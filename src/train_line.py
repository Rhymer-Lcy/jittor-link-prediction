# -*- coding: utf-8 -*-
"""LINE 图嵌入 + 相似用户协同打分 + 虚拟边自训练迭代管线。

流程：每训练 TRAIN_CYCLE 轮 LINE，导出节点嵌入 -> 基于嵌入余弦相似度构建
双区间衰减相似用户缓存 -> 对 test 集候选(c1..c100)矢量化协同打分并输出
提交文件 -> 高置信候选生成虚拟边并入下一轮训练 -> 尾部留一 MRR 评估。

历史成绩备注（原 1.py 头部记录）："21：重来：0.424"。
"""
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ===================== 全局随机种子 =====================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===================== 全局超参配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LINE模型超参
emb_total_dim = 400
sub_dim = emb_total_dim // 2
neg_ratio = 5
epochs = 400
batch_size = 1024
save_interval = 5
EMB_PRECISION = 6
LOSS_ALPHA = 0.5
USE_FP16 = False
GRAD_CLIP = 1.0

# 迭代控制
TRAIN_CYCLE = 10
# 学习率配置
INIT_LR = 1e-4
RESET_LR = 1e-4
LR_RESET_EVERY_N_PREDICT = 1
# 虚拟边重复复制倍数，提升样本训练频次
VIRT_REPEAT_TIMES = 2

# ===================== 路径配置（相对项目根目录） =====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = os.environ.get("DATASET", "dataset2")  # 可切换 dataset1 / dataset2
assert DATASET in ("dataset1", "dataset2"), f"unknown dataset: {DATASET}"
DATA_DIR = PROJECT_ROOT / "data" / "data_A" / DATASET
OUTPUT_DIR = PROJECT_ROOT / "outputs" / DATASET
ckpt_dir = OUTPUT_DIR / "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

train_csv = DATA_DIR / "train.csv"
test_csv = DATA_DIR / "test.csv"
latest_emb_path = OUTPUT_DIR / "line_latest_emb.csv"
# Test预测结果输出模板，按epoch区分文件
test_result_template = str(OUTPUT_DIR / "result_epoch_{}.csv")
best_ckpt_path = ckpt_dir / "line_best.pt"
last_ckpt_path = ckpt_dir / "line_last.pt"
# 虚拟边存储文件
virtual_edge_csv = ckpt_dir / "virtual_edges.csv"

# ===================== 预测配置 =====================
NEG_SAMPLE_NUM = 99
# 相似度两段衰减配置
TOP_N_FIRST = 200
TOP_N_SECOND = 2000
DECAY_W1 = 1.0
DECAY_W2 = 0.2
VAL_PER_SRC_TAIL = 1
ADD_TOP_K_HIST = 2
BACK_FILL_THRESHOLD = 0.97
REPLACE_WEIGHT_INSTEAD_ADD = False
MCC_SAMPLE_COUNT = 10000

# 全局缓存容器
src_dst_cache = dict()
src_top_sim = dict()
src2row = dict()
sim_neigh_arr = None
sim_weight_arr = None

base_src_dst_cache = dict()
prev_virt_single_for_cache = set()
global_real_src_np = np.array([])
# 按src预建的历史交互索引：src -> (升序time数组, 对应dst数组)
src_hist_times = dict()
src_hist_dsts = dict()
# Test候选列
c_cols = [f"c{i}" for i in range(1, 101)]
# 记录已执行的预测周期次数
predict_run_count = 0

# ===================== LINE模型定义 =====================
class LINE(nn.Module):
    def __init__(self, n_node, d_sub):
        super().__init__()
        self.emb_first = nn.Embedding(n_node, d_sub)
        self.emb_node = nn.Embedding(n_node, d_sub)
        self.emb_ctx = nn.Embedding(n_node, d_sub)
        torch.manual_seed(SEED)
        nn.init.xavier_uniform_(self.emb_first.weight)
        torch.manual_seed(SEED)
        nn.init.xavier_uniform_(self.emb_node.weight)
        torch.manual_seed(SEED)
        nn.init.xavier_uniform_(self.emb_ctx.weight)

    def score_first(self, s, d):
        es = self.emb_first(s)
        ed = self.emb_first(d)
        return torch.sum(es * ed, dim=-1)

    def score_second(self, s, d):
        es = self.emb_node(s)
        ed = self.emb_ctx(d)
        return torch.sum(es * ed, dim=-1)

    def get_final_emb(self):
        e1 = self.emb_first.weight.detach()
        e2 = self.emb_node.weight.detach()
        return torch.cat([e1, e2], dim=-1)

# ===================== 工具1：按用户尾部切分train/val =====================
def split_train_val_by_tail(df):
    # 每个src取时间序尾部 VAL_PER_SRC_TAIL 条为val；不足的src整组进val、不进train
    df = df.sort_values(["src", "time"]).reset_index(drop=True)
    val_df = df.groupby("src", sort=False).tail(VAL_PER_SRC_TAIL)
    train_df = df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df

# ===================== 工具2：负采样 =====================
def gen_neg_batch(batch_pos_cpu, full_pos_set, all_node_list):
    neg_s, neg_d = [], []
    for u, _ in batch_pos_cpu:
        for _ in range(neg_ratio):
            while True:
                rand_v = random.choice(all_node_list)
                if (int(u), rand_v) not in full_pos_set:
                    neg_s.append(u)
                    neg_d.append(rand_v)
                    break
    return torch.LongTensor(neg_s).to(device), torch.LongTensor(neg_d).to(device)

# ===================== 缓存基础工具 =====================
def add_src_dst_record(cache_dict, src_id: int, dst_id: int):
    if src_id not in cache_dict:
        cache_dict[src_id] = dict()
    dst_dict = cache_dict[src_id]
    if REPLACE_WEIGHT_INSTEAD_ADD:
        dst_dict[dst_id] = 1.0
    else:
        dst_dict[dst_id] = dst_dict.get(dst_id, 0.0) + 1.0

def build_history_index(full_df):
    # 一次性预建 src -> 按time升序的(dst,time)索引，替代逐行全表扫描
    src_hist_times.clear()
    src_hist_dsts.clear()
    df = full_df.sort_values("time", kind="mergesort")
    for src, g in df.groupby("src", sort=False):
        src_hist_times[int(src)] = g["time"].values.astype(float)
        src_hist_dsts[int(src)] = g["dst"].values.astype(np.int64)

def get_dst_before_time(src_id: int, cutoff_time: float) -> np.ndarray:
    times = src_hist_times.get(src_id)
    if times is None:
        return np.array([], dtype=np.int64)
    k = np.searchsorted(times, cutoff_time, side="left")
    return src_hist_dsts[src_id][:k]

# 原始逐点打分（兼容val与MRR评估）
def get_sim_agg_score(target_src: int, dst_id: int, cache_dict):
    sim_list = src_top_sim.get(target_src, [])
    total = 0.0
    for sim_src, sim_w in sim_list:
        dst_w = cache_dict.get(sim_src, {}).get(dst_id, 0.0)
        total += sim_w * dst_w
    return total

# ===================== 矢量化全套工具 =====================
def cache_dict_to_matrix(base_cache: dict, add_pair_set: set, max_node: int):
    # 稀疏CSR矩阵：dataset2节点id超12万，稠密(N+1)^2 float32需60+GB内存
    rows, cols, vals = [], [], []
    for src, ddict in base_cache.items():
        for d, w in ddict.items():
            rows.append(src)
            cols.append(d)
            vals.append(w)
    for u, d in add_pair_set:
        rows.append(u)
        cols.append(d)
        vals.append(1.0)
    n = max_node + 1
    mat = sp.coo_matrix(
        (np.array(vals, dtype=np.float32), (np.array(rows), np.array(cols))),
        shape=(n, n),
    ).tocsr()
    return mat

def batch_sim_score(target_src: int, dst_batch: np.ndarray, cache_mat):
    row_idx = src2row.get(target_src, -1)
    if row_idx == -1:
        return np.zeros_like(dst_batch, dtype=np.float32)
    neigh_ids = sim_neigh_arr[row_idx]
    neigh_w = sim_weight_arr[row_idx]
    weight_slice = cache_mat[neigh_ids][:, dst_batch].toarray()
    scores = neigh_w @ weight_slice
    return scores.astype(np.float32)

def build_sim_cache(emb_matrix, real_src_np):
    global src_top_sim, src2row, sim_neigh_arr, sim_weight_arr
    src_top_sim.clear()
    src2row.clear()
    all_node_emb = emb_matrix.astype(np.float32)
    max_emb_id = all_node_emb.shape[0] - 1

    valid_targets = [int(s) for s in real_src_np if 0 <= s <= max_emb_id]
    all_candidate_ids = np.arange(max_emb_id + 1, dtype=np.int64)

    if len(valid_targets) == 0:
        sim_neigh_arr = np.zeros((0, TOP_N_SECOND), dtype=np.int64)
        sim_weight_arr = np.zeros((0, TOP_N_SECOND), dtype=np.float32)
        return

    target_vecs = all_node_emb[valid_targets]

    target_norm = np.linalg.norm(target_vecs, axis=1, keepdims=True)
    target_norm[target_norm < 1e-8] = 1.0
    target_vecs_norm = target_vecs / target_norm

    candidate_norm = np.linalg.norm(all_node_emb, axis=1, keepdims=True)
    candidate_norm[candidate_norm < 1e-8] = 1.0
    candidate_vecs_norm = all_node_emb / candidate_norm

    sim_matrix = target_vecs_norm @ candidate_vecs_norm.T
    decay_mask = np.full(TOP_N_SECOND, DECAY_W2, dtype=np.float32)
    decay_mask[:TOP_N_FIRST] = DECAY_W1

    neigh_list = []
    weight_list = []
    for row_idx, src_id in enumerate(tqdm(valid_targets, desc="构建双区间衰减相似用户缓存+矢量化数组")):
        sim_row = sim_matrix[row_idx]
        sim_row[src_id] = 0.0
        row_len = len(sim_row)
        if row_len > TOP_N_SECOND:
            top_idx = np.argpartition(sim_row, -TOP_N_SECOND)[-TOP_N_SECOND:]
            top_idx = top_idx[np.argsort(sim_row[top_idx])[::-1]]
            valid_mask = np.ones(TOP_N_SECOND, dtype=bool)
        else:
            top_idx = np.argsort(sim_row)[::-1]
            pad_num = TOP_N_SECOND - len(top_idx)
            valid_mask = np.concatenate(
                [np.ones(len(top_idx), dtype=bool), np.zeros(pad_num, dtype=bool)]
            )
            top_idx = np.pad(top_idx, (0, pad_num), mode="constant", constant_values=0)
        top_sim_node = all_candidate_ids[top_idx].copy()
        top_sim_val = sim_row[top_idx].copy()
        # 补位条目权重置0（节点id无影响），修复原版用-1索引回绕导致的补位污染
        top_sim_val[~valid_mask] = 0.0
        weighted_sim = top_sim_val * decay_mask
        weighted_sim[weighted_sim < 1e-8] = 0.0
        sim_list = [
            (int(nid), float(w)) for nid, w in zip(top_sim_node, weighted_sim) if w > 1e-8
        ]
        src_top_sim[int(src_id)] = sim_list
        src2row[int(src_id)] = row_idx
        neigh_list.append(top_sim_node)
        weight_list.append(weighted_sim)
    sim_neigh_arr = np.array(neigh_list, dtype=np.int64)
    sim_weight_arr = np.array(weight_list, dtype=np.float32)

# ===================== Test矢量化预测函数（仅屏蔽真实历史交互） =====================
def predict_test(test_df, emb_matrix, last_pair_set, real_src_np, current_epoch):
    global src_dst_cache
    src_dst_cache.clear()
    build_sim_cache(emb_matrix, real_src_np)

    # 拷贝基础真实交互，打分缓存合并当前虚拟边
    curr_cache = {usr: ddict.copy() for usr, ddict in base_src_dst_cache.items()}
    for (u, d) in last_pair_set:
        add_src_dst_record(curr_cache, u, d)

    curr_round_all_pair = set()
    max_node_id = emb_matrix.shape[0] - 1
    cache_mat = cache_dict_to_matrix(curr_cache, set(), max_node_id)

    test_src_arr = test_df["src"].values.astype(np.int64)
    test_time_arr = test_df["time"].values.astype(float)
    cand_mat = test_df[c_cols].values.astype(np.int64)

    test_prob_rows = []

    for i in tqdm(range(len(test_df)), desc="Test集时序打分生成虚拟边【矢量化批量打分】"):
        src = int(test_src_arr[i])
        curr_time = float(test_time_arr[i])
        all_candidates = cand_mat[i]

        raw_scores = batch_sim_score(src, all_candidates, cache_mat)

        # 仅屏蔽当前时间前真实历史交互dst；本轮内部重复候选不屏蔽
        history_real_d = get_dst_before_time(src, curr_time)
        hist_mask = np.isin(all_candidates, history_real_d)
        raw_scores[hist_mask] = 0.0

        row_max = raw_scores.max()
        if row_max > 5:
            prob_list = (raw_scores / row_max).tolist()
        else:
            prob_list = (raw_scores / 500).tolist()
        test_prob_rows.append(prob_list)

        cand_prob = list(zip(all_candidates.tolist(), prob_list))
        cand_prob.sort(key=lambda x: x[1], reverse=True)
        top_k_items = cand_prob[:ADD_TOP_K_HIST]
        valid_items = [(d, p) for d, p in top_k_items if p > BACK_FILL_THRESHOLD]

        for d, p in valid_items:
            # set自动去重，重复预测不会重复存入
            curr_round_all_pair.add((src, d))

    # 覆盖写入新虚拟边，替换旧文件
    if len(curr_round_all_pair) > 0:
        df_virt = pd.DataFrame(sorted(curr_round_all_pair), columns=["src", "dst"])
        df_virt.to_csv(virtual_edge_csv, mode="w", header=True, index=False)
        print(f"✅ 本轮新虚拟边覆盖写入 {virtual_edge_csv}，共{len(curr_round_all_pair)}条，旧边已替换")
    else:
        pd.DataFrame([], columns=["src", "dst"]).to_csv(virtual_edge_csv, mode="w", header=True, index=False)
        print("⚠️ 本轮无虚拟边，已清空csv文件")

    out_df = pd.DataFrame(test_prob_rows)
    save_path = test_result_template.format(current_epoch)
    out_df.to_csv(save_path, index=False, header=False)
    print(f"\n✅ Test提交格式结果已保存：{save_path}")
    print("屏蔽规则：仅屏蔽当前时间前真实交互dst；本轮内部重复候选不屏蔽，最终虚拟边自动去重")

    return curr_round_all_pair

# ===================== MRR评估函数（尾部留一） =====================
def calc_mrr_eval(train_df, emb_matrix, real_src_np, sample_num=10000):
    global src_dst_cache
    src_dst_cache = {src: dst_map.copy() for src, dst_map in base_src_dst_cache.items()}
    # 打分同步加载当前虚拟边
    for (u, d) in prev_virt_single_for_cache:
        add_src_dst_record(src_dst_cache, u, d)
    build_sim_cache(emb_matrix, real_src_np)

    tail_records = []
    for src, g in train_df.groupby("src"):
        last_row = g.loc[g["time"].idxmax()]
        tail_records.append({
            "src": int(last_row["src"]),
            "true_dst": int(last_row["dst"]),
            "time": float(last_row["time"]),
        })
    if len(tail_records) > sample_num:
        eval_samples = random.sample(tail_records, sample_num)
    else:
        eval_samples = tail_records

    dst_all = train_df["dst"].values
    dst_min, dst_max = int(dst_all.min()), int(dst_all.max())
    total_cnt = len(eval_samples)
    total_mrr = 0.0
    for item in tqdm(eval_samples, desc="MRR指标评估打分"):
        src_id = item["src"]
        true_dst = item["true_dst"]
        pred_t = item["time"]
        negs = []
        while len(negs) < NEG_SAMPLE_NUM:
            cand = random.randint(dst_min, dst_max)
            if cand != src_id and cand != true_dst and cand not in negs:
                negs.append(cand)
        candidates = negs + [true_dst]
        history_d_set = set(get_dst_before_time(src_id, pred_t).tolist())
        score_map = {}
        for d in candidates:
            if d in history_d_set:
                score_map[d] = 0.0
            else:
                score_map[d] = get_sim_agg_score(src_id, d, src_dst_cache)
        scores = [score_map[d] for d in candidates]

        row_max = max(scores)
        if row_max > 1e-6:
            scores = [x / row_max for x in scores]
        cand_prob = list(zip(candidates, scores))
        cand_prob.sort(key=lambda x: x[1], reverse=True)

        rank = next(idx + 1 for idx, (d, _) in enumerate(cand_prob) if d == true_dst)
        total_mrr += 1.0 / rank

    avg_mrr = total_mrr / total_cnt if total_cnt > 0 else 0.0
    return avg_mrr

# ===================== 主训练入口 =====================
if __name__ == "__main__":
    df_raw = pd.read_csv(train_csv)
    df_raw = df_raw.drop_duplicates(subset=["src", "dst", "time"]).reset_index(drop=True)
    df_raw["src"] = df_raw["src"].astype(np.int64)
    df_raw["dst"] = df_raw["dst"].astype(np.int64)
    df_raw["time"] = df_raw["time"].astype(float)
    train_df_split, val_df_split = split_train_val_by_tail(df_raw)
    print(f"拆分训练切片：{len(train_df_split)}，验证切片：{len(val_df_split)}，全量总数据：{len(df_raw)}")

    # test不排序，保持原始顺序
    test_df = pd.read_csv(test_csv)
    test_df = test_df[["src", "time"] + c_cols].copy()
    test_df["src"] = test_df["src"].astype(np.int64)
    test_df["time"] = test_df["time"].astype(float)
    print(f"测试集样本数：{len(test_df)}，未做时间排序，保留原始读取顺序")

    num_entity = int(max(df_raw.src.max(), df_raw.dst.max())) + 1
    all_node_list = list(range(num_entity))

    train_src_set = set(df_raw["src"].unique())
    test_src_set = set(test_df["src"].unique())
    union_src = sorted(train_src_set.union(test_src_set))
    global_real_src_np = np.array(union_src)
    print(f"相似度计算候选用户总数(全量train+test)：{len(global_real_src_np)}")

    print("预构建全量训练交互基础缓存与按src历史索引...")
    pair_w = df_raw.groupby(["src", "dst"], sort=False).size()
    for (s, d), w in pair_w.items():
        base_src_dst_cache.setdefault(int(s), dict())[int(d)] = (
            1.0 if REPLACE_WEIGHT_INSTEAD_ADD else float(w)
        )
    build_history_index(df_raw)
    print("全量基础交互缓存构建完成")

    all_train_edges = df_raw[["src", "dst"]].values
    base_single_edges = []
    base_pos_set = set()
    for u, v in all_train_edges:
        base_single_edges.append([u, v])
        base_single_edges.append([v, u])
        base_pos_set.add((int(u), int(v)))
        base_pos_set.add((int(v), int(u)))
    base_single_edges = torch.LongTensor(base_single_edges).to(device)

    current_virt_bi_edges = torch.empty((0, 2), dtype=torch.long, device=device)
    prev_virt_single_for_cache = set()

    # 程序启动读取历史虚拟边csv
    if os.path.exists(virtual_edge_csv):
        df_load = pd.read_csv(virtual_edge_csv, dtype={"src": int, "dst": int})
        prev_virt_single_for_cache = set(
            zip(df_load["src"].astype(int), df_load["dst"].astype(int))
        )
        print(f"\n✅ 加载初始虚拟边文件 {virtual_edge_csv}，读取 {len(prev_virt_single_for_cache)} 条虚拟边，本轮纳入训练与打分")
        # 初始加载的虚拟边构建双向训练样本
        virt_bi_list = []
        for u, v in prev_virt_single_for_cache:
            virt_bi_list.append([u, v])
            virt_bi_list.append([v, u])
            base_pos_set.add((u, v))
            base_pos_set.add((v, u))
        virt_tensor = torch.LongTensor(virt_bi_list).to(device)
        current_virt_bi_edges = torch.repeat_interleave(virt_tensor, repeats=VIRT_REPEAT_TIMES, dim=0)
    else:
        print(f"\n⚠️ 虚拟边文件 {virtual_edge_csv} 不存在，本轮无虚拟边参与训练")

    model = LINE(num_entity, sub_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scaler = GradScaler(enabled=USE_FP16)
    start_epoch = 0
    best_loss = float("inf")

    if os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt["epoch"] + 1
        best_loss = ckpt["best_loss"]
        predict_run_count = ckpt.get("predict_run_count", 0)
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        print(f"✅ 加载断点，从epoch {start_epoch} 继续训练，已完成预测周期数：{predict_run_count}")

    cycle_counter = start_epoch % TRAIN_CYCLE
    total_line_epoch = epochs
    for ep in range(start_epoch, total_line_epoch):
        if cycle_counter >= TRAIN_CYCLE:
            cycle_counter = 0
            current_epoch_num = ep + 1
            print(f"\n===== 完成{TRAIN_CYCLE}轮LINE训练，执行Test矢量化预测 + MRR评估（Epoch:{current_epoch_num}） ====")
            current_emb = model.get_final_emb().cpu().numpy()
            emb_np = np.round(current_emb, decimals=EMB_PRECISION)
            emb_df = pd.DataFrame(emb_np)
            emb_df.insert(0, "node_id", list(range(num_entity)))
            emb_df.to_csv(latest_emb_path, index=False)

            # 预测生成新虚拟边，覆盖替换文件
            new_virt_set = predict_test(test_df, emb_np, prev_virt_single_for_cache, global_real_src_np, current_epoch_num)
            print(f"Test本轮生成单向虚拟交互：{len(new_virt_set)} 条")
            val_mrr = calc_mrr_eval(df_raw, emb_np, global_real_src_np, sample_num=MCC_SAMPLE_COUNT)
            print(f"全量数据尾部验证MRR:{val_mrr:.4f}")

            predict_run_count += 1
            print(f"当前累计完成预测周期：{predict_run_count}")

            # 重置Adam动量缓存，适配新增虚拟样本训练
            print("执行Adam动量缓存重置，适配新增虚拟样本训练...")
            for group in opt.param_groups:
                for p in group["params"]:
                    state = opt.state[p]
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].zero_()
                    # 就地清零保留dtype/device：新建int64张量会与torch 2.x的float step不符导致崩溃
                    if "step" in state:
                        if torch.is_tensor(state["step"]):
                            state["step"].zero_()
                        else:
                            state["step"] = 0

            # 更新全局虚拟边集合，重建下一轮训练样本
            prev_virt_single_for_cache = new_virt_set
            virt_bi_list = []
            for u, v in prev_virt_single_for_cache:
                virt_bi_list.append([u, v])
                virt_bi_list.append([v, u])
                base_pos_set.add((u, v))
                base_pos_set.add((v, u))
            virt_tensor = torch.LongTensor(virt_bi_list).to(device)
            current_virt_bi_edges = torch.repeat_interleave(virt_tensor, repeats=VIRT_REPEAT_TIMES, dim=0)
            print("✅ 已切换至本轮新生成虚拟边，下一轮训练/打分使用该批数据")

        # 拼接原始真实边 + 当前虚拟边联合训练
        full_train_graph = torch.cat([base_single_edges, current_virt_bi_edges], dim=0)
        pos_cnt = full_train_graph.shape[0]
        perm = torch.randperm(pos_cnt, device=device)
        pos_shuffle = full_train_graph[perm]
        batch_total = (pos_cnt + batch_size - 1) // batch_size
        total_loss = 0.0
        pbar = tqdm(range(0, pos_cnt, batch_size), desc=f"LINE Epoch {ep+1}/{total_line_epoch}")

        for start_idx in pbar:
            end_idx = min(start_idx + batch_size, pos_cnt)
            batch_pos = pos_shuffle[start_idx:end_idx]
            s_pos = batch_pos[:, 0]
            d_pos = batch_pos[:, 1]
            batch_cpu = batch_pos.cpu()
            s_neg, d_neg = gen_neg_batch(batch_cpu, base_pos_set, all_node_list)

            s_all = torch.cat([s_pos, s_neg])
            d_all = torch.cat([d_pos, d_neg])
            label = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)])

            opt.zero_grad()
            with autocast(enabled=USE_FP16):
                scr1 = model.score_first(s_all, d_all)
                scr2 = model.score_second(s_all, d_all)
                loss1 = F.binary_cross_entropy_with_logits(scr1, label.float())
                loss2 = F.binary_cross_entropy_with_logits(scr2, label.float())
                loss = loss1 + LOSS_ALPHA * loss2

            if USE_FP16:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
            total_loss += loss.item()
            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / batch_total
        cycle_counter += 1
        print(f"\n[LINE Epoch {ep+1}] 平均Loss:{avg_loss:.4f}")

        # 保存断点
        save_dict = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "opt_state": opt.state_dict(),
            "best_loss": best_loss,
            "dim": emb_total_dim,
            "node_num": num_entity,
            "scaler_state": scaler.state_dict(),
            "seed": SEED,
            "predict_run_count": predict_run_count,
        }
        torch.save(save_dict, last_ckpt_path)
        if (ep + 1) % save_interval == 0:
            torch.save(save_dict, ckpt_dir / f"line_epoch_{ep+1}.pt")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(save_dict, best_ckpt_path)
            print("🏆 更新最优模型权重")
