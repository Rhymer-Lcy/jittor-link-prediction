# -*- coding: utf-8 -*-
"""LINE embedding + similar-user collaborative scoring + virtual-edge self-training.

Pipeline: every TRAIN_CYCLE epochs of LINE training, export node embeddings ->
build a two-band decayed similar-user cache from cosine similarity -> score the
100 candidates (c1..c100) of each test query with vectorized collaborative
scoring and write the submission file -> promote high-confidence candidates to
virtual edges for the next training round -> evaluate leave-one-out tail MRR.

Historical score note (from the original 1.py header): "21: redo: 0.424".
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
from tqdm import tqdm

# ===================== Global random seed =====================
# Override via env (e.g. SEED=123) for multi-seed ensemble runs; non-default
# seeds write to outputs/<dataset>-s<seed>/ so runs never clobber each other
SEED = int(os.environ.get("SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ===================== Hyperparameters =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# LINE model
emb_total_dim = 400
sub_dim = emb_total_dim // 2
neg_ratio = 5
# Total epochs; override via env to extend a finished run (e.g. EPOCHS=500)
epochs = int(os.environ.get("EPOCHS", "400"))
batch_size = 1024
# Snapshot sparingly: dataset2 snapshots are ~1 GB each and filled the disk
save_interval = 50
EMB_PRECISION = 6
LOSS_ALPHA = 0.5
USE_FP16 = False
GRAD_CLIP = 1.0

# Self-training cycle control
TRAIN_CYCLE = 10
# Learning rate
INIT_LR = 1e-4
RESET_LR = 1e-4
LR_RESET_EVERY_N_PREDICT = 1
# Virtual edges are repeated this many times to raise their sampling frequency
VIRT_REPEAT_TIMES = 2

# ===================== Paths (relative to project root) =====================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Switch dataset via env var, e.g.  DATASET=dataset1 python src/train_line.py
DATASET = os.environ.get("DATASET", "dataset2")
assert DATASET in ("dataset1", "dataset2"), f"unknown dataset: {DATASET}"
DATA_DIR = PROJECT_ROOT / "data" / "data_A" / DATASET
# Staged self-training (teammate's idea): virtual edges are only harvested
# from test rows whose time falls inside the current stage window, which
# advances one stage per predict cycle. Scoring/output always covers all rows.
STAGED = os.environ.get("STAGED", "0") == "1"
N_STAGES = int(os.environ.get("N_STAGES", "5"))

_suffix = ("" if SEED == 42 else f"-s{SEED}") + ("-staged" if STAGED else "")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / (DATASET + _suffix)
ckpt_dir = OUTPUT_DIR / "checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

train_csv = DATA_DIR / "train.csv"
test_csv = DATA_DIR / "test.csv"
latest_emb_path = OUTPUT_DIR / "line_latest_emb.csv"
# Submission output template, one file per predict epoch
test_result_template = str(OUTPUT_DIR / "result_epoch_{}.csv")
best_ckpt_path = ckpt_dir / "line_best.pt"
last_ckpt_path = ckpt_dir / "line_last.pt"
# Virtual edge storage
virtual_edge_csv = ckpt_dir / "virtual_edges.csv"

# ===================== Prediction config =====================
NEG_SAMPLE_NUM = 99
# Two-band similarity decay
TOP_N_FIRST = 200
TOP_N_SECOND = 2000
DECAY_W1 = 1.0
DECAY_W2 = 0.2
VAL_PER_SRC_TAIL = 1
ADD_TOP_K_HIST = 2
BACK_FILL_THRESHOLD = 0.97
REPLACE_WEIGHT_INSTEAD_ADD = False
MCC_SAMPLE_COUNT = 10000
# Per-dataset history policy, measured on train tails: in dataset1 66% of next
# interactions repeat a past partner, in dataset2 0% do. So dataset2 masks
# historical dsts to zero, while dataset1 boosts them instead (offline MRR on
# dataset1: mask 0.284 / no-mask 0.802 / boost x20 0.915).
MASK_HISTORY = DATASET == "dataset2"
HIST_BOOST = 20.0 if DATASET == "dataset1" else 0.0
# Submission ranking policy per dataset, tuned on a real-candidate offline eval
# (negatives drawn from actual test candidate pools, which reproduced the
# online ordering; random-negative MRR was misleading for dataset2):
# - dataset1: blend embedding-CF with co-occurrence CF (online 0.776 -> 0.803)
# - dataset2: co-occurrence hurts online (0.526 -> 0.505, candidates skew
#   unpopular so cooc yields false positives); use pure embedding-CF plus a
#   small recent-popularity prior (last 20% of train time, +0.005 offline).
# Virtual-edge generation keeps using the raw embedding-CF signal.
# Online-validated recipe (ds1 0.803 / ds2 ~0.548). CAUTION: the offline
# real-candidate eval systematically overrates recency-flavored features —
# time-decayed history (harness +0.018) scored 0.7905 online (-0.013), so
# gamma=5 and the q98 recency window were rolled back together with it.
# One variable per isolated submission from here on.
COOC_GAMMA = 1.0 if DATASET == "dataset1" else 0.0
RPOP_DELTA = 0.3 if DATASET == "dataset2" else 0.0
RPOP_TIME_QUANTILE = 0.8
# Retained for experiments only; the production blend uses plain counts
HIST_TD_HALFLIFE_FRAC = 0.01
HIST_FLOOR = 5.0
# Sequential transition feature (dataset2): global next-dst transition counts
# from the user's most recent dst, sqrt-popularity normalized. Leak-free
# real-candidate eval: 0.608 -> 0.716, the largest ds2 gain found so far.
TRANS_GAMMA = 1.0 if DATASET == "dataset2" else 0.0

# Global caches
src_dst_cache = dict()
src_top_sim = dict()
src2row = dict()
sim_neigh_arr = None
sim_weight_arr = None

base_src_dst_cache = dict()
prev_virt_single_for_cache = set()
# Co-occurrence CF structures (train user-item matrix)
ui_mat = None
ui_mat_csc = None
dst_pop = None
dst_rpop_log = None
trans_mat = None
trans_pop = None
global_real_src_np = np.array([])
# Per-src history index: src -> (time array sorted ascending, matching dst array)
src_hist_times = dict()
src_hist_dsts = dict()
train_time_span = 1.0
# Test candidate columns
c_cols = [f"c{i}" for i in range(1, 101)]
# Completed predict cycles
predict_run_count = 0

# ===================== LINE model =====================
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

# ===================== Tool 1: tail split per src =====================
def split_train_val_by_tail(df):
    # Last VAL_PER_SRC_TAIL rows (by time) of each src go to val; srcs with too
    # few rows put the whole group in val and contribute nothing to train.
    df = df.sort_values(["src", "time"]).reset_index(drop=True)
    val_df = df.groupby("src", sort=False).tail(VAL_PER_SRC_TAIL)
    train_df = df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df

# ===================== Tool 2: negative sampling =====================
def build_pos_csr(pos_set, n_node: int):
    # Sparse membership matrix over positive pairs, for vectorized rejection
    arr = np.array(list(pos_set), dtype=np.int64)
    return sp.coo_matrix(
        (np.ones(len(arr), dtype=np.int8), (arr[:, 0], arr[:, 1])),
        shape=(n_node, n_node),
    ).tocsr()

def gen_neg_batch(s_pos_np, pos_csr, n_node: int):
    # Vectorized rejection sampling: uniform draws, redraw the few that hit a
    # positive pair (same distribution as the original per-sample loop)
    s_rep = np.repeat(s_pos_np, neg_ratio)
    d = np.random.randint(0, n_node, size=len(s_rep))
    for _ in range(30):
        hits = np.asarray(pos_csr[s_rep, d]).ravel() > 0
        if not hits.any():
            break
        d[hits] = np.random.randint(0, n_node, size=int(hits.sum()))
    return torch.LongTensor(s_rep).to(device), torch.LongTensor(d).to(device)

# ===================== Cache utilities =====================
def add_src_dst_record(cache_dict, src_id: int, dst_id: int):
    if src_id not in cache_dict:
        cache_dict[src_id] = dict()
    dst_dict = cache_dict[src_id]
    if REPLACE_WEIGHT_INSTEAD_ADD:
        dst_dict[dst_id] = 1.0
    else:
        dst_dict[dst_id] = dst_dict.get(dst_id, 0.0) + 1.0

def build_history_index(full_df):
    # One-off per-src index sorted by time, replacing full-table scans
    global train_time_span
    src_hist_times.clear()
    src_hist_dsts.clear()
    df = full_df.sort_values("time", kind="mergesort")
    for src, g in df.groupby("src", sort=False):
        src_hist_times[int(src)] = g["time"].values.astype(float)
        src_hist_dsts[int(src)] = g["dst"].values.astype(np.int64)
    train_time_span = float(full_df["time"].max() - full_df["time"].min()) or 1.0

def get_dst_before_time(src_id: int, cutoff_time: float) -> np.ndarray:
    times = src_hist_times.get(src_id)
    if times is None:
        return np.array([], dtype=np.int64)
    k = np.searchsorted(times, cutoff_time, side="left")
    return src_hist_dsts[src_id][:k]

def get_hist_before_time(src_id: int, cutoff_time: float):
    # Like get_dst_before_time but also returns the matching timestamps
    times = src_hist_times.get(src_id)
    if times is None:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    k = np.searchsorted(times, cutoff_time, side="left")
    return src_hist_dsts[src_id][:k], times[:k]

def decayed_count_in_history(candidates: np.ndarray, hist_d: np.ndarray, hist_t: np.ndarray, now: float) -> np.ndarray:
    # Sum of exponentially time-decayed occurrence weights per candidate
    if hist_d.size == 0:
        return np.zeros(candidates.shape[0], dtype=np.float64)
    w = 0.5 ** ((now - hist_t) / (HIST_TD_HALFLIFE_FRAC * train_time_span))
    return np.array([w[hist_d == c].sum() for c in candidates], dtype=np.float64)

def count_in_history(candidates: np.ndarray, hist: np.ndarray) -> np.ndarray:
    # Occurrence count of each candidate in the (possibly repeating) history
    if hist.size == 0:
        return np.zeros(candidates.shape[0], dtype=np.float32)
    values, counts = np.unique(hist, return_counts=True)
    idx = np.searchsorted(values, candidates)
    idx[idx >= len(values)] = 0
    hit = values[idx] == candidates
    return np.where(hit, counts[idx], 0).astype(np.float32)

def rownorm(v: np.ndarray) -> np.ndarray:
    mx = v.max()
    return v / mx if mx > 1e-12 else v

def build_cooc(df, n_node: int):
    global ui_mat, ui_mat_csc, dst_pop, dst_rpop_log
    ui_mat = sp.coo_matrix(
        (np.ones(len(df), dtype=np.float32), (df["src"].values, df["dst"].values)),
        shape=(n_node, n_node),
    ).tocsr()
    ui_mat_csc = ui_mat.tocsc()
    dst_pop = np.asarray(ui_mat.sum(axis=0)).ravel() + 1.0
    # Recent popularity: dst interaction counts in the last time window
    tcut = df["time"].quantile(RPOP_TIME_QUANTILE)
    rpop = np.zeros(n_node, dtype=np.float64)
    for d, c in df[df["time"] >= tcut].groupby("dst").size().items():
        rpop[int(d)] = float(c)
    dst_rpop_log = np.log1p(rpop)
    # Global sequential transition counts d_i -> d_{i+1} within src timelines
    global trans_mat, trans_pop
    df_sorted = df.sort_values(["src", "time"], kind="mergesort")
    srcs_arr = df_sorted["src"].values
    dsts_arr = df_sorted["dst"].values
    same_src = srcs_arr[1:] == srcs_arr[:-1]
    frm = dsts_arr[:-1][same_src]
    to = dsts_arr[1:][same_src]
    trans_mat = sp.coo_matrix(
        (np.ones(len(frm), dtype=np.float32), (frm, to)), shape=(n_node, n_node)
    ).tocsr()
    trans_pop = np.asarray(trans_mat.sum(axis=0)).ravel() + 1.0

def trans_scores(last_dst: int, cands: np.ndarray) -> np.ndarray:
    # Transition counts from the user's most recent dst, sqrt-pop normalized
    if trans_mat is None or last_dst < 0 or last_dst >= trans_mat.shape[0]:
        return np.zeros(len(cands), dtype=np.float64)
    row = np.asarray(trans_mat[last_dst, cands].todense()).ravel().astype(np.float64)
    return row / np.sqrt(trans_pop[cands])

def cooc_scores(src: int, cands: np.ndarray) -> np.ndarray:
    # Popularity-normalized co-occurrence CF: users overlapping src's history,
    # weighted by overlap size, aggregated over their interactions with cands
    if ui_mat is None or src >= ui_mat.shape[0]:
        return np.zeros(len(cands), dtype=np.float64)
    x = np.asarray((ui_mat @ ui_mat[src].T).todense()).ravel()
    x[src] = 0.0
    sc = np.asarray((sp.csr_matrix(x) @ ui_mat_csc[:, cands]).todense()).ravel()
    return sc.astype(np.float64) / np.sqrt(dst_pop[cands])

def precompute_cooc_rows(test_src_arr: np.ndarray, cand_mat: np.ndarray) -> np.ndarray:
    # Group rows by src so the expensive user-overlap vector is built once per src
    out = np.zeros(cand_mat.shape, dtype=np.float64)
    order = np.argsort(test_src_arr, kind="stable")
    i = 0
    while i < len(order):
        j = i
        s = int(test_src_arr[order[i]])
        while j < len(order) and test_src_arr[order[j]] == s:
            j += 1
        if ui_mat is not None and s < ui_mat.shape[0]:
            x = np.asarray((ui_mat @ ui_mat[s].T).todense()).ravel()
            x[s] = 0.0
            xr = sp.csr_matrix(x)
            for k in order[i:j]:
                cands = cand_mat[k]
                sc = np.asarray((xr @ ui_mat_csc[:, cands]).todense()).ravel()
                out[k] = sc / np.sqrt(dst_pop[cands])
        i = j
    return out

# Pointwise scoring (used by the MRR evaluation)
def get_sim_agg_score(target_src: int, dst_id: int, cache_dict):
    sim_list = src_top_sim.get(target_src, [])
    total = 0.0
    for sim_src, sim_w in sim_list:
        dst_w = cache_dict.get(sim_src, {}).get(dst_id, 0.0)
        total += sim_w * dst_w
    return total

# ===================== Vectorized scoring =====================
def cache_dict_to_matrix(base_cache: dict, add_pair_set: set, max_node: int):
    # Sparse CSR: dataset2 has 139k+ node ids, a dense (N+1)^2 float32 matrix
    # would need ~78 GB
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
    for row_idx, src_id in enumerate(tqdm(valid_targets, desc="Building two-band decayed similar-user cache")):
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
        # Zero out padded entries (node id is then irrelevant); fixes the
        # original -1 index wrap-around contamination
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

# ===================== Test prediction (masks real history only) =====================
def predict_test(test_df, emb_matrix, last_pair_set, real_src_np, current_epoch):
    global src_dst_cache
    src_dst_cache.clear()
    build_sim_cache(emb_matrix, real_src_np)

    # Copy the base real interactions, then merge current virtual edges
    curr_cache = {usr: ddict.copy() for usr, ddict in base_src_dst_cache.items()}
    for (u, d) in last_pair_set:
        add_src_dst_record(curr_cache, u, d)

    curr_round_all_pair = set()
    max_node_id = emb_matrix.shape[0] - 1
    cache_mat = cache_dict_to_matrix(curr_cache, set(), max_node_id)

    # Stage window for virtual-edge harvesting (all rows are always scored)
    if STAGED:
        stage_frac = min(1.0, (predict_run_count + 1) / N_STAGES)
        stage_cut = float(test_df["time"].quantile(stage_frac))
        print(f"[staged] cycle {predict_run_count + 1}: harvesting virtual edges from test rows with time <= {stage_frac:.0%} quantile")
    else:
        stage_cut = float("inf")

    test_src_arr = test_df["src"].values.astype(np.int64)
    test_time_arr = test_df["time"].values.astype(float)
    cand_mat = test_df[c_cols].values.astype(np.int64)

    if COOC_GAMMA > 0:
        print("Precomputing co-occurrence CF scores...")
        cooc_rows = precompute_cooc_rows(test_src_arr, cand_mat)
    else:
        cooc_rows = None

    test_prob_rows = []

    for i in tqdm(range(len(test_df)), desc="Scoring test queries (vectorized)"):
        src = int(test_src_arr[i])
        curr_time = float(test_time_arr[i])
        all_candidates = cand_mat[i]

        collab = batch_sim_score(src, all_candidates, cache_mat)
        history_real_d, history_real_t = get_hist_before_time(src, curr_time)

        extra = np.zeros(len(all_candidates), dtype=np.float64)
        if COOC_GAMMA > 0:
            extra += COOC_GAMMA * rownorm(cooc_rows[i])
        if RPOP_DELTA > 0:
            extra += RPOP_DELTA * rownorm(dst_rpop_log[np.clip(all_candidates, 0, len(dst_rpop_log) - 1)])
        if TRANS_GAMMA > 0 and len(history_real_d) > 0:
            cc = np.clip(all_candidates, 0, trans_mat.shape[0] - 1)
            extra += TRANS_GAMMA * rownorm(trans_scores(int(history_real_d[-1]), cc))

        if MASK_HISTORY:
            # dataset2: real interactions before the query time never repeat
            hist_mask = np.isin(all_candidates, history_real_d)
            raw_scores = collab.copy()
            raw_scores[hist_mask] = 0.0
            blend = rownorm(collab.astype(np.float64)) + extra
            blend[hist_mask] = 0.0
        else:
            # dataset1: repeats dominate, boost candidates by own history count
            own_cnt = count_in_history(all_candidates, history_real_d)
            raw_scores = collab + HIST_BOOST * own_cnt
            blend = HIST_BOOST * own_cnt.astype(np.float64) + rownorm(collab.astype(np.float64)) + extra

        # Legacy confidence-gated probabilities drive virtual-edge generation only
        row_max = raw_scores.max()
        if row_max > 5:
            prob_list = (raw_scores / row_max).tolist()
        else:
            prob_list = (raw_scores / 500).tolist()
        # Submission output ranks by the ensemble (MRR is rank-based)
        test_prob_rows.append(rownorm(blend).tolist())

        cand_prob = list(zip(all_candidates.tolist(), prob_list))
        cand_prob.sort(key=lambda x: x[1], reverse=True)
        if curr_time > stage_cut:
            continue
        top_k_items = cand_prob[:ADD_TOP_K_HIST]
        valid_items = [(d, p) for d, p in top_k_items if p > BACK_FILL_THRESHOLD]
        if not MASK_HISTORY:
            # Pairs already in real history are real edges — no virtual copy
            hist_set = set(history_real_d.tolist())
            valid_items = [(d, p) for d, p in valid_items if d not in hist_set]

        for d, p in valid_items:
            # Set dedupes automatically
            curr_round_all_pair.add((src, d))

    # Overwrite the virtual edge file with this round's edges
    if len(curr_round_all_pair) > 0:
        df_virt = pd.DataFrame(sorted(curr_round_all_pair), columns=["src", "dst"])
        df_virt.to_csv(virtual_edge_csv, mode="w", header=True, index=False)
        print(f"[OK] Wrote {len(curr_round_all_pair)} virtual edges to {virtual_edge_csv} (old file replaced)")
    else:
        pd.DataFrame([], columns=["src", "dst"]).to_csv(virtual_edge_csv, mode="w", header=True, index=False)
        print("[WARN] No virtual edges this round, csv cleared")

    out_df = pd.DataFrame(test_prob_rows)
    save_path = test_result_template.format(current_epoch)
    out_df.to_csv(save_path, index=False, header=False)
    print(f"\n[OK] Submission file saved: {save_path}")
    print("Masking rule: only real interactions before the query time are masked; virtual edges are deduped")

    return curr_round_all_pair

# ===================== MRR evaluation (leave-one-out tail) =====================
def calc_mrr_eval(train_df, emb_matrix, real_src_np, sample_num=10000):
    global src_dst_cache
    src_dst_cache = {src: dst_map.copy() for src, dst_map in base_src_dst_cache.items()}
    # Include current virtual edges in the scoring cache
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
    for item in tqdm(eval_samples, desc="MRR evaluation"):
        src_id = item["src"]
        true_dst = item["true_dst"]
        pred_t = item["time"]
        negs = []
        while len(negs) < NEG_SAMPLE_NUM:
            cand = random.randint(dst_min, dst_max)
            if cand != src_id and cand != true_dst and cand not in negs:
                negs.append(cand)
        cand_arr = np.array(negs + [true_dst], dtype=np.int64)
        history_d, history_t = get_hist_before_time(src_id, pred_t)
        collab = np.array([get_sim_agg_score(src_id, int(d), src_dst_cache) for d in cand_arr], dtype=np.float64)
        cnt = count_in_history(cand_arr, history_d).astype(np.float64)
        extra = np.zeros(len(cand_arr), dtype=np.float64)
        if COOC_GAMMA > 0:
            extra += COOC_GAMMA * rownorm(cooc_scores(src_id, cand_arr))
        if RPOP_DELTA > 0:
            extra += RPOP_DELTA * rownorm(dst_rpop_log[np.clip(cand_arr, 0, len(dst_rpop_log) - 1)])
        if TRANS_GAMMA > 0 and len(history_d) > 0:
            cc = np.clip(cand_arr, 0, trans_mat.shape[0] - 1)
            extra += TRANS_GAMMA * rownorm(trans_scores(int(history_d[-1]), cc))
        if MASK_HISTORY:
            blend = rownorm(collab) + extra
            blend[cnt > 0] = 0.0
        else:
            blend = HIST_BOOST * cnt + rownorm(collab) + extra

        # Pessimistic tie handling: true dst ranks after all equal scores
        rank = 1 + int((blend > blend[-1]).sum()) + int((blend[:-1] == blend[-1]).sum())
        total_mrr += 1.0 / rank

    avg_mrr = total_mrr / total_cnt if total_cnt > 0 else 0.0
    return avg_mrr

# ===================== Main =====================
if __name__ == "__main__":
    print(f"Dataset: {DATASET} | device: {device}")
    df_raw = pd.read_csv(train_csv)
    df_raw = df_raw.drop_duplicates(subset=["src", "dst", "time"]).reset_index(drop=True)
    df_raw["src"] = df_raw["src"].astype(np.int64)
    df_raw["dst"] = df_raw["dst"].astype(np.int64)
    df_raw["time"] = df_raw["time"].astype(float)
    train_df_split, val_df_split = split_train_val_by_tail(df_raw)
    print(f"Train slice: {len(train_df_split)}, val slice: {len(val_df_split)}, total rows: {len(df_raw)}")

    # Keep the original test row order (no sorting)
    test_df = pd.read_csv(test_csv)
    test_df = test_df[["src", "time"] + c_cols].copy()
    test_df["src"] = test_df["src"].astype(np.int64)
    test_df["time"] = test_df["time"].astype(float)
    print(f"Test rows: {len(test_df)} (original order preserved)")

    num_entity = int(max(df_raw.src.max(), df_raw.dst.max())) + 1

    train_src_set = set(df_raw["src"].unique())
    test_src_set = set(test_df["src"].unique())
    union_src = sorted(train_src_set.union(test_src_set))
    global_real_src_np = np.array(union_src)
    print(f"Similarity target srcs (train+test union): {len(global_real_src_np)}")

    print("Building base interaction cache and per-src history index...")
    pair_w = df_raw.groupby(["src", "dst"], sort=False).size()
    for (s, d), w in pair_w.items():
        base_src_dst_cache.setdefault(int(s), dict())[int(d)] = (
            1.0 if REPLACE_WEIGHT_INSTEAD_ADD else float(w)
        )
    build_history_index(df_raw)
    build_cooc(df_raw, num_entity)
    print("Base caches ready")

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

    # Load virtual edges left by a previous run, if any
    if os.path.exists(virtual_edge_csv):
        df_load = pd.read_csv(virtual_edge_csv, dtype={"src": int, "dst": int})
        prev_virt_single_for_cache = set(
            zip(df_load["src"].astype(int), df_load["dst"].astype(int))
        )
        print(f"\n[OK] Loaded {len(prev_virt_single_for_cache)} virtual edges from {virtual_edge_csv}")
        # Build bidirectional training samples from the loaded virtual edges
        virt_bi_list = []
        for u, v in prev_virt_single_for_cache:
            virt_bi_list.append([u, v])
            virt_bi_list.append([v, u])
            base_pos_set.add((u, v))
            base_pos_set.add((v, u))
        virt_tensor = torch.LongTensor(virt_bi_list).to(device)
        current_virt_bi_edges = torch.repeat_interleave(virt_tensor, repeats=VIRT_REPEAT_TIMES, dim=0)
    else:
        print(f"\n[WARN] Virtual edge file {virtual_edge_csv} not found, none used this round")

    pos_csr = build_pos_csr(base_pos_set, num_entity)

    model = LINE(num_entity, sub_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=INIT_LR)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_FP16)
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
        print(f"[OK] Resumed from checkpoint at epoch {start_epoch}, predict cycles done: {predict_run_count}")

    cycle_counter = start_epoch % TRAIN_CYCLE
    total_line_epoch = epochs
    for ep in range(start_epoch, total_line_epoch):
        if cycle_counter >= TRAIN_CYCLE:
            cycle_counter = 0
            current_epoch_num = ep + 1
            print(f"\n===== {TRAIN_CYCLE} LINE epochs done, running test prediction + MRR eval (epoch {current_epoch_num}) =====")
            current_emb = model.get_final_emb().cpu().numpy()
            emb_np = np.round(current_emb, decimals=EMB_PRECISION)
            emb_df = pd.DataFrame(emb_np)
            emb_df.insert(0, "node_id", list(range(num_entity)))
            emb_df.to_csv(latest_emb_path, index=False)

            # Predict, generating a new virtual edge set (file is overwritten)
            new_virt_set = predict_test(test_df, emb_np, prev_virt_single_for_cache, global_real_src_np, current_epoch_num)
            print(f"Virtual edges generated this round: {len(new_virt_set)}")
            val_mrr = calc_mrr_eval(df_raw, emb_np, global_real_src_np, sample_num=MCC_SAMPLE_COUNT)
            print(f"Leave-one-out tail MRR: {val_mrr:.4f}")

            predict_run_count += 1
            print(f"Predict cycles completed: {predict_run_count}")

            # Reset Adam moment buffers to adapt to the new virtual samples
            print("Resetting Adam moment buffers...")
            for group in opt.param_groups:
                for p in group["params"]:
                    state = opt.state[p]
                    if "exp_avg" in state:
                        state["exp_avg"].zero_()
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].zero_()
                    # Zero in place to keep dtype/device: assigning a fresh
                    # int64 tensor crashes torch 2.x, which expects float step
                    if "step" in state:
                        if torch.is_tensor(state["step"]):
                            state["step"].zero_()
                        else:
                            state["step"] = 0

            # Swap in the new virtual edge set for the next training rounds
            prev_virt_single_for_cache = new_virt_set
            virt_bi_list = []
            for u, v in prev_virt_single_for_cache:
                virt_bi_list.append([u, v])
                virt_bi_list.append([v, u])
                base_pos_set.add((u, v))
                base_pos_set.add((v, u))
            virt_tensor = torch.LongTensor(virt_bi_list).to(device)
            current_virt_bi_edges = torch.repeat_interleave(virt_tensor, repeats=VIRT_REPEAT_TIMES, dim=0)
            pos_csr = build_pos_csr(base_pos_set, num_entity)
            print("[OK] Switched to this round's virtual edges")

        # Train on real edges + current virtual edges
        full_train_graph = torch.cat([base_single_edges, current_virt_bi_edges], dim=0)
        pos_cnt = full_train_graph.shape[0]
        perm = torch.randperm(pos_cnt, device=device)
        pos_shuffle = full_train_graph[perm]
        batch_total = (pos_cnt + batch_size - 1) // batch_size
        total_loss = 0.0
        pbar = tqdm(range(0, pos_cnt, batch_size), desc=f"LINE epoch {ep+1}/{total_line_epoch}")

        for start_idx in pbar:
            end_idx = min(start_idx + batch_size, pos_cnt)
            batch_pos = pos_shuffle[start_idx:end_idx]
            s_pos = batch_pos[:, 0]
            d_pos = batch_pos[:, 1]
            s_neg, d_neg = gen_neg_batch(s_pos.cpu().numpy(), pos_csr, num_entity)

            s_all = torch.cat([s_pos, s_neg])
            d_all = torch.cat([d_pos, d_neg])
            label = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)])

            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=USE_FP16):
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
        print(f"\n[LINE epoch {ep+1}] avg loss: {avg_loss:.4f}")

        # Save checkpoint
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
            print("[BEST] Updated best model weights")
