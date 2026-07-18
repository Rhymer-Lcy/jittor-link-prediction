# jittor-link-prediction

计图（Jittor）人工智能挑战赛——动态图时序链路预测赛题的参赛代码。

给定带时间戳的交互边 `(src, dst, time)`，对测试集中每条查询 `(src, time)` 的
100 个候选节点 `c1..c100` 输出交互概率（每行 100 个概率值，无表头提交）。
离线评估采用尾部留一 + 99 负采样的 MRR。

## 目录结构

```
jittor-link-prediction/
├── data/                  # 数据（不入库）
│   ├── data_A.zip         # 官方原始数据包
│   └── data_A/
│       ├── dataset1/      # train 69 万条边；test 6.1 万条查询
│       └── dataset2/      # train 226 万条边（含 split 列）；test 15.3 万条查询
├── src/
│   └── train_line.py      # LINE 嵌入 + 相似用户协同打分 + 虚拟边自训练主脚本
├── outputs/               # 运行产物（不入库）：checkpoints / 嵌入 / 提交文件
├── requirements.txt
└── README.md
```

## 环境与运行

```bash
pip install -r requirements.txt
python src/train_line.py
```

- 数据集切换：改 `src/train_line.py` 中 `DATASET = "dataset1" / "dataset2"`。
- 所有路径均相对项目根目录；断点、虚拟边、提交文件写入 `outputs/<dataset>/`。
- 支持断点续训：存在 `outputs/<dataset>/checkpoints/line_last.pt` 时自动恢复。

## 算法流程

1. 全量真实边（双向）训练 LINE（一阶 + 二阶邻近，联合 BCE 损失）；
2. 每 `TRAIN_CYCLE=10` 轮导出节点嵌入，按余弦相似度为每个 src 构建
   Top-2000 相似用户缓存（前 200 权重 1.0，后 1800 权重 0.2 双区间衰减）；
3. 对 test 每条查询的 100 个候选做相似用户协同打分（稀疏矩阵矢量化），
   屏蔽该 src 在查询时刻之前的真实历史交互，归一化后输出提交文件；
4. 打分超过阈值 `BACK_FILL_THRESHOLD=0.97` 的 Top-2 候选生成虚拟边，
   复制 `VIRT_REPEAT_TIMES=2` 倍并入下一轮训练（自训练迭代），同时重置 Adam 动量；
5. 全量数据尾部留一 + 99 随机负采样计算 MRR 作为离线指标。

## 相对原始脚本（1.py）的修复与优化

重构 commit 的 diff 可完整审阅，关键点：

| # | 类别 | 说明 |
|---|------|------|
| 1 | **正确性** | 训练正样本目的节点误用 `s_pos`（自环对 `(u,u)`），修正为 `d_pos`。此修复会改变训练结果，历史成绩 0.424 是在带该 bug 的版本上取得的 |
| 2 | **必崩** | 评估打印引用未定义变量 `val_mcc`（NameError，每 10 轮首次评估即崩溃），修正为 `val_mrr` |
| 3 | **正确性** | MRR 累加器写在样本循环内部被反复清零，实际只返回最后一个样本的贡献；已改为正确累加求均值 |
| 4 | **内存** | 打分缓存原为稠密 `(N+1)²` float32 矩阵，dataset2 节点 id 超 12 万需 60+ GB 内存；改为 scipy CSR 稀疏矩阵 |
| 5 | **正确性** | 相似用户不足 2000 时用 `-1` 补位索引会回绕到末位节点造成污染，改为显式掩码置零 |
| 6 | **性能** | 历史交互查询由逐行全表扫描（每条 test 查询扫 226 万行）改为预建按 src 的时间排序索引 + 二分查找 |
| 7 | **性能** | 基础交互缓存构建由双层 iterrows 改为 `groupby(["src","dst"]).size()`；test 循环由 iterrows 改为预取 numpy 数组 |
| 8 | **可移植** | 移除指向其他机器的硬编码绝对路径，全部改为项目相对路径 |
| 9 | **清理** | 移除未用 import（sklearn、defaultdict）、死代码、未引用常量；`calc_mcc_eval` 更名为 `calc_mrr_eval`（该函数实际计算 MRR） |
| 10 | **必崩** | Adam 动量重置将 `step` 覆写为 int64 新张量，与 torch 2.x 要求的 float step 张量 dtype 不符，重置后首次 `opt.step()` 即崩溃；改为就地 `zero_()` 保留 dtype/device |

以上修复已通过合成小数据端到端冒烟验证（训练 → 预测输出 → 虚拟边 → MRR 评估 → Adam 重置 → 断点续训）。

## 待办

- [ ] **Jittor 迁移**：当前实现基于 PyTorch；若赛题要求最终提交使用计图框架，需将 LINE 模型与训练循环迁移至 Jittor（模型结构简单，仅 3 个 Embedding 层，迁移成本低）。
- [ ] 修复 #1 后重跑并与历史成绩 0.424 对比。
- [ ] 负采样 `gen_neg_batch` 仍为逐样本 Python 循环，是训练吞吐瓶颈，可矢量化。

## 历史记录

- 原始文件为 `新建文件夹/1.py` + `data_A.zip`，2026-07-18 重组为本项目结构。
- 原脚本头部成绩备注："21：重来：0.424"。
