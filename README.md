# jittor-link-prediction

Competition code for the Jittor AI Challenge temporal link prediction track.

Given timestamped interaction edges `(src, dst, time)`, output an interaction
probability for each of the 100 candidate nodes `c1..c100` of every test query
`(src, time)` (submission: 100 probabilities per row, no header). Offline
evaluation uses leave-one-out tail MRR with 99 sampled negatives.

Submission workflow (per teammate): run locally, submit the generated answer
file (2–10 submissions/day depending on the day); the Jittor version of the
code must be open-sourced at the end of the competition.

## Layout

```
jittor-link-prediction/
├── data/                  # data (git-ignored)
│   ├── data_A.zip         # official data package A
│   └── data_A/
│       ├── dataset1/      # train 691k edges, 43k nodes; test 61k queries
│       └── dataset2/      # train 2.26M edges, 140k nodes (extra split col); test 153k queries
├── src/
│   └── train_line.py      # LINE embedding + collaborative scoring + virtual-edge self-training
├── outputs/               # run artifacts (git-ignored): checkpoints / embeddings / submissions
├── requirements.txt
└── README.md
```

## Setup and run

```bash
pip install -r requirements.txt
DATASET=dataset1 python src/train_line.py   # or DATASET=dataset2 (default)
```

- All paths are relative to the project root; checkpoints, virtual edges and
  submission files go to `outputs/<dataset>/`.
- Training resumes automatically when
  `outputs/<dataset>/checkpoints/line_last.pt` exists.
- A submission file `outputs/<dataset>/result_epoch_<N>.csv` is written every
  `TRAIN_CYCLE=10` epochs.

## Algorithm

1. Train LINE (first- + second-order proximity, joint BCE loss) on all real
   edges (bidirectional).
2. Every `TRAIN_CYCLE=10` epochs, export node embeddings and build a top-2000
   similar-user cache per src by cosine similarity (two-band decay: weight 1.0
   for the top 200, 0.2 for the rest).
3. Score the 100 candidates of each test query by similar-user collaborative
   scoring (sparse-matrix vectorized), mask the src's real interactions before
   the query time, normalize, and write the submission file.
4. Candidates scoring above `BACK_FILL_THRESHOLD=0.97` (top-2 per query) become
   virtual edges, repeated `VIRT_REPEAT_TIMES=2` times and merged into the next
   training rounds (self-training); Adam moments are reset at each cycle.
5. Compute leave-one-out tail MRR with 99 random negatives as the offline metric.

## Fixes and optimizations vs. the original script (1.py)

The refactor commit diff is fully reviewable; key points:

| # | Type | Description |
|---|------|-------------|
| 1 | **Correctness** | Positive training pairs used `s_pos` as the destination (self-loop pairs `(u,u)`); fixed to `d_pos`. This changes training results — the historical score 0.424 was obtained with this bug present |
| 2 | **Crash** | Evaluation print referenced the undefined variable `val_mcc` (NameError on the first eval every 10 epochs); fixed to `val_mrr` |
| 3 | **Correctness** | The MRR accumulator was reset inside the sample loop, so only the last sample contributed; now accumulates correctly |
| 4 | **Memory** | The scoring cache was a dense `(N+1)²` float32 matrix — ~78 GB for dataset2's 139k+ node ids; replaced with a scipy CSR sparse matrix |
| 5 | **Correctness** | When a src had fewer than 2000 similar users, `-1` padding indices wrapped around to the last node and contaminated the cache; replaced with an explicit mask |
| 6 | **Performance** | History lookups now use a per-src time-sorted index with binary search instead of scanning the full 2.26M-row table per test query |
| 7 | **Performance** | Base cache construction switched from nested `iterrows` to `groupby(["src","dst"]).size()`; the test loop reads pre-extracted numpy arrays instead of `iterrows` |
| 8 | **Portability** | Removed hard-coded absolute paths pointing to another machine; all paths are project-relative, dataset selected via the `DATASET` env var |
| 9 | **Cleanup** | Removed unused imports (sklearn, defaultdict), dead code, unreferenced constants; renamed `calc_mcc_eval` to `calc_mrr_eval` (it computes MRR) |
| 10 | **Crash** | The Adam moment reset overwrote `step` with a fresh int64 tensor, which torch 2.x rejects (expects float step); now zeroed in place preserving dtype/device |

All fixes verified end-to-end on a synthetic mini dataset (training ->
prediction output -> virtual edges -> MRR eval -> Adam reset -> checkpoint
resume).

## TODO

- [ ] **Jittor port**: the current implementation is PyTorch. The final
  open-source release must use Jittor; only the LINE model (3 embedding
  layers), Adam and BCE are framework-specific — everything else is
  numpy/scipy. Port and verify on CPU (RTX 5080/5090 are Blackwell sm_120;
  no reliable evidence of Jittor GPU support on them — use the competition
  server for Jittor GPU runs).
- [ ] Obtain data package B (`data_B`): the teammate mentioned two data
  packages, a and b; only `data_A.zip` is present locally.
- [ ] Re-run after fix #1 and compare against the historical 0.424.
- [ ] `gen_neg_batch` is still a per-sample Python loop and the training
  throughput bottleneck; vectorize.

## History

- Original files: `新建文件夹/1.py` + `data_A.zip`; reorganized into this
  project structure on 2026-07-18.
- Original script header note: "21: redo: 0.424"; teammate's current estimated
  score: 1.36+ (aggregate across datasets).
