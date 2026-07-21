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
│   ├── train_line.py      # LINE embedding + collaborative scoring + virtual-edge self-training
│   ├── train_bpr.py       # BPR-MF embedding trainer (pairwise ranking loss, pop075 negatives)
│   └── ensemble_predict.py# predict-only scoring (multi-run ensemble, item-CF + BPR blend, real-candidate eval)
├── outputs/               # run artifacts (git-ignored): checkpoints / embeddings / submissions
├── requirements.txt
└── README.md
```

## Setup and run

```bash
pip install -r requirements.txt

# Train the LINE embedding (resumes from outputs/<dataset>/checkpoints/line_last.pt)
DATASET=dataset1 python src/train_line.py                 # or DATASET=dataset2 (default)

# Train the BPR embedding used by the default blend (fast: minutes, not hours)
DATASET=dataset1 python src/train_bpr.py                  # -> outputs/<dataset>-bpr/bpr_emb.npy

# Score a submission from the trained embedding(s)
DATASET=dataset1 python src/ensemble_predict.py           # -> outputs/<dataset>-ensemble/result_ensemble.csv
DATASET=dataset2 python src/ensemble_predict.py --eval    # real-candidate offline MRR instead of a submission
```

- All paths are relative to the project root; checkpoints, virtual edges and
  submission files go to `outputs/<dataset>/`.
- Training resumes automatically when
  `outputs/<dataset>/checkpoints/line_last.pt` exists.
- A submission file `outputs/<dataset>/result_epoch_<N>.csv` is written every
  `TRAIN_CYCLE=10` epochs.
- Optional training knobs (env vars): `EMB_DIM` (default 400), `NEG_RATIO`
  (default 5), `NEG_DIST` (`uniform` | `pop075` for degree^0.75 sampling),
  `EPOCHS`, `SEED`, `STAGED`. A non-default `SEED`/`STAGED`/`NEG_DIST`/`EMB_DIM`
  writes to a suffixed `outputs/<dataset>-<...>/` dir so runs never clobber.

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

### Production ranking blends (`ensemble_predict.py`, online-validated)

Item-CF — the cosine similarity between a candidate's embedding and the src's
historical dst embeddings (mean over all, plus mean of the top-3 closest) — is
the one feature that improved both datasets online:

- **dataset1** (66% of next interactions repeat a past partner):
  `16*own_history_count + 0.5*usercf + 0.3*cooc_cf + 2.0*itemcf_mean +
  1.2*itemcf_top3 + 9.0*bpr_ens5(tau=0.25span)`, scored as a uniform +
  pop075(d512) two-embedding ensemble — online 0.803 -> 0.813 (item-CF) ->
  0.8177 (ensemble) -> 0.8285 (holdout-retuned weights) -> 0.8508 (BPR term)
  -> 0.8526 (5-seed BPR) -> 0.8554 (recency-weighted BPR, `BPR_TAU_FRAC`).
- **dataset2** (0% repeats; history masked to zero):
  `0.5*usercf + 1.0*recent_popularity + 0.85*itemcf_mean + 0.8*itemcf_top3 +
  0.7*bpr_ens5` — online 0.5441 -> 0.5587 (item-CF) -> 0.5607
  (holdout-retuned weights) -> 0.5712 (BPR term) -> 0.5766 (5-seed BPR).
- **BPR direct score** (`train_bpr.py`): a separate BPR-MF embedding
  (single node table, pairwise softplus ranking loss, degree^0.75 negatives,
  d256/120ep) whose raw `e_src . e_cand` blends in as an extra term. The
  ranking loss is the point — the same LINE embedding's direct score is
  honestly negative. Degree^0.75 negatives are essential (uniform gave
  +0.0009); 240ep overfits; d512 no gain; seed-stable.
- **5-seed BPR ensemble** (`bpr_ens5`, the default via `BPR_RUNS`): rownormed
  direct scores from seeds 42/123/777/2024/31337 averaged. Unlike the refuted
  LINE multi-seed ensemble this works — individual seeds tie (all pairwise
  deltas insignificant) but averaging denoises: honest +0.0058 (ds2) /
  +0.0020 (ds1), online +0.0054 / +0.0018 (folds 0.93 / 0.90). Three seeds
  were not significant; five were.
- Best combined online total **1.4320** (validated 2026-07-21;
  ds1 0.8554 + ds2 0.5766, the sum landing within 1e-5 of the two
  separately-validated component scores).
- Transfer rule: when the honest (holdout) and leaky evals agree on a change
  it transfers online nearly 1:1 (ds1 retune 0.96, ds1 BPR 0.96, 5-seed
  0.90-0.93; ds2 BPR 0.62); when they disagree, the honest direction still
  wins but folds to ~0.3 (ds2 retune). EXCEPTION (2026-07-20): recency-
  weighted TRAINING on dataset2 (tau=0.10span BPR) regressed online
  (0.5766 -> 0.5724) even though both protocols predicted a gain — the
  offline positives sit at the training boundary while the online test spans
  a full year, so both protocols overrate training-boundary myopia. The same
  change at tau=0.25span on dataset1 transferred at 0.93. Keep dataset2
  training recency-free; treat any ds2 recency knob as online-unverifiable.

Offline evaluation that tracks the online ordering: negatives drawn from the
src's actual test candidate pools (`ensemble_predict.py --eval`). CAUTION: it
systematically overrates recency-flavored features — time-decayed history and
the sequential transition feature both showed large offline gains and lost
online — and inflates even the item-CF gain ~2.4x, so a new embedding-geometry
feature needs a large offline margin to be worth a submission. Trust it only
for non-temporal signals, and always confirm online.

For embedding-geometry features, prefer the leak-free protocol: train a
holdout embedding with `EVAL_HOLDOUT=1` (per-src tail rows removed from
training) and evaluate against it — the held-out tails are then genuinely
unseen pairs. Measured on dataset2 (2026-07-20): honest item-CF delta +0.0265
vs online +0.0146, i.e. a ~0.55 folding factor (vs ~0.42 for the leaky
protocol's inflated delta). The protocol has an online scalp: the dataset2
weight retune it proposed (and the leaky eval opposed) scored 0.5587 -> 0.5607
— when the two protocols disagree, trust the holdout. First victims: the direct
LINE scores (`emb_node[src]·emb_ctx[cand]` and the first-order dot) looked
like +0.12 under the leaky eval and are NEGATIVE at every weight honestly —
pure tail-edge memorization (the tail is the one trained pair that history
masking leaves alive, so the leaky eval turns the model head into an oracle).

Refuted directions (isolated online submissions, code removed): time-decayed
history (0.803 -> 0.7905), co-occurrence CF on dataset2 (0.526 -> 0.505),
sequential transition feature (0.5441 -> 0.5305), test-candidate-frequency
prior ("tpop": appearance counts of candidates in excess of the uniform
baseline as a test-period popularity estimate — 0.5587 -> 0.5199 replacing
rpop, actively harmful, implying candidate negatives are NOT uniform pool
draws but likely popularity- or similarity-biased distractors); also
multi-seed ensemble (offline tie) and +50 training epochs past ~70 (flat).

Virtual-edge self-training is a no-op under the production blend (holdout
verdict, 2026-07-21): retraining the holdout LINE with `VIRT_MODE=off` moves
the full-blend MRR by +0.0001 (ds1, P=0.39) / +0.0004 (ds2, P=0.44); a curated
ds1 edge set with 3x the measured gate precision (11.0% vs the degenerate
gate's 3.8% against held-out tails; dataset2's edges are ~0.04%, pure noise)
still changes nothing (-0.0002, P=0.70); and merging dataset2's ~100k virtual
edges into the predict-time user-CF cache is likewise null (+0.0007, P=0.30).
The blend's ranking power lives in the item-CF / BPR terms, not the LINE
self-training loop — train with `VIRT_MODE=off` for faster retrains and a
simpler port; the mechanism is kept only for reproducing historical runs.

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
  layers), the BPR trainer, Adam and BCE are framework-specific — everything
  else is numpy/scipy. Port and verify on CPU (RTX 5080/5090 are Blackwell sm_120;
  no reliable evidence of Jittor GPU support on them — use the competition
  server for Jittor GPU runs).
- [ ] Data package B (`data_B`): not yet released by the organizers (confirmed
  by teammate 2026-07-18); `data_A.zip` with its two datasets is everything
  currently available. Rerun the pipeline on `data_B` once it is released.
- [x] Re-run after fix #1 — far above the historical 0.424 (best total 1.3892).
- [x] Negative sampling fully vectorized: one per-epoch pregeneration pass with
  GPU `searchsorted` draws and sorted-key membership rejection (the previous
  per-batch scipy indexing dominated epoch time).
- [x] Similar-user cache build moved to GPU (chunked matmul + `torch.topk`);
  MRR eval scoring switched to the same sparse-matrix path as prediction.

## History

- Original files: `新建文件夹/1.py` + `data_A.zip`; reorganized into this
  project structure on 2026-07-18.
- Original script header note: "21: redo: 0.424"; teammate's estimate 1.36.
  This code reached a combined online total of 1.4320 on 2026-07-21.
