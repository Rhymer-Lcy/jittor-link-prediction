# data_B rerun runbook

End-to-end sequence from a freshly released `data_B.zip` to validated
submission packs, distilled from the data_A campaign (online best 1.51309,
2026-07-23). Written so the rerun is mechanical under deadline pressure.

## Principles

- **Run in a fresh clone.** Output dirs are keyed by `DATASET` only
  (`outputs/dataset1`, `outputs/dataset2-bpr-s123`, ...), so an in-place
  data_B run would clobber the data_A artifacts. Clone, drop the data in,
  and export `DATA_PACK=data_B` for every command below (it defaults to
  `data_A`; all scripts resolve data through `train_line.py`'s
  `DATA_DIR = data/<DATA_PACK>/<DATASET>`).
- **Structural exploits are data_A generator fingerprints, not laws.** The
  row-order chain (triple / pair / equality-CRF), the basket passes, and the
  footprint features earned their keep on data_A evidence. Each has a gate
  below (stage 1) that must be re-measured on data_B before its component is
  applied. Blends and the ranker are portable; the exploits are conditional.
- **Never sort or re-order `test.csv`.** The raw row order is load-bearing
  for the entire row-order chain.

## Stage 0 — data audit (minutes)

1. Unzip to `data/data_B/dataset1/` and `data/data_B/dataset2/`
   (`train.csv` + `test.csv` each).
2. Schema check: train `src,dst,time` (+ `split` on the ds2 analogue), test
   `src,time,c1..c100`. Confirm which dataset is the repeat-heavy short-horizon
   one (data_A ds1: 66% repeat, 61-day test span) and which is the zero-repeat
   long-horizon one (data_A ds2: 0% repeat, 1-year span, split column) — the
   chains below attach to the REGIME, not the file name.
3. Record: edge counts, node id max, test row counts, candidate universe size
   (distinct ids in candidate columns; data_A: ds1 23,852 / ds2 110,368), test
   time span vs train max.

## Stage 1 — invariant gates (label-free, on the new files; ~30 min)

Re-measure on data_B before enabling each conditional component:

| Gate | Statistic (data_A value) | Component enabled if it holds |
|------|--------------------------|-------------------------------|
| G1 repeat regime | split-tail replay: share of next-interactions repeating a past partner (ds1 66% / ds2 0%) | ds1: `HIST_BOOST` + masked iBPR; ds2: `MASK_HISTORY` |
| G2 basket | share of test rows sharing a `(src,time)` event with another row (ds2 75.8%, ds1 0.2%) | ranker passes 2/3 (basket feedback) |
| G3 row-order adjacency | mean adjacent candidate-slate intersection vs offset>=5 placebo (0.1771 vs 0.0906 = uniform 100^2/universe) | equality-CRF |
| G4 hard-rule mass | strict triple windows and pair links vs offset placebos (2242 triples vs 8/4 placebo, FDR 0.36%; 2716 pairs, FDR ~7.5%) | triple / pair rules |
| G5 footprint | split-1 replay cfreq separation truth-vs-negatives (ds2 ~1.16, ds1 1.039 = drowned) | cfreq/popratio stay in the ranker (they are harmless if weak) |

G3/G4 placebo method: recompute the same statistic after shifting one side of
each pair by >=5 rows; the excess over placebo is the exploitable mass. If G3
fails but G4 holds, ship hard rules without the CRF; if both fail, ship the
raw ranker base. Scratchpad references: `verify_roworder_ds2.py`,
`replay_harness_ds2.py` (data_A session, 2026-07-23).

## Stage 2 — dataset1-regime chain (~1-2 h GPU, serial)

```bash
# LINE, two runs (uniform d400 + pop075 d512)
DATA_PACK=data_B DATASET=dataset1 python src/train_line.py
DATA_PACK=data_B DATASET=dataset1 NEG_DIST=pop075 EMB_DIM=512 python src/train_line.py
# -> outputs/dataset1, outputs/dataset1-negpop-d512

# BPR ens5, tau=0.25span recency
for S in 42 123 777 2024 31337:
  DATA_PACK=data_B DATASET=dataset1 BPR_TAU_FRAC=0.25 SEED=$S python src/train_bpr.py
# -> outputs/dataset1-bpr-t0.25[-s<seed>]

# innovation-only BPR ens5 (first-occurrence edges, W_IBPR=12 default)
for S in 42 123 777 2024 31337:
  DATA_PACK=data_B DATASET=dataset1 BPR_INNOV=1 BPR_TAU_FRAC=0.25 SEED=$S python src/train_bpr.py
# -> outputs/dataset1-bpr-innov-t0.25[-s<seed>]

# serve (defaults encode the production weights incl. W_IBPR=12)
DATA_PACK=data_B DATASET=dataset1 python src/ensemble_predict.py \
    --runs outputs/dataset1 outputs/dataset1-negpop-d512
```

`VIRT_MODE=off` is safe for both LINE runs (validated no-op, faster). Weights
are data_A-tuned; sanity-check on data_B with the split-tail holdout eval
(`--eval`) but do NOT retune by training on tails (the LambdaRank meta-blend
lesson: tail-trained rankers regressed online).

## Stage 3 — dataset2-regime chain (~4-5 h)

```bash
# CUT = max(time) over split==0 rows of the new train.csv (data_A: 1261958400)
# serve-side embeddings (full train)
DATA_PACK=data_B DATASET=dataset2 VIRT_MODE=off python src/train_line.py
for S in 42 123 777 2024 31337:
  DATA_PACK=data_B DATASET=dataset2 SEED=$S python src/train_bpr.py

# train-side embeddings (split0-frozen)
DATA_PACK=data_B DATASET=dataset2 VIRT_MODE=off LINE_TIME_MAX=<CUT> python src/train_line.py
for S in 42 123 777 2024 31337:
  DATA_PACK=data_B DATASET=dataset2 BPR_TIME_MAX=<CUT> SEED=$S python src/train_bpr.py

# 18-feature LambdaRank + basket passes (drop passes 2/3 if G2 fails)
DATA_PACK=data_B DATASET=dataset2 python src/ranker_basket_ds2.py
# -> outputs/dataset2-ranker/ranker_basket3_dataset2.csv

# row-order postprocessor + packaging (only the gates that passed)
DATA_PACK=data_B python src/crf_promote.py \
    --base outputs/dataset2-ranker/ranker_basket3_dataset2.csv \
    --ds1-from outputs/submissions/<ds1 pack>.zip \
    --out outputs/submissions/submission_data_B.zip \
    --data data/data_B/dataset2
```

`ranker_basket_ds2.py` retrains on data_B's split-1 labels natively (that is
its design — full-horizon labels, features frozen at CUT). The CRF (tau=0.25,
B=100) sat on a broad plateau on data_A; if G3 passes but the adjacency
density differs a lot, re-tune on the split-1 replay (odd-day tune / even-day
validate) before shipping.

## Stage 4 — pack validation (always, minutes)

- Shape rows x 100, no NaN/inf, range [0,1], `%.6f`, HEADERLESS, no
  zero-variance rows.
- **md5 lineage audit** of every member CSV against its intended source file
  (`hashlib.md5`); the data_A campaign shipped a refuted ds1 file for seven
  packs because "byte reuse" was assumed instead of checked.
- When only one dataset changed, the other member must be byte-identical to
  the previous pack (isolated online delta).
- Top-1 flip rate vs the previous pack: report it; a surprise magnitude means
  a config drifted.

## Known transfer discounts (for reading data_B online results)

- Feature-class changes: offline delta x ~0.41.
- More-data / specialization changes (both-calibers-agree): x ~0.9-1.3.
- Test-file-observed exploits (row-order): x ~1.0 scaled by the measured
  density ratio (CRF folded by 0.72 = test/replay shared-pair density ratio).
- ds2-regime recency training knobs: online-unverifiable, keep OFF.
