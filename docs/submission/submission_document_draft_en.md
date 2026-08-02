# Submission Document — Temporal Link Prediction (Track 1)

**DRAFT FOR OWNER REVIEW. NOT THE FINAL SUBMISSION DOCUMENT.**
This file is the editable source for the required PDF. It is not named
`提交说明文档.pdf` and must not be submitted until every placeholder below has been replaced by the
owner and the content has been reviewed.

Placeholders still open, all of which require owner input and none of which may be inferred from
project history: `<TEAM_NAME>`, `<A_BOARD_RANK>`, `<A_BOARD_BEST_TOTAL_SCORE>`, `<CONTACT_NAME>`,
`<WECHAT_ID>`, `<PHONE_NUMBER>`, `<SUBMISSION_DATE>`.

---

## 1. Team Information

| Field | Value |
|---|---|
| Team name | `<TEAM_NAME>` |
| A-board rank | `<A_BOARD_RANK>` |
| Best A-board total score | `<A_BOARD_BEST_TOTAL_SCORE>` |
| Contact person | `<CONTACT_NAME>` |
| WeChat | `<WECHAT_ID>` |
| Phone | `<PHONE_NUMBER>` |
| Submission date | `<SUBMISSION_DATE>` |
| Archive | `contest1_<TEAM_NAME>_003.zip` |

The A-board total is the strictly additive sum of the two dataset components. Each component was
measured by uploading a single-member archive in isolation.

---

## 2. Project Overview

### 2.1 The task

Temporal link prediction on two dynamic interaction graphs. Each test query supplies a source node,
a timestamp and 100 candidate destination nodes; the submission assigns a score to every candidate,
and the metric is MRR over the held-out true destination. The causal boundary is strict: no feature
may read an interaction later than the query it scores.

### 2.2 The two scenarios

The two datasets are structurally different, and the difference is measured rather than assumed —
`python main.py --dataset <name> --stage preprocess` reports it from the raw files.

| Property | Dataset 1 | Dataset 2 |
|---|---|---|
| Graph type | non-bipartite (source and destination roles overlap) | bipartite (empty role overlap) |
| Test queries | 61,051 | 153,420 |
| Candidates per query | 100 | 100 |
| Submission member | `dataset1.csv`, 61,051 x 100 | `dataset2.csv`, 153,420 x 100 |

Because the graphs differ structurally, the two chains share their embedding stage and their
learned-ranker stage but diverge afterwards: Dataset 1 uses history-aware ranking with two
deterministic postprocessors, Dataset 2 uses basket-feedback ranking with a matrix-factorisation
geometry, an equality CRF and a final cross-time decoder.

### 2.3 Solution outline

1. **Representation.** Two embedding families are trained per dataset with Jittor: a LINE model
   (first-order and second-order objectives trained jointly) and a BPR-MF model (Bayesian
   personalised ranking with degree-proportional negative sampling). Each is trained twice — once on
   the full training history, once on history truncated at a fixed time cut — so that the ranker can
   be trained on a causally honest replay of the future.
2. **Learned ranking.** A LightGBM LambdaRank model scores candidates from hand-built features
   (collaborative filtering, item-CF over embedding similarity, popularity dynamics, interaction
   history, and embedding dot products). Features are frozen at the cut for training and at the end
   of the training history for inference; forward-extrapolating time features are clipped to the
   training-observed range.
3. **Structural post-processing.** Deterministic rules derived from measured invariants of the data
   generator, applied after ranking. They are not learned and have no free parameters fitted on test
   data.
4. **Final serialisation.** A mandatory output gate asserts shape, finiteness and the `[0, 1]` domain
   before any member is written.

### 2.4 Principal innovations

* **Per-row min-max serialisation of LambdaRank margins.** LambdaRank produces unbounded raw margins
  that straddle zero; the submission format requires `[0, 1]`. The chain applies a per-row min-max,
  which is strictly order-preserving and therefore has no effect on MRR, but is load-bearing because
  both Dataset-1 postprocessors and the output gate assume the unit interval.
* **Structural-invariant post-processing.** Several rules exploit invariants that were measured on
  the training data rather than assumed: for example, a (source, destination) pair never recurs at
  two distinct timestamps in the Dataset-2 training data (0 of 2,264,807 pairs), which makes a
  labelled cross-time answer a negative for other queries of the same source.
* **Basket feedback.** A large fraction of Dataset-2 test rows share a `(source, time)` event. A
  three-pass ranker feeds sibling predictions within an event back as features.
* **Matrix-factorisation sibling geometry.** Pass-2 and pass-3 sibling messages are carried through a
  rank-128 truncated SVD of the split-0 source-by-destination matrix.
* **An equality CRF over row order.** The raw Dataset-2 test row order preserves same-time,
  same-answer runs across sources; a chain CRF over adjacent rows exploits this coupling.
* **A stage-completion contract.** Engineering rather than modelling, but it is what makes the
  reproduction trustworthy; see section 4.4.

---

## 3. Canonical Architecture

### 3.1 Jittor only

The official pipeline is **Jittor only**. All neural training runs through `src/train_line_jt.py`
and `src/train_bpr_jt.py`. There is no backend command-line option, no backend environment variable
and no fallback path. A static import-closure analysis of the packaged tree reaches exactly
`jittor`, `lightgbm`, `numpy`, `pandas`, `scipy` and `tqdm`; PyTorch is not reachable, is not
installed in the reproduction environment, and is not declared in either environment file.

Historical PyTorch implementations of the two trainers exist in the project's version history. They
are **not included in this package**, because shipping a second backend in a Jittor-mandated
competition would create ambiguity about which code produced the result.

### 3.2 Framework-neutral shared components

`src/pipeline_common.py` holds the run-directory and data-root path contracts and the shared data
utilities (history indexing, co-occurrence, similar-user caching, row normalisation). It imports
neither Jittor nor PyTorch. This is deliberate: the ranking stages obtain their utilities from it
without pulling in a deep-learning framework, and a producer and its consumer derive artifact paths
from the same function, so they cannot disagree about where an artifact lives.

### 3.3 What Jittor computes

Jittor owns the model parameters, the loss and the optimiser. Sampling, shuffling and negative
rejection run in seeded NumPy in both trainers. This keeps the framework-dependent surface small and
version-robust, and makes the sampling stream reproducible independently of the framework.

### 3.4 The stage-completion contract

Every stage of the pipeline writes one artifact and, next to it, a completion record
`<output>.done.json`. A stage is treated as already complete **only** when that record proves it:
the producing commit, a digest over the exact source files the stage depends on, the size and SHA256
of every input, a digest of the scientific configuration, the seed, the requested and achieved epoch
counts, the size and SHA256 of the output itself, and the result of a stage-specific validation.

The record is published last, by atomic rename after `fsync`, so its presence implies a validated
artifact. Any mismatch — a missing record, a changed input, a different commit, a different
configuration, a short epoch count, a leftover partial file — **stops the run with a diagnostic**. A
stage is never skipped because its output file happens to exist, and nothing is silently deleted or
repaired.

This matters for reproduction. The neural trainers export their embedding periodically during
training, so an interrupted run leaves a file that looks like a finished export. Under the contract
that file has no completion record and the next run refuses it.

### 3.5 JittorGeometric

JittorGeometric is installed and version-pinned in the official environment, at upstream commit
`ff7d8ffac7bf3d95cc1962e091c52dc5737492d4`. **The final maintained production chain does not
directly import it.** The models used here are a LINE embedding, a BPR-MF embedding and gradient-
boosted ranking over hand-built features; none of those stages requires a graph-neural-network
library. No import was added to the code for the sole purpose of claiming usage.

---

## 4. Code Structure

All paths below are relative to `code/` inside the archive.

| Path | Responsibility |
|---|---|
| `main.py` | the entry point; executes the canonical graph for one dataset |
| `src/canonical_pipeline.py` | the stage graph for both datasets, and the fail-closed runner |
| `src/stage_contract.py` | completion records, atomic publication, artifact validators |
| `src/pipeline_common.py` | path contracts and framework-neutral data utilities |
| `src/train_line_jt.py` | LINE embedding trainer (Jittor) |
| `src/train_bpr_jt.py` | BPR-MF embedding trainer (Jittor) |
| `src/ensemble_predict.py` | embedding loading and collaborative scoring shared by both rankers |
| `src/ranker_ds1.py` | Dataset-1 LambdaRank over 21 features |
| `src/ranker_basket_ds2.py` | Dataset-2 LambdaRank over 18 features, three-pass basket feedback |
| `src/ds2_basket_featurizer.py` | Dataset-2 feature construction and the train-feature cache |
| `src/ds2_mf_basket_pack.py` | Dataset-2 MF sibling-message geometry; the production base matrix |
| `src/crf_promote.py` | Dataset-2 equality CRF and structural-invariant demotions |
| `src/build_ds1_member.py` | Dataset-1 postprocessor chain; writes the final member |
| `src/build_ds2_member.py` | Dataset-2 cross-time exclusivity decode; writes the final member |
| `src/strategies/registry.py` | the ordered, order-sensitive Dataset-1 postprocessor chain |
| `src/strategies/shared/frozen_ops.py` | frozen numeric primitives and the submission-format gate |
| `src/strategies/ds1/source_slate_recurrence.py` | Dataset-1 postprocessor 1 |
| `src/strategies/ds1/test_graph_reciprocity.py` | Dataset-1 postprocessor 2 |
| `src/strategies/ds2/cross_time_exclusivity.py` | the Dataset-2 final decoder |
| `configs/production.json` | machine-readable description of the production chain |
| `tools/submission/package_component.py` | independent verifier of submission-format members |

### 4.1 Dataset-1 stage graph (16 stages)

```
train.csv
  +- ds1_line_full        train_line_jt.py                    -> outputs/dataset1-novirt/line_latest_emb.csv
  +- ds1_line_cut         train_line_jt.py  LINE_TIME_MAX      -> outputs/dataset1-novirt-tmax1.1548e+08/…
  +- ds1_bpr_serve_s{5}   train_bpr_jt.py   5 seeds            -> outputs/dataset1-bpr-t0.25[-sN]/bpr_emb.npy
  +- ds1_bpr_cut_s{5}     train_bpr_jt.py   5 seeds + cut      -> outputs/dataset1-bpr-t0.25-tmax…[-sN]/…
  +- ds1_bpr_innov_serve  train_bpr_jt.py   BPR_INNOV=1        -> outputs/dataset1-bpr-innov-t0.25/…
  \- ds1_bpr_innov_cut    train_bpr_jt.py   BPR_INNOV=1 + cut  -> outputs/dataset1-bpr-innov-t0.25-tmax…/…
        |
  ds1_ranker              ranker_ds1.py                       -> outputs/dataset1-ensemble/result_ranker.csv
        |
  ds1_member              build_ds1_member.py                 -> outputs/members/dataset1.csv
```

### 4.2 Dataset-2 stage graph (17 stages)

```
train.csv
  +- ds2_line_full / ds2_line_cut          train_line_jt.py   -> outputs/dataset2-novirt[-tmax…]/…
  \- ds2_bpr_{serve,cut}_s{5}              train_bpr_jt.py    -> outputs/dataset2-bpr[-tmax…][-sN]/…
        |
  ds2_ranker              ranker_basket_ds2.py  -> outputs/dataset2-ranker/ranker_basket3_dataset2.csv
        |
  ds2_mf_pack             ds2_mf_basket_pack.py -> outputs/dataset2-ranker/mf_basket3_dataset2.csv
        |
  ds2_ds1_passthrough     (in-process)          -> outputs/dataset2-crf/ds1_passthrough.zip
        |
  ds2_crf                 crf_promote.py        -> outputs/dataset2-crf/ds2_crf_intermediate.zip
        |
  ds2_member              build_ds2_member.py   -> outputs/members/dataset2.csv
```

### 4.3 Training and inference logic

**Cut-split protocol, both datasets.** The learned ranker cannot be trained on the test labels, so it
is trained on a replay of the recent past: interactions before a fixed time cut form the feature
history (split 0), interactions after it supply the positive examples (split 1). For each positive,
negatives are drawn from the candidate pool that the test file exposes for that source, so the
training query has the same shape as a test query. Features are computed with all history frozen at
the cut.

At inference the identical feature code runs with history frozen at the end of the training data,
against the real test candidates. Two forward-extrapolating time features are clipped to the range
observed during training, so the model never sees an out-of-support value.

**Embedding training.** LINE is trained on bidirectional real edges with a joint objective
`loss1 + 0.5 * loss2` over a first-order pair table and a second-order node/context table pair;
negatives are drawn by rejection sampling against the observed edge set. BPR-MF is trained on
`-log sigmoid(e_src . e_dst - e_src . e_neg)` with degree^0.75 negative sampling.

---

## 5. Environment

| Component | Value |
|---|---|
| Operating system | Ubuntu 22.04 |
| GPU | NVIDIA RTX 4090 (compute capability 8.9) |
| NVIDIA driver | 580.76.05 (validated) |
| CUDA Toolkit | 12.4 (`/usr/local/cuda` -> `/usr/local/cuda-12.4`, nvcc 12.4.131) |
| Python | 3.10 |
| Jittor | 1.3.10 |
| JittorGeometric | commit `ff7d8ffac7bf3d95cc1962e091c52dc5737492d4` (setup.py version 2.0.0) |
| NumPy | 1.26.4 |
| Pandas | 2.2.3 |
| SciPy | 1.15.2 |
| scikit-learn | 1.6.1 |
| LightGBM | 4.5.0 |
| tqdm | 4.67.1 |
| PyTorch | **not required and not installed** |

### 5.1 Installation

```bash
conda env create -f environment.yaml
conda activate jittor-link-prediction-inspect

# JittorGeometric is not published on PyPI and upstream publishes no release
# tags, so it is pinned by commit hash and installed as a separate step.
pip install --no-build-isolation --no-deps \
  git+https://github.com/AlgRUC/JittorGeometric.git@ff7d8ffac7bf3d95cc1962e091c52dc5737492d4
```

`requirements.txt` installs the same versions into an existing Python 3.10 interpreter. No step
requires a credential, a token or a private repository.

`--no-deps` prevents a transitive requirement from silently upgrading the pinned numerical stack.
Upstream declares no `install_requires`, so nothing is suppressed by it.

**NumPy is held below 2.0 deliberately.** The frozen postprocessors depend on exact float64
comparison and stable argsort semantics, and the accepted member hashes were produced under the 1.x
behaviour.

### 5.2 Required Jittor runtime settings

```bash
export conv_opt=1
export use_mkl=0
```

These are **not** performance tuning; they are required on this image. Jittor 1.3.10 loads the
cuDNN 8 component libraries by name (`libcudnn_ops_infer.so` and siblings), and cuDNN 9 merged them
into `libcudnn.so.9`; `conv_opt=1` restricts Jittor to the cuBLAS path it can resolve. `use_mkl=0`
avoids a DNNL verification binary that does not terminate on this host.

This was validated directly on the target host: with the two variables unset, the Jittor smoke test
**aborts** (exit 134, `terminate called without an active exception`); with them set, the same smoke
test passes — CUDA enabled, `sm_89` detected, matrix multiplication, autograd and an Adam step all
succeed.

The entry point applies both variables to every stage it launches, prints them at the start of a
run, and records them in every completion record. If the surrounding environment already sets either
variable to a conflicting value, the run **stops** rather than overriding it.

---

## 6. Official Data Layout

The archive contains **no competition data**. The expected layout is:

```
<data-root>/
\-- data_A/
    +-- dataset1/
    |   +-- train.csv        columns: src, dst, time
    |   \-- test.csv         columns: src, time, c1 … c100
    \-- dataset2/
        +-- train.csv
        \-- test.csv
```

The default `<data-root>` is `code/data`; `--data-root` points a run anywhere else. No stage reads
any file outside this root except the artifacts it produces itself.

**No test-label ground truth is used anywhere.** Test labels are not available to this code and the
training protocol does not require them: the ranker is trained on a cut-split replay of the training
data. **No frozen prediction is used as a computational shortcut** — no reference prediction, no
previously submitted member and no historical score matrix is an input to any stage.

---

## 7. Complete Run Procedure

### 7.1 Order

```bash
python main.py --dataset dataset1
python main.py --dataset dataset2
```

**The order is required.** `crf_promote.py` emits a two-member container and therefore needs
Dataset-1 member bytes as passthrough input. Those bytes are never read as scores — the Dataset-2
member builder extracts `dataset2.csv` alone — but they must be present. The entry point supplies
them from a Dataset-1 member produced under a valid completion record. If none exists, the Dataset-2
run stops with an explicit message; it does not fabricate a placeholder and does not substitute a
frozen member. A validated member from elsewhere may be supplied with `--ds1-member <path>`.

### 7.2 Commands

| Purpose | Command |
|---|---|
| Show the graph and each stage's state; execute nothing | `python main.py --dataset dataset1 --stage plan` |
| Full run | `python main.py --dataset dataset1` |
| Validated resume (the default) | `python main.py --dataset dataset1` — completed stages with a valid record are reused |
| Refuse all reuse | `python main.py --dataset dataset1 --fresh` |
| Raw-data root | `--data-root /path/to/data` |
| Artifact root | `--output-root /path/to/outputs` |
| Stage logs | `--log-dir /path/to/logs` (default `<output-root>/_logs`) |
| Inspect the raw data | `python main.py --dataset dataset1 --stage preprocess` |
| Describe the production chain | `python main.py --stage describe` |

### 7.3 Outputs

| Artifact | Path |
|---|---|
| Dataset-1 base score matrix | `<output-root>/dataset1-ensemble/result_ranker.csv` |
| Dataset-1 final member | `<output-root>/members/dataset1.csv` (61,051 x 100) |
| Dataset-2 base score matrix | `<output-root>/dataset2-ranker/mf_basket3_dataset2.csv` |
| Dataset-2 final member | `<output-root>/members/dataset2.csv` (153,420 x 100) |
| Completion record, per stage | `<artifact>.done.json` |
| Stage logs | `<output-root>/_logs/<stage_id>.log` |

The final A-board archive is these two members, `dataset1.csv` and `dataset2.csv`, at the root of one
ZIP. Note that the two accepted members carry different line endings — `dataset1.csv` CRLF and
`dataset2.csv` LF — because each is in the exact form that was scored; neither may be normalised.

### 7.4 Failure behaviour

The run stops at the first stage that fails and returns a non-zero exit code. Subprocess failures
propagate. No completion record is written for a stage that exited non-zero, produced a partial
artifact, failed validation or completed fewer epochs than requested. A stopped run leaves its
artifacts in place — they are the evidence — and prints the stage, the reason code and the
properties that differed.

---

## 8. Key Hyperparameters

All values are the production defaults, read from the source and from `configs/production.json`.

### 8.1 LINE embedding (`train_line_jt.py`)

| Parameter | Value |
|---|---|
| Total embedding dimension | 400 (two 200-dimensional halves) |
| Epochs | 400 |
| Batch size | 1,024 |
| Negatives per positive | 5 |
| Negative distribution | uniform |
| Optimiser | Adam, learning rate 1e-4 |
| Loss | `BCE(first-order) + 0.5 * BCE(second-order)` |
| Gradient-norm clip | 1.0 |
| Seed | 42 |
| Export precision | 6 decimals; export every 10 epochs |
| Dataset-1 time cut | `LINE_TIME_MAX=115480000` for the cut run |
| Dataset-2 time cut | `LINE_TIME_MAX=1261958400` for the cut run |
| Virtual-edge self-training | disabled (measured no-op) |

### 8.2 BPR-MF embedding (`train_bpr_jt.py`)

| Parameter | Value |
|---|---|
| Dimension | 256 |
| Epochs | 120 |
| Batch size | 8,192 |
| Optimiser | Adam, learning rate 3e-3, weight decay 1e-6 |
| Negative sampling | destination degree^0.75 |
| Seeds | 42, 123, 777, 2024, 31337 |
| Recency weighting | `BPR_TAU_FRAC=0.25` on Dataset 1; 0 on Dataset 2 |
| Innovation-only variant | Dataset 1 only (`BPR_INNOV=1`, first occurrence of each pair) |

### 8.3 Dataset-1 ranker (`ranker_ds1.py`)

| Parameter | Value |
|---|---|
| Model | LightGBM `LGBMRanker` |
| Objective / metric | `lambdarank` / NDCG@10 |
| Estimators | 400 |
| Learning rate | 0.05 |
| Leaves | 31 |
| Minimum child samples | 100 |
| `label_gain` | `[0, 1]` |
| Random state | 42 |
| Features | 21 |
| Time cut | 115,480,000 |
| Negatives per training query | 99 |
| Negative-sampling seed | 20260727 |
| Clipped time features | `time_gap`, `last_gap` |
| Serialisation | per-row min-max into `[0, 1]` |

### 8.4 Dataset-2 ranker and geometry

| Parameter | Value |
|---|---|
| Model | LightGBM `LGBMRanker`, `lambdarank`, NDCG@10 |
| Estimators / learning rate / leaves | 400 / 0.05 / 31 |
| Minimum child samples | 100 |
| Feature fraction | 0.8 |
| Features | 18 |
| Basket feedback passes | 3 |
| Time cut | 1,261,958,400 |
| Seeds | 42, 123, 777, 2024, 31337 |
| Negatives per training query | 99 |
| MF geometry | rank-128 truncated SVD of the split-0 source x destination matrix |

### 8.5 Dataset-2 CRF (`crf_promote.py`)

| Parameter | Value |
|---|---|
| `--tau` | 0.20 |
| `--B` | 70 |
| `--W` (band half-width) | 1 |
| `--p` (neighbour-marginal exponent) | 1.0 |
| `--eta` (candidate-reliability exponent) | 0.0 |
| Triple rule | enabled |
| Pair rule | **disabled** (`--no-pair`) — it double-counts the chain the CRF already models |
| Cross-time zero-repeat exclusion | enabled (`--zr-exclude`) |
| Duplicate-slate demotion | enabled (`--demote-dups`) |
| Same-time structural exclusion | enabled (`--st-exclude`) |

### 8.6 Post-processing

| Stage | Effect on the member |
|---|---|
| `ds1_source_slate_recurrence` | 3,653 rows |
| `ds1_test_graph_reciprocity` | 1,365 rows |
| `xte_cross_time_exclusivity_decode` (Dataset 2) | 7,815 rows / 15,630 cells |

The Dataset-1 chain is **order-sensitive**: reciprocity reads the rank-2 candidate of the
recurrence-adjusted matrix, so swapping the two produces a different file. The order above is the
one that was adjudicated online.

---

## 9. How the Final Submission Files Are Produced

1. The ranker writes a base score matrix — one row per test query, 100 values per row, `%.6f`.
   This is an **intermediate**, not a submission file.
2. Dataset 1: `build_ds1_member.py` applies the two frozen postprocessors in order and writes
   `dataset1.csv`.
3. Dataset 2: `crf_promote.py` applies the CRF and the invariant demotions, then
   `build_ds2_member.py` applies the cross-time exclusivity decode and writes `dataset2.csv`.
4. Before any byte is serialised, `validate_submission_matrix` asserts the expected shape, that every
   value is finite, and that every value lies within `[0, 1]`. **It reports and aborts; it never
   clamps.** A member that fails this gate is a signal about the chain that produced it, and silently
   repairing the values would destroy the evidence and change the ranking.
5. The serialised file is then checked structurally — row count and per-row field count — which
   catches a truncated or ragged write that an in-memory check cannot see.
6. The Dataset-2 decoder is **byte-preserving**: it swaps the two affected score *tokens* in the base
   member's own bytes rather than re-serialising 15,342,000 float values, so every unaffected cell is
   the exact byte the base produced. The token swap is then verified to agree with the float path on
   every row's candidate ranking.
7. The final A-board archive contains exactly `dataset1.csv` and `dataset2.csv` at its root.

---

## 10. Runtime and Resource Expectations

Measured on the validation host (RTX 4090 24 GB, Xeon Gold 6430, 16 cores, 120 GiB), not estimated.

| Stage | Device | Measured |
|---|---|---|
| Dataset-1 LINE, full history | GPU | 2,309 s |
| Dataset-1 LINE, cut history | GPU | 1,802 s |
| Dataset-1 BPR x 12 runs | GPU | included in the total below |
| Dataset-1 ranker | CPU | included below |
| **Dataset-1, raw data to member** | | **1 h 51 m 42 s** |
| Dataset-2 train-feature cache | CPU | 10,115 s |
| Dataset-2 MF basket pack | CPU | 8,837 s |
| Dataset-2 CRF | CPU | 120 s |
| Dataset-2 member decode | CPU | 9 s |
| **Dataset-2, raw data to member** | | **approximately 11–13 h** |

Peak disk: approximately 5 GB of artifacts per dataset, plus a 1.92 GiB Dataset-2 train-feature
cache. Peak GPU memory is modest; the embedding tables dominate and fit comfortably in 24 GB.

**Resumability.** Every stage boundary is a resume point, governed by the completion contract. There
are no resume points *inside* a stage: an interrupted trainer or an interrupted feature build restarts
that stage from the beginning. The longest single unresumable stage is the Dataset-2 feature build at
roughly 2.8 h.

---

## 11. Reproducibility and Nondeterminism

* Both raw-to-final lineages have been executed successfully from the official raw data on the target
  environment.
* **GPU training is not guaranteed byte-deterministic across independent runs.** Jittor kernel
  scheduling on the GPU is not bit-reproducible, and the model state of the original A-board runs was
  not preserved.
* Consequently, **a fresh reproduction may not be byte-identical to the historical A-board member**.
  A measured comparison of one clean Dataset-1 reconstruction against the historical member gave
  95.60% top-1 agreement and mean per-row Spearman 0.873. This is the expected behaviour of a
  retrained pipeline, not a defect.
* **No claim of hidden-score equivalence is made.** Test labels are unavailable, so no offline
  statement about the score of a reconstruction is possible, and none is made here.
* Sampling, shuffling and negative rejection run in seeded NumPy and are reproducible independently
  of the framework.
* Completion records prevent the opposite failure — silently reusing a stale artifact from a
  different commit, different inputs or a different configuration and presenting the result as a
  reproduction.

---

## 12. Known Issues and Notes

1. **Skip-on-file-existence is removed.** Earlier drivers treated a stage as complete whenever its
   output file existed. That is no longer possible on the canonical path: reuse requires a valid
   completion record.
2. **The historical 1.92 GiB Dataset-2 feature cache is not reusable.** It was produced before the
   contract existed and has no completion record, so a Dataset-2 run rebuilds it (~2.8 h). It is not
   adopted on faith, because writing a record for an artifact of unknown provenance would defeat the
   contract.
3. **Line terminators.** The Dataset-1 *intermediate* ranker CSV is written with the platform default
   terminator (LF on Linux, CRLF on Windows). Parsed values are unaffected, and the final member is
   unaffected because its writer pins CRLF explicitly. Only the intermediate file's bytes differ
   across platforms.
4. **JittorGeometric is installed and pinned but not directly imported** by the production chain
   (section 3.5).
5. **The documented Jittor runtime is required, not optional.** Without `conv_opt=1` and `use_mkl=0`
   the Jittor smoke test aborts on this image (section 5.2).
6. **No PyTorch dependency.** The canonical package contains no Torch import and the environment does
   not install Torch.
7. **Two entry-point stages are informational in the package.** `--stage describe` and
   `--stage package --verify` read hash anchors that refer to the accepted A-board archive, which is
   not redistributable and is not included. Reproduction does not use them; `--stage run` is
   self-contained.

---

## 13. A/B Consistency and Compliance

* The submitted scientific pipeline is **consistent with the frozen A-board algorithm**. Model
  architectures, features, hyperparameters, seeds, ranking rules, normalisation semantics, CRF logic
  and decoder semantics are unchanged.
* The engineering changes since the A board concern the Jittor migration, path-contract safety,
  artifact validation, execution provenance and reproducibility. None of them alters a computed
  value.
* **No test ground truth is used or leaked.** The ranker is trained on a cut-split replay of the
  training data.
* **No frozen prediction is used as a computational shortcut.** No reference prediction and no
  previously accepted member is an input to any stage.
* No credentials, keys, tokens or private material are included in this package.
* B-board operation will remain consistent with the algorithm and configuration documented here;
  any adaptation required by the B-board data scale will be documented explicitly.

---

## 14. Verification a Reviewer Can Perform

```bash
# 1. The graph, and why each stage is in its current state. Executes nothing.
python main.py --dataset dataset1 --stage plan

# 2. Confirm the raw data is read correctly and the graph type is as documented.
python main.py --dataset dataset1 --stage preprocess

# 3. Full reproduction.
python main.py --dataset dataset1
python main.py --dataset dataset2

# 4. Inspect the provenance of any produced artifact.
cat <output-root>/members/dataset1.csv.done.json
```

Every completion record states the commit, the command, the environment, the identity of every input
and the identity of the output. A reviewer can therefore establish exactly what produced each
artifact without trusting this document.
