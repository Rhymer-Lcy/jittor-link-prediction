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
`python main.py --dataset dataset1 --stage preprocess` (and the same with `dataset2`) reports it
from the raw files.

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
  reproduction trustworthy; see section 4.2.

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

## 4. Code Structure and Execution Architecture

All paths are relative to `code/` inside the archive. Every statement in this section was verified
against the packaged source; §4.10 lists all 26 packaged files and their reachability.

**Terms used throughout, defined once.**

| Term | Meaning |
|---|---|
| **serve model** (or full-history model) | a model trained on the complete training history, used to score the real test queries |
| **cutoff model** | the same model trained only on interactions at or before a fixed timestamp, used to build causally honest training features |
| **cut-split replay** | the scheme that manufactures ranker training labels without test labels: interactions before the time cut form the feature history, interactions after it supply the positives |
| **basket event** | a set of Dataset-2 test rows sharing the same `(source, time)` pair; they are answered together by the generator |
| **sibling feedback** | using the scores assigned to the *other* rows of the same basket event as features for a later ranking pass |
| **MF sibling geometry** | a rank-128 truncated SVD of the source x destination interaction matrix, used to carry sibling messages. Distinct from the BPR-**MF** embedding model |
| **equality CRF** | a chain conditional random field over adjacent test rows that couples candidates the raw row order shows to be related |
| **cross-time exclusivity** | the measured invariant that a fixed `(source, answer)` pair never recurs at two distinct timestamps |
| **completion record** | `<artifact>.done.json`, the document that proves a stage ran to completion; see §4.2 |

### 4.1 Entrypoints and orchestration

Control flows strictly downward. Each level does one job and delegates:

```
run_all.py                one command; runs both datasets in the required order
  |
  +-> main.py             one dataset; selects a stage mode and a run context
        |
        +-> src/canonical_pipeline.py    the stage graph and the fail-closed runner
              |
              +-> stage modules (trainers, rankers, CRF)   one subprocess per stage
                    |
                    +-> src/build_ds1_member.py / src/build_ds2_member.py
                          |
                          +-> outputs/members/dataset{1,2}.csv  + .done.json
```

#### `run_all.py` — one-command wrapper

*Why it exists*: so a reviewer can reproduce everything with one command.
*Invoked by*: the reviewer. *Invokes*: `main.py`, twice.

It imports **only the Python standard library** (`argparse`, `subprocess`, `sys`, `pathlib`) and
contains no model, feature, ranking, CRF or decoding logic. It resolves the packaged directory from
`Path(__file__).resolve().parent`, so it works from any working directory, and launches children
with `sys.executable` so the interpreter running it is the interpreter running the pipeline.

`build_command()` constructs the child command and `preflight()` performs the pre-run checks;
both are pure functions, which is what makes the wrapper testable without executing anything.

| Behaviour | Implementation |
|---|---|
| Order | Dataset 1, then Dataset 2 — required, because `crf_promote.py` writes a two-member container and needs Dataset-1 member bytes as passthrough input |
| Dataset-1 failure | Dataset 2 is **not started**; the child's exit code is returned unchanged |
| Argument forwarding | `--data-root`, `--data-pack`, `--output-root`, `--log-dir`, `--fresh`, `--quiet` are passed to both datasets |
| Validated resume | the default, inherited from `main.py`: a stage is reused only when its completion record proves it |
| `--fresh` | forwarded to both datasets; refuses all reuse |
| `--plan` | adds `--stage plan` to both children; **no computation occurs** |
| Preflight | verifies `main.py` exists, that `--data-root` and `--output-root` differ, and that both datasets' `train.csv` and `test.csv` are present; exits 2 with a named reason otherwise |
| Output | child stdout/stderr stream to the console as they are produced, never captured and hidden |
| Interruption | returns 130; the child is signalled rather than orphaned |

#### `main.py` — per-dataset entry point

*Invoked by*: `run_all.py`, or the reviewer directly. *Invokes*: `canonical_pipeline`.

Parses `--dataset {dataset1,dataset2,all}` and `--stage {run,plan,describe,preprocess,postprocess,
package}`, plus `--data-root`, `--data-pack`, `--output-root`, `--log-dir`, `--config`, `--fresh`,
`--ds1-member`, `--verify`, `--quiet`. `run` is the default stage.

`contexts()` turns the arguments into one `RunContext` per dataset — the object that carries the
resolved data root, output root, log directory and resume policy. `stage_run()` then calls
`canonical_pipeline.execute()`; `stage_plan()` calls `canonical_pipeline.plan()` and executes
nothing. `stage_describe()` prints the chain from `configs/production.json`; `stage_preprocess()`
loads the raw files and reports the measured graph type; `stage_postprocess()` re-runs only the
member builders; `stage_package()` invokes the submission verifier.

There is **no backend option**. The graph names the Jittor trainers and nothing else can be
substituted.

#### `src/canonical_pipeline.py` — the stage graph and the runner

*Invoked by*: `main.py`. *Invokes*: every stage module, as a subprocess.

`dataset1_stages()` and `dataset2_stages()` declare the graphs — 16 and 17 `StageSpec` objects, in
execution order. `build_stages()` selects by dataset. Each specification carries the stage id, the
exact command, the environment additions, the declared inputs and output, the scientific
configuration, the relevant source files, the requested epoch count and the validator.

`run_stage()` executes one stage in a fixed, load-bearing order:

1. `stage_contract.evaluate()` decides `RUN`, `REUSE` or `STOP` from the completion record alone;
2. the stage runs, writing into an isolated `<output>.part` wherever it takes its output as an
   argument;
3. the staged artifact is validated;
4. it is published atomically by `os.replace`;
5. the completion record is written — last.

`stage_environment()` applies the Jittor runtime settings `conv_opt=1` and `use_mkl=0` and exports
this run's `DATA_ROOT` and `OUTPUTS_ROOT` to the child. If the surrounding environment already sets
either Jittor variable to a conflicting value the run **stops** rather than overriding it.
`_run_subprocess()` streams the child's output to both the console and `<output-root>/_logs/
<stage_id>.log`. `execute()` loops over the graph and returns non-zero at the first failure.

**Dataset-2's dependency on Dataset 1** is implemented by `_passthrough_source()`: it requires a
Dataset-1 member that carries a valid completion record, or an explicit `--ds1-member`. If neither
exists the run stops with an explicit message. No placeholder is generated and no frozen member is
substituted.

### 4.2 The canonical stage graph and the execution contract

#### `src/stage_contract.py` — completion records

*Why it exists*: because a file existing on disk does not prove the stage that should have produced
it ever finished. *Invoked by*: `canonical_pipeline`, and directly by `ds2_basket_featurizer` for
the feature cache.

Schema version 1. The record lives beside its artifact as `<output>.done.json`, so the two travel
together and deleting an output directory cannot leave an orphaned claim.

Principal API: `StageSpec` (the dataclass describing a stage), `evaluate()` (the reuse decision,
returning a `Decision`), `complete_stage()` (validate then publish), `publish_record()` (the atomic
write), `atomic_output()` (a context manager yielding a `.part` path), and `quarantine()` (the only
sanctioned way past a stop). Validators are `validate_embedding_csv`, `validate_embedding_npy`,
`validate_score_matrix_csv`, `validate_member_csv` and `validate_npz_cache`.

A record binds: `stage_id`, `dataset`, `artifact_kind`, `producing_commit`, a `code_identity`
digest over the exact source files the stage depends on, the `command` and `command_env`, the
`environment` (interpreter and installed package versions, read from metadata — nothing is
imported), every input's path, size and SHA256, the `config` and its digest, the `seed`, the
`requested_units` and `achieved_units` (epochs), the output's size and SHA256, the `validation`
result, the `exit_code` and both timestamps.

Reuse requires **all** of: the output exists; a record exists; it parses and declares a supported
schema version; `exit_code` is 0; every input still hashes to its recorded value and the input set
is unchanged; the configuration digest matches; the code digest **and** the producing commit match;
the output's size and SHA256 match; achieved epochs equal requested epochs; validation passed. Any
single mismatch is a **stop**, never a silent re-run and never a repair.

*Publication order and interruption*: the record is written only after the artifact is validated
and published, by `fsync` then atomic rename. A stage interrupted at any earlier point therefore
leaves no record, and the next run refuses its leftovers. This matters concretely: the LINE trainer
exports its embedding every 10 epochs, so an interrupted run leaves a file that *looks* like a
finished export — under the contract that file has no record and is refused.

#### The two graphs

Dataset 1, 16 stages: two LINE runs (serve, cutoff); ten BPR runs (five seeds x serve/cutoff); two
innovation-only BPR runs; the ranker; the member builder.

Dataset 2, 17 stages: two LINE runs; ten BPR runs; the ranker; the MF basket pack; the Dataset-1
passthrough archive; the CRF; the member builder.

§4.9 enumerates all 33 with their inputs, outputs and types.

### 4.3 Shared path, data and numeric utilities

#### `src/pipeline_common.py` — the path contract and framework-neutral utilities

*Why it exists*: so a producer and its consumer can never disagree about where an artifact lives.
*Invoked by*: both trainers, both rankers, the featurizer, the MF pack, `canonical_pipeline`.

It imports **neither Jittor nor PyTorch**. The ranking stages obtain their utilities from it without
pulling in a deep-learning framework.

*Path contract*: `data_root()` and `data_dir(dataset, pack)` resolve raw data, honouring
`DATA_ROOT`; `outputs_root()` resolves the artifact root, honouring `OUTPUTS_ROOT` in preference to
any argument; `line_run_dir()`, `bpr_run_dir()` and `ranker_dir()` derive a run directory from its
scientific parameters — seed, negative distribution, embedding dimension, time cutoff, innovation
flag, holdout flag. Because the trainer that *writes* `line_run_dir("dataset2", time_max=CUT)` and
the ranker that *reads* it call the same function, they cannot be pointed at different roots by a
redirected run; a literal path in either place would break exactly that guarantee.

*Data utilities*: `split_train_val_by_tail()`; `build_history_index()` and `get_hist_before_time()`
(per-source interaction history, sorted by time, queried by cutoff — this is what enforces the
causal boundary); `count_in_history()`; `build_cooc()` and `cooc_scores()` (popularity-normalised
co-occurrence collaborative filtering); `cache_dict_to_matrix()` and `batch_sim_score()`;
`build_sim_cache()` (two-band decayed similar-user cache by chunked cosine top-k, in NumPy);
`rownorm()` (divide a row by its maximum).

#### `src/ensemble_predict.py` — embedding loading and collaborative scoring

*Invoked by*: `ranker_ds1`, `ranker_basket_ds2`, `ds2_basket_featurizer`, `ds2_mf_basket_pack` —
for `load_embedding()`, `build_base_cache()` and `grouped_collab()`.

`load_embedding(dir, expected_rows)` reads a `line_latest_emb.csv` and asserts the row count, which
is how a mis-sized embedding table is caught at the boundary rather than deep inside featurisation.
`build_base_cache()` builds the decayed similar-user weights; `grouped_collab()` produces the
collaborative-filtering score for each (query, candidate) pair.

The module also carries a legacy `run_predict()` / `main()` path that writes
`result_ensemble.csv` — the superseded hand-tuned linear blend. **It is not reached by any of the
33 canonical stages**; it remains packaged because the four functions above are on the canonical
path and live in the same file.

### 4.4 Jittor model-training modules

These two files are the **only** ones that import Jittor. Everything downstream is NumPy, SciPy and
LightGBM.

In both trainers the division of labour is deliberate: **seeded NumPy owns all sampling, shuffling
and negative rejection; Jittor owns only the parameter tables, the loss and the optimiser.** This
keeps the framework-dependent surface small and makes the sampling stream reproducible independently
of the framework.

#### `src/train_line_jt.py` — LINE embeddings

*Stage owner*: `ds1_line_full`, `ds1_line_cut`, `ds2_line_full`, `ds2_line_cut`.
*Reads*: `<data-root>/<pack>/<dataset>/train.csv`. *Writes*: `line_latest_emb.csv` in the run
directory given by `pipeline_common.line_run_dir(...)`.

Principal objects: `class LINE` (three embedding tables), `gen_neg_epoch()`, `build_pos_keys()`,
`export_emb()`, `main()`.

1. Load `train.csv`, drop duplicate `(src, dst, time)` rows, cast types.
2. **Size the entity table from the unfiltered frame, before any time cutoff.** This is load-bearing:
   sizing it after the cutoff would shrink the table to nodes seen before the cut and silently
   misalign every downstream node id.
3. Apply `LINE_TIME_MAX` when set — this is what distinguishes a cutoff run from a serve run.
4. Build bidirectional edges, interleaved `[u,v],[v,u]`, and a sorted key set of observed pairs.
5. `class LINE` holds three tables of `sub_dim = EMB_DIM/2 = 200`: `emb_first` (first-order),
   `emb_node` and `emb_ctx` (second-order). All three are Xavier-uniform initialised from **one**
   NumPy stream — reseeding per table made them identical at initialisation.
6. Per epoch: NumPy permutes the positives and `gen_neg_epoch()` draws `NEG_RATIO = 5` negatives per
   positive by **rejection sampling** against the observed-pair key set, retrying up to 30 times.
7. Per batch of 1,024: score first-order `emb_first(s)·emb_first(d)` and second-order
   `emb_node(s)·emb_ctx(d)`; the joint loss is
   `BCE(first) + LOSS_ALPHA * BCE(second)` with `LOSS_ALPHA = 0.5`; Adam at `INIT_LR = 1e-4`;
   gradient-norm clipping at 1.0.
8. Every `EXPORT_EVERY = 10` epochs and once at the end, `export_emb()` writes `node_id` plus the
   concatenation of `emb_first` and `emb_node`, rounded to 6 decimals, via a temporary file and
   `os.replace`.

400 epochs, seed 42, uniform negatives. Virtual-edge self-training is omitted — a measured no-op,
which is what the `-novirt` marker in the run-directory name records. Dataset 1 and Dataset 2 differ
only in `DATASET` and the cutoff timestamp; the architecture and all other parameters are identical.

`validate_embedding_csv` then checks the header, the field count and the row count against the
measured entity count before any completion record is written.

#### `src/train_bpr_jt.py` — BPR-MF embeddings

*Stage owner*: the ten BPR stages per dataset, plus two innovation-only stages on Dataset 1.
*Reads*: the same `train.csv`. *Writes*: `bpr_emb.npy` in `pipeline_common.bpr_run_dir(...)`.

A **single** embedding table of `BPR_DIM = 256` — source and destination share it — trained with the
Bayesian personalised-ranking objective `softplus(-(e_src·e_dst - e_src·e_neg))`, i.e.
`-log sigmoid(...)`. Negatives are drawn from the destination-degree^0.75 distribution by inverse-CDF
sampling in NumPy. Adam, learning rate 3e-3, weight decay 1e-6, batch 8,192, 120 epochs.

Four switches distinguish the runs, and all are applied in NumPy before training:

| Switch | Effect | Used by |
|---|---|---|
| `SEED` | reseeds the NumPy stream and Jittor | all ten runs: 42, 123, 777, 2024, 31337 |
| `BPR_TIME_MAX` | keeps only interactions at or before the cut | the five cutoff runs |
| `BPR_TAU_FRAC` | recency-weighted positive sampling with `tau = frac * time span` | **Dataset 1 only**, at 0.25; Dataset 2 uses 0 |
| `BPR_INNOV` | keeps only the first occurrence of each `(src, dst)` pair | **Dataset 1 only**, two extra runs |

The entity table is again sized before the cutoff. The output is a single `float32` array of shape
`(num_entity, 256)`, written to a temporary path and `os.replace`d; `validate_embedding_npy` checks
shape, dtype and finiteness.

`conv_opt=1` and `use_mkl=0` are **runtime compatibility settings for this image, not scientific
parameters** — they select which Jittor kernels load (see §5.2). They change no model, no
hyperparameter and no result. **Neither trainer imports JittorGeometric.**

### 4.5 Dataset-1 ranking and inference chain

Raw files to `dataset1.csv`, in order.

**1–4. Embeddings.** The four stage families of §4.4 produce, for Dataset 1: a serve LINE
embedding, a cutoff LINE embedding, five serve and five cutoff BPR embeddings, and two
innovation-only BPR embeddings (serve and cutoff).

**5. Cut-split replay** — how the ranker gets labels without test labels. `src/ranker_ds1.py`
splits the training data at `CUT = 115,480,000`: `split0` (before the cut) supplies all history and
all features; `split1` (after the cut) supplies the positives. For each `split1` interaction whose
source appears in the test file, the true destination becomes the positive and up to `NEG = 99`
negatives are sampled — **from that source's own test candidate pool** — so a training query has the
same shape as a test query. The sampling RNG is seeded at 20260727. **No test label is read; only
candidate identities from `test.csv`.**

**6–7. Feature construction.** `featurize()` produces a 21-column matrix, identically for training
and inference; only the frozen history and the embeddings differ. In source order:

| # | Feature | Group | Meaning |
|---|---|---|---|
| 0 | `f_collab` | collaborative | similar-user collaborative score, row-normalised |
| 1 | `f_rpop` | popularity | recent destination popularity |
| 2 | `f_icfm` | LINE-derived | mean cosine of the candidate to the source's history, in LINE space |
| 3 | `f_icf3` | LINE-derived | the same, over the top-3 most similar history items |
| 4 | `f_bpr` | BPR-derived | mean of the five seeds' BPR dot products |
| 5 | `f_cooc` | collaborative | popularity-normalised co-occurrence |
| 6 | `gr` | recency | how late in the history the candidate last appeared |
| 7 | `dst_pop_log` | popularity | log destination degree |
| 8 | history length | history | number of prior interactions of the source |
| 9 | `hm` | history | is the candidate a historical partner (0/1) |
| 10 | `hcnt` | history | how many times, in history |
| 11 | `seen` | structural | has the candidate ever been a destination |
| 12 | `time_gap` | time | `(query time - freeze time)/year` — **clipped at inference** |
| 13 | slate size | structural | candidates in this query |
| 14 | `rs` | popularity | popularity in the last 5% of the time window |
| 15 | `rl` | popularity | popularity in the last 40% |
| 16 | `rs - rl` | popularity | short-minus-long popularity trend |
| 17 | `f_cfreq` | generator footprint | how often the candidate appears across all test slates |
| 18 | `f_popratio` | generator footprint | destination degree divided by that slate frequency |
| 19 | `last_gap` | time | time since the source's last interaction — **clipped at inference** |
| 20 | `ib` | BPR-derived | innovation-only BPR score, zeroed on historical partners |

**8. Ranker training.** `lgb.LGBMRanker` with `objective="lambdarank"`, `metric="ndcg"`,
`ndcg_eval_at=[10]`, `n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`,
`min_child_samples=100`, `label_gain=[0,1]`, `random_state=42`, fitted with per-query group sizes.

**9. Inference.** The same `featurize()` runs with history frozen at the end of the training data,
the *serve* embeddings, and the real test candidates. The two forward-extrapolating time features
(columns 12 and 19) are **clipped to the range observed during training**, so the model never sees
an out-of-support value. `mdl.predict(Xte)` yields raw LambdaRank margins.

**10. Serialisation domain.** LambdaRank margins are unbounded and mostly negative;
`serialisation_normalise()` maps each row by **per-row min-max** into `[0, 1]`, mapping a degenerate
constant row to 0.5. This is strictly order-preserving — it cannot change MRR — but it is
load-bearing, because both postprocessors and the output gate assume the unit interval. The result
is written atomically to `outputs/dataset1-ensemble/result_ranker.csv` at `%.6f`.

**11–13. Deterministic post-processing.** `src/build_ds1_member.py` reads that matrix and applies
`registry.DS1_POSTPROCESSOR_CHAIN` in order. Neither stage trains anything; both are frozen rules
with no fitted parameters, and both read only candidate identities and the training history —
never a label.

* `ds1_source_slate_recurrence` — the inference batch is itself evidence. With `recurrence(c)` = the
  number of *other* test slates of the same source containing `c`: act only if the current top-1 is
  **not** a historical partner; consider only non-historical candidates; take the candidate with the
  **strictly unique** maximum recurrence (ties disqualify the row); require it to appear in at least
  2 other slates; promote it to strict top-1. Acts on 3,653 of 61,051 rows.
* `ds1_test_graph_reciprocity` — with `c1`, `c2` the rank-1 and rank-2 candidates of the
  *recurrence-adjusted* matrix, act iff neither is a historical partner, the reverse test-exposure
  edge exists for `c2`, and does not for `c1`; then promote `c2`. Acts on 1,365 rows.

**The order is load-bearing.** Reciprocity reads the rank-2 candidate of the matrix the recurrence
stage produced, so swapping the two yields a different file.

**14–16.** `validate_submission_matrix` gates the float matrix, `write_score_matrix` serialises it
at `%.6f` with pinned CRLF through a `.part` file, `verify_submission_file` checks the serialised
row and field counts, and `canonical_pipeline` publishes the artifact and its completion record.

### 4.6 Dataset-2 ranking and inference chain

**1. Raw loading and role measurement.** `main.py --stage preprocess` measures the source/destination
identity overlap; an empty overlap is the bipartite signature, and Dataset 2 is bipartite. Nothing
in the chain assumes it — the measurement is reported, not asserted.

**2–3. Embeddings.** Two LINE runs and ten BPR runs, as in §4.4. Dataset 2 uses no recency weighting
and has no innovation-only variant.

**4–6. Features and cache.** `src/ds2_basket_featurizer.py` performs the same cut-split replay at
`CUT = 1,261,958,400` and builds an **18-column** base matrix in `build_features()`:

| # | Feature | Group |
|---|---|---|
| 0–1 | `f_collab`, `f_rpop` | collaborative, popularity |
| 2–3 | `f_icfm`, `f_icf3` | LINE-derived similarity to history |
| 4 | `f_bpr` | BPR-derived, five-seed mean |
| 5 | `blend` | the legacy weighted blend of the above, retained as one column |
| 6–7 | `gr`, `dst_pop_log` | recency, popularity |
| 8–10 | history length, `hm`, `seen` | history and structural |
| 11–12 | `time_gap`, slate size | time and structural |
| 13–15 | `rs`, `rl`, `rs - rl` | short/long popularity and trend |
| 16–17 | `f_cfreq`, `f_popratio` | generator footprint |

`build_or_load_features()` caches this matrix as `train_features_all.npz`. The cache is governed by
its own completion record: written through `atomic_output` and bound to the SHA256 of both official
input files, the cut, the negatives per sample, both sampling seeds, the entity count, the expected
array names and the relevant-code digest, then validated array-by-array on every reuse.

**7–10. Three-pass basket feedback.** `src/ranker_basket_ds2.py` trains three LambdaRank models with
the same parameters as Dataset 1 plus `feature_fraction = 0.8`:

* **pass 1** fits `m1` on the 18 features; a 2-fold source-disjoint cross-fit produces unbiased
  in-sample scores `s1_tr`;
* `fb_features()` converts those scores into **two** sibling-feedback columns — the maximum and mean
  score assigned to each candidate across the *other* rows of the same basket event;
* **pass 2** fits `m2` on 18 + 2 = 20 columns; cross-fit again to get `s2_tr`;
* **pass 3** fits `m3` on 20 columns built from `s2_tr`.

At serve time the three saved boosters (`ranker_pass{1,2,3}.txt`) are applied in the same order to
the real test rows, each pass regenerating the sibling features from the previous pass's scores.
The final row scores are min-max normalised with historical partners masked out, and written to
`outputs/dataset2-ranker/ranker_basket3_dataset2.csv`.

**11–13. MF sibling geometry.** `src/ds2_mf_basket_pack.py` is the production variant of that
chain. `mf_geom()` computes a **rank-128 truncated SVD** (`scipy.sparse.linalg.svds`) of the
split-0 source x destination interaction matrix and L2-normalises the rows; `fb_dense_ent()` then
carries the sibling messages through that dense geometry instead of the sparse item profiles. The
three-pass structure is otherwise identical, and the module reuses the featurizer's cache and
`build_features()` verbatim. Output: `mf_basket3_dataset2.csv`, the **production base matrix**.

> **"MF" means two different things in this codebase.** The *BPR-MF embedding* (§4.4) is a trained
> model. The *MF sibling geometry* here is an untrained SVD factorisation used only to carry
> messages between rows of a basket event. They are unrelated.

**14. Dataset-1 passthrough.** `crf_promote.py` writes a two-member container and therefore needs
Dataset-1 member bytes. `canonical_pipeline._write_passthrough()` packages them from a Dataset-1
member that carries a valid completion record. Those bytes are never read as scores — the Dataset-2
member builder extracts `dataset2.csv` alone.

**15–19. Equality CRF and structural rules.** `src/crf_promote.py` runs with
`--tau 0.20 --B 70 --W 1 --zr-exclude --demote-dups --st-exclude --no-pair`:

* `build_band()` couples adjacent test rows that the raw row order shows to be related;
  `equality_crf()` propagates agreement across that chain with strength `tau` and `B`;
* `find_triples()` applies the **triple rule** — a structurally forced answer is promoted to strict
  top-1;
* the **pair rule is disabled** (`--no-pair`): it double-counts the chain the CRF already models;
* `zero_repeat_demotions()` applies **cross-time exclusivity** — because a `(source, answer)` pair
  never recurs at two distinct timestamps, a labelled cross-time answer is a negative elsewhere —
  and **duplicate-slate demotion**, since negatives are drawn with replacement so a repeated slate
  id cannot be the answer;
* `same_time_structural_demotions()` adds the same-time structural rule: within one timestamp a
  shared answer forms one contiguous run in raw order, so a warm candidate in a shorter disjoint
  same-time component is a negative;
* `apply_demotions()` pushes demoted candidates strictly below every retained score.

The module re-reads its own output and re-asserts every invariant before exiting. Output:
`ds2_crf_intermediate.zip`.

**20–24. Final decode and serialisation.** `src/build_ds2_member.py` extracts `dataset2.csv` from
that container and applies `strategies/ds2/cross_time_exclusivity.apply()`: group rows by
`(source, served top-1)`; a group spanning two or more distinct timestamps contradicts the
invariant; keep the timestamp cluster with the largest maximum row margin, ties to the smallest
timestamp; swap the rank-1 and rank-2 scores in every row of every other cluster. One pass, no
thresholds, no refit; 7,815 rows.

Serialisation is **byte-preserving**: `swap_score_tokens()` rewrites only the two affected score
tokens inside the base member's own bytes, so the other 15,326,370 cells are exactly the bytes the
base produced and no value is re-serialised from a float. The result is then re-read and its
per-row candidate ranking compared against the float path; a disagreement fails the stage. A
structural census is asserted against the physical test file on every run.

### 4.7 Final-member validation and serialisation

#### `src/strategies/shared/frozen_ops.py`

Shared by both member builders and by both Dataset-1 postprocessors.

| Function | Role |
|---|---|
| `validate_submission_matrix` | the **output gate**: shape, finiteness, and every value within `[0, 1]`. It reports and aborts; it never clamps, because a member failing here is a signal about the chain and repairing it would destroy the evidence and change the ranking |
| `verify_submission_file` | structural gate on the *serialised* bytes — row count and per-row field count, which catches a truncated or ragged write an in-memory check cannot see |
| `write_score_matrix` | writes `%.6f` with **pinned CRLF** through a `.part` file, verifies the bytes, then renames |
| `row_max_normalise` | divides each row by its maximum; a row whose maximum is not strictly positive is shifted by its own minimum instead, and a constant row maps to 0.5 |
| `promoted_value` | the frozen strict-top-1 promotion value used by both Dataset-1 postprocessors |
| `stable_rank_order` | descending order, ties by ascending column index — the single ranking convention |
| `history_mask`, `pair_keys`, `membership` | directed `src -> dst` history membership |
| `dense_group_ids`, `csv_record_spans` | grouping and byte-offset support for the Dataset-2 decoder |

#### `src/strategies/registry.py`

Holds `DS1_POSTPROCESSOR_CHAIN` and `DS2_POSTPROCESSOR_CHAIN` — the ordered `(strategy id, module,
apply callable)` tuples the member builders execute. Its lifecycle-inventory helpers
(`load_inventory`, `strategies_by_status`) read `docs/strategy_inventory.json`, which is **not
shipped**; they are lazy and never called on the reproduction path, so their absence does not
affect reproduction, but they will raise if invoked inside the package.

### 4.8 Training versus inference — summary matrix

| Module / stage | Training input | Learned state | Inference input | Produces | Framework / device | Uses test labels? | Deterministic once inputs fixed? |
|---|---|---|---|---|---|:--:|---|
| LINE | `train.csv` edges (full or cutoff) | 3 embedding tables, 200-dim each | — (an output, not a scorer) | `line_latest_emb.csv` | **Jittor**, GPU | **No** | No — GPU kernel scheduling is not bit-reproducible |
| BPR-MF | `train.csv` pairs (+ cutoff / recency / innovation filters) | 1 embedding table, 256-dim | — | `bpr_emb.npy` | **Jittor**, GPU | **No** | No — same reason |
| Dataset-1 LambdaRank | cut-split replay: `split0` features, `split1` positives, 99 sampled negatives from the test candidate pool | LightGBM booster, 400 trees | 21 features on the real test slates, serve embeddings, time features clipped | `result_ranker.csv` | LightGBM, CPU | **No** | Yes |
| Dataset-2 LambdaRank | cut-split replay, 18 features | 3 boosters (`pass1/2/3`) | the same 18 features on test rows | pass-3 scores | LightGBM, CPU | **No** | Yes |
| Basket-feedback passes | cross-fit scores of the previous pass | none of its own — 2 derived columns | sibling scores within a basket event | 20-column matrices | NumPy + LightGBM, CPU | **No** | Yes |
| MF sibling geometry | split-0 interaction matrix | **none — an untrained SVD factorisation** | candidate ids | rank-128 dense geometry | SciPy, CPU | **No** | Yes |
| Equality CRF | — **no training** | none | base scores, candidate ids, timestamps, raw row order | CRF container | NumPy, CPU | **No** | Yes |
| Dataset-1 postprocessors | — **no training** | none | score matrix, `test.csv` candidates, training history | adjusted matrix | NumPy, CPU | **No** | Yes |
| Dataset-2 final decoder | — **no training** | none | CRF base, `test.csv` sources and timestamps | member bytes | NumPy, CPU | **No** | Yes |
| Final serialisation | — | none | validated float matrix | `dataset{1,2}.csv` | NumPy/pandas, CPU | **No** | Yes |

Reading the matrix: **test candidate identities are used** — for inference, and to shape the
negative pool of the cut-split replay, which is what makes a training query resemble a test query.
**Test ground-truth labels are never used**, are not available to this code, and appear nowhere in
any stage. Every ranker label derives from training-data interactions after the time cut. Only LINE
and BPR-MF use Jittor; the six deterministic stages train nothing.

### 4.9 Stage-to-module-to-artifact matrix

All 33 canonical stages. Repeated BPR stages are shown as one row per seed so that every instance is
explicit. Every stage requires a completion record before it may be reused.

**Dataset 1 — 16 stages**

| # | Stage id | Module | Type | Principal input | Principal output | Device | Jittor |
|---|---|---|---|---|---|---|:--:|
| 1 | `ds1_line_full` | `train_line_jt.py` | TRAIN | `train.csv` | `dataset1-novirt/line_latest_emb.csv` | GPU | yes |
| 2 | `ds1_line_cut` | `train_line_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset1-novirt-tmax…/line_latest_emb.csv` | GPU | yes |
| 3 | `ds1_bpr_serve_s42` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset1-bpr-t0.25/bpr_emb.npy` | GPU | yes |
| 4 | `ds1_bpr_cut_s42` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset1-bpr-t0.25-tmax…/bpr_emb.npy` | GPU | yes |
| 5 | `ds1_bpr_serve_s123` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset1-bpr-t0.25-s123/bpr_emb.npy` | GPU | yes |
| 6 | `ds1_bpr_cut_s123` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `…-tmax…-s123/bpr_emb.npy` | GPU | yes |
| 7 | `ds1_bpr_serve_s777` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset1-bpr-t0.25-s777/bpr_emb.npy` | GPU | yes |
| 8 | `ds1_bpr_cut_s777` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `…-tmax…-s777/bpr_emb.npy` | GPU | yes |
| 9 | `ds1_bpr_serve_s2024` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset1-bpr-t0.25-s2024/bpr_emb.npy` | GPU | yes |
| 10 | `ds1_bpr_cut_s2024` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `…-tmax…-s2024/bpr_emb.npy` | GPU | yes |
| 11 | `ds1_bpr_serve_s31337` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset1-bpr-t0.25-s31337/bpr_emb.npy` | GPU | yes |
| 12 | `ds1_bpr_cut_s31337` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `…-tmax…-s31337/bpr_emb.npy` | GPU | yes |
| 13 | `ds1_bpr_innov_serve` | `train_bpr_jt.py` | TRAIN | `train.csv`, first occurrences only | `dataset1-bpr-innov-t0.25/bpr_emb.npy` | GPU | yes |
| 14 | `ds1_bpr_innov_cut` | `train_bpr_jt.py` | TRAIN | as above, t <= cut | `dataset1-bpr-innov-t0.25-tmax…/bpr_emb.npy` | GPU | yes |
| 15 | `ds1_ranker` | `ranker_ds1.py` | RANKER_TRAIN_AND_INFER | `train.csv`, `test.csv`, stages 1–14 | `dataset1-ensemble/result_ranker.csv` | CPU | no |
| 16 | `ds1_member` | `build_ds1_member.py` | DETERMINISTIC_POSTPROCESS + VALIDATE_AND_SERIALISE | stage 15, `train.csv`, `test.csv` | `members/dataset1.csv` | CPU | no |

**Dataset 2 — 17 stages**

| # | Stage id | Module | Type | Principal input | Principal output | Device | Jittor |
|---|---|---|---|---|---|---|:--:|
| 1 | `ds2_line_full` | `train_line_jt.py` | TRAIN | `train.csv` | `dataset2-novirt/line_latest_emb.csv` | GPU | yes |
| 2 | `ds2_line_cut` | `train_line_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset2-novirt-tmax…/line_latest_emb.csv` | GPU | yes |
| 3 | `ds2_bpr_serve_s42` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset2-bpr/bpr_emb.npy` | GPU | yes |
| 4 | `ds2_bpr_cut_s42` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset2-bpr-tmax…/bpr_emb.npy` | GPU | yes |
| 5 | `ds2_bpr_serve_s123` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset2-bpr-s123/bpr_emb.npy` | GPU | yes |
| 6 | `ds2_bpr_cut_s123` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `…-tmax…-s123/bpr_emb.npy` | GPU | yes |
| 7 | `ds2_bpr_serve_s777` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset2-bpr-s777/bpr_emb.npy` | GPU | yes |
| 8 | `ds2_bpr_cut_s777` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `…-tmax…-s777/bpr_emb.npy` | GPU | yes |
| 9 | `ds2_bpr_serve_s2024` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset2-bpr-s2024/bpr_emb.npy` | GPU | yes |
| 10 | `ds2_bpr_cut_s2024` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `…-tmax…-s2024/bpr_emb.npy` | GPU | yes |
| 11 | `ds2_bpr_serve_s31337` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset2-bpr-s31337/bpr_emb.npy` | GPU | yes |
| 12 | `ds2_bpr_cut_s31337` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `…-tmax…-s31337/bpr_emb.npy` | GPU | yes |
| 13 | `ds2_ranker` | `ranker_basket_ds2.py` | RANKER_TRAIN_AND_INFER | `train.csv`, `test.csv`, stages 1–12 | `dataset2-ranker/ranker_basket3_dataset2.csv` | CPU | no |
| 14 | `ds2_mf_pack` | `ds2_mf_basket_pack.py` (+ `ds2_basket_featurizer.py`) | FEATURE_BUILD + RANKER_TRAIN_AND_INFER | `train.csv`, `test.csv`, stages 1–12, feature cache | `dataset2-ranker/mf_basket3_dataset2.csv` | CPU | no |
| 15 | `ds2_ds1_passthrough` | `canonical_pipeline.py` (in-process) | PASSTHROUGH | validated `members/dataset1.csv` | `dataset2-crf/ds1_passthrough.zip` | CPU | no |
| 16 | `ds2_crf` | `crf_promote.py` | DETERMINISTIC_POSTPROCESS | stage 14, stage 15, `train.csv`, `test.csv` | `dataset2-crf/ds2_crf_intermediate.zip` | CPU | no |
| 17 | `ds2_member` | `build_ds2_member.py` | DETERMINISTIC_POSTPROCESS + VALIDATE_AND_SERIALISE | stage 16, `test.csv` | `members/dataset2.csv` | CPU | no |

The Dataset-2 feature cache (`src/bagging_cache/train_features_all.npz`) is built inside stage 14
and carries its own completion record; it is not a separate driver stage.

### 4.10 Complete packaged-code inventory

All 26 files under `code/`. Reachability: **DIRECT** = named in a stage command or the entrypoint
chain; **TRANSITIVE** = imported by a DIRECT module; **SUPPORTING_TOOL** = a working utility not on
the reproduction path; **INFORMATIONAL_ONLY** = present for completeness, never executed.

| Path | Category | Purpose | Reachability | Principal caller | Runs in a full reproduction? |
|---|---|---|---|---|---|
| `run_all.py` | entrypoint | one-command reproduction, both datasets | DIRECT | the reviewer | yes |
| `main.py` | entrypoint | per-dataset entry, stage modes | DIRECT | `run_all.py` | yes |
| `src/canonical_pipeline.py` | orchestration | stage graphs, fail-closed runner | DIRECT | `main.py` | yes |
| `src/stage_contract.py` | orchestration | completion records, atomic publication, validators | TRANSITIVE | `canonical_pipeline`, `ds2_basket_featurizer` | yes |
| `src/pipeline_common.py` | shared | path contract, history, co-occurrence, similarity | TRANSITIVE | trainers, rankers, featurizer | yes |
| `src/train_line_jt.py` | training | LINE embeddings (Jittor) | DIRECT | `canonical_pipeline` | yes, 4 stages |
| `src/train_bpr_jt.py` | training | BPR-MF embeddings (Jittor) | DIRECT | `canonical_pipeline` | yes, 22 stages |
| `src/ensemble_predict.py` | inference | embedding loading, collaborative scoring | TRANSITIVE | both rankers, featurizer, MF pack | yes (helpers only; its legacy `main()` is never called) |
| `src/ranker_ds1.py` | ranking | Dataset-1 LambdaRank, 21 features | DIRECT | `canonical_pipeline` | yes |
| `src/ranker_basket_ds2.py` | ranking | Dataset-2 LambdaRank, 3-pass basket feedback | DIRECT | `canonical_pipeline` | yes |
| `src/ds2_basket_featurizer.py` | features | Dataset-2 features and the contract-governed cache | TRANSITIVE | `ds2_mf_basket_pack` | yes |
| `src/ds2_mf_basket_pack.py` | ranking | MF sibling geometry, production base matrix | DIRECT | `canonical_pipeline` | yes |
| `src/crf_promote.py` | postprocessing | equality CRF and structural demotions | DIRECT | `canonical_pipeline` | yes |
| `src/build_ds1_member.py` | final output | Dataset-1 postprocessor chain and member | DIRECT | `canonical_pipeline` | yes |
| `src/build_ds2_member.py` | final output | Dataset-2 decode and member | DIRECT | `canonical_pipeline` | yes |
| `src/strategies/registry.py` | postprocessing | ordered postprocessor chains | TRANSITIVE | both member builders | yes (chain constants; its inventory helpers are not reachable — see §4.7) |
| `src/strategies/shared/frozen_ops.py` | postprocessing | output gate, frozen primitives, serialisation | TRANSITIVE | member builders, postprocessors | yes |
| `src/strategies/ds1/source_slate_recurrence.py` | postprocessing | Dataset-1 postprocessor 1 | TRANSITIVE | `build_ds1_member` | yes |
| `src/strategies/ds1/test_graph_reciprocity.py` | postprocessing | Dataset-1 postprocessor 2 | TRANSITIVE | `build_ds1_member` | yes |
| `src/strategies/ds2/cross_time_exclusivity.py` | postprocessing | Dataset-2 final decoder | TRANSITIVE | `build_ds2_member` | yes |
| `src/strategies/__init__.py` | package marker | package initialiser | TRANSITIVE | import machinery | yes |
| `src/strategies/ds1/__init__.py` | package marker | package initialiser | TRANSITIVE | import machinery | yes |
| `src/strategies/ds2/__init__.py` | package marker | package initialiser | TRANSITIVE | import machinery | yes |
| `src/strategies/shared/__init__.py` | package marker | package initialiser | TRANSITIVE | import machinery | yes |
| `configs/production.json` | configuration | machine-readable chain description and accepted-artifact anchors | DIRECT | `main.py --stage describe`, `--verify` | no — informational; reproduction does not read it |
| `tools/submission/package_component.py` | verification | independent submission-format verifier | SUPPORTING_TOOL | `main.py --stage package --verify` | no — offered so a reviewer can check a member independently |

**Two files do not execute during a normal reproduction**, and both are included deliberately:
`configs/production.json` records the accepted artifact hashes and the chain description so a
reviewer can compare against them, and `tools/submission/package_component.py` lets a reviewer
validate a produced member without trusting our tooling. Neither is required by
`python run_all.py`. Note that `--stage describe` and `--stage package --verify` reference the
accepted A-board archive, which is not redistributable and is not included; those two modes are
informational in the package.

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

### 7.1 One command

```bash
python run_all.py --data-root /path/to/data --output-root /path/to/outputs
```

That is the whole reproduction. It runs both datasets, in the required order, and stops at the
first failure.

`run_all.py` is an **orchestration wrapper and nothing else**. It invokes the two canonical dataset
pipelines as child processes and contains no model, feature, ranking, normalisation, CRF or
decoding logic; it imports no training module and reads no configuration of its own. It uses only
the Python standard library. Every scientific decision stays in `main.py` and the canonical stage
graph, so the wrapper cannot change a result — it can only change whether and in what order the two
pipelines run.

Its behaviour:

| Property | Behaviour |
|---|---|
| Order | Dataset 1, then Dataset 2 — required, see below |
| Dataset-1 failure | Dataset 2 is **not started**; the child's exit code is propagated unchanged |
| Reuse | validated resume by default, inherited from `main.py` — a stage is reused only when its completion record proves it |
| `--fresh` | forwarded to both datasets; refuses all reuse |
| `--plan` | prints both stage graphs and performs **no computation** |
| Preflight | verifies that both datasets' `train.csv` and `test.csv` exist before starting anything |
| Artifacts | nothing is deleted, repaired or adopted automatically |

### 7.2 The transparent equivalent

`run_all.py` is exactly these two commands, and running them directly is equivalent in every
respect:

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

### 7.3 Commands

| Purpose | Command |
|---|---|
| **Full reproduction, both datasets** | `python run_all.py --data-root /path/to/data --output-root /path/to/outputs` |
| Show both stage graphs; execute nothing | `python run_all.py --plan` |
| Full reproduction, refusing all reuse | `python run_all.py --fresh` |
| Show one graph and each stage's state; execute nothing | `python main.py --dataset dataset1 --stage plan` |
| Run one dataset | `python main.py --dataset dataset1` |
| Validated resume (the default) | `python main.py --dataset dataset1` — completed stages with a valid record are reused |
| Refuse all reuse | `python main.py --dataset dataset1 --fresh` |
| Raw-data root | `--data-root /path/to/data` |
| Artifact root | `--output-root /path/to/outputs` |
| Stage logs | `--log-dir /path/to/logs` (default `<output-root>/_logs`) |
| Inspect the raw data | `python main.py --dataset dataset1 --stage preprocess` |
| Describe the production chain | `python main.py --stage describe` |

### 7.4 Outputs

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

### 7.5 Failure behaviour

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

Measured on the validation host (RTX 4090 24 GB, Xeon Gold 6430, 16 cores, 120 GiB). Only retained
measurements are reported; no runtime below is estimated.

### 10.1 Dataset 1 — complete

Every stage was timed by the run driver. The sixteen stage durations sum to 6,698 s; the driver's
wall clock was 6,702 s, the 4 s difference being inter-stage orchestration.

| Stage | Device | Jittor | Measured |
|---|---|:--:|---:|
| LINE, full history | GPU | yes | 2,309 s |
| LINE, cut history | GPU | yes | 1,802 s |
| BPR, 5 serve runs | GPU | yes | 72 / 58 / 58 / 58 / 57 s |
| BPR, 5 cut runs | GPU | yes | 42 / 41 / 42 / 42 / 42 s |
| BPR, 2 innovation-only runs | GPU | yes | 16 / 12 s |
| Ranker (LightGBM) | CPU | no | 2,028 s |
| Member build | CPU | no | 19 s |
| **Raw data to member** | | | **1 h 51 m 42 s** |

Device split: GPU 4,651 s (69.4%), CPU 2,047 s (30.6%), orchestration 4 s. The first BPR run is
longer than the rest because it absorbs Jittor's first-call kernel compilation.

### 10.2 Dataset 2 — partial

Dataset 2 was produced across separate sessions rather than as one continuous run, and the driver
log covering its training stages was not retained. Five stage measurements survive:

| Stage | Device | Jittor | Measured | Status |
|---|---|:--:|---:|---|
| LINE, full history | GPU | yes | 15,467 s | exact |
| LINE, cut history | GPU | yes | — | **not recorded** |
| BPR, 5 serve + 5 cut runs | GPU | yes | — | **not recorded** |
| Ranker (LightGBM) | CPU | no | — | **not recorded** |
| Train-feature cache | CPU | no | 10,115 s | aggregate only |
| MF basket pack (LightGBM) | CPU | no | 8,837 s | exact |
| CRF | CPU | no | 120 s | exact |
| Member decode | CPU | no | 9 s | exact |

**At least 9 h 35 m 48 s (34,548 s) is directly accounted for by these five retained stage
measurements. No complete end-to-end wall-clock measurement was retained, and twelve of the
seventeen stages remain untimed; the total runtime is therefore not reported.**

Among the retained measurements only, Jittor stages account for 15,467 s and non-Jittor CPU stages
for 19,081 s. This is a **partial-measurement comparison and not a statement about the complete
runtime**: eleven of the twelve untimed stages use Jittor and one is CPU, so both sides would grow
by unknown amounts.

**Only LINE and BPR use Jittor.** The train-feature cache, the MF basket pack, the CRF and both
member builders are CPU stages built on NumPy, SciPy and LightGBM.

### 10.3 Resources

Peak disk: approximately 5 GB of artifacts per dataset, plus a 1.92 GiB Dataset-2 train-feature
cache. Peak GPU memory is modest; the embedding tables dominate and fit comfortably in 24 GB.

**Resumability.** Every stage boundary is a resume point, governed by the completion contract. There
are no resume points *inside* a stage: an interrupted trainer or an interrupted feature build restarts
that stage from the beginning. The longest single unresumable stage is the Dataset-2 feature build at
roughly 2.8 h.

---

## 11. Reproducibility and Nondeterminism

* Both raw-to-final lineages have been executed successfully from the official raw data on the target
  environment (Dataset 1 and Dataset 2, each as a complete Jittor run on an RTX 4090). Those runs
  are the scientific evidence for this submission.
* The canonical entry point and the stage-completion contract were separately validated on the same
  target environment against real Jittor training stages.
* The released package additionally received targeted downstream and clean-package integration
  validation covering **all 33 canonical stages** (16 for Dataset 1, 17 for Dataset 2): the archive
  was extracted into an empty directory and, from that extracted copy alone, every stage command was
  resolved and contract-checked, and the real LightGBM ranker, the real matrix-factorisation
  geometry, the real feature-cache contract, the real CRF and both real member builders were
  executed on bounded production-schema fixtures, with the final output gates asserted on both
  produced members.
* A defect in which a non-default `--output-root` was ignored by two Dataset-2 consumers was found
  by that validation, corrected, and revalidated from a fresh extraction (see section 12, item 9).
* **A second complete dual-dataset re-execution was not performed**, by decision, because it would
  have repeated training already evidenced without testing anything the validations above do not
  already cover.
* **GPU training is not guaranteed byte-deterministic across independent runs.** Jittor kernel
  scheduling on the GPU is not bit-reproducible, and the model state of the original A-board runs was
  not preserved.
* **The historical accepted archive is the artifact associated with the recorded A-board score.**
  A reproduction produced by this code is a distinct artifact and carries no score of its own until
  it is itself scored.
* Consequently, **a fresh reproduction may not be byte-identical to the historical A-board member**.
  A measured comparison of one clean Dataset-1 reconstruction against the historical member gave
  **95.603676%** top-1 agreement and mean per-row Spearman **0.872865**. Quantifying how far the
  difference reaches: the historical rank-1 candidate appears within the reproduction's top two in
  **98.348921%** of rows, and the reproduction's rank-1 candidate within the historical top two in
  **98.352197%** — so of the 2,684 rows that disagree at rank 1, about a fifth differ beyond the
  first two positions. This is the expected behaviour of a retrained pipeline, not a defect.
* **No claim of hidden-score equivalence is made.** Test labels are unavailable, so no offline
  statement about the score of a reconstruction is possible, and none is made here. In particular,
  the ranking-overlap figures above do not imply an MRR relationship: under reciprocal-rank scoring
  a rank-1 and a rank-2 placement contribute 1 and 0.5 respectively, so they are materially
  different outcomes.
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
8. **Completion records produced from the extracted package report
   `producing_commit: "UNKNOWN"`.** The package contains no `.git` directory, by design, so the
   commit cannot be read. This is deliberate and fail-safe: `UNKNOWN` never compares equal to a real
   commit, so a record written inside the package can never be mistaken for one written in a
   checkout, and vice versa. Every other identity a record binds — input hashes, configuration
   digest, relevant-code digest, output hash, epoch counts — is unaffected and still enforced.
9. **`--output-root` routing is correct throughout, including the Dataset-2 stages.** Every
   producer and every consumer derives its run directory from the shared path contract
   (`pipeline_common.line_run_dir`, `bpr_run_dir`, `outputs_root`), so an artifact written under a
   requested output root is read from that root and from nowhere else. Two Dataset-2 consumers
   previously spelled the directory out as a literal and therefore ignored the option; that was
   found by targeted validation on 2026-08-02 and corrected. Under the default output root the
   corrected expressions resolve to byte-identical paths, so no existing artifact is orphaned, and
   `tests/strategies/test_ds2_output_root_contract.py` pins the identity, the isolated-root routing
   and the stale-default-tree case behaviourally.

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

# 3. Full reproduction, one command.
python run_all.py --data-root /path/to/data --output-root /path/to/outputs

# 4. Inspect the provenance of any produced artifact.
cat <output-root>/members/dataset1.csv.done.json
```

Every completion record states the commit, the command, the environment, the identity of every input
and the identity of the output. A reviewer can therefore establish exactly what produced each
artifact without trusting this document.
