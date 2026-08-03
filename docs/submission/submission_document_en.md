# Submission Document — Temporal Link Prediction (Track 1)

## 1. Team Information

| Field | Content |
|---|---|
| Team name | 皮卡丘 |
| A-board rank | 3rd |
| Best A-board total score | 1.576996059163449 |
| Project contact | [public contact redacted] |
| WeChat | [public contact redacted] |
| Telephone | [public contact redacted] |

## 2. Project Overview

### 2.1 Core task

This project performs temporal link prediction on two dynamic interaction
graphs. The input is a training set of timestamped historical interaction
triples `(src, dst, time)`; for each `(source node, time)` pair in the test set,
the 100 corresponding candidate nodes are scored for MRR. The evaluation total
is the direct sum of the MRR obtained on Dataset 1 and on Dataset 2.

### 2.2 Differences between the datasets

The two datasets differ markedly in structure. The difference is not a prior
assumption; it is measured directly from the raw files.

1. **Dataset 1**: not bipartite — source and destination roles interchange, and
   over the full training history 21,253 nodes appear on both sides. Historical
   interaction features are therefore extremely valuable, and the problem falls
   within conventional temporal graph prediction. Two distinct structural
   statistics support this, and they are **not** interchangeable:

   | Scope | Unit of analysis | Statistic | Value |
   |---|---|---|---|
   | Dataset-1 `split0` (pre-cut training subset) | distinct directed `(source, destination)` pairs | the reverse-direction pair also occurs | **63.31%** (89,814 of 141,867) |
   | Dataset-1 full deduplicated history | chronological interaction rows | the same directed pair occurred earlier for that source | **72.56%** (500,700 of 690,007) |

   In words: in the pre-cut training subset, 63.31% of distinct directed
   interaction pairs have a reciprocal reverse-direction pair — of the 21,534
   nodes present in that subset, 18,019 occupy both roles. Separately, after
   exact duplicate interaction triples are removed by the production data loader,
   500,700 of 690,007 Dataset-1 interactions, or 72.56%, repeat a directed
   `(source, destination)` pair previously observed for that source. The first is
   a reciprocity measure over pairs; the second is a recurrence measure over
   rows. Only 3.49% of test candidate slots are cold.
2. **Dataset 2**: strictly bipartite — the source and destination node sets are
   disjoint, and no historical pair is ever repeated (0 of 2,211,275 distinct
   pairs recurs at a second timestamp). 53.59% of candidate slots are cold. The
   problem falls within collaborative-filtering recommendation.

### 2.3 Overall approach

Temporal prediction and sequential recommendation are classical core problems in
information retrieval and personalised matching. Since the beginning of this
century the relevant algorithms have evolved from traditional temporal models —
Markov chains, collaborative filtering, RNN/LSTM — through to the graph-learning
paradigms that dominate today, such as graph neural networks and graph
contrastive learning. Leading industrial deployments generally adopt hybrid
architectures combining multi-feature fusion with multi-model ensembling, and
research groups in the field have successively proposed representative
algorithms such as SASRec, BERT4Rec, LightGCN and CRAFT, many of which have
performed strongly in top-tier recommendation and link-prediction competitions
including KDD Cup, RecSys, Alibaba Tianchi and Baidu PaddlePaddle.

During the preliminary research phase of this competition, our team
systematically reviewed the winning and runner-up solutions of recent major
recommendation and graph-learning competitions, and fully reproduced the
baseline performance of each class of benchmark model in its original
competition setting. Once applied to this link-prediction task, however, we
found that those mature solutions adapt poorly to the graph structure, temporal
distribution and sparsity characteristics of the present dataset: the best
traditional competition model we reproduced only matched the CRAFT baseline
supplied by the organisers, holding MRR stably around 0.77 on Dataset 1 and
around 0.48 on Dataset 2.

In subsequent iterations we found that moderately deepening the Transformer
encoder layers and enlarging the number of sampled semantic and structural
neighbours raised the fit of the model quickly, lifting the metrics on both
datasets in step to around 1.40 in total. That optimisation path, however, has a
fatal engineering weakness: the computational cost of the model explodes
non-linearly with the number of layers and the number of neighbours, so training
and inference efficiency become severely inadequate. Such a model cannot be
adapted to large-scale real business datasets with millions of nodes and
hundreds of millions of interaction edges, and therefore has no practical value
for online deployment or large-scale rollout.

Our team surveyed the relevant Chinese and international literature broadly,
completing a systematic review of existing models and recent papers in temporal
link prediction and sequential recommendation, and quantified the intrinsic
meaning, the time and space complexity, and the actual contribution share of
each computational module of the neural networks concerned. The complete
in-depth technical analysis report will be submitted to the competition
committee at the defence stage; what follows outlines only the basic design
rationale of the present solution.

All temporal prediction algorithms share a common basic execution paradigm.
First, heterogeneous multi-dimensional features — source-destination (`src-dst`)
interaction behaviour, timestamps, edge attributes, node attributes — are
uniformly encoded, compressing each class of information into a fixed-dimension
low-dimensional embedding vector through a differentiated mapping. Sequence
representations are then reconstructed by a temporal architecture such as an RNN
or a Transformer, aggregating node-neighbour and global-context information.
Finally, a fully connected network or a traditional machine-learning classifier
produces the prediction of future interaction behaviour and latent links.

Examined at the level of the underlying architectural logic, the structural
design of each class of model is essentially the concrete embodiment of a
research team's understanding of the dataset distribution and of the evolution
of the network. The core reason these algorithms have any predictive power at
all is that temporal interaction behaviour objectively contains stable,
exploitable regularities — typically recency effects (recent behaviour carries
greater influence), long- and short-period preferences, repeat consumption and
repeat interaction, the evolution of local neighbour associations, and the
gradual decay of node interest. Each model architecture is, in essence, a
mathematical and structural fit to those objective behavioural regularities.

It follows that there is essentially no universal model capable of
accommodating every temporal prediction scenario: each class of behavioural
regularity must be mined specifically and designed for in a bespoke way, in
combination with the real business conditions and the dataset itself. The
generality of a model is positively correlated with its computational cost — the
broader the adaptation sought, the higher the computational cost. Since the
advent of large models, moreover, the traditional development path of trading
stacked structure for generality has reached a stage of convergence, and Google
is the most typical leader and practitioner in that direction.

Google has on the one hand released Gemini, a general-purpose multimodal large
model, attempting to bridge the boundaries between text, temporal and
graph-structured tasks; on the other hand, Google Research has developed
TimesFM, a temporal foundation model pre-trained on hundreds of billions of
temporal data points, in an attempt to build a unified temporal prediction model
spanning finance, energy, recommendation and every other domain. Yet even with
the full efforts of leading enterprises such as Google, the inherent weaknesses
of general-purpose models remain unresolved: general large models generally
exhibit high inference latency and enormous computational consumption, and still
require substantial fine-tuning on unfamiliar niche datasets. TimesFM likewise
suffers from a lack of native support for joint multivariate modelling,
sensitivity to abrupt changes in the data, and poor inference efficiency in CPU
environments; it cannot adapt to arbitrary tasks entirely free of
scenario-specific customisation.

This conclusion is borne out within the present competition as well. Blindly
deepening the Transformer layers and enlarging the neighbour count can raise the
metrics in the short term, but the computational cost explodes non-linearly and
the result cannot be adapted to large-scale graph datasets with millions of
nodes and hundreds of millions of interactions. By contrast, the solutions that
have consistently reached the top in previous competitions were all built to
order around the regularities peculiar to their dataset; crude general-purpose
stacking is unlikely to lead stably over the long run.

### 2.4 Model selection

On this basis, our team drew on the industry algorithmic framework and designed
three components separately before integrating them: feature-vector training,
interaction-feature mining, and feature classification and prediction. This
reduces the precision of feature-vector training and of the other components,
but greatly facilitates computation and can accommodate the introduction of
larger-scale data later.

**First, feature-vector training.** For each dataset, two classes of embedding
model are trained using the **Jittor** framework: (i) a LINE model expressing
first-order and second-order structural features; and (ii) a BPR-MF model
expressing the structural difference between positive and negative samples,
which performs Bayesian personalised ranking using a negative-sampling
distribution derived from node degree (specifically, destination degree raised
to the power 0.75; see 5.2.2).

**Second, interaction-feature mining.** This covers collaborative filtering,
item-CF based on feature-vector similarity, popularity dynamics, interaction
history, and the feature vectors themselves. There are in addition features
peculiar to each dataset — for example, Dataset 1 exhibits an extremely strong
dependence on historically visited nodes, whereas Dataset 2 never revisits a
historical node. These ought to relate to the experimental background, but the
organisers did not supply that background, so they were mined by our team from
the datasets themselves.

**Third, feature classification and prediction.** Classification on the basis of
features is not itself an innovation. This task uses a **LightGBM LambdaRank**
model to predict over the candidate nodes on the basis of the several features
constructed above; the results are then adjusted according to the features
peculiar to each dataset, and the data is regularised before output.

### 2.5 Principal innovations

1. Mainstream graph-learning models are not adopted. Embedding vectors are
   trained directly with mature models such as LINE, which both reduces the
   information loss of the model and improves training efficiency.
2. Embedding vectors formed from a multi-seed mixture of BPR-MF models and LINE
   models, minimising as far as possible the error arising from the absence of
   prediction labels.
3. Noise in the collaborative-filtering matrix is effectively reduced by means
   such as SVD matrix factorisation, substantially improving the accuracy of the
   model.

### 2.6 Evidence boundary: score, lineage and reproduction

This subsection states precisely what the recorded score refers to and what is
and is not claimed about the submitted code. Five statements that are often
collapsed into one are kept separate.

**Score ownership.** The total 1.576996059163449 (Dataset 1: 0.8963013747474597;
Dataset 2: 0.6806946844159895) was measured online and belongs to the
historically accepted A-board prediction artifact. It belongs to that artifact
and to nothing else.

**No rescoring.** After the A board closed, the pipeline in this package was
re-executed independently and produced its own prediction files. Those files
were **not uploaded and were not rescored**. They therefore carry no score of
their own.

**No score interval.** No numeric score range is reported for the independently
reproduced outputs. Their similarity to the accepted files has been measured, but
prediction agreement, cross-top-2 coverage and rank correlation do not
mathematically determine MRR: the metric depends on where the correct answer
falls, and the test labels are held by the organiser. The reproduced rankings are
structurally close to the historically accepted rankings — for context, the
leading candidate agrees on 95.6% of Dataset-1 rows and 87.4% of Dataset-2 rows —
but their leaderboard score is not identifiable from the retained evidence,
because they were not rescored and no valid calibration is available.

**No byte-identity guarantee.** A fresh independent reproduction is not
guaranteed to be byte-identical to the historically accepted files. The reasons
are documented in 3.5 under *Reproduction boundary relative to the historical
A-board artifact*.

**Lineage, stated at four separate levels.**

| Level | Claim |
|---|---|
| Lineage consistency | The submitted pipeline follows the historical A-board production lineage: the same stage sequence, the same data cut, the same model families. |
| Downstream algorithm consistency | The feature sets, the LambdaRank parameters, the equality CRF, the two Dataset-1 postprocessors and the Dataset-2 decoder are retained without scientific change. |
| Exact producer identity | **Not claimed.** The submitted embedding trainers are a reconstruction, with the documented training differences in 3.5. |
| Byte identity and score equivalence | **Not claimed**, for either dataset. |

No test ground-truth label is used anywhere. No historical prediction file is
substituted for a reproduced member: every member this package produces is
computed by the packaged code from the raw competition data. The official
execution path is Jittor-only.

## 3. Code Structure

All paths in this section are relative to `code/` inside the archive. Every
statement was verified against the packaged source. The package contains 26 code
and configuration files and implements 33 canonical stages — 16 for Dataset 1
and 17 for Dataset 2.

Terms used throughout, defined once:

| Term | Meaning |
|---|---|
| **serve model** | a model trained on the complete training history, used to score the real test queries |
| **cutoff model** | the same model trained only on interactions at or before a fixed timestamp, used to build causally honest training features |
| **cut-split replay** | the scheme that manufactures ranker training labels without test labels: interactions before the time cut form the feature history, interactions after it supply the positives |
| **basket event** | a set of Dataset-2 test rows sharing the same `(source, time)` pair; they are answered together |
| **sibling feedback** | using the scores assigned to the *other* rows of the same basket event as features for a later ranking pass |
| **MF sibling geometry** | a rank-128 truncated SVD of the source x destination interaction matrix, used only to carry sibling messages. This is **not** the BPR-**MF** embedding model, which is a trained model |
| **completion record** | `<artifact>.done.json`, the document that proves a stage ran to completion; see 3.3 |

### 3.1 Directory layout

```
code/
+-- run_all.py                     one-command reproduction, both datasets
+-- main.py                        per-dataset entry point
+-- configs/
|   \-- production.json            machine-readable chain description
+-- src/
|   +-- canonical_pipeline.py      the 16- and 17-stage graphs, and the runner
|   +-- stage_contract.py          completion records, atomic publication
|   +-- pipeline_common.py         path contract and shared data utilities
|   +-- train_line_jt.py           LINE embeddings          (Jittor)
|   +-- train_bpr_jt.py            BPR-MF embeddings        (Jittor)
|   +-- ensemble_predict.py        embedding loading, collaborative scoring
|   +-- ranker_ds1.py              Dataset-1 LambdaRank, 21 features
|   +-- ranker_basket_ds2.py       Dataset-2 LambdaRank, 3-pass basket feedback
|   +-- ds2_basket_featurizer.py   Dataset-2 features and the feature cache
|   +-- ds2_mf_basket_pack.py      MF sibling geometry, production base matrix
|   +-- crf_promote.py             equality CRF and structural demotions
|   +-- build_ds1_member.py        Dataset-1 postprocessors and member
|   +-- build_ds2_member.py        Dataset-2 decode and member
|   \-- strategies/
|       +-- registry.py            the ordered postprocessor chains
|       +-- shared/frozen_ops.py   output gate, frozen primitives, serialisation
|       +-- ds1/source_slate_recurrence.py    Dataset-1 postprocessor 1
|       +-- ds1/graph_reciprocity.py     Dataset-1 postprocessor 2
|       \-- ds2/cross_time_exclusivity.py     Dataset-2 final decoder
\-- tools/submission/package_component.py     independent member verifier
```

Control flows strictly downward; each level does one job and delegates:

```
run_all.py                one command; both datasets, in the required order
  |
  +-> main.py             one dataset; selects a stage mode and a run context
        |
        +-> src/canonical_pipeline.py    the stage graph and the runner
              |
              +-> stage modules (trainers, rankers, CRF)  one subprocess per stage
                    |
                    +-> src/build_ds1_member.py / src/build_ds2_member.py
                          |
                          +-> outputs/members/dataset1.csv
                              outputs/members/dataset2.csv
```

### 3.2 Orchestration

#### `run_all.py` — one-command wrapper

*Invoked by*: the reviewer. *Invokes*: `main.py`, twice.

It imports **only the Python standard library** (`argparse`, `subprocess`,
`sys`, `pathlib`) and contains no model, feature, ranking, CRF or decoding
logic — it cannot change a result, only whether and in what order the two
pipelines run. It resolves the packaged directory from
`Path(__file__).resolve().parent`, so it works from any working directory, and
launches children with `sys.executable` so the interpreter running it is the
interpreter running the pipeline. `build_command()` constructs the child command
and `preflight()` performs the pre-run checks; both are pure functions.

| Behaviour | Implementation |
|---|---|
| Order | Dataset 1, then Dataset 2 — required, because `crf_promote.py` writes a two-member container and needs Dataset-1 member bytes as passthrough input |
| Dataset-1 failure | Dataset 2 is **not started**; the child's exit code is returned unchanged |
| Argument forwarding | `--data-root`, `--data-pack`, `--output-root`, `--log-dir`, `--fresh`, `--quiet` are passed to both datasets |
| Raw-data preflight | verifies that `main.py` exists, that `--data-root` and `--output-root` differ, and that both datasets' `train.csv` and `test.csv` are present; exits 2 with a named reason otherwise |
| `--plan` | adds `--stage plan` to both children; **no computation occurs** |
| `--fresh` | forwarded to both datasets; refuses all reuse |
| Validated resume | the default, inherited from `main.py`: a stage is reused only when its completion record proves it |
| Child output | streamed to the console as produced, never captured and hidden |
| Failure | fail-fast; the child's exit code propagates unchanged. Interruption returns 130 |

The expected raw-data layout, which the preflight checks, is:

```
<data-root>/data_A/dataset1/train.csv     columns: src, dst, time
<data-root>/data_A/dataset1/test.csv      columns: src, time, c1 ... c100
<data-root>/data_A/dataset2/train.csv
<data-root>/data_A/dataset2/test.csv
```

The archive contains **no competition data**. The default `<data-root>` is
`code/data`; `--data-root` points a run anywhere else.

#### `main.py` — per-dataset entry point

*Invoked by*: `run_all.py`, or the reviewer directly. *Invokes*:
`canonical_pipeline`.

Parses `--dataset {dataset1,dataset2,all}` and `--stage {run,plan,describe,
preprocess,postprocess,package}`, plus `--data-root`, `--data-pack`,
`--output-root`, `--log-dir`, `--config`, `--fresh`, `--ds1-member`,
`--verify` and `--quiet`. `run` is the default stage.

`contexts()` turns the arguments into one `RunContext` per dataset — the object
carrying the resolved data root, output root, log directory and resume policy.
`stage_run()` calls `canonical_pipeline.execute()`; `stage_plan()` calls
`canonical_pipeline.plan()` and executes nothing; `stage_describe()` prints the
chain from `configs/production.json`; `stage_preprocess()` loads the raw files
and reports the measured graph type; `stage_postprocess()` re-runs only the
member builders; `stage_package()` invokes the submission verifier. Runtime
variables are enforced here rather than left to the shell. There is **no backend
option**: the graph names the Jittor trainers and nothing else can be
substituted.

#### `src/canonical_pipeline.py` — the stage graph and the runner

*Invoked by*: `main.py`. *Invokes*: every stage module, as a subprocess.

`dataset1_stages()` and `dataset2_stages()` declare the two graphs — 16 and 17
`StageSpec` objects, in execution order — and `build_stages()` selects by
dataset. Each specification carries the stage id, the exact command, the
environment additions, the declared inputs and output, the scientific
configuration, the relevant source files, the requested epoch count and the
validator.

`run_stage()` executes one stage in a fixed, load-bearing order:

1. `stage_contract.evaluate()` decides `RUN`, `REUSE` or `STOP` from the
   completion record alone;
2. the stage runs, writing into an isolated `<output>.part`;
3. the staged artifact is validated;
4. it is published atomically by `os.replace`;
5. the completion record is written — last.

`stage_environment()` applies the Jittor runtime settings `conv_opt=1` and
`use_mkl=0` and exports this run's `DATA_ROOT` and `OUTPUTS_ROOT` to the child;
if the surrounding environment already sets either Jittor variable to a
conflicting value the run **stops** rather than overriding it.
`_run_subprocess()` streams the child's output to both the console and
`<output-root>/_logs/<stage_id>.log`. `execute()` loops over the graph and
returns non-zero at the first failure; no later stage is attempted.

**Dataset 2's dependency on Dataset 1** is implemented by `_passthrough_source()`:
it requires a Dataset-1 member carrying a valid completion record, or an
explicit `--ds1-member`. If neither exists the run stops with an explicit
message. No placeholder is generated and no frozen member is substituted.

### 3.3 The stage-completion contract

#### `src/stage_contract.py`

*Why it exists*: because a file existing on disk does not prove that the stage
which should have produced it ever finished. *Invoked by*: `canonical_pipeline`,
and directly by `ds2_basket_featurizer` for the feature cache.

Schema version 1. The record lives beside its artifact as `<output>.done.json`,
so the two travel together and deleting an output directory cannot leave an
orphaned claim. The principal API is `StageSpec` (the stage description),
`evaluate()` (the reuse decision), `complete_stage()` (validate then publish),
`publish_record()` (the atomic write), `atomic_output()` (a context manager
yielding a `.part` path) and `quarantine()` (the only sanctioned way past a
stop). Validators are `validate_embedding_csv`, `validate_embedding_npy`,
`validate_score_matrix_csv`, `validate_member_csv` and `validate_npz_cache`.

A record binds: the stage identity (`stage_id`, `dataset`, `artifact_kind`); the
`producing_commit`; a `code_identity` digest over the exact source files the
stage depends on; the `command` and `command_env`; the `environment`
(interpreter and installed package versions, read from metadata — nothing is
imported); every input's path, size and **SHA256**; the scientific `config` and
its digest; the `seed`; the `requested_units` and `achieved_units` (epochs); the
output's size and SHA256; the `validation` result; the `exit_code`; and both
timestamps.

Reuse requires **all** of the following to hold: the output exists; a record
exists; it parses and declares a supported schema version; `exit_code` is 0;
every input still hashes to its recorded value and the input set is unchanged;
the configuration digest matches; the code digest **and** the producing commit
match; the output's size and SHA256 match; achieved epochs equal requested
epochs; validation passed. **Any single mismatch is a stop** — never a silent
re-run, never a repair, never a deletion.

*Publication order and interruption.* The record is written only after the
artifact has been validated and published, by `fsync` followed by atomic rename.
A stage interrupted at any earlier point therefore leaves **no** record, and the
next run refuses its leftovers. This matters concretely: the LINE trainer
exports its embedding every 10 epochs, so an interrupted run leaves a file that
*looks* like a finished export — structurally perfect, correct row count, wrong
number of epochs behind it. Under the contract that file has no record and is
refused. This is why file existence alone never means completion.

### 3.4 Shared path, data and numeric utilities

#### `src/pipeline_common.py`

*Why it exists*: so that a producer and its consumer can never disagree about
where an artifact lives. *Invoked by*: both trainers, both rankers, the
featurizer, the MF pack and `canonical_pipeline`. It imports no deep-learning
framework at all, so the ranking stages obtain their utilities without pulling
one in.

*Path contract.* `data_root()` and `data_dir(dataset, pack)` resolve raw data,
honouring `DATA_ROOT`; `outputs_root()` resolves the artifact root, honouring
`OUTPUTS_ROOT` in preference to any argument; `line_run_dir()`, `bpr_run_dir()`
and `ranker_dir()` derive a run directory from its scientific parameters — seed,
negative distribution, embedding dimension, time cutoff, innovation flag,
holdout flag. Because the trainer that *writes* `line_run_dir("dataset2",
time_max=CUT)` and the ranker that *reads* it call the same function, a
redirected run cannot point them at different roots; a literal path in either
place would break exactly that guarantee.

*Data utilities.* `split_train_val_by_tail()`; `build_history_index()` and
`get_hist_before_time()` — per-source interaction history, sorted by time and
queried by cutoff, which is what enforces the causal boundary;
`count_in_history()`; `build_cooc()` and `cooc_scores()`, popularity-normalised
co-occurrence collaborative filtering; `cache_dict_to_matrix()` and
`batch_sim_score()`; `build_sim_cache()`, a two-band decayed similar-user cache
built by chunked cosine top-k in NumPy; and `rownorm()`, which divides a row by
its maximum.

#### `src/ensemble_predict.py`

*Invoked by*: `ranker_ds1`, `ranker_basket_ds2`, `ds2_basket_featurizer` and
`ds2_mf_basket_pack`, for `load_embedding()`, `build_base_cache()` and
`grouped_collab()`. `load_embedding(dir, expected_rows)` reads a
`line_latest_emb.csv` and asserts the row count, catching a mis-sized embedding
table at the boundary rather than deep inside featurisation.

The module also carries a legacy `run_predict()` / `main()` path that writes
`result_ensemble.csv` — a superseded hand-tuned linear blend. **It is not
reached by any of the 33 canonical stages**; it remains packaged because the
three functions above are on the canonical path and live in the same file.

### 3.5 Model training

These two modules are the **only** ones that import Jittor. Everything
downstream is NumPy, SciPy and LightGBM. In both trainers the division of labour
is deliberate: **seeded NumPy owns all sampling, shuffling and negative
rejection; Jittor owns only the parameter tables, the loss and the optimiser.**
This keeps the framework-dependent surface small and makes the sampling stream
reproducible independently of the framework. **Neither trainer imports
JittorGeometric.**

#### `src/train_line_jt.py` — LINE embeddings

*Owns the stages*: `ds1_line_full`, `ds1_line_cut`, `ds2_line_full`,
`ds2_line_cut`. *Reads*: `<data-root>/<pack>/<dataset>/train.csv`. *Writes*:
`line_latest_emb.csv` in the run directory given by
`pipeline_common.line_run_dir(...)`.

Principal objects: `class LINE`, `gen_neg_epoch()`, `build_pos_keys()`,
`export_emb()`, `main()`.

1. Load `train.csv`, drop duplicate `(src, dst, time)` rows, cast types.
2. **Size the entity table from the unfiltered frame, before any time cutoff.**
   This is load-bearing: sizing it after the cutoff would shrink the table to
   nodes seen before the cut and silently misalign every downstream node id.
3. Apply `LINE_TIME_MAX` when set — this is what distinguishes a cutoff run from
   a serve run.
4. Build bidirectional edges, interleaved `[u,v],[v,u]`, and a sorted key set of
   observed pairs.
5. `class LINE` holds three tables of `sub_dim = EMB_DIM/2 = 200`: `emb_first`
   (first-order), and `emb_node` with `emb_ctx` (second-order). All three are
   Xavier-uniform initialised from **one** NumPy stream; reseeding per table
   would make them identical at initialisation.
6. Per epoch, NumPy permutes the positives and `gen_neg_epoch()` draws
   `NEG_RATIO = 5` negatives per positive by **rejection sampling** against the
   observed-pair key set, retrying up to 30 times.
7. Per batch of 1,024: score first-order `emb_first(s)·emb_first(d)` and
   second-order `emb_node(s)·emb_ctx(d)`; the joint loss is
   `BCE(first) + LOSS_ALPHA * BCE(second)` with `LOSS_ALPHA = 0.5`; Adam at
   `INIT_LR = 1e-4`; gradient-norm clipping at 1.0.
8. Every `EXPORT_EVERY = 10` epochs and once at the end, `export_emb()` writes
   `node_id` plus the concatenation of `emb_first` and `emb_node`, rounded to 6
   decimals, through a temporary file and `os.replace`.

400 epochs, seed 42, uniform negatives. The current Jittor reproduction trains
LINE **only** from the retained training graph: it performs no
prediction-to-edge feedback and no virtual-edge retraining, and there is no
switch that would enable either. The `-novirt` marker in the run-directory name
records exactly that. Dataset 1 and Dataset 2 differ **only** in `DATASET` and
the cutoff timestamp; the architecture and all other parameters are identical.
`validate_embedding_csv` then checks the header, the field count and the row
count against the measured entity count before any completion record is written.

#### `src/train_bpr_jt.py` — BPR-MF embeddings

*Owns the stages*: the ten BPR stages per dataset, plus two innovation-only
stages on Dataset 1. *Reads*: the same `train.csv`. *Writes*: `bpr_emb.npy` in
`pipeline_common.bpr_run_dir(...)`.

A **single** embedding table of `BPR_DIM = 256` — source and destination share
it — trained with the Bayesian personalised-ranking objective
`softplus(-(e_src·e_dst - e_src·e_neg))`, that is `-log sigmoid(...)`. Negatives
are drawn from the destination-degree^0.75 distribution by inverse-CDF sampling
in NumPy. Adam, learning rate 3e-3, weight decay 1e-6, batch 8,192, 120 epochs.

Four switches distinguish the runs, all applied in NumPy before training:

| Switch | Effect | Used by |
|---|---|---|
| `SEED` | reseeds the NumPy stream and Jittor | all ten runs: 42, 123, 777, 2024, 31337 |
| `BPR_TIME_MAX` | keeps only interactions at or before the cut | the five cutoff runs |
| `BPR_TAU_FRAC` | recency-weighted positive sampling, `tau = frac * time span` | **Dataset 1 only**, at 0.25; Dataset 2 uses 0 |
| `BPR_INNOV` | keeps only the first occurrence of each `(src, dst)` pair | **Dataset 1 only**, two extra runs |

The entity table is again sized before the cutoff. The output is a single
`float32` array of shape `(num_entity, 256)`, written to a temporary path and
`os.replace`d; `validate_embedding_npy` checks shape, dtype and finiteness.

`conv_opt=1` and `use_mkl=0` are **runtime compatibility settings for this
image, not scientific parameters** (see 4.2). They select which Jittor kernels
load; they change no model, no hyperparameter and no result.

#### Reproduction boundary relative to the historical A-board artifact

A static comparison was made between the historical A-board trainer and the
current Jittor trainer. It found four differences in embedding training that can
change learned embeddings and therefore candidate rankings. They are recorded
here because a reader comparing a fresh reproduction against the historically
accepted files should know why the two need not match.

1. **Random-number-stream ownership and consumption differ.** The current trainer
   draws initialisation, epoch shuffling and negatives from one seeded NumPy
   generator; the historical A-board trainer drew them from its framework's own
   generator on the GPU. The same seed therefore yields a different sequence:
   different initial weights, a different edge order in every epoch and different
   negatives throughout.
2. **Optimiser state is handled differently.** The historical A-board trainer
   reset the Adam moment buffers at each 10-epoch export boundary — roughly
   thirty-nine resets across 400 epochs. The current trainer keeps optimiser
   state continuously for the whole run.
3. **Virtual-edge feedback.** The historical A-board serve-time LINE runs used
   virtual-edge feedback, in which high-confidence predicted pairs were merged
   back into the training graph and retrained; the retained Dataset-2 serve run
   holds a harvested set of 120,129 such edges. The historical cutoff runs did
   not use it, and the current Jittor reproduction does not implement it at all.
   This is one verified reason why independently reproduced embeddings and
   prediction rankings need not match the historically accepted artifact exactly.
4. **GPU kernels and floating-point reduction order may differ.** The historical
   hardware, driver and framework build were not retained, so no comparison of
   execution environments is possible and none is asserted.

What the same comparison found to be **unchanged**:

* the negative-sampling distribution, the rejection bound against observed pairs
  and the 30-attempt retry limit are aligned between the two versions;
* every downstream feature definition, count and order is unchanged;
* the LightGBM ranker hyperparameters and group construction are unchanged;
* `crf_promote.py`, the source-slate recurrence postprocessor and the test-graph
  reciprocity postprocessor have **no executable scientific difference**;
* every other difference identified outside embedding training is an engineering
  contract, orchestration, validation or serialisation difference — not a new or
  altered ranking algorithm.

Which of the four factors above contributes most is **not determined**. The
intermediate artifacts needed to separate them were not retained, so no dominant
cause is claimed.

### 3.6 Dataset-1 training and inference chain

From the raw files to `dataset1.csv`, in order.

**1–4. Embeddings.** The stage families of 3.5 produce, for Dataset 1: a serve
LINE embedding, a cutoff LINE embedding, five serve and five cutoff BPR
embeddings, and two innovation-only BPR embeddings.

**5. Cut-split replay** — how the ranker obtains labels without test labels.
`src/ranker_ds1.py` splits the training data at `CUT = 115,480,000`: `split0`
(before the cut) supplies all history and all features; `split1` (after the cut)
supplies the positives. For each `split1` interaction whose source appears in
the test file, the true destination becomes the positive and up to `NEG = 99`
negatives are sampled — **from that source's own test candidate pool** — so that
a training query has the same shape as a test query. The sampling RNG is seeded
at 20260727. **No test label is read; only candidate identities from
`test.csv`.**

**6–7. Feature construction.** `featurize()` produces a 21-column matrix,
identically for training and for inference; only the frozen history and the
embeddings differ. In source order:

| # | Feature | Group | Meaning |
|---|---|---|---|
| 0 | `f_collab` | collaborative | similar-user collaborative score, row-normalised |
| 1 | `f_rpop` | popularity | recent destination popularity |
| 2 | `f_icfm` | LINE-derived | mean cosine of the candidate to the source's history |
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

**8. Ranker training.** `lgb.LGBMRanker` with `objective="lambdarank"`,
`metric="ndcg"`, `ndcg_eval_at=[10]`, `n_estimators=400`, `learning_rate=0.05`,
`num_leaves=31`, `min_child_samples=100`, `label_gain=[0,1]`,
`random_state=42`, fitted with per-query group sizes.

**9. Inference.** The same `featurize()` runs with history frozen at the end of
the training data, the *serve* embeddings and the real test candidates. The two
forward-extrapolating time features (columns 12 and 19) are **clipped to the
range observed during training**, so the model never sees an out-of-support
value. `mdl.predict(Xte)` yields raw LambdaRank margins.

**10. Serialisation domain.** LambdaRank margins are unbounded and mostly
negative; `serialisation_normalise()` maps each row by **per-row min-max** into
`[0, 1]`, mapping a degenerate constant row to 0.5. This is strictly
order-preserving — it cannot change MRR — but it is load-bearing, because both
postprocessors and the output gate assume the unit interval. The result is
written atomically to `outputs/dataset1-ensemble/result_ranker.csv` at `%.6f`.

This transform is a **restoration of behaviour demonstrated by the accepted
artifact**, not an unverified change. The historically accepted Dataset-1
artifact carries a per-row min-max fingerprint on all 61,051 rows — row minimum
exactly 0 and row maximum exactly 1 on every row — which no other candidate
transform reproduces. The tracked historical ranker source at the tagged snapshot
instead divides each row by its maximum, and that snapshot is therefore **not
proven to be the literal producer of the accepted write site**; the producing
script's write site was not retained. An exhaustive static comparison over all
302,202,450 within-row candidate pairs found **zero** ranking changes
attributable to the difference between the two transforms. The consequence is
stated narrowly: the choice of transform changes numeric values and file hashes,
and it **cannot** explain a top-1 difference or any within-row ranking
difference.

**11–13. Deterministic post-processing.** `src/build_ds1_member.py` reads that
matrix and applies `registry.DS1_POSTPROCESSOR_CHAIN` in order. Neither stage
trains anything; both are frozen rules with no fitted parameters, and both read
only candidate identities and the training history — never a label.

* `ds1_source_slate_recurrence` — the inference batch is itself evidence. With
  `recurrence(c)` = the number of *other* test slates of the same source
  containing `c`: act only if the current top-1 is **not** a historical partner;
  consider only non-historical candidates; take the candidate with the
  **strictly unique** maximum recurrence (ties disqualify the row); require it
  to appear in at least 2 other slates; promote it to strict top-1. Acts on
  3,653 of 61,051 rows.
* `ds1_graph_reciprocity` — with `c1`, `c2` the rank-1 and rank-2
  candidates of the *recurrence-adjusted* matrix, act if and only if neither is
  a historical partner, the reverse test-exposure edge exists for `c2` and does
  not for `c1`; then promote `c2`. Acts on 1,365 rows.

**The order is load-bearing.** Reciprocity reads the rank-2 candidate of the
matrix the recurrence stage produced, so swapping the two yields a different
file.

**14–16.** `validate_submission_matrix` gates the float matrix,
`write_score_matrix` serialises it at `%.6f` with pinned CRLF through a `.part`
file, `verify_submission_file` checks the serialised row and field counts, and
`canonical_pipeline` publishes the artifact and its completion record.

### 3.7 Dataset-2 training and inference chain

**1. Raw loading and role measurement.** `main.py --stage preprocess` measures
the source/destination identity overlap; an empty overlap is the bipartite
signature. Nothing in the chain assumes it — the measurement is reported, not
asserted.

**2–3. Embeddings.** Two LINE runs and ten BPR runs, as in 3.5. Dataset 2 uses
no recency weighting and has no innovation-only variant.

**4–6. Features and cache.** `src/ds2_basket_featurizer.py` performs the same
cut-split replay at `CUT = 1,261,958,400` and builds an **18-column** base
matrix in `build_features()`:

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

`build_or_load_features()` caches this matrix as `train_features_all.npz`. The
cache is governed by its own completion record: written through `atomic_output`
and bound to the SHA256 of both official input files, the cut, the negatives per
sample, both sampling seeds, the entity count, the expected array names and the
relevant-code digest, then validated array-by-array on every reuse.

**7–10. Three-pass basket feedback.** `src/ranker_basket_ds2.py` trains three
LambdaRank models with the same parameters as Dataset 1 plus
`feature_fraction = 0.8`:

* **pass 1** fits `m1` on the 18 features; a 2-fold source-disjoint cross-fit
  produces unbiased in-sample scores `s1_tr`;
* `fb_features()` converts those scores into **two** sibling-feedback columns —
  the maximum and the mean score assigned to each candidate across the *other*
  rows of the same basket event;
* **pass 2** fits `m2` on 18 + 2 = 20 columns; cross-fit again to get `s2_tr`;
* **pass 3** fits `m3` on 20 columns built from `s2_tr`.

At serve time the three saved boosters (`ranker_pass{1,2,3}.txt`) are applied in
the same order to the real test rows, each pass regenerating the sibling
features from the previous pass's scores. The final row scores are min-max
normalised with historical partners masked out, and written to
`outputs/dataset2-ranker/ranker_basket3_dataset2.csv`.

**11–13. MF sibling geometry.** `src/ds2_mf_basket_pack.py` is the production
variant of that chain. `mf_geom()` computes a **rank-128 truncated SVD**
(`scipy.sparse.linalg.svds`) of the split-0 source x destination interaction
matrix and L2-normalises the rows; `fb_dense_ent()` then carries the sibling
messages through that dense geometry instead of the sparse item profiles. The
three-pass structure is otherwise identical, and the module reuses the
featurizer's cache and `build_features()` unchanged. Output:
`mf_basket3_dataset2.csv`, the **production base matrix**.

> **"MF" denotes two different things in this codebase.** The *BPR-MF embedding*
> of 3.5 is a **trained** model. The *MF sibling geometry* here is an
> **untrained** SVD factorisation used only to carry messages between the rows
> of a basket event. They are unrelated.

**14. Dataset-1 passthrough.** `crf_promote.py` writes a two-member container
and therefore needs Dataset-1 member bytes.
`canonical_pipeline._write_passthrough()` packages them from a Dataset-1 member
carrying a valid completion record. Those bytes are never read as scores — the
Dataset-2 member builder extracts `dataset2.csv` alone.

**15–19. Equality CRF and structural rules.** `src/crf_promote.py` runs with
`--tau 0.20 --B 70 --W 1 --zr-exclude --demote-dups --st-exclude --no-pair`:

* `build_band()` couples adjacent test rows that the raw row order shows to be
  related; `equality_crf()` propagates agreement across that chain with strength
  `tau` and `B`;
* `find_triples()` applies the **triple rule** — a structurally forced answer is
  promoted to strict top-1;
* the **pair rule is disabled** (`--no-pair`): it double-counts the chain the
  CRF already models;
* `zero_repeat_demotions()` applies **cross-time exclusivity** — because a
  `(source, answer)` pair never recurs at two distinct timestamps, a labelled
  cross-time answer is a negative elsewhere — and **duplicate-slate demotion**,
  since negatives are drawn with replacement, so a repeated slate id cannot be
  the answer;
* `same_time_structural_demotions()` adds the same-time structural rule: within
  one timestamp a shared answer forms one contiguous run in raw order, so a warm
  candidate in a shorter disjoint same-time component is a negative;
* `apply_demotions()` pushes demoted candidates strictly below every retained
  score.

The module re-reads its own output and re-asserts every invariant before
exiting. Output: `ds2_crf_intermediate.zip`.

**20–25. Final decode and serialisation.** `src/build_ds2_member.py` extracts
`dataset2.csv` from that container and applies
`strategies/ds2/cross_time_exclusivity.apply()`: group rows by
`(source, served top-1)`; a group spanning two or more distinct timestamps
contradicts the invariant; keep the timestamp cluster with the largest maximum
row margin, ties to the smallest timestamp; swap the rank-1 and rank-2 scores in
every row of every other cluster. One pass, no thresholds, no refit; 7,815 rows.

### 3.8 Final prediction generation, validation and serialisation

#### `src/strategies/shared/frozen_ops.py`

Shared by both member builders and by both Dataset-1 postprocessors.

| Function | Role |
|---|---|
| `validate_submission_matrix` | the **output gate**: shape, finiteness, and every value within `[0, 1]`. It reports and aborts; it **never clamps**, because a member failing here is a signal about the chain, and repairing it would destroy the evidence and change the ranking |
| `verify_submission_file` | structural gate on the *serialised* bytes — row count and per-row field count, catching a truncated or ragged write that an in-memory check cannot see |
| `write_score_matrix` | writes `%.6f` with pinned CRLF through a `.part` file, verifies the bytes, then renames |
| `row_max_normalise` | divides each row by its maximum; a row whose maximum is not strictly positive is shifted by its own minimum instead, and a constant row maps to 0.5 |
| `promoted_value` | the frozen strict-top-1 promotion value used by both Dataset-1 postprocessors |
| `stable_rank_order` | descending order, ties by ascending column index — the single ranking convention |
| `history_mask`, `pair_keys`, `membership` | directed `src -> dst` history membership |
| `dense_group_ids`, `csv_record_spans` | grouping and byte-offset support for the Dataset-2 decoder |

#### `src/strategies/registry.py`

Holds `DS1_POSTPROCESSOR_CHAIN` and `DS2_POSTPROCESSOR_CHAIN` — the ordered
`(strategy id, module, apply callable)` tuples that the member builders execute.
Its lifecycle-inventory helpers (`load_inventory`, `strategies_by_status`) read
`docs/strategy_inventory.json`, a development record that is **not shipped**;
they are lazy and are never called on the reproduction path.

#### How the two submission files are produced

1. The ranker writes a base score matrix — one row per test query, 100 values
   per row, `%.6f`. This is an **intermediate**, not a submission file.
2. Dataset 1: `build_ds1_member.py` applies the two frozen postprocessors in
   order and writes `dataset1.csv` (61,051 x 100).
3. Dataset 2: `crf_promote.py` applies the CRF and the invariant demotions, then
   `build_ds2_member.py` applies the cross-time exclusivity decode and writes
   `dataset2.csv` (153,420 x 100).
4. Before any byte is serialised, `validate_submission_matrix` asserts the
   expected shape, that every value is finite, and that every value lies within
   `[0, 1]`.
5. The serialised file is then checked structurally — row count and per-row
   field count.
6. The Dataset-2 decoder is **byte-preserving**: `swap_score_tokens()` rewrites
   only the two affected score *tokens* inside the base member's own bytes, so
   the other 15,326,370 cells are exactly the bytes the base produced and no
   value is re-serialised from a float. The result is re-read and its per-row
   candidate ranking compared against the float path; a disagreement fails the
   stage.

Outputs and logs:

| Artifact | Path |
|---|---|
| Dataset-1 base score matrix | `<output-root>/dataset1-ensemble/result_ranker.csv` |
| Dataset-1 final member | `<output-root>/members/dataset1.csv` |
| Dataset-2 base score matrix | `<output-root>/dataset2-ranker/mf_basket3_dataset2.csv` |
| Dataset-2 final member | `<output-root>/members/dataset2.csv` |
| Completion record, per stage | `<artifact>.done.json` |
| Stage log, per stage | `<output-root>/_logs/<stage_id>.log` |

### 3.9 Training versus inference

| Component | Training input | Learned state | Inference input | Produces | Framework / device | Jittor | Deterministic once inputs fixed | Uses test labels |
|---|---|---|---|---|---|:--:|---|:--:|
| LINE | `train.csv` edges (full or cutoff) | 3 embedding tables, 200-dim each | — (an output, not a scorer) | `line_latest_emb.csv` | **Jittor**, GPU | yes | No — GPU kernel scheduling is not bit-reproducible | **No** |
| BPR-MF | `train.csv` pairs, plus cutoff / recency / innovation filters | 1 embedding table, 256-dim | — | `bpr_emb.npy` | **Jittor**, GPU | yes | No — same reason | **No** |
| Dataset-1 LambdaRank | cut-split replay: `split0` features, `split1` positives, 99 sampled negatives | LightGBM booster, 400 trees | 21 features on the real test slates, serve embeddings, time features clipped | `result_ranker.csv` | LightGBM, CPU | no | Yes | **No** |
| Dataset-2 LambdaRank | cut-split replay, 18 features | 3 boosters (`pass1/2/3`) | the same 18 features on test rows | pass-3 scores | LightGBM, CPU | no | Yes | **No** |
| Basket-feedback passes | cross-fit scores of the previous pass | none of its own — 2 derived columns | sibling scores within a basket event | 20-column matrices | NumPy + LightGBM, CPU | no | Yes | **No** |
| MF sibling geometry | split-0 interaction matrix | **none — an untrained SVD factorisation** | candidate ids | rank-128 dense geometry | SciPy, CPU | no | Yes | **No** |
| Equality CRF | — **no training** | none | base scores, candidate ids, timestamps, raw row order | CRF container | NumPy, CPU | no | Yes | **No** |
| Dataset-1 postprocessors | — **no training** | none | score matrix, `test.csv` candidates, training history | adjusted matrix | NumPy, CPU | no | Yes | **No** |
| Dataset-2 final decoder | — **no training** | none | CRF base, `test.csv` sources and timestamps | member bytes | NumPy, CPU | no | Yes | **No** |
| Final serialisation | — | none | validated float matrix | `dataset{1,2}.csv` | NumPy/pandas, CPU | no | Yes | **No** |

Reading the table: **test candidate identities are used** — for inference, and
to shape the negative pool of the cut-split replay, which is what makes a
training query resemble a test query. **Test ground-truth labels are never
used**, are not available to this code, and appear nowhere in any stage; every
ranker label derives from training-data interactions after the time cut. The
postprocessors train nothing. Only LINE and BPR-MF use Jittor directly.

### 3.10 Stage traceability

All 33 canonical stages. Repeated BPR stages are shown one row per seed so that
every instance is explicit. Every stage requires a completion record before it
may be reused; every artifact is published atomically. The complete
machine-readable form, including declared inputs, is
`CANONICAL_STAGE_TRACEABILITY.csv`.

**Dataset 1 — 16 stages**

| # | Stage id | Module | Type | Principal input | Principal output | Device | Jittor |
|---|---|---|---|---|---|---|:--:|
| 1 | `ds1_line_full` | `train_line_jt.py` | TRAIN | `train.csv` | `dataset1-novirt/line_latest_emb.csv` | GPU | yes |
| 2 | `ds1_line_cut` | `train_line_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset1-novirt-tmax1.1548e+08/line_latest_emb.csv` | GPU | yes |
| 3 | `ds1_bpr_serve_s42` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset1-bpr-t0.25/bpr_emb.npy` | GPU | yes |
| 4 | `ds1_bpr_cut_s42` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset1-bpr-t0.25-tmax1.1548e+08/bpr_emb.npy` | GPU | yes |
| 5 | `ds1_bpr_serve_s123` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset1-bpr-t0.25-s123/bpr_emb.npy` | GPU | yes |
| 6 | `ds1_bpr_cut_s123` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset1-bpr-t0.25-tmax1.1548e+08-s123/bpr_emb.npy` | GPU | yes |
| 7 | `ds1_bpr_serve_s777` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset1-bpr-t0.25-s777/bpr_emb.npy` | GPU | yes |
| 8 | `ds1_bpr_cut_s777` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset1-bpr-t0.25-tmax1.1548e+08-s777/bpr_emb.npy` | GPU | yes |
| 9 | `ds1_bpr_serve_s2024` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset1-bpr-t0.25-s2024/bpr_emb.npy` | GPU | yes |
| 10 | `ds1_bpr_cut_s2024` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset1-bpr-t0.25-tmax1.1548e+08-s2024/bpr_emb.npy` | GPU | yes |
| 11 | `ds1_bpr_serve_s31337` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset1-bpr-t0.25-s31337/bpr_emb.npy` | GPU | yes |
| 12 | `ds1_bpr_cut_s31337` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset1-bpr-t0.25-tmax1.1548e+08-s31337/bpr_emb.npy` | GPU | yes |
| 13 | `ds1_bpr_innov_serve` | `train_bpr_jt.py` | TRAIN | `train.csv`, first occurrences only | `dataset1-bpr-innov-t0.25/bpr_emb.npy` | GPU | yes |
| 14 | `ds1_bpr_innov_cut` | `train_bpr_jt.py` | TRAIN | as above, t <= cut | `dataset1-bpr-innov-t0.25-tmax1.1548e+08/bpr_emb.npy` | GPU | yes |
| 15 | `ds1_ranker` | `ranker_ds1.py` | RANKER_TRAIN_AND_INFER | `train.csv`, `test.csv`, stages 1–14 | `dataset1-ensemble/result_ranker.csv` | CPU | no |
| 16 | `ds1_member` | `build_ds1_member.py` | DETERMINISTIC_POSTPROCESS + VALIDATE_AND_SERIALISE | stage 15, `train.csv`, `test.csv` | `members/dataset1.csv` | CPU | no |

**Dataset 2 — 17 stages**

| # | Stage id | Module | Type | Principal input | Principal output | Device | Jittor |
|---|---|---|---|---|---|---|:--:|
| 1 | `ds2_line_full` | `train_line_jt.py` | TRAIN | `train.csv` | `dataset2-novirt/line_latest_emb.csv` | GPU | yes |
| 2 | `ds2_line_cut` | `train_line_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset2-novirt-tmax1.26196e+09/line_latest_emb.csv` | GPU | yes |
| 3 | `ds2_bpr_serve_s42` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset2-bpr/bpr_emb.npy` | GPU | yes |
| 4 | `ds2_bpr_cut_s42` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset2-bpr-tmax1.26196e+09/bpr_emb.npy` | GPU | yes |
| 5 | `ds2_bpr_serve_s123` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset2-bpr-s123/bpr_emb.npy` | GPU | yes |
| 6 | `ds2_bpr_cut_s123` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset2-bpr-tmax1.26196e+09-s123/bpr_emb.npy` | GPU | yes |
| 7 | `ds2_bpr_serve_s777` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset2-bpr-s777/bpr_emb.npy` | GPU | yes |
| 8 | `ds2_bpr_cut_s777` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset2-bpr-tmax1.26196e+09-s777/bpr_emb.npy` | GPU | yes |
| 9 | `ds2_bpr_serve_s2024` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset2-bpr-s2024/bpr_emb.npy` | GPU | yes |
| 10 | `ds2_bpr_cut_s2024` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset2-bpr-tmax1.26196e+09-s2024/bpr_emb.npy` | GPU | yes |
| 11 | `ds2_bpr_serve_s31337` | `train_bpr_jt.py` | TRAIN | `train.csv` | `dataset2-bpr-s31337/bpr_emb.npy` | GPU | yes |
| 12 | `ds2_bpr_cut_s31337` | `train_bpr_jt.py` | TRAIN | `train.csv` (t <= cut) | `dataset2-bpr-tmax1.26196e+09-s31337/bpr_emb.npy` | GPU | yes |
| 13 | `ds2_ranker` | `ranker_basket_ds2.py` | RANKER_TRAIN_AND_INFER | `train.csv`, `test.csv`, stages 1–12 | `dataset2-ranker/ranker_basket3_dataset2.csv` | CPU | no |
| 14 | `ds2_mf_pack` | `ds2_mf_basket_pack.py` with `ds2_basket_featurizer.py` | FEATURE_BUILD + RANKER_TRAIN_AND_INFER | `train.csv`, `test.csv`, stages 1–12, feature cache | `dataset2-ranker/mf_basket3_dataset2.csv` | CPU | no |
| 15 | `ds2_ds1_passthrough` | `canonical_pipeline.py`, in-process | PASSTHROUGH | validated `members/dataset1.csv` | `dataset2-crf/ds1_passthrough.zip` | CPU | no |
| 16 | `ds2_crf` | `crf_promote.py` | DETERMINISTIC_POSTPROCESS | stage 14, stage 15, `train.csv`, `test.csv` | `dataset2-crf/ds2_crf_intermediate.zip` | CPU | no |
| 17 | `ds2_member` | `build_ds2_member.py` | DETERMINISTIC_POSTPROCESS + VALIDATE_AND_SERIALISE | stage 16, `test.csv` | `members/dataset2.csv` | CPU | no |

The Dataset-2 feature cache (`src/bagging_cache/train_features_all.npz`) is
built inside stage 14 and carries its own completion record; it is not a
separate driver stage.

### 3.11 Complete packaged-code inventory

All 26 files under `code/`. Reachability: **DIRECT** = named in a stage command
or in the entrypoint chain; **TRANSITIVE** = imported by a DIRECT module;
**SUPPORTING_TOOL** = a working utility not on the reproduction path;
**INFORMATIONAL_ONLY** = present for completeness, never executed. The complete
machine-readable form, including the principal caller, input and output of each
file, is `PACKAGED_CODE_INVENTORY.csv`.

| Path | Category | Function | Reachability | Runs in a full reproduction? |
|---|---|---|---|---|
| `run_all.py` | entrypoint | one-command reproduction, both datasets | DIRECT | yes |
| `main.py` | entrypoint | per-dataset entry, stage modes | DIRECT | yes |
| `src/canonical_pipeline.py` | orchestration | stage graphs, fail-closed runner | DIRECT | yes |
| `src/stage_contract.py` | orchestration | completion records, atomic publication, validators | TRANSITIVE | yes |
| `src/pipeline_common.py` | shared | path contract, history, co-occurrence, similarity | TRANSITIVE | yes |
| `src/train_line_jt.py` | training | LINE embeddings (Jittor) | DIRECT | yes, 4 stages |
| `src/train_bpr_jt.py` | training | BPR-MF embeddings (Jittor) | DIRECT | yes, 22 stages |
| `src/ensemble_predict.py` | inference | embedding loading, collaborative scoring | TRANSITIVE | yes (helpers only; its legacy `main()` is never called) |
| `src/ranker_ds1.py` | ranking | Dataset-1 LambdaRank, 21 features | DIRECT | yes |
| `src/ranker_basket_ds2.py` | ranking | Dataset-2 LambdaRank, 3-pass basket feedback | DIRECT | yes |
| `src/ds2_basket_featurizer.py` | features | Dataset-2 features and the contract-governed cache | TRANSITIVE | yes |
| `src/ds2_mf_basket_pack.py` | ranking | MF sibling geometry, production base matrix | DIRECT | yes |
| `src/crf_promote.py` | postprocessing | equality CRF and structural demotions | DIRECT | yes |
| `src/build_ds1_member.py` | final output | Dataset-1 postprocessor chain and member | DIRECT | yes |
| `src/build_ds2_member.py` | final output | Dataset-2 decode and member | DIRECT | yes |
| `src/strategies/registry.py` | postprocessing | ordered postprocessor chains | TRANSITIVE | yes (chain constants only) |
| `src/strategies/shared/frozen_ops.py` | postprocessing | output gate, frozen primitives, serialisation | TRANSITIVE | yes |
| `src/strategies/ds1/source_slate_recurrence.py` | postprocessing | Dataset-1 postprocessor 1 | TRANSITIVE | yes |
| `src/strategies/ds1/graph_reciprocity.py` | postprocessing | Dataset-1 postprocessor 2 | TRANSITIVE | yes |
| `src/strategies/ds2/cross_time_exclusivity.py` | postprocessing | Dataset-2 final decoder | TRANSITIVE | yes |
| `src/strategies/__init__.py` | package marker | package initialiser | TRANSITIVE | yes |
| `src/strategies/ds1/__init__.py` | package marker | package initialiser | TRANSITIVE | yes |
| `src/strategies/ds2/__init__.py` | package marker | package initialiser | TRANSITIVE | yes |
| `src/strategies/shared/__init__.py` | package marker | package initialiser | TRANSITIVE | yes |
| `configs/production.json` | configuration | machine-readable chain description and artifact anchors | INFORMATIONAL_ONLY | no — reproduction does not read it |
| `tools/submission/package_component.py` | verification | independent submission-format verifier | SUPPORTING_TOOL | no — offered so a reviewer can check a member independently |

**Two files do not execute during a normal reproduction**, and both are included
deliberately: `configs/production.json` records the chain description and the
artifact anchors so that a reviewer can compare against them, and
`tools/submission/package_component.py` lets a reviewer validate a produced
member without trusting our tooling. Neither is required by `python run_all.py`.

### 3.12 Failure, interruption and validated resume

The run stops at the first stage that fails and returns a non-zero exit code;
subprocess failures propagate. **No completion record is written** for a stage
that exited non-zero, produced a partial artifact, failed validation or
completed fewer epochs than requested. A stopped run leaves its artifacts in
place — they are the evidence — and prints the stage, the reason code and the
properties that differed. Nothing is deleted and nothing is silently repaired.

Every stage boundary is a resume point. There are no resume points *inside* a
stage: an interrupted trainer or feature build restarts that stage from its
beginning.

## 4. Environment Setup

### 4.1 Base environment

| Name | Content |
|---|---|
| Operating system | Ubuntu 22.04 |
| GPU | NVIDIA RTX 4090 |
| NVIDIA driver | 580.76.05 |
| CUDA Toolkit | 12.4 |
| Python | 3.10 |
| Jittor | 1.3.10 |
| JittorGeometric | commit `ff7d8ffac7bf3d95cc1962e091c52dc5737492d4` (setup.py version 2.0.0) |
| NumPy | 1.26.4 |
| Pandas | 2.2.3 |
| SciPy | 1.15.2 |
| scikit-learn | 1.6.1 |
| LightGBM | 4.5.0 |
| tqdm | 4.67.1 |

```
conda env create -f environment.yaml
conda activate jittor-link-prediction-inspect

# JittorGeometric is not on PyPI and upstream publishes
# no formal release tag, so the version is pinned by
# commit hash and installed as a separate step.
pip install --no-build-isolation --no-deps \
  git+https://github.com/AlgRUC/JittorGeometric.git@ff7d8ffac7bf3d95cc1962e091c52dc5737492d4
```

### 4.2 Required Jittor runtime settings

```
export conv_opt=1
export use_mkl=0
```

These settings are not performance-tuning options; they are required for correct
operation in the current image. Jittor 1.3.10 loads the cuDNN 8 component
libraries by name (`libcudnn_ops_infer.so` and siblings), whereas cuDNN 9 has
merged those components into `libcudnn.so.9`. With `conv_opt=1` set, Jittor uses
the cuBLAS path that can currently be resolved. Setting `use_mkl=0` avoids
invoking a DNNL verification program that does not terminate correctly on this
host.

## 5. Run Procedure

### 5.1 Reproduction commands

```
python main.py --dataset dataset1
python main.py --dataset dataset2
```

The program uses the `run` full-pipeline stage by default and will proceed
automatically through data preprocessing, Jittor model training and inference
scoring, and byte-fixed post-processing, finally producing the standardised
submission files in the `output-root` directory. A single-command wrapper,
`python run_all.py`, runs both datasets in the required order; see 3.2.

### 5.2 Key hyperparameters

#### 5.2.1 LINE embedding (`train_line_jt.py`)

| Parameter | Value |
|---|---|
| Total embedding dimension | 400 (two 200-dimensional parts) |
| Training epochs | 400 |
| Batch size | 1,024 |
| Negatives per positive | 5 |
| Negative-sampling distribution | uniform |
| Optimiser | Adam, learning rate 1e-4 |
| Loss function | `BCE(first-order) + 0.5 * BCE(second-order)` |
| Gradient-norm clipping | 1.0 |
| Random seed | 42 |
| Export precision | 6 decimal places; exported every 10 epochs |
| Dataset-1 time cut | cutoff run uses `LINE_TIME_MAX=115480000` |
| Dataset-2 time cut | cutoff run uses `LINE_TIME_MAX=1261958400` |
| Virtual-edge self-training | disabled (measured to have no effect) |

#### 5.2.2 BPR-MF embedding (`train_bpr_jt.py`)

| Parameter | Value |
|---|---|
| Dimension | 256 |
| Training epochs | 120 |
| Batch size | 8,192 |
| Optimiser | Adam, learning rate 3e-3, weight decay 1e-6 |
| Negative sampling | destination degree to the power 0.75 |
| Random seeds | 42, 123, 777, 2024, 31337 |
| Temporal recency weighting | Dataset 1 uses `BPR_TAU_FRAC=0.25`; Dataset 2 uses 0 |
| Innovation-only variant | Dataset 1 only (`BPR_INNOV=1`, keeping only the first occurrence of each node pair) |

#### 5.2.3 Dataset-1 ranker (`ranker_ds1.py`)

| Parameter | Value |
|---|---|
| Model | LightGBM `LGBMRanker` |
| Objective / metric | `lambdarank` / NDCG@10 |
| Number of base learners | 400 |
| Learning rate | 0.05 |
| Number of leaves | 31 |
| Minimum samples per leaf | 100 |
| `label_gain` | `[0, 1]` |
| Random state | 42 |
| Number of features | 21 |
| Time cut | 115,480,000 |
| Negatives per training query | 99 |
| Negative-sampling random seed | 20260727 |
| Time features requiring clipping | `time_gap`, `last_gap` |
| Serialisation | per-row min-max normalisation into `[0, 1]` |

#### 5.2.4 Dataset-2 ranker (`ranker_basket_ds2.py`)

| Parameter | Value |
|---|---|
| Model | LightGBM `LGBMRanker`, `lambdarank`, NDCG@10 |
| Base learners / learning rate / leaves | 400 / 0.05 / 31 |
| Minimum samples per leaf | 100 |
| Feature sampling fraction | 0.8 |
| Number of features | 18 |
| Same-event-group feedback passes | 3 |
| Time cut | 1,261,958,400 |
| Random seeds | 42, 123, 777, 2024, 31337 |
| Negatives per training query | 99 |
| MF geometry | rank-128 truncated SVD of the split-0 source x destination matrix |

#### 5.2.5 Dataset-2 CRF (`crf_promote.py`)

| Parameter | Value |
|---|---|
| `--tau` | 0.20 |
| `--B` | 70 |
| `--W` (band radius) | 1 |
| `--p` (neighbour marginal-probability exponent) | 1.0 |
| `--eta` (candidate-reliability exponent) | 0.0 |
| Triple rule | enabled |
| Pair rule | disabled (`--no-pair`) — this rule would double-count the chain relationship the CRF already models |
| Cross-time zero-repeat exclusion | enabled (`--zr-exclude`) |
| Duplicate-candidate-set demotion | enabled (`--demote-dups`) |
| Same-time structural exclusion | enabled (`--st-exclude`) |

## 6. Runtime and Resource Requirements

### 6.1 Measured runtimes

The validation host is configured with an RTX 4090 24 GB, a Xeon Gold 6430, 16
CPU cores and 120 GiB of memory.

| Stage | Device | Measured time |
|---|---|---|
| Dataset-1 LINE, full history | GPU | 2,309 s |
| Dataset-1 LINE, cutoff history | GPU | 1,802 s |
| Dataset-1 BPR, 12 runs in total | GPU | included in the total below |
| Dataset-1 ranker | CPU | included in the total below |
| **Dataset 1: raw data to member file** | | **1 h 51 m 42 s** |
| Dataset-2 LINE, full history | GPU | 15,467 s |
| Dataset-2 training feature cache | CPU | 10,115 s |
| Dataset-2 MF same-event-group packing | CPU | 8,837 s |
| Dataset-2 CRF | CPU | 120 s |
| Dataset-2 member file decode | CPU | 9 s |
| **Dataset 2: raw data to member file** | | **not reported — see the note below** |

Dataset 1 was measured as one continuous run, and its sixteen stage durations
sum to 6,698 s against a driver wall clock of 6,702 s, the 4 s difference being
inter-stage orchestration.

Dataset 2 was produced across separate sessions rather than as one continuous
run, and the driver log covering its training stages was not retained. The five
retained stage measurements above account for **at least 9 h 35 m 48 s
(34,548 s)**, but twelve of the seventeen stages remain untimed and no complete
end-to-end wall-clock measurement was retained. A total runtime for Dataset 2 is
therefore **not** stated here rather than estimated.

### 6.2 Resources and resumability

Peak disk usage is approximately 5 GB of artifacts per dataset, plus roughly
1.92 GiB for the Dataset-2 training feature cache. Peak GPU memory is
comparatively low, dominated by the embedding tables, and runs comfortably
within 24 GB.

**On resumability.** Every stage boundary is a resume point, governed by the
stage-completion contract. No resume point is offered inside a single stage: if
a trainer or a feature-build stage is interrupted, it must be re-executed from
the beginning of that stage. The longest stage that cannot be resumed part-way
is the Dataset-2 feature build, at approximately 2.8 h.

### 6.3 Known issues and reproduction constraints

Everything below is relevant to reproducing this submission. Nothing here is a
defect in the model.

**Target environment.** Ubuntu 22.04, Python 3.10, CUDA 12.4, Jittor 1.3.10 and
JittorGeometric 2.0.0, on an NVIDIA GPU of compute capability 8.9 or compatible.
Other combinations were not validated.

**Required runtime settings.** `conv_opt=1` and `use_mkl=0` must be exported
before any stage runs (see 4.2). They are compatibility settings for this image,
not tuning options.

**cuDNN component probe on the target image.** Jittor 1.3.10 resolves the cuDNN 8
component libraries by name, and cuDNN 9 has merged those components into a
single shared object, so the component probe does not succeed on this image. The
validated response is the runtime configuration above, under which every stage
completes and every validation gate passes. This is a statement about the probe
and the validated configuration only — no claim is made that cuDNN is healthy in
general on this image, and no other configuration was tested.

**Dataset-2 resource envelope.** Dataset 2 dominates the run. Its training
feature cache alone is roughly 1.92 GiB, its longest non-resumable stage is about
2.8 h, and its stages are substantially more demanding than Dataset 1's in disk,
memory and time. Plan the run accordingly.

**Stage-level resume is validated; sub-stage resume is not offered.** See 6.2.

**Dataset-2 end-to-end wall clock was not measured.** Five retained stage
measurements account for at least 9 h 35 m 48 s (34,548 s); twelve of the
seventeen Dataset-2 stages have no retained timing, and no complete end-to-end
measurement exists. A total is therefore not stated rather than estimated.

**Independent outputs may not be byte-identical to the historically accepted
artifact.** The reasons are set out in 2.6 and 3.5. This is expected behaviour
for a re-execution of stochastic embedding training, not an error condition, and
every validation gate in the pipeline still applies.
