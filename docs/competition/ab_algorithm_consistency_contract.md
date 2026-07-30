---
title: A/B-Board Algorithm Consistency Contract
document_id: JITTOR_2026_AB_ALGORITHM_CONSISTENCY_CONTRACT
version: 1.0.0
status: FROZEN
language: English
frozen_utc: 2026-07-30
companion_dossier: docs/competition/official_competition_dossier.md
companion_source_register: docs/competition/source_register.md
---

# A/B-Board Algorithm Consistency Contract

## 1. Purpose and status

This contract freezes the algorithmic skeleton that the B-board implementation must share with the
A-board submission reviewed at code inspection. It exists so that every B-board change can be
classified, before it is made, as either a permitted configuration adaptation or a prohibited
undeclared change.

It is **frozen** as of 2026-07-30, before B-board development begins. Amending it requires the
change-control procedure in the dossier: date, source, affected clause, prior position, revised
position, project impact.

Each clause carries one of four labels:

| label | meaning |
|---|---|
| **OFFICIAL** | stated in official competition material, cited to a source ID in [source_register.md](source_register.md) |
| **POLICY** | a conservative project commitment adopted while an ambiguity is open. It binds this project and carries no official authority |
| **CLARIFICATION REQUIRED** | an ambiguity that needs written organiser confirmation before it can be relied on |
| **COMMITMENT** | an implementation obligation this project accepts and can be audited against |

## 2. The governing tension, unresolved

**CLARIFICATION REQUIRED.** Two official statements pull in different directions:

- `SRC-003` (Track 1 page) permits neural-network models, learned weights and training parameters to
  **differ across scenarios**.
- `SRC-004` (A-board code-inspection notice) requires the B-board code, algorithm and model to be
  **fully consistent** with the A-board submission and prohibits unexplained modification.

This contract **does not claim the tension is resolved**, and nothing below should be read as an
organiser ruling. It is recorded as `TENSION-001` in the source register and `OCI-001` in the
dossier. The narrow questions are drafted for submission to the organisers.

**POLICY** pending that answer: adopt the stricter reading. One reviewed algorithmic family, one
reviewed code path, scenario differences expressed only through disclosed configuration and trained
weights that the reviewed A-board package already supports. If the organisers later permit
scenario-specific architectures, this contract can be relaxed deliberately; the reverse is not
possible after the inspection package is submitted.

## 3. The invariant skeleton

**COMMITMENT.** These are frozen for both boards. Changing any of them is an algorithm change, not a
parameter change, and must be declared.

| # | Invariant | Frozen position |
|---|---|---|
| 3.1 | Task formulation | Temporal candidate-set reranking. Given a source, a query timestamp and a supplied candidate slate, score every candidate. Never generate candidates, never reorder or drop them |
| 3.2 | Causal boundary | A feature may read only interactions at or before the applicable boundary. Train-side features are frozen at the training cut; serve-side features at the full-train maximum. No test label, no future interaction, no answer-derived quantity, no other query's answer |
| 3.3 | Graph-type detection | Measured, never assumed: the source/destination identity overlap is computed from the training edges. Empty overlap means bipartite. Both types must execute |
| 3.4 | Role semantics | Interactions are directed source to destination. History is directed. A reverse edge is evidence of reciprocity, never a duplicate of the forward edge |
| 3.5 | Feature meanings | A feature name denotes one quantity across both boards. Collaborative score, item-CF similarity, recency, degree/popularity, history membership, slate footprint and basket-sibling geometry keep their A-board definitions |
| 3.6 | Training-objective family | Pairwise and listwise ranking only: BPR pairwise ranking and LINE proximity for representations, LambdaRank for slate scoring. No switch to pointwise regression, classification or a generative objective |
| 3.7 | Representation family | Shallow embedding tables over node identity, plus a matrix-factorisation geometry for basket messages. No sequence model, no attention stack, no pretrained external encoder |
| 3.8 | Candidate-set interface | The slate is an ordered list of candidate identities per query. Candidate column position is never evidence (measured on both A-board scenarios: z = +0.07) |
| 3.9 | Candidate scoring | One real score per candidate, written in `[0, 1]` with six decimal places, row order and candidate order exactly as supplied |
| 3.10 | Post-processing semantics | Each post-processing stage is a frozen structural rule with a stated invariant, not a tuned filter. The rules are: slate recurrence, test-graph reciprocity, equality CRF over the same-time band, and cross-time exclusivity decode |
| 3.11 | Stage ordering | Fixed and order-sensitive; see the code-path matrix in section 6 |
| 3.12 | External-data boundary | Competition-provided data only. No external records, metadata, popularity statistics, entity matching or another team's assets |
| 3.13 | Pretrained-artifact provenance | Every model artifact is trained from competition data by tracked code in this repository. No externally pretrained weights are used on either board |
| 3.14 | Submission generation | Produced by tracked code through `main.py` and the `src/build_ds*_member.py` entry points, then packaged and hash-verified by `tools/submission/package_component.py`. Never hand-edited |
| 3.15 | Validation requirements | Row count, candidate count, value range, schema, member hash and container fields are asserted before any submission |

## 4. Permitted variation

**POLICY.** These may differ between scenarios and between boards, through configuration or training
output only, and must be disclosed in the reproduction document. Every one of them is already a
configuration surface of the reviewed A-board code; none requires new code.

Embedding dimension; hidden dimension; batch size; epoch count; learning rate; regularisation
strength; negative-sampling count; sampling rate and recency-weighting fraction; graph partition
size; cache size; candidate-processing batch size; number of workers; trained weight values;
scenario-specific calibration of an existing frozen rule's declared parameter; and
hardware-specific execution settings such as device placement, precision and thread count.

Two conditions apply to all of them:

1. the knob must already exist in the reviewed A-board code, and
2. the value used for each board must be recorded in the reproduction document.

## 5. Prohibited without declaration

**COMMITMENT.** Each of the following is an algorithm change. None may appear on the B board unless
it is present and documented in the reviewed A-board package, or an organiser clarification permits
it and is registered as `ORGANISER_CLARIFICATION`.

- a different scientific mechanism, or a new post-processing stage absent from the reviewed code;
- a different training-objective family;
- a B-only model with no counterpart in the reviewed A-board code;
- a different graph-role interpretation, including merging disjoint role identifiers or splitting a
  shared identity;
- a different causal boundary, including any widening of what a feature may read;
- use of external data;
- undisclosed pretrained weights;
- manual editing of a result file;
- any access to test ground truth;
- removing or reordering a frozen stage;
- turning a frozen structural rule into a swept hyperparameter.

## 6. Code-path matrix

**COMMITMENT.** The reviewed implementation, per scenario and mode. The A-board values are measured,
not asserted: `python main.py --stage preprocess` computes them from the released data.

| axis | dataset1 | dataset2 |
|---|---|---|
| graph type | NON_BIPARTITE (role overlap 21,253; 18,076 self loops) | BIPARTITE (role overlap 0; 0 self loops) |
| nodes | 22,093 sources, 23,012 destinations | 12,708 sources, 50,640 destinations |
| interactions | 690,848 | 2,261,283 |
| queries | 61,051 | 153,420 |
| candidates per query | 100 | 100 |
| representation | `src/train_line.py` / `src/train_line_jt.py`, `src/train_bpr.py` / `src/train_bpr_jt.py` | same modules, same objectives |
| slate scorer | `src/ranker_ds1.py` | `src/ranker_basket_ds2.py` with `src/ds2_mf_basket_pack.py` |
| post-processing, in order | `strategies/ds1/source_slate_recurrence.py`, then `strategies/ds1/test_graph_reciprocity.py` | `src/crf_promote.py`, then `strategies/ds2/cross_time_exclusivity.py` |
| member builder | `src/build_ds1_member.py` | `src/build_ds2_member.py` |
| entry point | `python main.py --dataset dataset1` | `python main.py --dataset dataset2` |

**Stage ordering is load-bearing.** On dataset1, reciprocity reads the rank-2 candidate of the
recurrence-adjusted matrix. On dataset2, the cross-time decode groups by *served* top-1 and so must
observe the CRF's output, and the CRF depends on the raw test row order, which must never be sorted.

Mode coverage. `A-scale mode` is the released A-board data and is the configuration under review.
`B-scale mode` is the same code path with the scale configuration of section 4; it has **not** been
exercised, because the B-board data has not been released. Every B-scale figure in this repository is
a projection and is labelled as one.

### 6.1 Bipartite scenarios

**COMMITMENT.** Source and destination role sets are disjoint. The implementation must:

- keep the two role sets distinct and never assume a shared namespace without evidence from the data;
- avoid accidental node-identity coupling: a source id and a destination id with the same integer
  value are different entities unless the measured overlap says otherwise;
- treat only cross-role interactions as valid, since no same-role interaction exists;
- use role-conditioned or separately indexed representations.

dataset2 is the reviewed bipartite scenario, with a measured overlap of exactly 0.

### 6.2 Non-bipartite scenarios

**COMMITMENT.** Source and destination share one node universe. The implementation must:

- use one shared node index, so a node's representation is the same in both roles;
- support dual-role nodes, which dataset1 has 21,253 of;
- support self-loops, which dataset1 has 18,076 of, without treating them as degenerate;
- keep behaviour role-specific where the mechanism is directional, notably the directed history
  definition and the reciprocity rule;
- never double-count an interaction by adding it to both a forward and a reverse structure that are
  later summed.

dataset1 is the reviewed non-bipartite scenario.

## 7. B-board scale requirements

**OFFICIAL** (`SRC-003`): B-board data may contain millions of nodes and tens of millions of
interactions, with explicit efficiency constraints.

Notation: `N` nodes, `E` interactions, `Q` queries, `C` candidates per query, `D` representation
dimension.

**COMMITMENT.** The implementation must satisfy:

| requirement | obligation |
|---|---|
| preprocessing | sparse or streaming, `O(E)` time and `O(E)` memory. No dense adjacency |
| training | chunkable: mini-batch over edges, `O(E * D)` per epoch, memory bounded by batch size and `O(N * D)` for the embedding table |
| inference | bounded memory, streaming over queries in batches, `O(Q * C * D)` total work with per-batch memory independent of `Q` |
| scoring | candidate-slate scoring only, `O(C)` per query |

**Prohibited by construction**, at any scale:

- a dense `N x N` matrix of any kind;
- all-pairs scoring of nodes or candidates;
- full-graph quadratic attention;
- unbounded per-node interaction history held in memory;
- a Python-object-per-edge design, which does not survive tens of millions of edges.

The A-board post-processing stages are already linear or near-linear: the frozen rules are
implemented as vectorised grouping over the query matrix, and the cross-time decode is a single
`O(Q log Q)` lexicographic grouping pass with no per-node history.

Two structural invariants need re-measurement rather than assumption on B-board data, because each
is a property of a released file and not a law:

1. the same-time adjacency agreement the equality CRF exploits;
2. the zero-repeat exclusivity invariant the cross-time decode relies on.

**COMMITMENT.** Both are measured on the B-board data before the corresponding stage is enabled. If
an invariant does not hold, the stage is disabled by configuration rather than retuned.

## 8. Configuration matrix

**COMMITMENT.** Invariant semantics against the A-board value and the permitted B-board adaptation.

| invariant semantics | A-board value | permitted B-board adaptation | maximum scope of change | review evidence |
|---|---|---|---|---|
| embedding capacity | LINE 400 total, BPR 256 | any dimension | value only | `configs/production.json`, trainer knobs |
| training length | LINE 400 epochs, BPR 120 | any epoch count | value only | trainer knobs |
| negative sampling | ratio 5, degree exponent 0.75 | any ratio; the exponent stays 0.75 | ratio value only | `src/train_bpr.py` |
| recency weighting | dataset1 0.25, dataset2 0.10 | any fraction, including 0 | value only | `BPR_TAU_FRAC` |
| seed ensemble | dataset1 10 seeds, dataset2 5 | any count | value only | `ranker_ds1.BPR_SEEDS`, `ranker_basket_ds2.SEEDS` |
| basket geometry rank | d128 unit-norm SVD | any rank | value only | `src/ds2_mf_basket_pack.py` |
| equality CRF sharpening | tau 0.20, B 70, W 1 | re-measured per scenario, or disabled | declared parameters, or off | `src/crf_promote.py` |
| slate recurrence support | at least 2 other slates, uniqueness required | value only; uniqueness stays required | threshold value only | `strategies/ds1/source_slate_recurrence.py` |
| reciprocity rank depth | 2 | value only | depth value only | `strategies/ds1/test_graph_reciprocity.py` |
| cross-time exclusivity | at least 2 distinct timestamps; one pass | nothing tunable; enabled or disabled | on or off only | `strategies/ds2/cross_time_exclusivity.py` |
| execution settings | single GPU, float32 | device, precision, workers, batch, partition, cache | performance only, no numeric semantics | `environment.yaml` |

The last two rows are the tightest on purpose. The cross-time decode has no threshold to tune, and
the frozen rules' declared parameters were fixed by online adjudication; re-fitting them would turn a
structural rule into a swept hyperparameter, which section 5 prohibits.

## 9. Framework requirement and the disclosed provenance gap

**OFFICIAL** (`SRC-002`, `SRC-003`, `SRC-004`): Jittor must be used for model design, training and
prediction. Failure to do so is grounds for invalidation.

**Disclosed fact, not an interpretation.** The accepted A-board embeddings were produced by the
PyTorch trainers `src/train_line.py` and `src/train_bpr.py`. The Jittor ports
`src/train_line_jt.py` and `src/train_bpr_jt.py` implement the same models, objectives, knobs and
output formats, and Jittor owns the model, loss and optimiser in them. Sampling and shuffling run in
seeded NumPy in both implementations, so the two agree statistically but **not** byte for byte: a
Jittor retrain produces a different embedding table by construction, and therefore a different
member hash.

**COMMITMENT.**

1. Both trainers ship in the inspection package, and the provenance above is stated in the
   reproduction document rather than implied.
2. The Jittor implementation is the framework-compliant path and is the one used for the B board.
3. Everything downstream of the embeddings is framework-neutral NumPy, SciPy and LightGBM. This is a
   deliberate design property, not an omission: the frozen structural rules are exact integer and
   comparison logic, where a JIT-compiled tensor framework adds risk without adding capability.
   `main.py --stage describe` names the implementation of every stage so the boundary is auditable.
4. `main.py --framework jittor` is the default for the train stage.

**CLARIFICATION REQUIRED.** Whether a reviewer requires the *accepted A-board artifacts* to have
been produced under Jittor, or requires the *submitted code* to implement the models in Jittor, is
not settled by the wording available. The conservative reading is the former; this project can
satisfy the latter today and cannot retroactively satisfy the former without retraining, which
would change the accepted member hashes. This is registered as a clarification question.

## 10. Reproduction status carried into the B board

**COMMITMENT.** Stated exactly, with no rounding up:

| claim | status |
|---|---|
| dataset1 member from the hash-pinned ranker CSV | **BYTE_EXACT**, verified on Ubuntu 22.04 with Python 3.10 |
| dataset2 member from the hash-pinned base matrix | **BYTE_EXACT**, verified on Ubuntu 22.04 with Python 3.10 |
| dataset2 base matrix from raw data | **NOT REPRODUCED**; needs cached features and a multi-hour rebuild (ambiguity A1) |
| either member from raw data through a Jittor retrain | **NOT REPRODUCIBLE BYTE-EXACTLY** by construction, see section 9 |

The project does not claim the full path from raw competition data to the accepted members is
reproducible. Closing that gap is B-board preparation work, and the honest statement is recorded here
so no later document can quietly upgrade it.

## 11. Audit hooks

Any reviewer, and any future maintainer, can check this contract against the code:

```bash
python main.py --stage describe                    # every stage, implementation and status
python main.py --stage preprocess                  # measured graph type and causal boundary
python src/build_ds1_member.py --verify            # dataset1 byte-exact reproduction
python src/build_ds2_member.py --verify            # dataset2 byte-exact reproduction
python -m unittest discover -s tests -t .          # 97 tests
```

The frozen-rule tests assert that no prohibited knob exists, which is how section 5 is enforced
mechanically rather than by convention.

## 12. Open clarification items carried by this contract

| ID | Question | This contract's position pending an answer |
|---|---|---|
| `OCI-001` | May model architecture differ across scenarios when all alternatives are present in the reviewed A-board code? | Assume no. One family, one code path |
| `OCI-002` | Are externally pretrained weights permitted under the external-data prohibition? | Assume no. None are used on either board |
| `OCI-003` | Are B-scale partitioning, sampling, reduced dimensions and changed schedules parameter changes? | Assume yes, but only for knobs already present in the reviewed code |
| `OCI-004` | What numerical tolerance is accepted when reproducing the best A-board submission? | Target byte identity where deterministic; disclose the training-stage variance of section 9 |
| `OCI-005` | May hash-pinned trained weights accompany the source rather than be regenerated? | Provide a complete from-scratch training path; treat any checkpoint as an optional accompaniment |

The narrow wording sent to the organisers is drafted privately and is not part of the public
repository. No answer may be assumed; each must be registered under `SRC-007` when received.
