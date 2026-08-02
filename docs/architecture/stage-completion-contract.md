# Submission engineering, phase 1 — the stage-completion contract and the canonical entrypoint

Design and implementation record. Two defects are closed here:

1. **A stage was considered done when its output file existed.** During the dataset-1 clean
   reexecution all sixteen artifacts were present and the maintained driver would have reported a
   complete success in a few seconds. They had to be moved aside by hand. The same driver was
   untracked scratch (see [Deviation 1](#deviations-from-the-brief)).
2. **`main.py` printed commands rather than running them**, and offered `--framework torch`, putting
   a PyTorch execution path inside the organiser-facing interface of a Jittor-mandated competition.

Nothing scientific changed. No model, feature, hyperparameter, seed, ranking rule, normalisation,
CRF rule or decode rule was touched, and no score transformation was added. What changed is *when*
a stage runs, *how* its artifact is published, and *what* is recorded about it.

---

## 1. The maintained stage graph

Measured from `src/canonical_pipeline.py`, which is now the single description of it. Diagnostic and
historical PyTorch stages are excluded by construction: they are not in the graph.

### dataset1 — 16 stages

| # | stage id | command | output | units | validation |
|---|---|---|---|---|---|
| 1 | `ds1_line_full` | `DATASET=dataset1 python src/train_line_jt.py` | `outputs/dataset1-novirt/line_latest_emb.csv` | 400 epochs | header, width, row count |
| 2 | `ds1_line_cut` | + `LINE_TIME_MAX=115480000` | `outputs/dataset1-novirt-tmax1.1548e+08/line_latest_emb.csv` | 400 epochs | as above |
| 3–12 | `ds1_bpr_{serve,cut}_s{42,123,777,2024,31337}` | `DATASET=dataset1 SEED=s BPR_TAU_FRAC=0.25 [BPR_TIME_MAX] python src/train_bpr_jt.py` | `outputs/dataset1-bpr-t0.25[-tmax…][-sN]/bpr_emb.npy` | 120 epochs | shape, dtype, finiteness |
| 13–14 | `ds1_bpr_innov_{serve,cut}` | + `BPR_INNOV=1` | `outputs/dataset1-bpr-innov-t0.25[-tmax…]/bpr_emb.npy` | 120 epochs | as above |
| 15 | `ds1_ranker` | `DATASET=dataset1 python src/ranker_ds1.py` | `outputs/dataset1-ensemble/result_ranker.csv` | — | shape, finiteness, `[0, 1]` |
| 16 | `ds1_member` | `python src/build_ds1_member.py --control … --output …` | `outputs/members/dataset1.csv` | — | score gate + serialised row/field shape |

### dataset2 — 17 stages

| # | stage id | command | output | units | validation |
|---|---|---|---|---|---|
| 1–2 | `ds2_line_{full,cut}` | `DATASET=dataset2 [LINE_TIME_MAX=1261958400] python src/train_line_jt.py` | `outputs/dataset2-novirt[-tmax1.26196e+09]/line_latest_emb.csv` | 400 epochs | header, width, row count |
| 3–12 | `ds2_bpr_{serve,cut}_s{…}` | `DATASET=dataset2 SEED=s [BPR_TIME_MAX] python src/train_bpr_jt.py` | `outputs/dataset2-bpr[-tmax…][-sN]/bpr_emb.npy` | 120 epochs | shape, dtype, finiteness |
| 13 | `ds2_ranker` | `DATASET=dataset2 python src/ranker_basket_ds2.py` | `outputs/dataset2-ranker/ranker_basket3_dataset2.csv` | — | shape, finiteness, `[0, 1]` |
| 14 | `ds2_mf_pack` | `python src/ds2_mf_basket_pack.py --geom MF --out …` | `outputs/dataset2-ranker/mf_basket3_dataset2.csv` | — | as above |
| 15 | `ds2_ds1_passthrough` | in-process | `outputs/dataset2-crf/ds1_passthrough.zip` | — | exactly one member, non-empty |
| 16 | `ds2_crf` | `python src/crf_promote.py --base … --ds1-from … --out … --tau 0.20 --B 70 --W 1 --zr-exclude --demote-dups --st-exclude --no-pair` | `outputs/dataset2-crf/ds2_crf_intermediate.zip` | — | both members, `dataset2.csv` shape/range/variance |
| 17 | `ds2_member` | `python src/build_ds2_member.py --base-zip … --base-member dataset2.csv --output …` | `outputs/members/dataset2.csv` | — | score gate + serialised shape |

`ds2_mf_pack` internally builds or reuses `src/bagging_cache/train_features_all.npz`, which is a
stage of the contract in its own right (§5) rather than a driver stage.

Every path above is produced by a `pipeline_common` helper — `line_run_dir`, `bpr_run_dir`,
`ranker_dir` — so a producer and a consumer cannot disagree, and every one matches what the
historical driver used.

### Restart hazards the graph carries

| property | stages | note |
|---|---|---|
| internal checkpoints | `ds1/ds2_line_*` | `export_emb` publishes every 10 epochs |
| an interrupted output can look complete | `ds1/ds2_line_*` | a run killed at epoch 250 leaves a well-formed 400-epoch-shaped CSV; caught by the epoch count |
| previously skipped on file existence | all | the defect closed here |
| non-atomic publication (before) | `ds1_ranker`, `ds2_ranker`, `ds2_mf_pack`, the NPZ cache | now atomic |
| already atomic (before) | `*_line_*`, `*_bpr_*`, `ds1_member`, `ds2_member` | `os.replace` / `write_score_matrix` |

---

## 2. The completion record

`SCHEMA_VERSION = 1`. One record per stage, at `<output>.done.json`.

**Location policy.** The record is a sibling of its output, so the two travel together. Deleting an
output directory deletes its records, which fails safe: the next run sees "no output, no record" and
re-runs, rather than finding an orphan record that claims a vanished artifact.

**Bound fields.** `schema_version`, `stage_id`, `dataset`, `artifact_kind`, `description`,
`producing_commit`, `code_identity` (per-file SHA256 plus a digest), `command`, `command_env`,
`environment` (interpreter, platform, package versions read from installed metadata — nothing is
imported), `inputs` (path, size, SHA256 each), `config`, `config_digest`, `seed`, `unit_name`,
`requested_units`, `achieved_units`, `output` (path, size, SHA256, plus validator detail such as
shape and dtype), `validation` (status, validator name, detail), `exit_code`, `started_at`,
`completed_at`.

**Reuse policy — strict, and documented as such.** A stage is reusable only when *all* hold:

1. the output exists;
2. a record exists beside it;
3. it parses and declares a supported `schema_version`;
4. `exit_code` is 0;
5. every declared input still hashes to the recorded value, and the input *set* is unchanged;
6. `config_digest` matches;
7. `code_identity.digest` matches **and** `producing_commit` matches;
8. the output's size and SHA256 match;
9. `achieved_units == requested_units` where the stage has units;
10. `validation.status` is `PASS`.

Both the commit and the finer code digest are recorded and both are compared. There is no rule that
treats a commit difference as ignorable because "the change was only documentation": that judgement
is not mechanical, and this module does not make it. A commit change therefore invalidates reuse —
strict, and deliberately so.

Some configuration is only measurable *after* a stage runs (how many training queries the dataset-2
cache built, how wide its feature matrix is). Such keys are recorded, and replayed into the
validator on load, but are excluded from the reuse digest via `StageSpec.digest_keys` — a
before-the-fact comparison cannot use an after-the-fact measurement.

---

## 3. Atomic write and failure semantics

Publication order, in `stage_contract.complete_stage` and `canonical_pipeline.run_stage`:

1. the stage writes into an isolated `<output>.part`, wherever it takes its output as an argument;
2. the file is closed;
3. a non-zero exit is rejected before anything else;
4. the achieved unit count is read from the stage's own log and compared with the requested count;
5. stage-specific validation runs on the staged artifact;
6. the artifact is published by `os.replace` — atomic within a directory;
7. the record is serialised to `<output>.done.json.part`, flushed and `fsync`ed;
8. it is renamed into place. **This is the last step of a stage.**

Stages whose output path is fixed rather than an argument — the trainers and the two rankers —
publish their own artifact atomically and are validated one step later, in place. The record is
still written only after validation passes, so the guarantee "a record implies a validated
artifact" holds in both modes.

`publish_record` refuses outright to write a record whose `exit_code` is non-zero or whose
validation status is not `PASS`, so the invariant cannot be bypassed by a caller assembling a
document by hand.

### Behaviour, per condition

| condition | behaviour |
|---|---|
| no output, no record, no `.part` | `RUN` |
| stale `.part` present | `STOP` `STALE_PART_FILE` — a previous run died mid-write |
| output exists, record missing | `STOP` `MISSING_COMPLETION_RECORD` — the historical false success |
| record exists, output missing | `STOP` `MISSING_OUTPUT` |
| malformed JSON | `STOP` `MALFORMED_RECORD` |
| unsupported schema version | `STOP` `UNSUPPORTED_SCHEMA` |
| required field absent | `STOP` `INCOMPLETE_RECORD` |
| input hash, input size or input set differs | `STOP` `IDENTITY_MISMATCH` |
| configuration or seed differs | `STOP` `IDENTITY_MISMATCH` |
| commit or code digest differs | `STOP` `IDENTITY_MISMATCH` |
| output hash or size differs | `STOP` `IDENTITY_MISMATCH` |
| requested units differ, or achieved < requested | `STOP` `IDENTITY_MISMATCH` |
| record claims a non-zero exit or a failed validation | `STOP` `IDENTITY_MISMATCH` |
| interrupted epoch count at completion time | no record written; the stage fails |
| truncated NPZ | validation fails; no record written |

Every stop names the stage, the reason code and the differing properties. Nothing is deleted and
nothing is repaired: a `STOP` leaves the artifact exactly where it was, because it is the evidence.

**The escape hatch is `quarantine`, and it is the only one.** It moves an unusable output and its
record into `outputs/_quarantine/NNN/` with a `QUARANTINE.json` recording the reason, then the next
run starts clean. There is no `--skip-checks`, no `--force-reuse` and no option on `main.py` that
waves a stage through; a test asserts this against the declared command-line surface.

---

## 4. The dataset-2 feature cache

`src/bagging_cache/train_features_all.npz` (2,057,882,178 bytes, ~2.8 h to build) was written by a
bare `np.savez` straight to its final name, and reused whenever `feat.is_file()` was true, guarded
only by a single `num_entity` assertion. A cache from other inputs, other code or an interrupted
write was indistinguishable from a good one.

It is now a contract stage. **Feature values are unchanged** — the build path is untouched; only
publication and the reuse decision moved.

* Written through `atomic_output`: `np.savez` into `<name>.part`, closed, then `os.replace`.
* Reuse is bound to: the SHA256 of `train.csv` and `test.csv`, `CUT`, negatives per sample,
  `max_queries`, both sampling seeds, `num_entity`, the expected array names, the cache tag, the
  relevant-code digest, the producing commit, and the file's own size and SHA256.
* Internal validation on every write **and** every reuse: `np.load` succeeds; all eight arrays are
  present and are actually read; `lens` is `(queries,)`; `qsrc_tr`/`qt_tr`/`qorig_tr` match it;
  `Xf` and `yf` have `lens.sum()` rows; `cands_concat` has `lens.sum()` entries; `Xf` has the
  recorded feature width; `num_entity` matches; label positives land exactly at `offsets[1:] - 1`;
  `Xf` and `yf` are finite.
* `--force` quarantines the previous cache and its record rather than deleting them.

**Known cost.** The 2 GB cache retained on the powered-off instance has no completion record, so
under this contract it is not reusable and the dataset-2 run will rebuild it (~2.8 h). No adoption
path was added: writing a record for an artifact of unknown provenance is exactly the reuse-on-faith
this contract exists to remove. Flagged as an owner decision, not taken here.

---

## 5. The canonical entrypoint

```bash
python main.py --dataset dataset1
python main.py --dataset dataset2
```

`run` is the default stage. The other stages are `plan` (print the graph and each stage's current
disposition, execute nothing), `describe`, `preprocess`, `postprocess` and `package`. The former
`train` and `rank` stages are gone: they only printed commands, and `run` executes them.

Documented controls: `--data-root`, `--data-pack`, `--output-root`, `--log-dir`, `--config`,
`--fresh` (refuse to reuse anything), `--ds1-member`, `--verify`, `--quiet`.

`--data-root` and `--output-root` are honoured through `pipeline_common.data_root()` /
`outputs_root()`, which read `DATA_ROOT` / `OUTPUTS_ROOT`; the driver exports both to every child,
so a producer and a consumer are always on the same roots. `train_line_jt` and `train_bpr_jt` now
call `pipeline_common.data_dir()` instead of repeating the expression, closing a third copy of it.

**Jittor runtime.** `conv_opt=1` and `use_mkl=0` are applied by the driver, printed at the start of
every run and recorded in every completion record. They are not tuning: the target image ships
cuDNN 9, which merged the component libraries Jittor 1.3.10 looks for by name, and without them the
first convolution aborts with `libcudnn_ops_infer.so not found`. If the caller's environment already
sets either variable to a different value the run **stops** rather than overriding it — silently
winning over an operator's explicit setting is the hidden behaviour this entrypoint removes.

**The dataset-2 passthrough.** `crf_promote` writes a two-member archive and therefore needs
dataset-1 bytes even when only dataset-2 is being produced; the dataset-2 member builder then
extracts `dataset2.csv` alone, so those bytes are never read as scores. The driver packages them
from a dataset-1 member **this repository produced under a valid completion record**. If none
exists, the run stops and says so. No frozen accepted member is substituted and no placeholder score
row is fabricated.

---

## 6. Torch separation, and the Jittor-only closure

`main.py --framework torch` is removed. The historical trainers are not deleted — their provenance
value is real — but they are reachable only from `tools/diagnostics/compare_backends.py`, which
announces itself as out of the submission, writes to distinct directories, and refuses to run
without a deliberately installed CPU-wheel PyTorch.

An AST walk of the transitive import graph from `main.py` reaches only `jittor`, `lightgbm`,
`numpy`, `pandas`, `scipy` and `tqdm`. The walker is bracketed by an anti-vacuity test that points
it at a file which really does `import torch` and confirms it notices.

Proposed staging exclusions for torch, pinned by a live scan in
`tests/integration/test_official_package_inventory.py`:

* `src/train_line.py`
* `src/train_bpr.py`
* `tools/diagnostics/compare_backends.py` (reaches torch by spawning, not importing, so the scan
  alone would miss it; the list names it explicitly)

---

## 7. JittorGeometric

| property | value |
|---|---|
| upstream | `https://github.com/AlgRUC/JittorGeometric.git` |
| release tags published | none |
| pinned commit | `ff7d8ffac7bf3d95cc1962e091c52dc5737492d4` (2026-06-03) |
| `setup.py` / distribution metadata version | 2.0.0 |
| package `__version__` attribute | 0.0.1 (an upstream inconsistency; the commit is authoritative) |
| install | `pip install --no-build-isolation --no-deps git+…@ff7d8ff…` |
| imported by any canonical module | **no** |
| required indirectly | no |

It is pinned and available because the competition requires it in the environment. The maintained
production chain does not import it: link prediction here is a LINE / BPR-MF embedding pair plus
LambdaRank over hand-built features, and no stage of that needs a graph-neural-network library. No
import was added to manufacture usage. The accurate sentence for the submission document is:

> JittorGeometric is installed and commit-pinned in the official environment; the final maintained
> production chain does not directly import it.

A test compares the claim in `environment.yaml` against a live import scan, in both directions, so
the documentation cannot drift from the code.

---

## 8. Tests

138 new tests, all offline; none needs a GPU or the competition data.

* `tests/pipeline/test_stage_contract.py` (62) — every stop condition, epoch accounting, atomic
  publication, quarantine, digest determinism and the NPZ validator. Two anti-vacuity tests bracket
  it: `test_the_happy_path_actually_reuses` proves the checks can pass at all, so a blanket STOP
  would fail the suite, and `test_every_stop_reason_is_reachable` proves each stop code has its own
  constructible condition rather than one over-broad guard.
* `tests/pipeline/test_canonical_entrypoint.py` (48) — graph shape and ordering for both datasets,
  path-contract derivation, external data and output roots, "no reference artifact is ever an
  input", real subprocess dispatch with test doubles, failure propagation, validated resume,
  `--fresh` refusal, incompatible-output refusal, missing inputs, the Jittor runtime reaching the
  child, the conflicting-setting stop, output gates, and the torch-free import closure with its own
  anti-vacuity probe.
* `tests/pipeline/test_ds2_feature_cache.py` (16) — the featurizer wiring, the atomic score-matrix
  writers, the unchanged float format, and the two root overrides.
* `tests/integration/test_official_package_inventory.py` (12) — the canonical file list, no
  canonical dependency outside the repository, no hard-coded host path, the torch exclusion list
  against a live scan, and the JittorGeometric contract in both directions.

Full suite: **341 tests**, zero failures.

---

## Deviations from the brief

**1. `canonical_run.sh` was never a maintained driver.** The brief describes it as maintained and
names its skip-on-existence behaviour as defect 1. It is real, and it is worse than described: the
file exists only at
`scratchpad/pre-b-board-p0-opus/remote_environment_preparation/scripts/canonical_run.sh`, under a
gitignored directory, and is not one of the repository's 76 tracked files. It also stops after the
dataset-2 ranker, printing `remaining: ds2 MF geometry, CRF, decoder, member -- see driver notes`.
So there was no tracked driver to repair; `src/canonical_pipeline.py` is a replacement, and it
covers the four stages the script never reached. Effect: the §11 "no canonical dependency lives only
in scratchpad" check would have failed before this phase, and now passes.

**2. `main.py --framework torch` never executed anything.** The brief calls it "an explicit Torch
backend path". It selected `src/train_line.py` and `src/train_bpr.py` — and then printed them,
because the `train` stage ended in `NOT EXECUTED by main.py`. The ambiguity in the submission was
real (a torch option in a Jittor competition's interface) but no torch code was reachable at
runtime, so removing the flag changed no behaviour. Effect on readiness: none; the ambiguity is
closed either way.

**3. `crf_promote` needs dataset-1 bytes to produce dataset-2.** Not mentioned in the brief. Its CLI
writes a two-member archive and requires `--ds1-from`; the historical run passed a hand-made
`ds1_passthrough.zip`. Those bytes are pure passthrough — `build_ds2_member` extracts `dataset2.csv`
alone — but the driver still has to obtain them from somewhere. Resolved by requiring a dataset-1
member produced by this repository under a valid completion record, with `--ds1-member` as the
explicit override, and by failing closed otherwise. No frozen member is substituted and no score row
is fabricated. Effect: `--dataset dataset2` alone now requires dataset-1 to have completed;
`--dataset all` runs them in that order and is unaffected.

**4. `--output-root` could not have worked without a path-contract change.** The trainers call
`pipeline_common.line_run_dir(..., root=PROJECT_ROOT)`, so an explicit root argument always won and
a caller's `--output-root` would have been silently ignored by every producer while the driver
believed it applied. `outputs_root()` now reads `OUTPUTS_ROOT` in preference to its argument, and
`data_root()`/`data_dir()` are new siblings. Defaults are byte-identical; a test pins the precedence
in both directions. Effect: the required control is real rather than nominal.

**5. Commit-strict reuse is stricter than "fail closed" alone requires.** Any commit invalidates
every completion record, including a documentation-only one. The brief forbids an informal
"documentation-only" reuse rule, and the only mechanical alternative — reusing on the code digest
while ignoring the commit — would silently accept an artifact produced by a different tree. Both are
recorded and both are compared. Effect: a run should finish before the tree is committed, or accept
a rebuild. Listed under known limitations rather than resolved.

---

## 9. Known limitations

* **Not executed on a GPU.** The contract has never gated a real Jittor training stage. Dispatch is
  proven with test doubles; the graph is proven by construction and by path comparison against the
  historical driver. §10 is the plan that closes this.
* **The retained remote NPZ cache is not reusable** under the contract (§4).
* **Commit-strict reuse is coarse.** Any commit to the repository invalidates every completion
  record, including a documentation-only commit. This is deliberate — the alternative requires a
  mechanical judgement about which changes are inert, which is exactly the reasoning the P0 record
  had to make by hand for the dataset-2 cache. In practice it means finishing a run before
  committing, or accepting a rebuild.
* **`achieved_units` is read from stage stdout.** A trainer that changed its progress-line format
  would report `None` and fail closed, which is the safe direction, but it is a textual coupling.
* **The line-terminator finding is unaddressed**, deliberately: `ranker_ds1` writes the platform
  default (LF on Linux) where the frozen artifact is CRLF. Classified
  `PORTABILITY_BYTE_DIFFERENCE_WITHOUT_RANKING_OR_MEMBER_EFFECT`; not repaired in this phase.
* **`src/ranker_basket_ab_ds2.py` and `src/footprint_ab_probe.py`** still rebuild `outputs/dataset2`
  independently. Neither is in the canonical graph; the existing path-contract test fails if either
  ever enters it.

---

## 10. Remote-validation plan

Not started. The instance is powered off and was not contacted during this phase.

**Prerequisite.** Sync the remote to the merge commit of this work by the established
credential-free bundle workflow (`git bundle create` → `scp -P` → `git fetch <bundle>
main:refs/bundle/N` → `git merge --ff-only` → delete the ref and the file). The remote is at
`d5fc39a`, four commits behind before this work.

**Phase A — contract validation, ~1 h GPU, ~1 h wall.**

```bash
python main.py --dataset dataset1 --stage plan
```

Expected: every stage reports `STOP (MISSING_COMPLETION_RECORD)` against the artifacts of the
2026-08-02 clean run. That result *is* the first acceptance criterion — it demonstrates the gate
refusing artifacts it cannot vouch for, on real data, where the old driver reported success.

Then, on a quarantined copy:

1. run `ds1_bpr_serve_s42` to completion (~10 min GPU) and assert the record binds the commit, both
   raw-input hashes, seed 42, 120/120 epochs and the artifact hash;
2. re-run the same stage and assert `REUSE` with no recomputation;
3. interrupt a LINE stage at ~30 epochs and assert no record exists and the next run stops;
4. touch `train.csv`'s mtime only, and assert reuse still holds — the contract binds content, not
   timestamps;
5. rebuild the dataset-2 feature cache (~2.8 h) and assert its internal validation and record.

**Phase B — full raw-to-final, ~14–16 h GPU.** `python main.py --dataset dataset1` (~2 h, measured
1 h 51 m on 2026-08-02) then `python main.py --dataset dataset2` (~11–13 h: cache ~2.8 h, MF pack
~2.5 h, ranker ~2.5 h, trainers the rest).

GPU is required for the LINE and BPR stages only; the rankers, the MF pack, the CRF and both member
builders are CPU.

**Is Phase B required?** For the *contract*, no: Phase A exercises every mechanism. For the
*submission*, yes, once — the organiser-facing command has never produced a member end to end, and
"the entrypoint works" is a claim the package makes. Recommendation: Phase A first, review, then
Phase B as a separate authorised task.

**Acceptance criteria.** Every stage exits 0; every stage has a record that re-validates on a second
`--stage plan`; both members pass the output gate; `dataset1.csv` is a valid 61,051 × 100 member and
`dataset2.csv` a valid 153,420 × 100 member; no stage is skipped on file existence; no `.part`
survives. Byte-equality with the historical members is **not** an acceptance criterion and is not
expected — Jittor GPU training is not bit-reproducible and the historical model state was never
preserved.

**Shutdown criteria.** Report first; the owner decides shutdown from the web console. Never invoke
any shutdown binary.
