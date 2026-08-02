# Provisional inventory for the official submission package

**The package is not built here.** This is the inventory a later, explicitly authorised build will
draw from, recorded now so the decisions are reviewable independently of whoever runs the build.
`tests/integration/test_official_package_inventory.py` pins every list below and checks the torch
exclusions against a live import scan, so the file cannot quietly go stale.

## Included under `code/`

| file | role |
|---|---|
| `main.py` | the organiser-facing entrypoint |
| `src/canonical_pipeline.py` | the stage graph and the fail-closed runner |
| `src/stage_contract.py` | the completion contract |
| `src/pipeline_common.py` | path contract and framework-neutral utilities |
| `src/train_line_jt.py` | LINE embedding trainer (Jittor) |
| `src/train_bpr_jt.py` | BPR-MF embedding trainer (Jittor) |
| `src/ensemble_predict.py` | embedding loading, collaborative scoring |
| `src/ranker_ds1.py` | dataset-1 LambdaRank |
| `src/ranker_basket_ds2.py` | dataset-2 three-pass basket LambdaRank |
| `src/ds2_basket_featurizer.py` | dataset-2 feature construction and cache |
| `src/ds2_mf_basket_pack.py` | dataset-2 MF sibling-message geometry |
| `src/crf_promote.py` | dataset-2 equality CRF and invariant demotions |
| `src/build_ds1_member.py` | dataset-1 frozen postprocessor chain |
| `src/build_ds2_member.py` | dataset-2 cross-time exclusivity decode |
| `src/strategies/registry.py` | the ordered dataset-1 chain |
| `src/strategies/shared/frozen_ops.py` | frozen primitives and the submission gate |
| `src/strategies/ds1/source_slate_recurrence.py` | postprocessor 1 |
| `src/strategies/ds1/test_graph_reciprocity.py` | postprocessor 2 |
| `src/strategies/ds2/cross_time_exclusivity.py` | the final decoder |

Plus `environment.yaml` and `requirements.txt` at the package root.

## Excluded — PyTorch

| file | reason |
|---|---|
| `src/train_line.py` | historical PyTorch trainer; not canonical |
| `src/train_bpr.py` | historical PyTorch trainer; not canonical |
| `tools/diagnostics/compare_backends.py` | local dual-backend diagnostic; spawns the two above |

The canonical backend is Jittor only, torch is not a runtime dependency, and torch is not declared
in either environment file. Shipping a second backend in a Jittor-mandated competition would be an
ambiguity in the submission, not a feature. The files stay in the repository and in git history for
provenance.

## Excluded — everything else

| path | reason |
|---|---|
| `artifacts_durable/` | audit evidence and retained artifacts; not code |
| `audit_exports/` | deterministic evidence archives |
| `scratchpad/` | untracked working notes and one-off drivers |
| `docs_local/` | local consultation briefs |
| `data/` | competition data; not redistributable |
| `outputs/` | run artifacts, including every completion record |
| `reference/` | frozen predictions; never a computational input |
| `F:/jlp-p1-diag-wt`, `F:/jlp-p1-int-wt`, `F:/jlp-schemeC-wt` | git worktrees; no canonical file lives only there |
| `src/ranker_basket_ab_ds2.py`, `src/footprint_*.py`, `src/triple_promote.py`, `src/validate_footprint_packs.py` | out-of-chain probes and A/B tooling; not in `configs/production.json` |

## Runtime-dependency location check

No canonical runtime dependency exists only in an untracked worktree, in `scratchpad/`, in
`artifacts_durable/`, in `audit_exports/`, on the powered-off instance, or behind a local absolute
path. Every file in the first table is tracked by git and resolves inside the repository; a test
asserts both, and a third asserts that no canonical file hard-codes a host path.

**One blocker was found and closed by this phase.** `canonical_run.sh` — the driver the
submission-engineering brief described as "maintained" — was never tracked: it lived only at
`scratchpad/pre-b-board-p0-opus/remote_environment_preparation/scripts/canonical_run.sh`, under a
gitignored directory, and it stopped after the dataset-2 ranker with a comment naming four unbuilt
stages. It is superseded by `src/canonical_pipeline.py`, which is tracked, covers all 33 stages
across both datasets, and is the graph `main.py` executes.

## Organiser-facing commands

```bash
conda env create -f environment.yaml
conda activate jittor-link-prediction-inspect
pip install --no-build-isolation --no-deps \
  git+https://github.com/AlgRUC/JittorGeometric.git@ff7d8ffac7bf3d95cc1962e091c52dc5737492d4

python main.py --dataset dataset1 --stage plan     # what will run, and why
python main.py --dataset dataset1                  # raw data -> outputs/members/dataset1.csv
python main.py --dataset dataset2                  # raw data -> outputs/members/dataset2.csv
```

Expected outputs: `outputs/members/dataset1.csv` (61,051 x 100) and `outputs/members/dataset2.csv`
(153,420 x 100), each beside its `.done.json` completion record, with every intermediate artifact
and record under `outputs/` and every stage log under `outputs/_logs/`.

## Sections of the future submission document affected

* **Reproduction instructions** — now two commands rather than a hand-ordered stage list.
* **Environment** — the JittorGeometric commit pin, and the accurate statement that the chain does
  not import it.
* **Framework compliance** — Jittor-only, with the import closure as the evidence.
* **Restart and provenance** — a new section: what a `.done.json` binds and why a stage is not
  complete merely because its output exists.
* **Known deviations** — the historical members were produced under PyTorch on Windows; the Jittor
  reproduction is a legitimately different artifact, not a byte reproduction.
