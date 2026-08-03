# Jittor Link Prediction

A reproducible Jittor implementation for Track 1, Dynamic Recommendation Based
on Graph Learning, of the Sixth Jittor Artificial Intelligence Challenge.

第六届计图人工智能挑战赛赛题一“基于图学习的动态推荐”的计图复现与提交工程。

## Project status

| Item | Value |
|---|---|
| Canonical framework | Jittor |
| Accepted A-board score | `1.576996059163449` |
| Recorded final rank | 3 |
| Dataset 1 component | `0.8963013747474597` |
| Dataset 2 component | `0.6806946844159895` |
| Configuration source | `configs/production.json` |
| Supported Python version | 3.10 |

The rank is based on the retained final-board observation described in
`docs/leaderboard_final.md`; it is not presented as an independently fetched
official timestamped record.

## Framework boundary

The production and submission paths use Jittor only:

- `src/train_line_jt.py` implements LINE training;
- `src/train_bpr_jt.py` implements BPR-MF training;
- `src/canonical_pipeline.py` selects those implementations explicitly;
- the official package inventory excludes every PyTorch file.

Earlier PyTorch implementations are retained under `reference/pytorch/` for
research and provenance. They are opt-in, use a separate dependency file, and
write to suffixed output directories so they cannot overwrite canonical Jittor
artifacts. See `reference/pytorch/README.md`.

## Reproducibility statement

The project distinguishes four claims:

1. The accepted members and package are hash-pinned in
   `configs/production.json`.
2. The deterministic post-processing chain regenerates the accepted member
   bytes from its pinned upstream matrices.
3. The raw-data-to-final Jittor pipeline was executed end to end on the target
   host on 1 and 2 August 2026, with every stage validated.
4. A fresh Jittor neural-training run is not expected to reproduce the
   historically accepted PyTorch-derived embeddings byte for byte because the
   frameworks use different random streams and optimiser trajectories.

These statements establish executable and algorithmic reproducibility without
claiming cross-framework byte identity.

## Repository layout

```text
.
├── configs/                 Production configuration and accepted hashes
├── data/                    Local official data; only the policy is tracked
├── docs/                    Public technical and competition documentation
├── outputs/                 Local generated artifacts; only the policy is tracked
├── reference/pytorch/       Optional PyTorch reference backend
├── src/                     Canonical Jittor pipeline and strategies
├── tests/                   Unit and integration tests
├── tools/                   Diagnostics and submission tooling
├── environment.yaml         Conda environment for inspection and reproduction
├── main.py                  Single-dataset canonical entry point
├── requirements.txt         Pinned Jittor runtime dependencies
└── run_all.py               Ordered two-dataset orchestration entry point
```

Private experiment records, personal communications, large artifacts, and
pre-publication audit material are deliberately kept outside the public
repository.

## Environment

Create the pinned inspection environment:

```bash
conda env create -f environment.yaml
conda activate jittor-link-prediction-inspect
python -m pip install --no-build-isolation --no-deps \
  git+https://github.com/AlgRUC/JittorGeometric.git@ff7d8ffac7bf3d95cc1962e091c52dc5737492d4
```

The JittorGeometric revision is pinned for environment completeness. The
canonical runtime import closure does not currently depend on it.

## Data preparation

Place the official data under `data/data_A/` as documented in
`data/README.md`, or pass another root explicitly:

```bash
python main.py --dataset dataset1 --stage preprocess --data-root /path/to/data
python main.py --dataset dataset2 --stage preprocess --data-root /path/to/data
```

Input hashes are validated against `configs/production.json`.

## Inspect the pipeline

The following commands do not train models:

```bash
python main.py --dataset dataset1 --stage describe
python main.py --dataset dataset2 --stage describe
python run_all.py --plan --data-root /path/to/data --output-root /path/to/outputs
```

`describe` prints the frozen strategy chain. `plan` resolves the stage graph and
reports whether each stage is runnable or reusable.

## Run the canonical pipeline

Run both datasets in the required order:

```bash
python run_all.py --data-root /path/to/data --output-root /path/to/outputs
```

Or run one dataset explicitly:

```bash
python main.py --dataset dataset1 --data-root /path/to/data --output-root /path/to/outputs
python main.py --dataset dataset2 --data-root /path/to/data --output-root /path/to/outputs
```

Dataset 2 consumes the Dataset 1 member as pass-through input when constructing
the two-member container, so `run_all.py` always runs Dataset 1 first.

Validated resume is the default. A stage is reused only when its completion
record binds the current code revision, configuration, inputs, command, and
output hash. Use `--fresh` to reject all reuse.

Expected members:

```text
<output-root>/members/dataset1.csv
<output-root>/members/dataset2.csv
```

## Verify accepted artifacts

The accepted members have distinct line-ending conventions and must be treated
as opaque bytes. Do not re-serialise or normalise them.

```bash
python src/build_ds1_member.py --verify
python src/build_ds2_member.py --verify
python tools/submission/package_component.py verify \
  --zip outputs/submissions/round30_final/r30_xte_final.zip
```

The accepted package has SHA-256
`efe790a56a715ec451a809a560f25ac6d3ecb0cde1d71a46e4de7fcdb6a4a536`.

## Build an organiser handoff package

The staging tool requires an approved team name and the final PDF. It does not
silently substitute a placeholder or a draft document.

```bash
python tools/submission/stage_package.py \
  --team-name APPROVED_TEAM_NAME \
  --document /path/to/final-submission-document.pdf
python tools/submission/stage_package.py \
  --audit \
  --root submission_staging/contest1_APPROVED_TEAM_NAME_003
```

The official inventory is maintained in
`docs/competition/official_package_inventory.md`.

## Tests and static checks

```bash
python -m pytest
ruff check --no-cache .
ruff format --check --no-cache .
```

The test count is intentionally not duplicated in documentation; the command
output is authoritative.

## Artifact policy

`data/` and `outputs/` are excluded from Git except for their policy files.
Generated packages, caches, rendered documents, local credentials, and editor
metadata are also excluded. Unknown work directories are not globally ignored:
new material must remain visible until it is classified, archived, or removed.

Before deleting an expensive or historically significant artifact, verify an
independent SHA-256-identical archive copy. See `outputs/README.md` for the
retention rules.

## Documentation

- `docs/CURRENT_PRODUCTION.md`: accepted state and exact strategy chain.
- `docs/architecture/pipeline-overview.md`: pipeline design and evaluation.
- `docs/competition/`: official-rule dossier, sources, and package inventory.
- `docs/SUBMISSION_PROTOCOL.md`: archive construction and validation.
- `docs/submission/submission_document_en.md`: editable English submission source.
- `docs/data-b-runbook.md`: controlled procedure for a later Dataset B release.

Historical experimentation is preserved in Git history and private verified
archives, not duplicated in the current public documentation tree.
