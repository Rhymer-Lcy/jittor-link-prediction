# Dataset B runbook

This procedure applies when an official replacement data package is released.
It does not assume that Dataset A invariants transfer unchanged.

## 1. Preserve the accepted state

Create a clean branch or clone. Keep the accepted Dataset A artifacts and their
hashes immutable. Use a separate data pack name and output root so a Dataset B
run cannot overwrite them.

```text
data/data_B/dataset1/{train.csv,test.csv}
data/data_B/dataset2/{train.csv,test.csv}
```

Record the source, receipt date, byte size, and SHA-256 of every official input.
Update the production configuration only after those values have been reviewed.

## 2. Audit the new data

Before training, verify:

- schema, row counts, dtypes, nulls, and identifier ranges;
- train/test timestamp ranges and split semantics;
- source-destination repeat rates across time;
- same-event and adjacent-row candidate overlap;
- whether row order carries a documented structural meaning;
- whether every Dataset A structural invariant still holds.

Treat Dataset A measurements as hypotheses. Disable a conditional strategy when
its defining invariant is not re-established on Dataset B.

Never sort or reorder the test file before completing the structural audit.

## 3. Inspect the resolved graph

```bash
python main.py --dataset dataset1 --stage preprocess \
  --data-root /path/to/data --data-pack data_B
python main.py --dataset dataset2 --stage preprocess \
  --data-root /path/to/data --data-pack data_B
python run_all.py --plan \
  --data-root /path/to/data \
  --data-pack data_B \
  --output-root /path/to/outputs-data-b
```

The plan must contain only Jittor training modules on the canonical path.
PyTorch reference runs, if scientifically useful, must use their separate
environment and suffixed output directories; they are comparisons, not a
fallback production path.

## 4. Execute the canonical Jittor pipeline

After the configuration and invariant review:

```bash
python run_all.py \
  --data-root /path/to/data \
  --data-pack data_B \
  --output-root /path/to/outputs-data-b
```

Use validated resume for interrupted work. Use `--fresh` only when every stage
must be recomputed; it rejects reuse but does not delete prior artifacts.

## 5. Validate outputs

For each member, verify:

- the required row count and exactly 100 candidate columns;
- finite numeric values and non-degenerate rows;
- the prescribed serialization and line-ending convention;
- a SHA-256 lineage record from the producing stage;
- the completion record against the current code, inputs, configuration,
  command, and output bytes.

For a one-component change, prove that the other accepted member is byte-
identical to the intended control. Report changed-row and top-1-change counts.

## 6. Package and review

Build candidate packages with `tools/submission/package_component.py`. Run the
repository tests and static checks before any handoff. Record online results as
observations; do not infer a component score from an unverified combined upload.

Promote a Dataset B result to production only through a reviewed change that
updates configuration, documentation, tests, hashes, and the release tag in one
consistent state.
