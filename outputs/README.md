# Generated artifacts

This directory contains local run artifacts and is excluded from Git except
for this policy file. Its contents are not source code and are not required to
inspect the repository.

The canonical entry points accept `--output-root`; lower-level scripts use the
resolved output root supplied by the pipeline. Source code must not embed a
machine-specific absolute path.

## Artifact classes

- `members/`: generated submission members and completion records.
- `_logs/`: stage logs.
- `_quarantine/`: artifacts rejected by the completion contract.
- model and feature directories: potentially expensive but reproducible run
  outputs.
- `submissions/`: byte-sensitive packages and their verification metadata.

`configs/production.json` is the source of truth for accepted hashes, member
shapes, strategy order, and named production artifacts. A path under this
directory must not be renamed or deleted solely for cosmetic consistency.

## Retention policy

Before removing an artifact:

1. prove that no active configuration, source file, test, or current document
   reads it;
2. classify it as reproducible, superseded, or historically significant;
3. retain and verify a SHA-256-identical archive copy when the artifact is not
   cheaply reproducible or supports a published result;
4. record the decision outside this ignored directory.

Caches and failed runs may be removed after the same dependency check. Never
normalise, re-serialise, or edit an accepted CSV or ZIP in place.

For a clean run that leaves the repository tree unchanged, provide an output
root outside the repository:

```bash
python run_all.py --data-root /path/to/data --output-root /path/to/outputs
```
