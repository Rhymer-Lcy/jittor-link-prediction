# Competition data

The competition datasets are not redistributed in this repository. Place the
official files under the following structure before running the pipeline:

```text
data/
└── data_A/
    ├── dataset1/
    │   ├── train.csv
    │   └── test.csv
    └── dataset2/
        ├── train.csv
        └── test.csv
```

The expected SHA-256 values are recorded under `immutable_inputs` in
`configs/production.json`. The pipeline validates inputs before computation.

Use `--data-root` to read data from another location. `DATA_ROOT` is supported
by lower-level scripts where documented. Source code must not contain a
machine-specific absolute path.

The original distribution archive is intentionally excluded. Retain it in a
separately backed-up private archive if required for provenance.
