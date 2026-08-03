# Official package inventory

`tools/submission/stage_package.py` is the executable source of truth for
package membership. The tool stages only tracked bytes, records the current Git
commit and SHA-256 of each member, and audits the resulting tree.

## Code members

The organiser-facing `code/` directory contains:

| Group | Members |
|---|---|
| entry points | `run_all.py`, `main.py` |
| orchestration | `src/canonical_pipeline.py`, `src/stage_contract.py`, `src/pipeline_common.py` |
| Jittor training | `src/train_line_jt.py`, `src/train_bpr_jt.py` |
| ranking and features | `src/ensemble_predict.py`, `src/ranker_ds1.py`, `src/ranker_basket_ds2.py`, `src/ds2_basket_featurizer.py`, `src/ds2_mf_basket_pack.py` |
| post-processing | `src/crf_promote.py`, `src/build_ds1_member.py`, `src/build_ds2_member.py` |
| strategy package | `src/strategies/__init__.py`, `src/strategies/registry.py`, `src/strategies/shared/__init__.py`, `src/strategies/shared/frozen_ops.py`, `src/strategies/ds1/__init__.py`, `src/strategies/ds1/source_slate_recurrence.py`, `src/strategies/ds1/graph_reciprocity.py`, `src/strategies/ds2/__init__.py`, `src/strategies/ds2/cross_time_exclusivity.py` |
| configuration and verification | `configs/production.json`, `tools/submission/package_component.py` |

`requirements.txt` and `environment.yaml` are copied to the package root.
The approved final PDF is supplied explicitly and staged under the required
name `提交说明文档.pdf`.

## Exclusions

- `reference/pytorch/` and `tools/diagnostics/compare_backends.py`: optional
  PyTorch reference backend and dual-backend diagnostic.
- `tests/`: maintained repository tests, not runtime package content.
- `docs/` and `README.md`: public repository documentation; the approved PDF is
  supplied separately.
- `data/`: official competition data, not redistributed.
- `outputs/`: run artifacts, completion records, and generated members.
- `.git/` and the staging directory itself.

No canonical runtime dependency may exist only in an ignored directory, an
auxiliary worktree, or a host-specific absolute path. Integration tests verify
that every canonical member exists, is tracked, remains inside the repository,
and does not import PyTorch.

## Build and audit

The team name and final PDF are required. The tool refuses placeholders and an
incorrect staging-directory name.

```bash
python tools/submission/stage_package.py \
  --team-name APPROVED_TEAM_NAME \
  --document /path/to/final-submission-document.pdf

python tools/submission/stage_package.py \
  --audit \
  --root submission_staging/contest1_APPROVED_TEAM_NAME_003
```

The build writes `STAGING_MANIFEST.json` inside the staged package and
`STAGING_REPORT.json` beside it. It does not create the final ZIP.

## Audit gates

- staged bytes equal their declared sources;
- no undeclared file is present;
- the Python import closure contains no PyTorch dependency;
- every imported third-party dependency is version-pinned or explicitly
  justified;
- no credential, host path, private key, model artifact, data file, prediction
  member, cache, log, or nested archive is present;
- no staged file exceeds the package's explicit size limit.

Any failed gate blocks handoff.
