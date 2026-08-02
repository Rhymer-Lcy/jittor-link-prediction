# Submission package — draft audit

> **Editor's note, 2026-08-02.** This is a dated record of the staging state as it stood when the
> audit ran, and its measurements are left verbatim rather than restated. Two names have since
> changed, and every occurrence below should be read through this map:
>
> | Name used below | Current name |
> |---|---|
> | `docs/submission/submission_document_draft_en.{md,docx,pdf}` | `docs/submission/submission_document_en.{md,docx,pdf}` |
> | `stage_package.DRAFT_DOCUMENT` | `stage_package.REVIEW_DOCUMENT` |
>
> The page count, character count and placeholder count recorded in section 6 describe the earlier
> rendering and were not updated. The current renderings are described in
> `docs/submission/submission_document_en.md` and in the consolidated review package.

**This audits a DRAFT.** No archive has been built, nothing has been uploaded, and no file carries
the official name `提交说明文档.pdf`. Final package readiness is **NOT** claimed anywhere in this
document.

Regenerate everything below with:

```bash
python tools/submission/stage_package.py          # build and audit
python tools/submission/stage_package.py --audit  # audit an existing tree
```

---

## 1. Staging tree

| Property | Value |
|---|---|
| Staging root | `submission_staging/contest1_TEAM_NAME_PLACEHOLDER_003/` (git-ignored) |
| Intended archive name | `contest1_<TEAM_NAME>_003.zip` |
| Source commit | `c6322346c09641430aaaea2fa4fd7ecdb9257f9b` |
| Files | 28 (25 under `code/`) |
| Total size | 466,062 bytes |
| Final archive built | **no** |
| Draft validation snapshot | built, extracted and discarded twice (remote 128,424 B; local 249,166 B — the local snapshot also carries the rendered PDF) |

```
contest1_<TEAM_NAME>_003/
+-- code/
+-- requirements.txt
+-- environment.yaml
\-- submission_document_draft_en.pdf     <- draft name, NOT the official filename
```

**Deviation: the directory placeholder.** The brief specifies the literal token `<TEAM_NAME>` in the
staging path. Windows rejects `<` and `>` in a path component, so the directory uses
`TEAM_NAME_PLACEHOLDER` instead. It is a placeholder in exactly the same sense — it must be replaced
before freeze — and the intended archive name is recorded verbatim in `STAGING_REPORT.json` and in
this document. No team name was invented.

## 2. Contents of `code/`

Twenty-five files, all tracked at the source commit and all byte-identical to their tracked source
(verified by the fidelity audit). Every file is on the canonical path.

| Group | Files |
|---|---|
| Entrypoint | `main.py` |
| Orchestration | `src/canonical_pipeline.py`, `src/stage_contract.py` |
| Shared | `src/pipeline_common.py`, `src/ensemble_predict.py` |
| Training (Jittor) | `src/train_line_jt.py`, `src/train_bpr_jt.py` |
| Ranking | `src/ranker_ds1.py`, `src/ranker_basket_ds2.py`, `src/ds2_mf_basket_pack.py` |
| Features | `src/ds2_basket_featurizer.py` |
| Post-processing | `src/crf_promote.py`, `src/strategies/registry.py`, `src/strategies/shared/frozen_ops.py`, `src/strategies/ds1/source_slate_recurrence.py`, `src/strategies/ds1/test_graph_reciprocity.py`, `src/strategies/ds2/cross_time_exclusivity.py` |
| Final output | `src/build_ds1_member.py`, `src/build_ds2_member.py` |
| Configuration | `configs/production.json` |
| Verification | `tools/submission/package_component.py` |
| Package markers | four `__init__.py` |

## 3. Exclusions

### 3.1 PyTorch

| File | Reason |
|---|---|
| `src/train_line.py` | historical PyTorch trainer; not on the canonical path |
| `src/train_bpr.py` | historical PyTorch trainer; not on the canonical path |
| `tools/diagnostics/compare_backends.py` | local dual-backend diagnostic; spawns the two above |

The exclusion list is checked against a **live import scan** of the repository, not trusted: a new
file that starts importing torch fails the audit until the list is updated deliberately. The
diagnostic driver is named explicitly because it reaches torch by spawning rather than importing, so
the scan alone would not catch it.

### 3.2 Everything else

| Path | Reason |
|---|---|
| `tests/` | the maintained suite; not required to reproduce a result |
| `docs/`, `README.md` | internal engineering records and the score ledger |
| `data/` | competition data; not redistributable, never bundled |
| `outputs/` | run artifacts, completion records, intermediate and final members |
| `reference/` | reference predictions; never a computational input |
| `artifacts_durable/`, `audit_exports/` | audit evidence |
| `scratchpad/`, `docs_local/` | working notes, one-off drivers, consultation briefs |
| `src/ranker_basket_ab_ds2.py`, `src/footprint_*.py`, `src/triple_promote.py`, `src/validate_footprint_packs.py` | out-of-chain probes and superseded channels; not in `configs/production.json` |
| `.git/`, git bundles, `submission_staging/` | version control and this tree itself |

## 4. Audits

| Audit | Verdict | What it checks |
|---|---|---|
| Fidelity | **PASS** | every staged byte equals its tracked source; no undeclared file in the tree |
| Import closure | **PASS** | static walk of the staged tree; reaches `jittor`, `lightgbm`, `numpy`, `pandas`, `scipy`, `tqdm` and nothing else |
| Hygiene | **PASS** | 0 findings across name patterns and content scanning |
| Environment | **PASS** | every imported package declared with an explicit version; nothing declared that is neither imported nor justified |

### 4.1 Hygiene scan

Scanned for private keys, public-key blobs, credential literals, GitHub tokens, the rented-instance
endpoint, `/root/autodl-tmp`, Windows user and drive paths, worktree paths, and PyTorch imports —
by filename pattern **and** by content. Forbidden name patterns cover `*.npz`, `*.npy`, `*.pt`,
`*.pth`, `*.ckpt`, `*.zip`, `*.bundle`, `*.part`, `*.log`, `*.pyc`, `*.done.json`, `id_rsa*`,
`*.pem`, `*.key`, `.env*`, `train.csv`, `test.csv`, `dataset1.csv`, `dataset2.csv`,
`result_ranker.csv`. Any file over 2 MB fails.

**Zero blocking findings. Nine advisories**, all of the same kind: provenance comments in production
source and in `configs/production.json` that mention a `scratchpad/` path an artifact was originally
ported from. They are historical notes, not runtime dependencies. Editing production source to
remove them would make the package diverge from the code that was actually executed, which is worse
for a reviewer than an unresolved comment reference, so they are recorded rather than removed.

Two scanner defects were found and fixed while building this: an unanchored `from torch` pattern
matched the prose *"differs from torch's"* in a docstring, and package **directories** were not in
the local-module set, so `import strategies` was misread as an undeclared third-party dependency.
Both would have made the scan's negatives worthless.

### 4.2 Environment

`requirements.txt` and `environment.yaml` pin identical versions:
`numpy==1.26.4`, `pandas==2.2.3`, `scipy==1.15.2`, `scikit-learn==1.6.1`, `lightgbm==4.5.0`,
`tqdm==4.67.1`, `jittor==1.3.10`, Python 3.10. Torch is declared in neither.

The staged code imports `jittor`, `lightgbm`, `numpy`, `pandas`, `scipy`, `tqdm`.

**`scikit-learn` is declared without being imported.** This is deliberate and recorded: the official
inspection notice names it explicitly as a version-pinned requirement of the target environment. The
canonical chain uses LightGBM for its learned components and does not import sklearn.

**JittorGeometric** is not a `requirements.txt` line because it is not on PyPI. It is pinned by
upstream commit `ff7d8ffac7bf3d95cc1962e091c52dc5737492d4` in the comment header of both environment
files, with the install command. Upstream publishes no release tags, so a commit hash is the only
sound pin. It is **not imported** by the canonical chain, and the document says so.

All versions were verified installed on the target host during Phase A.

## 5. Official requirement matrix

| Requirement | Verdict | Evidence |
|---|---|---|
| Archive named `contest1_<TEAM_NAME>_003.zip` | PARTIALLY_SATISFIED | name fixed and recorded; team placeholder open; archive not built |
| Root holds `code/`, environment file, PDF | PARTIALLY_SATISFIED | structure built; the PDF carries its **draft** name |
| Code covers data processing | SATISFIED | `pipeline_common`, `ds2_basket_featurizer`, `ranker_*` feature construction |
| Code covers model training | SATISFIED | `train_line_jt.py`, `train_bpr_jt.py`; both executed for real in Phase A |
| Code covers inference and prediction | SATISFIED | `ranker_ds1.py`, `ranker_basket_ds2.py`, `ds2_mf_basket_pack.py` |
| Code covers core algorithm logic | SATISFIED | `crf_promote.py`, `strategies/` |
| Code covers final prediction generation | SATISFIED | `build_ds1_member.py`, `build_ds2_member.py` |
| Starts from official raw data alone | SATISFIED | no stage reads a reference prediction or a frozen member; Phase-A case M proves the dataset-2 passthrough refuses an unrecorded member |
| No test-label ground truth | SATISFIED | ranker trained on a cut-split replay of training data; test labels are unavailable to the code |
| No frozen prediction as a shortcut | SATISFIED | verified by the staging exclusions and by case M |
| Ubuntu 22.04 | SATISFIED | validated on the target host |
| RTX 4090 / CUDA 12.4 | SATISFIED | sm_89, nvcc 12.4.131, validated |
| Python 3.10 | SATISFIED | 3.10.20 validated |
| Jittor >= 1.3.10 | SATISFIED | 1.3.10.0 validated; real training executed |
| JittorGeometric present | SATISFIED | 2.0.0 at the pinned commit, installed and importable |
| NumPy / Pandas / scikit-learn pinned | SATISFIED | explicit versions in both files |
| Other dependencies with explicit versions | SATISFIED | environment audit PASS |
| No Torch dependency | SATISFIED | absent from the host, from both environment files, and from the import closure |
| Complete reproduction commands | SATISFIED | document sections 7.1–7.3 |
| Documented input and output paths | SATISFIED | document sections 6 and 7.3 |
| Documented hyperparameters | SATISFIED | document section 8, read from source and `configs/production.json` |
| Known-issues disclosure | SATISFIED | document section 12 |
| Explanation of final-format generation | SATISFIED | document section 9 |
| A/B algorithm consistency | SATISFIED | document section 13 |
| Package size reasonable | SATISFIED | 461,580 bytes |
| No credentials or private material | SATISFIED | secret scan PASS |
| Archive extraction integrity | SATISFIED | deterministic snapshot extracted into an empty directory on the target host and locally: exit 0, no path traversal, no corrupt entry, **0 manifest hash mismatches**, root layout correct |
| Package independence | SATISFIED | all 10 canonical modules resolve inside the extracted copy; no repository, worktree or scratchpad path; torch unimportable |
| Final PDF ready | NOT_SATISFIED | seven placeholders open; owner review not done |
| Real-GPU validation | SATISFIED_FOR_SCOPE | **Phase A PASS** (real Jittor stages, contract mechanics); **targeted release-candidate validation PASS** (real LightGBM, MF geometry, feature cache, CRF, both member builders, both output gates, from the extracted package). A second full dual-dataset run was **not** performed, by owner decision |

## 6. The reviewer document

| Artifact | Path | State |
|---|---|---|
| Editable source | `docs/submission/submission_document_draft_en.md` | tracked |
| DOCX | `docs/submission/submission_document_draft_en.docx` | generated, git-ignored |
| PDF | `docs/submission/submission_document_draft_en.pdf` | generated, 18 pages, git-ignored |

The PDF was verified after rendering: 18 pages, not encrypted, 32,035 extractable characters, all
seven placeholders present, and no occurrence of `autodl-tmp`, `seetacloud`, a Windows drive path,
`scratchpad`, the local username, or any invented rank or score.

**Placeholders requiring owner input** — none may be inferred from project history, and none was:

`<TEAM_NAME>`, `<A_BOARD_RANK>`, `<A_BOARD_BEST_TOTAL_SCORE>`, `<CONTACT_NAME>`, `<WECHAT_ID>`,
`<PHONE_NUMBER>`, `<SUBMISSION_DATE>`.

## 7. The output-root defect — CLOSED

`src/ds2_basket_featurizer.py:413-416` and `src/ds2_mf_basket_pack.py:183-187` resolved the
Dataset-2 embedding directories from a literal `<project>/outputs/...`, so `OUTPUTS_ROOT` never
reached them and `--output-root` was honoured by the driver but ignored by those consumers. Fixed in
`b4776a2` by calling `tl.bpr_run_dir` / `tl.line_run_dir`, plus removal of one dead
`tdir = tl.PROJECT_ROOT / "outputs"` assignment in `ranker_basket_ds2.py` whose two call arguments
already used the helpers.

Classification: **`PATH_AND_ORCHESTRATION_ONLY`**. No filename, scientific configuration,
environment variable, stage ID or output format changed. Under the default output root the new
expressions resolve byte-for-byte to the former literals for all five required cases plus every
seed, so no existing artifact is orphaned. Both frozen replay tests still reproduce the accepted
members byte for byte.

`tests/strategies/test_ds2_output_root_contract.py` adds 14 tests. They evaluate the **actual source
expressions**, extracted from the AST rather than retyped, so they cannot pass against a copy of the
fix while the shipped code still holds a literal. Against the pre-fix tree at `6d2ff15`, **10 of the
14 fail** — including the stale-default-tree trap, the isolated-root routing, the real
`build_or_load_features` subprocess check and both static guards.

Full analysis: `artifacts_durable/rc_validation_20260802/RC_VALIDATION_AUDIT.md` section 8.

## 8. Blockers to a final package

1. **Owner input** for the seven placeholders.
2. **Owner review** of the document content.
3. **Rename** the reviewed PDF to `提交说明文档.pdf`.
4. **Build and verify the final archive**: deterministic ZIP, extraction dry run, SHA256 freeze.

Closed since the previous audit:

* **A second full dual-dataset raw-to-final run** — replaced, by owner decision, with the targeted
  release-candidate validation recorded above.
* **The `ENVIRONMENT_PACKAGES` negative-dependency extension** —
  `DEFERRED_AS_NONESSENTIAL` by owner decision; `stage_contract.py` was not modified. Torch absence
  is already evidenced by the environment audit, the import closure, the runtime import guard, the
  zero Torch files in `code/`, and both environment specifications.
