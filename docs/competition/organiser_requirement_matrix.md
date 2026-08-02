# Organiser requirement traceability matrix

Every official requirement traced to a specific English-document section, staged file, command or
validation result. No requirement is left unclassified.

Verdicts: **SATISFIED** / **BLOCKED_BY_OWNER_PLACEHOLDER** / **BLOCKED_BY_LATER_TRANSLATION** /
**NOT_APPLICABLE_WITH_REASON** / **DEFECT**.

State at `49314cd` (staged code at the validated hotfix `b4776a2`).

## 1. Archive

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| 1.1 | Archive named `contest1_<TEAM_NAME>_003.zip` | BLOCKED_BY_OWNER_PLACEHOLDER | name template recorded in `STAGING_REPORT.json` as `intended_archive_name`; the staging directory uses `TEAM_NAME_PLACEHOLDER` because Windows rejects `<`/`>` in a path component |
| 1.2 | Root contains `code/` | SATISFIED | staging root holds exactly one `code/` directory, 25 files |
| 1.3 | Root contains `requirements.txt` or `environment.yaml` | SATISFIED | **both** are staged (stricter than required) |
| 1.4 | Root contains `提交说明文档.pdf` | **BLOCKED_BY_LATER_TRANSLATION** | the final Chinese PDF is produced after this task from the final English source; no placeholder or substitute PDF has been inserted |
| 1.5 | Archive extracts cleanly, not damaged or incomplete | SATISFIED | deterministic snapshot extracted into an empty directory on the target host and locally; `testzip` clean, no path traversal, 0 manifest hash mismatches |

## 2. Code package

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| 2.1 | Reproduces the best A-board result | SATISFIED | both raw-to-final Jittor executions completed on the target RTX 4090; document §11 |
| 2.2 | Data processing | SATISFIED | `src/pipeline_common.py`, `src/ds2_basket_featurizer.py`; document §4 |
| 2.3 | Model training | SATISFIED | `src/train_line_jt.py`, `src/train_bpr_jt.py`; both executed for real in Phase A |
| 2.4 | Inference and prediction | SATISFIED | `src/ranker_ds1.py`, `src/ranker_basket_ds2.py`, `src/ds2_mf_basket_pack.py`; real LightGBM path executed from the extracted package |
| 2.5 | Core algorithm logic | SATISFIED | `src/crf_promote.py`, `src/strategies/`; real CRF executed from the extracted package |
| 2.6 | Final prediction generation | SATISFIED | `src/build_ds1_member.py`, `src/build_ds2_member.py`; both executed, output gates asserted |
| 2.7 | Starts from official raw data alone | SATISFIED | document §6, §7; no stage reads a reference prediction or frozen member; the Dataset-2 passthrough refuses an unrecorded member |
| 2.8 | Training and inference run independently | SATISFIED | `python main.py --dataset dataset1` / `dataset2`; 33 stages contract-checked |
| 2.9 | Produces both final members | SATISFIED | `outputs/members/dataset1.csv` (61,051 x 100), `outputs/members/dataset2.csv` (153,420 x 100); document §7.3 |
| 2.10 | No test labels used | SATISFIED | document §6, §13; ranker trained on a cut-split replay of training data |
| 2.11 | No frozen prediction as a shortcut | SATISFIED | document §6, §13; enforced by the staging exclusions and by the passthrough contract |
| 2.12 | Consistent with the frozen A-board algorithm | SATISFIED | document §13; AST comparison of every scientific constant and function body against the full-run baselines; both frozen replays byte-exact at the release candidate |
| 2.13 | B-board adaptation disclosed | NOT_APPLICABLE_WITH_REASON | no B-board adaptation has been made or authorised; document §13 states future operation must remain consistent |
| 2.14 | Code manually inspectable and reproducible | SATISFIED | document §14 gives four reviewer commands; every command verified from an extracted copy |

## 3. Environment

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| 3.1 | Ubuntu 22.04 | SATISFIED | validated on the target host (22.04.4); document §5 |
| 3.2 | NVIDIA RTX 4090 | SATISFIED | validated, sm_89, 24,564 MiB; document §5 |
| 3.3 | CUDA 12.4 compatibility | SATISFIED | `/usr/local/cuda` -> `cuda-12.4`, nvcc 12.4.131; document §5 |
| 3.4 | Python 3.10 | SATISFIED | 3.10.20 validated; pinned in `environment.yaml` |
| 3.5 | Jittor >= 1.3.10 | SATISFIED | 1.3.10.0 validated, real training executed; `jittor==1.3.10` pinned |
| 3.6 | JittorGeometric | SATISFIED | 2.0.0 at pinned commit `ff7d8ffac7bf3d95cc1962e091c52dc5737492d4`, installed and importable; document §3.5, §5.1 |
| 3.7 | NumPy pinned | SATISFIED | `numpy==1.26.4` in both files |
| 3.8 | Pandas pinned | SATISFIED | `pandas==2.2.3` in both files |
| 3.9 | scikit-learn pinned | SATISFIED | `scikit-learn==1.6.1` in both files; declared because the notice requires it, and documented as not imported by the chain |
| 3.10 | All other required dependencies with explicit versions | SATISFIED | `scipy==1.15.2`, `lightgbm==4.5.0`, `tqdm==4.67.1`; environment audit compares the declared set against the staged import closure |
| 3.11 | Special runtime settings documented | SATISFIED | `conv_opt=1`, `use_mkl=0`; document §5.2 with the measured evidence that the default shell aborts |

## 4. Required PDF content

The final PDF is produced from the English source by later translation. Each item is traced to the
English section that will supply it.

| # | Item | Verdict | English section |
|---|---|---|---|
| 4.1 | Team name | BLOCKED_BY_OWNER_PLACEHOLDER | §1 `<TEAM_NAME>` |
| 4.2 | A-board rank | BLOCKED_BY_OWNER_PLACEHOLDER | §1 `<A_BOARD_RANK>` |
| 4.3 | Best A-board score | BLOCKED_BY_OWNER_PLACEHOLDER | §1 `<A_BOARD_BEST_TOTAL_SCORE>` |
| 4.4 | Contact person | BLOCKED_BY_OWNER_PLACEHOLDER | §1 `<CONTACT_NAME>` |
| 4.5 | WeChat | BLOCKED_BY_OWNER_PLACEHOLDER | §1 `<WECHAT_ID>` |
| 4.6 | Phone number | BLOCKED_BY_OWNER_PLACEHOLDER | §1 `<PHONE_NUMBER>` |
| 4.7 | Project overview | SATISFIED | §2 |
| 4.8 | Task and solution | SATISFIED | §2.1–2.3 |
| 4.9 | Model selection | SATISFIED | §2.3, §4.3 |
| 4.10 | Algorithm innovations | SATISFIED | §2.4 |
| 4.11 | Code structure | SATISFIED | §4 |
| 4.12 | Training and inference logic | SATISFIED | §4.3 |
| 4.13 | Operating system | SATISFIED | §5 |
| 4.14 | CUDA | SATISFIED | §5 |
| 4.15 | Python | SATISFIED | §5 |
| 4.16 | Dependency installation | SATISFIED | §5.1 |
| 4.17 | Full Dataset-1 reproduction command | SATISFIED | §7.1, §7.2 |
| 4.18 | Full Dataset-2 reproduction command | SATISFIED | §7.1, §7.2 |
| 4.19 | Input paths | SATISFIED | §6 |
| 4.20 | Output paths | SATISFIED | §7.3 |
| 4.21 | Important hyperparameters | SATISFIED | §8.1–8.6 |
| 4.22 | How final submission-format files are generated | SATISFIED | §9 |
| 4.23 | Known issues | SATISFIED | §12 (nine items) |
| 4.24 | Reproduction notes | SATISFIED | §11, §14 |
| 4.25 | JittorGeometric installation and usage status | SATISFIED | §3.5, §5.1 — installed, commit-pinned, **not directly imported**, no artificial import added |
| 4.26 | Rendered as `提交说明文档.pdf` | **BLOCKED_BY_LATER_TRANSLATION** | English preview exists as `submission_document_draft_en.pdf`; the reserved Chinese filename is deliberately unused |

## 5. Compliance

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| 5.1 | No credentials or private material | SATISFIED | secret scan over every staged file, by filename pattern and content |
| 5.2 | Archive not damaged, extractable, complete | SATISFIED | see 1.5 |
| 5.3 | Manual inspection and reproduction feasible | SATISFIED | §14 reviewer commands; package independence verified from an extracted copy |
| 5.4 | Consistency between A-board and B-board | SATISFIED | §13 |

## 6. Higher standards applied beyond the requirements

| Item | Rationale |
|---|---|
| Both `requirements.txt` and `environment.yaml` shipped | the organiser asks for either; shipping both removes an installation choice from the reviewer |
| Machine-readable `STAGING_MANIFEST.json` inside the tree, `STAGING_REPORT.json` outside it | per-file SHA256 lets a reviewer verify any file independently |
| Every command verified from an extracted copy, never from the repository | proves package independence rather than asserting it |
| SHA256 for every staged file, plus a composite tree hash | makes the package state citable |
| Duplicate, hidden-file, symlink and absolute-path scans | catches packaging faults the organiser would otherwise find |
| Clean-room extraction report | evidence that extraction works before the organiser tries it |
| This traceability matrix | no requirement can be silently unclassified |

## 7. Summary

| Verdict | Count |
|---|---:|
| SATISFIED | 46 |
| BLOCKED_BY_OWNER_PLACEHOLDER | 6 |
| BLOCKED_BY_LATER_TRANSLATION | 2 |
| NOT_APPLICABLE_WITH_REASON | 1 |
| **DEFECT** | **0** |

The eight blocked items are the seven owner-supplied values and the final Chinese PDF. Neither
class is a defect: both are awaiting an input this task deliberately does not produce.
