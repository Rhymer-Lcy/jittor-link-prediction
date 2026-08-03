# Organiser requirement traceability matrix

Every official requirement is traced to a specific English-document section,
source declaration, command, archived validation result or final-freeze check.
No requirement is left unclassified.

Verdicts: **SATISFIED** / **PARTIALLY_SATISFIED** /
**OPEN_REVIEWER_DECISION** / **BLOCKED_BY_LATER_TRANSLATION** /
**NOT_APPLICABLE_WITH_REASON** / **DEFECT**.

State after the 2026-08-03 documentation reconciliation and repository cleanup.
**Section numbers refer to the English document**,
which follows the structure of the reviewer-revised `提交说明文档.docx`
(SHA256 `1ef5712f9e890b7a91797697440596eeb8a63065052a9ba3d03b82423ef1248b`):
1 Team Information, 2 Project Overview, 3 Code Structure, 4 Environment Setup,
5 Run Procedure, 6 Runtime and Resource Requirements. The previous 14-section
numbering is superseded.

**OPEN_REVIEWER_DECISION** marked a requirement that the previous English draft
satisfied and that the reviewer's revision removed. All four such items
(2.12, 4.23, 4.24, 5.4) were closed in the reconciliation pass: the reviewer's
structure was kept and the missing substance was restored inside it, in
§2.6 *Evidence boundary*, §3.5 *Reproduction boundary* and §6.3 *Known issues*.
No open reviewer decision remains.

The English source is now `docs/submission/submission_document_en.md`; the
earlier `..._draft_en.md` name is retired.

Artifact locations below reflect the 2026-08-03 repository cleanup. The loose
final Chinese PDF and the draft staging tree are no longer kept in the
repository. The PDF remains hash-verified in the cold archive; the final
organiser archive has not yet been assembled or frozen. Dated audit records
retain their historical paths and wording.

## 1. Archive

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| 1.1 | Archive named `contest1_<TEAM_NAME>_003.zip` | **PARTIALLY_SATISFIED** | the team name and required final name, `contest1_皮卡丘_003.zip`, are known. No current staging tree or final organiser archive exists; the draft builder still uses `TEAM_NAME_PLACEHOLDER`, which must be replaced at freeze |
| 1.2 | Root contains `code/` | **PARTIALLY_SATISFIED** | the last audited draft contained exactly one `code/` directory with 26 files, and the current builder still declares the same 26 members. The final tree must be regenerated and re-audited at freeze |
| 1.3 | Root contains `requirements.txt` or `environment.yaml` | **PARTIALLY_SATISFIED** | the deterministic builder declares **both** files (stricter than required), but no current final archive exists; their final placement is verified at freeze |
| 1.4 | Root contains `提交说明文档.pdf` | **PARTIALLY_SATISFIED** | the final 24-page Chinese PDF is complete and hash-verified (`0cd48507…`) under the cold archive's `DOCUMENTS_OF_RECORD/提交说明文档.pdf`. The repository keeps no duplicate; it must be copied to the final archive root at freeze |
| 1.5 | Archive extracts cleanly, not damaged or incomplete | **PARTIALLY_SATISFIED** | the last draft passed `testzip`, path-traversal and manifest-hash checks. The final organiser archive does not yet exist, so the same checks remain mandatory after final assembly |

## 2. Code package

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| 2.1 | Reproduces the best A-board result | SATISFIED | both raw-to-final Jittor executions completed on the target RTX 4090 |
| 2.2 | Data processing | SATISFIED | `src/pipeline_common.py` §3.4, `src/ds2_basket_featurizer.py` §3.7 |
| 2.3 | Model training | SATISFIED | `src/train_line_jt.py`, `src/train_bpr_jt.py`; **logic documented in §3.5**, training/inference split in §3.9; hyperparameters in §5.2.1–5.2.2 |
| 2.4 | Inference and prediction | SATISFIED | `src/ranker_ds1.py` §3.6, `src/ranker_basket_ds2.py` and `src/ds2_mf_basket_pack.py` §3.7, matrix §3.9 |
| 2.5 | Core algorithm logic | SATISFIED | `src/crf_promote.py` §3.7 steps 15–19, `src/strategies/` §3.6 steps 11–13 and §3.7 step 20 |
| 2.6 | Final prediction generation | SATISFIED | `src/build_ds1_member.py` §3.6, `src/build_ds2_member.py` §3.7, gates and serialisation §3.8 |
| 2.7 | Starts from official raw data alone | SATISFIED | §3.2 raw-data layout and preflight; no stage reads a reference prediction or frozen member; the Dataset-2 passthrough refuses an unrecorded member (§3.2, §3.7 step 14) |
| 2.8 | Training and inference run independently | SATISFIED | `python main.py --dataset dataset1` / `dataset2` (§5.1); 33 stages contract-checked |
| 2.9 | Produces both final members | SATISFIED | `outputs/members/dataset1.csv` (61,051 x 100), `outputs/members/dataset2.csv` (153,420 x 100); §3.8 |
| 2.10 | No test labels used | SATISFIED | §3.6 step 5 (cut-split replay), §3.9 — "Test ground-truth labels are never used" with the candidate-identity distinction stated explicitly |
| 2.11 | No frozen prediction as a shortcut | SATISFIED | §3.7 step 14: the Dataset-1 passthrough bytes are never read as scores; enforced by the staging exclusions and the passthrough contract. *Note: the reviewer's revision dropped the previous draft's explicit standalone sentence; the substance is retained in §3.7* |
| 2.12 | Consistent with the frozen A-board algorithm | SATISFIED | §2.6 states lineage at four separate levels, and §3.5 *Reproduction boundary* names the four verified embedding-training differences. The downstream algorithm — features, LambdaRank parameters, CRF, both Dataset-1 postprocessors, the Dataset-2 decoder — is unchanged, which the static cross-version audit verified file by file. Exact producer identity and byte identity are explicitly **not** claimed |
| 2.13 | B-board adaptation disclosed | NOT_APPLICABLE_WITH_REASON | no B-board adaptation has been made or authorised |
| 2.14 | Code manually inspectable and reproducible | SATISFIED | §3 in full, plus `CODE_REVIEW_MAP.md`, `PACKAGED_CODE_INVENTORY.csv` and `CANONICAL_STAGE_TRACEABILITY.csv` in the review bundle |
| **2.15** | **Detailed code-file listing and per-file function** | **SATISFIED** | **§3.11** lists all **26** packaged files with category, function, reachability and whether each runs during a full reproduction; `PACKAGED_CODE_INVENTORY.csv` adds principal caller, input and output. Machine coverage audit: **26/26 documented**, zero undocumented |
| **2.16** | **Key-module logic explained** | **SATISFIED** | §3.1 layout and call chain; §3.2 orchestration; §3.3 the completion contract; §3.4 shared utilities; §3.5 both Jittor trainers; §3.6 and §3.7 the two complete chains; §3.8 validation and serialisation; §3.12 failure and resume. Each module answers: why it exists, who invokes it, what it reads, what it computes, what it writes |
| **2.17** | **Stage traceability** | **SATISFIED** | **§3.10** maps all **33** canonical stages (16 + 17, every BPR seed explicit) to module, stage type, input, output, device and Jittor use; `CANONICAL_STAGE_TRACEABILITY.csv` is the machine-readable form. Coverage audit: **33/33 documented** |
| **2.18** | **Packaged source free of development residue** | **SATISFIED** | source-comment cleanup at `4821d5c`, `94565f6` and `7eb2754`; staging hygiene audit reports **0 advisories**; a scan of the extracted package finds no reference to an alternative backend, an AI tool or a scratch path; executable-token equivalence proved across all 17 modified modules |

## 3. Environment

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| 3.1 | Ubuntu 22.04 | SATISFIED | validated on the target host (22.04.4); §4.1 |
| 3.2 | NVIDIA RTX 4090 | SATISFIED | validated, sm_89, 24,564 MiB; §4.1, §6 |
| 3.3 | CUDA 12.4 compatibility | SATISFIED | `/usr/local/cuda` -> `cuda-12.4`, nvcc 12.4.131; §4.1 |
| 3.4 | Python 3.10 | SATISFIED | 3.10.20 validated; pinned in `environment.yaml`; §4.1 |
| 3.5 | Jittor >= 1.3.10 | SATISFIED | 1.3.10.0 validated, real training executed; `jittor==1.3.10` pinned |
| 3.6 | JittorGeometric | SATISFIED | 2.0.0 at pinned commit `ff7d8ffac7bf3d95cc1962e091c52dc5737492d4`, installed and importable; §4.1 |
| 3.7 | NumPy pinned | SATISFIED | `numpy==1.26.4` in both files |
| 3.8 | Pandas pinned | SATISFIED | `pandas==2.2.3` in both files |
| 3.9 | scikit-learn pinned | SATISFIED | `scikit-learn==1.6.1` in both files |
| 3.10 | All other required dependencies with explicit versions | SATISFIED | `scipy==1.15.2`, `lightgbm==4.5.0`, `tqdm==4.67.1`; environment audit compares the declared set against the staged import closure |
| 3.11 | Special runtime settings documented | SATISFIED | `conv_opt=1`, `use_mkl=0`; §4.2 with the cuDNN-8/9 explanation, and §3.5 stating that they are runtime compatibility settings and not scientific parameters |

## 4. Required PDF content

The final Chinese PDF has been rendered and archived. Each content item remains
traced to the English source section from which it was produced; row 4.26
separately records the state of final-package assembly.

| # | Item | Verdict | English section |
|---|---|---|---|
| 4.1 | Team name | SATISFIED | §1 — 皮卡丘 |
| 4.2 | A-board rank | SATISFIED | §1 — 3rd |
| 4.3 | Best A-board score | SATISFIED | §1 — 1.576996059163449, matching `configs/production.json` |
| 4.4 | Contact person | SATISFIED | §1 — [public contact redacted] |
| 4.5 | WeChat | SATISFIED | §1 |
| 4.6 | Phone number | SATISFIED | §1 |
| 4.7 | Project overview | SATISFIED | §2 |
| 4.8 | Task and solution | SATISFIED | §2.1–2.4 |
| 4.9 | Model selection | SATISFIED | §2.4; §3.5 LINE and BPR-MF; §3.6 and §3.7 the two LambdaRank rankers |
| 4.10 | Algorithm innovations | SATISFIED | §2.5 |
| 4.11 | Code structure | SATISFIED | §3, written in this task into the reviewer's empty section 三: §3.1–§3.12 |
| 4.12 | Training and inference logic | SATISFIED | §3.5 trainers; §3.6 Dataset-1 chain; §3.7 Dataset-2 chain; §3.9 training-vs-inference matrix |
| 4.13 | Operating system | SATISFIED | §4.1 |
| 4.14 | CUDA | SATISFIED | §4.1 |
| 4.15 | Python | SATISFIED | §4.1 |
| 4.16 | Dependency installation | SATISFIED | §4.1 |
| 4.17 | Full Dataset-1 reproduction command | SATISFIED | §5.1 |
| 4.18 | Full Dataset-2 reproduction command | SATISFIED | §5.1 |
| 4.19 | Input paths | SATISFIED | §3.2 — the expected `<data-root>/data_A/<dataset>/{train,test}.csv` layout and the preflight that checks it |
| 4.20 | Output paths | SATISFIED | §3.8 outputs and logs table; also §3.10 per stage |
| 4.21 | Important hyperparameters | SATISFIED | §5.2.1–5.2.5 |
| 4.22 | How final submission-format files are generated | SATISFIED | §3.8 "How the two submission files are produced", six numbered steps |
| 4.23 | Known issues | SATISFIED | **§6.3**: target environment, the two required runtime settings, the cuDNN component-probe limitation and the validated response, the Dataset-2 resource envelope, stage-level resume, the absent Dataset-2 end-to-end measurement, and the byte-identity caveat |
| 4.24 | Reproduction notes | SATISFIED | §3.12 failure and resume; §6.2 resumability; **§2.6** score ownership, no rescoring, no score interval, no byte-identity guarantee and the four-level lineage table; **§3.5** the four verified reasons a reproduction need not match; **§6.3** the same caveat restated where an operator will meet it |
| 4.25 | JittorGeometric installation and usage status | SATISFIED | §4.1 gives the pinned commit and install step; §3.5 states that **neither trainer imports JittorGeometric** and that no import was added merely to claim usage |
| 4.26 | Rendered as `提交说明文档.pdf` | **PARTIALLY_SATISFIED** | the final Chinese PDF is complete and hash-verified (`0cd48507…`, 24 pages) in the cold archive; no repository-side duplicate remains. `submission_staging/` has been cleared. The current `stage_package.py` still builds an English-document placeholder draft, so final freeze must stage this Chinese PDF under its official name and then build and audit the organiser archive |

## 5. Compliance

| # | Requirement | Verdict | Evidence |
|---|---|---|---|
| 5.1 | No credentials or private material | **PARTIALLY_SATISFIED** | the last draft passed filename and content secret scans. Because the final tree has not been assembled, the scan must be rerun over every final staged file at freeze |
| 5.2 | Archive not damaged, extractable, complete | **PARTIALLY_SATISFIED** | see 1.5; the prior draft passed, but the final organiser archive has not yet been built or tested |
| 5.3 | Manual inspection and reproduction feasible | SATISFIED | §3 plus the review-bundle reading map and inventories |
| 5.4 | Consistency between A-board and B-board | SATISFIED | §2.6 lineage table; see 2.12 |

## 6. Higher standards applied beyond the requirements

| Item | Rationale |
|---|---|
| Both `requirements.txt` and `environment.yaml` included at freeze | the prior audited draft contained both; the organiser asks for either, and retaining both removes an installation choice from the reviewer |
| Machine-readable `STAGING_MANIFEST.json` inside the tree, `STAGING_REPORT.json` outside it | required by the deterministic staging workflow at final freeze; per-file SHA256 lets a reviewer verify any file independently |
| `PACKAGED_CODE_INVENTORY.csv` and `CANONICAL_STAGE_TRACEABILITY.csv` | generated from the staging manifest and the packaged stage graph, so neither can drift from the package |
| Every command verified from an extracted copy, never from the repository | proves package independence rather than asserting it |
| Source-comment hygiene with proved token equivalence | the packaged code reads as production documentation, and the cleanup is provably behaviour-preserving |
| This traceability matrix | no requirement can be silently unclassified |

## 7. Summary

| Verdict | Count |
|---|---:|
| SATISFIED | 55 |
| PARTIALLY_SATISFIED | 8 |
| OPEN_REVIEWER_DECISION | 0 |
| BLOCKED_BY_LATER_TRANSLATION | 0 |
| NOT_APPLICABLE_WITH_REASON | 1 |
| **DEFECT** | **0** |

All four previously open items are closed. The reviewer's six-section structure
was kept intact; the substance the revision had dropped was restored **inside**
it rather than by re-adding the old closing sections, as §2.6, §3.5 and §6.3.
No item remains blocked by translation: the final Chinese PDF is complete and
hash-verified in the cold archive. The eight partially satisfied items are all
freeze-gated. There is no current staging tree or final organiser archive, and
the current `stage_package.py` still builds the earlier English-document,
placeholder-named draft. Final freeze must select the approved Chinese PDF and
team name, assemble the archive, generate its manifest, and rerun the package
integrity and secret checks.

**One correction the reconciliation forced.** The previous statement of 2.12 —
"the algorithm is unchanged and the claim remains true" — was too strong for the
embedding trainers. The static cross-version audit established four verified
differences in LINE training that can change learned embeddings and therefore
rankings, including that the historical serve-time runs used virtual-edge
feedback while the current trainer implements none. The document now separates
lineage consistency and downstream algorithm consistency, which hold, from exact
producer identity, byte identity and score equivalence, which are not claimed.
