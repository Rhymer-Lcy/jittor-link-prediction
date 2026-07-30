---
title: Sixth Jittor Artificial Intelligence Challenge — Source Register
document_id: JITTOR_2026_SOURCE_REGISTER
version: 1.0.0
status: ACTIVE
language: English
canonical_competition_url: https://www.educoder.net/competitions/Jittor-7
companion_dossier: docs/competition/official_competition_dossier.md
last_reviewed: 2026-07-30
---

# Sixth Jittor Artificial Intelligence Challenge — Source Register

## 1. Purpose

This register records the provenance, authority, verification status, and scope of every source used by:

`docs/competition/official_competition_dossier.md`

It exists to prevent:

- unattributed rule claims;
- silent replacement of official wording;
- confusion between official requirements and project interpretations;
- reliance on memory or chat history as the sole source of competition rules;
- loss of historical notices;
- unsupported resolution of conflicting requirements.

---

## 2. Source Status Vocabulary

Use only the following source-status values.

| Status | Meaning |
|---|---|
| `OFFICIAL_WEB_PAGE` | A live page hosted on an official competition or organiser domain |
| `OFFICIAL_NOTICE` | A dated official announcement or inspection notice |
| `OFFICIAL_FRAMEWORK_DOCUMENTATION` | Official documentation for the required Jittor framework |
| `OFFICIAL_SCREENSHOT` | A screenshot independently captured from an official source and preserved locally |
| `USER_PROVIDED_OFFICIAL_SNAPSHOT` | A screenshot or transcript supplied by the project owner and represented as originating from an official source |
| `ORGANISER_CLARIFICATION` | A written clarification issued by the competition organisers |
| `HISTORICAL_OFFICIAL_VERSION` | An archived earlier version of an official page or notice |
| `PROJECT_INTERPRETATION` | A conservative internal interpretation, not an official rule |
| `UNVERIFIED` | A claim whose official provenance has not yet been established |

---

## 3. Verification Status Vocabulary

| Status | Meaning |
|---|---|
| `ENDPOINT_CONFIRMED` | The official URL is reachable |
| `TEXT_INDEPENDENTLY_VERIFIED` | The relevant text was independently retrieved and compared |
| `VISUALLY_VERIFIED` | The relevant content was independently checked in the rendered page |
| `TRANSCRIPT_MATCHED_TO_SCREENSHOT` | A supplied transcript was checked against a supplied screenshot |
| `USER_ATTESTED` | The project owner identified the source or result |
| `PARTIALLY_VERIFIED` | Some but not all content was independently verified |
| `PENDING_DYNAMIC_PAGE_CAPTURE` | The official page is dynamically rendered and a stable local capture remains pending |
| `SUPERSEDED` | A later authoritative source supersedes the source within a defined scope |
| `NOT_APPLICABLE` | Independent verification is not applicable to the source type |

---

## 4. Authority Order

Use the following order when interpreting conflicting sources:

1. `ORGANISER_CLARIFICATION`
2. later `OFFICIAL_NOTICE`
3. track-specific `OFFICIAL_WEB_PAGE`
4. general competition `OFFICIAL_WEB_PAGE`
5. `OFFICIAL_FRAMEWORK_DOCUMENTATION`
6. `OFFICIAL_SCREENSHOT`
7. `USER_PROVIDED_OFFICIAL_SNAPSHOT`
8. `PROJECT_INTERPRETATION`
9. `UNVERIFIED`

A later source supersedes an earlier source only for the subject it explicitly governs.

---

## 5. Registered Sources

### SRC-001 — Official Competition Entry

| Field | Value |
|---|---|
| Source ID | `SRC-001` |
| Title | Sixth Jittor Artificial Intelligence Challenge |
| Source type | `OFFICIAL_WEB_PAGE` |
| URL | `https://www.educoder.net/competitions/Jittor-7` |
| Publisher | EduCoder / competition organisers |
| Access date | 2026-07-30 |
| Verification status | `ENDPOINT_CONFIRMED`, `PENDING_DYNAMIC_PAGE_CAPTURE` |
| Authority scope | Competition entry, navigation, general competition information |
| Local evidence | User-provided rendered-page screenshots and transcript |
| Notes | The page is dynamically rendered. The official endpoint was located, but stable machine-readable paragraph extraction was not available during this review. |

### SRC-002 — Competition Overview and Rules

| Field | Value |
|---|---|
| Source ID | `SRC-002` |
| Title | Competition Overview, Governance, Schedule, Rules, Awards, and Support |
| Source type | `USER_PROVIDED_OFFICIAL_SNAPSHOT` |
| Parent source | `SRC-001` |
| URL | `https://www.educoder.net/competitions/Jittor-7` |
| Page locator | Competition introduction and rules section |
| Snapshot date represented | 2026-07-30 |
| Verification status | `TRANSCRIPT_MATCHED_TO_SCREENSHOT`, `USER_ATTESTED` |
| Authority scope | Competition background, organisers, committees, tracks, schedule, participation, awards, publication, support |
| Dossier sections | 3–6, 18–23 |
| Notes | Requires an independently saved local rendered-page capture for full archival verification. |

### SRC-003 — Track 1: Dynamic Recommendation Based on Graph Learning

| Field | Value |
|---|---|
| Source ID | `SRC-003` |
| Title | Track 1 — Dynamic Recommendation Based on Graph Learning |
| Source type | `USER_PROVIDED_OFFICIAL_SNAPSHOT` |
| Parent source | `SRC-001` |
| URL | `https://www.educoder.net/competitions/Jittor-7` |
| Page locator | Track 1 task page |
| Snapshot date represented | 2026-07-30 |
| Verification status | `TRANSCRIPT_MATCHED_TO_SCREENSHOT`, `USER_ATTESTED` |
| Authority scope | Task definition, graph types, dataset scale, MRR, submission format, data restrictions, scenario-specific models and parameters |
| Dossier sections | 7–12 |
| Notes | Primary track-specific source. It overrides general statements where it is more specific. |

### SRC-004 — A-Board Code Inspection Notice

| Field | Value |
|---|---|
| Source ID | `SRC-004` |
| Title | Sixth Jittor Competition A-Board Code Inspection |
| Source type | `OFFICIAL_NOTICE` represented by `USER_PROVIDED_OFFICIAL_SNAPSHOT` |
| Parent source | `SRC-001` |
| URL | `https://www.educoder.net/competitions/Jittor-7` |
| Page locator | A-board code-inspection notice |
| Notice date | 2026-07-30 |
| Submission deadline | 2026-08-03 12:00:00 UTC+8 |
| Verification status | `TRANSCRIPT_MATCHED_TO_SCREENSHOT`, `USER_ATTESTED` |
| Authority scope | Inspection-package format, reproduction environment, documentation, A/B consistency, upload deadline, disqualification conditions |
| Dossier sections | 11, 13–17 |
| Source anomaly | The visible heading contains `20026/7/30`; normalised to `2026-07-30` based on context and the official schedule. |
| Notes | This is the controlling source for the A-board code-inspection package. |

### SRC-005 — Official Jittor Framework Page

| Field | Value |
|---|---|
| Source ID | `SRC-005` |
| Title | Jittor Deep-Learning Framework |
| Source type | `OFFICIAL_FRAMEWORK_DOCUMENTATION` represented by `USER_PROVIDED_OFFICIAL_SNAPSHOT` |
| Parent source | `SRC-001` |
| URL | `https://www.educoder.net/competitions/Jittor-7` |
| Page locator | Jittor framework information and installation page |
| Snapshot date represented | 2026-07-30 |
| Verification status | `TRANSCRIPT_MATCHED_TO_SCREENSHOT`, `USER_ATTESTED` |
| Authority scope | General Jittor description, installation modes, general compiler and runtime requirements |
| Dossier section | 13 |
| Notes | The competition-specific inspection environment in `SRC-004` is more specific and therefore governs reproduction. |

### SRC-006 — User-Provided Composite Evidence

| Field | Value |
|---|---|
| Source ID | `SRC-006` |
| Title | User-Provided Competition Screenshots and Full Transcript |
| Source type | `USER_PROVIDED_OFFICIAL_SNAPSHOT` |
| Source date | 2026-07-30 |
| Verification status | `USER_ATTESTED`, `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| Authority scope | Evidence supporting `SRC-002` through `SRC-005` |
| Intended local preservation | Private competition evidence archive |
| Public-release status | Exclude unless redistribution is appropriate and necessary |
| Notes | This evidence must not be treated as a substitute for a future organiser clarification. |

### SRC-007 — Future Organiser Clarifications

| Field | Value |
|---|---|
| Source ID | `SRC-007` |
| Title | Written Organiser Clarifications |
| Source type | `ORGANISER_CLARIFICATION` |
| Status | Not yet populated |
| Expected scope | A/B model consistency, B-scale adaptations, pretrained weights, reproduction tolerance |
| Verification status | `NOT_APPLICABLE` until received |
| Required preservation | Original message, sender identity, date, question, complete response, and project interpretation |
| Notes | Each clarification must receive a separate child source ID such as `SRC-007-A`. |

---

## 6. Claim-to-Source Matrix

| Claim ID | Claim | Primary source | Secondary source | Dossier section | Verification status |
|---|---|---|---|---|---|
| `CLM-001` | The competition was established in 2021 under the guidance of the Department of Information Sciences of the NSFC. | `SRC-002` | `SRC-006` | 3 | `PARTIALLY_VERIFIED` |
| `CLM-002` | The competition contains two warm-up tracks and two official tracks. | `SRC-002` | `SRC-006` | 5 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-003` | Passing at least one warm-up track is required before entering an official track. | `SRC-002` | `SRC-003` | 5 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-004` | The A-board closes on 2026-07-30 at 12:00 UTC+8. | `SRC-002` | `SRC-006` | 6 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-005` | A-board inspection materials are due on 2026-08-03 at 12:00 UTC+8. | `SRC-004` | `SRC-006` | 6, 14 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-006` | The B-board opens on 2026-08-10 at 12:00 UTC+8. | `SRC-002` | `SRC-006` | 6, 19 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-007` | The B-board closes on 2026-08-20 at 12:00 UTC+8. | `SRC-002` | `SRC-006` | 6, 19 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-008` | Track 1 is a temporal future-link prediction and candidate-reranking task. | `SRC-003` | `SRC-006` | 7 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-009` | Each test query contains one source, one timestamp, and 100 destination candidates. | `SRC-003` | `SRC-006` | 7 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-010` | A- and B-board data include both bipartite and non-bipartite graphs. | `SRC-003` | `SRC-006` | 8 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-011` | B-board datasets may contain millions of nodes and tens of millions of interactions. | `SRC-003` | `SRC-006` | 8, 19 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-012` | A-board score is the sum of scenario-level MRR values. | `SRC-003` | `SRC-006` | 9 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-013` | B-board evaluation uses 65% MRR component and 35% defence component. | `SRC-003` | `SRC-006` | 9 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-014` | Submission CSV values must lie in `[0,1]` and use eight decimal places. | `SRC-003` | `SRC-006` | 10 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-015` | Direct use of data outside the competition-provided datasets is prohibited. | `SRC-003` | `SRC-006` | 12 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-016` | A pretrained model source must be disclosed if a pretrained model is used. | `SRC-004` | `SRC-006` | 12 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-017` | Scenario-specific neural-network models, weights, and training parameters may differ. | `SRC-003` | `SRC-006` | 11 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-018` | The A-board inspection notice requires B-board code, algorithm, and model consistency with the A-board submission. | `SRC-004` | `SRC-006` | 11 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-019` | Jittor is mandatory for model design, training, and prediction. | `SRC-002`, `SRC-003`, `SRC-004` | `SRC-006` | 13, 16 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-020` | The inspection target environment is Ubuntu 22.04, RTX 4090, CUDA 12.4, Python 3.10, and Jittor 1.3.10. | `SRC-004` | `SRC-006` | 13–16 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-021` | The inspection archive must contain `code/`, an environment specification, and `提交说明文档.pdf`. | `SRC-004` | `SRC-006` | 14–15 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-022` | The submitted code must reproduce the best A-board result from the original training data. | `SRC-004` | `SRC-006` | 14–16 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-023` | Test ground truth must not be used. | `SRC-004` | `SRC-006` | 16 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-024` | A-board and B-board open-source obligations include GitHub and GitLink. | `SRC-002` | `SRC-006` | 18 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |
| `CLM-025` | B-board top teams are subject to code and model review, with the selected top ten entering the final defence. | `SRC-002` | `SRC-006` | 19–20 | `TRANSCRIPT_MATCHED_TO_SCREENSHOT` |

---

## 7. Identified Source Tensions

### TENSION-001 — Scenario-Specific Models versus A/B Identity

**Source A**

`SRC-003` permits different neural-network models, weights, and training parameters across scenarios.

**Source B**

`SRC-004` requires B-board code, algorithm, and model consistency with the A-board submission and prohibits unexplained modification.

**Status**

`OPEN_CLARIFICATION_ITEM`

**Current project policy**

Use one reviewed algorithmic family and code path. Express scenario and scale differences through disclosed configuration and trained weights already supported by the inspected A-board package.

**Required action**

Obtain written organiser clarification before introducing a materially different B-board architecture.

### TENSION-002 — External-Data Prohibition versus Pretrained-Model Disclosure

**Source A**

`SRC-003` prohibits data other than competition-provided datasets.

**Source B**

`SRC-004` requires source disclosure if a pretrained model is used.

**Status**

`OPEN_CLARIFICATION_ITEM`

**Current project policy**

Do not directly use external task data. Do not rely on externally pretrained weights without full provenance and written organiser clarification.

**Required action**

Ask whether externally pretrained weights are permitted and, if so, what restrictions apply to pretraining data, licences, frozen weights, and fine-tuning.

### TENSION-003 — General Jittor Requirements versus Inspection Environment

**Source A**

`SRC-005` describes broad supported Jittor environments.

**Source B**

`SRC-004` specifies Ubuntu 22.04, RTX 4090, CUDA 12.4, Python 3.10, and Jittor 1.3.10 for inspection.

**Status**

`RESOLVED_BY_SPECIFICITY`

**Resolution**

The A-board inspection environment governs the reproducibility package.

---

## 8. Pending Verification Work

| Action ID | Action | Priority | Completion condition |
|---|---|---:|---|
| `VER-001` | Save an independent rendered capture of the official competition overview. | P1 | Local evidence has date, source URL, and content hash |
| `VER-002` | Save an independent rendered capture of the Track 1 page. | P0 | Dataset, scoring, data restriction, and A/B statements are visually confirmed |
| `VER-003` | Save an independent copy of the A-board code-inspection notice. | P0 | Deadline, environment, package structure, and consistency wording are preserved |
| `VER-004` | Record the exact official page route or API identifier for each dynamic section. | P1 | Each registered source has a stable locator |
| `VER-005` | Request organiser clarification for `TENSION-001`. | P0 | Written response archived as `ORGANISER_CLARIFICATION` |
| `VER-006` | Request organiser clarification for `TENSION-002`. | P0 if pretrained weights are contemplated | Written response archived as `ORGANISER_CLARIFICATION` |
| `VER-007` | Confirm acceptable numerical tolerance for code-review reproduction. | P1 | Written inspection guidance recorded |
| `VER-008` | Confirm whether trained checkpoints may accompany, but not replace, the from-scratch reproduction path. | P1 | Written inspection guidance recorded |

---

## 9. Recommended Evidence Storage

Private or locally retained evidence should use a structure equivalent to:

```text
docs_local/competition/evidence/
└── 2026-07-30/
    ├── competition_overview/
    ├── track_1_dynamic_recommendation/
    ├── a_board_code_inspection/
    ├── jittor_framework/
    └── capture_manifest.json
```

Each evidence entry should record:

- source ID;
- source URL;
- capture time;
- capture method;
- file size;
- SHA256;
- page title;
- page locator;
- whether authentication was required;
- whether the page was dynamically rendered;
- whether personal information is present;
- redistribution status.

Raw screenshots and private notices should not automatically be committed to the public repository.

---

## 10. Organiser Clarification Record Template

Use the following structure for each clarification:

```markdown
### SRC-007-A — [Clarification title]

- **Question sent:** YYYY-MM-DD HH:MM UTC+8
- **Channel:** Official platform / email / official group
- **Recipient:** [Role or organiser identity]
- **Question:** [Complete question]
- **Response received:** YYYY-MM-DD HH:MM UTC+8
- **Response:** [Complete response]
- **Authority status:** ORGANISER_CLARIFICATION
- **Affected dossier sections:** [Section numbers]
- **Project interpretation:** [Implementation consequence]
- **Evidence path:** [Repository-relative or private evidence reference]
- **Evidence SHA256:** [Hash]
```

Do not paraphrase away material qualifications in an organiser response.

---

## 11. Review Checklist

Before treating this register as current, verify:

- [ ] The canonical official URL remains reachable.
- [ ] The competition schedule has not changed.
- [ ] The A-board inspection deadline remains 2026-08-03 12:00 UTC+8.
- [ ] The B-board opening remains 2026-08-10 12:00 UTC+8.
- [ ] The B-board closing remains 2026-08-20 12:00 UTC+8.
- [ ] The inspection environment has not changed.
- [ ] The required ZIP structure has not changed.
- [ ] The official upload link has not changed.
- [ ] The A/B consistency wording has not changed.
- [ ] The external-data rule has not changed.
- [ ] Any pretrained-model clarification has been recorded.
- [ ] Any reproduction-tolerance clarification has been recorded.
- [ ] The dossier and source register versions agree.
- [ ] All project policies remain labelled as project policies.
- [ ] No unverified claim is presented as independently confirmed.

---

## 12. Change Log

### 1.0.0 — 2026-07-30

- Created the initial canonical source register.
- Registered the official EduCoder competition entry.
- Registered the competition overview, Track 1 page, A-board inspection notice, and Jittor framework page.
- Recorded the dynamic-page extraction limitation.
- Registered the user-provided official screenshots and transcript.
- Added the claim-to-source matrix.
- Recorded the A/B consistency ambiguity.
- Recorded the external-data and pretrained-model ambiguity.
- Established the organiser-clarification workflow.