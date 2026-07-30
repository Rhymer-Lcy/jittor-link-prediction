---
title: Sixth Jittor Artificial Intelligence Challenge — Official Competition Dossier
document_id: JITTOR_2026_OFFICIAL_COMPETITION_DOSSIER
version: 1.0.0
status: ACTIVE
language: English
canonical_source_url: https://www.educoder.net/competitions/Jittor-7
source_snapshot_date: 2026-07-30
last_reviewed: 2026-07-30
maintainer_scope: Competition compliance, reproducibility, A/B-board consistency, code review, open-source release, and final defence preparation
companion_source_register: docs/competition/source_register.md
---

# Sixth Jittor Artificial Intelligence Challenge — Official Competition Dossier

## 1. Purpose and Scope

This document consolidates the official competition information relevant to the Sixth Jittor Artificial Intelligence Challenge and, in particular, Track 1: **Dynamic Recommendation Based on Graph Learning**.

It serves as the canonical repository reference for:

- competition governance;
- competition schedule;
- track definition;
- dataset and graph semantics;
- evaluation and submission requirements;
- A-board and B-board consistency obligations;
- data-use restrictions;
- Jittor framework requirements;
- A-board code-inspection requirements;
- reproducibility obligations;
- open-source obligations;
- B-board admission;
- final defence;
- project compliance decisions.

This document does not replace the official competition website or organiser notices. If the official website, a later organiser notice, or a written organiser clarification changes a requirement, the later and more specific official source governs.

The authoritative source register is maintained in:

`docs/competition/source_register.md`

---

## 2. Normative Status and Interpretation Rules

### 2.1 Source hierarchy

When official materials conflict or differ in specificity, apply the following hierarchy:

1. written organiser clarification addressed to the team;
2. later official competition notice;
3. track-specific official page;
4. general competition rules;
5. official framework documentation;
6. archived official screenshot or transcript;
7. project interpretation.

A more specific official requirement overrides a more general requirement only within its stated scope.

### 2.2 Separation of official rules and project policy

This document distinguishes between:

- **OFFICIAL REQUIREMENT**: directly stated in the official competition material;
- **PROJECT COMPLIANCE POLICY**: a conservative operational interpretation adopted by the project;
- **OPEN CLARIFICATION ITEM**: an ambiguity requiring written confirmation from the organisers.

Project policy must not be represented as an official rule.

### 2.3 Change control

Any material rule change must be recorded in both:

- this dossier;
- `docs/competition/source_register.md`.

Historical versions must not be silently rewritten. Material amendments require:

- date;
- source;
- affected section;
- prior interpretation;
- revised interpretation;
- project impact.

---

## 3. Competition Overview

### 3.1 Competition name

**Sixth Jittor Artificial Intelligence Challenge**

Chinese name:

**第六届计图人工智能算法挑战赛**

### 3.2 Background

The Jittor Artificial Intelligence Algorithm Challenge was established in 2021 under the guidance of the Department of Information Sciences of the National Natural Science Foundation of China.

The competition is organised around the Jittor deep-learning framework developed by Tsinghua University. It is open to students and practitioners in artificial intelligence and related fields.

The competition is intended to:

- improve algorithmic research capability;
- improve data-analysis and data-processing capability;
- promote practical application of artificial intelligence;
- strengthen the ecosystem of domestic artificial-intelligence platforms;
- advance artificial-intelligence research and application.

Tencent provides sponsorship support.

### 3.3 Participation scope

The competition is open to:

- enrolled students;
- artificial-intelligence practitioners;
- practitioners in related technical fields.

Participation requires registration through the EduCoder platform.

---

## 4. Governance and Organisation

### 4.1 Supervising organisation

- Department of Information Sciences, National Natural Science Foundation of China

### 4.2 Organisers

- Beijing National Research Center for Information Science and Technology
- Tsinghua University–Tencent Joint Laboratory for Internet Innovation Technology

### 4.3 Sponsorship and technical support

- Tencent Technology (Shenzhen) Co., Ltd.  
  Competition sponsorship, including a stated prize pool contribution of RMB 280,000.

- Tianjin Artificial Intelligence Computing Center / Hebei District Government and Data Bureau of Tianjin  
  Computing support based on Huawei hardware.

- Sugon Intelligent Computing Information Technology Co., Ltd.  
  Intelligent-computing support.

- Beijing Feishi Technology Co., Ltd.  
  Competition technical support.

### 4.4 Steering committee

Listed in the official material in alphabetical order:

1. 戴琼海 — Professor, Tsinghua University; Member of the Chinese Academy of Engineering; President of the Chinese Association for Artificial Intelligence
2. 胡事民 — Professor, Tsinghua University; Member of the Chinese Academy of Sciences; Vice President of the China Computer Federation
3. 刘克 — Former Executive Deputy Director, Department of Information Sciences, National Natural Science Foundation of China
4. 梅宏 — Professor, Peking University; Member of the Chinese Academy of Sciences
5. 沈向洋 — Chairman, International Digital Economy Academy; Foreign Member of the US National Academy of Engineering
6. 王怀民 — Professor, National University of Defense Technology; Member of the Chinese Academy of Sciences
7. 吴国政 — Director, Division II for Computing and Artificial Intelligence, Department of Information Sciences, National Natural Science Foundation of China
8. 徐宗本 — Professor, Xi'an Jiaotong University; Member of the Chinese Academy of Sciences
9. 查红彬 — Professor, Peking University
10. 张钹 — Professor, Tsinghua University; Member of the Chinese Academy of Sciences
11. 章毅 — Professor, Sichuan University
12. 周志华 — Professor, Nanjing University; Member of the Chinese Academy of Sciences; President of the International Joint Conferences on Artificial Intelligence Board of Trustees

### 4.5 Expert committee

Listed in the official material in alphabetical order:

1. 白翔 — Professor, School of Software Engineering, Huazhong University of Science and Technology
2. 程明明 — Professor, College of Computer Science, Nankai University
3. 董未名 — Researcher, Institute of Automation, Chinese Academy of Sciences
4. 高林 — Researcher, Institute of Computing Technology, Chinese Academy of Sciences
5. 郭延文 — Professor, Department of Computer Science and Technology, Nanjing University
6. 黄华 — Professor, School of Artificial Intelligence, Beijing Normal University
7. 李庆利 — Professor, School of Communication and Electronic Engineering, East China Normal University
8. 刘偲 — Professor, School of Artificial Intelligence, Beihang University
9. 吕琳 — Professor, School of Computer Science and Technology, Shandong University
10. 孟德宇 — Professor, School of Mathematics and Statistics, Xi'an Jiaotong University
11. 闵卫东 — Professor, School of Mathematics and Computer Sciences, Nanchang University
12. 童若锋 — Professor, College of Computer Science and Technology, Zhejiang University
13. 王巨宏 — Director, Tencent Technical Committee
14. 王志衡 — Deputy Director, Division II for Computing and Artificial Intelligence, Department of Information Sciences, National Natural Science Foundation of China
15. 魏哲巍 — Professor, Gaoling School of Artificial Intelligence, Renmin University of China
16. 严骏驰 — Professor, School of Computer Science, Shanghai Jiao Tong University
17. 张蕾 — Professor, College of Computer Science, Sichuan University
18. 张松海 — Tenured Associate Professor, Department of Computer Science and Technology, Tsinghua University
19. 郑伟诗 — Professor, School of Data and Computer Science, Sun Yat-sen University
20. 左旺孟 — Professor, School of Computer Science and Technology, Harbin Institute of Technology
21. 周明辉 — Professor, School of Computer Science, Peking University

### 4.6 Organising committee

Listed in the official material in alphabetical order:

1. 国孟昊 — Postdoctoral Researcher, Department of Computer Science and Technology, Tsinghua University
2. 穆太江 — Associate Researcher, Department of Computer Science and Technology, Tsinghua University
3. 杨国炜 — Co-founder, Feishi Technology
4. 郑宇飞 — Director of Industry–Academia Cooperation, Tencent

---

## 5. Competition Tracks

The competition contains:

- two warm-up tracks;
- two official competition tracks.

### 5.1 Warm-up tracks

1. Paper classification based on citation networks
2. Point-cloud classification

A team must pass at least one warm-up track before participating in either official track.

### 5.2 Official tracks

1. Dynamic recommendation based on graph learning
2. Three-dimensional point-cloud denoising based on deep learning

This repository concerns:

**Track 1 — Dynamic Recommendation Based on Graph Learning**

---

## 6. Official Schedule

All times are interpreted as China Standard Time, UTC+8, unless the official platform states otherwise.

| Date and time | Milestone |
|---|---|
| 2026-04-09 00:00:00 | Registration opens |
| 2026-04-11 12:00:00 | Warm-up evaluation opens; official-track training datasets become available |
| 2026-04-20 12:00:00 | A-board evaluation datasets become available |
| 2026-04-30 12:00:00 | A-board submission entry opens |
| 2026-07-15 12:00:00 | Registration closes; warm-up evaluation closes |
| 2026-07-30 12:00:00 | A-board closes |
| 2026-08-03 12:00:00 | A-board code-inspection materials deadline |
| 2026-08-10 12:00:00 | B-board evaluation datasets and submission entry open |
| 2026-08-20 12:00:00 | B-board closes; code and model inspection begins |
| 2026-08-25 12:00:00 | B-board top-ten teams announced |
| September 2026 | Final defence; exact time and venue to be announced separately |

The source heading for the A-board inspection notice contains the apparent typographical date `20026/7/30`. This dossier normalises it to `2026-07-30` based on the competition schedule and surrounding context. The source discrepancy remains recorded in the source register.

---

## 7. Track 1 — Task Definition

### 7.1 Problem domain

The track concerns future-link prediction in temporal graphs.

Applications include:

- recommendation systems;
- social networks;
- transaction networks;
- citation networks;
- web-navigation prediction;
- other time-dependent interaction graphs.

### 7.2 Input representation

The training data consists of temporal interaction triples:

\[
(u, v, t)
\]

where:

- \(u\) is the source node;
- \(v\) is the destination node;
- \(t\) is the interaction timestamp.

Each row indicates that source node \(u\) interacted with destination node \(v\) at time \(t\).

### 7.3 Test query representation

Each test row contains:

- one source node;
- one query timestamp;
- exactly 100 candidate destination nodes.

For each query, the participant must return a probability or ranking score for every candidate destination node.

### 7.4 Prediction objective

Given:

- historical temporal interactions;
- source node \(u\);
- query timestamp \(t\);
- a supplied candidate set \(C(u,t)\);

the model predicts which candidate destination node is most likely to interact with \(u\) at time \(t\).

The operational problem is therefore a temporal candidate-set reranking problem.

### 7.5 Temporal causality

Prediction features must be derived only from information available at or before the applicable prediction boundary.

Test labels, future interactions, and answer-derived information must not be used.

---

## 8. Dataset and Graph Semantics

### 8.1 A-board datasets

The A-board provides datasets of moderate difficulty. Their regularities are described as comparatively easier to discover.

### 8.2 B-board datasets

The B-board provides more difficult datasets with:

- millions of nodes;
- tens of millions of interactions;
- higher modelling difficulty;
- explicit computational-efficiency constraints.

### 8.3 Graph types

Both the simpler and more difficult datasets include:

- bipartite graphs;
- non-bipartite graphs.

### 8.4 Bipartite graph definition

In a bipartite scenario:

- source nodes and destination nodes belong to distinct node sets;
- source and destination identities must not be assumed to share one namespace unless explicitly established by the data.

### 8.5 Non-bipartite graph definition

In a non-bipartite scenario:

- source and destination nodes belong to the same node universe;
- the same node may occur in both interaction roles.

### 8.6 Required implementation property

The submitted algorithm must support both graph types.

The implementation must not:

- assume that every graph is bipartite;
- assume that every graph is non-bipartite;
- merge disjoint role-specific identifiers without evidence;
- split shared node identities into unrelated entities without justification.

---

## 9. Evaluation

### 9.1 Reciprocal rank

For a query \(i\), if the first relevant result appears at rank \(\operatorname{rank}_i\), its reciprocal-rank score is:

\[
RR_i = \frac{1}{\operatorname{rank}_i}
\]

### 9.2 Mean reciprocal rank

For a query set \(Q\):

\[
MRR = \frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{\operatorname{rank}_i}
\]

### 9.3 A-board score

The A-board score is the sum of the scenario-level MRR values:

\[
S_A = \sum_{d \in D_A} MRR_d
\]

### 9.4 B-board score

The official material states that the B-board result is composed of:

- 65% from the sum of scenario-level MRR values;
- 35% from the final defence score.

Conceptually:

\[
S_B = 0.65 \times S_{\text{MRR}} + 0.35 \times S_{\text{defence}}
\]

The organiser's final score normalisation and presentation govern.

### 9.5 Ranking and admission

A-board ranking is used to select and review teams for B-board admission.

Only A-board teams within the stated ranking scope and passing code inspection may enter the B-board.

The general rules refer to the A-board top 50 as the code-review admission population.

---

## 10. Prediction Submission Format

### 10.1 Archive layout

Each scenario produces one CSV file. All scenario CSV files are packaged in one ZIP archive.

Expected flat-root structure:

```text
result.zip
├── dataset1.csv
├── dataset2.csv
└── ...
```

### 10.2 CSV format

Each row contains one predicted value for each of the 100 supplied candidates.

Requirements:

- comma-separated values;
- ASCII comma `,`;
- values within `[0,1]`;
- eight decimal places;
- row order consistent with the official test data;
- candidate order consistent with the supplied candidate order;
- no additional columns;
- no omitted candidate;
- no duplicate row;
- no header unless the official dataset specification explicitly requires one.

Illustrative row:

```text
0.80000000,0.20000000,...,0.10000000
```

### 10.3 Submission validity

A package is invalid if it contains:

- malformed files;
- incorrect filenames;
- invalid row counts;
- missing candidates;
- corrupted ZIP members;
- unsupported nesting;
- values outside the required range;
- an inconsistent scenario mapping.

---

## 11. A/B-Board Algorithm Consistency

### 11.1 Official general requirement

The A-board and B-board algorithms must be fundamentally consistent and capable of supporting different data scales.

The official track page also states that, for different scenarios:

- neural-network models may differ;
- learned weights may differ;
- training parameters may differ.

### 11.2 A-board code-inspection requirement

The A-board inspection notice states that the code, algorithm, and model used for the B-board must be fully consistent with the A-board submission, and that unexplained modifications are prohibited.

### 11.3 Open interpretation issue

The following wording requires organiser clarification:

- the track page permits different models, weights, and training parameters across scenarios;
- the inspection notice requires A-board and B-board code, algorithms, and models to remain fully consistent.

This dossier does not claim that these statements are identical.

### 11.4 Conservative project compliance policy

Pending written organiser clarification, the project adopts the following policy:

The A-board and B-board must share the same:

- algorithmic family;
- scientific mechanism;
- feature semantics;
- graph-role treatment;
- training objective;
- candidate-scoring definition;
- inference-stage ordering;
- causal boundary;
- code path;
- configuration schema.

The following may vary only through disclosed configuration or training outputs already supported by the reviewed code:

- model dimensions;
- batch size;
- number of epochs;
- learning rate;
- regularisation;
- sample count;
- graph partition size;
- cache size;
- scenario-specific weights;
- scenario-specific hyperparameters;
- hardware-oriented batching.

The project must not introduce an undisclosed B-board-only algorithm after the A-board code inspection.

### 11.5 Required organiser clarification

The team should request written confirmation of whether:

1. scenario-specific model architectures are allowed when all alternatives are already present and documented in the A-board code package;
2. B-board scale adaptations such as partitioning, sampling, reduced dimension, and changed training schedules are considered parameter changes or algorithm changes;
3. a model family switch is permitted between scenarios under the statement that scenario-specific models may differ.

The clarification must be recorded as `ORGANISER_CLARIFICATION`.

---

## 12. Data-Use and Pretrained-Model Compliance

### 12.1 External task data

The official track page states that competition participants must not use data other than the datasets provided by the competition.

Therefore, direct use of the following is prohibited unless the organisers explicitly approve it:

- external interaction records;
- external knowledge graphs;
- external node metadata;
- web-derived node attributes;
- external labels;
- external popularity statistics;
- external entity matching;
- external test-answer information;
- data shared by another team.

### 12.2 Pretrained-model source disclosure

The code-review rules state that, if a pretrained model is used, its source must be provided.

This establishes a mandatory disclosure requirement.

### 12.3 Open pretrained-model ambiguity

The source-disclosure clause does not, by itself, conclusively resolve whether externally pretrained weights are permissible under the prohibition on additional data.

The project must not represent either of the following as confirmed without organiser clarification:

- all externally pretrained weights are permitted;
- all externally pretrained weights are prohibited.

### 12.4 Conservative project policy

Until written clarification is received:

- do not directly use external task data;
- do not query external databases during training or inference;
- disclose every pretrained model, checkpoint, repository, version, licence, and training-data description;
- do not use a pretrained artifact with unknown provenance;
- do not use a pretrained artifact that may encode competition answers;
- prefer competition-data-derived representations;
- obtain written organiser confirmation before relying on externally pretrained weights in the reviewed submission.

### 12.5 Prohibited sharing

Teams must not share:

- code;
- model weights;
- prediction results;
- answer information.

A violation may result in disqualification.

---

## 13. Jittor Framework Requirement

### 13.1 Mandatory framework

Participants must use the Jittor deep-learning framework for model design, training, and prediction.

Failure to implement the submission using Jittor is grounds for invalidation.

### 13.2 Jittor overview

Jittor is a just-in-time compiled deep-learning framework with:

- meta-operator-based design;
- dynamic execution interfaces;
- compiler-level optimisation;
- automatic code generation;
- support for computer vision, geometry learning, reinforcement learning, and related domains.

### 13.3 General framework requirements

The official Jittor page states general support for:

- Ubuntu 16.04 or later;
- Windows Subsystem for Linux;
- Python 3.7 or later;
- GCC 5.4 or later, or Clang 8.0 or later;
- optional NVCC 10.0 or later;
- optional CUDA and cuDNN acceleration;
- Docker, pip, or source installation.

### 13.4 Competition-specific reproduction environment

For A-board code inspection, the more specific required environment governs:

- Ubuntu 22.04;
- NVIDIA RTX 4090;
- CUDA 12.4 compatibility;
- Python 3.10;
- Jittor 1.3.10;
- JittorGeometric;
- version-pinned NumPy;
- version-pinned Pandas;
- version-pinned scikit-learn;
- all additional dependencies required by the project.

The submission must be reproducible in this target environment.

---

## 14. A-Board Code-Inspection Submission

### 14.1 Deadline

The A-board code-inspection package must be submitted by:

**2026-08-03 12:00:00 China Standard Time**

Late submissions are not accepted.

### 14.2 Required materials

Every A-board participant must submit:

1. a complete code package reproducing the team's best A-board result;
2. a detailed project and reproduction document;
3. one ZIP archive containing all required materials.

The code package must cover:

- raw-data processing;
- feature generation;
- model definition;
- model training;
- inference;
- post-processing;
- submission-file generation;
- all core algorithm logic required to reproduce the best A-board result.

### 14.3 Archive naming

Supported archive format:

`.zip`

Required naming convention:

```text
[contest1-or-contest2]_[team-name]_[A-board-rank].zip
```

Official examples:

```text
contest1_队伍名_001.zip
contest2_队伍名_001.zip
```

The exact team name and rank must match the official registration and leaderboard record.

### 14.4 Required archive structure

The ZIP root must contain:

```text
submission.zip
├── code/
├── requirements.txt
│   or
├── environment.yaml
└── 提交说明文档.pdf
```

The `code/` directory must contain all code required for:

- data processing;
- model construction;
- training;
- inference;
- result generation.

The environment file must pin all required versions.

### 14.5 Package size

The archive should remain within a reasonable size to reduce:

- upload failure;
- download failure;
- extraction failure;
- reviewer handling problems.

Large reproducible caches, raw datasets, temporary outputs, and unnecessary checkpoints should not be included.

---

## 15. Required Reproduction Document

The required PDF filename is:

```text
提交说明文档.pdf
```

The document must include at least the following sections.

### 15.1 Team information

- team name;
- A-board rank;
- best A-board total score;
- contact person;
- WeChat contact;
- telephone number.

Personal contact details must be included only in the private inspection package, not in the public repository.

### 15.2 Project overview

- task definition;
- overall solution;
- model selection;
- principal algorithmic components;
- innovation claims;
- relationship between the implementation and the submitted A-board result.

### 15.3 Code structure

- directory tree;
- file-level responsibilities;
- training entry point;
- inference entry point;
- data-processing entry point;
- post-processing modules;
- result-generation modules;
- relationship to `main.py` or the project's actual unified entry point.

### 15.4 Environment configuration

- operating-system version;
- CUDA version;
- Python version;
- Jittor version;
- JittorGeometric installation;
- dependency-installation commands;
- compiler requirements;
- environment variables;
- expected hardware;
- memory and storage requirements.

### 15.5 Execution instructions

For every dataset:

- full training command;
- inference command;
- input paths;
- output paths;
- expected intermediate artifacts;
- expected final artifacts;
- estimated runtime;
- expected peak memory;
- critical hyperparameters;
- random seeds;
- checkpoint-generation procedure;
- submission-file-generation procedure.

Illustrative interface:

```text
python main.py --dataset dataset1
python main.py --dataset dataset2
```

The actual commands must match the submitted implementation.

### 15.6 Parameter documentation

Document the meaning and value of all material parameters, including where applicable:

- batch size;
- epoch count;
- learning rate;
- embedding dimension;
- negative-sampling count;
- graph-propagation depth;
- tree count;
- random seed;
- thresholds;
- candidate-processing rules;
- post-processing order.

### 15.7 Known limitations

Document:

- known reproducibility risks;
- special installation requirements;
- platform-specific behaviour;
- expected numerical tolerance;
- deterministic and nondeterministic operations;
- required disk space;
- required model assets;
- known fallback behaviour.

---

## 16. Code-Inspection Reproducibility Requirements

The submitted code must:

- independently start from the original competition training data;
- train or construct all required models;
- process the official test data;
- generate predictions consistent with the team's best A-board submission;
- avoid all use of test ground truth;
- avoid answer leakage;
- run in the specified environment;
- contain no undisclosed dependency on local caches;
- contain no undocumented manually edited output;
- contain no dependency on another team's code or model;
- use Jittor for the relevant model implementation.

A submission may be invalidated if:

- code is missing;
- the implementation does not use Jittor;
- the result cannot be reproduced;
- code is substantially similar to another team's code;
- required assets are unavailable;
- the package is corrupt;
- the package cannot be extracted;
- the result-generation path is incomplete;
- hidden test information is used.

---

## 17. Submission Upload for Code Inspection

Official Tsinghua Cloud upload links:

Track 1:

```text
https://cloud.tsinghua.edu.cn/u/d/cd67f2e15fcc4996a9f5/
```

Track 2:

```text
https://cloud.tsinghua.edu.cn/u/d/13eb296cea3c4b8aa2d8/
```

This repository concerns Track 1.

The team must retain:

- uploaded filename;
- file size;
- SHA256;
- upload completion time;
- upload confirmation screenshot or receipt;
- final archive inventory.

---

## 18. Open-Source Requirements

### 18.1 Warm-up tracks

Warm-up projects may be published on GitLink and may qualify for commemorative rewards.

### 18.2 A-board

A-board participants must commit to open-sourcing their code on GitHub and GitLink after the B-board closes to remain eligible for B-board admission.

### 18.3 B-board

B-board participants must publish their code on both:

- GitHub;
- GitLink.

Open-source compliance is required for:

- prize eligibility;
- award-ceremony participation;
- recognition as a valid result.

### 18.4 Repository naming

The official naming format is:

```text
jittor-[team-name]-[project-name]
```

The repository README must follow the official open-source guide.

### 18.5 Public-release hygiene

The public repository must not contain:

- private contact information;
- credentials;
- test labels;
- private organiser communications;
- large raw competition datasets;
- unauthorised third-party materials;
- another team's assets;
- hidden submission-only files;
- machine-specific absolute paths.

---

## 19. B-Board Admission and Review

### 19.1 A-board admission gate

Only eligible A-board teams that pass code inspection may enter the B-board.

### 19.2 B-board evaluation

The B-board opens on:

**2026-08-10 12:00**

The B-board closes on:

**2026-08-20 12:00**

### 19.3 B-board scale

B-board datasets may contain:

- millions of nodes;
- tens of millions of interactions.

The implementation must therefore control:

- preprocessing complexity;
- graph-storage complexity;
- training complexity;
- inference complexity;
- peak memory;
- candidate-scoring cost.

### 19.4 B-board code and model review

The organisers review leading B-board teams and require reproduction of the best leaderboard result on the Jittor platform.

The reviewed top ten teams are selected for the final defence.

### 19.5 Top-ten announcement

The B-board top ten are scheduled to be announced on:

**2026-08-25 12:00**

---

## 20. Final Defence

Each finalist team prepares a presentation of approximately ten minutes.

The judging criteria include:

- technical approach;
- theoretical depth;
- B-board score;
- overall quality of the work;
- responses to judges' questions.

The precise date and venue are to be announced separately.

---

## 21. Awards and Publication Opportunities

### 21.1 Awards per official track

Each official track produces eight awarded teams:

| Award | Quantity | Pre-tax award |
|---|---:|---:|
| First Prize | 1 | RMB 50,000 and certificate |
| Second Prize | 2 | RMB 20,000 and certificate |
| Third Prize | 5 | RMB 10,000 and certificate |

### 21.2 Recruitment opportunities

Strong competition performance may provide access to Tencent campus-recruitment or internship advantages, including increased résumé visibility or interview opportunities.

### 21.3 Journal invitation

Projects with strong results and meaningful algorithmic innovation may, following expert-committee review and recommendation, receive an invitation from *Computational Visual Media*.

The official page states an SCI impact factor of 18.3. This dossier records that statement as an official competition claim and does not independently recalculate the metric.

---

## 22. Participation and Conduct Rules

### 22.1 Registration

Every participant must register through EduCoder.

Registration information must be accurate and valid.

Competition eligibility and prize payment depend on the registered information.

### 22.2 Team composition

A team may consist of:

- one participant;
- two participants;
- three participants.

Each participant may join only one team.

Team membership may not be changed after the registration deadline.

Multiple accounts used to participate in multiple teams may result in disqualification.

### 22.3 Team captain and team name

Each team must designate a captain.

The team name must:

- comply with Chinese law;
- comply with public-order and good-custom requirements;
- not contain misleading terms such as “official”.

The organising committee may dissolve a non-compliant team and cancel its result.

### 22.4 Conflict-of-interest restrictions

Members of the Tsinghua University Visual Media Research Center involved in organising the competition may not participate.

Tencent employees may participate, but their teams may not receive prize money. Such teams may still receive rankings and commemorative recognition.

Personnel from organisers or technical-support organisations who may have access to task answers may not participate.

### 22.5 Intellectual property and legality

Submission materials must:

- comply with laws and regulations;
- comply with competition rules;
- respect third-party rights;
- contain no plagiarised code;
- contain no unauthorised model or data;
- contain no shared competitor assets.

The organising committee retains final interpretation authority.

---

## 23. Communication and Support

### 23.1 Competition discussion

Official discussion channel:

- Jittor discussion community — competition section

### 23.2 Competition QQ group

```text
717152103
```

The application note should contain:

```text
team name + real name
```

### 23.3 Jittor framework support

Email:

```text
[public contact redacted]
```

Issue tracker:

```text
https://github.com/Jittor/jittor/issues
```

Jittor developer communication group:

```text
836860279
```

### 23.4 Computing support

The competition provides domestic-computing support for the two official tracks, including Huawei- and Hygon-based resources.

Teams lacking sufficient training resources may apply to the organisers.

---

## 24. Project Compliance Baseline

The following project rules are mandatory unless replaced by a more specific written organiser clarification.

### 24.1 Data boundary

- Use only competition-provided task data.
- Do not join external task records.
- Do not use external labels.
- Do not use hidden test answers.
- Do not use another team's predictions.
- Record the provenance of every model and artifact.

### 24.2 Framework boundary

- Production training and inference must be implemented through Jittor where required.
- Auxiliary analysis performed in another framework must not replace the reviewed Jittor implementation.
- The submitted reproduction path must run in the specified inspection environment.

### 24.3 A/B consistency boundary

- Preserve one algorithmic skeleton.
- Represent scenario differences through documented configuration.
- Do not introduce an undisclosed B-only algorithm.
- Ensure both bipartite and non-bipartite support.
- Ensure B-scale complexity is practical.

### 24.4 Reproducibility boundary

- Start from official raw training data.
- Generate all required derived artifacts.
- Train all required models.
- Generate the final submission files.
- Record deterministic seeds and nondeterministic limitations.
- Use repository-relative paths.
- Avoid undocumented manual operations.

### 24.5 Release boundary

- Keep the private inspection package separate from the public repository.
- Keep personal contact information out of the public repository.
- Pin dependencies.
- Record package and member hashes.
- Verify the archive after independent extraction.
- Preserve upload evidence.

---

## 25. Immediate Pre-B-Board Obligations

### 25.1 Required before 2026-08-03 12:00

1. Complete the reproducible implementation of the final A-board pipeline.
2. Ensure the final Dataset-2 decoder is represented in maintained code.
3. Freeze the A/B algorithm-consistency contract.
4. Build the inspection archive with the required structure.
5. Produce `提交说明文档.pdf`.
6. Validate the package in the target environment.
7. Verify that no test ground truth is used.
8. Verify that no private or unauthorised data is included.
9. Upload the package.
10. Record filename, size, SHA256, and upload confirmation.

### 25.2 Required before 2026-08-10 12:00

1. Resolve the Round-29 offline-to-online inversion sufficiently to prevent recurrence.
2. Complete B-scale memory and runtime planning.
3. Validate bipartite and non-bipartite execution paths.
4. Review retained local evidence and large generated artifacts.
5. Confirm organiser interpretation of A/B model consistency.
6. Confirm the permissibility of externally pretrained weights, if relevant.
7. Prepare B-board dataset-ingestion and scale-diagnostic procedures without changing the algorithmic family.

---

## 26. Open Clarification Items

| ID | Question | Current project policy | Required resolution |
|---|---|---|---|
| OCI-001 | May model architecture differ across A/B scenarios when the track page permits scenario-specific models but the inspection notice requires identical models? | Preserve one model family and one reviewed code path; vary only disclosed configuration. | Written organiser clarification |
| OCI-002 | Are externally pretrained weights permitted under the prohibition on data outside competition datasets? | Do not rely on external pretrained weights without written approval; disclose all provenance. | Written organiser clarification |
| OCI-003 | Are B-scale adaptations such as graph partitioning, neighbour sampling, reduced dimensions, and changed epochs considered permissible parameter changes? | Permit only configuration-driven scale adaptation already present in the reviewed A-board code. | Written organiser clarification |
| OCI-004 | What numerical tolerance is accepted when reproducing the best A-board submission? | Target exact ranking equivalence and document any unavoidable floating-point variance. | Inspection guidance or organiser clarification |
| OCI-005 | Must every generated model weight be reproduced during review, or may hash-pinned trained weights accompany the source? | Provide a complete from-scratch training path and document optional verified checkpoints separately. | Inspection guidance |
| OCI-006 | Does the A-board top-50 code-review statement apply uniformly to both official tracks and all tied ranks? | Treat top-50 eligibility as applicable pending specific notice. | Official confirmation |

---

## 27. Maintenance Requirements

This dossier must be reviewed:

- after every official competition notice;
- after every organiser clarification;
- before the A-board code-inspection upload;
- when B-board data is released;
- before the B-board submission;
- before public open-source release;
- before final defence.

Every review must update:

- document version;
- review date;
- source register;
- open clarification items;
- project-impact summary.

---

## 28. Canonical Source

Official competition entry:

```text
https://www.educoder.net/competitions/Jittor-7
```

Detailed source status, evidence locators, and verification limitations are recorded in:

`docs/competition/source_register.md`