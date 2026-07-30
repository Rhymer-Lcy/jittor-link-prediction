# Competition rules and sources

This directory is the canonical repository entry point for **official competition rules** for the
Sixth Jittor Artificial Intelligence Challenge, Track 1 (Dynamic Recommendation Based on Graph
Learning). Nothing here describes what the project built; it records what the competition requires.

For what the project built and shipped, see [../CURRENT_PRODUCTION.md](../CURRENT_PRODUCTION.md) and
[../rounds/README.md](../rounds/README.md).

## Index

| file | role |
|---|---|
| [official_competition_dossier.md](official_competition_dossier.md) | Consolidated official rules: governance, schedule, task definition, dataset semantics, evaluation, submission format, A/B consistency, data-use limits, framework requirement, code inspection, open-source obligations, defence, awards |
| [source_register.md](source_register.md) | Provenance and verification status of every source the dossier relies on, the claim-to-source matrix, identified source tensions, and the organiser-clarification workflow |
| `ab_algorithm_consistency_contract.md` | The frozen A/B algorithmic skeleton, the configuration surface permitted to vary, and the changes prohibited without disclosure. Added by the pre-B-board preparation work |

## Authority order

When sources conflict, the higher entry governs, and only for the subject it explicitly addresses:

1. written organiser clarification addressed to the team;
2. later official competition notice;
3. track-specific official page;
4. general official competition rules;
5. official Jittor framework documentation;
6. archived official screenshot or transcript;
7. project interpretation.

The register defines the source-status and verification-status vocabularies. Use those values; do
not invent new ones.

## This dossier does not replace later official information

The dossier is a snapshot consolidation, not the competition's own authority. A later official
notice, an updated official page, or a written organiser clarification **supersedes it** within the
scope that source addresses. When the two disagree, the official source governs and the dossier is
stale until reviewed.

## Organiser clarifications must be registered

An organiser clarification carries the highest authority, so it must not survive only in a chat log,
an inbox, or a screenshot on someone's machine. Every clarification received must be:

1. recorded in `source_register.md` under its own child source ID (`SRC-007-A`, `SRC-007-B`, ...),
   using the template in that file;
2. reflected in the affected dossier sections, with the prior interpretation preserved rather than
   overwritten;
3. propagated to the A/B consistency contract when it touches algorithmic scope.

Unresolved ambiguities live in the dossier's open-clarification table and the register's tension
records. An ambiguity must not be closed by project reasoning alone.

## Official requirement versus project policy

These are distinct and are never merged:

- **official requirement** — stated in official competition material, cited to a source ID;
- **project compliance policy** — a conservative internal interpretation the project adopts while an
  ambiguity is open. It binds the project's own work and carries no official authority;
- **open clarification item** — an ambiguity that needs written organiser confirmation.

Project policy must never be presented, internally or publicly, as an official rule. Where the
project chose to be stricter than the rules require, the documents say so explicitly.

## Maintenance

Both documents are **content-immutable in ordinary work**. Their prose, structure, terminology and
source classifications are changed only through a deliberate review that follows the change-control
procedure in the dossier (date, source, affected section, prior interpretation, revised
interpretation, project impact) and updates the register's change log in the same commit. Silent
correction of a claim is prohibited, including correction of an apparent error in an official
source: the anomaly is recorded and the normalisation is disclosed.

Every review must update the document version, the review date, the register, the open-clarification
table, and the project-impact summary. Version numbers of the two files are kept in agreement.

## Review triggers

Review both documents:

- after any official competition notice;
- after any organiser clarification;
- before the A-board code-inspection upload;
- when B-board data is released;
- before the B-board submission;
- before public open-source release;
- before the final defence.

The register's review checklist is the minimum set of facts to re-confirm each time, and its
pending-verification table lists the verification work still outstanding. Note that the official
competition entry is a dynamically rendered page: a non-browser fetch confirms the endpoint but
cannot capture the body, so archival verification needs a rendered capture.
