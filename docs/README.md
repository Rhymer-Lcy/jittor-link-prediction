# Documentation

The documentation is organised by current responsibility rather than by the
chronology of experimentation.

| Document | Purpose |
|---|---|
| [CURRENT_PRODUCTION.md](CURRENT_PRODUCTION.md) | Accepted scores, hashes, strategy chains, and reproducibility claims |
| [architecture/pipeline-overview.md](architecture/pipeline-overview.md) | Canonical Jittor pipeline and framework boundary |
| [architecture/stage-completion-contract.md](architecture/stage-completion-contract.md) | Rules for safe stage reuse and atomic artifact publication |
| [STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md) | Human-readable lifecycle summary of evaluated strategies |
| [strategy_inventory.json](strategy_inventory.json) | Machine-readable strategy lifecycle inventory |
| [SUBMISSION_PROTOCOL.md](SUBMISSION_PROTOCOL.md) | Submission archive construction and verification |
| [competition/](competition/) | Official-rule dossier, source register, consistency contract, and package inventory |
| [submission/submission_document_en.md](submission/submission_document_en.md) | Editable English submission document source |
| [data-b-runbook.md](data-b-runbook.md) | Controlled procedure for a later data release |
| [leaderboard_final.md](leaderboard_final.md) | Retained final-board observation and its evidence limits |
| [naming-standard.md](naming-standard.md) | Naming, language, path, and commit conventions |

`configs/production.json` is the machine-readable source of truth for accepted
hashes, scores, strategy order, and package policy. When prose conflicts with
that file, treat the prose as stale and correct both in one reviewed change.

Historical round reports, local conversations, cleanup audits, and large
experiment records are retained in private verified archives and Git history.
They are not duplicated in the current public documentation tree.

## Documentation requirements

- Write public documentation in formal English, except for official Chinese
  names and required functional literals.
- State whether a score is online-observed, derived, or inferred.
- Distinguish executable reproduction, algorithmic equivalence, and byte
  identity.
- Use repository-relative paths in durable records.
- Do not duplicate test counts, file counts, or mutable status values when a
  command or configuration file is authoritative.
- Do not publish personal communications, credentials, host paths, or private
  archive locations.
