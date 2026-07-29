# Naming standard

Applied 2026-07-29. Where the repository already had a working convention, that convention won.

## Python

| item | convention | example |
|---|---|---|
| modules and files | `snake_case` | `source_slate_recurrence.py` |
| classes | `PascalCase` | `Report` |
| functions, variables | `snake_case` | `row_max_normalise` |
| frozen constants | `UPPER_SNAKE_CASE` | `MINIMUM_OTHER_SLATES` |

## Markdown

Two tiers, both already present in the repository and both kept:

- **root and top-level `docs/` documents that define state**: `SCREAMING_SNAKE_CASE.md` —
  `README.md`, `CURRENT_PRODUCTION.md`, `STRATEGY_REGISTRY.md`, `SUBMISSION_PROTOCOL.md`.
- **everything nested**: `kebab-case.md` — `docs/rounds/round-21.md`,
  `docs/architecture/pipeline-overview.md`, `docs/maintenance/repository-reorganisation.md`.
  This follows the pre-existing `docs/data-b-runbook.md`, which was **not** renamed.

## Strategy ids

Stable, mechanism-oriented, lowercase `snake_case`. A strategy id must not encode a round number,
an agent name, a model name, a seed value or a transient hyperparameter — those belong in
manifests, not in names. `source_slate_recurrence`, not `r21_r2_codex_rule`.

The one admitted exception is a mechanism that *is* about seeds: `multi_seed_bpr_ensemble`
describes seed averaging itself, which the standard treats as scientifically essential.
`tests/integration/test_strategy_inventory.py` enforces this rule and encodes the exception.

## Rounds and agents

- round directories: `round-XX`, zero-padded — `round-03`, `round-21`, `round-22`
- agent directories: `codex`, `fable`, `opus`, `gpt`

Existing physical scratchpad directories (`round21_codex`, `round22_opus`, …) were **not**
renamed; they carry a canonical logical identity in `scratchpad/index.json` instead. See
ambiguity A7 — moving them breaks scripts that resolve the repository root by path depth.

## Artifact taxonomy

One canonical name per artifact kind. `brief` and `prompt` are **not** interchangeable: the
canonical name is `prompt.md`.

| artifact | file | lives in |
|---|---|---|
| the exact instruction sent to an agent | `prompt.md` | `docs_local/agent_runs/round-XX/<agent>/` |
| the conversational completion returned | `response.md` | `docs_local/agent_runs/round-XX/<agent>/` |
| the formal evidence-bearing report | `report.md` | `scratchpad/<run>/` |
| machine-readable provenance | `manifest.json` | `scratchpad/<run>/` |
| execution artifacts | `scripts/ probes/ artifacts/ logs/ tests/` | `scratchpad/<run>/` |
| concise project-level history | `round-XX.md` | `docs/rounds/` (tracked) |

When one run genuinely has several chronological prompts or responses, number them:
`prompt-01.md`, `response-01.md`. Do not number a single artifact.

## Ownership rule

- prompt and conversational response → `docs_local/agent_runs/` (local-only, ignored)
- formal report, manifest, scripts, probes, artifacts, logs → `scratchpad/` (ignored)
- concise conclusions, production state, strategy lifecycle → `docs/` and `configs/` (**tracked**)

Avoid duplication. If the same report exists in two places, the scratchpad copy is canonical and
the other should become a reference to it.

## Outputs

`outputs/<dataset>[-<suffix>]/`, one run per directory, non-default knobs appended as a suffix so
runs never clobber. **These names are load-bearing** — production source reads several by
hard-coded path. Do not rename one without updating every reference and re-running
`python src/build_ds1_member.py --verify`.
