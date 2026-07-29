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

## Run identifiers

One identifier names a run in **both** local trees, so they cross-reference directly:

```
round-<NN>[<sub-round letter>]-<agent>
```

- `NN` zero-padded: `round-03-gpt`, `round-21-codex`, `round-22-opus`
- agents: `codex`, `fable`, `opus`, `gpt` — the thread that **owns** the artifact (the recipient
  of a prompt in `docs_local/`, the executor of a run in `scratchpad/`). These can differ for the
  same round and that is meaningful: `docs_local/agent_runs/round-16-fable` holds the brief that
  `scratchpad/round-16-opus` executed.
- the sub-round letter is kept **only where it denotes a real sub-round**: `round-18a-opus`,
  `round-18b-fable`, `round-18c-opus`, `round-19a-opus`, `round-20a-opus`, `round-21a-opus`.
  Letters that merely abbreviated the agent — 22**C**odex, 22**F**able, 22**O**pus — are dropped,
  because the agent is already in the identifier.

The historical label written **inside** each document (for example `ROUND 22F`) is recorded
verbatim as `document_label` in the indexes and is never rewritten: the historical letters are
inconsistent, and normalising the text would falsify the record.

**Runs are flat children of their tree**, `scratchpad/round-22-opus/`, not
`scratchpad/round-22/opus/`. Nesting would add a path component, and several run scripts resolve
the repository root by counting components (`parents[3]`, `parents[4]`) — the exact off-by-one
that broke the R2 and RGR migrations. The 2026-07-29 rename preserved depth for that reason;
`scratchpad/migration/legacy-name-map.csv` and `docs_local/migration/legacy-name-map.csv` map
every old name to its new one.

## Artifact taxonomy

One canonical name per artifact kind. `brief` and `prompt` are **not** interchangeable: the
canonical name is `prompt.md`.

| artifact | file | lives in |
|---|---|---|
| the exact instruction sent to an agent | `prompt.md` | `docs_local/agent_runs/<run id>/` |
| the conversational completion returned | `response.md` | `docs_local/agent_runs/<run id>/` |
| the formal evidence-bearing report | `report.md` | `scratchpad/<run id>/` |
| machine-readable provenance | `manifest.json` | `scratchpad/<run id>/` |
| execution artifacts | `scripts/ probes/ artifacts/ logs/ tests/` | `scratchpad/<run id>/` |
| concise project-level history | `round-NN.md` | `docs/rounds/` (tracked) |

When one run genuinely has several chronological prompts or responses, number them:
`prompt-01.md`, `response-01.md`. Do not number a single artifact.

A run whose round cannot be identified from the evidence is named **topically** rather than given
a guessed number — `scratchpad/footprint-ab`, `scratchpad/ds1-offline-reconciliation` — and the
index records why it has no round.

## Ownership rule

- prompt and conversational response → `docs_local/agent_runs/` (local-only, ignored)
- formal report, manifest, scripts, probes, artifacts, logs → `scratchpad/<run id>/` (ignored)
- concise conclusions, production state, strategy lifecycle → `docs/` and `configs/` (**tracked**)

Avoid duplication. If the same report exists in two places, the scratchpad copy is canonical and
the other should become a reference to it.

## Outputs

`outputs/<dataset>[-<suffix>]/`, one run per directory, non-default knobs appended as a suffix so
runs never clobber. **These names are load-bearing** — production source reads several by
hard-coded path. Do not rename one without updating every reference and re-running
`python src/build_ds1_member.py --verify`.
