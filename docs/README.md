# Documentation index

Start here.

| document | answers |
|---|---|
| [CURRENT_PRODUCTION.md](CURRENT_PRODUCTION.md) | What is live right now? Which hashes define it? How do I reproduce it? |
| [leaderboard_final.md](leaderboard_final.md) | The closing A-board result and this project's final placement |
| [STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md) | Every strategy ever tried, with its lifecycle status and evidence |
| [SUBMISSION_PROTOCOL.md](SUBMISSION_PROTOCOL.md) | How the platform actually scores and ingests a submission |
| [architecture/pipeline-overview.md](architecture/pipeline-overview.md) | How the two pipelines fit together and why they differ |
| [naming-standard.md](naming-standard.md) | Conventions for code, docs, strategy ids and run directories |
| [rounds/](rounds/) | Concise canonical history, one file per round |
| [maintenance/](maintenance/) | What the consolidation changed, and what it deliberately left alone |
| [data-b-runbook.md](data-b-runbook.md) | Procedure for when data package B is released |

Machine-readable companions, both tracked:

- [`../configs/production.json`](../configs/production.json) — the accepted state: scores,
  hashes, the ordered strategy chain, build and verify commands, archive policy. **Source of
  truth**; if a document disagrees with it, the JSON wins.
- [`strategy_inventory.json`](strategy_inventory.json) — 65 strategies with lifecycle status,
  offline and online evidence, implementation paths and prohibited variants, plus
  `late_round_evidence_index`. `STRATEGY_REGISTRY.md` is generated from it.

## Quick answers

**What is the score?** **1.576996059163449, rank #3, final**, shipped 2026-07-30. It is
`0.8963013747474597 (dataset1) + 0.6806946844159895 (dataset2)`, whose exact decimal sum is
`1.5769960591634492` — the platform displays the same quantity one digit shorter. The A board is
closed; this state is final. The superseded Round-22 state was 1.5752524794476366.

**Am I on the same state as everyone else?**

```bash
python -m unittest discover -s tests -t .      # 55 tests; local-data tests skip cleanly
python src/build_ds1_member.py --verify        # needs local artifacts
```

**What must I not change?** See the "must not be changed casually" section of
[CURRENT_PRODUCTION.md](CURRENT_PRODUCTION.md), and the prohibited-variants section at the end of
[STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md).

## What a collaborator receives, and what they do not

`docs_local/`, `scratchpad/` and `outputs/` are **intentionally git-ignored and always will be.**
A normal `git pull` does not deliver them, and **nothing in this documentation depends on having
them.** Every tracked document is written to be complete on its own; where it cites a local
artifact it does so as provenance, not as a prerequisite.

The production truth travels entirely through tracked paths:

| tracked path | what it carries |
|---|---|
| `configs/production.json` | the accepted state: scores, hashes, ordered chain, commands, archive policy |
| `src/` | every mechanism holding the accepted score **except the final dataset2 decoder** (see the caveat below), including the two frozen dataset1 postprocessors |
| `tools/` | component validation and submission packaging |
| `tests/` | 55 tests; those needing local artifacts skip with an explicit reason |
| `docs/` | production state, strategy lifecycle, submission protocol, round history |

A collaborator can therefore verify they are on the same strategy state, reproduce the dataset1
member byte for byte (given the raw data and the ranker output), and build a submission — without
possessing a single historical run directory.

> **One caveat, added at A-board closure.** The final dataset2 decoder
> (`xte_cross_time_exclusivity_decode`, worth +0.0017435797158127 online) is **not yet graduated into
> `src/`**; its implementation exists only in the git-ignored provenance tree. The shipped member bytes
> are hash-pinned in `configs/production.json`, so the final state is reproducible **by extraction**
> from the accepted archive but **not yet from tracked code**. This is the largest outstanding
> reproducibility gap and is recorded in
> [maintenance/repository-reorganisation-ambiguities.md](maintenance/repository-reorganisation-ambiguities.md).

The ignored trees hold **local provenance only**: the conversational record
(`docs_local/agent_runs/`, indexed by `docs_local/agent_runs_index.json`) and the execution record
(`scratchpad/`, indexed by `scratchpad/index.json`). Their 2026-07-29 renames and indexes were a
local hygiene improvement with **no effect on what is delivered**. See
[naming-standard.md](naming-standard.md) for the ownership rule and
[maintenance/repository-reorganisation.md](maintenance/repository-reorganisation.md) for the
record-keeping policy.
