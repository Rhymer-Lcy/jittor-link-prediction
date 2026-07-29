# Documentation index

Start here.

| document | answers |
|---|---|
| [CURRENT_PRODUCTION.md](CURRENT_PRODUCTION.md) | What is live right now? Which hashes define it? How do I reproduce it? |
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
- [`strategy_inventory.json`](strategy_inventory.json) — 53 strategies with lifecycle status,
  offline and online evidence, implementation paths and prohibited variants.
  `STRATEGY_REGISTRY.md` is generated from it.

## Quick answers

**What is the score?** 1.5752524794476366, rank #3, accepted 2026-07-29. It is exactly
`0.8963013747474597 (dataset1) + 0.6789511047001768 (dataset2)`.

**Am I on the same state as everyone else?**

```bash
python -m unittest discover -s tests -t .      # 55 tests; local-data tests skip cleanly
python src/build_ds1_member.py --verify        # needs local artifacts
```

**What must I not change?** See the "must not be changed casually" section of
[CURRENT_PRODUCTION.md](CURRENT_PRODUCTION.md), and the prohibited-variants section at the end of
[STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md).

**Where do local, untracked working records live?** Prompts and conversational responses in
`docs_local/agent_runs/` (indexed by `docs_local/agent_runs_index.json`); formal reports,
manifests, scripts and evidence in `scratchpad/` (indexed by `scratchpad/index.json`). Both trees
are git-ignored by design — see [naming-standard.md](naming-standard.md) for the ownership rule.
