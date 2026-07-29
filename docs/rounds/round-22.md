# Round 22 — dataset1 rank-2 test-graph reciprocity (RGR)

**Outcome: shipped. Total 1.567242782636773 → 1.5752524794476366.** 2026-07-29.
Visible rank **#3** — the board moved up while we did (#1 1.5885, #2 1.5818).

## What shipped

`test_graph_reciprocity` — a second deterministic dataset1 postprocessor, applied on top of
`source_slate_recurrence`.

The physical test batch defines a directed exposure graph, `A(u, c) = 1` iff candidate `c`
appears in a test slate whose source is `u`. With `c1`/`c2` the stable score ranks 1 and 2, act
iff **all four** hold: `c1` non-historical, `c2` non-historical, `A(c2, source) = 1`,
`A(c1, source) = 0`. Then promote `c2` to strict top-1. Acts on **1,365 / 61,051 rows (2.24%)**.

| evidence | value |
|---|---|
| offline replay | +0.0026108377841567065 over 142,483 queries (1,604 repaired / 860 damaged) |
| isolated auxiliary read | 0.8963013747474597 vs baseline 0.8882916779365962 |
| online delta | **+0.0080096968108635** against a locked +0.0010 gate |
| accepted components | ds1 0.8963013747474597 + ds2 0.6789511047001768 |

The naive per-action transfer projected ≈ +0.00186; the online result was **4.3× that** — the
same direction of surprise as R2 (5× its replay). The replay keeps calling the sign correctly on
this family while badly understating the magnitude.

Origin: Codex round 22 rank 1. Executed in `scratchpad/round22_opus/rgr/`. Now tracked at
`src/strategies/ds1/test_graph_reciprocity.py`.

## Not executed

- **SPC (candidate source-peak crossing)** and **SRC (candidate source-role crossing)** — Codex
  ranks 2 and 3, held by instruction. Atlas evidence only; recorded as `PROBE`.
- **Fable round 22F returned NO CANDIDATE.** Its contribution is three failure mechanisms worth
  keeping: deep-rescue asymmetry under strict-top-1 promotion, replay-construction prevalence
  inflation, and the answer-sharing illusion.
- **XLIST** remains `STANDBY`.

## Infrastructure

A component validation and packaging harness was built first (`scratchpad/round22_opus/harness/`,
35 self-tests, three dry runs) and has since been graduated to
[`tools/submission/package_component.py`](../../tools/submission/package_component.py). It takes a
treatment CSV to a validated ZIP plus a machine-readable manifest in seconds, and closes the
component-score arithmetic after the leaderboard answers.

Two platform properties were measured and are now encoded there: the total is strictly additive,
and a dataset absent from the ZIP scores **0** rather than being carried over.

## Consolidation (same day, after acceptance)

The repository was consolidated: both shipped dataset1 postprocessors were graduated into tracked
production code and now **reproduce the accepted `dataset1.csv` byte for byte** from the ranker
output (`python src/build_ds1_member.py --verify`); the MF dataset2 pipeline was graduated
verbatim out of the scratchpad; the strategy lifecycle was reconstructed into
[`docs/strategy_inventory.json`](../strategy_inventory.json) (53 strategies).

That pass also **refuted the Round-21 archive-size cliff** — see
[SUBMISSION_PROTOCOL.md](../SUBMISSION_PROTOCOL.md).

Full working record: `scratchpad/round22_opus/rgr/{report.md,manifest.json}` and
`scratchpad/round22_opus/harness/`.
