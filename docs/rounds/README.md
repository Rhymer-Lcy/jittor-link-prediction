# Round history

Concise canonical history, one file per round. Each file records what was tested, the exact measured
evidence, the lifecycle decision and the closure scope.

| round | outcome | file |
|---|---|---|
| 21 | shipped — dataset1 source-slate recurrence (R2) | [round-21.md](round-21.md) |
| 22 | shipped — dataset1 rank-2 test-graph reciprocity (RGR); total 1.5752524794476366 | [round-22.md](round-22.md) |
| 23 | online-positive, **closed at the locked +0.002 gate** — dataset2 cross-time exclusivity decode (XTE) | [round-23.md](round-23.md) |
| 26 | no online candidate; produced the validated LINE-direct control and its two-control rule | [round-26.md](round-26.md) |
| 27 | no candidate — three model-interface families closed at their tested rungs | [round-27.md](round-27.md) |
| 28 | no candidate — largest offline gain in the project, refuted by its own controls | [round-28.md](round-28.md) |
| 29 | **online failure** — strongest offline evidence in the project inverted online | [round-29.md](round-29.md) |
| 30 | shipped, final — exact XTE asset graduation; total 1.576996059163449, rank 3 | [round-30.md](round-30.md) |

Final board snapshot: [../leaderboard_final.md](../leaderboard_final.md).

## Coverage

Rounds before 21 and rounds 24-25 do not have files here.

- **Rounds before 21.** Their outcomes are carried in the score ledger and closed-axes sections of the
  root [`README.md`](../../README.md) and in the per-strategy records of
  [`../strategy_inventory.json`](../strategy_inventory.json). They predate this one-file-per-round
  convention and were never migrated.
- **Rounds 24 and 25.** Both were adjudicated in full at the time, and both closed with no candidate.
  Round 24 produced the rank-tie / position-leak validation rule and Round 25 produced the
  null-preservation principle; both rules are still in force and are described where they are applied,
  in [round-26.md](round-26.md) and [round-27.md](round-27.md). Their adjudications live in the
  git-ignored local provenance tree (`scratchpad/round-24-opus/`, `scratchpad/round-25-opus/`), and
  their per-strategy records were **not** migrated into the tracked inventory.

The second gap is recorded deliberately rather than papered over. No production claim depends on those
two rounds: every candidate in both was closed and none shipped. The machine-readable index of which
late rounds have migrated strategy records is `late_round_evidence_index` in
[`../strategy_inventory.json`](../strategy_inventory.json).

## Reading the lifecycle vocabulary

Lifecycle and evidence terms are defined in
[`../../src/strategies/registry.py`](../../src/strategies/registry.py) and used consistently across
these files. The two that matter most for the late rounds:

- **`CLOSED_AT_LOCKED_GATE`** — the mechanism worked online but missed a threshold that was fixed
  before the read. Round 23 is the only instance.
- **`CLOSED_ONLINE_FAILURE`** — the mechanism passed every offline gate, was submitted, and lost
  score. Round 29 is the only instance.

One strategy carries two lifecycle axes at once. `xte_cross_time_exclusivity_decode` is historically
`CLOSED_AT_LOCKED_GATE` and operationally `SHIPPED_ACTIVE`, because the final board was maximised with
an asset that had been correctly rejected under a different objective. Both records stand; neither
cancels the other.
