# Round 26 — six candidates, no online candidate, one validated control

**Outcome: ROUND 26 CLOSED — NO ONLINE CANDIDATE.** Both agent threads independently recommended
`NO CANDIDATE`. Nothing was built, nothing was submitted.

The round's lasting product is not a candidate but a **control** and the evaluation rule that came
with it, which changed how Round 27 and everything after it had to be measured.

## Codex thread — three closures

| strategy | dataset1 delta | dataset2 delta | lifecycle |
|---|---:|---:|---|
| `role_aware_conditional_interaction_lift` | `-0.005505920004491764` | `-0.021595453502474844` | `CLOSED` |
| `head_to_head_candidate_competition` | `-0.0002526617210474232` | `-0.0014013177303569672` | `CLOSED` |
| `role_conditioned_winner_flow_residual` | `-0.004305776829516504` | `-0.025996082866227423` | `CLOSED` |

All three were negative on both datasets, both source-disjoint folds, both time halves **and** the
production-matched in-pool slices. The mechanism nulls and matched placebos show that much of the
movement is generic top-pair intervention cost rather than recoverable structural information — and a
null share cannot rescue a treatment that is negative before attribution.

`head_to_head_candidate_competition` was additionally physically sparse: 70 dataset1 and 637 dataset2
actions.

Each closure binds the exact tested construction at the deterministic-overlay rung, not its family.

## Fable thread — one closure, one standby, one rung closure

| strategy | OOF delta | lifecycle |
|---|---:|---|
| `directed_role_svd_affinity` | `+0.000075` (folds `-0.000533` / `+0.000679`) | `CLOSED` |
| `line_direct_candidate_affinity` | **`+0.004103`** | **`STANDBY`** |
| `slate_confidence_context` | `+0.000965` (null `-0.000378`) | `CLOSED` |

`directed_role_svd_affinity` was offline dead at the treatment stage with sign-unstable folds; its
null battery was never run to adjudication. The direction axis on dataset1 is now triply dead
(reverse-edge features, directed three-hop, directed factorisation), while role-factorised
representations remain production-validated on dataset2 via the shipped basket geometry.

`slate_confidence_context` is real and null-clean but below the minimum-interesting offline floor.
Its `+0.000965` is now the recorded floor any deeper gating or confidence-aware architecture must beat
before claiming mechanism value.

## The validated control

`line_direct_candidate_affinity` — the direct LINE source-candidate cosine, one deterministic column
appended to the accepted 21-column dataset1 ranker.

| property | value |
|---|---|
| control MRR (250 trees, paired) | `0.663812` |
| treatment MRR | `0.667915` |
| **paired OOF delta** | **`+0.004103`** |
| folds | `+0.003754` / `+0.00445` |
| time halves | `+0.006284` / `+0.001921` |
| non-repeat / cold-truth slices | `+0.009062` / `+0.033309` |
| repeat slice | `-0.00083` (flat; history features already dominate) |
| equal-capacity control | `-0.000747` (**0% share**) |
| slate-permutation semantic null | `-0.000671` (**0% share**) |
| physical coverage | 100% of 61,051 rows / 6,105,100 cells |
| conservative projected online gain | approximately `+0.00045` |

This is the strongest capacity-clean and null-clean offline gain since the ranker itself, and it was
**deliberately not promoted**: the projection sits below the `+0.002` learned gate and below the
`+0.001` clean-lightweight gate, and within reach of the observed online refit noise floor. The
Round-23 XTE precedent settled it — an online-*positive* `+0.0017435797158127` had already been closed
against its locked gate, so a sub-gate candidate does not justify spending a read.

It was recorded as `STANDBY` / data_B evidence. Being the sole survivor of a round is not evidence of
online economics.

## The rule this round produced

Any later treatment that uses LINE-direct must be measured against **both** the accepted control
**and** that control plus the validated LINE-direct scalar, paired on identical folds, seed, tree count
and harness. Reporting only against the bare accepted control is not admissible, because the bare
control no longer represents the best known cheap configuration.

Round 27 applied this immediately, and it mattered: all three of its model families read as
`+0.004`-class findings against the bare control and collapsed to approximately zero against the
control that already contained LINE-direct. See [round-27.md](round-27.md).

This sits alongside the two earlier standing rules — the Round-24 rank-tie/position-leak audit and the
Round-25 null-preservation principle — and none replaces another.

Local provenance: `scratchpad/round-26-codex/`, `scratchpad/round-26-fable/`,
`scratchpad/round-26-opus/adjudication.md`, `scratchpad/round-26-opus/round27_handoff.md`.
