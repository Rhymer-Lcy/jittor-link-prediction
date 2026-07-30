# Round 27 — three model-interface families, all closed at their tested rung

**Outcome: NO CANDIDATE. Nothing was built, nothing was submitted.** The full out-of-fold stage was
never entered (`SKIPPED_NO_SCREENING_SURVIVOR`).

Round 27 was the first round to run under the Round-26 two-control requirement, and it is the
strongest available demonstration that the requirement changes conclusions.

## The control protocol

| arm | full-replay MRR | screening MRR |
|---|---:|---:|
| control A — accepted production-aligned | `0.6638121262583483` | `0.6654911694330767` |
| control B — control A **plus the validated LINE-direct scalar** | `0.6679147193356217` | `0.6700460196448429` |

The LINE-direct anchor delta is `0.00410259307727342`, reproducing the Round-26 finding. Every
family's primary increment is measured against **control B**, so the double-counting prohibition was
enforced structurally rather than asserted.

## The three families

Deltas are incremental to control B.

| strategy | increment over control B | against control A alone | folds | lifecycle |
|---|---:|---:|---|---|
| `setwise_deepsets_residual` | **`-0.00021168145246953707`** | `+0.004343168759296733` | `-0.00068` / `+0.00024` | `CLOSED_AT_TESTED_DEEPSETS_RUNG` |
| `causal_event_sequence_ranker` | **`+0.00011493543092999476`** | `+0.004669785642696264` | `+0.00029` / `-0.00006` | `CLOSED_AT_TESTED_GRU_RUNG` |
| `objective_low_rank_adapter` | **`-0.00011367610365886334`** | `+0.004441174108107407` | `+0.00024` / `-0.00045` | `CLOSED_AT_TESTED_RANK8_ADAPTER_RUNG` |

**All three look like `+0.004`-class findings against control A and collapse to approximately zero
against control B.** Every apparent gain was the LINE-direct scalar that had already been measured
and recorded in Round 26. Reported against control A alone, this round would have produced three
false positives.

Source-clustered bootstrap over 2,048 sources, 1,000 draws: 95% intervals
`[-0.001435, +0.000867]`, `[-0.000578, +0.000804]`, `[-0.000100, +0.000980]`. Every interval straddles
zero.

### The decisive result: sequence information is empty at this rung

`causal_event_sequence_ranker` is the only family with a positive increment, and its own controls
refute it:

| arm | increment over control B | share of treatment gain |
|---|---:|---:|
| treatment (bounded event GRU, `H <= 64`, 18,513 parameters) | `+0.00011493543092999476` | 100% |
| matched `mean_pool_mlp` equal-capacity control | `+0.0002856881228210235` | **248.6%** |
| `zero_history` null | `+0.000185471820092053` | **161.4%** |
| `event_order_shuffle` null | `+0.00011674618912374544` | 101.6% |
| `timestamp_gap_removal` null | `+0.00011341339871122085` | 98.7% |

A mean pool with no sequence at all does better than the GRU; removing the history entirely still
reproduces most of the effect; and shuffling event order or deleting timestamp gaps changes nothing
measurable. **Event order and event timing carry no measurable ranking information at this rung.**
99.43% of dataset1 rows carry a non-empty history, so this is not a coverage artefact.

## Contracts

All three families: `METHOD_CONTRACT_PASS` on A/B consistency with no production candidate;
`METHOD_GENERAL` for non-bipartite and strictly bipartite role semantics with the **dataset2 gain
explicitly unmeasured** rather than projected; 100% physical coverage (61,051 rows, 6,105,100 cells);
exact-pass on the Round-24 candidate-permutation and truth-cell exchangeability audits (reciprocal-rank
maximum absolute deviation `0.0` over 64 queries). GPU peaks 43.5 MB, 264.2 MB, 216.0 MB; runtimes
14.1 s, 12.9 s, 9.9 s — all inside the single-card serial bound.

## Closure scopes

Each closure binds its tested rung, not its family:

- **DeepSets** — a residual over scalar inputs only. Does not close set-wise modelling that consumes
  candidate *representations* rather than candidate *scalars*.
- **Causal event GRU** — bounded event identities and times. Does not close deeper-history sequence
  modelling, but it does establish that the cheap version of the hypothesis is empty.
- **Rank-8 adapter** — the general design hypothesis that each embedding objective should contribute
  its own direct score is not refuted; what is refuted is that a supervised low-rank adapter extracts
  more once those direct scores are already present.

## Carried forward

The control-A/control-B divergence in this round should be treated as permanent protocol: a candidate
that shares an information source with an already-recorded finding must be measured against a control
that already contains it.

Local provenance: `scratchpad/round-27-codex/`, `scratchpad/round-27-opus/adjudication.md`.
