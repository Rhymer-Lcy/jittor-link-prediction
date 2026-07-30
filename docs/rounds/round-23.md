# Round 23 — dataset2 cross-time exclusivity decode (XTE): closed at the locked gate

**Outcome at the time: CLOSED_AT_LOCKED_GATE.** 2026-07-29. Online-positive and still not shipped.

**Outcome seven rounds later: shipped as the final A-board state**, under an explicit operational
override. Both facts belong to the record; see [round-30.md](round-30.md).

## The mechanism

`xte_cross_time_exclusivity_decode`. A measured generator invariant: for a fixed source and candidate
answer, the answer does not recur at two distinct timestamps — **0 of 2,264,807 dataset2 train pairs
and 0 of 244,056 replay truths**. The accepted member nevertheless served the same candidate as top-1
for the same source at two or more distinct times, which the invariant says cannot all be right.

The frozen rule, one pass and no thresholds. With `c1` and `c2` the stable descending rank-1 and
rank-2 candidates of a row and `margin = score(c1) - score(c2)`:

1. group rows by `(source, c1)`;
2. a group is eligible only if its rows span at least two distinct timestamps;
3. partition an eligible group into exact timestamp clusters;
4. keep the cluster whose maximum row margin is largest, ties to the smallest timestamp;
5. in every row of every other cluster, swap the two existing score values at `c1` and `c2`.

Nothing else changes. The treatment CSV is produced by swapping the two score **tokens** in the
accepted member's own bytes — unaffected rows are copied verbatim and no score is ever re-serialised
from a float.

| structural measurement | value |
|---|---:|
| rows / candidate columns | 153,420 / 100 |
| distinct sources | 2,180 |
| `(source, top-1)` groups | 142,693 |
| groups spanning >= 2 timestamps | 6,561 |
| rows in violation groups | 14,555 |
| **rows acted on** | **7,815** (5.09%) |
| cells changed / top-1 changes / strict inversions | 15,630 / 7,815 / 7,815 |
| pairs collapsed to a tie | 0 |
| violation groups remaining after one pass | 1,589 |

## The evidence, and the gate

| quantity | value | evidence class |
|---|---:|---|
| offline replay delta | `+0.02015932103723213` | `OFFLINE_REPLAY` |
| projected component gain | `+0.0062` to `+0.0069` | `PROJECTED` |
| accepted dataset2 control | `0.6789511047001768` | `ONLINE_OBSERVED` |
| **observed dataset2 score** | **`0.6806946844159895`** | `ONLINE_OBSERVED` |
| **observed gain** | **`+0.0017435797158127`** | `ONLINE_OBSERVED` |
| locked shipping gate | `+0.002` | pre-registered |
| score required under the gate | `0.6809511047001768` | — |
| **shortfall** | **`0.0002564202841873`** | — |

The realised gain was **3.6x to 4.0x smaller** than the projection, and the replay was not
byte-equivalent to the accepted chain. The mechanism was real and the direction was right; the
magnitude was not what the replay predicted.

**It was closed.** An online-positive result was rejected because it missed a gate that had been
locked before the read. That decision is what made the gate credible for the seven rounds that
followed, and it is repeatedly cited in later rounds as the reason not to spend a read on a sub-gate
candidate.

## Why it later shipped anyway

By Round 30 the situation had changed in three ways: the A board was closing, no further online reads
needed protecting, and the strongest offline candidate the project ever produced had just failed
online (see [round-29.md](round-29.md)). The objective became closing-score maximisation using an
existing validated asset.

The exact member bytes from this round — SHA256
`beb13345dc020f32283cea2d132efa072eb52c73d29806f86f60fbf24982f971` — were assembled with the accepted
dataset1 member and shipped. The gain reappeared exactly: `+0.0017435797158127` on the component, and
`+0.0017435797158124` on the observed total.

**XTE never passed the `+0.002` gate.** Its lifecycle carries two axes:

| axis | value |
|---|---|
| historical scientific lifecycle | `CLOSED_AT_LOCKED_GATE` |
| evidence status | `ONLINE_VALIDATED` |
| operational decision | `FINAL_BOARD_MAXIMISATION_OVERRIDE` |
| operational lifecycle | `SHIPPED_ACTIVE` |

The override is scoped to the exact online-observed member bytes. It does not reopen the family for
tuning, rescue or reconstruction, and it does not convert a sub-gate result into a gate pass.

## Outstanding

The decoder is **not yet tracked under `src/`**; its implementation lives only in the git-ignored
provenance tree. See [../CURRENT_PRODUCTION.md](../CURRENT_PRODUCTION.md) and
[../maintenance/repository-reorganisation-ambiguities.md](../maintenance/repository-reorganisation-ambiguities.md).

Local provenance: `scratchpad/round-23-opus/xte/`,
`outputs/submissions/round23_components/r23_xte_d2_swap_ds2_manifest.json`.
