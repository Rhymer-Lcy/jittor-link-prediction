# Final A-board leaderboard

Closing state of the A board. **Not a scientific score ledger** — nothing here is an experimental
result.

Evidence class: **`USER_OBSERVED_SCREENSHOT`**. Source: a leaderboard screenshot reported by the
project owner. It is an approximately final pre-closure observation, **not an independently fetched
platform record**. The screenshot exposes no official observation timestamp, so **none is recorded**;
this file's own commit time must not be read as one.

A-board closure: 2026-07-30 12:00 (UTC+8).

---

## Snapshot

| Rank | Display name | Institution | Displayed score |
|---:|---|---|---:|
| 1 | 杨乾坤 | 上海交通大学 | 1.5946 |
| 2 | 宋传承 | 中国科学院信息工程研究所 | 1.5885 |
| 3 | [public contact redacted] | 国防科技大学 | 1.577 |

## This project's final position

| item | value |
|---|---|
| final rank | **3** |
| displayed score | `1.577` |
| platform-observed total | **`1.576996059163449`** |
| component-derived exact decimal sum | `1.5769960591634492` |
| dataset1 component | `0.8963013747474597` |
| dataset2 component | `0.6806946844159895` |
| shipped archive | `outputs/submissions/round30_final/r30_xte_final.zip` |
| archive SHA256 | `efe790a56a715ec451a809a560f25ac6d3ecb0cde1d71a46e4de7fcdb6a4a536` |
| displayed gap to rank 2 | approximately `0.0115` |
| displayed gap to rank 1 | approximately `0.0176` |

## Movement over the final day

| state | total | rank | note |
|---|---:|---:|---|
| Round-22 accepted | `1.5752524794476366` | 3 | `SHIPPED_SUPERSEDED`, tag `a-board-r22-1.5752524794476366` |
| Round-29 submitted | `1.5494486598552333` | — | online failure, `CLOSED_ONLINE_FAILURE`; not left in place |
| Round-30 final | **`1.576996059163449`** | **3** | `SHIPPED_ACTIVE`, tag `a-board-r30-1.576996059163449` |

Improvement of the final state over the superseded Round-22 total: **`+0.0017435797158124`**,
delivered entirely by the dataset2 component (`+0.0017435797158127`) with dataset1 held
byte-for-byte identical.

The final graduation changed the score but not the placement: the project held rank 3 before and
after. A `+0.0017` gain does not close a displayed gap of `0.0115`.

## Reading notes

- Displayed values are **as shown on the board**, at the board's own rounding. Only this project's
  own total is known exactly; ranks 1 and 2 are known to four decimal places only, so the true gaps
  are bounded by the board's rounding and are not determinable from this evidence. The quoted gaps
  are therefore displayed-score differences and are labelled "approximately".
- The rank-3 displayed `1.577` is consistent with this project's observed total rounded to three
  decimals. This record does not assert, beyond that consistency, which submission the board is
  displaying.
- The exact decimal component sum ends in a digit the platform's floating-point display drops. Both
  forms are preserved and neither is reconciled into the other: `1.5769960591634492` is the
  arithmetic, `1.576996059163449` is the observation.
- This is a point-in-time observation of a live board and carries no claim about its state at any
  other moment.

## Related records

- Accepted production state: [CURRENT_PRODUCTION.md](CURRENT_PRODUCTION.md) and
  [`../configs/production.json`](../configs/production.json).
- How the final package was assembled and verified: [rounds/round-30.md](rounds/round-30.md).
- Why the Round-23 gate and the final-board override are both in the record:
  [rounds/round-23.md](rounds/round-23.md).
- The failure that made Round 30 necessary: [rounds/round-29.md](rounds/round-29.md).
