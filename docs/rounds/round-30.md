# Round 30 — final A-board package: exact XTE asset graduation

**Outcome: shipped, and final. Total 1.5752524794476366 → 1.576996059163449.** 2026-07-30.
Visible rank **#3** (board: #1 1.5946, #2 1.5885).

This was not a discovery round. No model was trained, fitted or tuned; no feature was generated; no
threshold, blend or rank depth was touched; no CSV was regenerated. The round consisted of
identifying an existing online-validated asset, auditing it, and assembling it deterministically.

## What shipped

`outputs/submissions/round30_final/r30_xte_final.zip`, SHA256
`efe790a56a715ec451a809a560f25ac6d3ecb0cde1d71a46e4de7fcdb6a4a536`, 63,984,486 bytes.

Two members, both **copied byte-for-byte from archives whose contents had already been scored
online**:

| member | SHA256 | source | component |
|---|---|---|---|
| `dataset1.csv` | `baa0dc21e1f4b93e579b4c895a8a44da2064b6f3cbbc3d2d1be18314e1125987` | accepted Round-22 archive | 0.8963013747474597 |
| `dataset2.csv` | `beb13345dc020f32283cea2d132efa072eb52c73d29806f86f60fbf24982f971` | Round-23 XTE component archive | 0.6806946844159895 |

The final archive is 248 bytes **smaller** than the superseded Round-22 archive and 1.38 MB below
the size the platform previously rejected, so the standing size caution was satisfied without
recompressing anything.

## Why this asset

The Round-23 XTE decode was the only remaining mechanism in the project with a **measured positive
online observation** that had not been shipped. It had been closed at its locked `+0.002` shipping
gate (see [round-23.md](round-23.md)), and the strongest offline candidate the project ever produced
had just failed online in Round 29 (see [round-29.md](round-29.md)). With no further online reads to
protect and no alternatives, the final objective changed from "protect the read budget" to "maximise
the closing score with an existing validated asset".

That decision is recorded as an explicit, separately named operational event —
`FINAL_BOARD_MAXIMISATION_OVERRIDE` — and **not** as a revision of the Round-23 verdict. Both
records stand: the gate decided correctly under its own objective, and the override decided
correctly under a different one.

## Score contract

| item | value | evidence class |
|---|---|---|
| dataset1 component | `0.8963013747474597` | `ONLINE_OBSERVED / ACCEPTED` |
| dataset2 component | `0.6806946844159895` | `ONLINE_OBSERVED` |
| exact decimal sum | `1.5769960591634492` | `DERIVED_FROM_ONLINE_OBSERVED_COMPONENTS` |
| platform-observed total | **`1.576996059163449`** | `ONLINE_OBSERVED` |
| improvement | **`+0.0017435797158124`** | `ONLINE_OBSERVED` |

The prediction held. Round 30 is the only case in the project where a combined total was projected
from two separately observed components **before** upload and then confirmed, which retires the
standing limitation that additivity had never been verified for an untested pairing.

## Integrity

The archive was built **twice by two independent processes** and the results were byte-identical —
same size, same SHA256, same member ordering, same member metadata. Determinism is structural: local
headers carry a fixed `1980-01-01` timestamp, so no wall clock enters the container. The promoted
archive was then reopened and re-extracted independently and every member hash re-verified against
its source archive.

46 of 46 integrity checks passed, covering member identity, byte-for-byte provenance, schema and row
counts (61,051 x 100 and 153,420 x 100, no malformed row, no duplicate, no NaN or infinity), exact
decimal score arithmetic, container structure, and the exclusion of every Round-29 byte-stream.

Two checks failed on a first pass and both were defects in the assertions rather than the package: one
demanded uniform LF line endings (`dataset1.csv` is legitimately CRLF and `dataset2.csv` LF — each is
the exact form already scored, and neither may be normalised), and one demanded that the accepted
total equal the exact decimal sum of its components when the locked anchor is the float sum. The
package hash never changed.

## Lifecycle

| axis | value |
|---|---|
| historical scientific lifecycle | `CLOSED_AT_LOCKED_GATE` — retained, not erased |
| evidence status | `ONLINE_VALIDATED` |
| operational lifecycle | `SHIPPED_ACTIVE` |
| package status | `ONLINE_OBSERVED_FINAL` |
| Round-22 production | `SHIPPED_SUPERSEDED` |

Release tag: `a-board-r30-1.576996059163449`.

## Outstanding

The decoder that produces the shipped `dataset2.csv` is **not yet tracked under `src/`**. Production
is reproducible by extraction from the hash-pinned archive but not yet from tracked code. This is the
largest open reproducibility item and is recorded in
[../maintenance/repository-reorganisation-ambiguities.md](../maintenance/repository-reorganisation-ambiguities.md).

Local provenance: `scratchpad/round-30-opus/` (`finalisation_audit.md`, `graduation_record.md`,
`online_outcome_addendum.md`, `manifest.json`).
