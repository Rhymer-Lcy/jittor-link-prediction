# Current production state

**Final A-board score 1.576996059163449 — visible rank #3**, shipped 2026-07-30. This is the
closing state of the competition; see [leaderboard_final.md](leaderboard_final.md) for the board
snapshot and [rounds/round-30.md](rounds/round-30.md) for how it was assembled.

Machine-readable source of truth: [`configs/production.json`](../configs/production.json).
Everything below is derived from it; if the two disagree, the JSON wins and this file is stale.

## What defines production

| item | value |
|---|---|
| accepted archive | `outputs/submissions/round30_final/r30_xte_final.zip` |
| archive SHA256 | `efe790a56a715ec451a809a560f25ac6d3ecb0cde1d71a46e4de7fcdb6a4a536` |
| archive bytes | 63,984,486 (deflate method 8, level 9) |
| `dataset1.csv` SHA256 | `baa0dc21e1f4b93e579b4c895a8a44da2064b6f3cbbc3d2d1be18314e1125987` |
| `dataset2.csv` SHA256 | `beb13345dc020f32283cea2d132efa072eb52c73d29806f86f60fbf24982f971` |
| dataset1 component | 0.8963013747474597 |
| dataset2 component | 0.6806946844159895 |

The platform total is **strictly additive**. The exact decimal component sum is
`1.5769960591634492`; the platform displays the same quantity as `1.576996059163449`. Both forms
are kept: the first is the arithmetic, the second is the observation. Each component was measured
directly by uploading a single-member ZIP to the auxiliary account, and Round 30 is the one case
where a combined total was predicted from two separately observed components and then confirmed
online.

The archive contains **existing verified member bytes only** — `dataset1.csv` copied byte-for-byte
from the superseded Round-22 archive and `dataset2.csv` copied byte-for-byte from the
online-observed Round-23 XTE component. Neither CSV was regenerated. Note that the two members
carry different line endings (`dataset1.csv` CRLF, `dataset2.csv` LF); each is in the exact form
that was scored online, and neither may be normalised.

## Superseded production

| item | value |
|---|---|
| archive | `outputs/submissions/round22_main/r22_rgr_main_d9.zip` |
| archive SHA256 | `6b9f0cd928c49ec2b6f48554c5d3ef2cfae6dbc34650dd17f19c03105dd9dae7` |
| total | 1.5752524794476366 |
| lifecycle | `SHIPPED_SUPERSEDED` |
| tag | `a-board-r22-1.5752524794476366` |

It held production from 2026-07-29 until the Round-30 graduation, is retained on disk, is
immutable, and remains the fallback state.

## The active strategy chain

### dataset1 — 0.8963013747474597

```
src/train_line.py  +  src/train_bpr.py            embeddings (10-seed BPR + innovation BPR w=12)
        |
src/ranker_ds1.py                                 cut-split LambdaRank, 21 columns
        |                                         -> outputs/dataset1-ensemble/result_ranker.csv
        |                                            sha256 cb4964ea...  component 0.8619654323294592
        |
src/strategies/ds1/source_slate_recurrence.py     3,653 actions   online +0.0263262456071370
        |
src/strategies/ds1/test_graph_reciprocity.py      1,365 actions   online +0.0080096968108635
        |
        -> dataset1.csv   component 0.8963013747474597
```

**dataset1 score-domain contract.** LambdaRank margins are unbounded and mostly negative, so
`src/ranker_ds1.py` serialises through a **per-row min-max**: every row of `result_ranker.csv` has
minimum exactly 0 and maximum exactly 1. The step is order-preserving, so it has no ranking or MRR
effect, but it is load-bearing — both ds1 postprocessors and the final output gate assume the
`[0, 1]` domain, and `frozen_ops.promoted_value` is "exactly 3.0" only when a row's range is 1.

**The chain is order-sensitive.** Reciprocity reads the rank-2 candidate of the
recurrence-adjusted matrix; swapping the two produces a different file. The order above is the
one that was adjudicated online and accepted.

Reproduce and prove it:

```bash
python src/build_ds1_member.py --verify
```

That re-runs both postprocessors over the ranker CSV and asserts the SHA256 of every stage
against `configs/production.json`. It currently reproduces the accepted member **exactly**
(and the intermediate stage reproduces the previously accepted Round-21 member exactly).

### dataset2 — 0.6806946844159895

```
src/train_line.py + src/train_bpr.py              embeddings (5-seed BPR)
        |
src/ranker_basket_ds2.py                          18-feature LambdaRank + 3-pass basket feedback
src/ds2_mf_basket_pack.py                         pass-2/3 sibling-message geometry: d128 SVD of
  (uses src/ds2_basket_featurizer.py)             split0 src x dst, replacing item_profiles
        |
src/crf_promote.py                                equality CRF tau=0.20 B=70
  --zr-exclude --demote-dups --st-exclude --no-pair  + triple / zero-repeat / same-time invariants
        |
        -> base matrix    component 0.6789511047001768   (held production 2026-07-28 to 2026-07-29)
        |                 sha256 dc6928c2...  hash-pinned as dataset2.decoder_input
        |
src/strategies/ds2/cross_time_exclusivity.py      cross-time answer-exclusivity decode
                                                  7,815 rows / 15,630 cells swapped
        |
        -> dataset2.csv   component 0.6806946844159895
```

The base chain through `crf_promote.py` was **byte-identical from 2026-07-28 to 2026-07-29**. The
final member adds one deterministic decode on top of it, described below.

#### The final decoder

Group rows by `(source, served top-1)`. A group spanning two or more distinct timestamps
contradicts the measured zero-repeat generator invariant — for a fixed source and candidate answer,
the answer does not recur at two distinct timestamps (0 of 2,264,807 dataset2 train pairs). Keep
the timestamp cluster whose maximum row margin is largest, ties to the smallest timestamp, and swap
the two existing score tokens at rank 1 and rank 2 in every row of every other cluster. One pass,
no thresholds, no refit. It changes 7,815 of 153,420 rows (5.09%), creates no ties, and moves the
component by **+0.0017435797158127** online.

Reproduce and prove it:

```bash
python src/build_ds2_member.py --verify
```

That re-derives the action set from the hash-pinned base matrix and the test file, asserts all twelve
physical census anchors and the four effect counts, and asserts the member SHA256. It currently
reproduces the accepted member **exactly**. The rebuild swaps the two affected score *tokens* in the
base member's own bytes, so the other 15,326,370 cells are the exact bytes that were scored online;
no score is re-serialised from a float.

**Graduated 2026-07-30**, closing what was ambiguity A8. Until then the decoder existed only in the
git-ignored provenance tree at `scratchpad/round-23-opus/xte/scripts/build_xte_member.py` and
production was reproducible only by extraction.

> **Reproduction status — raw-to-final EXECUTED; byte identity NOT claimed.** Two statements that
> must not be merged into one:
>
> 1. **The chain runs from raw data to a final member.** On 2026-08-01 the complete Dataset-2 chain
>    was executed on the target host from the official raw data: `ds2_mf_basket_pack.py` produced
>    the MF base, `crf_promote.py` the CRF container and `build_ds2_member.py` the member, each
>    validated on exit. The Dataset-1 chain was executed the same way on 2026-08-02. This closes
>    ambiguity A1, which recorded that the modules had been graduated verbatim but never re-run.
> 2. **A rerun is not guaranteed to be byte-identical to the accepted member.** Embedding training
>    is stochastic and the reconstruction differs from the historical producer in the random stream,
>    the optimiser trajectory and the virtual-edge path. The measured divergence and its causes are
>    in the cross-version audit; the accepted member remains reproducible **by extraction** from the
>    hash-pinned archive, and the decoder stage reproduces it byte for byte from that base.
>
> See [maintenance/repository-reorganisation-ambiguities.md](maintenance/repository-reorganisation-ambiguities.md).

## What must not be changed casually

- **The accepted archive and both accepted CSV members are immutable.** Tools open them
  read-only and assert their hashes before doing anything.
- **The frozen postprocessor rules.** Thresholds, uniqueness requirements, row gates, the history
  definition and the score transform are all frozen by an online adjudication. Their prohibited
  variants are listed per strategy in [STRATEGY_REGISTRY.md](STRATEGY_REGISTRY.md).
- **Candidate column position is never evidence** on either dataset (measured: z = +0.07).
- **The raw dataset2 test row order is load-bearing** — never sort before applying `crf_promote`.
- **The pair rule stays off** (`--no-pair`): it double-counts the chain the equality CRF models.

## How a collaborator confirms they are on the same state

```bash
git rev-parse HEAD
python main.py --stage describe                      # every stage, implementation and status
python -m unittest discover -s tests -t .            # data-dependent tests skip cleanly
python main.py --stage postprocess --verify          # rebuild both members and assert their hashes
python tools/submission/package_component.py verify \
    --zip outputs/submissions/round30_final/r30_xte_final.zip
```

The verify command prints both member hashes. `dataset1.csv` reports `IS the accepted member`;
`dataset2.csv` reports as changed against the pre-XTE baseline, which is the expected and intended
result of the final graduation. If the local artifacts are absent, the integration tests skip with
an explicit reason and the config-level tests still assert that the manifest, the registry and the
inventory agree.

## Building a new submission

The A board is closed and this state is final. The protocol below is retained for data package B.

See [SUBMISSION_PROTOCOL.md](SUBMISSION_PROTOCOL.md). In short: measure one component in
isolation on the auxiliary account with a single-member ZIP, and only merge it into a main pack
with the frozen other member after it has passed online.

One correction the final rounds forced on that protocol: an offline replay gain is not evidence of
an online gain when the replay is not a faithful image of the physical serving chain. Round 29
carried the strongest offline evidence in the project — a bit-identical reconstruction, an
independent second seed, clean nulls and a strictly positive source-clustered interval — and
inverted online, from a predicted +0.0020876888876181 to an observed -0.0258038195924032. See
[rounds/round-29.md](rounds/round-29.md).
