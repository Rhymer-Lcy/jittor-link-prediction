# Current production state

**Accepted main score 1.5752524794476366 — visible rank #3** (board: #1 1.5885, #2 1.5818),
accepted 2026-07-29.

Machine-readable source of truth: [`configs/production.json`](../configs/production.json).
Everything below is derived from it; if the two disagree, the JSON wins and this file is stale.

## What defines production

| item | value |
|---|---|
| accepted archive | `outputs/submissions/round22_main/r22_rgr_main_d9.zip` |
| archive SHA256 | `6b9f0cd928c49ec2b6f48554c5d3ef2cfae6dbc34650dd17f19c03105dd9dae7` |
| archive bytes | 63,984,734 (deflate method 8, level 9) |
| `dataset1.csv` SHA256 | `baa0dc21e1f4b93e579b4c895a8a44da2064b6f3cbbc3d2d1be18314e1125987` |
| `dataset2.csv` SHA256 | `dc6928c266190c5bab97ec4426bc647b526f8717dc589d6381cb60b1dee18a99` |
| dataset1 component | 0.8963013747474597 |
| dataset2 component | 0.6789511047001768 |

The platform total is **strictly additive**: `0.8963013747474597 + 0.6789511047001768` is the
accepted total to the last digit. Each component was measured directly by uploading a
single-member ZIP to the auxiliary account.

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

### dataset2 — 0.6789511047001768

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
        -> dataset2.csv   component 0.6789511047001768
```

The dataset2 member has been **byte-identical since 2026-07-28** (shipped at total
1.540916537029636). Every gain since then is on dataset1.

> **Reproduction risk — HIGH.** `ds2_mf_basket_pack.py` and `ds2_basket_featurizer.py` were
> graduated from the scratchpad on 2026-07-29 **verbatim but not re-executed**, and the pipeline
> also needs the cached train features that `build_or_load_features` produces. dataset2 is
> therefore tracked but not currently proven reproducible from a clean checkout. See
> [maintenance/repository-reorganisation-ambiguities.md](maintenance/repository-reorganisation-ambiguities.md).

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
python -m unittest discover -s tests -t .            # 55 tests; data-dependent ones skip cleanly
python tools/submission/package_component.py verify \
    --zip outputs/submissions/round22_main/r22_rgr_main_d9.zip
```

The verify command prints both member hashes and reports `IS the accepted member` for each. If
the local artifacts are absent, the integration tests skip with an explicit reason and the
config-level tests still assert that the manifest, the registry and the inventory agree.

## Building a new submission

See [SUBMISSION_PROTOCOL.md](SUBMISSION_PROTOCOL.md). In short: measure one component in
isolation on the auxiliary account with a single-member ZIP, and only merge it into a main pack
with the frozen other member after it has passed online.
