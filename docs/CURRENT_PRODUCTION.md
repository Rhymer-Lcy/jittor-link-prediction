# Current production state

`configs/production.json` is the authoritative machine-readable record. This
document explains that record without replacing it.

## Accepted result

| Item | Value |
|---|---|
| A-board total | `1.576996059163449` |
| Exact decimal component sum | `1.5769960591634492` |
| Recorded final rank | 3 |
| Dataset 1 component | `0.8963013747474597` |
| Dataset 2 component | `0.6806946844159895` |
| Accepted archive | `outputs/submissions/round30_final/r30_xte_final.zip` |
| Archive SHA-256 | `efe790a56a715ec451a809a560f25ac6d3ecb0cde1d71a46e4de7fcdb6a4a536` |
| Dataset 1 member SHA-256 | `baa0dc21e1f4b93e579b4c895a8a44da2064b6f3cbbc3d2d1be18314e1125987` |
| Dataset 2 member SHA-256 | `beb13345dc020f32283cea2d132efa072eb52c73d29806f86f60fbf24982f971` |

The platform total is additive. Each component was observed through an isolated
single-member submission before the combined package was assembled. The rank is
based on the retained final-board observation documented in
`leaderboard_final.md`; it is not presented as an independently fetched
official timestamped record.

The accepted package reuses existing verified CSV bytes. Dataset 1 uses CRLF
line endings and Dataset 2 uses LF. Both are immutable byte artifacts and must
not be normalised or re-serialised.

## Canonical framework

The maintained raw-data pipeline uses Jittor:

```text
src/train_line_jt.py
src/train_bpr_jt.py
```

The accepted historical embeddings were produced by the earlier PyTorch
implementations now retained under `reference/pytorch/`. Those files are
provenance and research references; no production entry point imports or
selects them.

## Dataset 1 chain

```text
Jittor LINE and BPR-MF trainers
  -> src/ensemble_predict.py
  -> src/ranker_ds1.py
  -> src/strategies/ds1/source_slate_recurrence.py
  -> src/strategies/ds1/graph_reciprocity.py
  -> dataset1.csv
```

The pinned ranker input is
`outputs/dataset1-ensemble/result_ranker.csv`, SHA-256
`cb4964ea21dcefdecb2a34f13ca8500adcc54120684f10052f78a87df88da16b`.
The two deterministic postprocessors are order-sensitive: graph reciprocity
reads the rank-2 candidate after source-slate recurrence has run.

`src/ranker_ds1.py` serialises each row through an order-preserving min-max
transform. The resulting `[0, 1]` score-domain contract is required by the
frozen postprocessors and final output gate.

Verify the pinned downstream reconstruction with:

```bash
python src/build_ds1_member.py --verify
```

## Dataset 2 chain

```text
Jittor LINE and BPR-MF trainers
  -> src/ensemble_predict.py
  -> src/ranker_basket_ds2.py
  -> src/ds2_mf_basket_pack.py
  -> src/crf_promote.py
  -> src/strategies/ds2/cross_time_exclusivity.py
  -> dataset2.csv
```

The base matrix has component score `0.6789511047001768`. The final
cross-time-exclusivity decoder changes 7,815 rows by swapping existing rank-1
and rank-2 score tokens, producing an observed component gain of
`0.0017435797158127`. It performs no score re-serialisation.

Verify the pinned downstream reconstruction with:

```bash
python src/build_ds2_member.py --verify
```

## Reproducibility claims

The following claims are deliberately separate:

1. The accepted archive and both accepted members are hash-pinned.
2. The deterministic post-processing stages regenerate accepted member bytes
   from their pinned upstream matrices.
3. The raw-data-to-final Jittor chains were executed on the target host on
   1 and 2 August 2026, with all stages validated.
4. A fresh neural retraining run is not expected to match the historically
   accepted PyTorch-derived member bytes because framework random streams and
   optimiser trajectories differ.

Therefore the repository establishes executable and algorithmic reproduction,
and byte-exact downstream reconstruction, without claiming cross-framework
neural-training byte identity.

## Immutable production constraints

- Do not edit accepted CSV or ZIP artifacts in place.
- Do not change strategy order, thresholds, row gates, score transforms, or
  history definitions without a new adjudication.
- Do not sort Dataset 2 test rows before the structural postprocessors.
- Keep the Dataset 2 pair-promotion rule disabled.
- Do not use candidate column position as predictive evidence.

The detailed lifecycle and prohibited variants are recorded in
`STRATEGY_REGISTRY.md` and `strategy_inventory.json`.

## Validation commands

```bash
python main.py --dataset dataset1 --stage describe
python main.py --dataset dataset2 --stage describe
python -m pytest
ruff check --no-cache .
ruff format --check --no-cache .
```

Artifact-dependent verification commands fail or skip explicitly when the
private data or output artifacts are absent; they do not silently weaken the
claim.
