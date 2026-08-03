# Pipeline overview

The repository contains two dataset-specific pipelines and one deterministic
packaging layer. The platform score is the sum of the two component scores.

## Canonical execution graph

```text
official train.csv and test.csv
             |
     Jittor LINE and BPR-MF
             |
   collaborative score features
             |
     +-------+--------+
     |                |
  Dataset 1        Dataset 2
  LambdaRank       basket LambdaRank
     |                |
  recurrence       MF basket geometry
     |                |
  reciprocity      equality CRF and invariants
     |                |
  dataset1.csv     cross-time exclusivity
                      |
                   dataset2.csv
     |                |
     +-------+--------+
             |
       verified ZIP package
```

`run_all.py` invokes Dataset 1 before Dataset 2 because the Dataset 2 packaging
stage passes the Dataset 1 member through into the two-member container.

## Dataset-specific structure

Dataset 1 is repeat-dominated. Its learned ranker combines historical and
embedding-derived evidence, while both accepted postprocessors act only on
eligible non-history candidates.

Dataset 2 has no repeated source-destination pair across distinct timestamps in
the measured training census. Its accepted gains rely on event structure,
same-time row adjacency, basket geometry, and the cross-time exclusivity
invariant. Raw test-row order is therefore part of the input contract.

## Framework boundary

`src/train_line_jt.py` and `src/train_bpr_jt.py` are the only trainers selected
by the canonical graph. The optional PyTorch reference backend is isolated in
`reference/pytorch/`, uses a separate environment, and writes to suffixed output
directories. It is never selected by an implicit fallback or environment
switch.

The Jittor and PyTorch implementations share objectives and parameter
conventions, but they do not share optimiser trajectories or random-number
streams. Cross-framework output bytes are not expected to match.

## Stage contract

Every expensive stage publishes a completion record beside its artifact. The
record binds the code revision, command, input hashes, configuration, output
hash, and validation result. A file without a valid completion record is not a
reusable stage result.

See `stage-completion-contract.md` for the complete reuse rules.

## Evaluation discipline

Offline evaluation is used to reject weak proposals and diagnose mechanisms;
it is not treated as proof of an online gain. Accepted strategy claims identify
whether evidence is online-observed, derived from component arithmetic, or
limited to offline evaluation. Historical negative results remain in the
strategy inventory so they are not silently reopened.
