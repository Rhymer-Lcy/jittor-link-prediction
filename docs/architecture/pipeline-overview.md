# Pipeline overview

Two datasets, two independent pipelines, one additive score. Nothing crosses between them: the
dataset1 and dataset2 members are produced by completely separate chains and the platform simply
adds their MRRs.

```
                    data/data_A/{dataset1,dataset2}/{train.csv,test.csv}
                                        |
              +-------------------------+-------------------------+
              |                                                   |
        DATASET 1                                           DATASET 2
              |                                                   |
    train_line.py  (LINE embedding)                   train_line.py  (LINE embedding)
    train_bpr.py   (BPR-MF, 10 seeds                  train_bpr.py   (BPR-MF, 5 seeds)
                    + innovation BPR w=12)                          |
              |                                        ranker_basket_ds2.py
    ensemble_predict.py  (feature source;              18 features + 3-pass basket feedback
     the hand-tuned blend itself is superseded)                     |
              |                                        ds2_mf_basket_pack.py
    ranker_ds1.py  (cut-split LambdaRank,              d128 SVD sibling-message geometry
     21 columns, train-cut / infer-full)               (uses ds2_basket_featurizer.py)
              |                                                     |
     result_ranker.csv   0.86197                        crf_promote.py
              |                                         equality CRF tau=0.20 B=70
    strategies/ds1/source_slate_recurrence.py           + triple + zero-repeat + same-time
     3,653 actions                                      (--no-pair)
              |                                                     |
    strategies/ds1/test_graph_reciprocity.py                        |
     1,365 actions                                                  |
              |                                                     |
     dataset1.csv  0.89630                              dataset2.csv  0.67895
              |                                                     |
              +----------------------+------------------------------+
                                     |
                    tools/submission/package_component.py
                     deflate-9 ZIP, both members, byte-verified
                                     |
                          total 1.5752524794476366
```

## Why the two sides look so different

**dataset1 is repeat-dominated.** 66% of next interactions repeat a past partner, so the history
count term carries the leaderboard number and the headroom lives entirely in the non-repeat block.
That is why the two shipped dataset1 postprocessors both act *only* when the incumbent top-1 is
**non-historical** — they never touch the rows the history term already gets right.

**dataset2 has 0% repeats** (history is masked to zero) and is strictly bipartite. Its gains have
come from *structure in the benchmark file itself*: same-timestamp rows form contiguous runs that
share answers, which the equality CRF and the triple / zero-repeat / same-time invariants exploit.
The raw test row order is load-bearing there.

## The three insertion points that have actually paid

1. **Embedding geometry** — what vector space sibling messages and item-CF are computed in.
   The MF basket geometry (+0.0145 online) replaced a hand-built profile matrix.
2. **Learned ranking over frozen features** — LambdaRank trained on an interior *time cut*, never
   on holdout tails. The same architecture trained on tails regressed online; the cut-split
   version shipped.
3. **Deterministic postprocessors over the batch itself** — the inference batch is evidence. Both
   Round-21/22 dataset1 gains (+0.0263, +0.0080) came from counting candidate identities across
   the test slates, with no model retrained and no threshold swept.

The third is by far the cheapest per point of score and is where the last two rounds have lived.

## Evaluation protocols, and what each is good for

| protocol | use | caution |
|---|---|---|
| leaky eval (`ensemble_predict.py --eval`) | ordering non-temporal features | overrates recency features; inflates item-CF ~2.4x |
| holdout / `EVAL_HOLDOUT=1` | embedding-geometry features | never train a model on the tails |
| cut-split replay | learned rankers | the only protocol that made a learned ds1 ranker ship |
| source-disjoint OOF (244k queries) | ds2 full-chain marginals | a SCREEN, not a shipping gate |
| **isolated online A/B** | **the shipping gate** | costs one auxiliary submission; always worth it |

The standing rule, learned the expensive way: **an offline marginal is a screen, not a shipping
gate.** Two full-stack, null-controlled, fold-stable offline wins have inverted online
(five-channel basket -0.0116, within-event collision -0.0147). Every gain shipped since is one
that was adjudicated by an isolated online A/B *before* it entered the main pack.

## Where the remaining headroom is believed to be

- dataset1 non-repeat cold ranking (the postprocessors only reach ~8% of rows so far);
- a B-board-viable unified NN (survival-hazard or path-propagation), still unbuilt;
- dataset2's four sub-axes (features / post-processing / hyperparameters / ensemble structure) are
  all closed — its 18-feature ablation leaves only two load-bearing features, both structural
  invariants.

## Framework note

The competition requires Jittor for model design, training and prediction. `train_line_jt.py` and
`train_bpr_jt.py` are the Jittor implementations and are the framework-compliant path; only the LINE
model, the BPR trainer, Adam and BCE are framework-specific, everything else is numpy/scipy.

**The accepted A-board embeddings were produced by the PyTorch trainers**, not by these ports. Both
implementations share the objective, the knobs and the output format, and both run sampling in seeded
NumPy, so they agree statistically but **not** byte for byte: a Jittor retrain yields a different
embedding table by construction, and therefore a different member hash. This provenance gap is stated
rather than implied, and its consequences for reproduction are set out in
[../competition/ab_algorithm_consistency_contract.md](../competition/ab_algorithm_consistency_contract.md)
section 9.

Everything downstream of the ranker is framework-neutral and **is** byte-exact: `python
src/build_ds1_member.py --verify` and `python src/build_ds2_member.py --verify` regenerate both
accepted members exactly, verified on Ubuntu 22.04 with Python 3.10.
