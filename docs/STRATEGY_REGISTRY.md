# Strategy registry

Generated from [`docs/strategy_inventory.json`](strategy_inventory.json), which is
reconstructed from the README score ledger, `configs/production.json`, the submission
artifacts, the round reports under `scratchpad/` and durable agent memory. The JSON is
the source of truth and carries fuller evidence per record than the tables below.

Lifecycle vocabulary is defined in [`src/strategies/registry.py`](../src/strategies/registry.py).

| status | count |
|---|---|
| `SHIPPED_ACTIVE` | 14 |
| `SHIPPED_SUPERSEDED` | 6 |
| `ONLINE_VALIDATED` | 0 |
| `CANDIDATE` | 0 |
| `STANDBY` | 1 |
| `PROBE` | 3 |
| `CLOSED` | 29 |
| **total** | **53** |

> ONLINE_VALIDATED is empty by construction: in this project every mechanism that passed an isolated online test was shipped in the next main pack, and every mechanism that failed one was closed permanently.

## SHIPPED_ACTIVE

Part of the current accepted submission. Changing any of these changes production.

| strategy | dataset | mechanism | evidence | implementation |
|---|---|---|---|---|
| `line_embedding` | both | LINE first- + second-order proximity embedding trained on all real edges, exported as cat([emb_first, emb_node]) | present in every accepted submission since 1.4341 | `src/train_line.py` |
| `bpr_embedding` | both | BPR-MF single node table, pairwise softplus ranking loss, degree^0.75 negatives | ds1 0.8285 -> 0.8508, ds2 0.5607 -> 0.5712 | `src/train_bpr.py` |
| `multi_seed_bpr_ensemble` | both | rownormed BPR direct scores averaged over independent seeds (ds1 10, ds2 5) | +0.0054 / +0.0018; ds1 ten-seed isolated read 0.8595 -> 0.86000 | `src/train_bpr.py` |
| `innovation_bpr` | dataset1 | BPR trained only on the FIRST occurrence of each (src,dst) pair, zeroed on history candidates before rownorm | 1.50890 -> 1.51309 (+0.00419 incl. a lineage fix) | `src/train_bpr.py` |
| `item_cf_blend_terms` | both | cosine between a candidate embedding and the source's historical dst embeddings (mean over all + mean of top-3) | ds1 0.803 -> 0.813, ds2 0.5441 -> 0.5587 | `src/ensemble_predict.py` |
| `ds1_lambdarank_ranker` | dataset1 | cut-split LambdaRank over 21 columns; features frozen at a 0.75 time-quantile cut for training, full-train frozen at serve | +0.00205 (ds1 0.86000 -> 0.862047); component 0.8619654323294592 before the postprocessors | `src/ranker_ds1.py` |
| `source_slate_recurrence` | dataset1 | promote the unique maximum-recurrence non-history candidate when it appears in >= 2 other slates of the same source and the control top-1 is non-hi... | isolated auxiliary A/B before shipping: 0.9136871602584966 -> 0.9400134058656336, delta +0.0263262456071370 | `src/strategies/ds1/source_slate_recurrence.py` |
| `test_graph_reciprocity` | dataset1 | promote the rank-2 candidate when the reverse test-exposure edge candidate->source exists for it and not for rank 1, both being non-historical | isolated auxiliary read 0.8963013747474597 vs baseline 0.8882916779365962, delta +0.0080096968108635 against a +0.0010 gate | `src/strategies/ds1/test_graph_reciprocity.py` |
| `ds2_basket_lambdarank` | dataset2 | 18-feature LightGBM LambdaRank with 3-pass basket feedback (sibling messages within a (src,time) event) | part of every accepted ds2 member since 1.50890 | `src/ranker_basket_ds2.py` |
| `mf_basket_geometry` | dataset2 | d128 unit SVD of the split0 src x dst interaction used as the pass-2/3 sibling-MESSAGE geometry, replacing the hand-built item_profiles | single-variable auxiliary A/B: 0.7166850834516475 -> 0.7312368936166302, +0.0145518 (a ~2.2x UPWARD transfer) | `src/ds2_mf_basket_pack.py` |
| `structural_row_order_crf` | dataset2 | equality CRF (exact sum-product over same-time adjacency chains, pairwise potential 1 + B*delta(answer equality), unaries softmax(rownorm/tau)) plu... | equality-CRF +0.01636 isolated; chain 1.49254 -> 1.50890 | `src/crf_promote.py` |
| `triple_promotion` | dataset2 | promote the unique common warm candidate of every strict three-row same-time window (three distinct sources) to top-1 in all three rows | 1.45725 -> 1.47499 (+0.01775), matching the risk-adjusted projection ~1:1 | `src/crf_promote.py` |
| `zero_repeat_exclusion` | dataset2 | fifth invariant -- 0 of 2.2M (src,dst) pairs recur at distinct times, so a labelled cross-time answer is a negative; excluded, plus duplicate demotion | +0.001764 (1.5149319 -> 1.5166958) | `src/crf_promote.py` |
| `same_time_structural_exclusion` | dataset2 | sixth invariant -- within a timestamp a shared answer forms one contiguous raw-order run, so a warm candidate in a shorter disjoint same-time compo... | +0.0006142 (1.5166958 -> 1.5173100) | `src/crf_promote.py` |

## SHIPPED_SUPERSEDED

Was in an accepted submission and was later replaced. Kept for reproduction of historical runs; do not re-enable without a fresh online adjudication.

| strategy | dataset | mechanism | evidence | implementation |
|---|---|---|---|---|
| `ds1_linear_blend` | dataset1 | hand-tuned linear blend of history count, user-CF, co-occurrence, item-CF and BPR terms | carried ds1 from 0.803 to 0.86000 across many submissions | `src/ensemble_predict.py` |
| `ds2_linear_blend` | dataset2 | hand-tuned linear blend (user-CF, recent popularity, item-CF, BPR) | ds2 0.5441 -> 0.5787 | `src/ensemble_predict.py` |
| `temporal_footprint_residual` | dataset2 | 85-column candidate x query-time exposure block distilled into a per-candidate residual added before the shared CRF | shipped at total 1.5264, ds2 isolated MRR 0.6664472907 | `src/footprint_feature.py` |
| `pair_promotion` | dataset2 | adjacent same-time different-source rows sharing a warm candidate that is top-1 in one row promote it in the other | shipped for +0.00955; REMOVED for +0.0004173 (1.5145953 -> 1.5149319) | `src/crf_promote.py` |
| `triple_promote_standalone` | dataset2 | the triple rule applied on its own, without the equality CRF | +0.01775 when first shipped alone | `src/triple_promote.py` |
| `virtual_edge_self_training` | both | candidates above BACK_FILL_THRESHOLD become virtual edges merged into later LINE training rounds | present in early accepted runs; contribution measured as null | `src/train_line.py` |

## ONLINE_VALIDATED

*None.* Passed an isolated online test but is not in the accepted state.

## CANDIDATE

*None.* Passed the offline gate, no online read yet.

## STANDBY

Weak-but-positive evidence, deliberately inactive.

| strategy | dataset | mechanism | evidence | implementation |
|---|---|---|---|---|
| `xlist_objective_diversity` | dataset1 | objective-diversity blend over the ranker output | none | -- |

## PROBE

Exploratory only; no promotion decision was taken.

| strategy | dataset | mechanism | evidence | implementation |
|---|---|---|---|---|
| `candidate_source_peak_crossing` | dataset1 | act when rank 2 sits at the maximum physical source exposure attained by that candidate across sources and rank 1 does not | none | -- |
| `candidate_source_role_crossing` | dataset1 | act when rank 2 appears as a physical test query source and rank 1 does not | none | -- |
| `ds2_stable_core_envelope` | dataset2 | rank 2 has other physical exposures before and after the query horizon while rank 1 does not | none | -- |

## CLOSED

Refuted by offline or online evidence. **Do not reopen in the same formulation.**

| strategy | dataset | mechanism | evidence | implementation |
|---|---|---|---|---|
| `within_event_collision_exclusion` | dataset2 | demote a candidate claimed as top-1 by a sibling row of the same event (E1 / E1b / E1c / T1 family) | control 0.7312368936166302 -> treatment 0.7165484296355199, delta -0.0146884639811103 | -- |
| `chronological_prefix_standin` | dataset2 | rolling chronological-prefix stand-in training instead of source-disjoint OOF | worse on both future blocks (MRR -0.000797 / -0.001431) and worse calibrated (ECE 0.0446 vs 0.0141) | -- |
| `test_native_pseudo_label_ranker` | dataset2 | residual ranker trained on test-native pseudo-labels with matched negatives | best unlabelled post-CRF marginal +0.000005 against a required +0.002; the unit-label-weight ablation beats the treatment | -- |
| `five_channel_basket_representation` | dataset2 | serve 5 disentangled basket channels instead of the shipped 2 entangled ones | control 0.7313145109421939 -> treatment 0.719693486657827, delta -0.0116210242843669 | -- |
| `footprint_on_mf_base` | dataset2 | refit the footprint residual against the MF base and stack both | pre-CRF footprint marginal over MF +0.00813 but full-chain post-CRF +0.002066 < the +0.003 gate; anti-correlated with MF (-0.259) and -0.029 on the rows MF already fixed | -- |
| `ds1_reverse_edge_features` | dataset1 | explicit reverse-edge indicator and count columns (candidate -> source) in the ranker | overall +0.000372 against a +0.004 gate; a source-activity x candidate-popularity matched null reaches 100.4% of the gain; 17,786 repaired vs 17,959 damaged | -- |
| `ds2_line_block_geometry_fix` | dataset2 | drop or renormalise the emb_node block whose inner product LINE never optimised | standalone item-CF readout 0.208141 -> 0.426181, but the full shipped chain scores pass-3 -0.001165 and post-CRF -0.001110, negative on every slice | -- |
| `directed_three_hop` | dataset1 | directed walk A^3 as explicit count / exists / weighted ranker columns | marginals -0.000498..+0.000310, folds and seeds disagree in sign, and a source-activity x popularity shuffled null (+0.000305) beats every real arm | -- |
| `co_event_ppmi_geometry` | dataset2 | co-event PPMI-SVD(d128) as the basket feedback geometry | naive pass-3 +0.008457 but capacity-MF and random-vector nulls scored 166% and 184%; identity-deconfounded it LOSES to the incumbent P-profile (-0.00758) and to the plain-MF null (-0.01558) | -- |
| `graph_motif_ds1` | dataset1 | graph-motif features on ds1 against the 21-column ranker | closed against the full ranker | -- |
| `non_repeat_specialist_ds1` | dataset1 | a specialist model for the non-repeat block | closed against the full ranker | -- |
| `candidate_column_position` | both | use the candidate's column index as evidence | truth column histogram chi-square 99.9 on 99 df, z = +0.07 | -- |
| `temporal_decay_ds1` | dataset1 | time-decayed history counts / recency-weighted aggregation | 0.803 -> 0.7905 when shipped (confounded submission) | -- |
| `slate_restricted_negative_sampling` | both | draw embedding-training negatives from the model-ranked bottom of each source's own test slate | false-negative rate 3.6x worse than global uniform; model-top-10 contamination 38x worse; the shuffled-slate null sits closer to the incumbent than the treatment | -- |
| `virtual_edge_bpr_feedback` | both | feed virtual edges into BPR training | would make ~17.9% of BPR's positive mass synthetic from sources whose destinations are 12.2x popularity-skewed | -- |
| `architecture_sweep_sasrec_ials_metablend` | both | SASRec-lite next-item score, iALS direct scores, LambdaRank meta-blend over blend terms | meta-blend regressed 0.8554 -> 0.8524 | -- |
| `tie_breaking` | dataset2 | break the ~21 tied candidates per row | the truth never lands in a tie (0.000% of 244,056 replay queries) | -- |
| `cooc_term_removal` | dataset1 | drop the co-occurrence blend term (W_COOC = 0) | -0.0001 | -- |
| `ds1_blend_retune` | dataset1 | coordinate-descent retune of the blend weights | +0.00096 in sample, +0.00053 (P=0.13) cross-fitted, per-fold vectors disagree wildly; the eleven-weight joint tune is worse (+0.00027, P=0.26) | -- |
| `refuted_online_feature_family` | both | time-decayed history, co-occurrence CF on ds2, sequential transition feature, test-candidate-frequency prior (tpop) | 0.803->0.7905, 0.526->0.505, 0.5441->0.5305, 0.5587->0.5199 | -- |
| `basket_soft_factor_graph` | dataset2 | full posterior decoding over the basket factor graph instead of the top-3 decoder | the full posterior beats the shipped decoder by only +0.0006 against a +0.072 oracle | -- |
| `truth_free_variance_selector` | dataset2 | a deployable selector over semantic evidence | oracle +0.0935 but the deployable gate is +0.00038 at 32% precision and loses to a shuffle null | -- |
| `pass3_pair_decider` | dataset2 | pairwise 'which of A,B ranks higher' decider over pass-3 sibling-consistency features | 60.2% precision / +0.0075 OOF, but a stratified shuffle of the pass-3 diff features beat it (+0.014) | -- |
| `query_bootstrap_bagging` | dataset2 | Poisson(1) query-level bootstrap bagging of the basket ranker | pilot did not clear its pre-registered kill criteria | `src/ds2_basket_featurizer.py` |
| `adjacency_slate_features` | dataset2 | adjacency slate-membership features inside the ranker | -0.0065: the ranker re-extracts 98% of the same signal worse than the CRF, and stacking double-counts | -- |
| `cohort_label_shift_reweighting` | dataset2 | query reweighting for cohort label shift | +0.0002 -- the ranker's activity features already price staleness per row | -- |
| `crf_refinement_axis` | dataset2 | band width W, potential exponent p, candidate-reliability coupling eta, semi-Markov, CRF round-2 / ensemble, ds1 row-order | the (tau,B) plane is one w-curve; --W monotone-negative (double-counts the chain); --p dead; --eta killed even with perfect per-bin truth | -- |
| `rank_derivative_features` | both | ordinal derivatives of transferring features (e.g. cfreq_rank) | -0.0028 | -- |
| `in_pool_query_reweighting` | dataset2 | reweight queries toward the production-matched in-pool slice | monotonically negative (w3 -0.00177, w6 -0.00410) | -- |

## Prohibited variants on active strategies

These were frozen by an online adjudication. Re-introducing one silently invalidates the
accepted hashes and the evidence behind them.

**`source_slate_recurrence`** -- threshold tuning; relaxing uniqueness; changing the row gate or history definition; altering the score transform; using candidate column position.

**`test_graph_reciprocity`** -- inspecting ranks below 2; reverse-edge counts or thresholds; widening eligibility; adding recurrence / source-peak / source-role conditions.

