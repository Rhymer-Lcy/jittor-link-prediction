# -*- coding: utf-8 -*-
"""dataset1 learned ranker: LightGBM LambdaRank over 21 hand-built features.

This is the scoring stage of the dataset1 chain. It replaces a fixed
global-weight linear blend of the same evidence (collaborative filtering,
item-CF over embedding similarity, BPR and innovation-BPR dot products, history
strength) with a learned per-row ranking model, so that the relative weight of
each signal can depend on the row rather than being one global constant.

Labels come from a cut-split replay of the training data, never from test
labels: interactions at or before CUT supply the features and history,
interactions after CUT supply the positives, and up to 99 negatives per query
are drawn from that source's own test candidate pool so a training query has the
same shape as a test query.

Protocol (train-cut / infer-full, identical to the ds2 ranker):
  TRAIN  features frozen at CUT (split0) predict split1 labels, cut embeddings.
  INFER  features frozen at full-train end, full-train embeddings, real test
         candidates. Forward-extrapolating time features (time_gap, last_gap)
         are CLIPPED to the training-observed range so inference never sees an
         out-of-support value.

Output: outputs/dataset1-ensemble/result_ranker.csv  (one row per test query,
100 per-row min-max normalised scores, %.6f). This is an intermediate, not a
submission file: src/build_ds1_member.py turns it into the member.

Run: DATASET=dataset1 python src/ranker_ds1.py
"""
import os
import time
from pathlib import Path

os.environ.setdefault("DATASET", "dataset1")
import numpy as np
import pandas as pd
import lightgbm as lgb

import pipeline_common as tl   # framework-neutral shared components
import ensemble_predict as ep
import stage_contract as sc    # atomic publication helper (imports no framework)

REPO = Path(__file__).resolve().parent.parent
CUT = 115480000.0
YEAR = 365.0 * 86400.0
NEG = 99
BPR_SEEDS = (42, 123, 777, 2024, 31337)
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10], n_estimators=400,
              learning_rate=0.05, num_leaves=31, min_child_samples=100, random_state=42,
              n_jobs=6, verbosity=-1, label_gain=[0, 1])
RNG = np.random.default_rng(20260727)
TIME_COLS_CLIP = (12, 19)     # time_gap, last_gap -- clip to train range at infer
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def serialisation_normalise(row: np.ndarray) -> np.ndarray:
    """Map one LambdaRank score row into [0, 1] the way the frozen chain did.

    LambdaRank margins straddle zero: on the frozen A-board embeddings 97.5% of
    the 6,105,100 predicted cells are negative and every one of the 61,051 rows
    contains a negative value. ``pipeline_common.rownorm`` divides by the row
    maximum, which fixes the maximum at 1 but passes negatives straight through
    and, on a row whose maximum is not positive, is undefined. Serialising that
    yields a matrix the submission format rejects.

    The frozen A-board artifact
    ``outputs/dataset1-ensemble/result_ranker.csv`` (sha256 ``cb4964ea...``) does
    not have that shape. Every one of its 61,051 rows has minimum exactly 0.0 AND
    maximum exactly 1.0, with 1.0005 zeros and 1.0000 ones per row over 660,816
    distinct values. Among the candidate transforms that is uniquely the
    signature of a per-row min-max: ``rownorm`` cannot place an exact 0.0 in every
    row, clipping at 0 would leave about 97 zeros per row rather than one, a rank
    map would admit only 100 distinct values, and a bare shift would not fix the
    maximum at 1. So the frozen path applied this transform and the port to
    ``src/`` did not carry it across.

    It is a serialisation-domain step, not a scoring change. Subtracting a
    per-row constant and dividing by a positive per-row constant preserves the
    within-row order exactly, and MRR depends on nothing else; replaying the
    chain under both transforms produced identical postprocessor action counts
    (3,647 and 1,374).

    A degenerate row -- every candidate scored identically -- has no min-max.
    It takes the established repository convention for exactly this case:
    ``frozen_ops.row_max_normalise`` maps a constant row to 0.5, pinned by
    ``test_constant_negative_rows_map_to_one_half``. The value is in range,
    order-neutral and free of division by zero, and it is NOT invented here. No
    constant row occurs in any real artifact -- the frozen ranker, the frozen
    member and the pre-repair reconstruction contain zero of them -- and
    ``tools/submission/package_component.py`` independently fails a member that
    has any, so this branch cannot let one ship unnoticed.
    """
    lo = float(row.min())
    span = float(row.max()) - lo
    if span <= 1e-12:
        return np.full_like(row, 0.5)
    return (row - lo) / span


def popwin(dvals, tvals, hi, lo, frac, num_entity):
    m = tvals >= (hi - frac * (hi - lo))
    return np.log1p(np.bincount(dvals[m], minlength=num_entity).astype(np.float64))


def featurize(struct_df, freeze_t, queries, line_dir, bpr_dirs, ibpr_dir,
              num_entity, freq_cand, cfreq_log):
    """Return the (Nrows, 21) feature matrix for `queries` with history/pops/
    embeddings frozen on `struct_df` at `freeze_t`. queries: list of (src,t,cands).
    Identical computation for train (cut) and infer (full)."""
    tl.build_history_index(struct_df)
    tl.build_cooc(struct_df, num_entity)
    base_cache = ep.build_base_cache(struct_df)
    cache_mat = tl.cache_dict_to_matrix(base_cache, set(), num_entity - 1)
    dst_pop_log = np.log1p(tl.dst_pop)
    dst_pop_raw = tl.dst_pop.copy()
    tmin = float(struct_df["time"].min())
    gr = np.zeros(num_entity)
    for d, tv in struct_df.groupby("dst")["time"].max().items():
        gr[int(d)] = (float(tv) - tmin) / (freeze_t - tmin)
    seen = (np.bincount(struct_df["dst"].values, minlength=num_entity) > 0).astype(np.float64)
    rps = popwin(struct_df["dst"].values, struct_df["time"].values, freeze_t, tmin, 0.05, num_entity)
    rpl = popwin(struct_df["dst"].values, struct_df["time"].values, freeze_t, tmin, 0.40, num_entity)
    src_last = np.full(num_entity, tmin)
    for s, tv in struct_df.groupby("src")["time"].max().items():
        src_last[int(s)] = float(tv)

    bprs = [np.load(d / "bpr_emb.npy") for d in bpr_dirs]
    ibpr = np.load(ibpr_dir / "bpr_emb.npy")
    emb = ep.load_embedding(line_dir, expected_rows=num_entity)
    emb_n = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    srcs = [s for s, _, _ in queries]
    cands = [c for _, _, c in queries]
    need_srcs = np.unique(np.array(srcs, dtype=np.int64))
    tl.build_sim_cache(emb, need_srcs)
    collab_rows = ep.grouped_collab(cache_mat, srcs, cands)
    log(f"  caches ready; featurize {len(queries)} queries")

    rows = []
    for i, (src, t, cc_) in enumerate(queries):
        if i and i % 20000 == 0:
            log(f"    {i}/{len(queries)}")
        cc = np.clip(cc_, 0, num_entity - 1)
        m = len(cc_)
        hist_d, _ = tl.get_hist_before_time(int(src), freeze_t + 1)
        f_collab = tl.rownorm(collab_rows[i].astype(np.float64))
        f_rpop = tl.rownorm(tl.dst_rpop_log[cc])
        f_cooc = tl.rownorm(tl.cooc_scores(int(src), cc))
        f_icfm = np.zeros(m)
        f_icf3 = np.zeros(m)
        if len(hist_d) > 0:
            hvec = emb_n[np.clip(hist_d, 0, num_entity - 1)]
            sim = np.sort(emb_n[cc] @ hvec.T, axis=1)
            f_icfm = tl.rownorm(np.maximum(sim.mean(axis=1), 0.0))
            kk = min(3, sim.shape[1])
            f_icf3 = tl.rownorm(np.maximum(sim[:, -kk:].mean(axis=1), 0.0))
        f_bpr = np.zeros(m)
        for be in bprs:
            f_bpr += tl.rownorm(np.maximum(be[cc] @ be[min(int(src), be.shape[0] - 1)], 0.0))
        f_bpr /= len(bprs)
        hm = np.isin(cc_, hist_d).astype(float)
        hcnt = tl.count_in_history(cc_, hist_d).astype(np.float64)
        rs = tl.rownorm(rps[cc])
        rl = tl.rownorm(rpl[cc])
        f_cfreq = tl.rownorm(cfreq_log[cc])
        f_popratio = tl.rownorm(dst_pop_raw[cc] / (freq_cand[cc] + 1.0))
        last_gap = np.full(m, (t - src_last[int(src)]) / YEAR)
        ib = tl.rownorm(np.maximum(ibpr[cc] @ ibpr[min(int(src), ibpr.shape[0] - 1)], 0.0))
        ib[np.isin(cc_, hist_d)] = 0.0                          # iBPR reranks non-repeat block
        rows.append(np.column_stack([
            f_collab, f_rpop, f_icfm, f_icf3, f_bpr, f_cooc,
            gr[cc], dst_pop_log[cc], np.full(m, len(hist_d)), hm, hcnt,
            seen[cc], np.full(m, (t - freeze_t) / YEAR), np.full(m, m), rs, rl,
            rs - rl, f_cfreq, f_popratio, last_gap, ib]))
    return np.vstack(rows).astype(np.float32)


def main():
    df_raw = pd.read_csv(tl.train_csv)
    test_df = pd.read_csv(tl.test_csv)
    num_entity = int(max(df_raw["src"].max(), df_raw["dst"].max(),
                         test_df[tl.c_cols].values.max(), test_df["src"].max())) + 1
    split0 = df_raw[df_raw["time"] <= CUT].reset_index(drop=True)
    split1 = df_raw[df_raw["time"] > CUT].reset_index(drop=True)
    cand_mat = test_df[tl.c_cols].values.astype(np.int64)
    _allc = np.clip(cand_mat.ravel(), 0, num_entity - 1)
    freq_cand = np.bincount(_allc, minlength=num_entity).astype(np.float64)
    cfreq_log = np.log1p(freq_cand)
    log(f"num_entity={num_entity} split0={len(split0)} split1={len(split1)} test={len(test_df)}")

    # ---- TRAIN: cut-split replay (src in test pool, 99 sampled test-pool negs) ----
    pools = {}
    for i, s in enumerate(test_df["src"].values.astype(np.int64)):
        pools.setdefault(int(s), set()).update(cand_mat[i].tolist())
    lab = split1[split1["src"].isin(pools.keys())].reset_index()
    tq = []
    for row in lab.itertuples(index=False):
        src, dst, t = int(row.src), int(row.dst), float(row.time)
        pool = pools.get(src)
        negs = sorted(pool - {dst, src}) if pool else []
        if not negs:
            continue
        if len(negs) > NEG:
            negs = RNG.choice(negs, NEG, replace=False).tolist()
        tq.append((src, t, np.array(negs + [dst], dtype=np.int64)))
    log(f"train replay queries {len(tq)}")

    # Producer/consumer path contract: these names come from the same helpers the
    # Jittor trainers use, so a producer and its consumer cannot disagree.
    cut_bpr = [tl.bpr_run_dir("dataset1", seed=s, tau_frac=0.25, time_max=CUT)
               for s in BPR_SEEDS]
    cut_ibpr = tl.bpr_run_dir("dataset1", tau_frac=0.25, time_max=CUT, innov=True)
    cut_line = tl.line_run_dir("dataset1", time_max=CUT)
    log("featurize TRAIN (cut-frozen) ...")
    Xtr = featurize(split0, CUT, tq, cut_line, cut_bpr, cut_ibpr, num_entity, freq_cand, cfreq_log)
    lens = np.array([len(q[2]) for q in tq])
    off = np.concatenate([[0], np.cumsum(lens)])
    ytr = np.zeros(off[-1], np.float32)
    ytr[off[1:] - 1] = 1.0
    clip_lo = {c: float(np.min(Xtr[:, c])) for c in TIME_COLS_CLIP}
    clip_hi = {c: float(np.max(Xtr[:, c])) for c in TIME_COLS_CLIP}
    log(f"train matrix {Xtr.shape}; time-clip ranges {clip_lo} .. {clip_hi}")

    mdl = lgb.LGBMRanker(**PARAMS)
    mdl.fit(Xtr, ytr, group=lens.tolist())
    log("ranker trained on full cut-split replay")

    # ---- INFER: real test rows, features frozen at full-train end ----
    full_t = float(df_raw["time"].max())
    full_bpr = [tl.bpr_run_dir("dataset1", seed=s, tau_frac=0.25) for s in BPR_SEEDS]
    full_ibpr = tl.bpr_run_dir("dataset1", tau_frac=0.25, innov=True)
    full_line = tl.line_run_dir("dataset1")     # holds line_latest_emb.csv
    iq = [(int(test_df["src"].values[i]), float(test_df["time"].values[i]), cand_mat[i])
          for i in range(len(test_df))]
    log(f"featurize INFER (full-frozen, freeze_t={full_t:g}) ...")
    Xte = featurize(df_raw, full_t, iq, full_line, full_bpr, full_ibpr, num_entity, freq_cand, cfreq_log)
    for c in TIME_COLS_CLIP:                                    # forward-extrapolation guard
        Xte[:, c] = np.clip(Xte[:, c], clip_lo[c], clip_hi[c])
    log(f"infer matrix {Xte.shape}; predicting")

    scores = mdl.predict(Xte)
    out_dir = tl.ranker_dir("dataset1")     # the shared path contract, not a literal
    os.makedirs(out_dir, exist_ok=True)
    save_path = out_dir / "result_ranker.csv"
    out_rows = []
    pos = 0
    for i in range(len(iq)):
        m = len(iq[i][2])
        s = scores[pos:pos + m]
        pos += m
        out_rows.append(serialisation_normalise(s.astype(np.float64)).tolist())
    # Atomic publication. The formatter, the float format and therefore the bytes
    # are unchanged; only the moment the final name starts existing moves, so an
    # interrupted write can no longer leave a truncated matrix that a later stage
    # would mistake for a finished one.
    with sc.atomic_output(save_path) as staged:
        pd.DataFrame(out_rows).to_csv(staged, index=False, header=False, float_format="%.6f")
    log(f"[OK] ds1 ranker submission -> {save_path}  ({len(out_rows)} rows)")

    # quick sanity: fraction of rows whose argmax is a history (repeat) candidate,
    # vs the production expectation (~66% ds1 answers are repeats).
    tl.build_history_index(df_raw)
    rep_top = 0
    for i in range(len(iq)):
        src, t, cc_ = iq[i]
        hist_d, _ = tl.get_hist_before_time(int(src), full_t + 1)
        am = int(np.argmax(out_rows[i]))
        if np.isin(cc_[am], hist_d):
            rep_top += 1
    log(f"sanity: argmax is a repeat candidate on {rep_top}/{len(iq)} = {rep_top/len(iq):.3f} of rows")


if __name__ == "__main__":
    main()
