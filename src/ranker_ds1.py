# -*- coding: utf-8 -*-
"""dataset1 learned ranker -- replaces the hand-tuned linear blend.

Until now ds1's entire score came from ensemble_predict.blend_scores, a FIXED
global-weight linear combination (collab/item-CF/BPR/iBPR/HIST_BOOST). A 2-fold
src-disjoint OOF probe (scratchpad/ds1_ranker_probe.py + ds1_ranker_ibpr.py)
showed a LambdaRank on the same evidence beats the real shipped mechanism
(blend + 12*iBPR) by +0.0187 MRR on a production-matched cut-split replay:
  prod(blend+iBPR)=0.6442  opt-linear+iBPR=0.6340  RANK+iBPR=0.6629
The gain is dominated by nonlinear per-row conditioning (+0.029 vs the strongest
fixed-linear fit), robust to seed (noise 0.00013), does NOT sign-flip on the
latest horizon quartile (+0.032), and survives dropping every time feature
(+0.026) -- so it is not a recency/training-boundary artifact.

Protocol (train-cut / infer-full, identical to the ds2 ranker):
  TRAIN  features frozen at CUT (split0) predict split1 labels, cut embeddings.
  INFER  features frozen at full-train end, full-train embeddings, real test
         candidates. Forward-extrapolating time features (time_gap, last_gap)
         are CLIPPED to the training-observed range so inference never sees an
         out-of-support value.

Output: outputs/dataset1-ensemble/result_ranker.csv  (one row per test query,
100 rownormed scores, %.6f -- same format as result_ensemble.csv).

Run: DATASET=dataset1 D:/Anaconda3/envs/jittor/python.exe src/ranker_ds1.py
"""
import os
import time
from pathlib import Path

os.environ.setdefault("DATASET", "dataset1")
import numpy as np
import pandas as pd
import lightgbm as lgb

import train_line as tl
import ensemble_predict as ep

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

    cut_bpr = [REPO / "outputs" / ("dataset1-bpr-t0.25-tmax1.1548e+08" + (f"-s{s}" if s != 42 else ""))
               for s in BPR_SEEDS]
    cut_ibpr = REPO / "outputs" / "dataset1-bpr-innov-t0.25-tmax1.1548e+08"
    cut_line = REPO / "outputs" / "dataset1-novirt-tmax1.1548e+08"
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
    full_bpr = [REPO / "outputs" / ("dataset1-bpr-t0.25" + (f"-s{s}" if s != 42 else "")) for s in BPR_SEEDS]
    full_ibpr = REPO / "outputs" / "dataset1-bpr-innov-t0.25"
    full_line = REPO / "outputs" / "dataset1"                   # holds line_latest_emb.csv
    iq = [(int(test_df["src"].values[i]), float(test_df["time"].values[i]), cand_mat[i])
          for i in range(len(test_df))]
    log(f"featurize INFER (full-frozen, freeze_t={full_t:g}) ...")
    Xte = featurize(df_raw, full_t, iq, full_line, full_bpr, full_ibpr, num_entity, freq_cand, cfreq_log)
    for c in TIME_COLS_CLIP:                                    # forward-extrapolation guard
        Xte[:, c] = np.clip(Xte[:, c], clip_lo[c], clip_hi[c])
    log(f"infer matrix {Xte.shape}; predicting")

    scores = mdl.predict(Xte)
    out_dir = REPO / "outputs" / "dataset1-ensemble"
    os.makedirs(out_dir, exist_ok=True)
    save_path = out_dir / "result_ranker.csv"
    out_rows = []
    pos = 0
    for i in range(len(iq)):
        m = len(iq[i][2])
        s = scores[pos:pos + m]
        pos += m
        out_rows.append(tl.rownorm(s.astype(np.float64)).tolist())
    pd.DataFrame(out_rows).to_csv(save_path, index=False, header=False, float_format="%.6f")
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
