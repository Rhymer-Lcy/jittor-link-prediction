# GRADUATED 2026-07-29 from scratchpad/round-16-opus/build_mf_aux_pack.py
# (source sha256 d9c6204b92239b8871463988bc9bd76a1b4acc4b6f07b1e6bf251c65e6609ac4).
# Body is VERBATIM apart from two path-only changes: REPO is derived from __file__
# instead of a hard-coded absolute path on the original build host, and the featurizer is imported as
# src/ds2_basket_featurizer.py. Algorithm, parameters, seeds and normalisation untouched.
# NOT re-executed during the 2026-07-29 consolidation -- see configs/production.json.
# -*- coding: utf-8 -*-
"""Build the ds2 base CSV for the aux A/B: single-variable basket-geometry swap.

--geom P  : item_profiles (sparse, production geometry) + production fb_features (entangled).
            MUST reproduce outputs/dataset2-ranker/ranker_basket3_label_dataset2.csv (the live
            control) -> proves the pipeline is faithful, so --geom MF is a clean single variable.
--geom MF : plain-MF (SVD of src x dst, d128 unit) as the basket message geometry, entangled fb
            (self-sim included -> in a unit geometry the exact-hit self-sim is 1 regardless, so the
            ONLY change vs control is the off-diagonal geometry). MF frozen: split0 for train,
            full train for serve; no split1/test truth, no structural labels; rank fixed d128.

Everything else identical to ranker_basket_ab_ds2.py (label variant): 18 base features, LambdaRank
PARAMS, seeds, 2-fold rng-7 crossfit for train-side stand-in scores, structural-label stand-in,
row-minmax + history-mask serve normalisation. Reuses the cached train features and bag's verbatim
featurizer. Does NOT touch ensemble_predict scoring or any other production feature/param.

Run: DATASET=dataset2 python .../build_mf_aux_pack.py --geom MF --out <base.csv>
"""
import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("DATASET", "dataset2")
os.environ.setdefault("DATA_PACK", "data_A")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import svds  # noqa: E402
import lightgbm as lgb  # noqa: E402
import train_line as tl  # noqa: E402
import ensemble_predict as ep  # noqa: E402
import ds2_basket_featurizer as bag  # noqa: E402
bag.REPO = REPO

T0 = time.time()
DIM = 128
SEEDS = (42, 123, 777, 2024, 31337)
CUT = 1261958400.0
YEAR = 365.0 * 86400.0
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10], n_estimators=400,
              learning_rate=0.05, num_leaves=31, min_child_samples=100, random_state=42,
              n_jobs=6, verbosity=-1, label_gain=[0, 1])


def log(m):
    print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)


def l2n(E):
    n = np.linalg.norm(E, axis=1, keepdims=True); n[n == 0] = 1.0
    return (E / n).astype(np.float32)


def mf_geom(edges, num_entity, dim=DIM, seed=13):
    """d128 unit SVD of the src x dst interaction (the plain-MF basket geometry)."""
    s = edges["src"].to_numpy(np.int64); d = edges["dst"].to_numpy(np.int64)
    A = sp.csr_matrix((np.ones(len(s), np.float32), (s, d)), shape=(num_entity, num_entity))
    A.data[:] = 1.0
    k = min(dim, min(A.shape) - 1)
    _, S, Vt = svds(A.astype(np.float64), k=k, v0=np.random.default_rng(seed).standard_normal(A.shape[0]))
    return l2n(Vt.T * np.sqrt(np.maximum(S, 0.0))[None, :])


def fb_dense_ent(qsrc, qt, cands, s1, E, num_entity, labels=None):
    """Production fb_features (entangled: self-sim INCLUDED) but in dense geometry E."""
    n = len(qsrc)
    ev = defaultdict(list)
    for i in range(n):
        ev[(int(qsrc[i]), float(qt[i]))].append(i)
    fmax = [np.zeros(len(cands[i]), np.float32) for i in range(n)]
    fmean = [np.zeros(len(cands[i]), np.float32) for i in range(n)]
    for idxs in ev.values():
        if len(idxs) < 2:
            continue
        tvecs = []
        for j in idxs:
            lab = labels.get(j) if labels is not None else None
            if lab is not None:
                tvecs.append(E[np.clip(int(lab), 0, num_entity - 1)])
                continue
            s = np.asarray(s1[j], np.float64)
            o3 = np.argsort(-s)[:3]
            w = np.exp(s[o3] - s[o3].max()); w /= w.sum()
            tvecs.append((w[:, None] * E[np.clip(cands[j][o3], 0, num_entity - 1)]).sum(0))
        T = np.vstack(tvecs)
        for pos, i in enumerate(idxs):
            V = E[np.clip(cands[i], 0, num_entity - 1)] @ T.T
            V = np.delete(V, pos, axis=1)
            fmax[i] = V.max(1).astype(np.float32)
            fmean[i] = V.mean(1).astype(np.float32)
    return fmax, fmean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", choices=["P", "MF"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    is_mf = args.geom == "MF"

    # ---- shared train features (cache = verbatim ranker_basket_ab_ds2 train replay) ----
    D = bag.build_or_load_features(force=False, max_queries=0)
    Xf, lens, off = D["Xf"], D["lens"], D["off"]
    qsrc_tr, qt_tr, qorig_tr, cands_tr = D["qsrc_tr"], D["qt_tr"], D["qorig_tr"], D["cands_tr"]
    num_entity, warm_full, split0, df_raw = D["num_entity"], D["warm_full"], D["split0"], D["df_raw"]
    Q = len(lens)
    yf = np.zeros(off[-1], np.float32); yf[off[1:] - 1] = 1.0
    log(f"train features Q={Q} Xf={Xf.shape} geom={args.geom}")

    # candidate-frequency features for the serve featurize (from the test slate)
    test_df = pd.read_csv(tl.test_csv)[["src", "time"] + tl.c_cols].copy()
    test_df["src"] = test_df["src"].astype(np.int64)
    _allcand = np.clip(test_df[tl.c_cols].values.astype(np.int64).ravel(), 0, num_entity - 1)
    freq_cand = np.bincount(_allcand, minlength=num_entity).astype(np.float64)
    cfreq_log = np.log1p(freq_cand)

    # geometry: train (split0) + serve (full train)
    if is_mf:
        G_tr = mf_geom(split0, num_entity)
        G_sv = mf_geom(df_raw, num_entity)
        fb_tr = lambda qs, qt, cd, s, lab: fb_dense_ent(qs, qt, cd, s, G_tr, num_entity, lab)
        fb_sv = lambda qs, qt, cd, s, lab: fb_dense_ent(qs, qt, cd, s, G_sv, num_entity, lab)
    else:
        P_tr = bag.item_profiles(split0, num_entity)
        P_sv = bag.item_profiles(df_raw, num_entity)
        fb_tr = lambda qs, qt, cd, s, lab: bag.fb_features(qs, qt, cd, s, P_tr, num_entity, labels=lab)
        fb_sv = lambda qs, qt, cd, s, lab: bag.fb_features(qs, qt, cd, s, P_sv, num_entity, labels=lab)
    log("geometry built")

    # crossfit (2-fold src-disjoint rng 7) for train-side stand-in scores
    rng2 = np.random.default_rng(7)
    usrc = np.unique(qsrc_tr).copy(); rng2.shuffle(usrc)
    half = set(usrc[:len(usrc) // 2].tolist())
    fold = np.array([0 if s in half else 1 for s in qsrc_tr])
    qi = {f: np.flatnonzero(fold == f) for f in (0, 1)}
    ro = {f: np.concatenate([np.arange(off[i], off[i + 1]) for i in qi[f]]) for f in (0, 1)}

    def crossfit(Z):
        s_out = [None] * Q
        for f in (0, 1):
            m = lgb.LGBMRanker(**PARAMS)
            m.fit(Z[ro[1 - f]], yf[ro[1 - f]], group=lens[qi[1 - f]].tolist())
            pb = m.predict(Z[ro[f]]); pos = 0
            for i in qi[f]:
                s_out[i] = pb[pos:pos + lens[i]]; pos += lens[i]
        return s_out

    csets_tr = [frozenset(int(c) for c in a) for a in cands_tr]
    train_labels, ntr, npa = bag.structural_labels(qsrc_tr, qt_tr, qorig_tr, csets_tr, warm_full)
    log(f"train labels: triple {ntr} pair {npa} union {len(train_labels)}")

    m1 = lgb.LGBMRanker(**PARAMS); m1.fit(Xf, yf, group=lens.tolist()); log("pass1 fit")
    s1_tr = crossfit(Xf); log("pass1 crossfit")
    f1x, f1m = fb_tr(qsrc_tr, qt_tr, cands_tr, s1_tr, train_labels)
    Zf2 = np.hstack([Xf, np.concatenate(f1x)[:, None], np.concatenate(f1m)[:, None]])
    m2 = lgb.LGBMRanker(**PARAMS); m2.fit(Zf2, yf, group=lens.tolist())
    s2_tr = crossfit(Zf2); log("pass2 fit+crossfit")
    f2x, f2m = fb_tr(qsrc_tr, qt_tr, cands_tr, s2_tr, train_labels)
    Zf3 = np.hstack([Xf, np.concatenate(f2x)[:, None], np.concatenate(f2m)[:, None]])
    m3 = lgb.LGBMRanker(**PARAMS); m3.fit(Zf3, yf, group=lens.tolist()); log("pass3 fit")
    del Zf2, Zf3

    # ---- SERVE on the real test (full-train-frozen features via bag.build_features) ----
    test_srcs = test_df["src"].values.astype(np.int64)
    test_times = test_df["time"].values.astype(float)
    cand_mat = test_df[tl.c_cols].values.astype(np.int64)
    CUTp = float(df_raw["time"].max())
    prod_q = [(int(test_srcs[i]), float(test_times[i]), cand_mat[i]) for i in range(len(test_df))]
    N = len(prod_q)
    tdir = REPO / "outputs"
    log(f"serve featurize N={N} ...")
    Xall_list = bag.build_features(
        df_raw, CUTp, prod_q,
        [tdir / ("dataset2-bpr" + (f"-s{s}" if s != 42 else "")) for s in SEEDS],
        tdir / "dataset2",
        np.unique(test_srcs), num_entity, freq_cand, cfreq_log)
    Xall = np.vstack(Xall_list).astype(np.float32); del Xall_list
    lens_sv = np.array([len(c) for c in cand_mat])   # all 100
    assert Xall.shape == (N * 100, 18)
    hm_all = Xall.reshape(N, 100, 18)[:, :, 9].astype(bool)   # in-hist mask col (feature 9)

    s1_all = m1.predict(Xall).reshape(N, 100)
    test_csets = [frozenset(int(c) for c in row) for row in cand_mat]
    serve_labels, sntr, snpa = bag.structural_labels(test_srcs, test_times, np.arange(N), test_csets, warm_full)
    log(f"serve labels: triple {sntr} pair {snpa} union {len(serve_labels)}")

    cands_sv = list(cand_mat)
    g1x, g1m = fb_sv(test_srcs, test_times, cands_sv, list(s1_all), serve_labels)
    Z2 = np.concatenate([Xall, np.concatenate(g1x)[:, None], np.concatenate(g1m)[:, None]], axis=1)
    s2_all = m2.predict(Z2).reshape(N, 100); del Z2
    g2x, g2m = fb_sv(test_srcs, test_times, cands_sv, list(s2_all), serve_labels)
    Z3 = np.concatenate([Xall, np.concatenate(g2x)[:, None], np.concatenate(g2m)[:, None]], axis=1)
    s3 = m3.predict(Z3).reshape(N, 100); del Z3
    log("serve pass1/2/3 done")

    out = np.zeros((N, 100), np.float64)
    for i in range(N):
        s = s3[i].astype(np.float64)
        if hm_all[i].any():
            s = np.where(hm_all[i], s.min() - 1.0, s)
        lo, hi = s.min(), s.max()
        out[i] = (s - lo) / (hi - lo) if hi > lo else np.full(100, 0.5)
    pd.DataFrame(out).to_csv(args.out, index=False, header=False, float_format="%.6f")
    log(f"WROTE {args.out}  ({N} rows)")


if __name__ == "__main__":
    main()
