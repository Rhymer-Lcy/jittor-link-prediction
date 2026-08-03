# -*- coding: utf-8 -*-
"""dataset2 production base score matrix: three-pass basket ranking over MF geometry.

This is the stage that produces the dataset2 base matrix consumed by
``src/crf_promote.py``. It reuses ``src/ds2_basket_featurizer.py`` for the
18-column feature matrix and its contract-governed cache, and adds the sibling
message geometry.

``--geom`` selects how sibling messages are carried between the rows of one
basket event -- the rows sharing a ``(source, time)`` pair:

--geom MF : the production setting. A rank-128 truncated SVD of the split-0
            source x destination interaction matrix, L2-normalised by row, is
            used as a dense geometry. The factorisation is frozen: split0 for
            the training replay, the full training history for serving. It reads
            no split1 or test truth and no structural labels.
--geom P  : sparse ``item_profiles`` as the geometry. Retained as the control
            against which the MF geometry was introduced as a single variable.

NOTE ON THE NAME "MF": the geometry here is an UNTRAINED SVD factorisation used
only to carry messages between sibling rows. It is unrelated to the BPR-MF
embedding model trained in ``src/train_bpr_jt.py``.

Everything else matches the dataset1 ranking conventions: 18 base features, the
same LambdaRank parameters and seeds, a 2-fold source-disjoint cross-fit for
train-side stand-in scores, and row-min-max plus history-mask serve
normalisation.

Run: DATASET=dataset2 python src/ds2_mf_basket_pack.py --geom MF --out <base.csv>
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

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy.sparse.linalg import svds  # noqa: E402

import ds2_basket_featurizer as bag  # noqa: E402
import pipeline_common as tl  # framework-neutral shared components  # noqa: E402
import stage_contract as sc  # atomic publication helper  # noqa: E402

bag.REPO = REPO

T0 = time.time()
DIM = 128
SEEDS = (42, 123, 777, 2024, 31337)
CUT = 1261958400.0
YEAR = 365.0 * 86400.0
PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[10],
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    random_state=42,
    n_jobs=6,
    verbosity=-1,
    label_gain=[0, 1],
)


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def l2n(E):
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (E / n).astype(np.float32)


def mf_geom(edges, num_entity, dim=DIM, seed=13):
    """d128 unit SVD of the src x dst interaction (the plain-MF basket geometry)."""
    s = edges["src"].to_numpy(np.int64)
    d = edges["dst"].to_numpy(np.int64)
    A = sp.csr_matrix((np.ones(len(s), np.float32), (s, d)), shape=(num_entity, num_entity))
    A.data[:] = 1.0
    k = min(dim, min(A.shape) - 1)
    _, S, Vt = svds(
        A.astype(np.float64), k=k, v0=np.random.default_rng(seed).standard_normal(A.shape[0])
    )
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
            w = np.exp(s[o3] - s[o3].max())
            w /= w.sum()
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

    # ---- shared train features (the contract-governed cut-split replay cache) ----
    D = bag.build_or_load_features(force=False, max_queries=0)
    Xf, lens, off = D["Xf"], D["lens"], D["off"]
    qsrc_tr, qt_tr, qorig_tr, cands_tr = D["qsrc_tr"], D["qt_tr"], D["qorig_tr"], D["cands_tr"]
    num_entity, warm_full, split0, df_raw = (
        D["num_entity"],
        D["warm_full"],
        D["split0"],
        D["df_raw"],
    )
    Q = len(lens)
    yf = np.zeros(off[-1], np.float32)
    yf[off[1:] - 1] = 1.0
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

        def fb_tr(qs, qt, cd, scores, labels):
            return fb_dense_ent(qs, qt, cd, scores, G_tr, num_entity, labels)

        def fb_sv(qs, qt, cd, scores, labels):
            return fb_dense_ent(qs, qt, cd, scores, G_sv, num_entity, labels)

    else:
        P_tr = bag.item_profiles(split0, num_entity)
        P_sv = bag.item_profiles(df_raw, num_entity)

        def fb_tr(qs, qt, cd, scores, labels):
            return bag.fb_features(qs, qt, cd, scores, P_tr, num_entity, labels=labels)

        def fb_sv(qs, qt, cd, scores, labels):
            return bag.fb_features(qs, qt, cd, scores, P_sv, num_entity, labels=labels)

    log("geometry built")

    # crossfit (2-fold src-disjoint rng 7) for train-side stand-in scores
    rng2 = np.random.default_rng(7)
    usrc = np.unique(qsrc_tr).copy()
    rng2.shuffle(usrc)
    half = set(usrc[: len(usrc) // 2].tolist())
    fold = np.array([0 if s in half else 1 for s in qsrc_tr])
    qi = {f: np.flatnonzero(fold == f) for f in (0, 1)}
    ro = {f: np.concatenate([np.arange(off[i], off[i + 1]) for i in qi[f]]) for f in (0, 1)}

    def crossfit(Z):
        s_out = [None] * Q
        for f in (0, 1):
            m = lgb.LGBMRanker(**PARAMS)
            m.fit(Z[ro[1 - f]], yf[ro[1 - f]], group=lens[qi[1 - f]].tolist())
            pb = m.predict(Z[ro[f]])
            pos = 0
            for i in qi[f]:
                s_out[i] = pb[pos : pos + lens[i]]
                pos += lens[i]
        return s_out

    csets_tr = [frozenset(int(c) for c in a) for a in cands_tr]
    train_labels, ntr, npa = bag.structural_labels(qsrc_tr, qt_tr, qorig_tr, csets_tr, warm_full)
    log(f"train labels: triple {ntr} pair {npa} union {len(train_labels)}")

    m1 = lgb.LGBMRanker(**PARAMS)
    m1.fit(Xf, yf, group=lens.tolist())
    log("pass1 fit")
    s1_tr = crossfit(Xf)
    log("pass1 crossfit")
    f1x, f1m = fb_tr(qsrc_tr, qt_tr, cands_tr, s1_tr, train_labels)
    Zf2 = np.hstack([Xf, np.concatenate(f1x)[:, None], np.concatenate(f1m)[:, None]])
    m2 = lgb.LGBMRanker(**PARAMS)
    m2.fit(Zf2, yf, group=lens.tolist())
    s2_tr = crossfit(Zf2)
    log("pass2 fit+crossfit")
    f2x, f2m = fb_tr(qsrc_tr, qt_tr, cands_tr, s2_tr, train_labels)
    Zf3 = np.hstack([Xf, np.concatenate(f2x)[:, None], np.concatenate(f2m)[:, None]])
    m3 = lgb.LGBMRanker(**PARAMS)
    m3.fit(Zf3, yf, group=lens.tolist())
    log("pass3 fit")
    del Zf2, Zf3

    # ---- SERVE on the real test (full-train-frozen features via bag.build_features) ----
    test_srcs = test_df["src"].values.astype(np.int64)
    test_times = test_df["time"].values.astype(float)
    cand_mat = test_df[tl.c_cols].values.astype(np.int64)
    CUTp = float(df_raw["time"].max())
    prod_q = [(int(test_srcs[i]), float(test_times[i]), cand_mat[i]) for i in range(len(test_df))]
    N = len(prod_q)
    log(f"serve featurize N={N} ...")
    Xall_list = bag.build_features(
        df_raw,
        CUTp,
        prod_q,
        # Same contract as the LINE directory below: a literal path here would
        # ignore OUTPUTS_ROOT and silently read from the wrong root.
        [tl.bpr_run_dir("dataset2", seed=s) for s in SEEDS],
        # The LINE run directory is named by pipeline_common, not spelled out
        # here. The trainer writes the serve embedding to a run directory derived
        # from its scientific parameters, so a literal would resolve to a stale
        # directory and the stage would fail on the missing export. line_run_dir
        # is the same call the producer (train_line_jt) makes.
        tl.line_run_dir("dataset2"),
        np.unique(test_srcs),
        num_entity,
        freq_cand,
        cfreq_log,
    )
    Xall = np.vstack(Xall_list).astype(np.float32)
    del Xall_list
    assert Xall.shape == (N * 100, 18)
    hm_all = Xall.reshape(N, 100, 18)[:, :, 9].astype(bool)  # in-hist mask col (feature 9)

    s1_all = m1.predict(Xall).reshape(N, 100)
    test_csets = [frozenset(int(c) for c in row) for row in cand_mat]
    serve_labels, sntr, snpa = bag.structural_labels(
        test_srcs, test_times, np.arange(N), test_csets, warm_full
    )
    log(f"serve labels: triple {sntr} pair {snpa} union {len(serve_labels)}")

    cands_sv = list(cand_mat)
    g1x, g1m = fb_sv(test_srcs, test_times, cands_sv, list(s1_all), serve_labels)
    Z2 = np.concatenate([Xall, np.concatenate(g1x)[:, None], np.concatenate(g1m)[:, None]], axis=1)
    s2_all = m2.predict(Z2).reshape(N, 100)
    del Z2
    g2x, g2m = fb_sv(test_srcs, test_times, cands_sv, list(s2_all), serve_labels)
    Z3 = np.concatenate([Xall, np.concatenate(g2x)[:, None], np.concatenate(g2m)[:, None]], axis=1)
    s3 = m3.predict(Z3).reshape(N, 100)
    del Z3
    log("serve pass1/2/3 done")

    out = np.zeros((N, 100), np.float64)
    for i in range(N):
        s = s3[i].astype(np.float64)
        if hm_all[i].any():
            s = np.where(hm_all[i], s.min() - 1.0, s)
        lo, hi = s.min(), s.max()
        out[i] = (s - lo) / (hi - lo) if hi > lo else np.full(100, 0.5)
    # Atomic publication: identical formatter, identical bytes. This stage costs
    # ~2.5 h, so a truncated base matrix left under the real name is expensive
    # in exactly the way the completion contract exists to prevent.
    with sc.atomic_output(args.out) as staged:
        pd.DataFrame(out).to_csv(staged, index=False, header=False, float_format="%.6f")
    log(f"WROTE {args.out}  ({N} rows)")


if __name__ == "__main__":
    main()
