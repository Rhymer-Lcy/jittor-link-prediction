# GRADUATED 2026-07-29 from scratchpad/archive/ds2_closed_veins/bagging_ensemble_ds2.py
# (source sha256 bc6f7fd889915f2ec86c23e8fc0af8e135c5cdd797e14d0d215adf99c0133fc1).
# Body is VERBATIM apart from the stale run-hint path in the docstring; REPO resolves
# correctly from src/. Only the featurizer helpers are on the production path:
# build_or_load_features, build_features, item_profiles, fb_features, structural_labels.
# The bagging CLI below is a CLOSED experiment, retained verbatim and never run.
# -*- coding: utf-8 -*-
"""Query-bootstrap bagging ensemble PILOT for the ds2 basket ranker.

Strict, additive probe: it does NOT modify any live source file. It replicates
the train-side replay of src/ranker_basket_ab_ds2.py (the "label" variant that
produced the shipped ds2 base) function-for-function, then wraps a bagging
ensemble around it exactly as specified:

  * bootstrap unit = a COMPLETE ranking query (all candidates), never a single
    candidate row -- Poisson(1) group weights;
  * each member: distinct seed, feature_fraction=0.8, every other LightGBM
    parameter identical to production;
  * MRR is measured out-of-fold on the same 2-fold src-disjoint split
    (default_rng(7)) the project's gate uses; truth = slate column -1.

Two gate levels:
  --pass1-only  (FAST pilot, GPT's original base-18 gate level): bag the pass-1
                base ranker. Queries are independent at pass-1, so this is a
                clean, cheap direction call and the arm deltas are exact.
  (default)     full 3-pass: each pass trained per member, then a CONSENSUS
                (row-min-max per query, averaged over members) builds the next
                pass's basket feedback (fb_max/fb_mean). Faithful to production;
                heavy (244k queries x 3 passes x members x 2 folds).

Four arms, one shared OOF protocol, pre-CRF base ranker score:
  base       single production model (no bootstrap, full features) -- the anchor
  identical  M copies of the base model averaged -- MUST equal base ranking
  seedonly   M members, distinct seeds + feature_fraction=0.8, NO bootstrap
  bootstrap  M members, distinct seeds + feature_fraction=0.8 + Poisson(1) query
             bootstrap -- the candidate

Kill criteria (GPT round-13 spec), judged on this pre-CRF gate:
  * bootstrap must beat seedonly by >= +0.002, else the query bootstrap is inert;
  * bootstrap must beat base by >= +0.006 to justify expanding to 8 members;
  * >= +0.010 to justify building a ds2-only aux-account A/B pack.

The absolute MRR is NOT expected to equal the old base-18 gate anchor (0.6123):
that gate ran on a CUT-frozen 60k population, this runs on the test-pool split1
tail (244k). Only the RELATIVE arm deltas are load-bearing, and all arms share
one protocol, so the comparison is exact.

Run (fast pilot):
  DATASET=dataset2 python \
      src/ds2_basket_featurizer.py --members 4 --pass1-only
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("DATASET", "dataset2")
if os.environ["DATASET"] != "dataset2":
    raise RuntimeError("this pilot is dataset2-only")
os.environ.setdefault("DATA_PACK", "data_A")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import train_line as tl
import ensemble_predict as ep

assert tl.DATASET == "dataset2", "this pipeline is dataset2-only"

SEEDS = (42, 123, 777, 2024, 31337)  # the five production BPR embedding seeds
CUT = 1261958400.0
W = {"collab": 0.25, "rpop": 1.25, "icfm": 0.5, "icf3": 1.0, "bpr": 0.7}
YEAR = 365.0 * 86400.0
RNG = np.random.default_rng(42)  # identical negative-sampling RNG as the live build
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10], n_estimators=400,
              learning_rate=0.05, num_leaves=31, min_child_samples=100, random_state=42,
              n_jobs=6, verbosity=-1, label_gain=[0, 1])
FEATURE_FRACTION = 0.8
CACHE = HERE / "bagging_cache"
METRICS = HERE / "bagging_ensemble_ds2_metrics.json"
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ---- feature/label machinery copied verbatim from ranker_basket_ab_ds2.py ----
def popwin(dvals, tvals, hi, lo, frac, num_entity):
    m = tvals >= (hi - frac * (hi - lo))
    return np.log1p(np.bincount(dvals[m], minlength=num_entity).astype(np.float64))


def item_profiles(edges, num_entity):
    P = sp.csr_matrix((np.ones(len(edges), np.float32),
                       (edges.dst.values.astype(np.int64), edges.src.values.astype(np.int64))),
                      shape=(num_entity, num_entity))
    P.data[:] = 1.0
    nrm = np.sqrt(P.multiply(P).sum(1)).A.ravel(); nrm[nrm == 0] = 1.0
    return (sp.diags(1.0 / nrm) @ P).tocsr()


def structural_labels(qsrc, qt, qorig, csets, warm):
    n = len(qsrc)
    trip, ambig = {}, set()
    for i in range(n - 2):
        if not (qorig[i + 1] == qorig[i] + 1 and qorig[i + 2] == qorig[i] + 2):
            continue
        if not (qt[i] == qt[i + 1] == qt[i + 2]):
            continue
        if len({qsrc[i], qsrc[i + 1], qsrc[i + 2]}) != 3:
            continue
        common = {x for x in (csets[i] & csets[i + 1] & csets[i + 2]) if warm[x]}
        if len(common) != 1:
            continue
        x = next(iter(common))
        for r in (i, i + 1, i + 2):
            if r in trip and trip[r] != x:
                ambig.add(r)
            else:
                trip[r] = x
    trip = {r: x for r, x in trip.items() if r not in ambig}
    pair, pambig = {}, set()
    for i in range(n - 1):
        if not (qorig[i + 1] == qorig[i] + 1 and qt[i] == qt[i + 1] and qsrc[i] != qsrc[i + 1]):
            continue
        if i in trip or (i + 1) in trip:
            continue
        common = {x for x in (csets[i] & csets[i + 1]) if warm[x]}
        if len(common) != 1:
            continue
        x = next(iter(common))
        for r in (i, i + 1):
            if r in pair and pair[r] != x:
                pambig.add(r)
            else:
                pair[r] = x
    pair = {r: x for r, x in pair.items() if r not in pambig}
    lab = dict(pair); lab.update(trip)
    return lab, len(trip), len(pair)


def fb_features(qsrc, qt, cands, s1, P, num_entity, labels=None):
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
                tvecs.append(P[np.clip(np.array([lab], np.int64), 0, num_entity - 1)])
                continue
            s = np.asarray(s1[j], np.float64)
            o3 = np.argsort(-s)[:3]
            w = np.exp(s[o3] - s[o3].max()); w /= w.sum()
            tvecs.append(sp.csr_matrix(w[None, :], dtype=np.float32) @
                         P[np.clip(cands[j][o3], 0, num_entity - 1)])
        T = sp.vstack(tvecs).T.tocsc()
        for pos, i in enumerate(idxs):
            V = (P[np.clip(cands[i], 0, num_entity - 1)] @ T).toarray()
            V = np.delete(V, pos, axis=1)
            fmax[i] = V.max(1).astype(np.float32)
            fmean[i] = V.mean(1).astype(np.float32)
    return fmax, fmean


def build_features(struct_df, freeze_t, queries, bpr_dirs, line_dir, need_srcs, num_entity,
                   freq_cand, cfreq_log):
    tl.build_history_index(struct_df)
    tl.build_cooc(struct_df, num_entity)
    base_cache = ep.build_base_cache(struct_df)
    cache_mat = tl.cache_dict_to_matrix(base_cache, set(), num_entity - 1)
    dst_pop_log = np.log1p(tl.dst_pop); rpop = tl.dst_rpop_log.copy(); dst_pop_raw = tl.dst_pop.copy()
    tmin = float(struct_df["time"].min())
    gr = np.zeros(num_entity)
    for d, tv in struct_df.groupby("dst")["time"].max().items():
        gr[int(d)] = (float(tv) - tmin) / (freeze_t - tmin)
    seen = (np.bincount(struct_df["dst"].values, minlength=num_entity) > 0).astype(np.float64)
    rps = popwin(struct_df["dst"].values, struct_df["time"].values, freeze_t, tmin, 0.05, num_entity)
    rpl = popwin(struct_df["dst"].values, struct_df["time"].values, freeze_t, tmin, 0.40, num_entity)
    bprs = [np.load(d / "bpr_emb.npy") for d in bpr_dirs]
    emb = ep.load_embedding(line_dir, expected_rows=num_entity)
    emb_n = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    tl.build_sim_cache(emb, need_srcs)
    srcs = [s for s, _, _ in queries]; cands = [c for _, _, c in queries]
    collab_rows = ep.grouped_collab(cache_mat, srcs, cands)
    log(f"build_features: caches ready, per-query loop over {len(queries)} queries")
    X = []
    for i, (src, t, cc_) in enumerate(queries):
        if i and i % 20000 == 0:
            log(f"  featurize {i}/{len(queries)}")
        cc = np.clip(cc_, 0, num_entity - 1); m = len(cc_)
        hist_d, _ = tl.get_hist_before_time(int(src), freeze_t + 1)
        f_collab = tl.rownorm(collab_rows[i].astype(np.float64)); f_rpop = tl.rownorm(rpop[cc])
        f_icfm = np.zeros(m); f_icf3 = np.zeros(m)
        if len(hist_d) > 0:
            hvec = emb_n[np.clip(hist_d, 0, num_entity - 1)]
            sim = np.sort(emb_n[cc] @ hvec.T, axis=1)
            f_icfm = tl.rownorm(np.maximum(sim.mean(axis=1), 0.0))
            kk = min(3, sim.shape[1]); f_icf3 = tl.rownorm(np.maximum(sim[:, -kk:].mean(axis=1), 0.0))
        f_bpr = np.zeros(m)
        for be in bprs:
            f_bpr += tl.rownorm(np.maximum(be[cc] @ be[min(int(src), be.shape[0] - 1)], 0.0))
        f_bpr /= len(bprs)
        blend = (W["collab"] * f_collab + W["rpop"] * f_rpop + W["icfm"] * f_icfm
                 + W["icf3"] * f_icf3 + W["bpr"] * f_bpr)
        hm = np.isin(cc_, hist_d)
        rs = tl.rownorm(rps[cc]); rl = tl.rownorm(rpl[cc])
        f_cfreq = tl.rownorm(cfreq_log[cc]); f_popratio = tl.rownorm(dst_pop_raw[cc] / (freq_cand[cc] + 1.0))
        X.append(np.column_stack([f_collab, f_rpop, f_icfm, f_icf3, f_bpr, blend, gr[cc], dst_pop_log[cc],
                                  np.full(m, len(hist_d)), hm.astype(float), seen[cc],
                                  np.full(m, (t - freeze_t) / YEAR), np.full(m, m), rs, rl, rs - rl,
                                  f_cfreq, f_popratio]))
    return X


# ---- pilot helpers ----
def reciprocal_ranks(score_list):
    rr = np.empty(len(score_list))
    for i, s in enumerate(score_list):
        s = np.asarray(s, np.float64)
        pos = s[-1]; neg = s[:-1]
        rr[i] = 1.0 / (1 + int((neg > pos).sum()) + int((neg == pos).sum()))
    return rr


def row_minmax_list(score_list):
    out = []
    for s in score_list:
        s = np.asarray(s, np.float64)
        lo, hi = s.min(), s.max()
        out.append((s - lo) / (hi - lo) if hi > lo else np.full(len(s), 0.5))
    return out


def consensus(member_score_lists):
    """Row-min-max each member per query, then average across members."""
    normed = [row_minmax_list(s) for s in member_score_lists]
    Q = len(member_score_lists[0])
    return [np.mean([normed[m][q] for m in range(len(normed))], axis=0) for q in range(Q)]


def top1(score_list):
    return np.array([int(np.argmax(s)) for s in score_list])


def build_or_load_features(force, max_queries):
    """Build (and cache) the expensive train-side feature matrix, or load it.

    Cached: Xf, yf, lens, qsrc_tr, qt_tr, qorig_tr, cands_concat, num_entity.
    The light df-derived objects (split0, warm_full, train_labels, P_tr) are
    rebuilt each run -- they are cheap next to the per-query feature loop.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = "all" if not max_queries else f"q{max_queries}"
    feat = CACHE / f"train_features_{tag}.npz"

    df_raw = pd.read_csv(tl.train_csv).drop_duplicates(subset=["src", "dst", "time"]).reset_index(drop=True)
    df_raw["src"] = df_raw["src"].astype(np.int64); df_raw["dst"] = df_raw["dst"].astype(np.int64)
    df_raw["time"] = df_raw["time"].astype(float)
    test_df = pd.read_csv(tl.test_csv)[["src", "time"] + tl.c_cols].copy()
    test_df["src"] = test_df["src"].astype(np.int64)
    num_entity = int(max(df_raw.src.max(), df_raw.dst.max())) + 1
    warm_full = np.zeros(num_entity, bool); warm_full[df_raw["dst"].values.astype(np.int64)] = True
    split0 = df_raw[df_raw["time"] <= CUT].reset_index(drop=True)

    if feat.is_file() and not force:
        log(f"loading cached features {feat}")
        z = np.load(feat, allow_pickle=False)
        Xf = z["Xf"]; yf = z["yf"]; lens = z["lens"]
        qsrc_tr = z["qsrc_tr"]; qt_tr = z["qt_tr"]; qorig_tr = z["qorig_tr"]
        cands_concat = z["cands_concat"]
        assert int(z["num_entity"]) == num_entity
        off = np.concatenate([[0], np.cumsum(lens)])
        cands_tr = [cands_concat[off[i]:off[i + 1]] for i in range(len(lens))]
        return dict(df_raw=df_raw, split0=split0, num_entity=num_entity, warm_full=warm_full,
                    Xf=Xf, yf=yf, lens=lens, off=off, qsrc_tr=qsrc_tr, qt_tr=qt_tr,
                    qorig_tr=qorig_tr, cands_tr=cands_tr)

    _allcand = np.clip(test_df[tl.c_cols].values.astype(np.int64).ravel(), 0, num_entity - 1)
    freq_cand = np.bincount(_allcand, minlength=num_entity).astype(np.float64)
    cfreq_log = np.log1p(freq_cand)

    split1 = df_raw[df_raw["time"] > CUT].reset_index(drop=True)
    pools = {}
    cand_mat = test_df[tl.c_cols].values.astype(np.int64)
    for i, s in enumerate(test_df["src"].values.astype(np.int64)):
        pools.setdefault(int(s), set()).update(cand_mat[i].tolist())
    lab = split1[split1["src"].isin(pools.keys())].reset_index()
    train_q, qorig_tr = [], []
    for row in lab.itertuples(index=False):
        src, dst, t = int(row.src), int(row.dst), float(row.time)
        pool = pools.get(src); negs = sorted(pool - {dst, src}) if pool else []
        if not negs:
            continue
        if len(negs) > ep.NEG_PER_SAMPLE:
            negs = RNG.choice(negs, ep.NEG_PER_SAMPLE, replace=False).tolist()
        train_q.append((src, t, np.array(negs + [dst], dtype=np.int64)))
        qorig_tr.append(int(row.index))
    if max_queries and len(train_q) > max_queries:
        # SAFE only at pass-1 (queries independent). Subsample whole queries
        # reproducibly; pass-1 MRR/deltas are unbiased under paired arms.
        sel = np.random.default_rng(20260726).choice(len(train_q), max_queries, replace=False)
        sel.sort()
        train_q = [train_q[i] for i in sel]
        qorig_tr = [qorig_tr[i] for i in sel]
    qorig_tr = np.array(qorig_tr, np.int64)
    log(f"train queries {len(train_q)}")

    tdir = REPO / "outputs"
    Xtr = build_features(split0, CUT, train_q,
                         [tdir / ("dataset2-bpr-tmax1.26196e+09" + (f"-s{s}" if s != 42 else "")) for s in SEEDS],
                         tdir / "dataset2-novirt-tmax1.26196e+09",
                         np.unique(np.array([s for s, _, _ in train_q], dtype=np.int64)),
                         num_entity, freq_cand, cfreq_log)
    lens = np.array([len(q[2]) for q in train_q])
    off = np.concatenate([[0], np.cumsum(lens)])
    Xf = np.vstack(Xtr).astype(np.float32); del Xtr
    yf = np.zeros(off[-1], np.float32); yf[off[1:] - 1] = 1.0
    qsrc_tr = np.array([q[0] for q in train_q], np.int64)
    qt_tr = np.array([q[1] for q in train_q], np.float64)
    cands_concat = np.concatenate([q[2] for q in train_q]).astype(np.int64)
    log(f"train matrix {Xf.shape}; caching -> {feat}")
    np.savez(feat, Xf=Xf, yf=yf, lens=lens, qsrc_tr=qsrc_tr, qt_tr=qt_tr,
             qorig_tr=qorig_tr, cands_concat=cands_concat, num_entity=np.array([num_entity]))
    cands_tr = [q[2] for q in train_q]
    return dict(df_raw=df_raw, split0=split0, num_entity=num_entity, warm_full=warm_full,
                Xf=Xf, yf=yf, lens=lens, off=off, qsrc_tr=qsrc_tr, qt_tr=qt_tr,
                qorig_tr=qorig_tr, cands_tr=cands_tr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--pass1-only", action="store_true",
                    help="fast base-ranker gate (GPT's base-18 level); no 3-pass basket feedback")
    ap.add_argument("--max-queries", type=int, default=0,
                    help="subsample train queries (pass1-only ONLY; corrupts basket structure otherwise)")
    ap.add_argument("--force", action="store_true", help="rebuild cached features")
    ap.add_argument("--out", type=Path, default=METRICS)
    args = ap.parse_args()
    if args.max_queries and not args.pass1_only:
        ap.error("--max-queries is only valid with --pass1-only (it breaks basket sibling structure)")
    M = args.members
    member_seeds = list(SEEDS[:M]) if M <= len(SEEDS) else \
        list(SEEDS) + [1_000 + i for i in range(M - len(SEEDS))]

    D = build_or_load_features(args.force, args.max_queries)
    Xf, yf, lens, off = D["Xf"], D["yf"], D["lens"], D["off"]
    qsrc_tr, qt_tr, qorig_tr, cands_tr = D["qsrc_tr"], D["qt_tr"], D["qorig_tr"], D["cands_tr"]
    num_entity, warm_full, split0 = D["num_entity"], D["warm_full"], D["split0"]
    Q = len(lens)
    log(f"features ready: Q={Q}  Xf={Xf.shape}  members={M} seeds={member_seeds}  pass1_only={args.pass1_only}")

    rng2 = np.random.default_rng(7)
    usrc = np.unique(qsrc_tr).copy(); rng2.shuffle(usrc)
    half = set(usrc[:len(usrc) // 2].tolist())
    fold = np.array([0 if s in half else 1 for s in qsrc_tr])
    rows_of = {f: np.concatenate([np.arange(off[i], off[i + 1]) for i in np.flatnonzero(fold == f)])
               for f in (0, 1)}
    qidx_of = {f: np.flatnonzero(fold == f) for f in (0, 1)}

    def poisson_row_weights(seed):
        wq = np.random.default_rng(seed).poisson(1.0, size=Q).astype(np.float64)
        return np.repeat(wq, lens), float(wq.mean())

    def fit_oof(Z, seed, colsample, row_w):
        params = dict(PARAMS); params["random_state"] = int(seed)
        if colsample is not None:
            params.update(colsample_bytree=float(colsample), feature_fraction_seed=int(seed),
                          bagging_seed=int(seed))
        s_out = [None] * Q
        for f in (0, 1):
            tr_rows, te_rows = rows_of[1 - f], rows_of[f]
            w = row_w[tr_rows] if row_w is not None else None
            mdl = lgb.LGBMRanker(**params)
            mdl.fit(Z[tr_rows], yf[tr_rows], group=lens[qidx_of[1 - f]].tolist(), sample_weight=w)
            pb = mdl.predict(Z[te_rows]); pos = 0
            for i in qidx_of[f]:
                s_out[i] = pb[pos:pos + lens[i]]; pos += lens[i]
        return s_out

    def member_disagreement(member_s):
        t1 = np.stack([top1(member_s[m]) for m in range(len(member_s))], axis=1)
        return float(np.mean([len(set(t1[q].tolist())) > 1 for q in range(Q)]))

    if args.pass1_only:
        log("PASS-1 gate: base ...")
        base_s = fit_oof(Xf, 42, None, None)
        base_c = consensus([base_s])
        ident_c = consensus([base_s] * M)  # identical members -> reuse, no refit
        log("PASS-1 gate: seedonly ...")
        seed_members = [fit_oof(Xf, s, FEATURE_FRACTION, None) for s in member_seeds]
        seed_c = consensus(seed_members)
        log("PASS-1 gate: bootstrap ...")
        boot_members, wq_means = [], []
        for s in member_seeds:
            rw, wqm = poisson_row_weights(s)
            boot_members.append(fit_oof(Xf, s, FEATURE_FRACTION, rw)); wq_means.append(wqm)
        boot_c = consensus(boot_members)
        seed_disagree = member_disagreement(seed_members)
        boot_disagree = member_disagreement(boot_members)
        # Dump OOF per-query scores so the confidence-gate sweep is instant
        # (no refits). Concatenated in query order; lens + fold reconstruct.
        np.savez(CACHE / "pass1_oof_scores.npz",
                 base=np.concatenate([np.asarray(s, np.float64) for s in base_c]),
                 seed=np.concatenate([np.asarray(s, np.float64) for s in seed_c]),
                 boot=np.concatenate([np.asarray(s, np.float64) for s in boot_c]),
                 lens=lens, fold=fold)
        log(f"dumped pass1 OOF scores -> {CACHE / 'pass1_oof_scores.npz'}")
    else:
        P_tr = item_profiles(split0, num_entity)
        csets_tr = [frozenset(int(c) for c in a) for a in cands_tr]
        train_labels, ntr, npa = structural_labels(qsrc_tr, qt_tr, qorig_tr, csets_tr, warm_full)
        log(f"train labels: triple {ntr} pair {npa} union {len(train_labels)}")

        def run_arm(members):
            m1 = [fit_oof(Xf, mm["seed"], mm["colsample"], mm["row_w"]) for mm in members]
            c1 = consensus(m1)
            f1max, f1mean = fb_features(qsrc_tr, qt_tr, cands_tr, c1, P_tr, num_entity, labels=train_labels)
            Z2 = np.hstack([Xf, np.concatenate(f1max)[:, None], np.concatenate(f1mean)[:, None]])
            m2 = [fit_oof(Z2, mm["seed"], mm["colsample"], mm["row_w"]) for mm in members]
            c2 = consensus(m2)
            f2max, f2mean = fb_features(qsrc_tr, qt_tr, cands_tr, c2, P_tr, num_entity, labels=train_labels)
            Z3 = np.hstack([Xf, np.concatenate(f2max)[:, None], np.concatenate(f2mean)[:, None]])
            m3 = [fit_oof(Z3, mm["seed"], mm["colsample"], mm["row_w"]) for mm in members]
            return consensus(m3), m3

        log("3-PASS: base ...")
        base_c, base_m = run_arm([dict(seed=42, colsample=None, row_w=None)])
        log("3-PASS: identical ...")
        ident_c, _ = run_arm([dict(seed=42, colsample=None, row_w=None) for _ in range(M)])
        log("3-PASS: seedonly ...")
        seed_c, seed_members = run_arm([dict(seed=s, colsample=FEATURE_FRACTION, row_w=None) for s in member_seeds])
        log("3-PASS: bootstrap ...")
        boot_specs, wq_means = [], []
        for s in member_seeds:
            rw, wqm = poisson_row_weights(s)
            boot_specs.append(dict(seed=s, colsample=FEATURE_FRACTION, row_w=rw)); wq_means.append(wqm)
        boot_c, boot_members = run_arm(boot_specs)
        seed_disagree = member_disagreement(seed_members)
        boot_disagree = member_disagreement(boot_members)

    base_t1 = top1(base_c)
    rr_base = reciprocal_ranks(base_c)
    rr_seed = reciprocal_ranks(seed_c)
    rr_boot = reciprocal_ranks(boot_c)
    mrr_base, mrr_ident = float(rr_base.mean()), float(reciprocal_ranks(ident_c).mean())
    mrr_seed, mrr_boot = float(rr_seed.mean()), float(rr_boot.mean())
    ident_ok = bool((top1(ident_c) == base_t1).all())

    base_truth_rank = np.array([1 + int((np.asarray(s)[:-1] > np.asarray(s)[-1]).sum())
                                + int((np.asarray(s)[:-1] == np.asarray(s)[-1]).sum()) for s in base_c])
    low = (base_truth_rank >= 2) & (base_truth_rank <= 10)

    d_base = mrr_boot - mrr_base
    d_seed = mrr_boot - mrr_seed
    verdict = {
        "identical_control_passes": ident_ok,
        "bootstrap_beats_seedonly_by_0.002": bool(d_seed >= 0.002),
        "bootstrap_beats_base_by_0.006": bool(d_base >= 0.006),
        "bootstrap_beats_base_by_0.010": bool(d_base >= 0.010),
        "decision": ("BUILD-AUX-PACK" if d_base >= 0.010 and d_seed >= 0.002 else
                     "EXPAND-TO-8" if d_base >= 0.006 and d_seed >= 0.002 else "STOP"),
    }
    reports = {
        "mode": "pass1-only" if args.pass1_only else "full-3-pass",
        "queries": Q, "members": M, "seeds": member_seeds, "feature_fraction": FEATURE_FRACTION,
        "arms": {
            "base": {"mrr": mrr_base},
            "identical": {"mrr": mrr_ident, "top1_identical_to_base": ident_ok},
            "seedonly": {"mrr": mrr_seed, "delta_vs_base": mrr_seed - mrr_base,
                         "member_top1_disagreement": seed_disagree},
            "bootstrap": {"mrr": mrr_boot, "delta_vs_base": d_base, "delta_vs_seedonly": d_seed,
                          "member_top1_disagreement": boot_disagree,
                          "top1_changed_vs_base": int((top1(boot_c) != base_t1).sum()),
                          "poisson_weight_mean_per_member": wq_means},
        },
        "low_confidence_subset": {
            "definition": "base truth rank in [2,10]", "n": int(low.sum()),
            "base_mrr": float(rr_base[low].mean()) if low.any() else None,
            "bootstrap_mrr": float(rr_boot[low].mean()) if low.any() else None,
            "bootstrap_delta": float(rr_boot[low].mean() - rr_base[low].mean()) if low.any() else None,
        },
        "verdict": verdict,
    }
    args.out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    log(f"base {mrr_base:.6f} | ident {mrr_ident:.6f}(ok={ident_ok}) | "
        f"seed {mrr_seed:.6f}({mrr_seed - mrr_base:+.6f}) | boot {mrr_boot:.6f}"
        f"(vs_base {d_base:+.6f}, vs_seed {d_seed:+.6f})")
    log(f"VERDICT: {verdict['decision']}  ; wrote {args.out}")


if __name__ == "__main__":
    main()
