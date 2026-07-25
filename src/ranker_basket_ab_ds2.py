# -*- coding: utf-8 -*-
"""A/B builder for the label-augmented basket feedback (candidate 1).

Faithful copy of ranker_basket_ds2.py that, in ONE run sharing pass-1 and the
(expensive) featurize, produces TWO ds2 base CSVs differing only in the basket
sibling stand-in:

  top3  : sibling stand-in = softmax-top3 of the sibling's pass score (shipped).
  label : where a sibling carries a triple/pair structural label, its stand-in
          is the LABEL one-hot (the sibling's near-certain true answer); else top3.

Replay proved label beats top3 as a ranker feature by +0.0104 (base sibling
top-1 accuracy is only 40.8%, so the label sharpens the similarity message).
Both bases go through the same crf_promote zrxst config; the aux-account
single-dataset A/B (top3 vs label ds2) isolates the label gain with no base
drift. Outputs: outputs/dataset2-ranker/ranker_basket3_{top3,label}_dataset2.csv.

Train-side labels use the deduped-train raw order (split1 original positions,
same-time consecutive adjacency) exactly as the replay; serve-side labels use
the test-file raw order. Run: DATASET=dataset2 python src/ranker_basket_ab_ds2.py
"""
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import train_line as tl
import ensemble_predict as ep

assert tl.DATASET == "dataset2", "this pipeline is dataset2-only"
SEEDS = (42, 123, 777, 2024, 31337)
CUT = 1261958400.0
W = {"collab": 0.25, "rpop": 1.25, "icfm": 0.5, "icf3": 1.0, "bpr": 0.7}
YEAR = 365.0 * 86400.0
RNG = np.random.default_rng(42)
OUT_DIR = tl.PROJECT_ROOT / "outputs" / "dataset2-ranker"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_T3 = str(OUT_DIR / "ranker_basket3_top3_dataset2.csv")
OUT_LB = str(OUT_DIR / "ranker_basket3_label_dataset2.csv")
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10], n_estimators=400,
              learning_rate=0.05, num_leaves=31, min_child_samples=100, random_state=42,
              n_jobs=6, verbosity=-1, label_gain=[0, 1])
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


df_raw = pd.read_csv(tl.train_csv).drop_duplicates(subset=["src", "dst", "time"]).reset_index(drop=True)
df_raw["src"] = df_raw["src"].astype(np.int64); df_raw["dst"] = df_raw["dst"].astype(np.int64)
df_raw["time"] = df_raw["time"].astype(float)
test_df = pd.read_csv(tl.test_csv)[["src", "time"] + tl.c_cols].copy()
test_df["src"] = test_df["src"].astype(np.int64)
num_entity = int(max(df_raw.src.max(), df_raw.dst.max())) + 1
warm_full = np.zeros(num_entity, bool); warm_full[df_raw["dst"].values.astype(np.int64)] = True

_allcand = np.clip(test_df[tl.c_cols].values.astype(np.int64).ravel(), 0, num_entity - 1)
freq_cand = np.bincount(_allcand, minlength=num_entity).astype(np.float64)
cfreq_log = np.log1p(freq_cand)


def popwin(dvals, tvals, hi, lo, frac):
    m = tvals >= (hi - frac * (hi - lo))
    return np.log1p(np.bincount(dvals[m], minlength=num_entity).astype(np.float64))


def item_profiles(edges):
    P = sp.csr_matrix((np.ones(len(edges), np.float32),
                       (edges.dst.values.astype(np.int64), edges.src.values.astype(np.int64))),
                      shape=(num_entity, num_entity))
    P.data[:] = 1.0
    nrm = np.sqrt(P.multiply(P).sum(1)).A.ravel(); nrm[nrm == 0] = 1.0
    return (sp.diags(1.0 / nrm) @ P).tocsr()


def structural_labels(qsrc, qt, qorig, csets, warm):
    """triple + pair answer labels {query_i: answer_id} via raw-order same-time
    adjacency (qorig = original file position; consecutive positions only)."""
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
    lab = dict(pair); lab.update(trip)  # triple wins
    return lab, len(trip), len(pair)


def fb_features(qsrc, qt, cands, s1, P, labels=None):
    """Sibling-feedback fb_max/fb_mean. If labels[j] is set, sibling j's stand-in
    is the label one-hot (P row of the labeled answer) instead of its top-3."""
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


def build_features(struct_df, freeze_t, queries, bpr_dirs, line_dir, need_srcs):
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
    rps = popwin(struct_df["dst"].values, struct_df["time"].values, freeze_t, tmin, 0.05)
    rpl = popwin(struct_df["dst"].values, struct_df["time"].values, freeze_t, tmin, 0.40)
    bprs = [np.load(d / "bpr_emb.npy") for d in bpr_dirs]
    emb = ep.load_embedding(line_dir, expected_rows=num_entity)
    emb_n = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    tl.build_sim_cache(emb, need_srcs)
    srcs = [s for s, _, _ in queries]; cands = [c for _, _, c in queries]
    collab_rows = ep.grouped_collab(cache_mat, srcs, cands)
    X = []
    for i, (src, t, cc_) in enumerate(queries):
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


# ---- TRAIN: split1 in deduped-train raw order (orig positions tracked) ----
split0 = df_raw[df_raw["time"] <= CUT].reset_index(drop=True)
split1 = df_raw[df_raw["time"] > CUT].reset_index(drop=True)
pools = {}
cand_mat = test_df[tl.c_cols].values.astype(np.int64)
for i, s in enumerate(test_df["src"].values.astype(np.int64)):
    pools.setdefault(int(s), set()).update(cand_mat[i].tolist())
lab = split1[split1["src"].isin(pools.keys())].reset_index()  # 'index' = split1 position
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
qorig_tr = np.array(qorig_tr, np.int64)
log(f"train queries {len(train_q)}")
tdir = tl.PROJECT_ROOT / "outputs"
Xtr = build_features(split0, CUT, train_q,
                     [tdir / ("dataset2-bpr-tmax1.26196e+09" + (f"-s{s}" if s != 42 else "")) for s in SEEDS],
                     tdir / "dataset2-novirt-tmax1.26196e+09",
                     np.unique(np.array([s for s, _, _ in train_q], dtype=np.int64)))
lens = np.array([len(q[2]) for q in train_q])
off = np.concatenate([[0], np.cumsum(lens)])
Xf = np.vstack(Xtr).astype(np.float32); del Xtr
yf = np.zeros(off[-1], np.float32); yf[off[1:] - 1] = 1.0
log(f"train matrix {Xf.shape}")

m1 = lgb.LGBMRanker(**PARAMS); m1.fit(Xf, yf, group=lens.tolist())
log("pass-1 trained")
qsrc_tr = np.array([q[0] for q in train_q], np.int64)
qt_tr = np.array([q[1] for q in train_q])
cands_tr = [q[2] for q in train_q]
csets_tr = [frozenset(int(c) for c in q[2]) for q in train_q]
rng2 = np.random.default_rng(7)
usrc = np.unique(qsrc_tr).copy(); rng2.shuffle(usrc)
half = set(usrc[:len(usrc) // 2].tolist())
fold = np.array([0 if s in half else 1 for s in qsrc_tr])
P_tr = item_profiles(split0)

train_labels, ntr, npa = structural_labels(qsrc_tr, qt_tr, qorig_tr, csets_tr, warm_full)
log(f"train labels: triple {ntr} pair {npa} union {len(train_labels)}")


def crossfit(Z):
    s_out = [None] * len(train_q)
    for f in (0, 1):
        a = np.flatnonzero(fold != f); b = np.flatnonzero(fold == f)
        ra = np.concatenate([np.arange(off[i], off[i + 1]) for i in a])
        rb = np.concatenate([np.arange(off[i], off[i + 1]) for i in b])
        mdl = lgb.LGBMRanker(**PARAMS); mdl.fit(Z[ra], yf[ra], group=lens[a].tolist())
        pb = mdl.predict(Z[rb]); pos = 0
        for i in b:
            s_out[i] = pb[pos:pos + lens[i]]; pos += lens[i]
    return s_out


s1_tr = crossfit(Xf)
log("pass-1 cross-fit done")


def train_variant(tag, labels):
    f1max, f1mean = fb_features(qsrc_tr, qt_tr, cands_tr, s1_tr, P_tr, labels=labels)
    Zf2 = np.hstack([Xf, np.concatenate(f1max)[:, None], np.concatenate(f1mean)[:, None]])
    m2 = lgb.LGBMRanker(**PARAMS); m2.fit(Zf2, yf, group=lens.tolist())
    s2 = crossfit(Zf2); del Zf2
    f2max, f2mean = fb_features(qsrc_tr, qt_tr, cands_tr, s2, P_tr, labels=labels)
    Zf3 = np.hstack([Xf, np.concatenate(f2max)[:, None], np.concatenate(f2mean)[:, None]])
    m3 = lgb.LGBMRanker(**PARAMS); m3.fit(Zf3, yf, group=lens.tolist()); del Zf3
    log(f"[{tag}] pass-2/3 trained")
    return m2, m3


m2_t3, m3_t3 = train_variant("top3", None)
m2_lb, m3_lb = train_variant("label", train_labels)

# ---- SERVE: full-train features (shared), then both variants' 3-pass chain ----
test_srcs = test_df["src"].values.astype(np.int64)
test_times = test_df["time"].values.astype(float)
CUTp = float(df_raw["time"].max())
prod_q = [(int(test_srcs[i]), float(test_times[i]), c) for i, c in enumerate(cand_mat)]
N = len(prod_q); CHUNK = 20000
Xall = np.zeros((N, 100, 18), np.float32); hm_all = np.zeros((N, 100), bool)
log("serve: featurize...")
tl.build_history_index(df_raw); tl.build_cooc(df_raw, num_entity)
base_cache = ep.build_base_cache(df_raw); cache_mat = tl.cache_dict_to_matrix(base_cache, set(), num_entity - 1)
dst_pop_log = np.log1p(tl.dst_pop); rpop = tl.dst_rpop_log.copy(); dst_pop_raw = tl.dst_pop.copy()
tmin = float(df_raw["time"].min())
gr = np.zeros(num_entity)
for d, tv in df_raw.groupby("dst")["time"].max().items():
    gr[int(d)] = (float(tv) - tmin) / (CUTp - tmin)
seen = (np.bincount(df_raw["dst"].values, minlength=num_entity) > 0).astype(np.float64)
rps = popwin(df_raw["dst"].values, df_raw["time"].values, CUTp, tmin, 0.05)
rpl = popwin(df_raw["dst"].values, df_raw["time"].values, CUTp, tmin, 0.40)
bprs = [np.load(tdir / ("dataset2-bpr" + (f"-s{s}" if s != 42 else "")) / "bpr_emb.npy") for s in SEEDS]
emb = ep.load_embedding(tdir / "dataset2", expected_rows=num_entity)
emb_n = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
tl.build_sim_cache(emb, np.unique(test_srcs))
for beg in range(0, N, CHUNK):
    end = min(beg + CHUNK, N)
    collab_rows = ep.grouped_collab(cache_mat, [prod_q[i][0] for i in range(beg, end)],
                                    [prod_q[i][2] for i in range(beg, end)])
    for j, i in enumerate(range(beg, end)):
        src, t, cc_ = prod_q[i]; cc = np.clip(cc_, 0, num_entity - 1); m = len(cc_)
        hist_d, _ = tl.get_hist_before_time(src, CUTp + 1)
        f_collab = tl.rownorm(collab_rows[j].astype(np.float64)); f_rpop = tl.rownorm(rpop[cc])
        f_icfm = np.zeros(m); f_icf3 = np.zeros(m)
        if len(hist_d) > 0:
            hvec = emb_n[np.clip(hist_d, 0, num_entity - 1)]
            sim = np.sort(emb_n[cc] @ hvec.T, axis=1)
            f_icfm = tl.rownorm(np.maximum(sim.mean(axis=1), 0.0))
            kk = min(3, sim.shape[1]); f_icf3 = tl.rownorm(np.maximum(sim[:, -kk:].mean(axis=1), 0.0))
        f_bpr = np.zeros(m)
        for be in bprs:
            f_bpr += tl.rownorm(np.maximum(be[cc] @ be[min(src, be.shape[0] - 1)], 0.0))
        f_bpr /= len(bprs)
        blend = (W["collab"] * f_collab + W["rpop"] * f_rpop + W["icfm"] * f_icfm
                 + W["icf3"] * f_icf3 + W["bpr"] * f_bpr)
        hm = np.isin(cc_, hist_d); rs = tl.rownorm(rps[cc]); rl = tl.rownorm(rpl[cc])
        f_cfreq = tl.rownorm(cfreq_log[cc]); f_popratio = tl.rownorm(dst_pop_raw[cc] / (freq_cand[cc] + 1.0))
        Xall[i] = np.column_stack([f_collab, f_rpop, f_icfm, f_icf3, f_bpr, blend, gr[cc], dst_pop_log[cc],
                                   np.full(m, len(hist_d)), hm.astype(float), seen[cc],
                                   np.full(m, (t - CUTp) / YEAR), np.full(m, m), rs, rl, rs - rl,
                                   f_cfreq, f_popratio])
        hm_all[i] = hm
    log(f"  featurize {end}/{N}")

s1_all = m1.booster_.predict(Xall.reshape(-1, 18)).reshape(N, 100)
P_sv = item_profiles(df_raw)
test_csets = [frozenset(int(c) for c in row) for row in cand_mat]
serve_labels, sntr, snpa = structural_labels(test_srcs, test_times, np.arange(N), test_csets, warm_full)
log(f"serve labels: triple {sntr} pair {snpa} union {len(serve_labels)}")


def serve_variant(tag, m2, m3, labels, out_path):
    f1max, f1mean = fb_features(test_srcs, test_times, list(cand_mat), list(s1_all), P_sv, labels=labels)
    Z2 = np.concatenate([Xall.reshape(-1, 18),
                         np.concatenate(f1max)[:, None], np.concatenate(f1mean)[:, None]], axis=1)
    s2_all = m2.booster_.predict(Z2).reshape(N, 100); del Z2
    f2max, f2mean = fb_features(test_srcs, test_times, list(cand_mat), list(s2_all), P_sv, labels=labels)
    Z3 = np.concatenate([Xall.reshape(-1, 18),
                         np.concatenate(f2max)[:, None], np.concatenate(f2mean)[:, None]], axis=1)
    s3 = m3.booster_.predict(Z3).reshape(N, 100); del Z3
    out = np.zeros((N, 100), np.float64)
    for i in range(N):
        s = s3[i].astype(np.float64)
        if hm_all[i].any():
            s = np.where(hm_all[i], s.min() - 1.0, s)
        lo, hi = s.min(), s.max()
        out[i] = (s - lo) / (hi - lo) if hi > lo else np.full(100, 0.5)
    pd.DataFrame(out).to_csv(out_path, index=False, header=False, float_format="%.6f")
    log(f"[{tag}] wrote {out_path}")


serve_variant("top3", m2_t3, m3_t3, None, OUT_T3)
serve_variant("label", m2_lb, m3_lb, serve_labels, OUT_LB)
log("done -- A/B bases ready (top3 vs label), diff = sibling stand-in only")
