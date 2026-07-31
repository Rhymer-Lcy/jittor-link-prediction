# -*- coding: utf-8 -*-
"""dataset2 production ranker: 18-feature LightGBM LambdaRank + 3-pass basket feedback.

This is the productionized form of the pipeline behind the 1.49254 basket3
base (online 2026-07-23); combined with the row-order postprocessor
(`crf_promote.py`) it produced the 1.50890 best. Chain:

  pass 1: 18-feature LambdaRank over each test row's 100-candidate slate.
      Features (order is load-bearing, the boosters index by position):
      collab, rpop, icf_mean, icf_top3, bpr_ens5, blend, graph_recency,
      dst_pop_log, hist_len, in_hist, seen, dt_years, slate_size,
      rpop_short(5%), rpop_long(40%), rs-rl, cfreq_log, popratio.
      cfreq/popratio are candidate-generator footprint features computed from
      the test file's candidate columns (identical for train and serve since
      training negatives are drawn from the same test pools).
  pass 2/3: basket feedback -- 75.8% of test rows share a (src, time) event
      with sibling rows; each row gets fb_max/fb_mean cosine features between
      its candidates and the softmax-top3 stand-in answers of its siblings
      (dst -> src interaction profiles, L2-normalized). Pass N feedback is
      rebuilt from pass N-1 scores; train-side stand-ins come from a by-src
      2-fold cross-fit so no model ever scores a src it trained on.

Train side: features frozen at CUT (split0 structures + split0-frozen
embeddings), labels = ALL of split1 (full horizon; the early 2/3 cut was an
experiment artifact worth -0.0017). Serve side: features frozen at full-train
max, same 100-candidate slates as the submission rows, history candidates
demoted to row minimum, per-row [0, 1] normalization, %.6f headerless CSV.

Prerequisites (embedding dirs under outputs/):
  train-side (split0-frozen, CUT = 1261958400):
    dataset2-novirt-tmax1.26196e+09   VIRT_MODE=off LINE_TIME_MAX=1261958400
                                      DATASET=dataset2 python src/train_line.py
    dataset2-bpr-tmax1.26196e+09[-s{123,777,2024,31337}]
                                      BPR_TIME_MAX=1261958400 [SEED=...]
                                      DATASET=dataset2 python src/train_bpr.py
  serve-side (full train):
    dataset2                          the production LINE run
    dataset2-bpr[-s{123,777,2024,31337}]  DATASET=dataset2 [SEED=...]
                                      python src/train_bpr.py

Run: DATASET=dataset2 python src/ranker_basket_ds2.py
Outputs (outputs/dataset2-ranker/): ranker_pass{1,2,3}.txt boosters and
ranker_basket3_dataset2.csv (the ds2 base for crf_promote.py).
Runtime: ~2.5 h (train cross-fits ~1 h, serve featurize ~1 h, feedback ~20 min).
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

import pipeline_common as tl   # framework-neutral shared components
import ensemble_predict as ep

assert tl.DATASET == "dataset2", "this pipeline is dataset2-only"
SEEDS = (42, 123, 777, 2024, 31337)
CUT = 1261958400.0
W = {"collab": 0.25, "rpop": 1.25, "icfm": 0.5, "icf3": 1.0, "bpr": 0.7}
YEAR = 365.0 * 86400.0
RNG = np.random.default_rng(42)
OUT_DIR = tl.PROJECT_ROOT / "outputs" / "dataset2-ranker"
os.makedirs(OUT_DIR, exist_ok=True)
OUT = str(OUT_DIR / "ranker_basket3_dataset2.csv")
PASS1_MODEL = str(OUT_DIR / "ranker_pass1.txt")
PASS2_MODEL = str(OUT_DIR / "ranker_pass2.txt")
PASS3_MODEL = str(OUT_DIR / "ranker_pass3.txt")
PARAMS = dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[10], n_estimators=400,
              learning_rate=0.05, num_leaves=31, min_child_samples=100, random_state=42,
              n_jobs=6, verbosity=-1, label_gain=[0, 1])
# Reproduction status vs the shipped A-board ranker (verified 2026-07-23). This
# unified script trains all three passes with the same features and PARAMS;
# passes 2 and 3 come out byte-identical to the shipped boosters (md5
# 32fbc4f6e7db, 53f538cd61a4). Pass 1 differs (md5 11b6799e29 vs shipped
# eab591d02d13) because the shipped pass-1 booster was produced by a SEPARATE
# full-horizon script (exp_ranker_fullhz_prod_ds2.py) with a different feature
# build -- NOT a thread-count difference: n_jobs 6 and -1 both yield 11b6799e29
# on this script's train matrix (a direct probe confirmed LightGBM histogram
# summation IS thread-sensitive in general, but not on this matrix). The port
# is faithful, not byte-exact: pass-1 scores agree with the shipped file on
# 99.17% of serve top-1 decisions (mean |delta| 3e-4), inside the ranker's own
# run-to-run noise. Byte-exact pass-1 reproduction would require importing the
# full-horizon feature build; not worth restructuring the unified pipeline, and
# irrelevant for data_B (which needs a correct pipeline, not the A-board bytes).
# n_jobs is fixed at 6 (not -1) so the result is portable across core counts.

df_raw = pd.read_csv(tl.train_csv).drop_duplicates(subset=["src", "dst", "time"]).reset_index(drop=True)
df_raw["src"] = df_raw["src"].astype(np.int64); df_raw["dst"] = df_raw["dst"].astype(np.int64)
df_raw["time"] = df_raw["time"].astype(float)
test_df = pd.read_csv(tl.test_csv)[["src", "time"] + tl.c_cols].copy()
test_df["src"] = test_df["src"].astype(np.int64)
num_entity = int(max(df_raw.src.max(), df_raw.dst.max())) + 1

# candidate-generator footprint (shared by train and serve by construction)
_allcand = np.clip(test_df[tl.c_cols].values.astype(np.int64).ravel(), 0, num_entity - 1)
freq_cand = np.bincount(_allcand, minlength=num_entity).astype(np.float64)
cfreq_log = np.log1p(freq_cand)


def popwin(dvals, tvals, hi, lo, frac):
    m = tvals >= (hi - frac * (hi - lo))
    return np.log1p(np.bincount(dvals[m], minlength=num_entity).astype(np.float64))


def item_profiles(edges):
    """L2-normalized dst -> src interaction profile rows (sparse)."""
    P = sp.csr_matrix((np.ones(len(edges), np.float32),
                       (edges.dst.values.astype(np.int64), edges.src.values.astype(np.int64))),
                      shape=(num_entity, num_entity))
    P.data[:] = 1.0
    nrm = np.sqrt(P.multiply(P).sum(1)).A.ravel()
    nrm[nrm == 0] = 1.0
    return (sp.diags(1.0 / nrm) @ P).tocsr()


def fb_features(qsrc, qt, cands, s1, P):
    """Sibling-feedback fb_max/fb_mean per row from pass scores s1."""
    n = len(qsrc)
    ev = defaultdict(list)
    for i in range(n):
        ev[(int(qsrc[i]), float(qt[i]))].append(i)
    fmax = [np.zeros(len(cands[i]), np.float32) for i in range(n)]
    fmean = [np.zeros(len(cands[i]), np.float32) for i in range(n)]
    done = 0; t0 = time.time()
    for idxs in ev.values():
        done += len(idxs)
        if len(idxs) < 2:
            continue
        tvecs = []
        for j in idxs:
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
        if done % 200000 < len(idxs):
            print(f"    feedback {done} rows ({time.time() - t0:.0f}s)", flush=True)
    return fmax, fmean


def build_features(struct_df, freeze_t, queries, bpr_dirs, line_dir, need_srcs):
    """18-column feature blocks for a list of (src, t, cands) queries."""
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


# ---- TRAIN: split0 features, full-split1 labels, 3-pass cross-fit chain ----
split0 = df_raw[df_raw["time"] <= CUT].reset_index(drop=True)
split1 = df_raw[df_raw["time"] > CUT].reset_index(drop=True)
pools = {}
cand_mat = test_df[tl.c_cols].values.astype(np.int64)
for i, s in enumerate(test_df["src"].values.astype(np.int64)):
    pools.setdefault(int(s), set()).update(cand_mat[i].tolist())
lab = split1[split1["src"].isin(pools.keys())].sort_values("time").reset_index(drop=True)
train_q = []
for row in lab.itertuples(index=False):
    src, dst, t = int(row.src), int(row.dst), float(row.time)
    pool = pools.get(src); negs = sorted(pool - {dst, src}) if pool else []
    if not negs:
        continue
    if len(negs) > ep.NEG_PER_SAMPLE:
        negs = RNG.choice(negs, ep.NEG_PER_SAMPLE, replace=False).tolist()
    train_q.append((src, t, np.array(negs + [dst], dtype=np.int64)))
print(f"train queries {len(train_q)}", flush=True)
tdir = tl.PROJECT_ROOT / "outputs"
Xtr = build_features(split0, CUT, train_q,
                     [tl.bpr_run_dir("dataset2", seed=s, time_max=CUT) for s in SEEDS],
                     tl.line_run_dir("dataset2", time_max=CUT),
                     np.unique(np.array([s for s, _, _ in train_q], dtype=np.int64)))
lens = np.array([len(q[2]) for q in train_q])
off = np.concatenate([[0], np.cumsum(lens)])
Xf = np.vstack(Xtr).astype(np.float32); del Xtr
yf = np.zeros(off[-1], np.float32)
yf[off[1:] - 1] = 1.0
print(f"train matrix {Xf.shape}", flush=True)

# pass-1 full-data booster (the serve pass-1 scorer) -- see the reproduction
# note next to PARAMS for why this differs from the shipped pass-1 booster
m1 = lgb.LGBMRanker(**PARAMS)
m1.fit(Xf, yf, group=lens.tolist())
m1.booster_.save_model(PASS1_MODEL)
print("pass-1 ranker trained + saved", flush=True)

qsrc_tr = np.array([q[0] for q in train_q], np.int64)
qt_tr = np.array([q[1] for q in train_q])
cands_tr = [q[2] for q in train_q]
rng2 = np.random.default_rng(7)
usrc = np.unique(qsrc_tr).copy(); rng2.shuffle(usrc)
half = set(usrc[:len(usrc) // 2].tolist())
fold = np.array([0 if s in half else 1 for s in qsrc_tr])
P_tr = item_profiles(split0)


def crossfit(Z, tag):
    """fold-out scores from a by-src two-fold cross-fit on feature matrix Z."""
    s_out = [None] * len(train_q)
    for f in (0, 1):
        a = np.flatnonzero(fold != f); b = np.flatnonzero(fold == f)
        ra = np.concatenate([np.arange(off[i], off[i + 1]) for i in a])
        rb = np.concatenate([np.arange(off[i], off[i + 1]) for i in b])
        mdl = lgb.LGBMRanker(**PARAMS)
        mdl.fit(Z[ra], yf[ra], group=lens[a].tolist())
        pb = mdl.predict(Z[rb])
        pos = 0
        for i in b:
            s_out[i] = pb[pos:pos + lens[i]]; pos += lens[i]
        print(f"{tag} cross-fit fold {f} done", flush=True)
    return s_out


s1_tr = crossfit(Xf, "pass-1")
print("building train fb1 (from pass-1 scores)...", flush=True)
f1max, f1mean = fb_features(qsrc_tr, qt_tr, cands_tr, s1_tr, P_tr)
Zf2 = np.hstack([Xf, np.concatenate(f1max)[:, None], np.concatenate(f1mean)[:, None]])
del f1max, f1mean
m2 = lgb.LGBMRanker(**PARAMS)
m2.fit(Zf2, yf, group=lens.tolist())
m2.booster_.save_model(PASS2_MODEL)
print("pass-2 ranker trained + saved", flush=True)
s2_tr = crossfit(Zf2, "pass-2")
del Zf2
print("building train fb2 (from pass-2 scores)...", flush=True)
f2max, f2mean = fb_features(qsrc_tr, qt_tr, cands_tr, s2_tr, P_tr)
Zf3 = np.hstack([Xf, np.concatenate(f2max)[:, None], np.concatenate(f2mean)[:, None]])
del Xf, f2max, f2mean
m3 = lgb.LGBMRanker(**PARAMS)
m3.fit(Zf3, yf, group=lens.tolist())
m3.booster_.save_model(PASS3_MODEL)
del Zf3
print("pass-3 ranker trained + saved", flush=True)

# ---- SERVE: full-train features, saved boosters, 3-pass chain ----
test_srcs = test_df["src"].values.astype(np.int64)
test_times = test_df["time"].values.astype(float)
CUTp = float(df_raw["time"].max())
prod_q = [(int(test_srcs[i]), float(test_times[i]), c) for i, c in enumerate(cand_mat)]
N = len(prod_q); CHUNK = 20000; t0 = time.time()
Xall = np.zeros((N, 100, 18), np.float32)
hm_all = np.zeros((N, 100), bool)
print("serve: featurize...", flush=True)
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
bprs = [np.load(tl.bpr_run_dir("dataset2", seed=s) / "bpr_emb.npy") for s in SEEDS]
emb = ep.load_embedding(tl.line_run_dir("dataset2"), expected_rows=num_entity)
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
    print(f"  {end}/{N} ({time.time() - t0:.0f}s)", flush=True)

pass1 = lgb.Booster(model_file=PASS1_MODEL)
pass2 = lgb.Booster(model_file=PASS2_MODEL)
pass3 = lgb.Booster(model_file=PASS3_MODEL)
s1_all = pass1.predict(Xall.reshape(-1, 18)).reshape(N, 100)
print("serve pass 1 scored", flush=True)
P_sv = item_profiles(df_raw)
print("serve fb1...", flush=True)
f1max_sv, f1mean_sv = fb_features(test_srcs, test_times, list(cand_mat), list(s1_all), P_sv)
Z2 = np.concatenate([Xall.reshape(-1, 18),
                     np.concatenate(f1max_sv)[:, None], np.concatenate(f1mean_sv)[:, None]], axis=1)
del f1max_sv, f1mean_sv
s2_all = pass2.predict(Z2).reshape(N, 100)
del Z2
print("serve pass 2 scored", flush=True)
print("serve fb2...", flush=True)
f2max_sv, f2mean_sv = fb_features(test_srcs, test_times, list(cand_mat), list(s2_all), P_sv)
Z3 = np.concatenate([Xall.reshape(-1, 18),
                     np.concatenate(f2max_sv)[:, None], np.concatenate(f2mean_sv)[:, None]], axis=1)
del Xall, f2max_sv, f2mean_sv
s3 = pass3.predict(Z3).reshape(N, 100)
del Z3
print("serve pass 3 scored", flush=True)

out = np.zeros((N, 100), np.float64)
for i in range(N):
    s = s3[i].astype(np.float64)
    if hm_all[i].any():
        s = np.where(hm_all[i], s.min() - 1.0, s)
    lo, hi = s.min(), s.max()
    out[i] = (s - lo) / (hi - lo) if hi > lo else np.full(100, 0.5)
print(f"pass3 vs pass2 top-1 disagreement: {(s3.argmax(1) != s2_all.argmax(1)).mean() * 100:.2f}%", flush=True)
pd.DataFrame(out).to_csv(OUT, index=False, header=False, float_format="%.6f")
print(f"[OK] wrote {OUT}", flush=True)
