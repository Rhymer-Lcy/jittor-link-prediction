# -*- coding: utf-8 -*-
"""Predict-only ensemble scoring over multiple LINE embedding runs.

Loads the exported embeddings of one or more training runs (e.g. the mainline
and a seed-123 run), rebuilds the collaborative-scoring caches from the train
data, averages the row-normalized embedding-CF scores across runs, applies the
per-dataset production blend (recent-popularity prior / co-occurrence CF /
history mask or boost), and writes a submission CSV. No training happens here.

Also provides a real-candidate offline evaluation (--eval): for every
leave-one-out train tail whose src appears in the test file, negatives are
drawn from that src's actual test candidate pools. This protocol reproduced
the online ordering far better than random negatives. Scoring caches are built
with the tail rows removed, so the protocol itself is leak-free (the
embeddings were trained on the full train set, which both configs share, so
single-vs-ensemble comparisons remain fair).

CAUTION (learned online, 2026-07-19): this evaluation systematically overrates
recency-flavored features — time-decayed history and the sequential transition
feature both showed large offline gains and lost online. Use it only to
compare non-temporal variants (e.g. single embedding vs. ensemble).

Usage:
  DATASET=dataset2 python src/ensemble_predict.py --eval
  DATASET=dataset2 python src/ensemble_predict.py
  DATASET=dataset2 python src/ensemble_predict.py --runs outputs/dataset2 outputs/dataset2-s123
"""
import argparse
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import train_line as tl

# Item-CF: cosine similarity between a candidate's embedding and the src's
# historical dst embeddings (mean over all, plus mean of the top-3 closest).
# Non-temporal, so the real-candidate eval is trustworthy for it. Tuned on
# dataset2 across 3 negative-sampling seeds: baseline 0.6132 -> 0.6463.
# Item-CF weights are fixed at the level validated online on dataset2
# (submission 2026-07-19: 0.5441 -> 0.5587). The real-candidate offline eval
# inflates the item-CF gain ~2.4x (it reads embedding geometry that memorized
# the held-out tail edge), so the offline-monotone climb to higher weights is
# not trusted — only this online-backed weight level is used.
if tl.DATASET == "dataset2":
    COLLAB_W = 0.3
    ITEMCF_MEAN_W = 0.7
    ITEMCF_TOP3_W = 0.5
    tl.RPOP_DELTA = 0.45
else:
    # dataset1 keeps full user-CF, history boost and cooc (train_line defaults);
    # item-CF added at the same online-backed weight (offline +0.021, 3 seeds)
    COLLAB_W = 1.0
    ITEMCF_MEAN_W = 0.7
    ITEMCF_TOP3_W = 0.5

NEG_PER_SAMPLE = 99


def load_embedding(run_dir: Path) -> np.ndarray:
    emb_df = pd.read_csv(run_dir / "line_latest_emb.csv")
    return emb_df.drop(columns=["node_id"]).values.astype(np.float32)


def load_virtual_edges(run_dir: Path) -> set:
    path = run_dir / "checkpoints" / "virtual_edges.csv"
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype={"src": int, "dst": int})
    return set(zip(df["src"].astype(int), df["dst"].astype(int)))


def build_base_cache(df: pd.DataFrame) -> dict:
    cache = dict()
    pair_w = df.groupby(["src", "dst"], sort=False).size()
    for (s, d), w in pair_w.items():
        cache.setdefault(int(s), dict())[int(d)] = float(w)
    return cache


def emb_rows_for_queries(emb, needed_srcs, cache_mat, q_srcs, q_times, q_cands):
    """Per-run embedding score: weighted user-CF plus item-CF terms."""
    tl.build_sim_cache(emb, needed_srcs)
    emb_norm = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    n_node = emb.shape[0]
    rows = []
    for src, t, cands in tqdm(
        list(zip(q_srcs, q_times, q_cands)), desc="Embedding scoring (user-CF + item-CF)"
    ):
        collab = tl.batch_sim_score(int(src), cands, cache_mat)
        score = COLLAB_W * tl.rownorm(collab.astype(np.float64))
        if ITEMCF_MEAN_W > 0 or ITEMCF_TOP3_W > 0:
            hist_d, _ = tl.get_hist_before_time(int(src), float(t))
            if len(hist_d) > 0:
                cc = np.clip(cands, 0, n_node - 1)
                hvec = emb_norm[np.clip(hist_d, 0, n_node - 1)]
                sim_all = emb_norm[cc] @ hvec.T
                score = score + ITEMCF_MEAN_W * tl.rownorm(
                    np.maximum(sim_all.mean(axis=1), 0.0))
                k = min(3, hvec.shape[0])
                top3 = np.sort(sim_all, axis=1)[:, -k:].mean(axis=1)
                score = score + ITEMCF_TOP3_W * tl.rownorm(np.maximum(top3, 0.0))
        rows.append(score)
    return rows


def extra_scores(src: int, cands: np.ndarray) -> np.ndarray:
    """Non-embedding blend terms shared by all runs (rpop prior, cooc CF)."""
    extra = np.zeros(len(cands), dtype=np.float64)
    if tl.COOC_GAMMA > 0:
        extra += tl.COOC_GAMMA * tl.rownorm(tl.cooc_scores(src, cands))
    if tl.RPOP_DELTA > 0:
        cc = np.clip(cands, 0, len(tl.dst_rpop_log) - 1)
        extra += tl.RPOP_DELTA * tl.rownorm(tl.dst_rpop_log[cc])
    return extra


def blend_scores(src, curr_time, cands, emb_score):
    """Production ranking blend on top of an (already averaged) embedding score."""
    hist_d, _ = tl.get_hist_before_time(int(src), float(curr_time))
    blend = emb_score + extra_scores(int(src), cands)
    if tl.MASK_HISTORY:
        blend[np.isin(cands, hist_d)] = 0.0
    else:
        cnt = tl.count_in_history(cands, hist_d).astype(np.float64)
        blend = tl.HIST_BOOST * cnt + blend
    return blend


def run_eval(df_raw, test_df, run_dirs, args):
    train_split, val_split = tl.split_train_val_by_tail(df_raw)
    print(f"Eval protocol: {len(train_split)} train rows (tails removed), {len(val_split)} tail rows")
    tl.build_history_index(train_split)
    tl.build_cooc(train_split, args.num_entity)
    base_cache = build_base_cache(train_split)
    cache_mat = tl.cache_dict_to_matrix(base_cache, set(), args.num_entity - 1)

    # Candidate pools: every candidate the src actually gets in the test file
    pools = {}
    cand_mat = test_df[tl.c_cols].values.astype(np.int64)
    for i, s in enumerate(test_df["src"].values.astype(np.int64)):
        pools.setdefault(int(s), set()).update(cand_mat[i].tolist())

    rng = np.random.default_rng(args.seed)
    samples = []
    for row in val_split.itertuples(index=False):
        src, true_dst, t = int(row.src), int(row.dst), float(row.time)
        pool = pools.get(src)
        if not pool:
            continue
        negs = sorted(pool - {true_dst, src})
        if not negs:
            continue
        if len(negs) > NEG_PER_SAMPLE:
            negs = rng.choice(negs, NEG_PER_SAMPLE, replace=False).tolist()
        samples.append((src, t, np.array(negs + [true_dst], dtype=np.int64)))
    pool_sizes = [len(c) - 1 for _, _, c in samples]
    print(f"Eval samples: {len(samples)} (src in test), negatives per sample: "
          f"min {min(pool_sizes)} / median {int(np.median(pool_sizes))} / max {max(pool_sizes)}")

    q_srcs = [s for s, _, _ in samples]
    q_times = [t for _, t, _ in samples]
    q_cands = [c for _, _, c in samples]
    needed_srcs = np.unique(np.array(q_srcs, dtype=np.int64))

    per_run_collab = []
    for run_dir in run_dirs:
        print(f"\n--- Scoring with {run_dir.name} ---")
        emb = load_embedding(run_dir)
        per_run_collab.append(
            emb_rows_for_queries(emb, needed_srcs, cache_mat, q_srcs, q_times, q_cands)
        )
        del emb
        gc.collect()

    def mrr_of(collab_rows):
        total = 0.0
        for (src, t, cands), collab in zip(samples, collab_rows):
            blend = blend_scores(src, t, cands, collab)
            rank = 1 + int((blend > blend[-1]).sum()) + int((blend[:-1] == blend[-1]).sum())
            total += 1.0 / rank
        return total / len(samples)

    print("\n===== Real-candidate MRR =====")
    for run_dir, collab_rows in zip(run_dirs, per_run_collab):
        print(f"  {run_dir.name:<24s} {mrr_of(collab_rows):.4f}")
    if len(run_dirs) > 1:
        mean_rows = [
            np.mean([per_run_collab[r][i] for r in range(len(run_dirs))], axis=0)
            for i in range(len(samples))
        ]
        print(f"  {'ensemble (mean)':<24s} {mrr_of(mean_rows):.4f}")


def run_predict(df_raw, test_df, run_dirs, args):
    tl.build_history_index(df_raw)
    tl.build_cooc(df_raw, args.num_entity)
    base_cache = build_base_cache(df_raw)

    q_srcs = test_df["src"].values.astype(np.int64)
    q_times = test_df["time"].values.astype(float)
    cand_mat = test_df[tl.c_cols].values.astype(np.int64)
    q_cands = list(cand_mat)
    needed_srcs = np.unique(q_srcs)
    print(f"Test rows: {len(test_df)}, distinct srcs: {len(needed_srcs)}")

    sum_collab = None
    for run_dir in run_dirs:
        print(f"\n--- Scoring with {run_dir.name} ---")
        # Each run's cache includes its own virtual edges, matching how that
        # run's production submission files were generated
        virt = load_virtual_edges(run_dir)
        print(f"Virtual edges merged: {len(virt)}")
        cache_mat = tl.cache_dict_to_matrix(base_cache, virt, args.num_entity - 1)
        emb = load_embedding(run_dir)
        rows = emb_rows_for_queries(emb, needed_srcs, cache_mat, q_srcs, q_times, q_cands)
        if sum_collab is None:
            sum_collab = rows
        else:
            for i in range(len(rows)):
                sum_collab[i] = sum_collab[i] + rows[i]
        del emb, cache_mat, rows
        gc.collect()

    out_rows = []
    for i in tqdm(range(len(q_srcs)), desc="Blending"):
        collab_mean = sum_collab[i] / len(run_dirs)
        blend = blend_scores(q_srcs[i], q_times[i], cand_mat[i], collab_mean)
        out_rows.append(tl.rownorm(blend).tolist())

    out_dir = tl.PROJECT_ROOT / "outputs" / f"{tl.DATASET}-ensemble"
    os.makedirs(out_dir, exist_ok=True)
    save_path = out_dir / "result_ensemble.csv"
    # 6 decimals: MRR is rank-based, so this preserves ordering exactly while
    # keeping the file small. Full float64 repr bloated item-CF outputs (every
    # candidate nonzero) to ~2.5x, pushing the packaged zip past the ~100 MB
    # submission upload limit.
    pd.DataFrame(out_rows).to_csv(save_path, index=False, header=False, float_format="%.6f")
    print(f"\n[OK] Ensemble submission saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--runs", nargs="+",
        default=[f"outputs/{tl.DATASET}", f"outputs/{tl.DATASET}-s123"],
        help="run directories containing line_latest_emb.csv",
    )
    parser.add_argument("--eval", action="store_true",
                        help="real-candidate offline evaluation instead of writing a submission")
    parser.add_argument("--seed", type=int, default=42,
                        help="rng seed for negative sampling in --eval")
    args = parser.parse_args()

    run_dirs = [tl.PROJECT_ROOT / r for r in args.runs]
    for d in run_dirs:
        assert (d / "line_latest_emb.csv").exists(), f"missing embedding: {d}"

    print(f"Dataset: {tl.DATASET} | runs: {[d.name for d in run_dirs]} | "
          f"blend: collab={COLLAB_W} itemcf_mean={ITEMCF_MEAN_W} "
          f"itemcf_top3={ITEMCF_TOP3_W} cooc={tl.COOC_GAMMA} rpop={tl.RPOP_DELTA} "
          f"mask_history={tl.MASK_HISTORY} hist_boost={tl.HIST_BOOST}")

    df_raw = pd.read_csv(tl.train_csv)
    df_raw = df_raw.drop_duplicates(subset=["src", "dst", "time"]).reset_index(drop=True)
    df_raw["src"] = df_raw["src"].astype(np.int64)
    df_raw["dst"] = df_raw["dst"].astype(np.int64)
    df_raw["time"] = df_raw["time"].astype(float)

    test_df = pd.read_csv(tl.test_csv)
    test_df = test_df[["src", "time"] + tl.c_cols].copy()
    test_df["src"] = test_df["src"].astype(np.int64)
    test_df["time"] = test_df["time"].astype(float)

    args.num_entity = int(max(df_raw.src.max(), df_raw.dst.max())) + 1

    if args.eval:
        run_eval(df_raw, test_df, run_dirs, args)
    else:
        run_predict(df_raw, test_df, run_dirs, args)


if __name__ == "__main__":
    main()
