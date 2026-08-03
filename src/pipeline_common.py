# -*- coding: utf-8 -*-
"""Framework-neutral shared components of the canonical pipeline.

This module is the SINGLE SOURCE OF TRUTH for two things the canonical stages
need:

1. run-directory naming, so a producer and its consumer can never disagree;
2. the framework-neutral data utilities (history indexing, co-occurrence,
   similarity caching, row normalisation, paths and constants).

It imports no deep-learning framework at all. That is the point: the canonical
ranking stages import this module, so nothing on the canonical path pulls in a
training framework merely to obtain a helper. Neural training lives in the
explicit Jittor entry points src/train_line_jt.py and src/train_bpr_jt.py.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET = os.environ.get("DATASET", "dataset2")

assert DATASET in ("dataset1", "dataset2"), f"unknown dataset: {DATASET}"

DATA_PACK = os.environ.get("DATA_PACK", "data_A")


def data_root() -> Path:
    """Root holding the official data packs.

    Defaults to ``<repo>/data``. ``DATA_ROOT`` overrides it, so the canonical
    entrypoint can point a run at competition data that lives outside the
    checkout without any module hard-coding a host path.
    """
    override = os.environ.get("DATA_ROOT")
    return Path(override) if override else PROJECT_ROOT / "data"


def data_dir(dataset: str, pack: str | None = None) -> Path:
    """Directory holding one dataset's ``train.csv`` and ``test.csv``.

    The single definition. The same expression was previously repeated in this
    module, ``train_line_jt`` and ``train_bpr_jt``, so a producer and a consumer
    could in principle be pointed at different data roots.
    """
    return data_root() / (pack or DATA_PACK) / dataset


DATA_DIR = data_dir(DATASET)

train_csv = DATA_DIR / "train.csv"

test_csv = DATA_DIR / "test.csv"

TOP_N_FIRST = 200

TOP_N_SECOND = 2000

DECAY_W1 = 1.0

DECAY_W2 = 0.2

VAL_PER_SRC_TAIL = 1

MASK_HISTORY = DATASET == "dataset2"

HIST_BOOST = float(os.environ.get("HIST_BOOST", "20.0" if DATASET == "dataset1" else "0.0"))

COOC_GAMMA = 1.0 if DATASET == "dataset1" else 0.0

RPOP_DELTA = 0.3 if DATASET == "dataset2" else 0.0

RPOP_TIME_QUANTILE = 0.8

src2row = dict()

sim_neigh_arr = None

sim_weight_arr = None

ui_mat = None

ui_mat_csc = None

dst_pop = None

dst_rpop_log = None

src_hist_times = dict()

src_hist_dsts = dict()

c_cols = [f"c{i}" for i in range(1, 101)]


def split_train_val_by_tail(df):
    # Last VAL_PER_SRC_TAIL rows (by time) of each src go to val; srcs with too
    # few rows put the whole group in val and contribute nothing to train.
    df = df.sort_values(["src", "time"]).reset_index(drop=True)
    val_df = df.groupby("src", sort=False).tail(VAL_PER_SRC_TAIL)
    train_df = df.drop(val_df.index).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    return train_df, val_df


def build_history_index(full_df):
    # One-off per-src index sorted by time, replacing full-table scans
    src_hist_times.clear()
    src_hist_dsts.clear()
    df = full_df.sort_values("time", kind="mergesort")
    for src, g in df.groupby("src", sort=False):
        src_hist_times[int(src)] = g["time"].values.astype(float)
        src_hist_dsts[int(src)] = g["dst"].values.astype(np.int64)


def get_hist_before_time(src_id: int, cutoff_time: float):
    # Like get_dst_before_time but also returns the matching timestamps
    times = src_hist_times.get(src_id)
    if times is None:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    k = np.searchsorted(times, cutoff_time, side="left")
    return src_hist_dsts[src_id][:k], times[:k]


def count_in_history(candidates: np.ndarray, hist: np.ndarray) -> np.ndarray:
    # Occurrence count of each candidate in the (possibly repeating) history
    if hist.size == 0:
        return np.zeros(candidates.shape[0], dtype=np.float32)
    values, counts = np.unique(hist, return_counts=True)
    idx = np.searchsorted(values, candidates)
    idx[idx >= len(values)] = 0
    hit = values[idx] == candidates
    return np.where(hit, counts[idx], 0).astype(np.float32)


def rownorm(v: np.ndarray) -> np.ndarray:
    mx = v.max()
    return v / mx if mx > 1e-12 else v


def build_cooc(df, n_node: int):
    global ui_mat, ui_mat_csc, dst_pop, dst_rpop_log
    ui_mat = sp.coo_matrix(
        (np.ones(len(df), dtype=np.float32), (df["src"].values, df["dst"].values)),
        shape=(n_node, n_node),
    ).tocsr()
    ui_mat_csc = ui_mat.tocsc()
    dst_pop = np.asarray(ui_mat.sum(axis=0)).ravel() + 1.0
    # Recent popularity: dst interaction counts in the last time window
    tcut = df["time"].quantile(RPOP_TIME_QUANTILE)
    rpop = np.zeros(n_node, dtype=np.float64)
    for d, c in df[df["time"] >= tcut].groupby("dst").size().items():
        rpop[int(d)] = float(c)
    dst_rpop_log = np.log1p(rpop)


def cooc_scores(src: int, cands: np.ndarray) -> np.ndarray:
    # Popularity-normalized co-occurrence CF: users overlapping src's history,
    # weighted by overlap size, aggregated over their interactions with cands
    if ui_mat is None or src >= ui_mat.shape[0]:
        return np.zeros(len(cands), dtype=np.float64)
    x = np.asarray((ui_mat @ ui_mat[src].T).todense()).ravel()
    x[src] = 0.0
    sc = np.asarray((sp.csr_matrix(x) @ ui_mat_csc[:, cands]).todense()).ravel()
    return sc.astype(np.float64) / np.sqrt(dst_pop[cands])


def cache_dict_to_matrix(base_cache: dict, add_pair_set: set, max_node: int):
    # Sparse CSR: dataset2 has 139k+ node ids, a dense (N+1)^2 float32 matrix
    # would need ~78 GB
    rows, cols, vals = [], [], []
    for src, ddict in base_cache.items():
        for d, w in ddict.items():
            rows.append(src)
            cols.append(d)
            vals.append(w)
    for u, d in add_pair_set:
        rows.append(u)
        cols.append(d)
        vals.append(1.0)
    n = max_node + 1
    mat = sp.coo_matrix(
        (np.array(vals, dtype=np.float32), (np.array(rows), np.array(cols))),
        shape=(n, n),
    ).tocsr()
    return mat


def batch_sim_score(target_src: int, dst_batch: np.ndarray, cache_mat):
    row_idx = src2row.get(target_src, -1)
    if row_idx == -1:
        return np.zeros_like(dst_batch, dtype=np.float32)
    neigh_ids = sim_neigh_arr[row_idx]
    neigh_w = sim_weight_arr[row_idx]
    weight_slice = cache_mat[neigh_ids][:, dst_batch].toarray()
    scores = neigh_w @ weight_slice
    return scores.astype(np.float32)


def build_sim_cache(emb_matrix, real_src_np):
    # Two-band decayed similar-user cache: chunked cosine top-k.
    #
    # NumPy implementation. This function is on the CANONICAL ranking path
    # (both rankers call it), so it deliberately depends on no deep-learning
    # framework. The steps are: L2-normalise, chunked matmul, zero the
    # self-similarity, take the k largest per row in descending order, pad,
    # then apply the two-band decay.
    global sim_neigh_arr, sim_weight_arr  # src2row is only mutated, not rebound
    src2row.clear()
    max_emb_id = emb_matrix.shape[0] - 1

    valid_targets = [int(s) for s in real_src_np if 0 <= s <= max_emb_id]

    if len(valid_targets) == 0:
        sim_neigh_arr = np.zeros((0, TOP_N_SECOND), dtype=np.int64)
        sim_weight_arr = np.zeros((0, TOP_N_SECOND), dtype=np.float32)
        return

    emb = np.ascontiguousarray(emb_matrix, dtype=np.float32)
    norm = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
    emb_norm = emb / norm
    targets = np.asarray(valid_targets, dtype=np.int64)

    k = min(TOP_N_SECOND, emb_norm.shape[0])
    chunk = 1024
    neigh_chunks = []
    weight_chunks = []
    for beg in tqdm(
        range(0, len(valid_targets), chunk), desc="Building two-band decayed similar-user cache"
    ):
        ids = targets[beg : beg + chunk]
        sim = emb_norm[ids] @ emb_norm.T
        # Exclude self-similarity, matching the original per-row zeroing
        sim[np.arange(len(ids)), ids] = 0.0
        # k largest per row, then order them descending. argpartition selects
        # the top block, the subsequent argsort orders it; ties resolve by
        # ascending column index, which is deterministic.
        part = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        part_vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_vals, axis=1, kind="stable")
        idx = np.take_along_axis(part, order, axis=1)
        vals = np.take_along_axis(part_vals, order, axis=1)
        neigh_chunks.append(idx.astype(np.int64))
        weight_chunks.append(vals)

    sim_neigh_arr = np.concatenate(neigh_chunks, axis=0)
    weights = np.concatenate(weight_chunks, axis=0)
    if k < TOP_N_SECOND:
        pad = TOP_N_SECOND - k
        sim_neigh_arr = np.pad(sim_neigh_arr, ((0, 0), (0, pad)), constant_values=0)
        weights = np.pad(weights, ((0, 0), (0, pad)), constant_values=0.0)

    decay_mask = np.full(TOP_N_SECOND, DECAY_W2, dtype=np.float32)
    decay_mask[:TOP_N_FIRST] = DECAY_W1
    weights = weights * decay_mask
    weights[weights < 1e-8] = 0.0
    sim_weight_arr = weights.astype(np.float32)

    for row_idx, src_id in enumerate(valid_targets):
        src2row[src_id] = row_idx


# ===================== Run-directory contract =====================
# One authoritative definition used by BOTH the producer (the Jittor trainers)
# and the consumers (the rankers). Before this existed they disagreed for the
# full-train LINE case -- the trainer wrote "<dataset>-novirt" while the ranker
# read "<dataset>" -- and the gap had to be bridged by a hand-made symlink.


def outputs_root(root: Path | None = None) -> Path:
    """Directory holding every run artifact.

    Defaults to ``<repo>/outputs``. ``OUTPUTS_ROOT`` overrides it and takes
    precedence over an explicit ``root`` argument, because the trainers pass
    their own ``PROJECT_ROOT`` and would otherwise ignore the entrypoint's
    ``--output-root``. Producer and consumer read the same variable, so they
    cannot be separated by it.
    """
    override = os.environ.get("OUTPUTS_ROOT")
    if override:
        return Path(override)
    return (Path(root) if root is not None else PROJECT_ROOT) / "outputs"


def line_run_dir(
    dataset: str,
    *,
    seed: int = 42,
    neg_dist: str = "uniform",
    emb_dim: int = 400,
    time_max: float = 0.0,
    holdout: bool = False,
    virtual_edges: bool = False,
    root: Path | None = None,
) -> Path:
    """Directory holding a LINE run's line_latest_emb.csv.

    ``virtual_edges`` defaults to False because virtual-edge self-training is a
    measured no-op in the production configuration and the Jittor trainer omits
    it, which is what the "-novirt" marker records.
    """
    suffix = "" if seed == 42 else f"-s{seed}"
    suffix += "-negpop" if neg_dist == "pop075" else ""
    suffix += f"-d{emb_dim}" if emb_dim != 400 else ""
    suffix += "" if virtual_edges else "-novirt"
    suffix += f"-tmax{time_max:g}" if time_max and time_max > 0 else ""
    suffix += "-holdout" if holdout else ""
    return outputs_root(root) / (dataset + suffix)


def bpr_run_dir(
    dataset: str,
    *,
    seed: int = 42,
    tau_frac: float = 0.0,
    dim: int = 256,
    time_max: float = 0.0,
    innov: bool = False,
    holdout: bool = False,
    root: Path | None = None,
) -> Path:
    """Directory holding a BPR run's bpr_emb.npy."""
    name = dataset + "-bpr"
    name += "-innov" if innov else ""
    name += f"-t{tau_frac:g}" if tau_frac and tau_frac > 0 else ""
    name += f"-d{dim}" if dim != 256 else ""
    name += f"-tmax{time_max:g}" if time_max and time_max > 0 else ""
    name += f"-s{seed}" if seed != 42 else ""
    name += "-holdout" if holdout else ""
    return outputs_root(root) / name


def ranker_dir(dataset: str, root: Path | None = None) -> Path:
    """Directory holding a dataset's ranker output."""
    suffix = "-ensemble" if dataset == "dataset1" else "-ranker"
    return outputs_root(root) / (dataset + suffix)
