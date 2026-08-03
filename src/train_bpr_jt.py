# -*- coding: utf-8 -*-
"""BPR-MF embedding trainer (Jittor). The second of the two neural stages.

A single node-embedding table -- source and destination share it -- trained with
the Bayesian personalised-ranking objective
-log sigmoid(e_src.e_dst - e_src.e_neg), with negatives drawn from the
destination-degree^0.75 distribution.

Four switches distinguish the production runs, all applied in numpy before
training: SEED (the five production seeds), BPR_TIME_MAX (hard time cutoff),
BPR_TAU_FRAC (recency-weighted positive sampling, dataset1 only) and BPR_INNOV
(innovation-only filtering, keeping the first occurrence of each pair, dataset1
only). EVAL_HOLDOUT provides a leak-free holdout mode for evaluation.

Division of labour: sampling and shuffling run in seeded numpy; Jittor owns only
the embedding table, the loss and Adam, so the framework-dependent surface is
minimal and version-robust.

Outputs bpr_emb.npy into the run directory derived from those parameters.
JT_OUT_SUFFIX (empty in the production release) can suffix the output directory
so a verification run cannot clobber a production artifact.

Usage: DATASET=dataset1 BPR_TAU_FRAC=0.25 python src/train_bpr_jt.py
"""

import os
import time
from pathlib import Path

import jittor as jt
import numpy as np
import pandas as pd
from jittor import nn

import pipeline_common as pc

SEED = int(os.environ.get("SEED", "42"))
DIM = int(os.environ.get("BPR_DIM", "256"))
EPOCHS = int(os.environ.get("BPR_EPOCHS", "120"))
TAU_FRAC = float(os.environ.get("BPR_TAU_FRAC", "0"))
TIME_MAX = float(os.environ.get("BPR_TIME_MAX", "0"))
INNOV = os.environ.get("BPR_INNOV", "0") == "1"
EVAL_HOLDOUT = os.environ.get("EVAL_HOLDOUT", "0") == "1"
VAL_PER_SRC_TAIL = 1
LR = 3e-3
L2 = 1e-6
BATCH = 8192

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = os.environ.get("DATASET", "dataset2")
assert DATASET in ("dataset1", "dataset2"), f"unknown dataset: {DATASET}"
DATA_PACK = os.environ.get("DATA_PACK", "data_A")
DATA_DIR = pc.data_dir(DATASET, DATA_PACK)  # one definition, shared with the consumers
# Canonical runs write the contract name; see train_line_jt.py.
JT_OUT_SUFFIX = os.environ.get("JT_OUT_SUFFIX", "")

# Single source of truth, shared with the rankers that consume this output.
_BASE_DIR = pc.bpr_run_dir(
    DATASET,
    seed=SEED,
    tau_frac=TAU_FRAC,
    dim=DIM,
    time_max=TIME_MAX,
    innov=INNOV,
    holdout=EVAL_HOLDOUT,
    root=PROJECT_ROOT,
)
OUT_DIR = _BASE_DIR.parent / (_BASE_DIR.name + JT_OUT_SUFFIX)
os.makedirs(OUT_DIR, exist_ok=True)

jt.flags.use_cuda = 1 if jt.has_cuda else 0


def split_train_val_by_tail(df):
    df = df.sort_values(["src", "time"]).reset_index(drop=True)
    val_df = df.groupby("src", sort=False).tail(VAL_PER_SRC_TAIL)
    train_df = df.drop(val_df.index).reset_index(drop=True)
    return train_df, val_df.reset_index(drop=True)


def main():
    print(
        f"Dataset: {DATASET} ({DATA_PACK}) | jittor cuda={jt.flags.use_cuda} | "
        f"dim {DIM} | epochs {EPOCHS} | seed {SEED} | tau_frac {TAU_FRAC:g}"
    )
    df = pd.read_csv(DATA_DIR / "train.csv")
    df = df.drop_duplicates(subset=["src", "dst", "time"]).reset_index(drop=True)
    df["src"] = df["src"].astype(np.int64)
    df["dst"] = df["dst"].astype(np.int64)
    df["time"] = df["time"].astype(float)
    if EVAL_HOLDOUT:
        n_before = len(df)
        df, _ = split_train_val_by_tail(df)
        print(f"[EVAL_HOLDOUT] Dropped {n_before - len(df)} per-src tail rows from training data")
    num_entity = int(max(df.src.max(), df.dst.max())) + 1
    if TIME_MAX > 0:
        n_before = len(df)
        df = df[df["time"] <= TIME_MAX].reset_index(drop=True)
        print(f"[BPR_TIME_MAX] Kept {len(df)}/{n_before} edges with time <= {TIME_MAX:g}")
    if INNOV:
        n_before = len(df)
        df = (
            df.sort_values("time")
            .drop_duplicates(["src", "dst"], keep="first")
            .reset_index(drop=True)
        )
        print(f"[BPR_INNOV] Kept {len(df)}/{n_before} first-time (src, dst) links")

    rng = np.random.default_rng(SEED)
    jt.set_global_seed(SEED)
    emb = nn.Embedding(num_entity, DIM)
    emb.weight.assign(jt.array(rng.normal(0.0, 0.1, (num_entity, DIM)).astype(np.float32)))
    opt = jt.optim.Adam(emb.parameters(), lr=LR, weight_decay=L2)

    pw = np.bincount(df["dst"].values, minlength=num_entity).astype(np.float64) + 1.0
    cdf = np.cumsum(pw**0.75 / (pw**0.75).sum())
    s_all = df["src"].values.astype(np.int32)
    d_all = df["dst"].values.astype(np.int32)
    n_pos = len(s_all)
    print(f"Training pairs: {n_pos}, nodes: {num_entity}")

    pair_cdf = None
    if TAU_FRAC > 0:
        t_arr = df["time"].values.astype(np.float64)
        tau = TAU_FRAC * (t_arr.max() - t_arr.min())
        pair_w = np.exp(-(t_arr.max() - t_arr) / tau)
        pair_cdf = np.cumsum(pair_w / pair_w.sum())

    t0 = time.time()
    for ep_i in range(EPOCHS):
        perm = None if pair_cdf is not None else rng.permutation(n_pos)
        total = 0.0
        batches = 0
        for beg in range(0, n_pos, BATCH):
            m = min(BATCH, n_pos - beg)
            if pair_cdf is not None:
                idx = np.searchsorted(pair_cdf, rng.random(m)).clip(0, n_pos - 1)
            else:
                idx = perm[beg : beg + m]
            vneg = np.searchsorted(cdf, rng.random(m)).clip(0, num_entity - 1).astype(np.int32)
            u = jt.array(s_all[idx])
            v = jt.array(d_all[idx])
            vn = jt.array(vneg)
            eu = emb(u)
            x = (eu * emb(v)).sum(1) - (eu * emb(vn)).sum(1)
            loss = nn.softplus(-x).mean()
            opt.step(loss)
            total += float(loss.item())
            batches += 1
        if (ep_i + 1) % 20 == 0:
            print(
                f"[BPR epoch {ep_i + 1}/{EPOCHS}] avg loss {total / batches:.4f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    out_path = OUT_DIR / "bpr_emb.npy"
    tmp_path = OUT_DIR / "bpr_emb.tmp.npy"
    np.save(str(tmp_path), emb.weight.numpy().astype(np.float32))
    os.replace(tmp_path, out_path)
    print(f"[OK] BPR embedding saved: {out_path}")


if __name__ == "__main__":
    main()
