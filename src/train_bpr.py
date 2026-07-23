# -*- coding: utf-8 -*-
"""BPR-MF embedding trainer (pairwise ranking loss).

Trains a single node-embedding table with the BPR objective
(-log sigmoid(e_src.e_dst - e_src.e_neg)) over the train edges, negatives
sampled proportional to dst-degree^0.75. Unlike LINE's BCE, this directly
optimizes a ranking criterion, and its direct score e_src.e_cand proved
complementary to the LINE blend on the leak-free holdout eval (dataset2
honest +0.017 at blend weight 0.7, 2026-07-20).

Usage:
  DATASET=dataset2 python src/train_bpr.py                # -> outputs/dataset2-bpr/bpr_emb.npy
  DATASET=dataset2 EVAL_HOLDOUT=1 python src/train_bpr.py # leak-free variant for offline eval
Knobs: BPR_DIM (256), BPR_EPOCHS (120), SEED (42), BPR_TAU_FRAC (0),
BPR_TIME_MAX (0), BPR_INNOV (0), BPR_SAVE_EVERY (0, intermediate epoch
snapshots).

BPR_TAU_FRAC > 0 enables recency-weighted positive sampling: pairs are drawn
with replacement proportional to exp(-(t_max - t) / (frac * time_span))
instead of a uniform permutation. Validated on the honest holdout AND the
leaky protocol together (both agreed, 2026-07-20): dataset2 frac=0.10,
dataset1 frac=0.25.

BPR_INNOV=1 trains the innovation-only variant: only the FIRST occurrence of
each (src, dst) pair is kept, so the embedding specializes in new-partner
selection instead of being dominated by recurring pairs (66% of dataset1
mass). Blended via ensemble_predict's W_IBPR term, masked to non-history
candidates. Validated 2026-07-23 on both calibers (honest +0.0033 at w12,
leaky same direction), targeting dataset1's non-repeat loss block.
"""
import os
import time

import numpy as np
import pandas as pd
import torch

import train_line as tl

SEED = int(os.environ.get("SEED", "42"))
DIM = int(os.environ.get("BPR_DIM", "256"))
EPOCHS = int(os.environ.get("BPR_EPOCHS", "120"))
TAU_FRAC = float(os.environ.get("BPR_TAU_FRAC", "0"))
# Optional hard time cutoff: train only on edges with time <= BPR_TIME_MAX.
# Used to build feature-period-only embeddings for time-sliced ranker labels.
TIME_MAX = float(os.environ.get("BPR_TIME_MAX", "0"))
# Innovation-only mode: keep only the first occurrence of each (src, dst) pair.
INNOV = os.environ.get("BPR_INNOV", "0") == "1"
# Optional intermediate snapshots: if > 0, also save bpr_emb_ep{N}.npy every N
# epochs. Lets a single run sweep the whole epoch axis for the under-training /
# early-stopping transfer study without retraining once per epoch count.
SAVE_EVERY = int(os.environ.get("BPR_SAVE_EVERY", "0"))
LR = 3e-3
L2 = 1e-6
BATCH = 8192

# Non-default knobs write to a suffixed dir (same convention as train_line)
# so multi-seed / time-weighted ensemble runs never clobber each other.
OUT_DIR = tl.PROJECT_ROOT / "outputs" / (
    tl.DATASET + "-bpr"
    + ("-innov" if INNOV else "")
    + (f"-t{TAU_FRAC:g}" if TAU_FRAC > 0 else "")
    + (f"-d{DIM}" if DIM != 256 else "")
    + (f"-tmax{TIME_MAX:g}" if TIME_MAX > 0 else "")
    + (f"-s{SEED}" if SEED != 42 else "")
    + ("-holdout" if tl.EVAL_HOLDOUT else "")
)
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    device = tl.device
    print(f"Dataset: {tl.DATASET} | device: {device} | dim {DIM} | epochs {EPOCHS} | "
          f"seed {SEED} | tau_frac {TAU_FRAC:g}")
    df = pd.read_csv(tl.train_csv)
    df = df.drop_duplicates(subset=["src", "dst", "time"]).reset_index(drop=True)
    df["src"] = df["src"].astype(np.int64)
    df["dst"] = df["dst"].astype(np.int64)
    df["time"] = df["time"].astype(float)
    if tl.EVAL_HOLDOUT:
        n_before = len(df)
        df, _ = tl.split_train_val_by_tail(df)
        print(f"[EVAL_HOLDOUT] Dropped {n_before - len(df)} per-src tail rows from training data")
    num_entity = int(max(df.src.max(), df.dst.max())) + 1
    if TIME_MAX > 0:
        n_before = len(df)
        df = df[df["time"] <= TIME_MAX].reset_index(drop=True)
        print(f"[BPR_TIME_MAX] Kept {len(df)}/{n_before} edges with time <= {TIME_MAX:g}")
    if INNOV:
        n_before = len(df)
        df = df.sort_values("time").drop_duplicates(["src", "dst"], keep="first").reset_index(drop=True)
        print(f"[BPR_INNOV] Kept {len(df)}/{n_before} first-time (src, dst) links")

    torch.manual_seed(SEED)
    emb = torch.nn.Embedding(num_entity, DIM).to(device)
    torch.nn.init.normal_(emb.weight, std=0.1)
    opt = torch.optim.Adam(emb.parameters(), lr=LR, weight_decay=L2)

    # degree^0.75 negative CDF (same convention as LINE's pop075 mode)
    pw = np.bincount(df["dst"].values, minlength=num_entity).astype(np.float64) + 1.0
    cdf = torch.from_numpy(np.cumsum(pw ** 0.75 / (pw ** 0.75).sum())).to(device)
    s_all = torch.from_numpy(df["src"].values).to(device)
    d_all = torch.from_numpy(df["dst"].values).to(device)
    n_pos = len(s_all)
    print(f"Training pairs: {n_pos}, nodes: {num_entity}")

    # Recency-weighted positive sampling CDF (BPR_TAU_FRAC > 0): pairs drawn
    # with replacement proportional to exp(-age / tau), tau = frac * span.
    pair_cdf = None
    if TAU_FRAC > 0:
        t_arr = df["time"].values.astype(np.float64)
        tau = TAU_FRAC * (t_arr.max() - t_arr.min())
        pair_w = np.exp(-(t_arr.max() - t_arr) / tau)
        pair_cdf = torch.from_numpy(np.cumsum(pair_w / pair_w.sum())).to(device)

    t0 = time.time()
    for ep_i in range(EPOCHS):
        perm = None if pair_cdf is not None else torch.randperm(n_pos, device=device)
        total = 0.0
        batches = 0
        for beg in range(0, n_pos, BATCH):
            if pair_cdf is not None:
                idx = torch.searchsorted(
                    pair_cdf,
                    torch.rand(min(BATCH, n_pos - beg), device=device, dtype=torch.float64),
                ).clamp_(0, n_pos - 1)
            else:
                idx = perm[beg:beg + BATCH]
            u, v = s_all[idx], d_all[idx]
            vneg = torch.searchsorted(
                cdf, torch.rand(len(idx), device=device, dtype=torch.float64)
            ).clamp_(0, num_entity - 1)
            eu = emb(u)
            x = (eu * emb(v)).sum(1) - (eu * emb(vneg)).sum(1)
            loss = torch.nn.functional.softplus(-x).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            batches += 1
        if (ep_i + 1) % 20 == 0:
            print(f"[BPR epoch {ep_i + 1}/{EPOCHS}] avg loss {total / batches:.4f} ({time.time() - t0:.0f}s)")
        if SAVE_EVERY > 0 and (ep_i + 1) % SAVE_EVERY == 0 and (ep_i + 1) != EPOCHS:
            snap = OUT_DIR / f"bpr_emb_ep{ep_i + 1}.npy"
            np.save(str(snap), emb.weight.detach().cpu().numpy().astype(np.float32))
            print(f"[snapshot] {snap}")

    # Atomic replace; np.save appends .npy itself, so give the tmp file that suffix
    out_path = OUT_DIR / "bpr_emb.npy"
    tmp_path = OUT_DIR / "bpr_emb.tmp.npy"
    np.save(str(tmp_path), emb.weight.detach().cpu().numpy().astype(np.float32))
    os.replace(tmp_path, out_path)
    print(f"[OK] BPR embedding saved: {out_path}")


if __name__ == "__main__":
    main()
