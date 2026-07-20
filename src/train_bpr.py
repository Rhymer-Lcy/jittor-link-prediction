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
Knobs: BPR_DIM (256), BPR_EPOCHS (120), SEED (42).
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
LR = 3e-3
L2 = 1e-6
BATCH = 8192

OUT_DIR = tl.PROJECT_ROOT / "outputs" / (tl.DATASET + "-bpr" + ("-holdout" if tl.EVAL_HOLDOUT else ""))
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    device = tl.device
    print(f"Dataset: {tl.DATASET} | device: {device} | dim {DIM} | epochs {EPOCHS} | seed {SEED}")
    df = pd.read_csv(tl.train_csv)
    df = df.drop_duplicates(subset=["src", "dst", "time"]).reset_index(drop=True)
    df["src"] = df["src"].astype(np.int64)
    df["dst"] = df["dst"].astype(np.int64)
    if tl.EVAL_HOLDOUT:
        n_before = len(df)
        df["time"] = df["time"].astype(float)
        df, _ = tl.split_train_val_by_tail(df)
        print(f"[EVAL_HOLDOUT] Dropped {n_before - len(df)} per-src tail rows from training data")
    num_entity = int(max(df.src.max(), df.dst.max())) + 1

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

    t0 = time.time()
    for ep_i in range(EPOCHS):
        perm = torch.randperm(n_pos, device=device)
        total = 0.0
        batches = 0
        for beg in range(0, n_pos, BATCH):
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

    # Atomic replace; np.save appends .npy itself, so give the tmp file that suffix
    out_path = OUT_DIR / "bpr_emb.npy"
    tmp_path = OUT_DIR / "bpr_emb.tmp.npy"
    np.save(str(tmp_path), emb.weight.detach().cpu().numpy().astype(np.float32))
    os.replace(tmp_path, out_path)
    print(f"[OK] BPR embedding saved: {out_path}")


if __name__ == "__main__":
    main()
