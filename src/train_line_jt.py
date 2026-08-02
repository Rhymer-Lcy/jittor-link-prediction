# -*- coding: utf-8 -*-
"""LINE embedding trainer (Jittor). One of the two neural stages of the pipeline.

Trains a LINE model on the bidirectional training edges and exports
`line_latest_emb.csv` in the exact format `ensemble_predict.load_embedding`
reads (node_id + concat(emb_first, emb_node), rounded to 6 decimals). Three
embedding tables are held: a first-order pair table and a second-order
node/context pair, optimised jointly under `loss1 + 0.5 * loss2`.

Everything downstream of this stage (similar-user cache, item-CF, blending,
ranker, CRF) is numpy/LightGBM and uses no training framework.

Division of labour: sampling, shuffling and negative rejection run in seeded
numpy; Jittor owns the embedding tables, the loss and Adam only. This keeps the
framework-dependent surface small and makes the sampling stream reproducible
independently of the framework.

Virtual-edge self-training is deliberately omitted (VIRT_MODE=off), a measured
no-op; the `-novirt` marker in the run-directory name records that choice. The
in-training predict/MRR cycle is likewise omitted; run `ensemble_predict.py
--eval` separately if an evaluation is wanted.

Knobs: DATASET, DATA_PACK, EMB_DIM (400 total, split into two sub_dim halves),
NEG_RATIO (5), NEG_DIST (uniform | pop075), EPOCHS (400), SEED (42),
LINE_TIME_MAX (0 = full history; set to the cut for a cutoff run),
LINE_TAU_FRAC (0), EVAL_HOLDOUT. JT_OUT_SUFFIX (empty in the production
release) can suffix the output directory so a verification run cannot clobber a
production artifact.

Usage: DATASET=dataset1 python src/train_line_jt.py
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import jittor as jt
from jittor import nn

import pipeline_common as pc

SEED = int(os.environ.get("SEED", "42"))
emb_total_dim = int(os.environ.get("EMB_DIM", "400"))
sub_dim = emb_total_dim // 2
neg_ratio = int(os.environ.get("NEG_RATIO", "5"))
epochs = int(os.environ.get("EPOCHS", "400"))
batch_size = 1024
EMB_PRECISION = 6
LOSS_ALPHA = 0.5
GRAD_CLIP = 1.0
INIT_LR = 1e-4
EXPORT_EVERY = 10
VAL_PER_SRC_TAIL = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = os.environ.get("DATASET", "dataset2")
assert DATASET in ("dataset1", "dataset2"), f"unknown dataset: {DATASET}"
DATA_PACK = os.environ.get("DATA_PACK", "data_A")
DATA_DIR = pc.data_dir(DATASET, DATA_PACK)   # one definition, shared with the consumers
NEG_DIST = os.environ.get("NEG_DIST", "uniform")
assert NEG_DIST in ("uniform", "pop075"), f"unknown NEG_DIST: {NEG_DIST}"
EVAL_HOLDOUT = os.environ.get("EVAL_HOLDOUT", "0") == "1"
LINE_TIME_MAX = float(os.environ.get("LINE_TIME_MAX", "0"))
LINE_TAU_FRAC = float(os.environ.get("LINE_TAU_FRAC", "0"))
# Canonical runs write the contract name. JT_OUT_SUFFIX remains available for
# side-by-side verification runs, but defaults to empty so a canonical run needs
# no environment variable and cannot silently land in a parallel directory.
JT_OUT_SUFFIX = os.environ.get("JT_OUT_SUFFIX", "")

# Single source of truth, shared with the rankers that consume this output.
_BASE_DIR = pc.line_run_dir(DATASET, seed=SEED, neg_dist=NEG_DIST, emb_dim=emb_total_dim,
                            time_max=LINE_TIME_MAX, holdout=EVAL_HOLDOUT, root=PROJECT_ROOT)
OUTPUT_DIR = _BASE_DIR.parent / (_BASE_DIR.name + JT_OUT_SUFFIX)
os.makedirs(OUTPUT_DIR, exist_ok=True)
latest_emb_path = OUTPUT_DIR / "line_latest_emb.csv"

jt.flags.use_cuda = 1 if jt.has_cuda else 0


class LINE(nn.Module):
    def __init__(self, n_node, d_sub, rng):
        super().__init__()
        self.emb_first = nn.Embedding(n_node, d_sub)
        self.emb_node = nn.Embedding(n_node, d_sub)
        self.emb_ctx = nn.Embedding(n_node, d_sub)
        # Xavier-uniform from one numpy stream: one seed for all three tables.
        # Reseeding per table would make the three identical at initialisation.
        bound = float(np.sqrt(6.0 / (n_node + d_sub)))
        for e in (self.emb_first, self.emb_node, self.emb_ctx):
            e.weight.assign(jt.array(
                rng.uniform(-bound, bound, (n_node, d_sub)).astype(np.float32)))

    def score_first(self, s, d):
        return (self.emb_first(s) * self.emb_first(d)).sum(-1)

    def score_second(self, s, d):
        return (self.emb_node(s) * self.emb_ctx(d)).sum(-1)

    def get_final_emb(self):
        return np.concatenate(
            [self.emb_first.weight.numpy(), self.emb_node.weight.numpy()], axis=1)


def split_train_val_by_tail(df):
    df = df.sort_values(["src", "time"]).reset_index(drop=True)
    val_df = df.groupby("src", sort=False).tail(VAL_PER_SRC_TAIL)
    train_df = df.drop(val_df.index).reset_index(drop=True)
    return train_df, val_df.reset_index(drop=True)


def build_pos_keys(pos_set, n_node):
    arr = np.fromiter((u * n_node + v for u, v in pos_set), dtype=np.int64, count=len(pos_set))
    arr.sort()
    return arr


def gen_neg_epoch(s_pos_np, pos_keys, n_node, neg_cdf, rng):
    """Numpy rejection sampling of the epoch's negatives, drawn against the
    observed-pair key set."""
    s_rep = np.repeat(s_pos_np.astype(np.int64), neg_ratio)
    base = s_rep * n_node

    def draw(n):
        if neg_cdf is not None:
            return np.searchsorted(neg_cdf, rng.random(n)).clip(0, n_node - 1)
        return rng.integers(0, n_node, n)

    def collides(pos):
        key = base[pos] + d[pos]
        loc = np.searchsorted(pos_keys, key).clip(max=len(pos_keys) - 1)
        return pos[pos_keys[loc] == key]

    d = draw(len(s_rep))
    bad = collides(np.arange(len(d)))
    for _ in range(30):
        if len(bad) == 0:
            break
        d[bad] = draw(len(bad))
        bad = collides(bad)
    return d.astype(np.int32)


def export_emb(model, num_entity):
    emb_np = np.round(model.get_final_emb(), decimals=EMB_PRECISION)
    emb_df = pd.DataFrame(emb_np)
    emb_df.insert(0, "node_id", list(range(num_entity)))
    tmp = latest_emb_path.with_suffix(".csv.tmp")
    emb_df.to_csv(tmp, index=False)
    os.replace(tmp, latest_emb_path)


def main():
    print(f"Dataset: {DATASET} ({DATA_PACK}) | jittor cuda={jt.flags.use_cuda} | "
          f"dim {emb_total_dim} | epochs {epochs} | seed {SEED} | neg {NEG_DIST}")
    df_raw = pd.read_csv(DATA_DIR / "train.csv")
    df_raw = df_raw.drop_duplicates(subset=["src", "dst", "time"]).reset_index(drop=True)
    df_raw["src"] = df_raw["src"].astype(np.int64)
    df_raw["dst"] = df_raw["dst"].astype(np.int64)
    df_raw["time"] = df_raw["time"].astype(float)
    if EVAL_HOLDOUT:
        n_before = len(df_raw)
        df_raw, _ = split_train_val_by_tail(df_raw)
        print(f"[EVAL_HOLDOUT] Dropped {n_before - len(df_raw)} per-src tail rows")
    # The entity table is sized from the UNFILTERED frame, before any time
    # cutoff, so node ids stay aligned with full-data artifacts. train_bpr_jt.py
    # sizes its table the same way, for the same reason.
    #
    # Sizing it after the cutoff instead shrinks the embedding table to only the
    # nodes seen before the cut (dataset1: 42,361 instead of 43,215), which
    # silently misaligns every downstream node id and makes ranker_ds1 fail with
    # "embedding has 42361 rows, expected 43215". Nodes that appear only after
    # the cut keep their initialised rows, which is the frozen behaviour.
    num_entity = int(max(df_raw.src.max(), df_raw.dst.max())) + 1
    if LINE_TIME_MAX > 0:
        n_before = len(df_raw)
        df_raw = df_raw[df_raw["time"] <= LINE_TIME_MAX].reset_index(drop=True)
        print(f"[LINE_TIME_MAX] Kept {len(df_raw)}/{n_before} edges with time <= {LINE_TIME_MAX:g}")

    # Bidirectional real edges, interleaved [u,v],[v,u]
    uv = df_raw[["src", "dst"]].values.astype(np.int64)
    edges = np.empty((2 * len(uv), 2), dtype=np.int64)
    edges[0::2] = uv
    edges[1::2] = uv[:, ::-1]
    pos_set = set()
    for u, v in uv:
        pos_set.add((int(u), int(v)))
        pos_set.add((int(v), int(u)))
    pos_keys = build_pos_keys(pos_set, num_entity)
    pos_cnt = len(edges)
    print(f"Training pairs (bidirectional): {pos_cnt}, nodes: {num_entity}")

    neg_cdf = None
    if NEG_DIST == "pop075":
        dst_pop = np.bincount(df_raw["dst"].values, minlength=num_entity).astype(np.float64) + 1.0
        _pw = dst_pop ** 0.75
        neg_cdf = np.cumsum(_pw / _pw.sum())
        print("Negative sampling: degree^0.75 (pop075)")

    pos_cdf = None
    if LINE_TAU_FRAC > 0:
        t_arr = np.repeat(df_raw["time"].values.astype(np.float64), 2)
        tau = LINE_TAU_FRAC * (t_arr.max() - t_arr.min())
        w = np.exp(-(t_arr.max() - t_arr) / tau)
        pos_cdf = np.cumsum(w / w.sum())
        print(f"Recency-weighted positive sampling: tau = {LINE_TAU_FRAC:g} * span")

    rng = np.random.default_rng(SEED)
    jt.set_global_seed(SEED)
    model = LINE(num_entity, sub_dim, rng)
    opt = jt.optim.Adam(model.parameters(), lr=INIT_LR)

    t0 = time.time()
    for ep_i in range(epochs):
        if pos_cdf is not None:
            perm = np.searchsorted(pos_cdf, rng.random(pos_cnt)).clip(0, pos_cnt - 1)
        else:
            perm = rng.permutation(pos_cnt)
        pos_shuffle = edges[perm]
        d_neg_epoch = gen_neg_epoch(pos_shuffle[:, 0], pos_keys, num_entity, neg_cdf, rng)
        total = 0.0
        batches = 0
        for beg in range(0, pos_cnt, batch_size):
            end = min(beg + batch_size, pos_cnt)
            s_pos = pos_shuffle[beg:end, 0].astype(np.int32)
            d_pos = pos_shuffle[beg:end, 1].astype(np.int32)
            s_neg = np.repeat(s_pos, neg_ratio)
            d_neg = d_neg_epoch[beg * neg_ratio:end * neg_ratio]
            s_batch = jt.array(np.concatenate([s_pos, s_neg]))
            d_batch = jt.array(np.concatenate([d_pos, d_neg]))
            label = jt.array(np.concatenate(
                [np.ones(len(s_pos), np.float32), np.zeros(len(s_neg), np.float32)]))
            scr1 = model.score_first(s_batch, d_batch)
            scr2 = model.score_second(s_batch, d_batch)
            loss = (nn.binary_cross_entropy_with_logits(scr1, label)
                    + LOSS_ALPHA * nn.binary_cross_entropy_with_logits(scr2, label))
            opt.zero_grad()
            opt.backward(loss)
            opt.clip_grad_norm(GRAD_CLIP, 2)
            opt.step()
            total += float(loss.item())
            batches += 1
        print(f"[LINE epoch {ep_i + 1}/{epochs}] avg loss {total / batches:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if (ep_i + 1) % EXPORT_EVERY == 0:
            export_emb(model, num_entity)

    export_emb(model, num_entity)
    print(f"[OK] Final embedding exported: {latest_emb_path}")


if __name__ == "__main__":
    main()
