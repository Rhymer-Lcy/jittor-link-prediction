#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strict dataset2-only A/B probe for the temporal-footprint channel.

This script deliberately leaves the live three-pass label-basket ranker
untouched.  It learns a *paired nuisance residual* on the exact hzeval replay:

    delta = row_minmax(score(base18 + footprint85))
            - row_minmax(score(base18))

The two nuisance models see identical labels, groups, folds, base columns and
LightGBM parameters.  On the hzeval gate, adding ``delta`` to the paired base
is algebraically identical to the 103-column model and reproduces the audited
round-11 footprint gate.  For the online probe, the paired full models predict
the residual on the physical full-test exposure population; that residual is
added once to the existing label-basket *pre-CRF* scores.  Both arms then use
the exact same CRF/triple/zrxst/no-pair implementation.

This is an isolation probe for whether the test-file footprint transfers.  It
is not a claim that the raw 85 columns were retrained inside all three live
basket passes.

Outputs (all under outputs/dataset2-footprint/ and therefore gitignored):

    pack_A_ds2only.zip
    pack_B_ds2only.zip
    pack_B_footprint_precrf.csv
    footprint_ab_probe_metrics.json
    models/*.txt                         (trained paired boosters)
    cache/*                              (resumable large arrays)

Examples:

    python src/footprint_ab_probe.py \
        --gate-only

    python src/footprint_ab_probe.py \
        --full
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# train_line selects its data and artifact family at import time.
if os.environ.get("DATASET", "dataset2") != "dataset2":
    raise RuntimeError("footprint_ab_probe.py is dataset2-only; unset DATASET or set it to dataset2")
os.environ["DATASET"] = "dataset2"
if os.environ.get("DATA_PACK", "data_A") != "data_A":
    raise RuntimeError("this exact A/B probe is pinned to DATA_PACK=data_A")
os.environ["DATA_PACK"] = "data_A"

import crf_promote as cp  # noqa: E402
import ensemble_predict as ep  # noqa: E402
import footprint_feature as ff  # noqa: E402
import train_line as tl  # noqa: E402


DATASET_DIR = REPO / "data" / "data_A" / "dataset2"
DEFAULT_HZEVAL = ff.DEFAULT_HZEVAL
LIVE_PRECRF = REPO / "outputs" / "dataset2-ranker" / "ranker_basket3_label_dataset2.csv"
KNOWN_A = REPO / "outputs" / "submissions" / "ds2_batch7_ab" / "label_ds2only.zip"

# All heavy artifacts (packs, caches, trained boosters, metrics) live under the
# gitignored reproduction directory, never inside the tracked src/ tree.
WORKDIR = REPO / "outputs" / "dataset2-footprint"
PACK_A = WORKDIR / "pack_A_ds2only.zip"
PACK_B = WORKDIR / "pack_B_ds2only.zip"
PRECRF_B = WORKDIR / "pack_B_footprint_precrf.csv"
METRICS = WORKDIR / "footprint_ab_probe_metrics.json"
MODEL_DIR = WORKDIR / "models"
CACHE_DIR = WORKDIR / "cache"

N_SLATE = 100
N_BASE = 18
N_FOOT = ff.N_FEATURES
N_AUG = N_BASE + N_FOOT
EXPECTED_ROWS = 153420
EXPECTED_GATE_BASE = 0.61228336
EXPECTED_GATE_DELTA = 0.008116656

KNOWN_A_ZIP_MD5 = "a7e731245863803b9a8cf06628439868"
KNOWN_A_INNER_MD5 = "ae750503904fd7275f1e6f6544d2a7c8"
KNOWN_PRECRF_MD5 = "658b3fd7d5ec7c4db4d561a46a7fd199"

PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    ndcg_eval_at=[10],
    label_gain=[0, 1],
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    random_state=42,
    n_jobs=6,
    verbosity=-1,
)
MODEL_SIGNATURE: Dict[str, object] = {}
ADOPT_UNMARKED_MODELS = False
SEEDS = (42, 123, 777, 2024, 31337)
W = {"collab": 0.25, "rpop": 1.25, "icfm": 0.5, "icf3": 1.0, "bpr": 0.7}
YEAR = 365.0 * 86400.0


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_md5(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def bytes_md5(value: bytes) -> str:
    return hashlib.md5(value).hexdigest()


def json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_metrics() -> dict:
    if METRICS.is_file():
        return json.loads(METRICS.read_text(encoding="utf-8"))
    return {}


def row_minmax(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_SLATE:
        raise ValueError(f"expected (Q,{N_SLATE}) scores; got {values.shape}")
    lo = values.min(axis=1, keepdims=True)
    hi = values.max(axis=1, keepdims=True)
    result = np.divide(
        values - lo,
        hi - lo,
        out=np.full_like(values, 0.5),
        where=(hi - lo) > 0,
    )
    if not np.isfinite(result).all():
        raise FloatingPointError("row_minmax produced non-finite values")
    return result


def crf_minmax(scores: np.ndarray) -> np.ndarray:
    """Match crf_promote.py exactly, including its constant-row behavior."""

    values = np.asarray(scores, dtype=np.float64)
    lo = values.min(axis=1, keepdims=True)
    hi = values.max(axis=1, keepdims=True)
    result = (values - lo) / np.maximum(hi - lo, 1e-12)
    if not np.isfinite(result).all():
        raise FloatingPointError("crf_minmax produced non-finite values")
    return result


def reciprocal_ranks(scores: np.ndarray) -> np.ndarray:
    """Match the project's pessimistic tie convention; truth is column 99."""

    scores = np.asarray(scores)
    positive = scores[:, -1]
    negative = scores[:, :-1]
    rank = (
        1
        + (negative > positive[:, None]).sum(axis=1)
        + (negative == positive[:, None]).sum(axis=1)
    )
    return 1.0 / rank


def query_cell_indices(query_indices: np.ndarray) -> np.ndarray:
    query_indices = np.asarray(query_indices, dtype=np.int64)
    return (
        query_indices[:, None] * N_SLATE
        + np.arange(N_SLATE, dtype=np.int64)[None, :]
    ).ravel()


def source_folds(src: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)
    unique = np.unique(np.asarray(src, dtype=np.int64)).copy()
    rng.shuffle(unique)
    left = set(unique[: len(unique) // 2].tolist())
    fold = np.fromiter(
        (0 if int(value) in left else 1 for value in src),
        dtype=np.int8,
        count=len(src),
    )
    assert set(np.unique(fold).tolist()) == {0, 1}
    assert not set(src[fold == 0]).intersection(set(src[fold == 1]))
    return fold


def ensure_npy(
    path: Path,
    shape: Tuple[int, ...],
    dtype: np.dtype,
) -> Optional[np.memmap]:
    if not path.is_file():
        return None
    array = np.load(path, mmap_mode="r")
    if tuple(array.shape) != tuple(shape) or array.dtype != np.dtype(dtype):
        raise AssertionError(
            f"stale cache {path}: shape/dtype {array.shape}/{array.dtype}, "
            f"expected {shape}/{np.dtype(dtype)}"
        )
    return array


def npy_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".meta.json")


def mark_npy_complete(path: Path, provenance: dict) -> dict:
    array = np.load(path, mmap_mode="r")
    metadata = {
        "path": str(path),
        "shape": list(map(int, array.shape)),
        "dtype": str(array.dtype),
        "bytes": int(path.stat().st_size),
        "md5": file_md5(path),
        "provenance": provenance,
    }
    del array
    json_write(npy_metadata_path(path), metadata)
    return metadata


def completed_npy(
    path: Path,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    *,
    adopt_unmarked: bool = False,
    adoption_provenance: Optional[dict] = None,
) -> Optional[np.memmap]:
    array = ensure_npy(path, shape, dtype)
    if array is None:
        return None
    metadata_path = npy_metadata_path(path)
    if not metadata_path.is_file():
        if not adopt_unmarked:
            raise AssertionError(
                f"cache exists without a completion marker: {path}; "
                "rerun with --force"
            )
        log(f"adopting explicitly requested unmarked cache: {path}")
        mark_npy_complete(
            path,
            {
                "adopted_unmarked_cache": True,
                **(adoption_provenance or {}),
            },
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("shape") != list(map(int, shape)):
        raise AssertionError(f"cache metadata shape mismatch: {path}")
    if metadata.get("dtype") != str(np.dtype(dtype)):
        raise AssertionError(f"cache metadata dtype mismatch: {path}")
    if metadata.get("bytes") not in (None, int(path.stat().st_size)):
        raise AssertionError(f"cache metadata byte-size mismatch: {path}")
    expected_md5 = metadata.get("md5")
    if expected_md5 and expected_md5 != file_md5(path):
        raise AssertionError(f"cache MD5 mismatch: {path}")
    return array


def build_cut_footprint(
    hzeval: Path,
    chunk_size: int,
    force: bool,
) -> Tuple[np.memmap, dict]:
    output = CACHE_DIR / "cut_footprint_60000x100x85.npy"
    expected = (60000, N_SLATE, N_FOOT)
    cached = None if force else completed_npy(output, expected, np.float32)
    if cached is None:
        log("replaying the complete 153,420-row CUT footprint population")
        generator, keep = ff.build_generator(
            "cut-frozen",
            DATASET_DIR,
            hzeval,
            verify_hzeval=True,
        )
        assert len(keep) == expected[0]
        ff.write_npy(generator, keep, output, chunk_size=chunk_size)
        del generator
        gc.collect()
        # ff.write_npy writes the same completion metadata contract.
        cached = completed_npy(output, expected, np.float32)
        assert cached is not None
    report = streamed_column_stats(cached, row_chunk=max(64, chunk_size))
    report["path"] = str(output)
    report["md5"] = file_md5(output)
    return cached, report


def build_gate_augmented(
    base: np.ndarray,
    footprint: np.ndarray,
    force: bool,
    chunk_queries: int = 1000,
) -> np.memmap:
    q = len(footprint)
    output = CACHE_DIR / "gate_aug103.npy"
    expected = (q * N_SLATE, N_AUG)
    cached = (
        None
        if force
        else completed_npy(
            output,
            expected,
            np.float32,
            adopt_unmarked=ADOPT_UNMARKED_MODELS,
            adoption_provenance={
                "kind": "gate_aug103",
                "model_signature": MODEL_SIGNATURE,
            },
        )
    )
    if cached is not None:
        # A cheap but strict sample check catches stale column order.
        sample_q = np.array([0, q // 2, q - 1], dtype=np.int64)
        sample_c = query_cell_indices(sample_q)
        assert np.array_equal(cached[sample_c, :N_BASE], base[sample_c])
        assert np.array_equal(
            cached[sample_c, N_BASE:].reshape(-1, N_SLATE, N_FOOT),
            footprint[sample_q],
        )
        return cached

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mapped = np.lib.format.open_memmap(
        output,
        mode="w+",
        dtype=np.float32,
        shape=expected,
    )
    for begin in range(0, q, chunk_queries):
        end = min(begin + chunk_queries, q)
        cells = slice(begin * N_SLATE, end * N_SLATE)
        mapped[cells, :N_BASE] = base[cells]
        mapped[cells, N_BASE:] = footprint[begin:end].reshape(-1, N_FOOT)
        assert np.array_equal(mapped[cells, :N_BASE], base[cells])
        mapped.flush()
        log(f"gate augmented matrix {end}/{q}")
    del mapped
    mark_npy_complete(
        output,
        {
            "kind": "gate_aug103",
            "base_columns": N_BASE,
            "footprint_columns": N_FOOT,
            "model_signature": MODEL_SIGNATURE,
        },
    )
    cached = completed_npy(output, expected, np.float32)
    assert cached is not None
    return cached


def streamed_column_stats(
    array: np.ndarray,
    *,
    row_chunk: int = 512,
) -> dict:
    if array.ndim != 3 or array.shape[1:] != (N_SLATE, N_FOOT):
        raise ValueError(f"unexpected footprint tensor: {array.shape}")
    total = np.zeros(N_FOOT, np.float64)
    total_sq = np.zeros(N_FOOT, np.float64)
    minimum = np.full(N_FOOT, np.inf)
    maximum = np.full(N_FOOT, -np.inf)
    count = 0
    for begin in range(0, len(array), row_chunk):
        part = np.asarray(array[begin : begin + row_chunk]).reshape(-1, N_FOOT)
        if not np.isfinite(part).all():
            raise FloatingPointError(f"non-finite footprint values near row {begin}")
        values = part.astype(np.float64, copy=False)
        count += len(values)
        total += values.sum(axis=0)
        total_sq += np.square(values).sum(axis=0)
        minimum = np.minimum(minimum, values.min(axis=0))
        maximum = np.maximum(maximum, values.max(axis=0))
    mean = total / count
    variance = np.maximum(total_sq / count - mean * mean, 0.0)
    return {
        "shape": list(map(int, array.shape)),
        "all_finite": True,
        "columns": [
            {
                "index": i,
                "name": ff.FEATURE_NAMES[i],
                "mean": float(mean[i]),
                "std": float(np.sqrt(variance[i])),
                "min": float(minimum[i]),
                "max": float(maximum[i]),
            }
            for i in range(N_FOOT)
        ],
    }


def fit_ranker(
    x: np.ndarray,
    y: np.ndarray,
    query_indices: np.ndarray,
    model_path: Path,
    *,
    force: bool,
) -> lgb.Booster:
    if not MODEL_SIGNATURE:
        raise RuntimeError("model-cache signature was not initialized")
    metadata_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    if model_path.is_file() and not force:
        if not metadata_path.is_file():
            if not ADOPT_UNMARKED_MODELS:
                raise AssertionError(
                    f"model cache lacks provenance metadata: {model_path}; "
                    "rerun with --force"
                )
            booster = lgb.Booster(model_file=str(model_path))
            if booster.num_feature() != int(x.shape[1]):
                raise AssertionError(
                    f"unmarked model feature-count mismatch: {model_path}"
                )
            if booster.num_trees() != int(PARAMS["n_estimators"]):
                raise AssertionError(
                    f"unmarked model tree-count mismatch: {model_path}"
                )
            json_write(
                metadata_path,
                {
                    "model": str(model_path),
                    "feature_count": int(x.shape[1]),
                    "training_queries": int(len(query_indices)),
                    "adopted_unmarked_model": True,
                    "signature": MODEL_SIGNATURE,
                },
            )
            return booster
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("signature") != MODEL_SIGNATURE:
            raise AssertionError(
                f"model cache signature mismatch: {model_path}; "
                "rerun with --force"
            )
        return lgb.Booster(model_file=str(model_path))
    all_queries = (
        len(query_indices) * N_SLATE == len(x)
        and np.array_equal(
            query_indices,
            np.arange(len(query_indices), dtype=np.int64),
        )
    )
    if all_queries:
        # Preserve the memmap for the 2.47 GB full augmented matrix instead of
        # making a second same-sized fancy-index copy.
        cells = None
        train_x = np.asarray(x)
        train_y = np.asarray(y)
    else:
        cells = query_cell_indices(query_indices)
        # Fancy indexing preserves complete contiguous 100-row groups while
        # bounding a fold materialisation to roughly half the full matrix.
        train_x = np.asarray(x[cells], dtype=np.float32)
        train_y = np.asarray(y[cells], dtype=np.float32)
    assert len(train_x) == len(query_indices) * N_SLATE
    model = lgb.LGBMRanker(**PARAMS)
    model.fit(
        train_x,
        train_y,
        group=[N_SLATE] * len(query_indices),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path))
    json_write(
        metadata_path,
        {
            "model": str(model_path),
            "feature_count": int(x.shape[1]),
            "training_queries": int(len(query_indices)),
            "signature": MODEL_SIGNATURE,
        },
    )
    booster = model.booster_
    del model, train_x, train_y, cells
    gc.collect()
    return booster


def predict_queries(
    model: lgb.Booster,
    x: np.ndarray,
    query_indices: np.ndarray,
    *,
    chunk_queries: int = 2000,
) -> np.ndarray:
    output = np.empty((len(query_indices), N_SLATE), np.float64)
    for begin in range(0, len(query_indices), chunk_queries):
        end = min(begin + chunk_queries, len(query_indices))
        part_q = query_indices[begin:end]
        cells = query_cell_indices(part_q)
        pred = model.predict(np.asarray(x[cells], dtype=np.float32))
        output[begin:end] = pred.reshape(-1, N_SLATE)
    return output


def run_gate(
    hzeval_path: Path,
    *,
    chunk_size: int,
    force: bool,
) -> Tuple[dict, lgb.Booster, lgb.Booster, np.ndarray, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log(f"loading hzeval: {hzeval_path}")
    with np.load(hzeval_path) as asset:
        x_base = np.asarray(asset["X"], dtype=np.float32)
        y = np.asarray(asset["y"], dtype=np.float32)
        src = np.asarray(asset["src"], dtype=np.int64)
        cand = np.asarray(asset["cand"], dtype=np.int32)
        assert int(asset["n_slate"][0]) == N_SLATE
    q = len(src)
    assert x_base.shape == (q * N_SLATE, N_BASE)
    assert y.shape == (q * N_SLATE,)
    assert cand.shape == (q, N_SLATE)
    assert np.all(y.reshape(q, N_SLATE)[:, -1] == 1)
    assert np.all(y.reshape(q, N_SLATE)[:, :-1] == 0)

    footprint, footprint_report = build_cut_footprint(
        hzeval_path, chunk_size, force
    )
    x_aug = build_gate_augmented(x_base, footprint, force)
    assert x_aug.shape == (q * N_SLATE, N_AUG)

    fold = source_folds(src)
    oof_base = np.empty((q, N_SLATE), np.float64)
    oof_aug = np.empty((q, N_SLATE), np.float64)
    fold_metrics = []
    for held in (0, 1):
        train_q = np.flatnonzero(fold != held)
        test_q = np.flatnonzero(fold == held)
        base_path = MODEL_DIR / f"gate_fold{held}_base18.txt"
        aug_path = MODEL_DIR / f"gate_fold{held}_aug103.txt"
        log(f"gate fold {held}: train {len(train_q)}, test {len(test_q)}")
        base_model = fit_ranker(
            x_base, y, train_q, base_path, force=force
        )
        aug_model = fit_ranker(
            x_aug, y, train_q, aug_path, force=force
        )
        oof_base[test_q] = predict_queries(base_model, x_base, test_q)
        oof_aug[test_q] = predict_queries(aug_model, x_aug, test_q)
        fold_base = float(reciprocal_ranks(oof_base[test_q]).mean())
        fold_aug = float(reciprocal_ranks(oof_aug[test_q]).mean())
        fold_metrics.append(
            {
                "fold": held,
                "queries": int(len(test_q)),
                "base18_mrr": fold_base,
                "aug103_mrr": fold_aug,
                "delta": fold_aug - fold_base,
            }
        )
        del base_model, aug_model
        gc.collect()

    norm_base = row_minmax(oof_base)
    norm_aug = row_minmax(oof_aug)
    delta = norm_aug - norm_base
    reconstructed = norm_base + delta
    algebra_max_abs = float(np.max(np.abs(reconstructed - norm_aug)))
    assert algebra_max_abs <= 1e-12

    base_mrr = float(reciprocal_ranks(norm_base).mean())
    aug_mrr = float(reciprocal_ranks(reconstructed).mean())
    marginal = aug_mrr - base_mrr
    if abs(base_mrr - EXPECTED_GATE_BASE) > 5e-5:
        raise AssertionError(
            f"base18 gate drift: {base_mrr:.9f}, expected {EXPECTED_GATE_BASE:.9f}"
        )
    if abs(marginal - EXPECTED_GATE_DELTA) > 5e-5:
        raise AssertionError(
            f"footprint gate drift: {marginal:+.9f}, "
            f"expected {EXPECTED_GATE_DELTA:+.9f}"
        )

    gate_report = {
        "hzeval_path": str(hzeval_path),
        "hzeval_md5": file_md5(hzeval_path),
        "queries": int(q),
        "slate_size": N_SLATE,
        "base_feature_count": N_BASE,
        "footprint_feature_count": N_FOOT,
        "augmented_feature_count": N_AUG,
        "base18_mrr": base_mrr,
        "aug103_mrr": aug_mrr,
        "delta": marginal,
        "expected_delta": EXPECTED_GATE_DELTA,
        "paired_residual_algebra_max_abs": algebra_max_abs,
        "folds": fold_metrics,
        "footprint_distribution": footprint_report,
    }
    log(f"gate base {base_mrr:.9f}; aug {aug_mrr:.9f}; delta {marginal:+.9f}")

    all_queries = np.arange(q, dtype=np.int64)
    full_base_path = MODEL_DIR / "full_base18.txt"
    full_aug_path = MODEL_DIR / "full_aug103.txt"
    log("training/loading paired full hzeval models")
    full_base = fit_ranker(
        x_base, y, all_queries, full_base_path, force=force
    )
    full_aug = fit_ranker(
        x_aug, y, all_queries, full_aug_path, force=force
    )
    return gate_report, full_base, full_aug, x_base, y


def popwin(
    dvals: np.ndarray,
    tvals: np.ndarray,
    hi: float,
    lo: float,
    frac: float,
    num_entity: int,
) -> np.ndarray:
    mask = tvals >= (hi - frac * (hi - lo))
    return np.log1p(
        np.bincount(
            dvals[mask].astype(np.int64),
            minlength=num_entity,
        ).astype(np.float64)
    )


class FullTestBase18:
    """Source-faithful, chunked reconstruction of live full-test base18."""

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.df_raw = (
            pd.read_csv(self.dataset_dir / "train.csv")
            .drop_duplicates(subset=["src", "dst", "time"])
            .reset_index(drop=True)
        )
        self.df_raw["src"] = self.df_raw["src"].astype(np.int64)
        self.df_raw["dst"] = self.df_raw["dst"].astype(np.int64)
        self.df_raw["time"] = self.df_raw["time"].astype(float)
        self.test = pd.read_csv(self.dataset_dir / "test.csv")
        self.candidate_columns = [f"c{i}" for i in range(1, N_SLATE + 1)]
        self.candidates = self.test[self.candidate_columns].to_numpy(np.int64)
        self.src = self.test["src"].to_numpy(np.int64)
        self.time = self.test["time"].to_numpy(float)
        assert self.candidates.shape == (EXPECTED_ROWS, N_SLATE)
        self.num_entity = int(
            max(self.df_raw["src"].max(), self.df_raw["dst"].max())
        ) + 1
        self.cutp = float(self.df_raw["time"].max())
        self.tmin = float(self.df_raw["time"].min())

        all_candidates = np.clip(
            self.candidates.ravel(), 0, self.num_entity - 1
        )
        self.freq_cand = np.bincount(
            all_candidates, minlength=self.num_entity
        ).astype(np.float64)
        self.cfreq_log = np.log1p(self.freq_cand)

        self._prepared = False
        self.cache_mat = None
        self.dst_pop_log = None
        self.rpop = None
        self.dst_pop_raw = None
        self.gr = None
        self.seen = None
        self.rps = None
        self.rpl = None
        self.bprs = None
        self.embedding = None
        self.embedding_norm = None

    def prepare(self) -> None:
        if self._prepared:
            return
        log("full-test base18: building history/cooccurrence/collab caches")
        tl.build_history_index(self.df_raw)
        tl.build_cooc(self.df_raw, self.num_entity)
        base_cache = ep.build_base_cache(self.df_raw)
        self.cache_mat = tl.cache_dict_to_matrix(
            base_cache, set(), self.num_entity - 1
        )
        self.dst_pop_log = np.log1p(tl.dst_pop)
        self.rpop = tl.dst_rpop_log.copy()
        self.dst_pop_raw = tl.dst_pop.copy()
        self.gr = np.zeros(self.num_entity)
        for dst, value in self.df_raw.groupby("dst")["time"].max().items():
            self.gr[int(dst)] = (
                (float(value) - self.tmin) / (self.cutp - self.tmin)
            )
        self.seen = (
            np.bincount(
                self.df_raw["dst"].to_numpy(np.int64),
                minlength=self.num_entity,
            )
            > 0
        ).astype(np.float64)
        self.rps = popwin(
            self.df_raw["dst"].to_numpy(),
            self.df_raw["time"].to_numpy(),
            self.cutp,
            self.tmin,
            0.05,
            self.num_entity,
        )
        self.rpl = popwin(
            self.df_raw["dst"].to_numpy(),
            self.df_raw["time"].to_numpy(),
            self.cutp,
            self.tmin,
            0.40,
            self.num_entity,
        )
        output_root = REPO / "outputs"
        self.bprs = [
            np.load(
                output_root
                / ("dataset2-bpr" + (f"-s{seed}" if seed != 42 else ""))
                / "bpr_emb.npy",
                mmap_mode="r",
            )
            for seed in SEEDS
        ]
        self.embedding = ep.load_embedding(
            output_root / "dataset2",
            expected_rows=self.num_entity,
        )
        self.embedding_norm = self.embedding / np.maximum(
            np.linalg.norm(self.embedding, axis=1, keepdims=True),
            1e-8,
        )
        tl.build_sim_cache(self.embedding, np.unique(self.src))
        self._prepared = True

    def transform_chunk(
        self, begin: int, end: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.prepare()
        assert self.cache_mat is not None
        assert self.rpop is not None
        assert self.dst_pop_log is not None
        assert self.dst_pop_raw is not None
        assert self.gr is not None
        assert self.seen is not None
        assert self.rps is not None and self.rpl is not None
        assert self.bprs is not None
        assert self.embedding_norm is not None
        query_src = self.src[begin:end]
        query_candidates = self.candidates[begin:end]
        collab_rows = ep.grouped_collab(
            self.cache_mat,
            query_src.tolist(),
            [row for row in query_candidates],
        )
        output = np.empty(
            (end - begin, N_SLATE, N_BASE), dtype=np.float32
        )
        history = np.zeros((end - begin, N_SLATE), dtype=bool)
        for local, global_row in enumerate(range(begin, end)):
            src = int(self.src[global_row])
            query_time = float(self.time[global_row])
            raw_candidates = self.candidates[global_row]
            candidates = np.clip(
                raw_candidates, 0, self.num_entity - 1
            )
            hist_dst, _ = tl.get_hist_before_time(src, self.cutp + 1)
            f_collab = tl.rownorm(
                collab_rows[local].astype(np.float64)
            )
            f_rpop = tl.rownorm(self.rpop[candidates])
            f_icfm = np.zeros(N_SLATE)
            f_icf3 = np.zeros(N_SLATE)
            if len(hist_dst) > 0:
                hist_vectors = self.embedding_norm[
                    np.clip(hist_dst, 0, self.num_entity - 1)
                ]
                similarity = np.sort(
                    self.embedding_norm[candidates] @ hist_vectors.T,
                    axis=1,
                )
                f_icfm = tl.rownorm(
                    np.maximum(similarity.mean(axis=1), 0.0)
                )
                k = min(3, similarity.shape[1])
                f_icf3 = tl.rownorm(
                    np.maximum(
                        similarity[:, -k:].mean(axis=1), 0.0
                    )
                )
            f_bpr = np.zeros(N_SLATE)
            for embedding in self.bprs:
                f_bpr += tl.rownorm(
                    np.maximum(
                        embedding[candidates]
                        @ embedding[min(src, embedding.shape[0] - 1)],
                        0.0,
                    )
                )
            f_bpr /= len(self.bprs)
            blend = (
                W["collab"] * f_collab
                + W["rpop"] * f_rpop
                + W["icfm"] * f_icfm
                + W["icf3"] * f_icf3
                + W["bpr"] * f_bpr
            )
            hm = np.isin(raw_candidates, hist_dst)
            rs = tl.rownorm(self.rps[candidates])
            rl = tl.rownorm(self.rpl[candidates])
            f_cfreq = tl.rownorm(self.cfreq_log[candidates])
            f_popratio = tl.rownorm(
                self.dst_pop_raw[candidates]
                / (self.freq_cand[candidates] + 1.0)
            )
            output[local] = np.column_stack(
                [
                    f_collab,
                    f_rpop,
                    f_icfm,
                    f_icf3,
                    f_bpr,
                    blend,
                    self.gr[candidates],
                    self.dst_pop_log[candidates],
                    np.full(N_SLATE, len(hist_dst)),
                    hm.astype(float),
                    self.seen[candidates],
                    np.full(
                        N_SLATE,
                        (query_time - self.cutp) / YEAR,
                    ),
                    np.full(N_SLATE, N_SLATE),
                    rs,
                    rl,
                    rs - rl,
                    f_cfreq,
                    f_popratio,
                ]
            ).astype(np.float32)
            history[local] = hm
        return output, history


def build_full_x18(
    builder: FullTestBase18,
    *,
    chunk_queries: int,
    force: bool,
    resume_from: Optional[int] = None,
) -> Tuple[np.memmap, np.memmap]:
    x_path = CACHE_DIR / "full_test_x18.npy"
    h_path = CACHE_DIR / "full_test_history.npy"
    progress_path = CACHE_DIR / "full_test_base18.progress.json"
    x_shape = (EXPECTED_ROWS, N_SLATE, N_BASE)
    h_shape = (EXPECTED_ROWS, N_SLATE)
    x_meta = npy_metadata_path(x_path)
    h_meta = npy_metadata_path(h_path)
    interrupted_pair = (
        x_path.is_file()
        and h_path.is_file()
        and not x_meta.is_file()
        and not h_meta.is_file()
    )
    if resume_from is not None and force:
        raise ValueError("resume_from and force are mutually exclusive")
    if resume_from is not None and not interrupted_pair:
        raise AssertionError(
            "--resume-full-base18-from requires exactly two unmarked cache "
            "arrays from an interrupted run"
        )

    if interrupted_pair and resume_from is not None:
        x_cached = ensure_npy(x_path, x_shape, np.float32)
        h_cached = ensure_npy(h_path, h_shape, np.bool_)
    else:
        x_cached = None if force else completed_npy(
            x_path, x_shape, np.float32
        )
        h_cached = None if force else completed_npy(
            h_path, h_shape, np.bool_
        )
    if x_cached is not None and h_cached is not None:
        if resume_from is None:
            return x_cached, h_cached
        start = int(resume_from)
        if start <= 0 or start >= EXPECTED_ROWS:
            raise ValueError("resume_from must be inside the unfinished row range")
        if start % chunk_queries:
            raise ValueError(
                "resume_from must align to the full-test chunk boundary"
            )
        # A console checkpoint is only emitted after both arrays have been
        # assigned and flushed.  Recompute three completed chunks (first,
        # middle and last) before trusting that checkpoint; this catches a
        # wrong row boundary, stale source artifacts or a mixed cache pair.
        probes = sorted(
            {
                0,
                max(0, (start // 2 // chunk_queries) * chunk_queries),
                start - chunk_queries,
            }
        )
        for probe_begin in probes:
            probe_end = min(probe_begin + chunk_queries, start)
            expected_x, expected_h = builder.transform_chunk(
                probe_begin, probe_end
            )
            if not np.array_equal(
                np.asarray(x_cached[probe_begin:probe_end]), expected_x
            ):
                raise AssertionError(
                    f"interrupted X18 cache fails sentinel at {probe_begin}"
                )
            if not np.array_equal(
                np.asarray(h_cached[probe_begin:probe_end]), expected_h
            ):
                raise AssertionError(
                    f"interrupted history cache fails sentinel at {probe_begin}"
                )
        json_write(
            progress_path,
            {
                "status": "resumed_after_external_timeout",
                "completed_rows": start,
                "chunk_queries": chunk_queries,
                "sentinel_chunk_starts": probes,
                "live_precrf_md5": KNOWN_PRECRF_MD5,
                "x_shape": list(x_shape),
                "h_shape": list(h_shape),
            },
        )
        log(
            "validated interrupted full-test base18 cache at "
            f"{len(probes)} sentinel chunks; resuming from {start}"
        )
        del x_cached, h_cached
        x_map = np.lib.format.open_memmap(x_path, mode="r+")
        h_map = np.lib.format.open_memmap(h_path, mode="r+")
    else:
        start = 0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        x_map = np.lib.format.open_memmap(
            x_path, mode="w+", dtype=np.float32, shape=x_shape
        )
        h_map = np.lib.format.open_memmap(
            h_path, mode="w+", dtype=np.bool_, shape=h_shape
        )
    for begin in range(start, EXPECTED_ROWS, chunk_queries):
        end = min(begin + chunk_queries, EXPECTED_ROWS)
        part_x, part_h = builder.transform_chunk(begin, end)
        x_map[begin:end] = part_x
        h_map[begin:end] = part_h
        x_map.flush()
        h_map.flush()
        json_write(
            progress_path,
            {
                "status": "building",
                "completed_rows": end,
                "chunk_queries": chunk_queries,
                "live_precrf_md5": KNOWN_PRECRF_MD5,
                "x_shape": list(x_shape),
                "h_shape": list(h_shape),
            },
        )
        log(f"full-test base18 {end}/{EXPECTED_ROWS}")
    del x_map, h_map
    provenance = {
        "kind": "full_test_live_base18",
        "live_precrf_md5": KNOWN_PRECRF_MD5,
        "rows": EXPECTED_ROWS,
    }
    mark_npy_complete(x_path, provenance)
    mark_npy_complete(h_path, provenance)
    json_write(
        progress_path,
        {
            "status": "complete",
            "completed_rows": EXPECTED_ROWS,
            "chunk_queries": chunk_queries,
            "live_precrf_md5": KNOWN_PRECRF_MD5,
            "x_shape": list(x_shape),
            "h_shape": list(h_shape),
        },
    )
    x_cached = completed_npy(x_path, x_shape, np.float32)
    h_cached = completed_npy(h_path, h_shape, np.bool_)
    assert x_cached is not None and h_cached is not None
    return x_cached, h_cached


class StreamingStats:
    def __init__(self, names: Sequence[str]) -> None:
        self.names = tuple(names)
        n = len(self.names)
        self.count = 0
        self.total = np.zeros(n, np.float64)
        self.total_sq = np.zeros(n, np.float64)
        self.minimum = np.full(n, np.inf)
        self.maximum = np.full(n, -np.inf)

    def add(self, matrix: np.ndarray) -> None:
        values = np.asarray(matrix).reshape(-1, len(self.names))
        if not np.isfinite(values).all():
            raise FloatingPointError("non-finite values in StreamingStats")
        values = values.astype(np.float64, copy=False)
        self.count += len(values)
        self.total += values.sum(axis=0)
        self.total_sq += np.square(values).sum(axis=0)
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))

    def report(self) -> dict:
        if self.count == 0:
            raise RuntimeError("empty StreamingStats")
        mean = self.total / self.count
        variance = np.maximum(
            self.total_sq / self.count - mean * mean, 0.0
        )
        return {
            "candidate_cells": int(self.count),
            "columns": [
                {
                    "index": i,
                    "name": name,
                    "mean": float(mean[i]),
                    "std": float(np.sqrt(variance[i])),
                    "min": float(self.minimum[i]),
                    "max": float(self.maximum[i]),
                }
                for i, name in enumerate(self.names)
            ],
        }


def predict_full_residual(
    base_model: lgb.Booster,
    aug_model: lgb.Booster,
    x18: np.ndarray,
    history: np.ndarray,
    footprint_generator: ff.FootprintFeatureGenerator,
    *,
    chunk_queries: int,
    force: bool,
) -> Tuple[np.memmap, dict]:
    output_path = CACHE_DIR / "full_test_paired_delta.npy"
    expected = (EXPECTED_ROWS, N_SLATE)
    cached = None if force else ensure_npy(output_path, expected, np.float32)
    if cached is not None:
        report_path = CACHE_DIR / "full_test_residual_report.json"
        if not report_path.is_file():
            raise AssertionError(
                "paired residual cache exists without its distribution report"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("md5") != file_md5(output_path):
            raise AssertionError("paired residual cache/report MD5 mismatch")
        report["cache_reused"] = True
        return cached, report

    mapped = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float32,
        shape=expected,
    )
    footprint_stats = StreamingStats(ff.FEATURE_NAMES)
    delta_stats = StreamingStats(("paired_delta",))
    history_cells = 0
    nonzero_history_before_zero = 0
    top1_changed = 0
    for begin in range(0, EXPECTED_ROWS, chunk_queries):
        end = min(begin + chunk_queries, EXPECTED_ROWS)
        base_x = np.asarray(x18[begin:end], dtype=np.float32)
        footprint = footprint_generator.transform_chunk(
            np.arange(begin, end, dtype=np.int64)
        )
        assert footprint.shape == (
            end - begin,
            N_SLATE,
            N_FOOT,
        )
        footprint_stats.add(footprint)
        aug_x = np.concatenate(
            [base_x, footprint], axis=2
        ).reshape(-1, N_AUG)
        assert np.array_equal(
            aug_x[:, :N_BASE],
            base_x.reshape(-1, N_BASE),
        )
        base_score = base_model.predict(
            base_x.reshape(-1, N_BASE)
        ).reshape(-1, N_SLATE)
        aug_score = aug_model.predict(aug_x).reshape(-1, N_SLATE)
        norm_base = row_minmax(base_score)
        norm_aug = row_minmax(aug_score)
        delta = norm_aug - norm_base
        hm = np.asarray(history[begin:end], dtype=bool)
        history_cells += int(hm.sum())
        nonzero_history_before_zero += int(
            (np.abs(delta[hm]) > 0).sum()
        )
        delta[hm] = 0.0
        assert np.all(delta[hm] == 0.0)
        top1_changed += int(
            (
                np.argmax(norm_base, axis=1)
                != np.argmax(norm_base + delta, axis=1)
            ).sum()
        )
        delta_stats.add(delta[:, :, None])
        mapped[begin:end] = delta.astype(np.float32)
        mapped.flush()
        del aug_x, footprint
        gc.collect()
        log(f"full-test paired residual {end}/{EXPECTED_ROWS}")
    del mapped
    cached = ensure_npy(output_path, expected, np.float32)
    assert cached is not None
    report = {
        "path": str(output_path),
        "md5": file_md5(output_path),
        "cache_reused": False,
        "history_candidate_cells": history_cells,
        "history_delta_nonzero_before_forced_zero": nonzero_history_before_zero,
        "paired_model_top1_changes": top1_changed,
        "delta_distribution": delta_stats.report(),
        "footprint_distribution": footprint_stats.report(),
    }
    json_write(CACHE_DIR / "full_test_residual_report.json", report)
    return cached, report


def load_precrf_base() -> np.ndarray:
    if file_md5(LIVE_PRECRF) != KNOWN_PRECRF_MD5:
        raise AssertionError(
            f"live pre-CRF base drifted: {LIVE_PRECRF}"
        )
    scores = pd.read_csv(LIVE_PRECRF, header=None).to_numpy(np.float64)
    assert scores.shape == (EXPECTED_ROWS, N_SLATE)
    assert np.isfinite(scores).all()
    return scores


def write_precrf_b(
    base: np.ndarray,
    delta: np.ndarray,
    history: np.ndarray,
) -> dict:
    assert base.shape == delta.shape == history.shape
    assert np.all(np.asarray(delta)[np.asarray(history)] == 0.0)
    adjusted = base + np.asarray(delta, dtype=np.float64)
    # The live ranker has already hard-demoted history candidates.  The paired
    # residual is zero there; reject (rather than silently repair) any row
    # where lowering non-history candidates would make a history item top-1.
    argmax = np.argmax(adjusted, axis=1)
    violations = int(
        np.asarray(history)[np.arange(len(adjusted)), argmax].sum()
    )
    if violations:
        raise AssertionError(
            f"paired residual revives history candidates in {violations} rows"
        )
    np.savetxt(PRECRF_B, adjusted, delimiter=",", fmt="%.6f")
    reread = pd.read_csv(PRECRF_B, header=None).to_numpy(np.float64)
    assert reread.shape == adjusted.shape
    assert np.max(np.abs(reread - adjusted)) <= 5.000001e-7
    return {
        "path": str(PRECRF_B),
        "rows": EXPECTED_ROWS,
        "columns": N_SLATE,
        "md5": file_md5(PRECRF_B),
        "history_top1_violations": violations,
        "score_min": float(reread.min()),
        "score_max": float(reread.max()),
    }


def crf_dataset2(
    precrf_scores: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> Tuple[np.ndarray, dict]:
    """Exact tau=.20/B70/triple/zrxst/no-pair postprocessor."""

    candidate_columns = [f"c{i}" for i in range(1, N_SLATE + 1)]
    candidates = test[candidate_columns].to_numpy(np.int64)
    src = test["src"].to_numpy(np.int64)
    query_time = test["time"].to_numpy(float)
    n_ids = int(max(train["dst"].max(), candidates.max())) + 1
    warm = np.zeros(n_ids, bool)
    warm[train["dst"].to_numpy(np.int64)] = True
    candidate_sets = [frozenset(row) for row in candidates]
    warm_mask = warm[candidates]
    candidate_index = [
        {int(value): col for col, value in enumerate(row)}
        for row in candidates
    ]

    scores0 = np.asarray(precrf_scores, dtype=np.float64)
    assert scores0.shape == (EXPECTED_ROWS, N_SLATE)
    normalized = crf_minmax(scores0)
    forward, backward = cp.build_band(
        candidates,
        src,
        query_time,
        warm_mask,
        candidate_index,
        1,
    )
    q = cp.equality_crf(
        normalized,
        forward,
        backward,
        0.20,
        70.0,
        1.0,
        None,
    )
    scores = crf_minmax(q)
    trip = cp.find_triples(
        candidates, candidate_sets, src, query_time, warm
    )
    for row, value in trip.items():
        col = int(np.where(candidates[row] == value)[0][0])
        scores[row] = cp.promote(scores[row].copy(), col)

    # --no-pair: no pair promotion.  Pair labels remain active for
    # --zr-exclude exactly as in src/crf_promote.py.
    pair_labels = cp.find_pair_labels(
        candidate_sets, src, query_time, warm, trip
    )
    demotions = cp.zero_repeat_demotions(
        candidates,
        candidate_sets,
        src,
        query_time,
        trip,
        pair_labels,
        True,
    )
    st_demotions = cp.same_time_structural_demotions(
        candidates,
        candidate_sets,
        query_time,
        warm,
        dict(trip),
    )
    for row, columns in st_demotions.items():
        for col, level in columns.items():
            demotions.setdefault(row, {}).setdefault(col, level)
    scores = cp.apply_demotions(scores, demotions)
    assert scores.shape == (EXPECTED_ROWS, N_SLATE)
    assert np.isfinite(scores).all()
    assert scores.min() >= 0.0 and scores.max() <= 1.0
    assert np.all(scores.std(axis=1) > 0)
    for row, value in trip.items():
        expected = int(np.where(candidates[row] == value)[0][0])
        assert int(np.argmax(scores[row])) == expected

    coupled_edges = int(
        (
            (query_time[1:] == query_time[:-1])
            & (src[1:] != src[:-1])
        ).sum()
    )
    report = {
        "tau": 0.20,
        "B": 70.0,
        "W": 1,
        "p": 1.0,
        "eta": 0.0,
        "triple_enabled": True,
        "pair_promotion_enabled": False,
        "zr_exclude": True,
        "demote_dups": True,
        "st_exclude": True,
        "triple_rows": int(len(trip)),
        "pair_label_rows_for_zr": int(len(pair_labels)),
        "rows_with_demotions": int(len(demotions)),
        "coupled_rows_with_edges": int(coupled_edges),
    }
    return scores, report


def formatted_csv_chunks(
    scores: np.ndarray,
    rows_per_chunk: int = 256,
) -> Iterator[bytes]:
    for begin in range(0, len(scores), rows_per_chunk):
        end = min(begin + rows_per_chunk, len(scores))
        lines = [
            ",".join(f"{value:.6f}" for value in row)
            for row in scores[begin:end]
        ]
        yield ("\n".join(lines) + "\n").encode("ascii")


def write_ds2_zip(scores: np.ndarray, output: Path) -> dict:
    digest = hashlib.md5()
    line_count = 0
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        with archive.open("dataset2.csv", "w", force_zip64=True) as member:
            for block in formatted_csv_chunks(scores):
                digest.update(block)
                line_count += block.count(b"\n")
                member.write(block)
    assert line_count == EXPECTED_ROWS
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["dataset2.csv"]
        info = archive.getinfo("dataset2.csv")
        assert info.file_size > 0
    return {
        "path": str(output),
        "rows": line_count,
        "columns": N_SLATE,
        "zip_entries": ["dataset2.csv"],
        "zip_md5": file_md5(output),
        "dataset2_csv_md5": digest.hexdigest(),
        "dataset2_csv_bytes": int(info.file_size),
    }


def ensure_pack_a() -> dict:
    if file_md5(KNOWN_A) != KNOWN_A_ZIP_MD5:
        raise AssertionError("known pack A archive drifted")
    with zipfile.ZipFile(KNOWN_A) as archive:
        assert archive.namelist() == ["dataset2.csv"]
        payload = archive.read("dataset2.csv")
    assert bytes_md5(payload) == KNOWN_A_INNER_MD5
    if not PACK_A.is_file():
        shutil.copy2(KNOWN_A, PACK_A)
    if file_md5(PACK_A) != KNOWN_A_ZIP_MD5:
        raise AssertionError("reference pack A is not byte-identical")
    return {
        "path": str(PACK_A),
        "rows": int(payload.count(b"\n")),
        "columns": N_SLATE,
        "zip_entries": ["dataset2.csv"],
        "zip_md5": KNOWN_A_ZIP_MD5,
        "dataset2_csv_md5": KNOWN_A_INNER_MD5,
        "dataset2_csv_bytes": len(payload),
        "byte_identical_to_known_live": True,
    }


def verify_crf_a_exact(
    base_precrf: np.ndarray,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    scores, crf_report = crf_dataset2(
        base_precrf, train, test
    )
    digest = hashlib.md5()
    count = 0
    for block in formatted_csv_chunks(scores):
        digest.update(block)
        count += block.count(b"\n")
    assert count == EXPECTED_ROWS
    if digest.hexdigest() != KNOWN_A_INNER_MD5:
        raise AssertionError(
            "pure-function CRF A does not reproduce the known live pack"
        )
    return {
        "dataset2_csv_md5": digest.hexdigest(),
        "exact_known_a": True,
        "crf": crf_report,
    }


def run_full(
    full_base_model: lgb.Booster,
    full_aug_model: lgb.Booster,
    *,
    chunk_queries: int,
    force: bool,
    resume_full_base18_from: Optional[int] = None,
) -> dict:
    ensure_pack_a()
    builder = FullTestBase18(DATASET_DIR)
    x18, history = build_full_x18(
        builder,
        chunk_queries=chunk_queries,
        force=force,
        resume_from=resume_full_base18_from,
    )
    generator, physical_rows = ff.build_generator(
        "full-test", DATASET_DIR
    )
    assert np.array_equal(
        physical_rows,
        np.arange(EXPECTED_ROWS, dtype=np.int64),
    )
    assert np.array_equal(generator.candidates, builder.candidates.astype(np.int32))
    assert np.array_equal(generator.src, builder.src)
    assert np.array_equal(generator.time, builder.time)
    delta, residual_report = predict_full_residual(
        full_base_model,
        full_aug_model,
        x18,
        history,
        generator,
        chunk_queries=chunk_queries,
        force=force,
    )

    base_precrf = load_precrf_base()
    train = builder.df_raw
    test = builder.test
    a_crf_verification = verify_crf_a_exact(
        base_precrf, train, test
    )
    precrf_report = write_precrf_b(
        base_precrf, delta, history
    )
    # Read the six-decimal boundary back exactly as src/crf_promote.py does.
    b_precrf_quantized = pd.read_csv(
        PRECRF_B, header=None
    ).to_numpy(np.float64)
    b_scores, b_crf_report = crf_dataset2(
        b_precrf_quantized, train, test
    )
    pack_b_report = write_ds2_zip(b_scores, PACK_B)
    pack_a_report = ensure_pack_a()
    assert pack_b_report["dataset2_csv_md5"] != KNOWN_A_INNER_MD5
    top1_a = None
    with zipfile.ZipFile(PACK_A) as archive:
        with archive.open("dataset2.csv") as handle:
            a_final = pd.read_csv(handle, header=None).to_numpy(np.float64)
    top1_a = np.argmax(a_final, axis=1)
    top1_b = np.argmax(b_scores, axis=1)
    return {
        "implementation": (
            "85-column footprint block distilled to a paired per-candidate "
            "residual after the common live label-basket base and before the "
            "common CRF; not a native three-pass raw-feature retrain"
        ),
        "pack_A": pack_a_report,
        "pack_B": pack_b_report,
        "precrf_B": precrf_report,
        "residual": residual_report,
        "A_crf_exact_reproduction": a_crf_verification,
        "B_crf": b_crf_report,
        "final_top1_changes_B_vs_A": int((top1_a != top1_b).sum()),
        "full_test_rows": EXPECTED_ROWS,
        "slate_size": N_SLATE,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    global ADOPT_UNMARKED_MODELS, MODEL_SIGNATURE
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--gate-only",
        action="store_true",
        help="reproduce the exact 2-fold hzeval delta and train paired full models",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="run gate, full-test residual, exact CRF and ds2-only packaging",
    )
    parser.add_argument(
        "--hzeval",
        type=Path,
        default=DEFAULT_HZEVAL,
    )
    parser.add_argument(
        "--feature-chunk",
        type=int,
        default=256,
        help="query rows per footprint generation chunk",
    )
    parser.add_argument(
        "--full-chunk",
        type=int,
        default=512,
        help="query rows per full-test X18/prediction chunk",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild cached arrays and retrain model text files",
    )
    parser.add_argument(
        "--adopt-unmarked-cache",
        action="store_true",
        help=(
            "recover outputs from an interrupted pre-metadata run; validates "
            "array shape and model tree/feature counts, then the normal gate "
            "assertions must still pass"
        ),
    )
    parser.add_argument(
        "--resume-full-base18-from",
        type=int,
        default=None,
        help=(
            "resume the two unmarked full-test base18 arrays at a logged, "
            "flushed row boundary; three completed chunks are recomputed "
            "bit-for-bit before continuation"
        ),
    )
    args = parser.parse_args(argv)
    if not args.hzeval.is_file():
        parser.error(f"hzeval not found: {args.hzeval}")
    if args.feature_chunk <= 0 or args.full_chunk <= 0:
        parser.error("chunk sizes must be positive")
    if args.force and args.adopt_unmarked_cache:
        parser.error("--force and --adopt-unmarked-cache are mutually exclusive")
    if args.resume_full_base18_from is not None and not args.full:
        parser.error("--resume-full-base18-from requires --full")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SIGNATURE = {
        "training_contract_version": 1,
        "hzeval_md5": file_md5(args.hzeval),
        "footprint_feature_md5": file_md5(HERE / "footprint_feature.py"),
        "base_features": N_BASE,
        "footprint_features": N_FOOT,
        "augmented_features": N_AUG,
        "slate_size": N_SLATE,
        "parameters": PARAMS,
        "source_fold_rng_seed": 7,
    }
    ADOPT_UNMARKED_MODELS = bool(args.adopt_unmarked_cache)
    gate, full_base, full_aug, _, _ = run_gate(
        args.hzeval,
        chunk_size=args.feature_chunk,
        force=args.force,
    )
    metrics = load_metrics()
    metrics.update(
        {
            "probe_version": 1,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "gate": gate,
            "model_parameters": PARAMS,
            "feature_names": list(ff.FEATURE_NAMES),
            "discipline": {
                "dataset2_only": True,
                "contains_dataset1": False,
                "live_source_files_modified": False,
                "paired_residual_beta": 1.0,
                "history_delta_forced_zero": True,
            },
        }
    )
    json_write(METRICS, metrics)
    if args.gate_only:
        log(f"gate complete; metrics: {METRICS}")
        return 0

    metrics["full"] = run_full(
        full_base,
        full_aug,
        chunk_queries=args.full_chunk,
        force=args.force,
        resume_full_base18_from=args.resume_full_base18_from,
    )
    json_write(METRICS, metrics)
    log(f"full probe complete; pack A: {PACK_A}")
    log(f"full probe complete; pack B: {PACK_B}")
    log(f"metrics: {METRICS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
