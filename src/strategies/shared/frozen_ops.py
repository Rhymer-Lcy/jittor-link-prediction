# -*- coding: utf-8 -*-
"""Primitives shared by the frozen dataset1 score postprocessors.

Every function here is part of a shipped, online-validated deployment rule. The
numeric behaviour is frozen: changing any of it invalidates the accepted
submission hashes recorded in ``configs/production.json``.

Serialisation note
------------------
The accepted ``dataset1.csv`` member uses ``%.6f`` formatting and CRLF line
endings, which is what ``pandas.DataFrame.to_csv`` produced on the Windows host
that built it. :func:`write_score_matrix` pins the line terminator explicitly so
the accepted bytes are reproducible on any platform rather than only on Windows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE_COLUMNS = [f"c{i}" for i in range(1, 101)]
SCORE_FORMAT = "%.6f"
LINE_TERMINATOR = "\r\n"


def sha256_file(path: str | Path) -> str:
    """Stream a file through SHA256 without loading it whole."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_score_matrix(path: str | Path) -> np.ndarray:
    """Read a headerless submission CSV into a float64 (rows, 100) matrix."""
    scores = pd.read_csv(path, header=None).to_numpy(np.float64)
    if scores.ndim != 2 or scores.shape[1] != 100:
        raise ValueError(f"expected a (rows, 100) score matrix, got {scores.shape}")
    if not np.isfinite(scores).all():
        raise FloatingPointError(f"non-finite score in {path}")
    return scores


def write_score_matrix(scores: np.ndarray, path: str | Path) -> str:
    """Write a headerless submission CSV in the accepted format; return its SHA256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).to_csv(
        path,
        index=False,
        header=False,
        float_format=SCORE_FORMAT,
        lineterminator=LINE_TERMINATOR,
    )
    return sha256_file(path)


def entity_upper_bound(*id_arrays: np.ndarray) -> int:
    """Smallest safe base for packing (a, b) node pairs into a single int64 key."""
    return int(max(int(np.asarray(a).max()) for a in id_arrays)) + 1


def pair_keys(left: np.ndarray, right: np.ndarray, num_entity: int) -> np.ndarray:
    """Pack a directed node pair into one int64 key."""
    return np.asarray(left, np.int64) * np.int64(num_entity) + np.asarray(right, np.int64)


def membership(sorted_keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Boolean membership of ``query`` in a sorted unique key array."""
    if len(sorted_keys) == 0:
        return np.zeros(len(query), dtype=bool)
    pos = np.searchsorted(sorted_keys, query)
    safe = np.minimum(pos, len(sorted_keys) - 1)
    return (pos < len(sorted_keys)) & (sorted_keys[safe] == query)


def history_mask(train, sources: np.ndarray, candidates: np.ndarray,
                 num_entity: int) -> np.ndarray:
    """``True`` where a candidate is a historical (training) partner of its query source.

    History is directed: only ``train.src -> train.dst`` edges count, matching the
    definition frozen in both shipped dataset1 postprocessors.
    """
    rows, cols = candidates.shape
    hist = np.unique(pair_keys(train["src"].to_numpy(np.int64),
                               train["dst"].to_numpy(np.int64), num_entity))
    cell_src = np.repeat(np.asarray(sources, np.int64), cols)
    return membership(hist, pair_keys(cell_src, candidates.ravel(), num_entity)).reshape(rows, cols)


def promoted_value(row: np.ndarray) -> float:
    """The frozen strict-top-1 promotion value for one score row.

    ``row.max() + max(ptp(row), 1.0) + 1.0`` -- evaluated on the row as it stands
    *before* the assignment. On a max-normalised row (every accepted dataset1 row
    peaks at exactly 1.0) this is exactly 3.0, so the subsequent row
    normalisation divides the row by exactly 3.
    """
    return float(row.max()) + max(float(np.ptp(row)), 1.0) + 1.0


def row_max_normalise(scores: np.ndarray) -> np.ndarray:
    """Divide each row by its maximum, guarding against a zero row."""
    return scores / np.maximum(scores.max(axis=1, keepdims=True), 1e-12)


def stable_rank_order(scores: np.ndarray) -> np.ndarray:
    """Descending score order per row, ties broken by ascending column index."""
    return np.argsort(-scores, axis=1, kind="stable")
