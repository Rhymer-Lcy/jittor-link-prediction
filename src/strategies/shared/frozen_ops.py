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
    """Divide each row by its maximum, mapping every row into [0, 1].

    Rows whose maximum is strictly positive are divided by it exactly as before,
    bit for bit. That is every row the frozen A-board chain ever produced, so the
    accepted member is unchanged.

    A row whose maximum is NOT strictly positive has no meaningful "divide by the
    maximum" normalisation. The previous guard, ``np.maximum(row_max, 1e-12)``,
    was written for an all-ZERO row; applied to an all-NEGATIVE row it divides by
    1e-12 and AMPLIFIES the row by twelve orders of magnitude. A canonical
    dataset1 run whose LambdaRank model scored 33 of 61,051 queries entirely
    negative produced member values down to -4.7e24, far outside the [0, 1] the
    submission format requires.

    Such a row is instead shifted by its own minimum and then divided by the
    shifted maximum. The shift is a per-row constant and the divisor is positive,
    so the candidate ORDER is preserved exactly, which is all MRR depends on. A
    constant non-positive row shifts to all zeros and is mapped to 0.5, which is
    in range, order-neutral and free of division by zero.

    Scope: this repair belongs to ``row_max_normalise`` alone. It must NOT be
    applied to ``pipeline_common.rownorm``, whose degenerate input is the all-ZERO
    row and which correctly returns it unchanged; mapping those to 0.5 would
    rewrite 117 measured dataset2 ``f_collab`` feature rows and change frozen
    dataset2 semantics.
    """
    row_max = scores.max(axis=1, keepdims=True)
    defined = row_max > 1e-12
    if defined.all():                       # the frozen path, untouched
        return scores / np.maximum(row_max, 1e-12)

    shifted = scores - scores.min(axis=1, keepdims=True)
    shifted_max = shifted.max(axis=1, keepdims=True)
    repaired = np.where(shifted_max > 1e-12,
                        shifted / np.maximum(shifted_max, 1e-12), 0.5)
    return np.where(defined, scores / np.maximum(row_max, 1e-12), repaired)


def stable_rank_order(scores: np.ndarray) -> np.ndarray:
    """Descending score order per row, ties broken by ascending column index."""
    return np.argsort(-scores, axis=1, kind="stable")


def dense_group_ids(*columns: np.ndarray) -> tuple[np.ndarray, int]:
    """Dense group id per row for a tuple of integer columns.

    Group ids are assigned in lexicographic order of the key tuple, which the
    dataset2 cross-time decoder relies on: it resolves a keeper tie by taking the
    first eligible cluster of a group, and lexicographic ``(group, time)``
    ordering makes that the smallest-timestamp cluster.

    Returns ``(group_id_per_row, number_of_groups)``.
    """
    keys = np.stack([np.asarray(c, np.int64) for c in columns], axis=1)
    order = np.lexsort(tuple(keys[:, i] for i in range(keys.shape[1] - 1, -1, -1)))
    sorted_keys = keys[order]
    boundary = np.ones(len(order), bool)
    boundary[1:] = (sorted_keys[1:] != sorted_keys[:-1]).any(axis=1)
    dense = np.empty(len(order), np.int64)
    dense[order] = np.cumsum(boundary) - 1
    return dense, int(boundary.sum())


def csv_record_spans(payload: bytes, expected_rows: int | None = None) -> np.ndarray:
    """Start/end byte offsets of each CSV record, the terminating newline excluded.

    Used by the byte-preserving dataset2 decoder: rewriting only the affected
    score tokens and copying every other byte is what makes the rebuilt member
    byte-identical to the online-scored one, since no score is ever
    re-serialised from a float.
    """
    newlines = np.flatnonzero(np.frombuffer(payload, np.uint8) == 0x0A)
    if not len(payload) or payload[-1] != 0x0A:
        raise ValueError("payload does not end with a newline")
    if expected_rows is not None and newlines.size != expected_rows:
        raise ValueError(f"{newlines.size} records for {expected_rows} expected rows")
    starts = np.empty(newlines.size, np.int64)
    starts[0] = 0
    starts[1:] = newlines[:-1] + 1
    return np.stack([starts, newlines], axis=1)
