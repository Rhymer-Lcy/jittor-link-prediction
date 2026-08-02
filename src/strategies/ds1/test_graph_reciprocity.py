# -*- coding: utf-8 -*-
"""dataset1 rank-2 test-graph reciprocity (strategy id ``test_graph_reciprocity``).

Status
------
Active. It is the SECOND of the two frozen dataset1 postprocessors and reads the
matrix produced by :mod:`src.strategies.ds1.source_slate_recurrence`, so the two
are order-sensitive; see :mod:`src.strategies.registry`.

Mechanism
---------
The physical test batch defines a directed exposure graph: ``A(u, c) = 1`` iff
candidate ``c`` appears in some test slate whose source is ``u``. A *reverse*
edge -- the query source itself appearing as a candidate in a slate belonging to
the runner-up candidate -- is mutual-exposure evidence that the generator treats
the two nodes as related. When the incumbent top-1 has no such reverse edge and
the runner-up does, and neither is a known partner, the runner-up is the better
guess. Like the recurrence rule this is truth-free, batch-native, and never uses
candidate column position.

Frozen deployment rule (do not tune)
------------------------------------
With ``c1`` = stable score rank 1 and ``c2`` = stable score rank 2 of the input
matrix, act iff all four conditions hold:

1. ``c1`` is not a historical partner of the query source;
2. ``c2`` is not a historical partner of the query source;
3. ``A(c2, source) = 1``;
4. ``A(c1, source) = 0``.

Then promote ``c2`` to strict top-1 via the frozen score transform and row
max-normalise.

Prohibited variants (deliberately frozen out): inspecting ranks below 2,
reverse-edge *counts* or thresholds instead of existence, widening the
eligibility population, adding recurrence / source-peak / source-role
conditions, or combining with other candidate mechanisms without revalidating.

Scope
-----
Acts on 1,365 of the 61,051 dataset1 test rows (2.24%), each action contributing
exactly one strict pair inversion; every other row passes through unchanged. The
rule was validated in isolation before being adopted, and its parameters are
frozen rather than fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..shared.frozen_ops import (
    CANDIDATE_COLUMNS,
    entity_upper_bound,
    history_mask,
    membership,
    pair_keys,
    promoted_value,
    row_max_normalise,
    stable_rank_order,
)

STRATEGY_ID = "test_graph_reciprocity"
DATASET = "dataset1"

#: Row/action anchors of the accepted deployment, asserted by the tracked tests.
EXPECTED_ROWS = 61051
EXPECTED_ACTIONS = 1365
EXPECTED_GRAPH_EDGES = 5918104
EXPECTED_REVERSE_CELLS = 105290


def exposure_graph(sources: np.ndarray, candidates: np.ndarray, num_entity: int) -> np.ndarray:
    """Sorted unique directed edges ``source -> candidate`` of the physical test batch."""
    cell_src = np.repeat(np.asarray(sources, np.int64), candidates.shape[1])
    return np.unique(pair_keys(cell_src, candidates.ravel(), num_entity))


def reverse_edge_mask(sources: np.ndarray, candidates: np.ndarray, edges: np.ndarray,
                      num_entity: int) -> np.ndarray:
    """``True`` where the reverse edge ``candidate -> query source`` exists."""
    rows, cols = candidates.shape
    cell_src = np.repeat(np.asarray(sources, np.int64), cols)
    return membership(edges, pair_keys(candidates.ravel(), cell_src, num_entity)).reshape(rows, cols)


def apply(scores: np.ndarray, test: pd.DataFrame, train: pd.DataFrame
          ) -> tuple[np.ndarray, dict]:
    """Apply the frozen rule to a dataset1 score matrix.

    Returns ``(treatment, info)``; ``treatment`` is a new max-normalised matrix.
    """
    candidates = test[CANDIDATE_COLUMNS].to_numpy(np.int64)
    if scores.shape != candidates.shape:
        raise ValueError(f"score shape {scores.shape} != candidate shape {candidates.shape}")

    sources = test["src"].to_numpy(np.int64)
    num_entity = entity_upper_bound(sources, candidates, train["src"], train["dst"])
    edges = exposure_graph(sources, candidates, num_entity)
    has_reverse = reverse_edge_mask(sources, candidates, edges, num_entity)
    is_history = history_mask(train, sources, candidates, num_entity)

    order = stable_rank_order(scores)
    c1, c2 = order[:, 0], order[:, 1]
    rows = np.arange(scores.shape[0])
    acted = (
        ~is_history[rows, c1]                       # 1. incumbent non-historical
        & ~is_history[rows, c2]                     # 2. runner-up non-historical
        & has_reverse[rows, c2]                     # 3. runner-up is reciprocated
        & ~has_reverse[rows, c1]                    # 4. incumbent is not
    )

    treatment = scores.copy()
    for q in np.flatnonzero(acted):
        treatment[q, c2[q]] = promoted_value(treatment[q])
    treatment = row_max_normalise(treatment)

    control_top = scores.argmax(axis=1)
    treatment_top = treatment.argmax(axis=1)
    changed = treatment_top != control_top
    if not np.array_equal(changed, acted):
        raise AssertionError("top-1 change mask does not equal the action mask")
    if not np.all(treatment_top[acted] == c2[acted]):
        raise AssertionError("a promoted runner-up did not become strict top-1")

    return treatment, {
        "strategy": STRATEGY_ID,
        "rows": int(scores.shape[0]),
        "actions": int(acted.sum()),
        "top1_changes": int(changed.sum()),
        "action_coverage": float(acted.mean()),
        "directed_graph_edges": int(edges.size),
        "reverse_positive_cells": int(has_reverse.sum()),
    }
