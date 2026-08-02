# -*- coding: utf-8 -*-
"""dataset1 source-slate recurrence promotion (strategy id ``source_slate_recurrence``).

Status
------
Active. It is the FIRST of the two frozen dataset1 postprocessors and runs
underneath :mod:`src.strategies.ds1.test_graph_reciprocity`, which reads the
matrix this module produces. The order is load-bearing; see
:mod:`src.strategies.registry`.

Mechanism
---------
The inference batch is itself evidence. Each dataset1 test row exposes a slate of
100 candidates for a source, and a candidate that the generator keeps re-drawing
into *other* slates of the same source is more likely to be that source's real
future partner than the ranker's top-1 when the ranker is guessing at a new
partner. The signal is truth-free and batch-native: it uses only the candidate
identities in ``test.csv``, never a label, never a score, and never the candidate
column position.

Frozen deployment rule (do not tune)
------------------------------------
For each physical test row, with ``recurrence(c)`` = the number of *other* test
slates of the same source that contain candidate ``c``:

1. row gate -- the control top-1 must NOT be a historical partner of the source;
2. eligibility -- consider only non-historical candidates;
3. winner -- the eligible candidate with the strictly unique maximum recurrence
   (ties disqualify the row);
4. support -- act only when that winner appears in >= 2 other slates
   (>= 3 slates in total for that source);
5. action -- promote the winner to strict top-1 via the frozen score transform,
   then row max-normalise.

Prohibited variants (measured or explicitly frozen out): tuning the recurrence
threshold, relaxing the uniqueness requirement, changing the row gate or the
history definition, altering the score transform, using candidate column
position, retraining, blending or sweeping.

Scope
-----
Acts on 3,653 of the 61,051 dataset1 test rows (5.98%); every other row passes
through unchanged. The rule was validated in isolation before being adopted, and
its parameters are frozen rather than fitted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..shared.frozen_ops import (
    CANDIDATE_COLUMNS,
    entity_upper_bound,
    history_mask,
    pair_keys,
    promoted_value,
    row_max_normalise,
)

STRATEGY_ID = "source_slate_recurrence"
DATASET = "dataset1"

#: A winner must appear in at least this many OTHER slates of the same source.
MINIMUM_OTHER_SLATES = 2

#: Row/action anchors of the accepted deployment, asserted by the tracked tests.
EXPECTED_ROWS = 61051
EXPECTED_ACTIONS = 3653


def slate_recurrence(sources: np.ndarray, candidates: np.ndarray, num_entity: int) -> np.ndarray:
    """Count, per cell, the OTHER same-source slates containing that candidate.

    Purely an identity count over the physical test matrix: no times, no scores,
    no column positions.
    """
    rows, cols = candidates.shape
    cell_src = np.repeat(np.asarray(sources, np.int64), cols)
    keys = pair_keys(cell_src, candidates.ravel(), num_entity)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    return (counts[inverse] - 1).reshape(rows, cols).astype(np.int64)


def action_mask(scores: np.ndarray, recurrence: np.ndarray, is_history: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """Resolve the frozen rule into ``(acted, winner_column)``.

    ``winner_column`` is -1 on rows that do not act.
    """
    rows = scores.shape[0]
    control_top = scores.argmax(axis=1)
    acted = np.zeros(rows, dtype=bool)
    winner = np.full(rows, -1, dtype=np.int64)

    for q in range(rows):
        if is_history[q, control_top[q]]:
            continue                                    # 1. row gate
        eligible = np.flatnonzero(~is_history[q])       # 2. eligibility
        if eligible.size == 0:
            continue
        values = recurrence[q, eligible]
        order = np.argsort(values, kind="stable")
        best = int(order[-1])
        best_value = int(values[best])
        second_value = int(values[order[-2]]) if order.size > 1 else -1
        if best_value < MINIMUM_OTHER_SLATES:           # 4. support
            continue
        if order.size > 1 and best_value == second_value:
            continue                                    # 3. uniqueness
        col = int(eligible[best])
        if col == int(control_top[q]):
            continue                                    # already top-1: no action
        acted[q] = True
        winner[q] = col
    return acted, winner


def apply(scores: np.ndarray, test: pd.DataFrame, train: pd.DataFrame
          ) -> tuple[np.ndarray, dict]:
    """Apply the frozen rule to a dataset1 score matrix.

    Parameters
    ----------
    scores : (rows, 100) float64 control score matrix, row order as ``test``.
    test   : the physical ``test.csv`` (needs ``src`` and ``c1..c100``).
    train  : the physical ``train.csv`` (needs ``src`` and ``dst``).

    Returns
    -------
    (treatment, info) where ``treatment`` is a new max-normalised matrix and
    ``info`` carries the action accounting.
    """
    candidates = test[CANDIDATE_COLUMNS].to_numpy(np.int64)
    if scores.shape != candidates.shape:
        raise ValueError(f"score shape {scores.shape} != candidate shape {candidates.shape}")

    sources = test["src"].to_numpy(np.int64)
    num_entity = entity_upper_bound(sources, candidates, train["src"], train["dst"])
    recurrence = slate_recurrence(sources, candidates, num_entity)
    is_history = history_mask(train, sources, candidates, num_entity)

    acted, winner = action_mask(scores, recurrence, is_history)

    treatment = scores.copy()
    for q in np.flatnonzero(acted):
        treatment[q, winner[q]] = promoted_value(treatment[q])   # 5. action
    treatment = row_max_normalise(treatment)

    control_top = scores.argmax(axis=1)
    treatment_top = treatment.argmax(axis=1)
    changed = treatment_top != control_top
    if not np.array_equal(changed, acted):
        raise AssertionError("top-1 change mask does not equal the action mask")
    if not np.all(treatment_top[acted] == winner[acted]):
        raise AssertionError("a promoted winner did not become strict top-1")

    return treatment, {
        "strategy": STRATEGY_ID,
        "rows": int(scores.shape[0]),
        "actions": int(acted.sum()),
        "top1_changes": int(changed.sum()),
        "action_coverage": float(acted.mean()),
        "minimum_other_slates": MINIMUM_OTHER_SLATES,
        "winner_min_other_slates": int(recurrence[acted, winner[acted]].min())
        if acted.any() else None,
    }
