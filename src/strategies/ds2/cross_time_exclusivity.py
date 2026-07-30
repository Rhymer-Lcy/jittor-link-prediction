# -*- coding: utf-8 -*-
"""dataset2 cross-time exclusivity decode (strategy id ``xte_cross_time_exclusivity_decode``).

Status
------
Shipped in the final accepted A-board member. Its lifecycle is recorded on three
axes in ``docs/strategy_inventory.json`` because they say different things and
none of them cancels the others:

``historical_lifecycle: CLOSED_AT_LOCKED_GATE``
    Round 23 measured the online gain as ``+0.0017435797158127`` against a
    pre-declared shipping gate of ``+0.002`` and closed the candidate for
    falling ``0.0002564202841873`` short. **This decoder never passed that
    gate and this module does not claim it did.**
``evidence_status: ONLINE_VALIDATED``
    The gain is a real, isolated, online-observed measurement, not an offline
    replay estimate.
``operational_lifecycle: SHIPPED_ACTIVE``
    At A-board closure the separate ``FINAL_BOARD_MAXIMISATION_OVERRIDE``
    decision shipped the exact online-observed member anyway, to maximise the
    final board position rather than to satisfy the scientific gate.

Mechanism
---------
The candidate generator has a measured exclusivity invariant: for a fixed source
and a fixed candidate answer, the answer does not recur at two distinct
timestamps (0 of 2,264,807 dataset2 train pairs; 0 of 244,056 replay truths). The
served member nevertheless offered the same candidate as top-1 for the same
source at two or more distinct times, so at most one of those rows can be right
and the rest are provably wrong under the invariant.

The decode is truth-free: it reads candidate identities, query timestamps and the
served scores. It never reads a label, and it never uses candidate column
position.

Frozen deployment rule (one pass, no thresholds, no refit)
----------------------------------------------------------
With ``c1``/``c2`` the stable descending rank-1/rank-2 candidates of a row and
``margin = score(c1) - score(c2)``:

1. group rows by ``(source, c1)``;
2. a group is eligible only if its rows span at least two distinct timestamps;
3. partition an eligible group into exact timestamp clusters;
4. keep the cluster whose maximum row margin is largest; ties resolve to the
   smallest timestamp;
5. in every row of every other cluster, swap the two score values at ``c1`` and
   ``c2``. Nothing else changes.

Prohibited variants: introducing a margin threshold, iterating to a second pass,
keeping the largest cluster by row count instead of by best margin, resolving
ties by anything other than the smallest timestamp, re-scoring, blending or
sweeping. The rule is frozen by an online adjudication.

Byte preservation
-----------------
:func:`apply` implements the rule on a float matrix, which is what the tests and
any downstream analysis need. The shipped member, however, was built by swapping
the two score **tokens** in the base member's own bytes: unaffected rows are
copied verbatim and no score is ever re-serialised from a float.
:func:`swap_score_tokens` is that path, and it is the one
``src/build_ds2_member.py`` uses to reproduce the accepted member exactly.

Evidence
--------
Isolated auxiliary online A/B of the dataset2 component: control
``0.6789511047001768`` -> treatment ``0.6806946844159895``, delta
``+0.0017435797158127``. Acts on 7,815 of 153,420 rows (5.09%), 15,630 cells,
7,815 top-1 changes, 0 pairs collapsed to a tie.

Provenance: proposed in round 23 and independently re-derived in
``scratchpad/round-23-opus/xte/``; graduated into ``src/`` for the A-board code
inspection. The graduation preserves the algorithm, the parameters (there are
none to tune) and the serving order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..shared.frozen_ops import (
    csv_record_spans,
    dense_group_ids,
    stable_rank_order,
)

STRATEGY_ID = "xte_cross_time_exclusivity_decode"
DATASET = "dataset2"

#: Minimum distinct timestamps a ``(source, top-1)`` group needs to be eligible.
MINIMUM_DISTINCT_TIMES = 2

#: Row/action anchors of the accepted deployment, asserted by the tracked tests.
EXPECTED_ROWS = 153420
EXPECTED_COLUMNS = 100
EXPECTED_ACTIONS = 7815

#: Full physical census of the accepted deployment. Every value is a property of
#: the base member and the test file together, so any disagreement means an input
#: is not the one that was shipped -- which must abort a rebuild, not warn.
CENSUS_ANCHORS = {
    "rows": 153420,
    "columns": 100,
    "sources": 2180,
    "src_top1_groups": 142693,
    "groups_multirow": 9288,
    "groups_multirow_same_time_only": 2727,
    "violation_groups_distinct_times_ge2": 6561,
    "rows_in_violation_groups": 14555,
    "guaranteed_wrong_top1_lower_bound": 7730,
    "planned_action_rows": 7815,
    "max_violation_group_size": 8,
    "violation_groups_after_one_pass": 1589,
}


def derive_actions(scores: np.ndarray, candidates: np.ndarray, sources: np.ndarray,
                   times: np.ndarray) -> dict:
    """Resolve the frozen rule into an action set plus the full physical census.

    Parameters
    ----------
    scores     : (rows, columns) float64 served score matrix, row order as the test file.
    candidates : (rows, columns) int64 candidate ids, same shape as ``scores``.
    sources    : (rows,) int64 query source per row.
    times      : (rows,) int64 query timestamp per row.

    Returns a dict carrying ``acted`` (bool mask), ``c1_col``/``c2_col`` (the
    rank-1/rank-2 column of every row), ``margin`` and ``census``.
    """
    scores = np.asarray(scores, np.float64)
    candidates = np.asarray(candidates, np.int64)
    sources = np.asarray(sources, np.int64)
    times = np.asarray(times, np.int64)
    if scores.shape != candidates.shape:
        raise ValueError(f"score shape {scores.shape} != candidate shape {candidates.shape}")
    if scores.shape[1] < 2:
        raise ValueError("the rule needs at least two candidates per row")
    if sources.shape[0] != scores.shape[0] or times.shape[0] != scores.shape[0]:
        raise ValueError("sources and times must have one entry per score row")

    rows = np.arange(scores.shape[0])
    order = stable_rank_order(scores)
    c1_col, c2_col = order[:, 0], order[:, 1]
    top1 = candidates[rows, c1_col]
    top2 = candidates[rows, c2_col]
    margin = scores[rows, c1_col] - scores[rows, c2_col]

    gid, n_groups = dense_group_ids(sources, top1)      # 1. (source, top-1 candidate)
    ctid, n_clusters = dense_group_ids(gid, times)       # 3. exact timestamp clusters

    group_size = np.bincount(gid, minlength=n_groups)
    cluster_size = np.bincount(ctid, minlength=n_clusters)
    cluster_gid = np.zeros(n_clusters, np.int64)
    cluster_gid[ctid] = gid
    distinct_times = np.bincount(cluster_gid, minlength=n_groups)

    cluster_max_margin = np.full(n_clusters, -np.inf)
    np.maximum.at(cluster_max_margin, ctid, margin)
    group_best_margin = np.full(n_groups, -np.inf)
    np.maximum.at(group_best_margin, cluster_gid, cluster_max_margin)

    # 4. Keeper cluster: largest cluster-maximum margin, ties to the smallest
    # timestamp. dense_group_ids assigns cluster ids in lexicographic (group,
    # time) order, so the first eligible cluster of a group is its earliest one.
    eligible = cluster_max_margin == group_best_margin[cluster_gid]
    eligible_idx = np.flatnonzero(eligible)
    first_gid, first_pos = np.unique(cluster_gid[eligible_idx], return_index=True)
    keeper = np.full(n_groups, -1, np.int64)
    keeper[first_gid] = eligible_idx[first_pos]

    violation_group = distinct_times >= MINIMUM_DISTINCT_TIMES    # 2. eligibility
    in_violation = violation_group[gid]
    acted = in_violation & (ctid != keeper[gid])                  # 5. action set

    # The guaranteed-wrong lower bound keeps the LARGEST cluster by row count.
    # That is a different quantity from the treatment's best-margin keeper and is
    # reported for the census only; it never drives an action.
    largest_cluster = np.zeros(n_groups, np.int64)
    np.maximum.at(largest_cluster, cluster_gid, cluster_size)
    guaranteed_wrong = int((group_size[violation_group]
                            - largest_cluster[violation_group]).sum())

    sizes = group_size[violation_group]
    census = {
        "rows": int(scores.shape[0]),
        "columns": int(scores.shape[1]),
        "sources": int(np.unique(sources).size),
        "src_top1_groups": int(n_groups),
        "groups_multirow": int((group_size >= 2).sum()),
        "groups_multirow_same_time_only": int(((group_size >= 2) & (distinct_times == 1)).sum()),
        "violation_groups_distinct_times_ge2": int(violation_group.sum()),
        "rows_in_violation_groups": int(in_violation.sum()),
        "guaranteed_wrong_top1_lower_bound": guaranteed_wrong,
        "planned_action_rows": int(acted.sum()),
        "keeper_rows_in_violation_groups": int(in_violation.sum() - acted.sum()),
        "max_violation_group_size": int(sizes.max()) if sizes.size else 0,
    }

    # One-pass discipline: report how many violation groups the swap itself
    # induces. They are deliberately NOT processed -- the rule is one pass.
    induced_top1 = top1.copy()
    induced_top1[acted] = top2[acted]
    induced_gid, induced_n = dense_group_ids(sources, induced_top1)
    induced_ctid, _ = dense_group_ids(induced_gid, times)
    induced_cluster_gid = np.zeros(int(induced_ctid.max()) + 1, np.int64)
    induced_cluster_gid[induced_ctid] = induced_gid
    census["violation_groups_after_one_pass"] = int(
        (np.bincount(induced_cluster_gid, minlength=induced_n) >= MINIMUM_DISTINCT_TIMES).sum())

    return {
        "acted": acted,
        "c1_col": c1_col,
        "c2_col": c2_col,
        "top1": top1,
        "top2": top2,
        "margin": margin,
        "census": census,
    }


def census_mismatches(census: dict, anchors: dict | None = None) -> dict:
    """Census entries that disagree with the accepted-deployment anchors."""
    anchors = CENSUS_ANCHORS if anchors is None else anchors
    return {key: {"expected": want, "observed": census.get(key)}
            for key, want in anchors.items() if census.get(key) != want}


def apply(scores: np.ndarray, test: pd.DataFrame, train: pd.DataFrame | None = None
          ) -> tuple[np.ndarray, dict]:
    """Apply the frozen decode to a dataset2 score matrix.

    Parameters
    ----------
    scores : (rows, 100) float64 served score matrix, row order as ``test``.
    test   : the physical ``test.csv``; needs ``src``, ``time`` and the candidate
             columns.
    train  : accepted and ignored. The decode needs no training data -- the
             exclusivity invariant it relies on was measured once and frozen.
             The parameter exists so this stage has the same call shape as the
             dataset1 postprocessors.

    Returns
    -------
    ``(treatment, info)``. ``treatment`` is a new matrix with the two affected
    score values swapped per acted row; every other value is untouched. Note
    that this path re-serialises floats if written with a float format -- use
    :func:`swap_score_tokens` when byte identity with the shipped member matters.
    """
    del train                       # documented above: the decode is training-free
    candidates = candidate_matrix(test)
    sources = test["src"].to_numpy(np.int64)
    times = test["time"].to_numpy(np.int64)

    derived = derive_actions(scores, candidates, sources, times)
    acted, c1_col, c2_col = derived["acted"], derived["c1_col"], derived["c2_col"]
    acted_rows = np.flatnonzero(acted)

    treatment = np.asarray(scores, np.float64).copy()
    a, b = c1_col[acted_rows], c2_col[acted_rows]
    treatment[acted_rows, a], treatment[acted_rows, b] = (
        treatment[acted_rows, b].copy(), treatment[acted_rows, a].copy())

    before = stable_rank_order(scores)[:, 0]
    after = stable_rank_order(treatment)[:, 0]
    changed = before != after
    margin = derived["margin"]

    # Two invariants hold for every input, and both are asserted:
    #   1. the decode never moves the top-1 of a row it did not act on;
    #   2. an acted row with a strictly positive margin always moves its top-1.
    # A zero-margin acted row swaps two EQUAL values and therefore changes
    # nothing. The accepted deployment contains no such row (min acted margin is
    # strictly positive), but a tie is legitimate input -- at B-board scale it
    # will occur -- so it is counted and reported rather than treated as an
    # error. This is why the two masks are not asserted equal.
    if changed[~acted].any():
        raise AssertionError("the decode moved the top-1 of a row it did not act on")
    strict = acted & (margin > 0)
    if not changed[strict].all():
        raise AssertionError("an acted row with a positive margin did not move its top-1")

    info = {
        "strategy": STRATEGY_ID,
        "rows": int(scores.shape[0]),
        "actions": int(acted.sum()),
        "top1_changes": int(changed.sum()),
        "cells_changed": int(2 * acted.sum()),
        "action_coverage": float(acted.mean()),
        "action_rows": acted_rows,
        "c1_col": c1_col,
        "c2_col": c2_col,
        "minimum_distinct_times": MINIMUM_DISTINCT_TIMES,
        "min_acted_margin": float(margin[acted].min()) if acted.any() else None,
        # Acted rows whose two swapped scores were already equal, so the swap is
        # a no-op on the ranking. Zero in the accepted deployment.
        "pairs_collapsed_to_tie": int((margin[acted] == 0).sum()) if acted.any() else 0,
        "acted_rows_without_a_top1_change": int((acted & ~changed).sum()),
        "census": derived["census"],
    }
    return treatment, info


def candidate_matrix(test: pd.DataFrame) -> np.ndarray:
    """Candidate id matrix from a dataset2 test frame.

    dataset2's candidate columns are discovered by prefix rather than assumed to
    be ``c1..c100``, because the column count is a property of the released data
    and the B-board scenarios are not promised to expose exactly 100.
    """
    columns = [c for c in test.columns if c.startswith("c") and c[1:].isdigit()]
    if not columns:
        raise ValueError("no candidate columns found in the dataset2 test frame")
    columns.sort(key=lambda c: int(c[1:]))
    return test[columns].to_numpy(np.int64)


def swap_score_tokens(payload: bytes, acted_rows: np.ndarray, c1_col: np.ndarray,
                      c2_col: np.ndarray, columns: int) -> bytes:
    """Rewrite only the two score tokens of each acted row; copy every other byte.

    This is what makes the rebuilt member byte-identical to the online-scored one:
    the surviving tokens are the base member's own bytes, so no rounding,
    formatting or locale decision can perturb them.
    """
    spans = csv_record_spans(payload)
    out = bytearray()
    cursor = 0
    for row in np.sort(np.asarray(acted_rows, np.int64)):
        start, end = int(spans[row, 0]), int(spans[row, 1])
        out += payload[cursor:start]
        record = payload[start:end]
        # Preserve a CRLF record terminator: split the payload on LF, then keep
        # any trailing CR outside the comma-separated field list.
        suffix = b""
        if record.endswith(b"\r"):
            record, suffix = record[:-1], b"\r"
        fields = record.split(b",")
        if len(fields) != columns:
            raise ValueError(f"row {row}: {len(fields)} fields, expected {columns}")
        a, b = int(c1_col[row]), int(c2_col[row])
        fields[a], fields[b] = fields[b], fields[a]
        out += b",".join(fields) + suffix
        cursor = end
    out += payload[cursor:]
    return bytes(out)
