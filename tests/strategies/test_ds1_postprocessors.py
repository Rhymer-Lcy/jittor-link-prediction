# -*- coding: utf-8 -*-
"""Unit tests for the frozen dataset1 score postprocessors.

Lightweight: synthetic slates only, no competition data, no local artifacts.
The end-to-end reproduction of the accepted member lives in
``tests/integration/test_accepted_reproduction.py`` and skips when the local
data is absent.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from strategies.ds1 import graph_reciprocity as tgr  # noqa: E402
from strategies.ds1 import source_slate_recurrence as ssr  # noqa: E402
from strategies.shared import frozen_ops  # noqa: E402


def make_test_frame(sources, slates):
    """Build a minimal test.csv-shaped frame; slates are padded to 100 columns.

    Padding ids are unique per row on purpose: shared padding would itself recur
    across slates and tie for the maximum recurrence, which the frozen rule
    correctly refuses to act on.
    """
    rows = []
    for row_index, (src, slate) in enumerate(zip(sources, slates)):
        pad = list(slate) + [10_000 + row_index * 1_000 + i for i in range(100 - len(slate))]
        rows.append([src, 0.0] + pad)
    return pd.DataFrame(rows, columns=["src", "time"] + frozen_ops.CANDIDATE_COLUMNS)


def uniform_scores(n_rows, top_col=0):
    """Rows whose max is exactly 1.0 at ``top_col`` -- the accepted format."""
    s = np.full((n_rows, 100), 0.1)
    s[:, top_col] = 1.0
    s[:, 1] = 0.9
    return s


class FrozenOpsTest(unittest.TestCase):
    def test_promotion_value_is_three_on_a_normalised_row(self):
        row = np.array([0.1, 0.5, 1.0] + [0.0] * 97)
        self.assertEqual(frozen_ops.promoted_value(row), 3.0)

    def test_row_max_normalise_leaves_normalised_rows_untouched(self):
        s = uniform_scores(3)
        self.assertTrue(np.array_equal(frozen_ops.row_max_normalise(s), s))

    def test_stable_rank_order_breaks_ties_by_column_index(self):
        s = np.array([[0.5, 0.5, 0.1] + [0.0] * 97])
        self.assertEqual(list(frozen_ops.stable_rank_order(s)[0][:2]), [0, 1])

    def test_history_mask_is_directed(self):
        test = make_test_frame([1], [[7, 8]])
        train = pd.DataFrame({"src": [7], "dst": [1]})  # reverse edge only
        num_entity = frozen_ops.entity_upper_bound(
            test["src"],
            test[frozen_ops.CANDIDATE_COLUMNS].to_numpy(np.int64),
            train["src"],
            train["dst"],
        )
        hm = frozen_ops.history_mask(
            train,
            test["src"].to_numpy(np.int64),
            test[frozen_ops.CANDIDATE_COLUMNS].to_numpy(np.int64),
            num_entity,
        )
        self.assertFalse(hm[0, 0], "a reverse training edge must not count as history")


class SourceSlateRecurrenceTest(unittest.TestCase):
    def setUp(self):
        # source 1 asks three times; candidate 7 recurs in all three slates,
        # candidate 8 only in the first. Column 0 holds the control top-1.
        self.test = make_test_frame([1, 1, 1], [[5, 7, 8], [6, 7], [4, 7]])
        self.train = pd.DataFrame({"src": [1], "dst": [999]})
        self.scores = uniform_scores(3)

    def test_acts_on_a_unique_recurring_non_history_candidate(self):
        treatment, info = ssr.apply(self.scores, self.test, self.train)
        self.assertEqual(info["actions"], 3)
        self.assertEqual(info["top1_changes"], 3)
        # candidate 7 sits at column 1 in row 0 and column 1 in rows 1-2
        self.assertEqual(int(treatment[0].argmax()), 1)

    def test_history_gate_blocks_the_row(self):
        train = pd.DataFrame({"src": [1, 1, 1], "dst": [5, 6, 4]})  # control top-1 is history
        _, info = ssr.apply(self.scores, self.test, train)
        self.assertEqual(info["actions"], 0)

    def test_history_candidates_are_ineligible_as_winners(self):
        train = pd.DataFrame({"src": [1], "dst": [7]})  # the recurring candidate is a partner
        _, info = ssr.apply(self.scores, self.test, train)
        self.assertEqual(info["actions"], 0)

    def test_threshold_requires_two_other_slates(self):
        two_slates = make_test_frame([1, 1], [[5, 7], [6, 7]])  # 7 has only ONE other slate
        _, info = ssr.apply(uniform_scores(2), two_slates, self.train)
        self.assertEqual(info["actions"], 0)
        self.assertEqual(ssr.MINIMUM_OTHER_SLATES, 2, "the support threshold is frozen at 2")

    def test_a_tie_on_maximum_recurrence_disqualifies_the_row(self):
        tied = make_test_frame([1, 1, 1], [[5, 7, 8], [6, 7, 8], [4, 7, 8]])
        _, info = ssr.apply(uniform_scores(3), tied, self.train)
        self.assertEqual(info["actions"], 0, "uniqueness of the maximum is required")

    def test_non_action_rows_are_untouched(self):
        train = pd.DataFrame({"src": [1], "dst": [7]})
        treatment, info = ssr.apply(self.scores, self.test, train)
        self.assertEqual(info["actions"], 0)
        self.assertTrue(np.array_equal(treatment, self.scores))

    def test_is_deterministic(self):
        a, _ = ssr.apply(self.scores, self.test, self.train)
        b, _ = ssr.apply(self.scores, self.test, self.train)
        self.assertTrue(np.array_equal(a, b))

    def test_rejects_a_shape_mismatch(self):
        with self.assertRaises(ValueError):
            ssr.apply(uniform_scores(2), self.test, self.train)


class TestGraphReciprocityTest(unittest.TestCase):
    def setUp(self):
        # row 0: source 1, rank-1 candidate 5, rank-2 candidate 2.
        # source 2's slate contains 1, so the reverse edge 2 -> 1 exists;
        # no slate of source 5 contains 1, so 5 -> 1 does not.
        self.test = make_test_frame([1, 2], [[5, 2], [1, 9]])
        self.train = pd.DataFrame({"src": [1], "dst": [999]})
        self.scores = uniform_scores(2)

    def test_promotes_the_reciprocated_runner_up(self):
        treatment, info = tgr.apply(self.scores, self.test, self.train)
        self.assertEqual(info["actions"], 1)
        self.assertEqual(int(treatment[0].argmax()), 1, "rank-2 candidate becomes top-1")
        self.assertEqual(int(treatment[1].argmax()), 0, "row 1 must not act")

    def test_requires_the_incumbent_to_lack_a_reverse_edge(self):
        # give source 5 a slate containing 1, so A(c1, source) = 1 and the rule stops
        test = make_test_frame([1, 2, 5], [[5, 2], [1, 9], [1, 3]])
        _, info = tgr.apply(uniform_scores(3), test, self.train)
        self.assertEqual(info["actions"], 0)

    def test_history_gates_apply_to_both_candidates(self):
        for dst in (5, 2):
            with self.subTest(historical=dst):
                train = pd.DataFrame({"src": [1], "dst": [dst]})
                _, info = tgr.apply(self.scores, self.test, train)
                self.assertEqual(info["actions"], 0)

    def test_only_rank_two_is_considered(self):
        # candidate 2 is reciprocated but demoted to rank 3; the rule must not see it
        scores = np.full((2, 100), 0.1)
        scores[:, 0] = 1.0
        scores[:, 3] = 0.9  # rank 2 is column 3 (candidate 903, not reciprocated)
        scores[:, 1] = 0.5  # candidate 2 is only rank 3
        _, info = tgr.apply(scores, self.test, self.train)
        self.assertEqual(info["actions"], 0, "ranks below 2 must never be inspected")

    def test_non_action_rows_are_untouched(self):
        treatment, _ = tgr.apply(self.scores, self.test, self.train)
        self.assertTrue(np.array_equal(treatment[1], self.scores[1]))

    def test_is_deterministic(self):
        a, _ = tgr.apply(self.scores, self.test, self.train)
        b, _ = tgr.apply(self.scores, self.test, self.train)
        self.assertTrue(np.array_equal(a, b))


class ChainOrderTest(unittest.TestCase):
    def test_registry_exposes_the_accepted_order(self):
        from strategies.registry import active_ds1_chain_ids

        self.assertEqual(active_ds1_chain_ids(), ["source_slate_recurrence", "graph_reciprocity"])

    def test_no_prohibited_knobs_are_exposed(self):
        """Frozen parameters must be module constants, not call arguments."""
        import inspect

        for fn in (ssr.apply, tgr.apply):
            params = list(inspect.signature(fn).parameters)
            self.assertEqual(
                params,
                ["scores", "test", "train"],
                f"{fn.__module__} must not expose tunable arguments",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
