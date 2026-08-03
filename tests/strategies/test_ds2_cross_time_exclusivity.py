# -*- coding: utf-8 -*-
"""Unit tests for the frozen dataset2 cross-time exclusivity decode.

Lightweight: synthetic slates only, no competition data, no local artifacts. The
byte-exact reproduction of the accepted member lives in
``tests/integration/test_accepted_reproduction.py`` and skips when the local
artifacts are absent.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from strategies.ds2 import cross_time_exclusivity as xte  # noqa: E402
from strategies.shared import frozen_ops  # noqa: E402

COLUMNS = 4


def make_frame(sources, times, slates):
    """A minimal dataset2 test.csv-shaped frame with ``COLUMNS`` candidates."""
    rows = [[src, time, *slate] for src, time, slate in zip(sources, times, slates)]
    names = ["src", "time"] + [f"c{i}" for i in range(1, COLUMNS + 1)]
    return pd.DataFrame(rows, columns=names)


def scores_from(order_values):
    """Score rows given explicitly, as float64."""
    return np.asarray(order_values, np.float64)


class SharedPrimitiveTest(unittest.TestCase):
    def test_dense_group_ids_are_lexicographic(self):
        # The keeper tie-break depends on this: ids ascend with the key tuple, so
        # the first cluster of a group is its smallest timestamp.
        gid, count = frozen_ops.dense_group_ids(np.array([2, 1, 2, 1]), np.array([50, 90, 10, 90]))
        self.assertEqual(count, 3)
        self.assertEqual(gid[1], gid[3])  # (1, 90) twice
        self.assertLess(gid[1], gid[2])  # (1, 90) before (2, 10)
        self.assertLess(gid[2], gid[0])  # (2, 10) before (2, 50)

    def test_csv_record_spans_locate_every_record(self):
        payload = b"1,2\n3,4\n5,6\n"
        spans = frozen_ops.csv_record_spans(payload, expected_rows=3)
        self.assertEqual([payload[a:b] for a, b in spans], [b"1,2", b"3,4", b"5,6"])

    def test_csv_record_spans_reject_a_missing_final_newline(self):
        with self.assertRaises(ValueError):
            frozen_ops.csv_record_spans(b"1,2\n3,4")

    def test_csv_record_spans_reject_a_row_count_mismatch(self):
        with self.assertRaises(ValueError):
            frozen_ops.csv_record_spans(b"1,2\n3,4\n", expected_rows=3)


class RuleTest(unittest.TestCase):
    def test_acts_on_every_non_keeper_cluster_of_a_cross_time_group(self):
        # One source, one shared top-1 candidate (7), three distinct timestamps.
        # Row 0 has the largest margin, so it keeps; rows 1 and 2 are swapped.
        test = make_frame([1, 1, 1], [10, 20, 30], [[7, 8, 9, 5], [7, 8, 9, 5], [7, 8, 9, 5]])
        scores = scores_from(
            [[0.90, 0.10, 0.05, 0.01], [0.60, 0.50, 0.05, 0.01], [0.55, 0.54, 0.05, 0.01]]
        )
        treatment, info = xte.apply(scores, test)
        self.assertEqual(info["actions"], 2)
        self.assertEqual(info["cells_changed"], 4)
        np.testing.assert_array_equal(info["action_rows"], [1, 2])
        np.testing.assert_allclose(treatment[0], scores[0])  # keeper untouched
        self.assertAlmostEqual(treatment[1, 0], 0.50)  # swapped
        self.assertAlmostEqual(treatment[1, 1], 0.60)

    def test_a_single_timestamp_group_is_never_eligible(self):
        # Same source, same top-1, but one timestamp: this is CRF territory and
        # the decode must leave it alone by construction.
        test = make_frame([1, 1, 1], [10, 10, 10], [[7, 8, 9, 5], [7, 8, 9, 5], [7, 8, 9, 5]])
        scores = scores_from([[0.9, 0.1, 0.05, 0.01]] * 3)
        treatment, info = xte.apply(scores, test)
        self.assertEqual(info["actions"], 0)
        np.testing.assert_array_equal(treatment, scores)

    def test_distinct_top1_candidates_are_distinct_groups(self):
        # Same source and different times, but different served top-1: no
        # exclusivity claim exists, so nothing is acted on.
        test = make_frame([1, 1], [10, 20], [[7, 8, 9, 5], [8, 7, 9, 5]])
        scores = scores_from([[0.9, 0.1, 0.05, 0.01], [0.9, 0.1, 0.05, 0.01]])
        _, info = xte.apply(scores, test)
        self.assertEqual(info["actions"], 0)

    def test_grouping_is_per_source(self):
        # Two different sources sharing a top-1 candidate at different times are
        # two groups. The invariant is per (source, answer), not per answer.
        test = make_frame([1, 2], [10, 20], [[7, 8, 9, 5], [7, 8, 9, 5]])
        scores = scores_from([[0.9, 0.1, 0.05, 0.01], [0.9, 0.1, 0.05, 0.01]])
        _, info = xte.apply(scores, test)
        self.assertEqual(info["actions"], 0)

    def test_keeper_is_the_best_margin_cluster_not_the_largest(self):
        # Timestamp 20 has two rows; timestamp 10 has one row with a bigger
        # margin. The frozen rule keeps the BEST MARGIN cluster, so the two rows
        # at t=20 are acted on and the single row at t=10 survives.
        test = make_frame([1, 1, 1], [10, 20, 20], [[7, 8, 9, 5]] * 3)
        scores = scores_from(
            [[0.95, 0.05, 0.02, 0.01], [0.60, 0.50, 0.02, 0.01], [0.58, 0.52, 0.02, 0.01]]
        )
        _, info = xte.apply(scores, test)
        np.testing.assert_array_equal(info["action_rows"], [1, 2])
        self.assertEqual(info["census"]["guaranteed_wrong_top1_lower_bound"], 1)

    def test_keeper_tie_resolves_to_the_smallest_timestamp(self):
        # Identical margins in both clusters: the earlier timestamp keeps.
        test = make_frame([1, 1], [30, 10], [[7, 8, 9, 5]] * 2)
        scores = scores_from([[0.9, 0.1, 0.05, 0.01], [0.9, 0.1, 0.05, 0.01]])
        _, info = xte.apply(scores, test)
        np.testing.assert_array_equal(info["action_rows"], [0])  # t=30 acted, t=10 kept

    def test_rank_order_ties_break_by_column_index(self):
        # Two candidates tie for the row maximum. The stable order takes the
        # lower column as rank 1, which fixes c1/c2 deterministically.
        test = make_frame([1, 1], [10, 20], [[7, 8, 9, 5]] * 2)
        scores = scores_from([[0.9, 0.9, 0.05, 0.01], [0.5, 0.4, 0.05, 0.01]])
        treatment, info = xte.apply(scores, test)
        self.assertEqual(int(info["c1_col"][0]), 0)
        self.assertEqual(int(info["c2_col"][0]), 1)
        # Row 0 is acted on (it is the weaker-margin cluster) but its two scores
        # are equal, so the swap cannot move the top-1. That is legitimate input,
        # counted and reported rather than raised. The accepted deployment has
        # none of these rows.
        np.testing.assert_array_equal(info["action_rows"], [0])
        self.assertEqual(info["pairs_collapsed_to_tie"], 1)
        self.assertEqual(info["acted_rows_without_a_top1_change"], 1)
        self.assertEqual(info["min_acted_margin"], 0.0)
        np.testing.assert_allclose(treatment[0], scores[0])

    def test_a_row_never_acted_on_keeps_its_top1(self):
        # Invariant 1: the decode must not disturb a row outside the action set.
        test = make_frame([1, 1, 2], [10, 20, 10], [[7, 8, 9, 5], [7, 8, 9, 5], [4, 3, 2, 1]])
        scores = scores_from(
            [[0.90, 0.10, 0.05, 0.01], [0.60, 0.50, 0.05, 0.01], [0.70, 0.20, 0.05, 0.01]]
        )
        treatment, info = xte.apply(scores, test)
        untouched = np.setdiff1d(np.arange(3), info["action_rows"])
        for row in untouched:
            np.testing.assert_allclose(treatment[row], scores[row])

    def test_row_and_candidate_counts_are_preserved(self):
        test = make_frame([1, 1, 1], [10, 20, 30], [[7, 8, 9, 5]] * 3)
        scores = scores_from(
            [[0.90, 0.10, 0.05, 0.01], [0.60, 0.50, 0.05, 0.01], [0.55, 0.54, 0.05, 0.01]]
        )
        treatment, _ = xte.apply(scores, test)
        self.assertEqual(treatment.shape, scores.shape)
        # The decode permutes values inside a row; it never adds, drops or
        # rewrites a score, so each row is a permutation of its original.
        for before, after in zip(scores, treatment):
            np.testing.assert_allclose(np.sort(before), np.sort(after))

    def test_candidate_set_is_never_modified(self):
        test = make_frame([1, 1], [10, 20], [[7, 8, 9, 5]] * 2)
        before = test.copy(deep=True)
        xte.apply(scores_from([[0.9, 0.1, 0.05, 0.01], [0.6, 0.5, 0.05, 0.01]]), test)
        pd.testing.assert_frame_equal(test, before)

    def test_is_deterministic(self):
        test = make_frame([1, 1, 1], [10, 20, 30], [[7, 8, 9, 5]] * 3)
        scores = scores_from(
            [[0.90, 0.10, 0.05, 0.01], [0.60, 0.50, 0.05, 0.01], [0.55, 0.54, 0.05, 0.01]]
        )
        first, info_a = xte.apply(scores, test)
        second, info_b = xte.apply(scores, test)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(info_a["census"], info_b["census"])

    def test_needs_no_training_data(self):
        # The decode is training-free. Passing a train frame must not change the
        # result, which is what lets the same stage serve any scenario.
        test = make_frame([1, 1], [10, 20], [[7, 8, 9, 5]] * 2)
        scores = scores_from([[0.9, 0.1, 0.05, 0.01], [0.6, 0.5, 0.05, 0.01]])
        without, info_a = xte.apply(scores, test)
        with_train, info_b = xte.apply(
            scores, test, pd.DataFrame({"src": [1], "dst": [7], "time": [1]})
        )
        np.testing.assert_array_equal(without, with_train)
        self.assertEqual(info_a["actions"], info_b["actions"])

    def test_uses_no_label_or_future_column(self):
        # Only src, time and the candidate columns are read. A label column, if
        # one were ever present, must not change the outcome.
        test = make_frame([1, 1], [10, 20], [[7, 8, 9, 5]] * 2)
        scores = scores_from([[0.9, 0.1, 0.05, 0.01], [0.6, 0.5, 0.05, 0.01]])
        _, plain = xte.apply(scores, test)
        poisoned = test.copy(deep=True)
        poisoned["label"] = [3, 3]
        poisoned["answer"] = [9, 9]
        _, with_labels = xte.apply(scores, poisoned)
        self.assertEqual(plain["actions"], with_labels["actions"])
        np.testing.assert_array_equal(plain["action_rows"], with_labels["action_rows"])

    def test_rejects_a_shape_mismatch(self):
        test = make_frame([1, 1], [10, 20], [[7, 8, 9, 5]] * 2)
        with self.assertRaises(ValueError):
            xte.apply(np.zeros((3, COLUMNS)), test)

    def test_rejects_a_frame_without_candidate_columns(self):
        frame = pd.DataFrame({"src": [1], "time": [10], "x": [3]})
        with self.assertRaises(ValueError):
            xte.candidate_matrix(frame)

    def test_candidate_columns_are_ordered_numerically(self):
        # c10 must not sort before c2. The rule reads candidate identities, so a
        # lexicographic column order would mislabel every top-1.
        names = ["src", "time"] + [f"c{i}" for i in range(1, 12)]
        frame = pd.DataFrame([[1, 0] + list(range(100, 111))], columns=names)
        np.testing.assert_array_equal(xte.candidate_matrix(frame)[0], list(range(100, 111)))


class ByteSwapTest(unittest.TestCase):
    def test_swaps_only_the_named_tokens(self):
        payload = b"0.100000,0.900000,0.010000\n0.500000,0.400000,0.020000\n"
        out = xte.swap_score_tokens(
            payload, np.array([0]), np.array([1, 0]), np.array([0, 1]), columns=3
        )
        self.assertEqual(out, b"0.900000,0.100000,0.010000\n0.500000,0.400000,0.020000\n")

    def test_byte_length_is_preserved_when_tokens_are_equal_width(self):
        payload = b"0.100000,0.900000,0.010000\n"
        out = xte.swap_score_tokens(payload, np.array([0]), np.array([1]), np.array([0]), columns=3)
        self.assertEqual(len(out), len(payload))

    def test_unacted_rows_are_copied_verbatim(self):
        payload = b"1,2,3\n4,5,6\n7,8,9\n"
        out = xte.swap_score_tokens(
            payload, np.array([1]), np.array([0, 0, 0]), np.array([0, 2, 0]), columns=3
        )
        self.assertEqual(out, b"1,2,3\n6,5,4\n7,8,9\n")

    def test_crlf_terminators_survive(self):
        payload = b"1,2,3\r\n4,5,6\r\n"
        out = xte.swap_score_tokens(payload, np.array([0]), np.array([0]), np.array([2]), columns=3)
        self.assertEqual(out, b"3,2,1\r\n4,5,6\r\n")

    def test_rejects_a_field_count_mismatch(self):
        with self.assertRaises(ValueError):
            xte.swap_score_tokens(b"1,2\n", np.array([0]), np.array([0]), np.array([1]), columns=3)

    def test_token_swap_matches_the_float_path(self):
        test = make_frame([1, 1, 1], [10, 20, 30], [[7, 8, 9, 5]] * 3)
        scores = scores_from(
            [
                [0.900000, 0.100000, 0.050000, 0.010000],
                [0.600000, 0.500000, 0.050000, 0.010000],
                [0.550000, 0.540000, 0.050000, 0.010000],
            ]
        )
        treatment, info = xte.apply(scores, test)
        payload = "".join(",".join(f"{v:.6f}" for v in row) + "\n" for row in scores).encode(
            "ascii"
        )
        swapped = xte.swap_score_tokens(
            payload, info["action_rows"], info["c1_col"], info["c2_col"], COLUMNS
        )
        expected = "".join(",".join(f"{v:.6f}" for v in row) + "\n" for row in treatment).encode(
            "ascii"
        )
        self.assertEqual(swapped, expected)


class RegistryTest(unittest.TestCase):
    def test_registry_exposes_the_dataset2_stage(self):
        from strategies.registry import POSTPROCESSOR_CHAINS, active_chain_ids, active_ds2_chain_ids

        self.assertEqual(active_ds2_chain_ids(), [xte.STRATEGY_ID])
        self.assertEqual(active_chain_ids("dataset2"), [xte.STRATEGY_ID])
        self.assertEqual(sorted(POSTPROCESSOR_CHAINS), ["dataset1", "dataset2"])

    def test_unknown_dataset_is_rejected(self):
        from strategies.registry import active_chain_ids

        with self.assertRaises(ValueError):
            active_chain_ids("dataset3")

    def test_no_tunable_threshold_is_exposed(self):
        # The rule is frozen and has no threshold. A knob appearing here would be
        # a prohibited variant, so the absence is asserted rather than assumed.
        self.assertEqual(xte.MINIMUM_DISTINCT_TIMES, 2)
        for forbidden in ("MARGIN_THRESHOLD", "TAU", "PASSES", "MAX_ITERATIONS"):
            self.assertFalse(hasattr(xte, forbidden), forbidden)

    def test_accepted_anchors_are_recorded(self):
        self.assertEqual(xte.EXPECTED_ROWS, 153420)
        self.assertEqual(xte.EXPECTED_COLUMNS, 100)
        self.assertEqual(xte.EXPECTED_ACTIONS, 7815)
        self.assertEqual(xte.CENSUS_ANCHORS["planned_action_rows"], xte.EXPECTED_ACTIONS)

    def test_census_mismatches_reports_disagreement(self):
        self.assertEqual(xte.census_mismatches(dict(xte.CENSUS_ANCHORS)), {})
        broken = dict(xte.CENSUS_ANCHORS, rows=1)
        self.assertIn("rows", xte.census_mismatches(broken))


if __name__ == "__main__":
    unittest.main(verbosity=2)
