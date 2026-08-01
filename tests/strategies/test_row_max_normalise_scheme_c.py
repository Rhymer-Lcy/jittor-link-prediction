# -*- coding: utf-8 -*-
"""``row_max_normalise`` must map every row into [0, 1], including rows whose
maximum is not strictly positive.

The frozen implementation divided by ``np.maximum(row_max, 1e-12)``. That guard
was written for an all-ZERO row. Applied to an all-NEGATIVE row it divides by
1e-12 and amplifies the row by twelve orders of magnitude. A canonical dataset1
reconstruction, whose LambdaRank model scored 33 of 61,051 queries entirely
negative, produced member values down to -4.7e24 -- an unsubmittable artifact
that every other check passed because its shape and row count were correct.

Scheme C repairs only the undefined branch. Rows with a positive maximum keep the
previous computation bit for bit, which is why the accepted A-board member is
unchanged.

Boundary: the repair is scoped to this function. ``pipeline_common.rownorm``
faces a different degenerate input -- the all-ZERO row -- and correctly returns it
unchanged. Applying Scheme C there would rewrite 117 measured dataset2 ``f_collab``
rows from 0.0 to 0.5 and change frozen dataset2 semantics, so
``test_scheme_c_is_not_applied_to_rownorm`` pins that boundary.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from strategies.shared.frozen_ops import row_max_normalise  # noqa: E402

EPS = 1e-12


def frozen_row_max_normalise(scores: np.ndarray) -> np.ndarray:
    """The pre-repair implementation, verbatim, as the bit-exactness reference."""
    return scores / np.maximum(scores.max(axis=1, keepdims=True), EPS)


class UnaffectedRowsAreBitExactTest(unittest.TestCase):
    """Everything the frozen chain ever produced must be untouched."""

    def test_ordinary_positive_rows_are_bit_identical(self):
        rows = np.random.default_rng(1).random((300, 100)) + 0.05
        self.assertTrue(np.array_equal(row_max_normalise(rows),
                                       frozen_row_max_normalise(rows)))

    def test_mixed_sign_rows_with_a_positive_maximum_are_bit_identical(self):
        rng = np.random.default_rng(2)
        rows = rng.standard_normal((400, 100))
        rows = rows[rows.max(axis=1) > EPS]
        self.assertGreater(len(rows), 0)
        self.assertTrue(np.array_equal(row_max_normalise(rows),
                                       frozen_row_max_normalise(rows)))

    def test_a_matrix_with_no_degenerate_row_takes_the_frozen_path(self):
        # The fast path must be the frozen expression itself, not a re-derivation
        # that happens to agree to floating-point tolerance.
        rows = np.random.default_rng(3).random((50, 100)) + 1.0
        self.assertTrue(np.array_equal(row_max_normalise(rows),
                                       frozen_row_max_normalise(rows)))

    def test_positive_rows_stay_bit_exact_when_a_degenerate_row_is_present(self):
        # The repair must not perturb its neighbours: the same positive rows are
        # normalised identically whether or not a negative row shares the matrix.
        rng = np.random.default_rng(4)
        positive = rng.random((20, 100)) + 0.05
        mixed = np.vstack([positive, -(rng.random((3, 100)) + 0.05)])
        self.assertTrue(np.array_equal(row_max_normalise(mixed)[:20],
                                       frozen_row_max_normalise(positive)))


class DegenerateRowsTest(unittest.TestCase):
    def test_all_negative_non_constant_rows_land_in_range(self):
        rows = -(np.random.default_rng(5).random((40, 100)) + 0.05)
        out = row_max_normalise(rows)
        self.assertTrue(np.isfinite(out).all())
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)

    def test_all_negative_rows_preserve_candidate_order(self):
        rows = -(np.random.default_rng(6).random((40, 100)) + 0.05)
        self.assertTrue(np.array_equal(
            np.argsort(-rows, axis=1, kind="stable"),
            np.argsort(-row_max_normalise(rows), axis=1, kind="stable")))

    def test_constant_negative_rows_map_to_one_half(self):
        rows = np.full((5, 100), -3.7)
        out = row_max_normalise(rows)
        self.assertTrue(np.array_equal(out, np.full((5, 100), 0.5)))

    def test_all_zero_rows_map_to_one_half(self):
        rows = np.zeros((5, 100))
        out = row_max_normalise(rows)
        self.assertTrue(np.array_equal(out, np.full((5, 100), 0.5)))

    def test_no_floating_point_error_is_raised_on_degenerate_rows(self):
        rows = np.vstack([np.zeros((2, 100)), np.full((2, 100), -1.0)])
        with np.errstate(all="raise"):
            out = row_max_normalise(rows)
        self.assertTrue(np.isfinite(out).all())

    def test_a_row_whose_maximum_is_exactly_zero_is_repaired_not_amplified(self):
        # max == 0 is not > 1e-12, so it takes the repair branch. Under the frozen
        # rule it would have been divided by 1e-12.
        row = np.linspace(-5.0, 0.0, 100)[None, :]
        out = row_max_normalise(row)
        self.assertLessEqual(out.max(), 1.0)
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertGreater(frozen_row_max_normalise(row).min(), 1e10 * -1e12)


class RegressionAgainstTheObservedDefectTest(unittest.TestCase):
    def test_the_frozen_rule_amplifies_and_the_repair_does_not(self):
        # Reproduces the shape of the canonical dataset1 failure: a row whose
        # maximum is a small negative number and whose minimum is large negative.
        row = np.full((1, 100), -3059.18)
        row[0, 0] = -0.457
        before = frozen_row_max_normalise(row)
        after = row_max_normalise(row)
        self.assertLess(before.min(), -1e15, "the frozen rule must show the defect")
        self.assertGreaterEqual(after.min(), 0.0)
        self.assertLessEqual(after.max(), 1.0)
        self.assertEqual(int(np.argmax(after)), int(np.argmax(row)))

    def test_every_row_of_a_hostile_matrix_is_finite_and_bounded_above(self):
        rng = np.random.default_rng(7)
        hostile = np.vstack([
            rng.random((10, 100)) + 0.05,           # ordinary
            rng.standard_normal((10, 100)),         # mixed sign, positive maximum
            -(rng.random((10, 100)) + 0.05),        # all negative
            np.zeros((3, 100)),                     # all zero
            np.full((3, 100), -42.0),               # constant negative
            np.full((3, 100), 7.0),                 # constant positive
        ])
        out = row_max_normalise(hostile)
        self.assertTrue(np.isfinite(out).all())
        self.assertLessEqual(out.max(), 1.0)
        # Every row the REPAIR touched is in [0, 1] ...
        repaired = hostile.max(axis=1) <= EPS
        self.assertGreaterEqual(out[repaired].min(), 0.0)
        self.assertLessEqual(out[repaired].max(), 1.0)


class ScopeIsNotAFullRangeGuaranteeTest(unittest.TestCase):
    """Scheme C alone does NOT make every possible matrix submittable, and must
    not be described as if it did.

    A row whose maximum is strictly positive keeps the frozen computation bit for
    bit. When such a row also contains negative values, ``scores / row_max`` is
    negative there. That is the frozen A-board behaviour and the repair
    deliberately leaves it alone -- changing it would rewrite rows the accepted
    chain actually produced.

    The guarantee that a member is submittable therefore belongs to the final
    output gate, not to this function.
    """

    def test_a_mixed_sign_row_with_a_positive_maximum_still_goes_negative(self):
        row = np.array([[-5.0, -2.0, 1.0]])
        out = row_max_normalise(row)
        self.assertLess(out.min(), 0.0)
        self.assertTrue(np.array_equal(out, frozen_row_max_normalise(row)),
                        "the frozen branch must remain bit-identical")

    def test_the_repair_branch_is_exactly_the_non_positive_maximum_rows(self):
        rng = np.random.default_rng(8)
        rows = np.vstack([rng.standard_normal((60, 100)),
                          -(rng.random((10, 100)) + 0.05),
                          np.zeros((2, 100))])
        changed = ~np.all(row_max_normalise(rows) == frozen_row_max_normalise(rows), axis=1)
        expected = rows.max(axis=1) <= EPS
        self.assertTrue(np.array_equal(changed, expected),
                        "only rows with a non-positive maximum may change")


class ScopeBoundaryTest(unittest.TestCase):
    def test_scheme_c_is_not_applied_to_rownorm(self):
        # pipeline_common.rownorm must keep returning an all-zero row unchanged.
        # Mapping it to 0.5 would rewrite 117 measured dataset2 f_collab rows.
        import pipeline_common as pc
        zero = np.zeros(100)
        self.assertTrue(np.array_equal(pc.rownorm(zero), zero))
        self.assertEqual(float(pc.rownorm(zero).max()), 0.0)

    def test_rownorm_still_normalises_an_ordinary_row(self):
        import pipeline_common as pc
        row = np.arange(1.0, 101.0)
        self.assertAlmostEqual(float(pc.rownorm(row).max()), 1.0)


if __name__ == "__main__":
    unittest.main()
