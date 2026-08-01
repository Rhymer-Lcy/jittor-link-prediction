# -*- coding: utf-8 -*-
"""The dataset1 ranker must serialise into the [0, 1] submission domain.

DIAGNOSTIC branch ``diag/ds1-score-domain``. Hypothesis under test:

    The frozen A-board production path applied a per-row MIN-MAX at
    serialisation; the port to ``src/ranker_ds1.py`` carried ``rownorm``
    (``v / v.max()``) instead, and that single substitution is what puts the
    reconstructed dataset1 member outside [0, 1].

The tests below pin the restored contract, pin the boundary that must NOT move,
and include an anti-vacuity case proving they discriminate -- ``rownorm`` on the
same hostile input fails the predicate that min-max passes.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("DATASET", "dataset1")

import pipeline_common as pc                              # noqa: E402
from ranker_ds1 import serialisation_normalise            # noqa: E402


def lambdarank_like(rows: int, cols: int, seed: int) -> np.ndarray:
    """Margins with the measured shape: straddling zero, mostly negative."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc=-1.5, scale=1.2, size=(rows, cols))


class SerialisationDomainTest(unittest.TestCase):
    def test_every_row_lands_in_the_unit_interval(self):
        for row in lambdarank_like(200, 100, seed=1):
            out = serialisation_normalise(row)
            self.assertGreaterEqual(out.min(), 0.0)
            self.assertLessEqual(out.max(), 1.0)

    def test_row_minimum_is_exactly_zero_and_maximum_exactly_one(self):
        # The fingerprint measured on the frozen artifact: 61,051/61,051 rows.
        for row in lambdarank_like(200, 100, seed=2):
            out = serialisation_normalise(row)
            self.assertEqual(float(out.min()), 0.0)
            self.assertEqual(float(out.max()), 1.0)

    def test_exactly_one_zero_and_one_one_per_row(self):
        # Distinguishes min-max from clipping at 0, which would leave ~97 zeros.
        for row in lambdarank_like(200, 100, seed=3):
            out = serialisation_normalise(row)
            self.assertEqual(int((out == 0.0).sum()), 1)
            self.assertEqual(int((out == 1.0).sum()), 1)

    def test_within_row_order_is_preserved_exactly(self):
        # MRR depends on nothing else, so this is what makes the step
        # score-neutral rather than a new scientific rule.
        for row in lambdarank_like(200, 100, seed=4):
            before = np.argsort(-row, kind="stable")
            after = np.argsort(-serialisation_normalise(row), kind="stable")
            self.assertTrue(np.array_equal(before, after))

    def test_an_all_negative_row_is_still_mapped_into_range(self):
        row = np.linspace(-9.0, -0.5, 100)
        out = serialisation_normalise(row)
        self.assertEqual(float(out.min()), 0.0)
        self.assertEqual(float(out.max()), 1.0)

    def test_a_degenerate_row_takes_the_established_one_half_convention(self):
        # NOT a convention invented here: frozen_ops.row_max_normalise already
        # maps a constant row to 0.5, pinned by
        # test_constant_negative_rows_map_to_one_half. This keeps the two ds1
        # [0, 1] normalisers agreeing on the one degenerate input they share.
        for row in (np.full(100, -2.5), np.full(100, 0.0), np.full(100, 7.0)):
            out = serialisation_normalise(row)
            self.assertTrue(np.all(np.isfinite(out)))
            self.assertTrue(np.array_equal(out, np.full(100, 0.5)))

    def test_the_degenerate_convention_matches_row_max_normalise(self):
        from strategies.shared.frozen_ops import row_max_normalise
        row = np.full(100, -2.5)
        self.assertTrue(np.array_equal(serialisation_normalise(row),
                                       row_max_normalise(row.reshape(1, -1))[0]))

    def test_output_is_finite_on_a_wide_dynamic_range(self):
        row = np.array([-1e12] + [0.0] * 98 + [1e12])
        out = serialisation_normalise(row)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)


class AntiVacuityTest(unittest.TestCase):
    """If rownorm also passed these, the tests would prove nothing."""

    def test_rownorm_leaves_negatives_in_the_serialised_matrix(self):
        rows = lambdarank_like(200, 100, seed=5)
        out = np.array([pc.rownorm(r) for r in rows])
        self.assertLess(out.min(), 0.0)
        self.assertEqual(int((out.min(axis=1) == 0.0).sum()), 0)

    def test_rownorm_amplifies_a_row_whose_maximum_is_not_positive(self):
        row = np.linspace(-9.0, -0.5, 100)
        self.assertLess(pc.rownorm(row).min(), -1.0)


class BoundaryTest(unittest.TestCase):
    """``pipeline_common.rownorm`` is shared with dataset2 and must not move."""

    def test_rownorm_is_unchanged_by_this_branch(self):
        row = np.array([0.0, 1.0, 2.0, 4.0])
        self.assertTrue(np.array_equal(pc.rownorm(row), row / 4.0))

    def test_rownorm_still_returns_an_all_zero_row_unchanged(self):
        zero = np.zeros(8)
        self.assertTrue(np.array_equal(pc.rownorm(zero), zero))


class FrozenRankerFixedPointTest(unittest.TestCase):
    """The patched persistence path must reproduce the frozen ranker byte for byte.

    SCOPE, stated precisely. This is a FIXED-POINT reproduction, not a
    regeneration from producer inputs. The frozen artifact already satisfies the
    min-max contract (every row minimum exactly 0, every row maximum exactly 1),
    so the restored transform must be the identity on it and re-serialisation
    must return the same bytes.

    A regeneration test is NOT possible: the 2026-07-27 booster was never
    persisted, and a retrained booster is not bit-identical (measured: 98.79%
    top-1 agreement). So this test proves the patch does not disturb the frozen
    contract; it cannot by itself prove min-max was the historical transform --
    ``rownorm`` is also the identity on an already-normalised matrix. The
    discriminating evidence is the row-minimum fingerprint, covered above.
    """

    EXPECTED = "cb4964ea21dcefdecb2a34f13ca8500adcc54120684f10052f78a87df88da16b"

    def setUp(self):
        self.frozen = REPO / "outputs" / "dataset1-ensemble" / "result_ranker.csv"
        if not self.frozen.is_file():
            self.skipTest(f"local asset not available: {self.frozen}")

    def test_patched_persistence_reproduces_the_frozen_ranker_bytes(self):
        import hashlib
        import tempfile

        import pandas as pd

        matrix = pd.read_csv(self.frozen, header=None).to_numpy(np.float64)
        self.assertEqual(matrix.shape, (61051, 100))
        out = [serialisation_normalise(row).tolist() for row in matrix]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "result_ranker.csv"
            pd.DataFrame(out).to_csv(path, index=False, header=False,
                                     float_format="%.6f")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(path.stat().st_size, self.frozen.stat().st_size)
        self.assertEqual(digest, self.EXPECTED,
                         "the patched persistence path no longer reproduces the "
                         "frozen dataset1 ranker artifact")

    def test_the_frozen_artifact_carries_the_min_max_fingerprint(self):
        # The discriminating measurement, pinned so it cannot silently change.
        import pandas as pd

        matrix = pd.read_csv(self.frozen, header=None).to_numpy(np.float64)
        self.assertEqual(int((matrix.min(axis=1) == 0.0).sum()), 61051)
        self.assertEqual(int((matrix.max(axis=1) == 1.0).sum()), 61051)
        self.assertEqual(int((matrix == 0.0).sum()), 61084)
        self.assertEqual(int((matrix == 1.0).sum()), 61051)


if __name__ == "__main__":
    unittest.main()
