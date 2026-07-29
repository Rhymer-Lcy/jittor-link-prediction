# -*- coding: utf-8 -*-
"""Unit tests for the submission packaging tool.

Structural checks run against a shrunken SPEC (a 6 x 5 stand-in) so they are
instant and need no 55 MB fixtures; the validation code under test is the same
code the real members go through. Checks that need the accepted archive skip
cleanly when it is absent.
"""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "submission"))

import package_component as pc  # noqa: E402


def report():
    return pc.Report(Path(tempfile.gettempdir()) / "package_component_test.log")


def as_csv(a: np.ndarray) -> bytes:
    buf = io.StringIO()
    pd.DataFrame(a).to_csv(buf, header=False, index=False, float_format="%.6f")
    return buf.getvalue().encode("ascii")


class SmallSpecMixin(unittest.TestCase):
    """Temporarily shrink the ds1 spec so structural tests stay instant."""

    def setUp(self):
        self._saved = pc.SPEC["ds1"].copy()
        pc.SPEC["ds1"] = {**self._saved, "rows": 6, "cols": 5, "expect_rowmax_one": 1.0}
        self.spec = pc.SPEC["ds1"]
        self.good = np.array([[0.1, 0.2, 0.3, 0.4, 1.0]] * 6)

    def tearDown(self):
        pc.SPEC["ds1"] = self._saved


class InputValidationTest(SmallSpecMixin):
    def test_well_formed_input_passes(self):
        rep = report()
        arr = pc.parse_matrix(as_csv(self.good), rep, "t", self.spec)
        pc.validate_values(arr, rep, "ds1", False)
        self.assertFalse(rep.failed)

    def test_nan_is_rejected(self):
        bad = self.good.copy()
        bad[2, 1] = np.nan
        rep = report()
        arr = pc.parse_matrix(as_csv(bad), rep, "t", self.spec)
        pc.validate_values(arr, rep, "ds1", False)
        self.assertTrue(rep.failed)

    def test_infinity_is_rejected(self):
        rep = report()
        arr = np.array(self.good, dtype=float)
        arr[0, 0] = np.inf
        pc.validate_values(arr, rep, "ds1", True)
        self.assertTrue(rep.failed)

    def test_degenerate_row_is_rejected(self):
        bad = self.good.copy()
        bad[3, :] = 0.5
        rep = report()
        arr = pc.parse_matrix(as_csv(bad), rep, "t", self.spec)
        pc.validate_values(arr, rep, "ds1", False)
        self.assertTrue(rep.failed)

    def test_out_of_range_is_rejected_but_can_be_downgraded(self):
        bad = self.good.copy()
        bad[0, 0] = 1.7
        rep = report()
        arr = pc.parse_matrix(as_csv(bad), rep, "t", self.spec)
        pc.validate_values(arr, rep, "ds1", False)
        self.assertTrue(rep.failed)

        rep2 = report()
        pc.validate_values(arr, rep2, "ds1", True)
        self.assertFalse(rep2.failed)
        self.assertTrue(rep2.warned)

    def test_short_file_is_rejected(self):
        rep = report()
        self.assertIsNone(pc.parse_matrix(as_csv(self.good[:5]), rep, "t", self.spec))
        self.assertTrue(rep.failed)

    def test_ragged_row_is_rejected(self):
        lines = as_csv(self.good).decode().split("\n")
        lines[2] = lines[2] + ",0.9"
        rep = report()
        self.assertIsNone(pc.parse_matrix("\n".join(lines).encode(), rep, "t", self.spec))
        self.assertTrue(rep.failed)

    def test_non_numeric_is_rejected(self):
        rep = report()
        raw = as_csv(self.good).replace(b"0.100000", b"abc", 1)
        self.assertIsNone(pc.parse_matrix(raw, rep, "t", self.spec))
        self.assertTrue(rep.failed)


class DiffAccountingTest(SmallSpecMixin):
    def test_counts_are_exact(self):
        trt = self.good.copy()
        trt[1] = [1.0, 0.2, 0.3, 0.4, 0.5]
        rep = report()
        d = pc.diff_against_baseline(trt, self.good, as_csv(trt), as_csv(self.good),
                                     rep, inversions=True)
        self.assertEqual(d["top1_changes"], 1)
        self.assertEqual(d["changed_rows_text"], 1)
        self.assertEqual(d["order_changed_rows"], 1)
        self.assertEqual(d["strict_inversions"], 4)

    def test_identical_input_warns_that_it_measures_nothing(self):
        rep = report()
        d = pc.diff_against_baseline(self.good, self.good, as_csv(self.good),
                                     as_csv(self.good), rep, inversions=False)
        self.assertEqual(d["changed_cells"], 0)
        self.assertTrue(rep.warned)


class SizePolicyTest(unittest.TestCase):
    def test_bands(self):
        cases = [
            (60_000_000, "OK"),
            (pc.SIZE_WARN + 50_000, "ACCEPTABLE-NO-MARGIN"),
            (pc.SIZE_REVIEW + 363_753, "REVIEW-REQUIRED"),
        ]
        for size, want in cases:
            with self.subTest(size=size):
                self.assertEqual(pc.size_policy(size, report()), want)

    def test_thresholds_are_ordered(self):
        self.assertLessEqual(pc.SIZE_TARGET, pc.SIZE_WARN)
        self.assertLess(pc.SIZE_WARN, pc.SIZE_REVIEW)

    def test_policy_knows_a_larger_archive_was_accepted(self):
        """Guards against re-asserting the refuted size-cliff claim."""
        self.assertGreater(pc.LARGEST_ACCEPTED, pc.SIZE_REVIEW)


class ZipWriterTest(unittest.TestCase):
    def test_writes_deflate_9_and_round_trips(self):
        rep = report()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t.zip"
            payload = b"0.1,0.2\n0.3,0.4\n"
            info = pc.write_zip(out, [("dataset1.csv", payload)], rep)
            self.assertEqual(info["compress_method"], 8)
            self.assertEqual(info["compress_level"], 9)
            self.assertFalse(rep.failed)
            with zipfile.ZipFile(out) as z:
                i = z.infolist()[0]
                self.assertEqual(z.comment, b"")
                self.assertEqual(len(z.infolist()), 1)
                self.assertFalse(i.is_dir())
                # container fields must match every accepted archive
                self.assertEqual((i.create_system, i.create_version,
                                  i.external_attr, i.flag_bits), (0, 20, 0x1800000, 0))
                self.assertEqual(z.read("dataset1.csv"), payload)

    def test_preserves_line_endings_verbatim(self):
        rep = report()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t.zip"
            crlf = b"0.1,0.2\r\n0.3,0.4\r\n"
            pc.write_zip(out, [("dataset1.csv", crlf)], rep)
            with zipfile.ZipFile(out) as z:
                self.assertEqual(z.read("dataset1.csv"), crlf,
                                 "CSV members must be treated as opaque bytes")

    def test_member_order_is_preserved(self):
        rep = report()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "t.zip"
            info = pc.write_zip(out, [("dataset1.csv", b"a"), ("dataset2.csv", b"b")], rep)
            self.assertEqual(info["members"], ["dataset1.csv", "dataset2.csv"])


class ScoreArithmeticTest(unittest.TestCase):
    def test_component_addition_reproduces_the_accepted_total(self):
        ds1 = pc.SPEC["ds1"]["baseline_score"]
        ds2 = pc.SPEC["ds2"]["baseline_score"]
        self.assertEqual(ds1 + ds2, pc.ACCEPTED_TOTAL)

    def test_spec_mirrors_the_production_manifest(self):
        self.assertEqual(pc.SPEC["ds1"]["baseline_sha256"],
                         pc.PROD["dataset1"]["member_sha256"])
        self.assertEqual(pc.SPEC["ds2"]["baseline_sha256"],
                         pc.PROD["dataset2"]["member_sha256"])


class ManifestSchemaTest(SmallSpecMixin):
    REQUIRED = [
        "experiment", "dataset", "source_csv_path", "baseline_csv_sha256",
        "treatment_csv_sha256", "zip_path", "zip_sha256", "member_names", "member_sha256",
        "rows", "cols", "changed_rows", "changed_cells", "top1_changes",
        "compress_method", "compress_level", "archive_bytes",
        "baseline_component_score", "treatment_component_score", "component_delta",
        "predicted_combined_total", "created_utc", "command_line",
    ]

    def test_manifest_carries_every_required_field(self):
        if not pc.GOLD_ZIP.exists():
            self.skipTest("accepted archive not available locally")
        import json
        manifests = list((REPO / "outputs" / "submissions").rglob("*_manifest.json"))
        if not manifests:
            self.skipTest("no packaging manifest present locally")
        payload = json.loads(manifests[0].read_text(encoding="utf-8"))
        for field in self.REQUIRED:
            self.assertIn(field, payload, f"manifest is missing {field}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
