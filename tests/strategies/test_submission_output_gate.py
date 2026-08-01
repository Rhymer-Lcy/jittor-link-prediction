# -*- coding: utf-8 -*-
"""No member may be serialised unless it is submittable.

A canonical dataset1 reconstruction produced a member whose values reached
-4.7e24. It had the right shape, the right row count and no NaN, so every check
the builders performed at the time accepted it and it was written to disk. The
gate closes that hole: shape, finiteness and the `[0, 1]` range are asserted on
the final matrix before serialisation, and the serialised bytes are structurally
re-checked before the file is renamed into place.

The gate reports and aborts. It never clamps, rescales or otherwise repairs the
values -- a rejected member is a signal about the chain that produced it, and
silently fixing it would destroy that evidence and change the ranking.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from strategies.shared.frozen_ops import (  # noqa: E402
    read_score_matrix,
    validate_submission_matrix,
    verify_submission_file,
    write_score_matrix,
)


def good(rows: int = 8, cols: int = 100) -> np.ndarray:
    return np.random.default_rng(0).random((rows, cols))


class MatrixGateAcceptsValidMembersTest(unittest.TestCase):
    def test_a_valid_member_passes(self):
        validate_submission_matrix(good())

    def test_the_closed_interval_endpoints_are_allowed(self):
        m = good()
        m[0, 0], m[0, 1] = 0.0, 1.0
        validate_submission_matrix(m)

    def test_a_matching_expected_shape_passes(self):
        validate_submission_matrix(good(8, 100), (8, 100))


class MatrixGateRejectsTest(unittest.TestCase):
    def reject(self, matrix, fragment, shape=None):
        with self.assertRaises(ValueError) as caught:
            validate_submission_matrix(matrix, shape)
        self.assertIn(fragment, str(caught.exception))

    def test_wrong_row_count(self):
        self.reject(good(7, 100), "shape", (8, 100))

    def test_wrong_column_count(self):
        self.reject(good(8, 99), "shape", (8, 100))

    def test_not_a_two_dimensional_matrix(self):
        self.reject(np.zeros(100), "ndim")

    def test_nan(self):
        m = good(); m[1, 1] = np.nan
        self.reject(m, "NaN")

    def test_positive_infinity(self):
        m = good(); m[2, 2] = np.inf
        self.reject(m, "+Inf")

    def test_negative_infinity(self):
        m = good(); m[3, 3] = -np.inf
        self.reject(m, "-Inf")

    def test_values_above_one(self):
        m = good(); m[4, 4] = 1.0000001
        self.reject(m, "outside [0, 1]")

    def test_values_below_zero(self):
        m = good(); m[5, 5] = -1e-9
        self.reject(m, "outside [0, 1]")

    def test_the_actual_observed_defect_value(self):
        m = good(); m[6, 6] = -4.723676999999999776e24
        self.reject(m, "outside [0, 1]")

    def test_the_gate_does_not_repair_the_matrix(self):
        m = good(); m[7, 7] = -5.0
        before = m.copy()
        with self.assertRaises(ValueError):
            validate_submission_matrix(m)
        self.assertTrue(np.array_equal(m, before), "the gate must not clamp")


class SerialisedFileGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_correctly_written_member_verifies(self):
        p = self.root / "m.csv"
        write_score_matrix(good(8, 100), p)
        verify_submission_file(p, (8, 100))

    def test_a_truncated_file_is_rejected(self):
        p = self.root / "m.csv"
        write_score_matrix(good(8, 100), p)
        payload = p.read_bytes()
        p.write_bytes(payload[: len(payload) // 2])
        with self.assertRaises(ValueError) as caught:
            verify_submission_file(p, (8, 100))
        self.assertIn("serialised rows", str(caught.exception))

    def test_a_ragged_row_is_rejected(self):
        p = self.root / "m.csv"
        write_score_matrix(good(8, 100), p)
        lines = p.read_bytes().split(b"\n")
        lines[3] = lines[3] + b",0.5"
        p.write_bytes(b"\n".join(lines))
        with self.assertRaises(ValueError) as caught:
            verify_submission_file(p, (8, 100))
        self.assertIn("field counts", str(caught.exception))


class AtomicWriteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_write_leaves_no_part_file_behind(self):
        p = self.root / "m.csv"
        write_score_matrix(good(8, 100), p)
        self.assertTrue(p.is_file())
        self.assertEqual(list(self.root.glob("*.part")), [])

    def test_the_accepted_serialisation_is_unchanged_by_the_atomic_write(self):
        # CRLF and %.6f are load-bearing for the accepted dataset1 bytes.
        p = self.root / "m.csv"
        m = np.array([[0.5] * 100, [0.25] * 100])
        write_score_matrix(m, p)
        payload = p.read_bytes()
        self.assertEqual(payload.count(b"\r\n"), 2)
        self.assertTrue(payload.split(b"\r\n")[0].startswith(b"0.500000,"))
        self.assertTrue(np.allclose(read_score_matrix(p), m))

    def test_a_structurally_bad_write_leaves_no_file_at_the_real_name(self):
        # verify_submission_file runs on the .part; a failure must remove it and
        # must not create the destination.
        p = self.root / "m.csv"
        m = good(8, 100)
        original = write_score_matrix.__globals__["verify_submission_file"]
        try:
            write_score_matrix.__globals__["verify_submission_file"] = \
                lambda *a, **k: (_ for _ in ()).throw(ValueError("forced"))
            with self.assertRaises(ValueError):
                write_score_matrix(m, p)
        finally:
            write_score_matrix.__globals__["verify_submission_file"] = original
        self.assertFalse(p.exists(), "no member may appear under the real name")
        self.assertEqual(list(self.root.glob("*.part")), [])


class BuilderCommandLineTest(unittest.TestCase):
    """The gate must actually be wired into both member builders."""

    def test_both_builders_import_and_call_the_gate(self):
        for name in ("build_ds1_member.py", "build_ds2_member.py"):
            with self.subTest(builder=name):
                text = (REPO / "src" / name).read_text(encoding="utf-8")
                self.assertIn("validate_submission_matrix", text)
                self.assertIn("OUTPUT GATE REJECTED", text)

    def test_the_gate_runs_before_the_FINAL_write_in_both_builders(self):
        # Ordering is the whole point: a gate that fires after serialisation has
        # already published the bad member. Checked structurally rather than by
        # driving the CLI, which needs a full-size competition test file.
        #
        # Only the FINAL write is gated. build_ds1_member also dumps optional
        # per-stage CSVs under --stage-dir; those are diagnostic artifacts and are
        # deliberately NOT gated, because an out-of-range intermediate is exactly
        # what someone debugging the chain needs to see on disk.
        import ast

        def final_write_line(main, writer, target):
            for node in ast.walk(main):
                if (isinstance(node, ast.Call)
                        and getattr(node.func, "id", getattr(node.func, "attr", "")) == writer
                        and any(ast.unparse(a) == target for a in node.args)):
                    return node.lineno
            return None

        for name, writer, target in (
                ("build_ds1_member.py", "write_score_matrix", "args.output"),
                ("build_ds2_member.py", "swap_score_tokens", "payload")):
            with self.subTest(builder=name):
                tree = ast.parse((REPO / "src" / name).read_text(encoding="utf-8"))
                main = next(n for n in ast.walk(tree)
                            if isinstance(n, ast.FunctionDef) and n.name == "main")
                gate = [n.lineno for n in ast.walk(main) if isinstance(n, ast.Call)
                        and getattr(n.func, "id", getattr(n.func, "attr", "")) ==
                        "validate_submission_matrix"]
                write = final_write_line(main, writer, target)
                self.assertTrue(gate, f"{name} never calls the gate")
                self.assertIsNotNone(write, f"{name}: no final {writer}({target}) found")
                self.assertLess(min(gate), write,
                                f"{name}: the gate must precede the final {writer}")

    def test_the_ds1_stage_dumps_are_deliberately_not_gated(self):
        # Pins the decision above so a later change cannot quietly start gating
        # (and therefore suppressing) the diagnostic stage CSVs.
        import ast
        tree = ast.parse((REPO / "src" / "build_ds1_member.py").read_text(encoding="utf-8"))
        main = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        stage_writes = [n for n in ast.walk(main) if isinstance(n, ast.Call)
                        and getattr(n.func, "id", "") == "write_score_matrix"
                        and any(ast.unparse(a) == "stage_path" for a in n.args)]
        self.assertEqual(len(stage_writes), 1,
                         "the optional per-stage dump must still exist and stay ungated")

    def test_both_builders_return_nonzero_without_writing_on_rejection(self):
        for name in ("build_ds1_member.py", "build_ds2_member.py"):
            with self.subTest(builder=name):
                text = (REPO / "src" / name).read_text(encoding="utf-8")
                block = text[text.index("validate_submission_matrix("):]
                block = block[: block.index("\n\n")] if "\n\n" in block else block
                self.assertIn("return 1", block)
                self.assertIn("no CSV was written", block)


if __name__ == "__main__":
    unittest.main()
