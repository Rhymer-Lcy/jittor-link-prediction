# -*- coding: utf-8 -*-
"""The dataset2 member builder must separate two different kinds of census check.

Three of the twelve census anchors -- ``rows``, ``columns``, ``sources`` -- are
properties of the physical test file alone. The other nine are functions of the
served top-1 ranking and therefore of the particular model that produced the base
matrix. Before this separation the builder asserted all twelve unconditionally,
so it could only ever run against the one frozen base: any legitimately retrained
model failed the census and no member was written.

These tests fix the boundary. Structural corruption must still abort. Shape and
candidate-set violations must still abort. A retrained base must be able to
complete the official CLI. Frozen-reference verification must still enforce every
original anchor. And the CLI must agree byte for byte with the internal decode.

Synthetic slates only; no competition data and no local artifacts.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from strategies.ds2 import cross_time_exclusivity as xte  # noqa: E402

BUILDER = REPO / "src" / "build_ds2_member.py"
COLUMNS = 6
ROWS = 900


def synthetic_test_frame(seed: int = 3, rows: int = ROWS,
                         columns: int = COLUMNS) -> pd.DataFrame:
    """A dataset2-shaped test frame dense enough to produce real violations."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "src": rng.integers(0, 40, rows),
        "time": rng.integers(0, 12, rows) * 100,
    })
    slates = rng.integers(500, 530, size=(rows, columns))
    for i in range(columns):
        frame[f"c{i + 1}"] = slates[:, i]
    return frame


def synthetic_scores(frame: pd.DataFrame, seed: int) -> np.ndarray:
    columns = sum(1 for c in frame.columns if c.startswith("c") and c[1:].isdigit())
    return np.random.default_rng(seed).random((len(frame), columns))


def write_base(path: Path, scores: np.ndarray) -> None:
    """Serialise like the production chain: headerless, %.6f, LF."""
    lines = [",".join(f"{v:.6f}" for v in row) for row in scores]
    path.write_bytes(("\n".join(lines) + "\n").encode())


def run_builder(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(BUILDER), *args],
                          capture_output=True, text=True, cwd=str(REPO))


class KeySetTest(unittest.TestCase):
    def test_the_two_key_sets_partition_the_anchors(self):
        self.assertEqual(xte.STRUCTURAL_ANCHOR_KEYS | xte.SCORE_DEPENDENT_ANCHOR_KEYS,
                         set(xte.CENSUS_ANCHORS))
        self.assertFalse(xte.STRUCTURAL_ANCHOR_KEYS & xte.SCORE_DEPENDENT_ANCHOR_KEYS)

    def test_structural_keys_are_exactly_the_score_independent_ones(self):
        # Empirical, not asserted from the constant: hold the frame fixed, vary
        # the scores, and require that only the structural keys stay put.
        frame = synthetic_test_frame()
        seen: dict[str, set] = {key: set() for key in xte.CENSUS_ANCHORS}
        for seed in range(6):
            _, info = xte.apply(synthetic_scores(frame, seed), frame)
            for key in seen:
                seen[key].add(info["census"].get(key))
        constant = {key for key, values in seen.items() if len(values) == 1}
        self.assertEqual(constant, set(xte.STRUCTURAL_ANCHOR_KEYS))

    def test_census_mismatches_key_filter_defaults_to_every_anchor(self):
        census = dict(xte.CENSUS_ANCHORS)
        self.assertEqual(xte.census_mismatches(census), {})
        census["planned_action_rows"] += 1
        self.assertEqual(set(xte.census_mismatches(census)), {"planned_action_rows"})
        self.assertEqual(
            xte.census_mismatches(census, keys=xte.STRUCTURAL_ANCHOR_KEYS), {})
        self.assertEqual(
            set(xte.census_mismatches(census, keys=xte.SCORE_DEPENDENT_ANCHOR_KEYS)),
            {"planned_action_rows"})


class StructuralValidationTest(unittest.TestCase):
    def test_a_correct_decode_has_no_structural_mismatch(self):
        frame = synthetic_test_frame()
        _, info = xte.apply(synthetic_scores(frame, 1), frame)
        self.assertEqual(xte.structural_mismatches(info["census"], frame), {})

    def test_structural_corruption_is_detected(self):
        frame = synthetic_test_frame()
        _, info = xte.apply(synthetic_scores(frame, 1), frame)
        for key, corrupt in (("rows", 1), ("columns", 3), ("sources", 999)):
            with self.subTest(key=key):
                census = dict(info["census"])
                census[key] = corrupt
                problems = xte.structural_mismatches(census, frame)
                self.assertIn(key, problems)
                self.assertEqual(problems[key]["observed"], corrupt)

    def test_a_census_from_a_different_frame_is_rejected(self):
        # The realistic corruption: a census computed against one test file is
        # checked against another. No key is hand-edited here.
        frame = synthetic_test_frame()
        other = synthetic_test_frame(seed=99, rows=ROWS // 2)
        _, info = xte.apply(synthetic_scores(frame, 1), frame)
        self.assertNotEqual(xte.structural_mismatches(info["census"], other), {})


class ShapeAndCandidateSetTest(unittest.TestCase):
    def test_score_and_candidate_shape_disagreement_still_raises(self):
        frame = synthetic_test_frame()
        wrong = synthetic_scores(frame, 1)[:, :-1]
        with self.assertRaises(ValueError):
            xte.apply(wrong, frame)

    def test_row_count_disagreement_still_raises(self):
        frame = synthetic_test_frame()
        wrong = synthetic_scores(frame, 1)[:-5]
        with self.assertRaises(ValueError):
            xte.apply(wrong, frame)

    def test_a_frame_without_candidate_columns_still_raises(self):
        frame = synthetic_test_frame()[["src", "time"]]
        with self.assertRaises(ValueError):
            xte.apply(np.zeros((len(frame), 2)), frame)

    def test_single_candidate_slates_are_refused(self):
        frame = synthetic_test_frame(columns=1)
        with self.assertRaises(ValueError):
            xte.apply(synthetic_scores(frame, 1), frame)


class CommandLineTest(unittest.TestCase):
    """End-to-end behaviour of the official build command on a retrained base."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.frame = synthetic_test_frame()
        self.test_csv = root / "test.csv"
        self.frame.to_csv(self.test_csv, index=False)
        self.scores = synthetic_scores(self.frame, 4)
        self.base = root / "base.csv"
        write_base(self.base, self.scores)
        self.output = root / "member.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_retraining_completes_the_official_cli(self):
        done = run_builder("--base", str(self.base), "--test", str(self.test_csv),
                           "--output", str(self.output))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertTrue(self.output.is_file())
        self.assertIn("structural census entries agree", done.stdout)
        self.assertIn("[NOTE]", done.stdout)
        # It must say so rather than pass silently: the divergence is recorded.
        self.assertIn("score-dependent census", done.stdout)

    def test_the_written_member_keeps_the_base_byte_length(self):
        run_builder("--base", str(self.base), "--test", str(self.test_csv),
                    "--output", str(self.output))
        self.assertEqual(self.output.stat().st_size, self.base.stat().st_size)

    def test_frozen_reference_verification_still_enforces_the_anchors(self):
        done = run_builder("--base", str(self.base), "--test", str(self.test_csv),
                           "--output", str(self.output), "--verify")
        self.assertEqual(done.returncode, 1, done.stdout + done.stderr)
        self.assertIn("census", done.stdout)
        self.assertIn("VERIFY FAILED", done.stdout)
        self.assertFalse(self.output.exists(), "no member may be written on failure")

    def test_structural_corruption_still_fails_the_cli(self):
        # A base whose column count disagrees with the test file. The decode's
        # own shape guard fires; the command must not write a member.
        narrow = self.base.parent / "narrow.csv"
        write_base(narrow, self.scores[:, :-1])
        done = run_builder("--base", str(narrow), "--test", str(self.test_csv),
                           "--output", str(self.output))
        self.assertNotEqual(done.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_row_count_violation_still_fails_the_cli(self):
        short = self.base.parent / "short.csv"
        write_base(short, self.scores[:-7])
        done = run_builder("--base", str(short), "--test", str(self.test_csv),
                           "--output", str(self.output))
        self.assertNotEqual(done.returncode, 0)
        self.assertFalse(self.output.exists())

    def test_cli_and_direct_internal_decode_agree_byte_for_byte(self):
        done = run_builder("--base", str(self.base), "--test", str(self.test_csv),
                           "--output", str(self.output))
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)

        payload = self.base.read_bytes()
        _, info = xte.apply(self.scores, self.frame)
        direct = xte.swap_score_tokens(payload, info["action_rows"], info["c1_col"],
                                       info["c2_col"], info["census"]["columns"])
        self.assertEqual(hashlib.sha256(self.output.read_bytes()).hexdigest(),
                         hashlib.sha256(direct).hexdigest())

    def test_the_decode_actually_acted_on_this_fixture(self):
        # Guards the suite against becoming vacuous: if the fixture stopped
        # producing violations, every CLI test above would pass trivially.
        _, info = xte.apply(self.scores, self.frame)
        self.assertGreater(info["actions"], 0)
        self.assertEqual(info["cells_changed"], 2 * info["actions"])


if __name__ == "__main__":
    unittest.main()
