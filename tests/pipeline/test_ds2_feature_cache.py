# -*- coding: utf-8 -*-
"""The dataset2 train-feature cache under the completion contract.

Two specific behaviours were removed and are pinned here as removed: a bare
``np.savez`` straight to the final name, and a reuse decision made from
``feat.is_file()``. Both are checked against the real source rather than
described, because the module binds its dataset at import and cannot be imported
inside a suite that has already selected the other one.

The generic NPZ validator is exercised in ``test_stage_contract.py``; what these
tests cover is that this producer is wired to it.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FeaturizerUsesTheContract(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (SRC / "ds2_basket_featurizer.py").read_text(encoding="utf-8")

    def test_the_cache_is_written_through_atomic_output(self):
        self.assertIn("sc.atomic_output(feat)", self.source)

    def test_no_bare_savez_to_the_final_path_remains(self):
        self.assertNotIn("np.savez(feat", self.source)

    def test_reuse_is_decided_by_the_contract_not_by_file_existence(self):
        self.assertNotIn("if feat.is_file() and not force", self.source)
        self.assertIn("sc.evaluate(spec, repo=REPO)", self.source)
        self.assertIn('decision.action == "STOP"', self.source)

    def test_the_cache_publishes_a_completion_record(self):
        self.assertIn("sc.complete_stage(built", self.source)

    def test_reuse_revalidates_the_arrays_and_does_not_trust_the_hash_alone(self):
        self.assertIn("sc.validate_npz_cache(reused, feat)", self.source)

    def test_force_quarantines_rather_than_deletes(self):
        self.assertRegex(self.source, r"sc\.quarantine\(\s*stale")
        self.assertNotIn("feat.unlink()", self.source)

    def test_the_geometry_keys_are_excluded_from_the_reuse_digest(self):
        constants = self.module_constants()
        for key in constants["CACHE_GEOMETRY_KEYS"]:
            self.assertNotIn(
                key,
                constants["CACHE_DIGEST_KEYS"],
                "geometry measurable only after the run cannot gate reuse",
            )
        for key in ("cut", "num_entity", "negatives_per_sample"):
            self.assertIn(key, constants["CACHE_DIGEST_KEYS"])
        self.assertEqual(
            set(constants["CACHE_KEYS"]),
            {"Xf", "yf", "lens", "qsrc_tr", "qt_tr", "qorig_tr", "cands_concat", "num_entity"},
        )

    def test_the_reuse_digest_binds_the_official_inputs(self):
        """The cache must be invalidated by a change to train.csv or test.csv."""
        self.assertIn("inputs=[tl.train_csv, tl.test_csv]", self.source)

    def test_the_constant_scanner_would_notice_a_rename(self):
        """Anti-vacuity: the static reader really does read these names."""
        constants = self.module_constants()
        self.assertNotIn("CACHE_NOT_A_REAL_CONSTANT", constants)
        self.assertTrue(all(isinstance(v, tuple) for v in constants.values()))

    def module_constants(self) -> dict:
        tree = ast.parse(self.source)
        found = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name.startswith("CACHE_"):
                    try:
                        found[name] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
        for required in ("CACHE_KEYS", "CACHE_GEOMETRY_KEYS", "CACHE_DIGEST_KEYS"):
            self.assertIn(required, found, "the cache contract constants moved")
        return found


class ProducersPublishAtomically(unittest.TestCase):
    """Every stage that writes a score matrix now stages it before publishing.

    The formatter, the float format and therefore the bytes are unchanged; only
    the moment the final name starts existing moved.
    """

    WRITERS = {
        "ranker_ds1.py": "sc.atomic_output(save_path)",
        "ranker_basket_ds2.py": "sc.atomic_output(OUT)",
        "ds2_mf_basket_pack.py": "sc.atomic_output(args.out)",
    }

    def test_each_score_matrix_writer_stages_its_output(self):
        for name, expected in self.WRITERS.items():
            source = (SRC / name).read_text(encoding="utf-8")
            self.assertIn(expected, source, f"{name} publishes non-atomically")

    def test_no_writer_calls_to_csv_on_its_final_path(self):
        for name in self.WRITERS:
            source = (SRC / name).read_text(encoding="utf-8")
            for forbidden in (".to_csv(save_path", ".to_csv(OUT,", ".to_csv(args.out"):
                self.assertNotIn(forbidden, source, f"{name} still writes in place")

    def test_the_float_format_is_untouched(self):
        """The bytes must not change: same formatter, same precision, no header."""
        for name in self.WRITERS:
            source = (SRC / name).read_text(encoding="utf-8")
            self.assertIn('float_format="%.6f"', source, name)
            self.assertIn("header=False", source, name)

    def test_rankers_resolve_their_directory_through_the_path_contract(self):
        self.assertIn(
            'tl.ranker_dir("dataset1")', (SRC / "ranker_ds1.py").read_text(encoding="utf-8")
        )
        self.assertIn(
            'tl.ranker_dir("dataset2")', (SRC / "ranker_basket_ds2.py").read_text(encoding="utf-8")
        )

    def test_trainers_resolve_their_data_directory_through_the_path_contract(self):
        for name in ("train_line_jt.py", "train_bpr_jt.py"):
            source = (SRC / name).read_text(encoding="utf-8")
            self.assertIn("pc.data_dir(DATASET, DATA_PACK)", source, name)
            self.assertNotIn('PROJECT_ROOT / "data" / DATA_PACK / DATASET', source, name)


class RootOverrides(unittest.TestCase):
    """--data-root and --output-root have to reach producer and consumer alike."""

    def test_the_output_root_variable_wins_over_an_explicit_root_argument(self):
        import importlib
        import os

        import pipeline_common as pc

        importlib.reload(pc)
        self.assertEqual(pc.outputs_root(Path("X")), Path("X") / "outputs")
        os.environ["OUTPUTS_ROOT"] = str(Path("Y") / "elsewhere")
        try:
            # The trainers pass their own PROJECT_ROOT, so the variable has to
            # win or --output-root would be silently ignored by every producer.
            self.assertEqual(pc.outputs_root(Path("X")), Path("Y") / "elsewhere")
        finally:
            os.environ.pop("OUTPUTS_ROOT", None)
        self.assertEqual(pc.outputs_root(Path("X")), Path("X") / "outputs")

    def test_the_data_root_variable_redirects_the_data_directory(self):
        import os

        import pipeline_common as pc

        default = pc.data_dir("dataset1")
        self.assertTrue(str(default).endswith(str(Path("data") / "data_A" / "dataset1")))
        os.environ["DATA_ROOT"] = str(Path("Z") / "official")
        try:
            self.assertEqual(
                pc.data_dir("dataset1"), Path("Z") / "official" / "data_A" / "dataset1"
            )
        finally:
            os.environ.pop("DATA_ROOT", None)
        self.assertEqual(pc.data_dir("dataset1"), default)


if __name__ == "__main__":
    unittest.main(verbosity=2)
