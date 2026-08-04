# -*- coding: utf-8 -*-
"""The dataset-2 consumers must resolve run directories through the path contract.

Two consumers spelled out ``<repo>/outputs/dataset2-...`` instead of calling
``pipeline_common``. ``OUTPUTS_ROOT`` therefore reached the trainers but not
them, so a run redirected with ``--output-root`` either failed on a missing
artifact or -- the dangerous case -- silently read whatever an earlier run had
left in the default tree.

The tests below evaluate the ACTUAL source expressions at the two call sites,
extracted from the AST rather than retyped, so they cannot pass against a
re-typed copy of the fix while the shipped code still holds a literal. Three
properties are pinned:

* under the default root the new expressions resolve byte-for-byte to the old
  literals, so no existing artifact is orphaned;
* under a non-default root every required artifact resolves under that root;
* when a stale witness sits in the default tree, none of the resolved paths
  point at it.

Every static check is paired with a positive control that fires on the
historical form, so a detector that has stopped detecting fails the suite.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pipeline_common as pc  # noqa: E402

DS2_CUT = 1261958400.0
SEEDS = (42, 123, 777, 2024, 31337)

#: The paths the historical literals produced under the default output root.
#: Spelled out here exactly as they appeared in the shipped source, so the
#: identity assertion is against the real former behaviour.
HISTORICAL_DEFAULT_PATHS = {
    "bpr_serve_s42": "dataset2-bpr",
    "bpr_serve_s123": "dataset2-bpr-s123",
    "bpr_serve_s777": "dataset2-bpr-s777",
    "bpr_serve_s2024": "dataset2-bpr-s2024",
    "bpr_serve_s31337": "dataset2-bpr-s31337",
    "bpr_cut_s42": "dataset2-bpr-tmax1.26196e+09",
    "bpr_cut_s123": "dataset2-bpr-tmax1.26196e+09-s123",
    "bpr_cut_s777": "dataset2-bpr-tmax1.26196e+09-s777",
    "bpr_cut_s2024": "dataset2-bpr-tmax1.26196e+09-s2024",
    "bpr_cut_s31337": "dataset2-bpr-tmax1.26196e+09-s31337",
    "line_cut": "dataset2-novirt-tmax1.26196e+09",
    "line_serve": "dataset2-novirt",
}

CALL_SITES = {
    # module, the callee name at the site, (bpr argument index, line argument index)
    "ds2_basket_featurizer.py": ("build_features", 3, 4),
    "ds2_mf_basket_pack.py": ("bag.build_features", 3, 4),
}


def callee_name(node: ast.Call) -> str:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
        return f"{target.value.id}.{target.attr}"
    return ""


def call_site_arguments(module: str) -> tuple[ast.expr, ast.expr]:
    """The two run-directory argument expressions, taken from the real source."""
    name, bpr_index, line_index = CALL_SITES[module]
    tree = ast.parse((SRC / module).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and callee_name(node) == name:
            if len(node.args) > line_index:
                return node.args[bpr_index], node.args[line_index]
    raise AssertionError(f"no {name}(...) call site found in {module}")


def evaluate(expression: ast.expr, **names):
    """Evaluate a source expression against the real pipeline_common."""
    namespace = {"tl": pc, "pc": pc, "SEEDS": SEEDS, "CUT": DS2_CUT, **names}
    return eval(
        compile(ast.Expression(ast.fix_missing_locations(expression)), "<call-site>", "eval"),
        namespace,
    )


class OutputsRootOverride:
    """Set OUTPUTS_ROOT for the duration of a block and restore it exactly."""

    def __init__(self, value: str | None) -> None:
        self.value = value

    def __enter__(self):
        self.previous = os.environ.get("OUTPUTS_ROOT")
        if self.value is None:
            os.environ.pop("OUTPUTS_ROOT", None)
        else:
            os.environ["OUTPUTS_ROOT"] = self.value
        return self

    def __exit__(self, *exc):
        if self.previous is None:
            os.environ.pop("OUTPUTS_ROOT", None)
        else:
            os.environ["OUTPUTS_ROOT"] = self.previous
        return False


class DefaultPathIdentity(unittest.TestCase):
    """The correction must orphan no existing artifact."""

    def test_helper_paths_equal_the_historical_literals(self):
        with OutputsRootOverride(None):
            default = pc.PROJECT_ROOT / "outputs"
            cases = {
                "bpr_serve_s42": pc.bpr_run_dir("dataset2", seed=42),
                "bpr_serve_s123": pc.bpr_run_dir("dataset2", seed=123),
                "bpr_cut_s42": pc.bpr_run_dir("dataset2", seed=42, time_max=DS2_CUT),
                "bpr_cut_s123": pc.bpr_run_dir("dataset2", seed=123, time_max=DS2_CUT),
                "line_cut": pc.line_run_dir("dataset2", time_max=DS2_CUT),
            }
            for key, produced in cases.items():
                self.assertEqual(
                    produced, default / HISTORICAL_DEFAULT_PATHS[key], f"{key} moved: {produced}"
                )

    def test_every_seed_matches_for_both_bpr_families(self):
        with OutputsRootOverride(None):
            default = pc.PROJECT_ROOT / "outputs"
            for seed in SEEDS:
                serve = pc.bpr_run_dir("dataset2", seed=seed)
                cut = pc.bpr_run_dir("dataset2", seed=seed, time_max=DS2_CUT)
                self.assertEqual(serve, default / HISTORICAL_DEFAULT_PATHS[f"bpr_serve_s{seed}"])
                self.assertEqual(cut, default / HISTORICAL_DEFAULT_PATHS[f"bpr_cut_s{seed}"])

    def test_the_real_call_site_expressions_reproduce_the_defaults(self):
        """Evaluated from the shipped source, not from a retyped copy."""
        with OutputsRootOverride(None):
            default = pc.PROJECT_ROOT / "outputs"
            bpr_expr, line_expr = call_site_arguments("ds2_basket_featurizer.py")
            self.assertEqual(
                [Path(p) for p in evaluate(bpr_expr)],
                [default / HISTORICAL_DEFAULT_PATHS[f"bpr_cut_s{s}"] for s in SEEDS],
            )
            self.assertEqual(
                Path(evaluate(line_expr)), default / HISTORICAL_DEFAULT_PATHS["line_cut"]
            )

            bpr_expr, line_expr = call_site_arguments("ds2_mf_basket_pack.py")
            self.assertEqual(
                [Path(p) for p in evaluate(bpr_expr)],
                [default / HISTORICAL_DEFAULT_PATHS[f"bpr_serve_s{s}"] for s in SEEDS],
            )
            self.assertEqual(
                Path(evaluate(line_expr)), default / HISTORICAL_DEFAULT_PATHS["line_serve"]
            )


class NonDefaultOutputRoot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "isolated-outputs"
        self.root.mkdir(parents=True)
        self.addCleanup(self._tmp.cleanup)

    def resolved(self, module: str) -> list[Path]:
        bpr_expr, line_expr = call_site_arguments(module)
        with OutputsRootOverride(str(self.root)):
            return [Path(p) for p in evaluate(bpr_expr)] + [Path(evaluate(line_expr))]

    def test_featurizer_resolves_every_artifact_under_the_requested_root(self):
        paths = self.resolved("ds2_basket_featurizer.py")
        self.assertEqual(len(paths), 6)
        for path in paths:
            self.assertTrue(
                str(path).startswith(str(self.root)), f"escaped the requested output root: {path}"
            )

    def test_mf_pack_resolves_every_artifact_under_the_requested_root(self):
        paths = self.resolved("ds2_mf_basket_pack.py")
        self.assertEqual(len(paths), 6)
        for path in paths:
            self.assertTrue(
                str(path).startswith(str(self.root)), f"escaped the requested output root: {path}"
            )

    def test_no_resolved_path_touches_the_repository_default_tree(self):
        default = pc.PROJECT_ROOT / "outputs"
        for module in CALL_SITES:
            for path in self.resolved(module):
                self.assertFalse(
                    str(path).startswith(str(default)), f"{module} reached the default tree: {path}"
                )


class StaleDefaultTreeTrap(unittest.TestCase):
    """The dangerous case: a populated default tree beside an isolated root."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.requested = base / "requested"
        self.default = base / "default"
        # The intended artifact lives only under the requested root; a DISTINCT
        # stale witness lives only in the default tree.
        for name in HISTORICAL_DEFAULT_PATHS.values():
            (self.requested / name).mkdir(parents=True, exist_ok=True)
            (self.requested / name / "witness.txt").write_text("intended", encoding="utf-8")
            (self.default / name).mkdir(parents=True, exist_ok=True)
            (self.default / name / "witness.txt").write_text("stale", encoding="utf-8")

    def witnesses(self, module: str) -> list[str]:
        bpr_expr, line_expr = call_site_arguments(module)
        with OutputsRootOverride(str(self.requested)):
            paths = [Path(p) for p in evaluate(bpr_expr)] + [Path(evaluate(line_expr))]
        return [
            (p / "witness.txt").read_text(encoding="utf-8")
            for p in paths
            if (p / "witness.txt").is_file()
        ]

    def test_only_the_intended_witness_is_read(self):
        for module in CALL_SITES:
            found = self.witnesses(module)
            self.assertEqual(len(found), 6, f"{module}: resolved {len(found)} artifacts")
            self.assertEqual(
                set(found), {"intended"}, f"{module} read a stale default-tree artifact"
            )

    def test_the_trap_would_catch_the_historical_implementation(self):
        """Anti-vacuity: the pre-fix expression reads the stale witness."""
        historical = [self.default / HISTORICAL_DEFAULT_PATHS[f"bpr_cut_s{s}"] for s in SEEDS]
        historical.append(self.default / HISTORICAL_DEFAULT_PATHS["line_cut"])
        stale = [(p / "witness.txt").read_text(encoding="utf-8") for p in historical]
        self.assertEqual(
            set(stale), {"stale"}, "the fixture does not actually distinguish the two trees"
        )


class MissingArtifactFailsClosed(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.requested = base / "requested"  # deliberately empty
        self.requested.mkdir(parents=True)
        self.default = base / "default"
        for name in HISTORICAL_DEFAULT_PATHS.values():
            (self.default / name).mkdir(parents=True, exist_ok=True)
            (self.default / name / "bpr_emb.npy").write_bytes(b"stale")

    def test_absent_artifact_is_not_satisfied_from_the_default_tree(self):
        import numpy as np

        for module in CALL_SITES:
            bpr_expr, _ = call_site_arguments(module)
            with OutputsRootOverride(str(self.requested)):
                paths = [Path(p) for p in evaluate(bpr_expr)]
            for path in paths:
                artifact = path / "bpr_emb.npy"
                self.assertFalse(
                    artifact.exists(),
                    f"{module}: resolved to an artifact that only the "
                    f"default tree provides: {artifact}",
                )
                with self.assertRaises((FileNotFoundError, OSError)):
                    np.load(artifact)


class StaticGuard(unittest.TestCase):
    """Detect reintroduction of the literal form at the affected sites."""

    HISTORICAL = (
        '"dataset2-bpr-tmax1.26196e+09"',
        '"dataset2-novirt-tmax1.26196e+09"',
        '("dataset2-bpr" + (f"-s{s}"',
    )

    def test_no_call_site_reintroduces_a_hard_coded_run_directory(self):
        for module in CALL_SITES:
            source = (SRC / module).read_text(encoding="utf-8")
            code = "\n".join(
                line for line in source.splitlines() if not line.strip().startswith("#")
            )
            for pattern in self.HISTORICAL:
                self.assertNotIn(pattern, code, f"{module} reintroduced {pattern}")

    def test_no_canonical_ds2_consumer_builds_an_outputs_root_literal(self):
        for module in list(CALL_SITES) + ["ranker_basket_ds2.py"]:
            source = (SRC / module).read_text(encoding="utf-8")
            code = "\n".join(
                line for line in source.splitlines() if not line.strip().startswith("#")
            )
            for pattern in ('REPO / "outputs"', 'PROJECT_ROOT / "outputs"'):
                self.assertNotIn(pattern, code, f"{module} still builds {pattern}")

    def test_the_guard_fires_on_the_historical_form(self):
        """Anti-vacuity: a detector that no longer detects must fail."""
        historical = (
            'tdir = REPO / "outputs"\n'
            "Xtr = build_features(split0, CUT, train_q,\n"
            '    [tdir / ("dataset2-bpr-tmax1.26196e+09" + (f"-s{s}" if s != 42 else "")) '
            "for s in SEEDS],\n"
            '    tdir / "dataset2-novirt-tmax1.26196e+09",\n'
            ")\n"
        )
        hits = [p for p in self.HISTORICAL if p in historical]
        self.assertGreaterEqual(len(hits), 2, "the guard patterns no longer match the defect")
        self.assertIn('REPO / "outputs"', historical)

    def test_the_call_site_extractor_actually_finds_both_sites(self):
        """Anti-vacuity for the AST extraction the behavioural tests depend on."""
        for module in CALL_SITES:
            bpr_expr, line_expr = call_site_arguments(module)
            self.assertIsInstance(bpr_expr, (ast.ListComp, ast.List))
            self.assertIsInstance(line_expr, ast.Call)
            self.assertIn("run_dir", ast.dump(line_expr))
            self.assertIn("run_dir", ast.dump(bpr_expr))


class BehaviouralFeaturizerRouting(unittest.TestCase):
    """Run the real build_or_load_features and capture what it actually passes.

    Source-level evidence is not enough on its own: this executes the shipped
    function in a subprocess against a tiny synthetic dataset and records the
    run directories it hands to build_features.
    """

    def test_the_real_function_passes_requested_root_directories(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            data = base / "data" / "data_A" / "dataset2"
            data.mkdir(parents=True)
            rng = __import__("numpy").random.default_rng(3)
            train = pd.DataFrame(
                {
                    "src": rng.integers(0, 20, 400),
                    "dst": rng.integers(20, 120, 400),
                    "time": sorted(rng.uniform(1.20e9, 1.32e9, 400)),
                }
            )
            train.to_csv(data / "train.csv", index=False)
            test = pd.DataFrame(
                {
                    "src": rng.integers(0, 20, 12),
                    "time": [1.33e9] * 12,
                    **{f"c{column}": rng.integers(20, 120, 12) for column in range(1, 101)},
                }
            )
            test.to_csv(data / "test.csv", index=False)

            requested = base / "isolated"
            requested.mkdir()
            script = (
                "import json, sys\n"
                f"sys.path.insert(0, {str(SRC)!r})\n"
                "import ds2_basket_featurizer as bag\n"
                "seen = {}\n"
                "def capture(split0, cut, queries, bpr_dirs, line_dir, *a, **k):\n"
                "    seen['bpr'] = [str(p) for p in bpr_dirs]\n"
                "    seen['line'] = str(line_dir)\n"
                "    raise SystemExit(json.dumps(seen))\n"
                "bag.build_features = capture\n"
                f"bag.CACHE = __import__('pathlib').Path({str(base / 'cache')!r})\n"
                "try:\n"
                "    bag.build_or_load_features(force=False, max_queries=0)\n"
                "except SystemExit as done:\n"
                "    print(done.code)\n"
            )
            environment = dict(
                os.environ,
                DATASET="dataset2",
                DATA_ROOT=str(base / "data"),
                OUTPUTS_ROOT=str(requested),
            )
            environment.pop("PYTHONPATH", None)
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=environment,
                cwd=str(REPO),
            )
            self.assertEqual(result.returncode, 0, result.stderr[-1500:])
            captured = json.loads(result.stdout.strip().splitlines()[-1])

        self.assertEqual(len(captured["bpr"]), 5)
        for path in captured["bpr"] + [captured["line"]]:
            self.assertTrue(
                path.startswith(str(requested)),
                f"the real function used {path}, outside the requested root",
            )
            self.assertNotIn(str(pc.PROJECT_ROOT / "outputs"), path)


if __name__ == "__main__":
    unittest.main(verbosity=2)
