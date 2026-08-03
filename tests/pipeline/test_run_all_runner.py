# -*- coding: utf-8 -*-
"""The one-command reproduction runner: ordering, forwarding and fail-closed behaviour.

The runner is a wrapper, so the properties that matter are behavioural: what
child command it builds, in what order it runs them, whether it stops when the
first one fails, and whether it can be fooled by the caller's working directory
or by a path containing a space. These tests drive the real module and observe
what it does; only one test reads source text, and it is paired with a
behavioural counterpart.
"""

from __future__ import annotations

import io
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_all  # noqa: E402


def parse(argv: list[str]):
    return run_all.main.__globals__["argparse"], run_all


def namespace(**overrides):
    """An argument namespace as argparse would build it."""
    import argparse

    defaults = dict(
        data_root=run_all.default_data_root(),
        data_pack="data_A",
        output_root=run_all.default_output_root(),
        log_dir=None,
        fresh=False,
        plan=False,
        quiet=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class DataFixture(unittest.TestCase):
    """A minimal well-formed raw-data tree, so preflight can pass."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.data = self.base / "data"
        for dataset in ("dataset1", "dataset2"):
            directory = self.data / "data_A" / dataset
            directory.mkdir(parents=True)
            (directory / "train.csv").write_text("src,dst,time\n0,1,1.0\n", encoding="utf-8")
            (directory / "test.csv").write_text("src,time\n0,2.0\n", encoding="utf-8")
        self.out = self.base / "outputs"
        self.out.mkdir()


class CommandConstruction(DataFixture):
    def test_dataset1_command(self):
        command = run_all.build_command("dataset1", namespace())
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(Path(command[1]), run_all.ENTRYPOINT)
        self.assertEqual(command[2:4], ["--dataset", "dataset1"])

    def test_dataset2_command(self):
        command = run_all.build_command("dataset2", namespace())
        self.assertEqual(command[2:4], ["--dataset", "dataset2"])

    def test_data_root_is_forwarded(self):
        command = run_all.build_command("dataset1", namespace(data_root=self.data))
        self.assertIn("--data-root", command)
        self.assertEqual(command[command.index("--data-root") + 1], str(self.data))

    def test_output_root_is_forwarded(self):
        command = run_all.build_command("dataset2", namespace(output_root=self.out))
        self.assertIn("--output-root", command)
        self.assertEqual(command[command.index("--output-root") + 1], str(self.out))

    def test_log_dir_is_forwarded_only_when_given(self):
        self.assertNotIn("--log-dir", run_all.build_command("dataset1", namespace()))
        logs = self.base / "logs"
        command = run_all.build_command("dataset1", namespace(log_dir=logs))
        self.assertEqual(command[command.index("--log-dir") + 1], str(logs))

    def test_fresh_is_forwarded_only_when_requested(self):
        self.assertNotIn("--fresh", run_all.build_command("dataset1", namespace()))
        self.assertIn("--fresh", run_all.build_command("dataset1", namespace(fresh=True)))

    def test_fresh_reaches_both_datasets(self):
        for dataset in ("dataset1", "dataset2"):
            self.assertIn("--fresh", run_all.build_command(dataset, namespace(fresh=True)))

    def test_plan_selects_the_plan_stage_and_nothing_else(self):
        command = run_all.build_command("dataset1", namespace(plan=True))
        self.assertEqual(command[command.index("--stage") + 1], "plan")
        self.assertNotIn("--fresh", command)

    def test_no_torch_diagnostic_or_historical_command_is_ever_built(self):
        forbidden = (
            "torch",
            "compare_backends",
            "train_line.py",
            "train_bpr.py",
            "canonical_run.sh",
            "reference",
            "r22_rgr",
            "r30_xte",
        )
        for dataset in ("dataset1", "dataset2"):
            for flags in (namespace(), namespace(fresh=True), namespace(plan=True)):
                blob = " ".join(run_all.build_command(dataset, flags)).lower()
                for token in forbidden:
                    # main.py is the only script the runner may name.
                    self.assertNotIn(token, blob.replace("_jt.py", ".JT"), token)

    def test_the_only_script_named_is_the_packaged_main(self):
        command = run_all.build_command("dataset1", namespace())
        scripts = [part for part in command if str(part).endswith(".py")]
        self.assertEqual(scripts, [str(run_all.ENTRYPOINT)])


class Ordering(DataFixture):
    """Ordering and failure propagation, observed by recording real calls."""

    def drive(self, argv: list[str], results: list[int]):
        calls: list[list[str]] = []
        sequence = list(results)

        def fake_run(command):
            calls.append(list(command))
            return sequence.pop(0)

        original = run_all.run
        run_all.run = fake_run
        buffer, errors = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(buffer), redirect_stderr(errors):
                code = run_all.main(argv)
        finally:
            run_all.run = original
        return code, calls, buffer.getvalue() + errors.getvalue()

    def base_argv(self, *extra: str) -> list[str]:
        return ["--data-root", str(self.data), "--output-root", str(self.out), *extra]

    def test_dataset1_runs_before_dataset2(self):
        code, calls, _ = self.drive(self.base_argv(), [0, 0])
        self.assertEqual(code, 0)
        self.assertEqual([c[c.index("--dataset") + 1] for c in calls], ["dataset1", "dataset2"])

    def test_dataset1_failure_prevents_dataset2(self):
        code, calls, text = self.drive(self.base_argv(), [7, 0])
        self.assertEqual(code, 7, "the child exit code must propagate unchanged")
        self.assertEqual(len(calls), 1, "dataset2 must not be started")
        self.assertEqual(calls[0][calls[0].index("--dataset") + 1], "dataset1")
        self.assertIn("dataset2 was NOT started", text)

    def test_dataset2_failure_propagates(self):
        code, calls, text = self.drive(self.base_argv(), [0, 3])
        self.assertEqual(code, 3)
        self.assertEqual(len(calls), 2)
        self.assertIn("FAILED dataset2", text)

    def test_no_silent_success_after_a_child_failure(self):
        for failure in (1, 2, 7, 90, 130):
            code, _, _ = self.drive(self.base_argv(), [failure, 0])
            self.assertNotEqual(code, 0, f"exit {failure} was reported as success")

    def test_plan_mode_runs_both_plans_and_no_training(self):
        code, calls, text = self.drive(self.base_argv("--plan"), [0, 0])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        for command in calls:
            self.assertEqual(command[command.index("--stage") + 1], "plan")
        self.assertIn("PLAN COMPLETE", text)
        self.assertIn("nothing was executed", text)

    def test_final_output_paths_are_reported(self):
        _, _, text = self.drive(self.base_argv(), [0, 0])
        self.assertIn(str(self.out / "members" / "dataset1.csv"), text)
        self.assertIn(str(self.out / "members" / "dataset2.csv"), text)

    def test_plan_mode_does_not_report_final_members_as_produced(self):
        _, _, text = self.drive(self.base_argv("--plan"), [0, 0])
        self.assertNotIn("REPRODUCTION COMPLETE", text)


class Preflight(DataFixture):
    def run_main(self, argv: list[str]):
        buffer, errors = io.StringIO(), io.StringIO()
        with redirect_stdout(buffer), redirect_stderr(errors):
            code = run_all.main(argv)
        return code, buffer.getvalue() + errors.getvalue()

    def test_missing_dataset1_data_fails_before_any_child(self):
        (self.data / "data_A" / "dataset1" / "train.csv").unlink()
        code, text = self.run_main(["--data-root", str(self.data), "--output-root", str(self.out)])
        self.assertEqual(code, 2)
        self.assertIn("missing official raw data", text)
        self.assertIn("dataset1", text)

    def test_missing_dataset2_data_fails_before_any_child(self):
        (self.data / "data_A" / "dataset2" / "test.csv").unlink()
        code, text = self.run_main(["--data-root", str(self.data), "--output-root", str(self.out)])
        self.assertEqual(code, 2)
        self.assertIn("missing official raw data", text)

    def test_plan_mode_does_not_require_the_raw_data(self):
        empty = self.base / "no-data"
        empty.mkdir()
        self.assertEqual(
            run_all.preflight(namespace(data_root=empty, output_root=self.out, plan=True)), []
        )

    def test_identical_data_and_output_root_is_refused(self):
        code, text = self.run_main(["--data-root", str(self.data), "--output-root", str(self.data)])
        self.assertEqual(code, 2)
        self.assertIn("same directory", text)

    def test_a_malformed_argument_is_refused(self):
        with self.assertRaises(SystemExit) as raised:
            with redirect_stderr(io.StringIO()):
                run_all.main(["--not-an-option"])
        self.assertNotEqual(raised.exception.code, 0)

    def test_a_missing_entrypoint_is_reported(self):
        original = run_all.ENTRYPOINT
        run_all.ENTRYPOINT = self.base / "absent" / "main.py"
        try:
            problems = run_all.preflight(namespace(data_root=self.data, output_root=self.out))
        finally:
            run_all.ENTRYPOINT = original
        self.assertTrue(any("entry point is missing" in p for p in problems))


class Portability(DataFixture):
    """The runner must not depend on the caller's working directory."""

    def test_runs_from_a_working_directory_outside_code(self):
        elsewhere = self.base / "some" / "other" / "place"
        elsewhere.mkdir(parents=True)
        script = textwrap.dedent(f"""
            import io, sys
            sys.path.insert(0, {str(REPO)!r})
            import run_all
            calls = []
            run_all.run = lambda c: (calls.append(c), 0)[1]
            code = run_all.main(["--data-root", {str(self.data)!r},
                                 "--output-root", {str(self.out)!r}, "--plan"])
            print("EXIT", code)
            print("CALLS", len(calls))
            print("ENTRY", calls[0][1])
        """)
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=str(elsewhere), capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr[-800:])
        self.assertIn("EXIT 0", result.stdout)
        self.assertIn("CALLS 2", result.stdout)
        self.assertIn(str(run_all.ENTRYPOINT), result.stdout)

    def test_paths_containing_spaces_are_passed_as_single_arguments(self):
        spaced_data = self.base / "raw data root"
        spaced_out = self.base / "run outputs"
        for dataset in ("dataset1", "dataset2"):
            directory = spaced_data / "data_A" / dataset
            directory.mkdir(parents=True)
            (directory / "train.csv").write_text("src,dst,time\n0,1,1.0\n", encoding="utf-8")
            (directory / "test.csv").write_text("src,time\n0,2.0\n", encoding="utf-8")
        spaced_out.mkdir()
        command = run_all.build_command(
            "dataset1", namespace(data_root=spaced_data, output_root=spaced_out)
        )
        self.assertIn(str(spaced_data), command)
        self.assertIn(str(spaced_out), command)
        # The argument list is passed to subprocess without shell quoting, so a
        # space must not split an element.
        self.assertEqual(sum(1 for part in command if part == str(spaced_data)), 1)
        self.assertEqual(
            run_all.preflight(namespace(data_root=spaced_data, output_root=spaced_out)), []
        )


class NoScientificLogic(unittest.TestCase):
    """The wrapper must stay a wrapper."""

    SOURCE = (REPO / "run_all.py").read_text(encoding="utf-8")

    def test_only_the_standard_library_is_imported(self):
        import ast

        tree = ast.parse(self.SOURCE)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        stdlib = set(getattr(sys, "stdlib_module_names", ()))
        self.assertEqual(
            imported - stdlib, set(), f"non-standard-library imports: {imported - stdlib}"
        )

    def test_no_training_or_scientific_module_is_imported(self):
        for module in (
            "jittor",
            "lightgbm",
            "numpy",
            "pandas",
            "scipy",
            "canonical_pipeline",
            "stage_contract",
            "pipeline_common",
            "train_line_jt",
            "train_bpr_jt",
            "ranker_ds1",
            "crf_promote",
        ):
            self.assertNotIn(f"import {module}", self.SOURCE, module)

    def test_it_defines_no_scientific_constant(self):
        for token in (
            "lambdarank",
            "n_estimators",
            "learning_rate",
            "num_leaves",
            "tau",
            "epochs =",
            "EMB_DIM",
            "SEED =",
            "CUT =",
            "0.20",
            "115480000",
        ):
            self.assertNotIn(token, self.SOURCE, f"scientific setting {token!r} leaked in")

    def test_the_module_stays_small(self):
        """A wrapper that grows past this is no longer only a wrapper."""
        code_lines = [
            line
            for line in self.SOURCE.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        self.assertLess(len(code_lines), 200, f"{len(code_lines)} non-comment lines")


class StagedForThePackage(unittest.TestCase):
    def test_the_runner_is_declared_for_the_official_package(self):
        sys.path.insert(0, str(REPO / "tools" / "submission"))
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "stage_package", REPO / "tools" / "submission" / "stage_package.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        staged = {source for source, _, _ in module.CODE_MEMBERS}
        self.assertIn("run_all.py", staged)
        self.assertIn("main.py", staged)


if __name__ == "__main__":
    unittest.main(verbosity=2)
