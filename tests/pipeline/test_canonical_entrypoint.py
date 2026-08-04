# -*- coding: utf-8 -*-
"""The canonical entrypoint: it executes, it is Jittor-only, and it fails closed.

``main.py`` used to print the commands a human should run. These tests pin the
properties of the replacement: the graph is real and ordered, stages are
dispatched as subprocesses whose failures propagate, reuse happens only through
a validated completion record, and neither the entrypoint nor anything it can
reach imports PyTorch.

Dispatch is exercised for real -- a subprocess runs, writes a staged file, is
validated, published and recorded -- using trivial test doubles rather than a
GPU trainer. Nothing here trains a model or requires the competition data.
"""

from __future__ import annotations

import ast
import io
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import canonical_pipeline as cp  # noqa: E402
import main as entrypoint  # noqa: E402
import stage_contract as sc  # noqa: E402


def synthetic_dataset(root: Path, dataset: str, *, rows: int = 6, candidates: int = 100) -> Path:
    """A tiny, well-formed stand-in for one official data pack.

    Small enough to be free, real enough that the graph builder measures its
    geometry the same way it measures the competition files.
    """
    directory = root / "data_A" / dataset
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"src": [0, 1, 2, 3], "dst": [4, 5, 6, 7], "time": [10.0, 20.0, 30.0, 40.0]}
    ).to_csv(directory / "train.csv", index=False)
    frame = {"src": list(range(rows)), "time": [50.0] * rows}
    for column in range(1, candidates + 1):
        frame[f"c{column}"] = [column % 8] * rows
    pd.DataFrame(frame).to_csv(directory / "test.csv", index=False)
    return directory


class GraphShape(unittest.TestCase):
    """The graph is the maintained one: same stages, same paths, same order."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.data = self.root / "data"
        synthetic_dataset(self.data, "dataset1")
        synthetic_dataset(self.data, "dataset2")

    def ctx(self, dataset: str, **overrides) -> cp.RunContext:
        defaults = dict(
            dataset=dataset, data_root=self.data, outputs_root=self.root / "outputs", repo=REPO
        )
        defaults.update(overrides)
        return cp.RunContext(**defaults)

    def test_dataset1_graph_is_the_maintained_sixteen_stages(self):
        stages = cp.build_stages(self.ctx("dataset1"))
        self.assertEqual(
            [s.stage_id for s in stages],
            [
                "ds1_line_full",
                "ds1_line_cut",
                "ds1_bpr_serve_s42",
                "ds1_bpr_cut_s42",
                "ds1_bpr_serve_s123",
                "ds1_bpr_cut_s123",
                "ds1_bpr_serve_s777",
                "ds1_bpr_cut_s777",
                "ds1_bpr_serve_s2024",
                "ds1_bpr_cut_s2024",
                "ds1_bpr_serve_s31337",
                "ds1_bpr_cut_s31337",
                "ds1_bpr_innov_serve",
                "ds1_bpr_innov_cut",
                "ds1_ranker",
                "ds1_member",
            ],
        )

    def test_dataset2_graph_runs_ranker_pack_crf_then_member(self):
        stages = [s.stage_id for s in cp.build_stages(self.ctx("dataset2"))]
        self.assertEqual(
            stages[-5:],
            ["ds2_ranker", "ds2_mf_pack", "ds2_ds1_passthrough", "ds2_crf", "ds2_member"],
        )
        self.assertEqual(len(stages), 17)

    def test_stage_outputs_come_from_the_path_contract(self):
        """Every artifact path is a pipeline_common expression, not a literal."""
        import pipeline_common as pc

        outputs = self.root / "outputs"
        os.environ["OUTPUTS_ROOT"] = str(outputs)
        try:
            expected = {
                "ds1_line_full": pc.line_run_dir("dataset1") / "line_latest_emb.csv",
                "ds1_line_cut": pc.line_run_dir("dataset1", time_max=float(cp.DS1_CUT))
                / "line_latest_emb.csv",
                "ds1_bpr_serve_s777": pc.bpr_run_dir("dataset1", seed=777, tau_frac=0.25)
                / "bpr_emb.npy",
                "ds1_ranker": pc.ranker_dir("dataset1") / "result_ranker.csv",
            }
        finally:
            os.environ.pop("OUTPUTS_ROOT", None)
        by_id = {s.stage_id: s for s in cp.build_stages(self.ctx("dataset1"))}
        for stage_id, path in expected.items():
            self.assertEqual(by_id[stage_id].output, path)

    def test_the_graph_honours_an_external_output_root(self):
        elsewhere = self.root / "somewhere-else"
        stages = cp.build_stages(self.ctx("dataset1", outputs_root=elsewhere))
        for spec in stages:
            self.assertTrue(
                str(spec.output).startswith(str(elsewhere.resolve())),
                f"{spec.stage_id} escaped --output-root: {spec.output}",
            )

    def test_the_graph_honours_an_external_data_root(self):
        stages = cp.build_stages(self.ctx("dataset1"))
        raw = {Path(p) for spec in stages for p in spec.inputs}
        official = {p for p in raw if p.name in ("train.csv", "test.csv")}
        self.assertTrue(official)
        for path in official:
            self.assertTrue(str(path).startswith(str(self.data.resolve())), path)

    def test_no_stage_reads_a_frozen_member_or_a_reference_prediction(self):
        """Reference artifacts are never computational inputs."""
        forbidden = (
            "reference",
            "submissions",
            "r22_rgr_main_d9.zip",
            "r30_xte_final.zip",
            "artifacts_durable",
            "result_ensemble.csv",
        )
        for dataset in ("dataset1", "dataset2"):
            for spec in cp.build_stages(self.ctx(dataset)):
                blob = " ".join([*map(str, spec.inputs), *map(str, spec.command)]).lower()
                for token in forbidden:
                    self.assertNotIn(token, blob, f"{spec.stage_id} references {token}")

    def test_every_stage_declares_inputs_config_and_code_identity(self):
        for dataset in ("dataset1", "dataset2"):
            for spec in cp.build_stages(self.ctx(dataset)):
                self.assertTrue(spec.code_files, f"{spec.stage_id} has no code identity")
                self.assertTrue(spec.config, f"{spec.stage_id} has no configuration")
                self.assertIsNotNone(spec.validator, f"{spec.stage_id} has no validator")
                for path in spec.code_files:
                    self.assertTrue(Path(path).is_file(), f"{spec.stage_id}: {path}")

    def test_trainers_declare_their_epoch_budget(self):
        for spec in cp.build_stages(self.ctx("dataset1")):
            if "line" in spec.stage_id:
                self.assertEqual(spec.requested_units, cp.LINE_EPOCHS)
            elif "bpr" in spec.stage_id:
                self.assertEqual(spec.requested_units, cp.BPR_EPOCHS)
            else:
                self.assertIsNone(spec.requested_units)

    def test_the_plan_reports_a_disposition_for_every_stage(self):
        text = cp.plan(self.ctx("dataset1"))
        self.assertIn("ds1_member", text)
        self.assertEqual(text.count("status :"), 16)
        self.assertIn("RUN (CLEAN_SLATE)", text)

    def test_the_dataset2_plan_renders_without_a_dataset1_member(self):
        text = cp.plan(self.ctx("dataset2"))
        self.assertIn("ds2_crf", text)
        self.assertIn("UNRESOLVED", text)  # the passthrough cannot resolve yet

    def test_an_unknown_dataset_is_refused(self):
        with self.assertRaises(cp.PipelineError):
            cp.build_stages(
                cp.RunContext(
                    dataset="dataset3", data_root=self.data, outputs_root=self.root / "outputs"
                )
            )


class Dispatch(unittest.TestCase):
    """Real subprocess dispatch, with trivial stand-ins for the heavy stages."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.ctx = cp.RunContext(
            dataset="dataset1",
            data_root=self.root / "data",
            outputs_root=self.root / "outputs",
            repo=self.root,
        )
        self.input = self.root / "data" / "in.txt"
        self.input.parent.mkdir(parents=True, exist_ok=True)
        self.input.write_text("payload\n", encoding="utf-8")
        self.output = self.root / "outputs" / "artifact.txt"

    def double(self, body: str, **overrides) -> sc.StageSpec:
        script = textwrap.dedent(body)
        defaults = dict(
            stage_id="double",
            dataset="dataset1",
            artifact_kind="text",
            command=[sys.executable, "-c", script, "<staged output>"],
            env={"DATASET": "dataset1"},
            output=self.output,
            inputs=[self.input],
            config={"k": 1},
            code_files=[self.input],
            description="a test double",
        )
        defaults.update(overrides)
        return sc.StageSpec(**defaults)

    WRITER = """
        import sys
        from pathlib import Path
        Path(sys.argv[1]).write_text("produced\\n", encoding="utf-8")
        print("[LINE epoch 1/1] done")
    """
    FAILER = """
        import sys
        print("something went wrong", file=sys.stderr)
        raise SystemExit(7)
    """

    def run_stage(self, spec: sc.StageSpec) -> str:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            return cp.run_stage(spec, self.ctx, echo=False)

    def test_a_stage_runs_publishes_and_records(self):
        self.assertEqual(self.run_stage(self.double(self.WRITER)), "RAN")
        self.assertEqual(self.output.read_text(encoding="utf-8"), "produced\n")
        record = json.loads(sc.record_path(self.output).read_text(encoding="utf-8"))
        self.assertEqual(record["exit_code"], 0)
        self.assertEqual(record["output"]["sha256"], sc.sha256_file(self.output))
        self.assertEqual(record["command_env"]["DATASET"], "dataset1")

    def test_a_completed_stage_is_reused_not_rerun(self):
        self.run_stage(self.double(self.WRITER))
        before = sc.record_path(self.output).read_text(encoding="utf-8")
        self.assertEqual(self.run_stage(self.double(self.WRITER)), "REUSE")
        self.assertEqual(sc.record_path(self.output).read_text(encoding="utf-8"), before)

    def test_a_fresh_run_refuses_to_reuse(self):
        self.run_stage(self.double(self.WRITER))
        self.ctx.resume = False
        with self.assertRaises(cp.PipelineError) as raised:
            self.run_stage(self.double(self.WRITER))
        self.assertIn("--fresh", str(raised.exception))

    def test_a_nonzero_exit_propagates_and_records_nothing(self):
        with self.assertRaises(cp.PipelineError) as raised:
            self.run_stage(self.double(self.FAILER))
        self.assertIn("exited 7", str(raised.exception))
        self.assertFalse(sc.record_path(self.output).exists())
        self.assertFalse(self.output.exists())

    def test_a_failing_stage_leaves_its_log(self):
        with self.assertRaises(cp.PipelineError):
            self.run_stage(self.double(self.FAILER))
        log = self.ctx.log_dir / "double.log"
        self.assertTrue(log.is_file())
        self.assertIn("something went wrong", log.read_text(encoding="utf-8"))

    def test_a_stage_that_writes_nothing_fails(self):
        silent = """
            import sys
            print("I did nothing")
        """
        with self.assertRaises(cp.PipelineError) as raised:
            self.run_stage(self.double(textwrap.dedent(silent)))
        self.assertIn("produced neither", str(raised.exception))

    def test_an_invalid_artifact_is_never_published(self):
        def refuse(spec, path):
            raise sc.StageContractError("wrong shape")

        with self.assertRaises(sc.StageContractError):
            self.run_stage(self.double(self.WRITER, validator=refuse))
        self.assertFalse(self.output.exists(), "an invalid artifact took the final name")
        self.assertFalse(sc.record_path(self.output).exists())

    def test_an_incompatible_pre_existing_output_stops_the_stage(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("from some other run\n", encoding="utf-8")
        with self.assertRaises(cp.PipelineError) as raised:
            self.run_stage(self.double(self.WRITER))
        self.assertIn("MISSING_COMPLETION_RECORD", str(raised.exception))
        self.assertEqual(
            self.output.read_text(encoding="utf-8"),
            "from some other run\n",
            "a stop must not delete or overwrite the artifact it stopped on",
        )

    def test_changed_inputs_stop_a_completed_stage(self):
        self.run_stage(self.double(self.WRITER))
        self.input.write_text("different\n", encoding="utf-8")
        with self.assertRaises(cp.PipelineError) as raised:
            self.run_stage(self.double(self.WRITER))
        self.assertIn("IDENTITY_MISMATCH", str(raised.exception))

    def test_a_missing_input_stops_before_the_subprocess(self):
        self.input.unlink()
        with self.assertRaises(cp.PipelineError) as raised:
            self.run_stage(self.double(self.WRITER))
        self.assertIn("inputs are absent", str(raised.exception))

    def test_an_interrupted_epoch_count_publishes_no_record(self):
        spec = self.double(
            self.WRITER,
            requested_units=400,
            unit_name="epochs",
            achieved_units_from_log=sc.LINE_EPOCH_READER,
        )
        with self.assertRaises(sc.StageContractError) as raised:
            self.run_stage(spec)
        self.assertIn("1 of 400", str(raised.exception))
        self.assertFalse(sc.record_path(self.output).exists())

    def test_the_jittor_runtime_settings_reach_the_child(self):
        probe = """
            import os, sys
            from pathlib import Path
            Path(sys.argv[1]).write_text(
                f"{os.environ.get('conv_opt')},{os.environ.get('use_mkl')}\\n",
                encoding="utf-8")
        """
        self.run_stage(self.double(textwrap.dedent(probe)))
        self.assertEqual(self.output.read_text(encoding="utf-8").strip(), "1,0")

    def test_a_conflicting_runtime_setting_is_a_stop_not_an_override(self):
        os.environ["use_mkl"] = "1"
        self.addCleanup(lambda: os.environ.pop("use_mkl", None))
        with self.assertRaises(cp.PipelineError) as raised:
            self.run_stage(self.double(self.WRITER))
        self.assertIn("use_mkl", str(raised.exception))

    def test_the_child_receives_this_runs_roots(self):
        probe = """
            import os, sys
            from pathlib import Path
            Path(sys.argv[1]).write_text(os.environ["OUTPUTS_ROOT"], encoding="utf-8")
        """
        self.run_stage(self.double(textwrap.dedent(probe)))
        self.assertEqual(self.output.read_text(encoding="utf-8"), str(self.ctx.outputs_root))


class PassthroughArchive(unittest.TestCase):
    """crf_promote needs dataset1 bytes; they must be produced, never invented."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.ctx = cp.RunContext(
            dataset="dataset2",
            data_root=self.root / "data",
            outputs_root=self.root / "outputs",
            repo=self.root,
        )

    def test_an_absent_dataset1_member_is_a_clear_stop(self):
        with self.assertRaises(cp.PipelineError) as raised:
            cp._passthrough_source(self.ctx)
        message = str(raised.exception)
        self.assertIn("--ds1-member", message)
        self.assertIn("No placeholder is generated", message)

    def test_a_member_without_a_completion_record_is_refused(self):
        member = self.ctx.member("dataset1")
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text("0.5,0.5\n", encoding="utf-8")
        with self.assertRaises(cp.PipelineError):
            cp._passthrough_source(self.ctx)

    def test_a_recorded_member_is_packaged_verbatim(self):
        member = self.ctx.member("dataset1")
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_text("0.5,0.5\n", encoding="utf-8")
        sc.record_path(member).write_text("{}", encoding="utf-8")
        destination = self.root / "outputs" / "pass.zip"
        cp._write_passthrough(self.ctx, destination)
        with zipfile.ZipFile(destination) as archive:
            self.assertEqual(archive.namelist(), ["dataset1.csv"])
            self.assertEqual(archive.read("dataset1.csv"), member.read_bytes())


class OutputGates(unittest.TestCase):
    """A stage artifact that leaves the submission domain never gets a record."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.path = self.root / "matrix.csv"

    def spec(self, **config) -> sc.StageSpec:
        base = {"rows": 3, "columns": 4}
        base.update(config)
        return sc.StageSpec(
            stage_id="m",
            dataset="dataset1",
            artifact_kind="csv",
            command=[],
            env={},
            output=self.path,
            inputs=[],
            config=base,
            code_files=[],
        )

    def write(self, matrix: np.ndarray) -> None:
        pd.DataFrame(matrix).to_csv(self.path, index=False, header=False, float_format="%.6f")

    def test_a_matrix_inside_the_unit_interval_passes(self):
        self.write(np.linspace(0, 1, 12).reshape(3, 4))
        detail = sc.validate_score_matrix_csv(self.spec(), self.path)
        self.assertEqual(detail["shape"], [3, 4])

    def test_a_negative_score_fails_the_gate(self):
        matrix = np.linspace(0, 1, 12).reshape(3, 4)
        matrix[1, 1] = -0.25
        self.write(matrix)
        with self.assertRaises(sc.StageContractError) as raised:
            sc.validate_score_matrix_csv(self.spec(), self.path)
        self.assertIn("outside [0, 1]", str(raised.exception))

    def test_a_score_above_one_fails_the_gate(self):
        matrix = np.linspace(0, 1, 12).reshape(3, 4)
        matrix[0, 0] = 1.5
        self.write(matrix)
        with self.assertRaises(sc.StageContractError):
            sc.validate_score_matrix_csv(self.spec(), self.path)

    def test_a_wrong_row_count_fails_the_gate(self):
        self.write(np.linspace(0, 1, 12).reshape(3, 4))
        with self.assertRaises(sc.StageContractError):
            sc.validate_score_matrix_csv(self.spec(rows=5), self.path)

    def test_a_crf_archive_missing_a_member_fails(self):
        archive_path = self.root / "crf.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("dataset2.csv", "0.1,0.2\n")
        with self.assertRaises(sc.StageContractError) as raised:
            cp.validate_crf_archive(self.spec(), archive_path)
        self.assertIn("dataset1.csv", str(raised.exception))

    def test_a_passthrough_archive_with_extra_members_fails(self):
        archive_path = self.root / "pass.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("dataset1.csv", "0.1\n")
            archive.writestr("dataset2.csv", "0.2\n")
        with self.assertRaises(sc.StageContractError):
            cp.validate_passthrough_archive(self.spec(), archive_path)


class JittorOnly(unittest.TestCase):
    """The canonical surface offers exactly one backend, and it is Jittor."""

    def test_the_entrypoint_has_no_backend_option(self):
        options = {
            option
            for action in entrypoint.build_parser()._actions
            for option in action.option_strings
        }
        for forbidden in ("--framework", "--backend", "--torch", "--use-torch"):
            self.assertNotIn(forbidden, options)

    def test_a_backend_argument_is_rejected(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--stage", "describe", "--framework", "torch"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrecognized arguments", result.stderr)

    def test_the_package_stage_refuses_instead_of_reporting_success(self):
        """Declining to package is a refusal, so it must not exit zero."""
        result = subprocess.run(
            [sys.executable, "main.py", "--stage", "package"],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("deliberately not", result.stderr)
        self.assertNotIn("deliberately not", result.stdout, "the refusal belongs on stderr")

    def test_no_backend_environment_variable_is_consulted(self):
        for path in (REPO / "main.py", SRC / "canonical_pipeline.py"):
            source = path.read_text(encoding="utf-8")
            for forbidden in ("FRAMEWORK", "BACKEND", "USE_TORCH"):
                self.assertNotIn(forbidden, source, f"{path.name} reads {forbidden}")

    def test_the_graph_names_only_the_jittor_trainers(self):
        ctx = cp.RunContext(
            dataset="dataset1", data_root=REPO / "data", outputs_root=REPO / "outputs", repo=REPO
        )
        commands = " ".join(str(part) for spec in cp.build_stages(ctx) for part in spec.command)
        self.assertIn("train_line_jt.py", commands)
        self.assertIn("train_bpr_jt.py", commands)
        for historical in ("reference/pytorch/train_line.py", "reference/pytorch/train_bpr.py"):
            self.assertNotIn(historical, commands.replace("_jt.py", ".JT"))

    def test_the_entrypoint_import_closure_contains_no_torch(self):
        """Walk the transitive import graph from main.py without executing it."""
        reached, third_party = self._closure(REPO / "main.py")
        self.assertNotIn(
            "torch", third_party, f"torch is reachable from main.py via {sorted(reached)}"
        )
        self.assertIn("canonical_pipeline", reached)
        self.assertIn("stage_contract", reached)

    def test_the_closure_walker_would_notice_torch(self):
        """Anti-vacuity: the same walker finds torch when torch is really there."""
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "probe.py"
            probe.write_text("import torch\n", encoding="utf-8")
            _, third_party = self._closure(probe)
        self.assertIn("torch", third_party)

    def _closure(self, entry: Path) -> tuple[set, set]:
        local = {p.stem for p in SRC.rglob("*.py")} | {"main", "canonical_pipeline"}
        reached: set[str] = set()
        third_party: set[str] = set()
        stack = [entry]
        while stack:
            path = stack.pop()
            if path.stem in reached:
                continue
            reached.add(path.stem)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            names: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names.add(node.module.split(".")[0])
            for name in names:
                if name in local:
                    for candidate in (SRC / f"{name}.py", REPO / f"{name}.py"):
                        if candidate.is_file():
                            stack.append(candidate)
                            break
                else:
                    third_party.add(name)
        return reached, third_party


class EntrypointBehaviour(unittest.TestCase):
    def test_run_is_the_default_stage(self):
        self.assertEqual(entrypoint.build_parser().parse_args([]).stage, "run")

    def test_dataset_selection_is_explicit(self):
        args = entrypoint.build_parser().parse_args(["--dataset", "dataset2"])
        self.assertEqual(args.dataset, "dataset2")
        with self.assertRaises(SystemExit):
            entrypoint.build_parser().parse_args(["--dataset", "dataset9"])

    def test_no_option_waves_a_stage_through_unchecked(self):
        options = {
            option
            for action in entrypoint.build_parser()._actions
            for option in action.option_strings
        }
        for forbidden in (
            "--skip-checks",
            "--force",
            "--no-verify",
            "--ignore-checks",
            "--framework",
            "--backend",
        ):
            self.assertNotIn(forbidden, options)

    def test_the_documented_controls_all_exist(self):
        options = {
            option
            for action in entrypoint.build_parser()._actions
            for option in action.option_strings
        }
        for required in (
            "--dataset",
            "--data-root",
            "--output-root",
            "--log-dir",
            "--config",
            "--fresh",
            "--stage",
        ):
            self.assertIn(required, options)

    def test_run_without_official_data_fails_with_a_named_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            buffer, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(buffer), redirect_stderr(errors):
                code = entrypoint.main(
                    [
                        "--dataset",
                        "dataset1",
                        "--data-root",
                        tmp,
                        "--output-root",
                        str(Path(tmp) / "out"),
                    ]
                )
            self.assertEqual(code, 2)
            self.assertIn("official raw data is absent", errors.getvalue())

    def test_describe_still_works_without_data(self):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = entrypoint.main(["--stage", "describe", "--dataset", "dataset1"])
        self.assertEqual(code, 0)
        self.assertIn("python main.py --dataset dataset1", buffer.getvalue())

    def test_the_module_docstring_states_the_real_entry_commands(self):
        for dataset in ("dataset1", "dataset2"):
            self.assertIn(f"python main.py --dataset {dataset}", entrypoint.__doc__)

    def test_main_no_longer_only_prints_commands(self):
        source = (REPO / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("NOT EXECUTED by main.py", source)
        self.assertIn("from canonical_pipeline import execute", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
