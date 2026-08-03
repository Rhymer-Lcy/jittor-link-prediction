# -*- coding: utf-8 -*-
"""The completion contract: every way a stage can look done without being done.

The property under test is one sentence: OUTPUT EXISTS != STAGE COMPLETE. Each
test constructs a stage that has finished correctly, then breaks exactly one of
the things a completion record binds, and asserts that the contract stops rather
than reusing, re-running or repairing.

Two anti-vacuity tests bracket the rest. ``test_the_happy_path_actually_reuses``
proves the checks can pass at all, so a blanket STOP would fail the suite; and
``test_every_stop_reason_is_reachable`` proves each stop code is produced by a
distinct, constructible condition rather than by one over-broad guard.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import stage_contract as sc  # noqa: E402


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class ContractHarness(unittest.TestCase):
    """A finished stage in a temporary tree, plus the levers to break it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.code = write_text(self.root / "code" / "producer.py", "VALUE = 1\n")
        self.source = write_text(self.root / "inputs" / "train.csv", "src,dst\n1,2\n")
        self.output = self.root / "outputs" / "artifact.txt"

    def spec(self, **overrides) -> sc.StageSpec:
        defaults = dict(
            stage_id="demo_stage",
            dataset="dataset1",
            artifact_kind="text",
            command=[sys.executable, "-c", "pass"],
            env={"DATASET": "dataset1"},
            output=self.output,
            inputs=[self.source],
            config={"alpha": 1, "beta": "two"},
            code_files=[self.code],
            seed=42,
            description="a stage that writes one line",
        )
        defaults.update(overrides)
        return sc.StageSpec(**defaults)

    def finish(
        self, spec: sc.StageSpec | None = None, *, payload: str = "done\n", log_text: str = ""
    ) -> dict:
        """Run the stage's effect and publish a real completion record."""
        spec = spec or self.spec()
        with sc.atomic_output(spec.output) as staged:
            staged.write_text(payload, encoding="utf-8")
        return sc.complete_stage(
            spec,
            repo=self.root,
            exit_code=0,
            started_at="2026-08-02T00:00:00Z",
            completed_at="2026-08-02T00:00:01Z",
            log_text=log_text,
        )

    def record(self) -> dict:
        return json.loads(sc.record_path(self.output).read_text(encoding="utf-8"))

    def rewrite_record(self, **changes) -> None:
        document = self.record()
        document.update(changes)
        sc.record_path(self.output).write_text(json.dumps(document, indent=2), encoding="utf-8")

    def assert_stop(self, code: str, spec: sc.StageSpec | None = None) -> sc.Decision:
        decision = sc.evaluate(spec or self.spec(), repo=self.root)
        self.assertEqual(
            decision.action, "STOP", f"expected a stop, got {decision.action} ({decision.code})"
        )
        self.assertEqual(decision.code, code, decision.detail)
        return decision


class HappyPath(ContractHarness):
    def test_a_clean_slate_runs(self):
        decision = sc.evaluate(self.spec(), repo=self.root)
        self.assertEqual(decision.action, "RUN")
        self.assertEqual(decision.code, "CLEAN_SLATE")

    def test_the_happy_path_actually_reuses(self):
        """Anti-vacuity: an untouched finished stage must be reusable."""
        self.finish()
        decision = sc.evaluate(self.spec(), repo=self.root)
        self.assertEqual(decision.action, "REUSE", decision.detail)
        self.assertEqual(decision.mismatches, [])

    def test_the_record_binds_every_required_property(self):
        self.finish()
        document = self.record()
        for key in sc.REQUIRED_RECORD_KEYS:
            self.assertIn(key, document)
        self.assertEqual(document["schema_version"], sc.SCHEMA_VERSION)
        self.assertEqual(document["exit_code"], 0)
        self.assertEqual(document["validation"]["status"], "PASS")
        self.assertEqual(document["inputs"][0]["sha256"], sc.sha256_file(self.source))
        self.assertEqual(document["output"]["sha256"], sc.sha256_file(self.output))
        self.assertIn("digest", document["code_identity"])
        self.assertIn("python", document["environment"])

    def test_the_commit_is_recorded_even_beside_a_code_digest(self):
        self.finish()
        document = self.record()
        self.assertTrue(document["producing_commit"])
        self.assertNotEqual(document["code_identity"]["digest"], "")


class OutputAndRecordDisagree(ContractHarness):
    def test_output_without_a_record_stops(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("done\n", encoding="utf-8")
        self.assert_stop("MISSING_COMPLETION_RECORD")

    def test_record_without_an_output_stops(self):
        self.finish()
        self.output.unlink()
        self.assert_stop("MISSING_OUTPUT")

    def test_a_stale_part_file_stops(self):
        sc.part_path(self.output).parent.mkdir(parents=True, exist_ok=True)
        sc.part_path(self.output).write_text("half", encoding="utf-8")
        self.assert_stop("STALE_PART_FILE")

    def test_a_stale_part_stops_even_beside_a_valid_record(self):
        self.finish()
        sc.part_path(self.output).write_text("half", encoding="utf-8")
        self.assert_stop("STALE_PART_FILE")


class RecordIntegrity(ContractHarness):
    def test_malformed_json_stops(self):
        self.finish()
        sc.record_path(self.output).write_text("{not json", encoding="utf-8")
        self.assert_stop("MALFORMED_RECORD")

    def test_a_json_scalar_is_not_a_record(self):
        self.finish()
        sc.record_path(self.output).write_text("42", encoding="utf-8")
        self.assert_stop("MALFORMED_RECORD")

    def test_an_unsupported_schema_version_stops(self):
        self.finish()
        self.rewrite_record(schema_version=sc.SCHEMA_VERSION + 99)
        self.assert_stop("UNSUPPORTED_SCHEMA")

    def test_a_record_missing_a_required_field_stops(self):
        self.finish()
        document = self.record()
        del document["config_digest"]
        sc.record_path(self.output).write_text(json.dumps(document), encoding="utf-8")
        self.assert_stop("INCOMPLETE_RECORD")


class IdentityMismatches(ContractHarness):
    def test_an_input_edited_without_changing_its_size_stops(self):
        """The hash, not the size, is what catches an equal-length edit."""
        self.finish()
        self.source.write_text("src,dst\n9,8\n", encoding="utf-8")
        decision = self.assert_stop("IDENTITY_MISMATCH")
        self.assertTrue(
            any(m["property"].endswith("sha256") for m in decision.mismatches), decision.mismatches
        )

    def test_an_input_that_grew_stops_on_size(self):
        self.finish()
        self.source.write_text("src,dst\n1,2\n3,4\n", encoding="utf-8")
        decision = self.assert_stop("IDENTITY_MISMATCH")
        self.assertTrue(
            any(m["property"].endswith("size") for m in decision.mismatches), decision.mismatches
        )

    def test_a_vanished_input_stops(self):
        self.finish()
        self.source.unlink()
        decision = self.assert_stop("IDENTITY_MISMATCH")
        self.assertTrue(any(m["current"] == "ABSENT" for m in decision.mismatches))

    def test_a_different_input_set_stops(self):
        self.finish()
        other = write_text(self.root / "inputs" / "test.csv", "a\n")
        self.assert_stop("IDENTITY_MISMATCH", self.spec(inputs=[self.source, other]))

    def test_a_changed_configuration_stops(self):
        self.finish()
        self.assert_stop("IDENTITY_MISMATCH", self.spec(config={"alpha": 2, "beta": "two"}))

    def test_a_changed_seed_stops(self):
        self.finish()
        self.assert_stop("IDENTITY_MISMATCH", self.spec(seed=7))

    def test_changed_code_stops(self):
        self.finish()
        self.code.write_text("VALUE = 2\n", encoding="utf-8")
        decision = self.assert_stop("IDENTITY_MISMATCH")
        self.assertTrue(any(m["property"] == "code_identity.digest" for m in decision.mismatches))

    def test_a_different_producing_commit_stops(self):
        self.finish()
        self.rewrite_record(producing_commit="0" * 40)
        decision = self.assert_stop("IDENTITY_MISMATCH")
        self.assertTrue(any(m["property"] == "producing_commit" for m in decision.mismatches))

    def test_a_changed_output_stops(self):
        self.finish()
        self.output.write_text("tampered\n", encoding="utf-8")
        decision = self.assert_stop("IDENTITY_MISMATCH")
        self.assertTrue(any("output" in m["property"] for m in decision.mismatches))

    def test_a_truncated_output_stops_on_size_before_hash(self):
        self.finish()
        self.output.write_text("do", encoding="utf-8")
        decision = self.assert_stop("IDENTITY_MISMATCH")
        self.assertIn("output size", [m["property"] for m in decision.mismatches])

    def test_a_record_claiming_a_nonzero_exit_stops(self):
        self.finish()
        self.rewrite_record(exit_code=1)
        self.assert_stop("IDENTITY_MISMATCH")

    def test_a_record_claiming_a_failed_validation_stops(self):
        self.finish()
        document = self.record()
        document["validation"]["status"] = "FAIL"
        sc.record_path(self.output).write_text(json.dumps(document), encoding="utf-8")
        self.assert_stop("IDENTITY_MISMATCH")

    def test_a_record_from_another_stage_stops(self):
        self.finish()
        self.assert_stop("IDENTITY_MISMATCH", self.spec(stage_id="a_different_stage"))


class EpochAccounting(ContractHarness):
    """A trainer that exports periodically can leave a complete-looking artifact."""

    def spec_with_epochs(self, **overrides) -> sc.StageSpec:
        return self.spec(
            requested_units=400,
            unit_name="epochs",
            achieved_units_from_log=sc.LINE_EPOCH_READER,
            **overrides,
        )

    def test_a_full_epoch_count_completes(self):
        log = "".join(f"[LINE epoch {i}/400] avg loss 0.1\n" for i in range(1, 401))
        record = self.finish(self.spec_with_epochs(), log_text=log)
        self.assertEqual(record["achieved_units"], 400)
        self.assertEqual(sc.evaluate(self.spec_with_epochs(), repo=self.root).action, "REUSE")

    def test_an_interrupted_epoch_count_writes_no_record(self):
        log = "".join(f"[LINE epoch {i}/400] avg loss 0.1\n" for i in range(1, 251))
        with self.assertRaises(sc.StageContractError) as raised:
            self.finish(self.spec_with_epochs(), log_text=log)
        self.assertIn("250 of 400", str(raised.exception))
        self.assertFalse(sc.record_path(self.output).exists())

    def test_a_silent_trainer_writes_no_record(self):
        with self.assertRaises(sc.StageContractError):
            self.finish(self.spec_with_epochs(), log_text="")

    def test_a_record_with_too_few_achieved_units_stops(self):
        log = "".join(f"[LINE epoch {i}/400] x\n" for i in range(1, 401))
        self.finish(self.spec_with_epochs(), log_text=log)
        self.rewrite_record(achieved_units=399)
        self.assert_stop("IDENTITY_MISMATCH", self.spec_with_epochs())

    def test_a_changed_requested_unit_count_stops(self):
        log = "".join(f"[LINE epoch {i}/400] x\n" for i in range(1, 401))
        self.finish(self.spec_with_epochs(), log_text=log)
        self.assert_stop(
            "IDENTITY_MISMATCH",
            self.spec(
                requested_units=600,
                unit_name="epochs",
                achieved_units_from_log=sc.LINE_EPOCH_READER,
            ),
        )

    def test_the_bpr_reader_reads_its_own_format(self):
        self.assertEqual(sc.BPR_EPOCH_READER("[BPR epoch 120/120] avg loss 0.02\n"), 120)
        self.assertIsNone(sc.BPR_EPOCH_READER("[LINE epoch 400/400]\n"))


class FailureNeverPublishes(ContractHarness):
    def test_a_nonzero_exit_writes_no_record(self):
        with sc.atomic_output(self.output) as staged:
            staged.write_text("partial\n", encoding="utf-8")
        with self.assertRaises(sc.StageContractError):
            sc.complete_stage(
                self.spec(), repo=self.root, exit_code=3, started_at="a", completed_at="b"
            )
        self.assertFalse(sc.record_path(self.output).exists())

    def test_a_missing_output_writes_no_record(self):
        with self.assertRaises(sc.StageContractError):
            sc.complete_stage(
                self.spec(), repo=self.root, exit_code=0, started_at="a", completed_at="b"
            )
        self.assertFalse(sc.record_path(self.output).exists())

    def test_a_leftover_part_writes_no_record(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("done\n", encoding="utf-8")
        sc.part_path(self.output).write_text("half\n", encoding="utf-8")
        with self.assertRaises(sc.StageContractError):
            sc.complete_stage(
                self.spec(), repo=self.root, exit_code=0, started_at="a", completed_at="b"
            )
        self.assertFalse(sc.record_path(self.output).exists())

    def test_a_failing_validator_writes_no_record(self):
        def refuse(spec, path):
            raise sc.StageContractError("the artifact is wrong")

        with sc.atomic_output(self.output) as staged:
            staged.write_text("done\n", encoding="utf-8")
        with self.assertRaises(sc.StageContractError):
            sc.complete_stage(
                self.spec(validator=refuse),
                repo=self.root,
                exit_code=0,
                started_at="a",
                completed_at="b",
            )
        self.assertFalse(sc.record_path(self.output).exists())

    def test_publish_record_refuses_a_failed_record(self):
        self.finish()
        document = self.record()
        document["exit_code"] = 1
        with self.assertRaises(sc.StageContractError):
            sc.publish_record(self.spec(), document)
        document["exit_code"] = 0
        document["validation"]["status"] = "FAIL"
        with self.assertRaises(sc.StageContractError):
            sc.publish_record(self.spec(), document)


class AtomicPublication(ContractHarness):
    def test_atomic_output_leaves_nothing_behind_on_failure(self):
        with self.assertRaises(RuntimeError):
            with sc.atomic_output(self.output) as staged:
                staged.write_text("half\n", encoding="utf-8")
                raise RuntimeError("the stage died")
        self.assertFalse(self.output.exists())
        self.assertFalse(sc.part_path(self.output).exists())

    def test_atomic_output_does_not_disturb_an_existing_file_on_failure(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("previous\n", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            with sc.atomic_output(self.output) as staged:
                staged.write_text("new\n", encoding="utf-8")
                raise RuntimeError("boom")
        self.assertEqual(self.output.read_text(encoding="utf-8"), "previous\n")

    def test_atomic_output_refuses_an_empty_publication(self):
        with self.assertRaises(sc.StageContractError):
            with sc.atomic_output(self.output):
                pass

    def test_the_record_is_published_by_rename_not_in_place(self):
        """The record must never be observable half-written."""
        source = (SRC / "stage_contract.py").read_text(encoding="utf-8")
        body = source.split("def publish_record", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("os.replace(tmp, target)", body)
        self.assertIn("os.fsync", body)
        self.assertNotIn("target.write_text", body)

    def test_the_record_is_written_after_the_output(self):
        """Anti-vacuity for the ordering claim, read off the real source."""
        source = (SRC / "stage_contract.py").read_text(encoding="utf-8")
        body = source.split("def complete_stage", 1)[1].split("\n# ---", 1)[0]
        self.assertLess(
            body.index("spec.validator(spec, output)"), body.index("publish_record(spec, record)")
        )
        self.assertLess(body.index("exit_code != 0"), body.index("spec.validator"))


class Quarantine(ContractHarness):
    def test_quarantine_moves_and_explains_but_never_deletes(self):
        self.finish()
        outputs = self.root / "outputs"
        slot = sc.quarantine(
            [self.output, sc.record_path(self.output)], outputs_root=outputs, reason="test"
        )
        self.assertFalse(self.output.exists())
        self.assertTrue((slot / self.output.name).exists())
        self.assertTrue((slot / sc.record_path(self.output).name).exists())
        note = json.loads((slot / "QUARANTINE.json").read_text(encoding="utf-8"))
        self.assertEqual(note["reason"], "test")
        self.assertEqual(sc.evaluate(self.spec(), repo=self.root).action, "RUN")

    def test_quarantine_slots_do_not_collide(self):
        outputs = self.root / "outputs"
        first = sc.quarantine([], outputs_root=outputs, reason="one")
        second = sc.quarantine([], outputs_root=outputs, reason="two")
        self.assertNotEqual(first, second)

    def test_the_contract_offers_no_way_to_wave_a_stage_through(self):
        """Checked against the code, not the prose, which does discuss the absence."""
        code = "\n".join(
            line
            for line in (SRC / "stage_contract.py").read_text(encoding="utf-8").splitlines()
            if not line.strip().startswith("#")
        )
        code = code.split('"""', 2)[-1]  # drop the module docstring
        for forbidden in ("skip_checks", "SKIP_CHECKS", "force_reuse", "ignore_mismatch"):
            self.assertNotIn(forbidden, code)

    def test_quarantine_is_the_only_documented_way_past_a_stop(self):
        exported = {name for name in dir(sc) if not name.startswith("_")}
        self.assertIn("quarantine", exported)
        for name in exported:
            self.assertNotIn("override", name.lower())
            self.assertNotIn("bypass", name.lower())


class NpzCacheContract(unittest.TestCase):
    """The dataset2 train-feature cache: written atomically, validated internally."""

    KEYS = ("Xf", "yf", "lens", "qsrc_tr", "qt_tr", "qorig_tr", "cands_concat", "num_entity")

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.path = self.root / "train_features_all.npz"
        self.arrays = self.make_arrays()
        self.write(self.arrays)

    def make_arrays(self, queries: int = 4, width: int = 18, num_entity: int = 50) -> dict:
        lens = np.full(queries, 5, np.int64)
        total = int(lens.sum())
        labels = np.zeros(total, np.float32)
        labels[np.cumsum(lens) - 1] = 1.0
        return {
            "Xf": np.arange(total * width, dtype=np.float32).reshape(total, width),
            "yf": labels,
            "lens": lens,
            "qsrc_tr": np.arange(queries, dtype=np.int64),
            "qt_tr": np.arange(queries, dtype=np.float64),
            "qorig_tr": np.arange(queries, dtype=np.int64),
            "cands_concat": np.arange(total, dtype=np.int64),
            "num_entity": np.array([num_entity]),
        }

    def write(self, arrays: dict) -> None:
        with self.path.open("wb") as handle:
            np.savez(handle, **arrays)

    def spec(self, **config) -> sc.StageSpec:
        base = {"queries": 4, "feature_width": 18, "num_entity": 50, "npz_keys": list(self.KEYS)}
        base.update(config)
        return sc.StageSpec(
            stage_id="cache",
            dataset="dataset2",
            artifact_kind="npz",
            command=["x"],
            env={},
            output=self.path,
            inputs=[],
            config=base,
            code_files=[],
        )

    def test_a_well_formed_cache_validates(self):
        detail = sc.validate_npz_cache(self.spec(), self.path)
        self.assertEqual(detail["queries"], 4)
        self.assertEqual(detail["rows"], 20)
        self.assertEqual(detail["num_entity"], 50)

    def test_a_truncated_npz_fails(self):
        payload = self.path.read_bytes()
        self.path.write_bytes(payload[: len(payload) // 2])
        with self.assertRaises(Exception):
            sc.validate_npz_cache(self.spec(), self.path)

    def test_a_missing_array_fails(self):
        reduced = {k: v for k, v in self.arrays.items() if k != "qorig_tr"}
        self.write(reduced)
        with self.assertRaises(sc.StageContractError) as raised:
            sc.validate_npz_cache(self.spec(), self.path)
        self.assertIn("qorig_tr", str(raised.exception))

    def test_a_wrong_query_count_fails(self):
        with self.assertRaises(sc.StageContractError):
            sc.validate_npz_cache(self.spec(queries=5), self.path)

    def test_a_wrong_entity_count_fails(self):
        with self.assertRaises(sc.StageContractError):
            sc.validate_npz_cache(self.spec(num_entity=51), self.path)

    def test_a_wrong_feature_width_fails(self):
        with self.assertRaises(sc.StageContractError):
            sc.validate_npz_cache(self.spec(feature_width=21), self.path)

    def test_inconsistent_lens_and_rows_fail(self):
        broken = dict(self.arrays)
        broken["lens"] = np.full(4, 6, np.int64)
        self.write(broken)
        with self.assertRaises(sc.StageContractError) as raised:
            sc.validate_npz_cache(self.spec(), self.path)
        self.assertIn("lens.sum()", str(raised.exception))

    def test_misplaced_labels_fail(self):
        broken = dict(self.arrays)
        labels = np.zeros(20, np.float32)
        labels[0] = 1.0
        broken["yf"] = labels
        self.write(broken)
        with self.assertRaises(sc.StageContractError) as raised:
            sc.validate_npz_cache(self.spec(), self.path)
        self.assertIn("positives", str(raised.exception))

    def test_non_finite_features_fail(self):
        broken = dict(self.arrays)
        broken["Xf"] = broken["Xf"].copy()
        broken["Xf"][3, 3] = np.nan
        self.write(broken)
        with self.assertRaises(sc.StageContractError) as raised:
            sc.validate_npz_cache(self.spec(), self.path)
        self.assertIn("non-finite", str(raised.exception))

    def test_a_mismatched_candidate_length_fails(self):
        broken = dict(self.arrays)
        broken["cands_concat"] = np.arange(19, dtype=np.int64)
        self.write(broken)
        with self.assertRaises(sc.StageContractError) as raised:
            sc.validate_npz_cache(self.spec(), self.path)
        self.assertIn("cands_concat", str(raised.exception))


class StopReasonsAreDistinct(ContractHarness):
    def test_every_stop_reason_is_reachable(self):
        """Anti-vacuity: each stop code has its own constructible condition."""
        reached = set()

        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text("x", encoding="utf-8")
        reached.add(sc.evaluate(self.spec(), repo=self.root).code)

        self.finish()
        self.output.unlink()
        reached.add(sc.evaluate(self.spec(), repo=self.root).code)

        self.finish()
        sc.record_path(self.output).write_text("{", encoding="utf-8")
        reached.add(sc.evaluate(self.spec(), repo=self.root).code)

        self.finish()
        self.rewrite_record(schema_version=99)
        reached.add(sc.evaluate(self.spec(), repo=self.root).code)

        self.finish()
        document = self.record()
        del document["inputs"]
        sc.record_path(self.output).write_text(json.dumps(document), encoding="utf-8")
        reached.add(sc.evaluate(self.spec(), repo=self.root).code)

        self.finish()
        self.source.write_text("changed\n", encoding="utf-8")
        reached.add(sc.evaluate(self.spec(), repo=self.root).code)

        self.finish()
        sc.part_path(self.output).write_text("half", encoding="utf-8")
        reached.add(sc.evaluate(self.spec(), repo=self.root).code)

        self.assertEqual(
            reached,
            {
                "MISSING_COMPLETION_RECORD",
                "MISSING_OUTPUT",
                "MALFORMED_RECORD",
                "UNSUPPORTED_SCHEMA",
                "INCOMPLETE_RECORD",
                "IDENTITY_MISMATCH",
                "STALE_PART_FILE",
            },
        )


class DigestDeterminism(unittest.TestCase):
    def test_the_configuration_digest_ignores_key_order(self):
        self.assertEqual(sc.digest_of({"a": 1, "b": 2}), sc.digest_of({"b": 2, "a": 1}))

    def test_the_configuration_digest_separates_different_values(self):
        self.assertNotEqual(sc.digest_of({"a": 1}), sc.digest_of({"a": 1.5}))
        self.assertNotEqual(sc.digest_of({"a": 1}), sc.digest_of({"a": "1"}))

    def test_digest_keys_restrict_the_comparison(self):
        def build(**config):
            return sc.StageSpec(
                stage_id="s",
                dataset="dataset1",
                artifact_kind="k",
                command=[],
                env={},
                output=Path("o"),
                inputs=[],
                config=config,
                code_files=[],
                digest_keys=("a",),
            )

        self.assertEqual(build(a=1, b=2).config_digest(), build(a=1, b=99).config_digest())
        self.assertNotEqual(build(a=1, b=2).config_digest(), build(a=2, b=2).config_digest())

    def test_missing_digest_keys_are_an_error_not_a_silent_pass(self):
        spec = sc.StageSpec(
            stage_id="s",
            dataset="dataset1",
            artifact_kind="k",
            command=[],
            env={},
            output=Path("o"),
            inputs=[],
            config={"b": 1},
            code_files=[],
            digest_keys=("a",),
        )
        with self.assertRaises(sc.StageContractError):
            spec.config_digest()

    def test_git_commit_reports_unknown_outside_a_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(sc.git_commit(Path(tmp)), "UNKNOWN")

    def test_git_commit_reports_the_head_of_this_checkout(self):
        commit = sc.git_commit(REPO)
        expected = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        self.assertEqual(commit, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
