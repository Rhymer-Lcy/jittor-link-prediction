# -*- coding: utf-8 -*-
"""Integration tests against the accepted submission artifacts.

These need local, git-ignored assets (the accepted archive, the ranker output
and the raw competition data). Every test skips with an explicit reason when its
asset is missing, so a clean checkout still runs the suite green.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

PRODUCTION = json.loads((REPO / "configs" / "production.json").read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def packaged_code_members() -> list[tuple[str, str, str]]:
    """The authoritative submission file set, read from the packaging tool.

    ``tools/submission/stage_package.py`` owns ``CODE_MEMBERS``: the explicit
    list that builds the archive. Reading it here means a dependency audit and
    the package can never disagree about what "the submission" is. The literal is
    parsed rather than imported, so no packaging side effect runs during tests.
    """
    import ast

    source = (REPO / "tools" / "submission" / "stage_package.py").read_text(encoding="utf-8")
    for node in ast.parse(source).body:
        target = None
        if isinstance(node, ast.AnnAssign):
            target = getattr(node.target, "id", None)
        elif isinstance(node, ast.Assign) and node.targets:
            target = getattr(node.targets[0], "id", None)
        if target == "CODE_MEMBERS":
            return [tuple(entry) for entry in ast.literal_eval(node.value)]
    raise AssertionError("stage_package.CODE_MEMBERS not found")


def packaged_python_sources() -> list[Path]:
    """Absolute paths of every packaged ``.py`` file, sorted."""
    return sorted(REPO / rel for rel, _, _ in packaged_code_members() if rel.endswith(".py"))


def declared_dependencies() -> str:
    """The functional lines of both specifications, with comments removed.

    Comments are stripped deliberately: a dependency mentioned only in prose is
    not installed, so it must not satisfy the import audit.
    """
    lines: list[str] = []
    for name in ("environment.yaml", "requirements.txt"):
        path = REPO / name
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)
    return "\n".join(lines)


class ProductionManifestTest(unittest.TestCase):
    """Pure-config checks -- these always run."""

    def test_components_sum_to_the_accepted_total(self):
        ds1 = PRODUCTION["dataset1"]["component_score"]
        ds2 = PRODUCTION["dataset2"]["component_score"]
        self.assertEqual(ds1 + ds2, PRODUCTION["accepted_total_score"])

    def test_chain_order_matches_the_registry(self):
        from strategies.registry import active_ds1_chain_ids

        configured = [
            s["strategy_id"] for s in PRODUCTION["dataset1"]["strategy_chain"] if s["order"] > 0
        ]
        self.assertEqual(configured, active_ds1_chain_ids())

    def test_dataset2_decoder_matches_the_registry(self):
        from strategies.registry import active_ds2_chain_ids

        decoder = PRODUCTION["dataset2"]["final_decoder"]
        self.assertEqual([decoder["strategy_id"]], active_ds2_chain_ids())

    def test_dataset2_decoder_implementation_is_tracked(self):
        decoder = PRODUCTION["dataset2"]["final_decoder"]
        self.assertEqual(decoder["implementation_status"], "GRADUATED")
        self.assertTrue(
            (REPO / decoder["implementation"]).exists(), f"missing {decoder['implementation']}"
        )

    def test_dataset2_decoder_gain_is_not_presented_as_a_gate_pass(self):
        # The decoder shipped by an explicit override, NOT by passing its gate.
        # Both figures must stay in the record so the distinction cannot be lost.
        decoder = PRODUCTION["dataset2"]["final_decoder"]
        gain = float(decoder["online_observed_gain"])
        gate = float(decoder["original_locked_gate"])
        self.assertLess(gain, gate)
        self.assertAlmostEqual(
            float(decoder["shortfall_from_original_gate"]), gate - gain, places=15
        )
        self.assertEqual(decoder["historical_lifecycle"], "CLOSED_AT_LOCKED_GATE")
        self.assertEqual(decoder["operational_decision"], "FINAL_BOARD_MAXIMISATION_OVERRIDE")

    def test_dataset2_decoder_component_arithmetic(self):
        decoder = PRODUCTION["dataset2"]["final_decoder"]
        before = decoder["component_score_before"]
        after = decoder["component_score_after"]
        self.assertEqual(after, PRODUCTION["dataset2"]["component_score"])
        self.assertEqual(
            before, PRODUCTION["dataset2"]["decoder_input"]["component_score_at_this_stage"]
        )
        self.assertAlmostEqual(after - before, float(decoder["online_observed_gain"]), places=15)

    def test_every_referenced_implementation_exists(self):
        for dataset in ("dataset1", "dataset2"):
            for stage in PRODUCTION[dataset]["strategy_chain"]:
                path = REPO / stage["implementation"]
                self.assertTrue(path.exists(), f"missing implementation {path}")

    def test_strategy_inventory_statuses_are_valid(self):
        from strategies.registry import LIFECYCLE_STATUSES

        inventory = json.loads(
            (REPO / "docs" / "strategy_inventory.json").read_text(encoding="utf-8")
        )
        for strategy in inventory["strategies"]:
            self.assertIn(
                strategy["lifecycle_status"],
                LIFECYCLE_STATUSES,
                f"{strategy['strategy_id']} has an unknown status",
            )

    def test_every_shipped_active_strategy_has_an_implementation(self):
        inventory = json.loads(
            (REPO / "docs" / "strategy_inventory.json").read_text(encoding="utf-8")
        )
        for strategy in inventory["strategies"]:
            if strategy["lifecycle_status"] != "SHIPPED_ACTIVE":
                continue
            impl = strategy.get("implementation_path")
            self.assertTrue(impl, f"{strategy['strategy_id']} is active with no implementation")
            self.assertTrue((REPO / impl).exists(), f"missing {impl}")


class AcceptedArchiveTest(unittest.TestCase):
    def setUp(self):
        self.archive = REPO / PRODUCTION["accepted_archive"]["path"]
        if not self.archive.exists():
            self.skipTest(f"accepted archive not present locally: {self.archive}")

    def test_archive_hash_and_size(self):
        self.assertEqual(self.archive.stat().st_size, PRODUCTION["accepted_archive"]["bytes"])
        self.assertEqual(sha256_file(self.archive), PRODUCTION["accepted_archive"]["sha256"])

    def test_member_hashes_and_container(self):
        with zipfile.ZipFile(self.archive) as z:
            self.assertIsNone(z.testzip())
            self.assertEqual(
                [i.filename for i in z.infolist()], PRODUCTION["accepted_archive"]["member_order"]
            )
            self.assertEqual(z.comment, b"")
            self.assertEqual(
                {i.compress_type for i in z.infolist()},
                {PRODUCTION["accepted_archive"]["compress_method"]},
            )
            self.assertFalse(any(i.is_dir() for i in z.infolist()))
            for member, dataset in (("dataset1.csv", "dataset1"), ("dataset2.csv", "dataset2")):
                digest = hashlib.sha256(z.read(member)).hexdigest()
                self.assertEqual(digest, PRODUCTION[dataset]["member_sha256"], member)


class ChainReproductionTest(unittest.TestCase):
    """The decisive test: tracked code must regenerate the accepted member exactly."""

    #: The frozen chain input is located by IDENTITY, never by path alone. After a
    #: clean raw-to-final run the canonical production path legitimately holds a
    #: freshly computed ranker, and asserting on whatever occupies that path fails
    #: for a correct reason while saying nothing about the chain. The assertion is
    #: NOT weakened: the located artifact must still hash to the frozen chain
    #: input, and the chain must still reproduce the accepted member byte for byte.
    CONTROL_CANDIDATES = (
        "outputs/dataset1-ensemble/result_ranker.csv",
        "reference/result_ranker.csv",
    )

    def setUp(self):
        self.test_csv = REPO / "data" / "data_A" / "dataset1" / "test.csv"
        self.train_csv = REPO / "data" / "data_A" / "dataset1" / "train.csv"
        for asset in (self.test_csv, self.train_csv):
            if not asset.exists():
                self.skipTest(f"local asset not available: {asset}")

    def _frozen_control(self) -> Path:
        """The frozen chain input, located by hash. Only the chain-reproduction
        test needs it, so the lookup must not gate the other tests in this class."""
        want = PRODUCTION["dataset1"]["chain_input"]["sha256"]
        seen = []
        for rel in self.CONTROL_CANDIDATES:
            candidate = REPO / rel
            if not candidate.exists():
                seen.append(f"{rel}: absent")
                continue
            got = sha256_file(candidate)
            if got == want:
                return candidate
            seen.append(f"{rel}: {got[:16]}...")
        self.skipTest(
            f"the frozen dataset1 chain input ({want[:16]}...) is "
            f"not available locally; searched {', '.join(seen)}"
        )

    def test_immutable_inputs_are_unchanged(self):
        for rel, want in PRODUCTION["immutable_inputs"].items():
            path = REPO / rel
            if not path.exists():
                self.skipTest(f"local asset not available: {path}")
            self.assertEqual(sha256_file(path), want, rel)

    def test_chain_reproduces_the_accepted_dataset1_member(self):
        import pandas as pd

        from strategies.registry import DS1_POSTPROCESSOR_CHAIN
        from strategies.shared.frozen_ops import read_score_matrix, write_score_matrix

        control = self._frozen_control()
        self.assertEqual(sha256_file(control), PRODUCTION["dataset1"]["chain_input"]["sha256"])
        test = pd.read_csv(self.test_csv)
        train = pd.read_csv(self.train_csv)
        scores = read_score_matrix(control)

        anchors = {s["strategy_id"]: s for s in PRODUCTION["dataset1"]["strategy_chain"]}
        for sid, _module, apply_fn in DS1_POSTPROCESSOR_CHAIN:
            scores, info = apply_fn(scores, test, train)
            self.assertEqual(info["actions"], anchors[sid]["actions"], f"{sid} action count")
            self.assertEqual(info["top1_changes"], anchors[sid]["top1_changes"], sid)

        import tempfile

        with tempfile.TemporaryDirectory() as td:
            digest = write_score_matrix(scores, Path(td) / "dataset1.csv")
        self.assertEqual(
            digest,
            PRODUCTION["dataset1"]["member_sha256"],
            "the tracked chain no longer reproduces the accepted member",
        )


class Dataset2DecoderReproductionTest(unittest.TestCase):
    """The dataset2 counterpart: tracked code must regenerate the member exactly."""

    def setUp(self):
        self.base_zip = REPO / PRODUCTION["dataset2"]["decoder_input"]["path"]
        self.test_csv = REPO / "data" / "data_A" / "dataset2" / "test.csv"
        for asset in (self.base_zip, self.test_csv):
            if not asset.exists():
                self.skipTest(f"local asset not available: {asset}")

    def test_decoder_reproduces_the_accepted_dataset2_member(self):
        import io

        import numpy as np
        import pandas as pd

        from strategies.ds2 import cross_time_exclusivity as xte

        decoder_input = PRODUCTION["dataset2"]["decoder_input"]
        with zipfile.ZipFile(self.base_zip) as archive:
            payload = archive.read(decoder_input["member"])
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            decoder_input["sha256"],
            "the base matrix is not the one the accepted member was built from",
        )

        scores = pd.read_csv(io.BytesIO(payload), header=None).to_numpy(np.float64)
        test = pd.read_csv(self.test_csv)
        self.assertEqual(scores.shape, tuple(PRODUCTION["dataset2"]["shape"]))

        _treatment, info = xte.apply(scores, test)

        self.assertEqual(
            xte.census_mismatches(info["census"]),
            {},
            "the physical census differs from the accepted deployment",
        )
        want = PRODUCTION["dataset2"]["final_decoder"]["effect_on_member"]
        self.assertEqual(info["actions"], want["rows_changed"])
        self.assertEqual(info["cells_changed"], want["cells_changed"])
        self.assertEqual(info["top1_changes"], want["top1_changes"])
        self.assertEqual(info["pairs_collapsed_to_tie"], want["pairs_collapsed_to_tie"])

        rebuilt = xte.swap_score_tokens(
            payload, info["action_rows"], info["c1_col"], info["c2_col"], info["census"]["columns"]
        )
        self.assertEqual(len(rebuilt), PRODUCTION["dataset2"]["member_bytes"])
        self.assertEqual(
            hashlib.sha256(rebuilt).hexdigest(),
            PRODUCTION["dataset2"]["member_sha256"],
            "the tracked decoder no longer reproduces the accepted member",
        )

    def test_rebuilt_member_satisfies_the_submission_schema(self):
        import io

        import numpy as np
        import pandas as pd

        from strategies.ds2 import cross_time_exclusivity as xte

        with zipfile.ZipFile(self.base_zip) as archive:
            payload = archive.read(PRODUCTION["dataset2"]["decoder_input"]["member"])
        scores = pd.read_csv(io.BytesIO(payload), header=None).to_numpy(np.float64)
        test = pd.read_csv(self.test_csv)
        treatment, _ = xte.apply(scores, test)

        rows, columns = PRODUCTION["dataset2"]["shape"]
        self.assertEqual(treatment.shape, (rows, columns))
        self.assertEqual(len(test), rows, "one prediction row per test query")
        self.assertTrue(np.isfinite(treatment).all(), "a non-finite score would be invalid")
        self.assertGreaterEqual(treatment.min(), 0.0, "scores must lie in [0, 1]")
        self.assertLessEqual(treatment.max(), 1.0, "scores must lie in [0, 1]")
        # Row-wise permutation: the decode may reorder scores within a row but
        # must never add, drop or rewrite one.
        self.assertTrue(np.array_equal(np.sort(treatment, axis=1), np.sort(scores, axis=1)))


class EntryPointTest(unittest.TestCase):
    """Both scenarios must expose a working, documented entry path."""

    def test_unified_entry_point_exists(self):
        self.assertTrue((REPO / "main.py").exists())

    def test_describe_stage_runs_for_both_datasets(self):
        import subprocess

        result = subprocess.run(
            [sys.executable, "main.py", "--stage", "describe"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for expected in (
            "dataset1",
            "dataset2",
            "xte_cross_time_exclusivity_decode",
            PRODUCTION["accepted_archive"]["sha256"],
        ):
            self.assertIn(expected, result.stdout)

    def test_every_dataset_declares_a_build_command(self):
        for dataset in ("dataset1", "dataset2"):
            command = PRODUCTION[dataset]["build_command"]
            self.assertTrue(
                command.startswith("python src/build_"),
                f"{dataset} build command is not a runnable entry point: {command}",
            )
            script = command.split()[1]
            self.assertTrue((REPO / script).exists(), f"missing {script}")

    def test_environment_specification_is_pinned(self):
        spec = REPO / "environment.yaml"
        if not spec.exists():
            self.skipTest("environment.yaml not present")
        text = spec.read_text(encoding="utf-8")
        for package in ("python", "jittor", "numpy", "pandas", "scikit-learn", "lightgbm"):
            self.assertIn(package, text, f"{package} is not pinned in environment.yaml")

    def test_third_party_imports_of_packaged_modules_are_declared(self):
        """Every third-party module the PACKAGED sources import must be declared.

        This guards a real defect: a canonical module importing a distribution
        that the environment specification does not install fails at import on
        the target host, so the stage cannot run and no member is produced.

        The audited set is the submission package itself, taken from
        ``stage_package.CODE_MEMBERS`` -- the same list that builds the archive,
        so the test cannot drift from what ships. The repository also carries two
        historical trainers, ``reference/pytorch/train_line.py`` and ``reference/pytorch/train_bpr.py``, which
        import torch and are deliberately NOT packaged; they are reference-only
        and unreachable from the canonical stage graph, and
        ``PackagedSetBoundaryTest`` below pins that separation. Auditing them here
        would demand a torch declaration for a dependency the submission does not
        have.
        """
        import ast

        stdlib = set(getattr(sys, "stdlib_module_names", ()))
        packaged = packaged_python_sources()
        self.assertTrue(packaged, "the packaged source list is empty")
        local = {p.stem for p in (REPO / "src").rglob("*.py")}
        local |= {"strategies", "tools"}

        imported: set[str] = set()
        for path in packaged:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.level == 0 and node.module:
                        imported.add(node.module.split(".")[0])
        third_party = {m for m in imported if m not in stdlib and m not in local}
        self.assertIn(
            "jittor",
            third_party,
            "the scan found no jittor import in the packaged sources, so "
            "its negatives prove nothing",
        )

        declared = declared_dependencies()
        # Import name to distribution name where they differ.
        distribution = {"sklearn": "scikit-learn", "yaml": "pyyaml"}
        missing = sorted(m for m in third_party if distribution.get(m, m) not in declared)
        self.assertEqual(missing, [], f"imported but not declared as a dependency: {missing}")

    def test_the_declaration_scan_rejects_an_undeclared_import(self):
        """Anti-vacuity: the same comparison must reject a module nobody declares."""
        declared = declared_dependencies()
        self.assertNotIn("tensorflow", declared)
        self.assertIn("jittor", declared)

    def test_a_declaration_in_a_comment_does_not_count(self):
        """A dependency named only in prose must not satisfy the audit.

        The declaration text is the functional lines of the two specifications,
        with comments stripped, so re-adding an explanatory mention of a
        framework cannot silently re-satisfy the import audit.
        """
        for name in ("requirements.txt", "environment.yaml"):
            raw = (REPO / name).read_text(encoding="utf-8")
            commented = [ln for ln in raw.splitlines() if ln.strip().startswith("#")]
            self.assertTrue(commented, f"{name} has no comments to distinguish")
            for line in commented:
                self.assertNotIn(
                    line.strip(),
                    declared_dependencies().splitlines(),
                    f"{name}: a comment line reached the declaration text",
                )


class PackagedSetBoundaryTest(unittest.TestCase):
    """The submission package boundary, and what deliberately sits outside it."""

    #: Historical trainers kept for provenance. They import torch, are not on the
    #: canonical path, and must never enter the package.
    REFERENCE_ONLY = ("reference/pytorch/train_line.py", "reference/pytorch/train_bpr.py")

    def test_the_package_declares_twenty_six_code_files(self):
        self.assertEqual(len(packaged_code_members()), 26)

    def test_reference_only_trainers_are_not_packaged(self):
        packaged = {rel for rel, _, _ in packaged_code_members()}
        for name in self.REFERENCE_ONLY:
            self.assertTrue(
                (REPO / name).exists(), f"{name} is expected to exist as a reference-only module"
            )
            self.assertNotIn(name, packaged, f"{name} must not be packaged")

    def test_reference_only_trainers_are_unreachable_from_the_stage_graph(self):
        graph = (REPO / "src" / "canonical_pipeline.py").read_text(encoding="utf-8")
        for name in self.REFERENCE_ONLY:
            self.assertNotIn(Path(name).name, graph, f"the canonical stage graph references {name}")

    def test_only_the_jittor_trainers_are_used_for_embeddings(self):
        graph = (REPO / "src" / "canonical_pipeline.py").read_text(encoding="utf-8")
        self.assertIn("train_line_jt.py", graph)
        self.assertIn("train_bpr_jt.py", graph)
        self.assertEqual(
            set(re.findall(r"train_\w+\.py", graph)), {"train_line_jt.py", "train_bpr_jt.py"}
        )

    def test_no_packaged_module_imports_torch(self):
        import ast

        offenders = []
        for path in packaged_python_sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                if any(n.split(".")[0] == "torch" for n in names):
                    offenders.append(path.relative_to(REPO).as_posix())
        self.assertEqual(
            sorted(set(offenders)),
            [],
            f"packaged modules importing torch: {sorted(set(offenders))}",
        )

    def test_the_dependency_specifications_never_name_torch(self):
        for name in ("requirements.txt", "environment.yaml"):
            text = (REPO / name).read_text(encoding="utf-8").lower()
            self.assertNotIn(
                "torch",
                text,
                f"{name} mentions torch; the submitted specification must "
                f"describe only what the packaged pipeline needs",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
