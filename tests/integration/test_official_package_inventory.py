# -*- coding: utf-8 -*-
"""What may and may not go into the official ``code/`` package.

The package is not built here. What is pinned here is the inventory it will be
built from, so that the decisions survive independently of whoever runs the
build: which files are canonical, which are excluded as PyTorch-only, and what
the environment specification is allowed to claim about JittorGeometric.

The exclusion list is checked against a live scan rather than trusted. If a new
file starts importing torch it appears in the scan, and the test fails until the
list is updated deliberately -- which is the point.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"

#: Every module the organiser-facing runtime needs, in dependency order.
CANONICAL_RUNTIME = [
    "run_all.py",
    "main.py",
    "src/canonical_pipeline.py",
    "src/stage_contract.py",
    "src/pipeline_common.py",
    "src/train_line_jt.py",
    "src/train_bpr_jt.py",
    "src/ensemble_predict.py",
    "src/ranker_ds1.py",
    "src/ranker_basket_ds2.py",
    "src/ds2_basket_featurizer.py",
    "src/ds2_mf_basket_pack.py",
    "src/crf_promote.py",
    "src/build_ds1_member.py",
    "src/build_ds2_member.py",
    "src/strategies/registry.py",
    "src/strategies/shared/frozen_ops.py",
    "src/strategies/ds1/source_slate_recurrence.py",
    "src/strategies/ds1/test_graph_reciprocity.py",
    "src/strategies/ds2/cross_time_exclusivity.py",
]

#: Required alongside the code.
REQUIRED_ENVIRONMENT_FILES = ["environment.yaml", "requirements.txt"]

#: Proposed staging exclusions for PyTorch. The historical trainers keep their
#: provenance value in git and stay in the working tree; they are not shipped,
#: because the canonical runtime is Jittor and shipping a second backend in a
#: Jittor-mandated competition is an ambiguity, not a feature.
TORCH_EXCLUSIONS = [
    "src/train_line.py",
    "src/train_bpr.py",
    "tools/diagnostics/compare_backends.py",
]

#: Excluded for reasons other than torch: audit evidence, scratch work, local
#: notes, competition data and frozen artifacts.
NON_CODE_EXCLUSIONS = [
    "artifacts_durable/", "audit_exports/", "scratchpad/", "docs_local/",
    "data/", "outputs/", "reference/",
]

JITTOR_GEOMETRIC_COMMIT = "ff7d8ffac7bf3d95cc1962e091c52dc5737492d4"


def module_imports(path: Path) -> set[str]:
    """Every module named by an import anywhere in the file, guarded or not."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def files_importing(package: str) -> set[str]:
    """Repository-relative paths of every tracked Python file importing ``package``."""
    found = set()
    for directory in (SRC, REPO / "tools"):
        for path in directory.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            if package in module_imports(path):
                found.add(path.relative_to(REPO).as_posix())
    if package in module_imports(REPO / "main.py"):
        found.add("main.py")
    return found


class CanonicalInventory(unittest.TestCase):
    def test_every_canonical_runtime_file_exists_and_is_tracked(self):
        import subprocess

        tracked = set(subprocess.run(["git", "-C", str(REPO), "ls-files"],
                                     capture_output=True, text=True).stdout.split())
        for name in CANONICAL_RUNTIME + REQUIRED_ENVIRONMENT_FILES:
            self.assertTrue((REPO / name).is_file(), f"{name} is missing")
            self.assertIn(name, tracked, f"{name} is not tracked by git")

    def test_no_canonical_runtime_dependency_lives_outside_the_repository(self):
        """A canonical file in a worktree, scratchpad or evidence tree is a blocker."""
        forbidden = ("scratchpad", "artifacts_durable", "audit_exports", "docs_local",
                     "jlp-p1-diag-wt", "jlp-p1-int-wt", "jlp-schemeC-wt")
        for name in CANONICAL_RUNTIME:
            resolved = (REPO / name).resolve().as_posix()
            for token in forbidden:
                self.assertNotIn(f"/{token}/", resolved, f"{name} resolves into {token}")

    def test_no_canonical_file_hardcodes_a_host_absolute_path(self):
        suspicious = ("/root/autodl-tmp", "C:\\\\Users", "/root/jittor", "F:\\\\")
        for name in CANONICAL_RUNTIME:
            text = (REPO / name).read_text(encoding="utf-8")
            for token in suspicious:
                self.assertNotIn(token.replace("\\\\", "\\"), text,
                                 f"{name} hard-codes a host path")


class TorchIsExcluded(unittest.TestCase):
    def test_the_exclusion_list_covers_every_torch_importing_file(self):
        """Live scan, so a new torch importer cannot slip past the list."""
        importers = files_importing("torch")
        self.assertTrue(importers, "the scanner found no torch importers at all")
        self.assertEqual(importers - set(TORCH_EXCLUSIONS), set(),
                         "a file imports torch and is not on the staging exclusion list")
        # The diagnostic driver is excluded too. It reaches torch by spawning the
        # historical trainers rather than by importing it, so the scan alone
        # would not catch it and the list has to name it explicitly.
        self.assertIn("tools/diagnostics/compare_backends.py", TORCH_EXCLUSIONS)
        self.assertEqual(set(TORCH_EXCLUSIONS) - importers,
                         {"tools/diagnostics/compare_backends.py"},
                         "the exclusion list names a file with no torch relationship")

    def test_no_canonical_runtime_file_imports_torch(self):
        for name in CANONICAL_RUNTIME:
            self.assertNotIn("torch", module_imports(REPO / name), name)

    def test_the_exclusions_and_the_canonical_inventory_do_not_overlap(self):
        self.assertEqual(set(TORCH_EXCLUSIONS) & set(CANONICAL_RUNTIME), set())

    def test_the_diagnostic_tool_declares_itself_out_of_the_package(self):
        text = (REPO / "tools/diagnostics/compare_backends.py").read_text(encoding="utf-8")
        self.assertIn("NOT PART OF THE SUBMISSION", text)
        self.assertIn("staging exclusion", text)
        self.assertIn("official_package_inventory.md", text)

    def test_the_environment_does_not_require_torch(self):
        for name in REQUIRED_ENVIRONMENT_FILES:
            for line in (REPO / name).read_text(encoding="utf-8").splitlines():
                stripped = line.strip().lstrip("-").strip()
                if stripped.startswith("#") or not stripped:
                    continue
                self.assertNotEqual(stripped.split("=")[0].strip(), "torch", name)


class JittorGeometricContract(unittest.TestCase):
    def test_it_is_pinned_by_commit_in_both_specifications(self):
        for name in REQUIRED_ENVIRONMENT_FILES:
            text = (REPO / name).read_text(encoding="utf-8")
            self.assertIn("JittorGeometric", text, name)
            self.assertIn(JITTOR_GEOMETRIC_COMMIT, text,
                          f"{name} does not pin the JittorGeometric commit")

    def test_no_canonical_module_imports_it(self):
        """The documentation claim must match the code, in this direction too."""
        importers = files_importing("jittor_geometric")
        self.assertEqual(importers, set(),
                         f"the chain now imports jittor_geometric: {sorted(importers)}")

    def test_the_documented_claim_matches_the_measurement(self):
        text = (REPO / "environment.yaml").read_text(encoding="utf-8")
        imported = bool(files_importing("jittor_geometric"))
        if imported:
            self.assertIn("does import", text)
        else:
            self.assertIn("does NOT import it", text)

    def test_the_scanner_would_notice_an_import(self):
        """Anti-vacuity: the same scan finds jittor, which the chain does import."""
        self.assertTrue(files_importing("jittor"),
                        "the scanner reports no jittor importers, so its negatives "
                        "prove nothing")
        self.assertIn("src/train_line_jt.py", files_importing("jittor"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
