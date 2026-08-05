# -*- coding: utf-8 -*-
"""The public licence, attribution and identity policy.

The repository is the work of more than one contributor, so three things must
stay true together: the licence names the project rather than a person, NOTICE
records who contributed the material that is not the project's own, and the
files retaining that material say so at the source.

These tests read the working tree only. Git history is immutable and is not a
subject of assertion here.
"""

from __future__ import annotations

import ast
import re
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The single approved copyright application notice.
COPYRIGHT_NOTICE = "Copyright 2026 Rhymer-Lcy and contributors"

#: The public identities. No real-world name belongs in the tracked tree.
PROJECT_IDENTITY = "Rhymer-Lcy"
CONTRIBUTOR_IDENTITY = "shadiaosjh"

#: Files that retain material expression from the contributed implementation.
ATTRIBUTED_SOURCES = (
    "reference/pytorch/train_line.py",
    "src/train_line_jt.py",
    "src/pipeline_common.py",
)

#: Informal wording replaced by the provenance statements. It must not return.
RETIRED_WORDING = (
    "teammate's idea",
    "original 1.py header",
)

COPYRIGHT_LINE = re.compile(r"^\s*Copyright\s+\d{4}\s+\S.*$", re.MULTILINE)

TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".ini", ".sh", ""}


def tracked_text_files() -> list[Path]:
    """Every tracked file that could carry a notice, resolved from git."""
    listing = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    files = []
    for name in listing:
        path = REPO / name
        if path.suffix.lower() in TEXT_SUFFIXES and path.is_file():
            files.append(path)
    return files


def read(path: str) -> str:
    return (REPO / path).read_text(encoding="utf-8")


class LicenceNoticeTest(unittest.TestCase):
    def test_the_licence_carries_exactly_one_application_notice(self):
        found = COPYRIGHT_LINE.findall(read("LICENSE"))
        self.assertEqual(len(found), 1, f"expected one application notice, found {len(found)}")
        self.assertEqual(found[0].strip(), COPYRIGHT_NOTICE)

    def test_the_licence_terms_are_unmodified_apache_two(self):
        text = read("LICENSE")
        self.assertIn("Apache License", text)
        self.assertIn("Version 2.0, January 2004", text)
        self.assertIn("http://www.apache.org/licenses/", text)

    def test_no_tracked_file_declares_a_different_copyright_holder(self):
        """A personal name must not reappear anywhere as a copyright holder."""
        for path in tracked_text_files():
            for line in COPYRIGHT_LINE.findall(path.read_text(encoding="utf-8", errors="replace")):
                self.assertEqual(
                    line.strip(),
                    COPYRIGHT_NOTICE,
                    f"{path.relative_to(REPO).as_posix()} declares another copyright holder",
                )


class NoticeFileTest(unittest.TestCase):
    def test_the_notice_file_exists(self):
        self.assertTrue((REPO / "NOTICE").is_file(), "NOTICE is required beside LICENSE")

    def test_the_notice_names_both_public_identities(self):
        text = read("NOTICE")
        self.assertIn(CONTRIBUTOR_IDENTITY, text)
        self.assertIn(PROJECT_IDENTITY, text)

    def test_the_notice_lists_every_attributed_source(self):
        text = read("NOTICE")
        for path in ATTRIBUTED_SOURCES:
            self.assertIn(path, text, f"NOTICE does not list {path}")

    def test_the_notice_claims_no_exclusive_authorship(self):
        self.assertIn("No single contributor is the exclusive author", read("NOTICE"))

    def test_the_notice_states_the_licence(self):
        self.assertIn("Apache License", read("NOTICE"))


class ReadmeAttributionTest(unittest.TestCase):
    def test_the_readme_documents_the_licence_and_attribution(self):
        text = read("README.md")
        self.assertIn("## Licence and attribution", text)
        for token in ("Apache License", "NOTICE", PROJECT_IDENTITY, CONTRIBUTOR_IDENTITY):
            self.assertIn(token, text)


class SourceProvenanceTest(unittest.TestCase):
    def test_every_attributed_source_names_the_contributor(self):
        for path in ATTRIBUTED_SOURCES:
            self.assertIn(
                CONTRIBUTOR_IDENTITY,
                read(path),
                f"{path} retains contributed expression and must say so",
            )

    def test_the_retired_informal_wording_does_not_return(self):
        for path in tracked_text_files():
            text = path.read_text(encoding="utf-8", errors="replace")
            for phrase in RETIRED_WORDING:
                self.assertNotIn(
                    phrase, text, f"{path.relative_to(REPO).as_posix()} revives {phrase!r}"
                )

    def test_attribution_is_documentation_only(self):
        """Provenance is prose. It must not become a runtime value."""
        for path in ATTRIBUTED_SOURCES:
            tree = ast.parse(read(path))
            # A docstring is a string expression in the first statement position
            # of a module, class or function. Identify those exact nodes, since
            # ast.get_docstring returns cleaned text that no longer compares
            # equal to the constant it came from.
            docstring_nodes = set()
            for node in ast.walk(tree):
                if not isinstance(
                    node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    continue
                first = node.body[0] if node.body else None
                if (
                    isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)
                ):
                    docstring_nodes.add(id(first.value))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                if CONTRIBUTOR_IDENTITY in node.value and id(node) not in docstring_nodes:
                    self.fail(f"{path} carries the contributor name in a runtime string")


class PackageInventoryTest(unittest.TestCase):
    """Attribution must not have disturbed the declared submission membership."""

    def members(self) -> list[str]:
        source = read("tools/submission/stage_package.py")
        for node in ast.parse(source).body:
            target = None
            if isinstance(node, ast.AnnAssign):
                target = getattr(node.target, "id", None)
            elif isinstance(node, ast.Assign) and node.targets:
                target = getattr(node.targets[0], "id", None)
            if target == "CODE_MEMBERS":
                return [entry[0] for entry in ast.literal_eval(node.value)]
        self.fail("stage_package.CODE_MEMBERS not found")

    def test_the_membership_list_is_an_explicit_literal_of_twenty_six(self):
        self.assertEqual(len(self.members()), 26)

    def test_the_attributed_package_members_are_still_declared(self):
        members = self.members()
        for path in ("src/train_line_jt.py", "src/pipeline_common.py"):
            self.assertIn(path, members)

    def test_the_notice_is_not_shipped_as_a_code_member(self):
        """NOTICE belongs to the repository, not to the organiser code tree."""
        self.assertNotIn("NOTICE", self.members())


if __name__ == "__main__":
    unittest.main()
