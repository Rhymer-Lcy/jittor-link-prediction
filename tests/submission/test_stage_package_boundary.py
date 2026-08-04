# -*- coding: utf-8 -*-
"""The staging tool must never delete a directory it does not own.

``stage_package.build`` removes the staging root before rebuilding it. These
tests pin the boundary of that removal: it reaches exactly one path, it refuses
to run over a tree the tool did not produce, and it refuses to run over an
existing tree at all unless the caller asks for a replacement.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "submission"))

import stage_package as sp  # noqa: E402

TEAM = "team"
NAME = sp.package_name(TEAM)


class StagingRootConfinementTest(unittest.TestCase):
    """Only ``submission_staging/<package name>`` may be accepted."""

    def test_the_derived_root_is_accepted(self):
        resolved = sp.check_staging_root(sp.staging_root_for(NAME), NAME)
        self.assertEqual(resolved.resolve(), (sp.STAGING_ROOT / NAME).resolve())

    def test_an_external_directory_with_a_matching_name_is_refused(self):
        """A correct basename outside the staging root must not be enough."""
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / NAME
            outside.mkdir()
            with self.assertRaises(SystemExit) as caught:
                sp.check_staging_root(outside, NAME)
            self.assertIn("staging root must be", str(caught.exception))

    def test_a_sibling_of_the_staging_root_is_refused(self):
        with self.assertRaises(SystemExit):
            sp.check_staging_root(sp.STAGING_ROOT.parent / NAME, NAME)

    def test_a_deeper_path_inside_the_staging_root_is_refused(self):
        """Confinement is to one level, not to any descendant."""
        with self.assertRaises(SystemExit):
            sp.check_staging_root(sp.STAGING_ROOT / "nested" / NAME, NAME)

    def test_a_traversal_back_out_of_the_staging_root_is_refused(self):
        with self.assertRaises(SystemExit):
            sp.check_staging_root(sp.STAGING_ROOT / ".." / NAME, NAME)

    def test_the_package_name_rejects_a_separator(self):
        """The name feeding the staging path may not carry a path component."""
        for bad in ("../evil", "a/b", "..", "."):
            with self.assertRaises(ValueError):
                sp.package_name(bad)


class StagingRootReplacementTest(unittest.TestCase):
    """An existing tree is destroyed only deliberately, and only if it is ours."""

    def test_a_missing_root_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / NAME
            sp.prepare_staging_root(root, replace=False)
            self.assertTrue(root.is_dir())

    def test_an_existing_root_is_refused_without_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / NAME
            root.mkdir()
            keep = root / "keep.txt"
            keep.write_text("payload", encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                sp.prepare_staging_root(root, replace=False)
            self.assertIn("--replace", str(caught.exception))
            self.assertTrue(keep.is_file(), "refusal must not delete anything")

    def test_replace_refuses_a_tree_this_tool_did_not_produce(self):
        """Without a staging manifest the directory is somebody else's data."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / NAME
            root.mkdir()
            keep = root / "keep.txt"
            keep.write_text("payload", encoding="utf-8")
            with self.assertRaises(SystemExit) as caught:
                sp.prepare_staging_root(root, replace=True)
            self.assertIn("STAGING_MANIFEST.json", str(caught.exception))
            self.assertTrue(keep.is_file(), "refusal must not delete anything")

    def test_replace_rebuilds_a_tree_this_tool_produced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / NAME
            root.mkdir()
            (root / "STAGING_MANIFEST.json").write_text(json.dumps([]), encoding="utf-8")
            stale = root / "stale.txt"
            stale.write_text("old", encoding="utf-8")
            sp.prepare_staging_root(root, replace=True)
            self.assertTrue(root.is_dir())
            self.assertFalse(stale.exists())
            self.assertFalse((root / "STAGING_MANIFEST.json").exists())


class AntiVacuityTest(unittest.TestCase):
    """The guards above must be reachable from the code path that deletes."""

    def test_build_delegates_to_the_guarded_preparation(self):
        source = Path(sp.__file__).read_text(encoding="utf-8")
        body = source.split("def build(", 1)[1]
        self.assertIn("prepare_staging_root(root, replace)", body)

    def test_build_does_not_remove_a_tree_directly(self):
        source = Path(sp.__file__).read_text(encoding="utf-8")
        body = source.split("def build(", 1)[1].split("\ndef ", 1)[0]
        self.assertNotIn("rmtree", body)

    def test_only_the_guarded_helper_removes_a_tree(self):
        source = Path(sp.__file__).read_text(encoding="utf-8")
        self.assertEqual(source.count("shutil.rmtree("), 1)
        guarded = source.split("def prepare_staging_root(", 1)[1].split("\ndef ", 1)[0]
        self.assertIn("shutil.rmtree(", guarded)


if __name__ == "__main__":
    unittest.main()
