# -*- coding: utf-8 -*-
"""Integration tests against the accepted submission artifacts.

These need local, git-ignored assets (the accepted archive, the ranker output
and the raw competition data). Every test skips with an explicit reason when its
asset is missing, so a clean checkout still runs the suite green.
"""

from __future__ import annotations

import hashlib
import json
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


class ProductionManifestTest(unittest.TestCase):
    """Pure-config checks -- these always run."""

    def test_components_sum_to_the_accepted_total(self):
        ds1 = PRODUCTION["dataset1"]["component_score"]
        ds2 = PRODUCTION["dataset2"]["component_score"]
        self.assertEqual(ds1 + ds2, PRODUCTION["accepted_total_score"])

    def test_chain_order_matches_the_registry(self):
        from strategies.registry import active_ds1_chain_ids
        configured = [s["strategy_id"] for s in PRODUCTION["dataset1"]["strategy_chain"]
                      if s["order"] > 0]
        self.assertEqual(configured, active_ds1_chain_ids())

    def test_every_referenced_implementation_exists(self):
        for dataset in ("dataset1", "dataset2"):
            for stage in PRODUCTION[dataset]["strategy_chain"]:
                path = REPO / stage["implementation"]
                self.assertTrue(path.exists(), f"missing implementation {path}")

    def test_strategy_inventory_statuses_are_valid(self):
        from strategies.registry import LIFECYCLE_STATUSES
        inventory = json.loads(
            (REPO / "docs" / "strategy_inventory.json").read_text(encoding="utf-8"))
        for strategy in inventory["strategies"]:
            self.assertIn(strategy["lifecycle_status"], LIFECYCLE_STATUSES,
                          f"{strategy['strategy_id']} has an unknown status")

    def test_every_shipped_active_strategy_has_an_implementation(self):
        inventory = json.loads(
            (REPO / "docs" / "strategy_inventory.json").read_text(encoding="utf-8"))
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
            self.assertEqual([i.filename for i in z.infolist()],
                             PRODUCTION["accepted_archive"]["member_order"])
            self.assertEqual(z.comment, b"")
            self.assertEqual({i.compress_type for i in z.infolist()},
                             {PRODUCTION["accepted_archive"]["compress_method"]})
            self.assertFalse(any(i.is_dir() for i in z.infolist()))
            for member, dataset in (("dataset1.csv", "dataset1"), ("dataset2.csv", "dataset2")):
                digest = hashlib.sha256(z.read(member)).hexdigest()
                self.assertEqual(digest, PRODUCTION[dataset]["member_sha256"], member)


class ChainReproductionTest(unittest.TestCase):
    """The decisive test: tracked code must regenerate the accepted member exactly."""

    def setUp(self):
        self.control = REPO / PRODUCTION["dataset1"]["chain_input"]["path"]
        self.test_csv = REPO / "data" / "data_A" / "dataset1" / "test.csv"
        self.train_csv = REPO / "data" / "data_A" / "dataset1" / "train.csv"
        for asset in (self.control, self.test_csv, self.train_csv):
            if not asset.exists():
                self.skipTest(f"local asset not available: {asset}")

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

        self.assertEqual(sha256_file(self.control),
                         PRODUCTION["dataset1"]["chain_input"]["sha256"])
        test = pd.read_csv(self.test_csv)
        train = pd.read_csv(self.train_csv)
        scores = read_score_matrix(self.control)

        anchors = {s["strategy_id"]: s for s in PRODUCTION["dataset1"]["strategy_chain"]}
        for sid, _module, apply_fn in DS1_POSTPROCESSOR_CHAIN:
            scores, info = apply_fn(scores, test, train)
            self.assertEqual(info["actions"], anchors[sid]["actions"], f"{sid} action count")
            self.assertEqual(info["top1_changes"], anchors[sid]["top1_changes"], sid)

        import tempfile
        with tempfile.TemporaryDirectory() as td:
            digest = write_score_matrix(scores, Path(td) / "dataset1.csv")
        self.assertEqual(digest, PRODUCTION["dataset1"]["member_sha256"],
                         "the tracked chain no longer reproduces the accepted member")


if __name__ == "__main__":
    unittest.main(verbosity=2)
