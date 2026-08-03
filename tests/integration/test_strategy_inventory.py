# -*- coding: utf-8 -*-
"""Consistency tests for the strategy inventory itself.

Cheap, dependency-free, always runs. Its job is to stop the inventory drifting
from the code and the production manifest.
"""

from __future__ import annotations

import collections
import json
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

INVENTORY = json.loads((REPO / "docs" / "strategy_inventory.json").read_text(encoding="utf-8"))
PRODUCTION = json.loads((REPO / "configs" / "production.json").read_text(encoding="utf-8"))


class InventoryConsistencyTest(unittest.TestCase):
    def test_declared_counts_match_the_records(self):
        actual = collections.Counter(s["lifecycle_status"] for s in INVENTORY["strategies"])
        for status, declared in INVENTORY["counts"].items():
            if status == "total":
                self.assertEqual(declared, len(INVENTORY["strategies"]))
            else:
                self.assertEqual(actual.get(status, 0), declared, status)

    def test_strategy_ids_are_unique(self):
        ids = [s["strategy_id"] for s in INVENTORY["strategies"]]
        duplicates = [i for i, n in collections.Counter(ids).items() if n > 1]
        self.assertEqual(duplicates, [])

    def test_strategy_ids_are_stable_and_mechanism_oriented(self):
        """No round numbers, agent names, model names or seed VALUES in ids.

        'multi_seed_bpr_ensemble' is deliberately allowed: seed averaging is the
        mechanism itself, which the naming standard admits as scientifically
        essential. What is banned is a concrete seed or round pinned into a name.
        """
        import re

        banned_tokens = ("round", "codex", "fable", "opus", "gpt")
        banned_patterns = (r"r\d{2}", r"seed\d+", r"_s\d+", r"v\d+$")
        for strategy in INVENTORY["strategies"]:
            sid = strategy["strategy_id"].lower()
            for token in banned_tokens:
                self.assertNotIn(token, sid, f"{sid} encodes {token!r}")
            for pattern in banned_patterns:
                self.assertIsNone(re.search(pattern, sid), f"{sid} matches {pattern!r}")

    def test_every_record_carries_the_required_fields(self):
        for strategy in INVENTORY["strategies"]:
            for field in ("strategy_id", "dataset", "mechanism", "lifecycle_status", "confidence"):
                self.assertIn(field, strategy, strategy.get("strategy_id"))

    def test_closed_strategies_state_why(self):
        for strategy in INVENTORY["strategies"]:
            if strategy["lifecycle_status"] == "CLOSED":
                self.assertIn("closure_reason", strategy, strategy["strategy_id"])

    def test_production_chain_strategies_are_all_shipped_active(self):
        by_id = {s["strategy_id"]: s for s in INVENTORY["strategies"]}
        for dataset in ("dataset1", "dataset2"):
            for stage in PRODUCTION[dataset]["strategy_chain"]:
                sid = stage["strategy_id"]
                self.assertIn(sid, by_id, f"{sid} is in production but absent from the inventory")
                self.assertEqual(by_id[sid]["lifecycle_status"], "SHIPPED_ACTIVE", sid)

    def test_shipped_strategies_carry_online_evidence(self):
        for strategy in INVENTORY["strategies"]:
            if strategy["lifecycle_status"].startswith("SHIPPED"):
                self.assertTrue(
                    strategy.get("online_evidence"),
                    f"{strategy['strategy_id']} claims shipped with no online evidence",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
