# -*- coding: utf-8 -*-
"""Every dataset1 production-chain module must name its run directories the same
way: through ``pipeline_common.line_run_dir`` / ``bpr_run_dir``.

This is the dataset1 counterpart of ``test_ds2_line_run_dir_contract``. Unlike
dataset2, **the dataset1 contract is already correct and no repair is needed**:
``train_line_jt.py`` writes through ``pc.line_run_dir(DATASET, ...)`` and
``ranker_ds1.py`` reads through ``tl.line_run_dir("dataset1")``, so producer and
consumer cannot disagree. These tests exist to pin that agreement, not to fix it.

The history worth pinning. At ``2d16036`` the ranker hard-coded two different
shapes -- ``outputs/dataset1-novirt-tmax1.1548e+08`` for the cut embedding but a
bare ``outputs/dataset1`` for the serve embedding. ``7b234e5`` moved both onto
the contract, which appends ``-novirt``. A retained local
``outputs/dataset1`` therefore predates the contract: it is a legacy artifact
name, not a directory the contract can produce for the serve embedding, and it
must never be treated as the canonical Jittor LINE output.

These tests fix the contract at the source level, where it can be checked without
competition data, without local artifacts and without executing the chain.
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import pipeline_common as pc  # noqa: E402

DATASET = "dataset1"
CUT = 115480000.0

#: The trainer that produces the embeddings, and the chain module that consumes
#: them. configs/production.json names ranker_ds1.py as dataset1 chain order 0.
PRODUCER = "train_line_jt.py"
BPR_PRODUCER = "train_bpr_jt.py"
CHAIN_CONSUMERS = ("ranker_ds1.py",)

CONTRACT_SUFFIX = re.compile(
    r"^dataset[12]"
    r"(-bpr|-innov|-novirt|-negpop|-holdout|-s\d+|-d\d+|-t[\d.]+|-tmax[\d.e+]+)*$")


def source(name: str) -> str:
    return (REPO / "src" / name).read_text(encoding="utf-8")


def owned_by_the_contract(literal: str) -> bool:
    return bool(CONTRACT_SUFFIX.match(literal))


def literal_path_joins(name: str) -> list[tuple[int, str]]:
    """Every ``<expr> / "<literal>"`` whose literal names a CONTRACT-owned run dir."""
    found = []
    for node in ast.walk(ast.parse(source(name))):
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, str)
                and owned_by_the_contract(node.right.value)):
            found.append((node.lineno, node.right.value))
    return found


class DetectorTest(unittest.TestCase):
    """Without this the suite could pass by matching nothing at all."""

    def test_detector_classifies_dataset1_ownership_correctly(self):
        for owned in ("dataset1", "dataset1-novirt", "dataset1-bpr-t0.25",
                      "dataset1-bpr-t0.25-s31337", "dataset1-novirt-tmax1.1548e+08",
                      "dataset1-bpr-innov-t0.25"):
            with self.subTest(owned=owned):
                self.assertTrue(owned_by_the_contract(owned))
        # Sibling OUTPUT directories are not run directories.
        for free in ("dataset1-ensemble", "dataset1-p1-rebuild", "dataset2-crf"):
            with self.subTest(free=free):
                self.assertFalse(owned_by_the_contract(free))

    def test_the_detector_finds_the_historical_defect_shape(self):
        # The exact expression 2d16036 used for the serve embedding. If this
        # stopped being detected the chain test below would be vacuous.
        tree = ast.parse('REPO / "outputs" / "dataset1"')
        hits = [n.right.value for n in ast.walk(tree)
                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)
                and isinstance(n.right, ast.Constant)
                and owned_by_the_contract(n.right.value)]
        self.assertEqual(hits, ["dataset1"])


class ProductionChainContractTest(unittest.TestCase):
    def test_no_chain_consumer_spells_out_a_run_directory(self):
        for name in CHAIN_CONSUMERS:
            with self.subTest(module=name):
                self.assertEqual(
                    literal_path_joins(name), [],
                    f"{name} rebuilds a run directory from a literal instead of "
                    "calling pipeline_common.line_run_dir/bpr_run_dir")

    def test_producer_and_consumer_resolve_the_same_directory(self):
        self.assertIn("pc.line_run_dir(DATASET", source(PRODUCER))
        self.assertIn('tl.line_run_dir("dataset1")', source("ranker_ds1.py"))

    def test_the_contract_appends_the_novirt_suffix_by_default(self):
        # The property that makes a retained bare ``outputs/dataset1`` a legacy
        # artifact rather than the canonical serve directory.
        serve = pc.line_run_dir(DATASET)
        self.assertEqual(serve.name, "dataset1-novirt")
        self.assertNotEqual(serve.name, "dataset1")

    def test_serve_and_time_cut_line_directories_stay_distinct(self):
        serve = pc.line_run_dir(DATASET)
        cut = pc.line_run_dir(DATASET, time_max=CUT)
        self.assertNotEqual(serve, cut)
        self.assertTrue(cut.name.startswith("dataset1-novirt-tmax"))

    def test_the_ranker_reads_the_cut_line_directory_through_the_contract(self):
        self.assertIn('tl.line_run_dir("dataset1", time_max=CUT)', source("ranker_ds1.py"))

    def test_bpr_contract_matches_the_names_the_ranker_requests(self):
        # dataset1 BPR carries tau 0.25, unlike dataset2. If bpr_run_dir's naming
        # changed, the ranker would silently load a different embedding set.
        for seed in (42, 123, 777, 2024, 31337):
            with self.subTest(seed=seed):
                sfx = "" if seed == 42 else f"-s{seed}"
                self.assertEqual(
                    pc.bpr_run_dir(DATASET, seed=seed, tau_frac=0.25).name,
                    "dataset1-bpr-t0.25" + sfx)
                self.assertEqual(
                    pc.bpr_run_dir(DATASET, seed=seed, tau_frac=0.25, time_max=CUT).name,
                    "dataset1-bpr-t0.25-tmax1.1548e+08" + sfx)

    def test_innovation_bpr_directory_is_distinct_from_the_plain_one(self):
        innov = pc.bpr_run_dir(DATASET, tau_frac=0.25, innov=True)
        plain = pc.bpr_run_dir(DATASET, tau_frac=0.25)
        self.assertEqual(innov.name, "dataset1-bpr-innov-t0.25")
        self.assertNotEqual(innov, plain)

    def test_the_bpr_producer_also_uses_the_contract(self):
        self.assertIn("bpr_run_dir", source(BPR_PRODUCER))


class ProductionManifestAgreementTest(unittest.TestCase):
    def test_the_manifest_names_the_ranker_as_the_dataset1_chain_producer(self):
        import json
        cfg = json.loads((REPO / "configs" / "production.json").read_text(encoding="utf-8"))
        chain = cfg["dataset1"]["strategy_chain"]
        base = next(s for s in chain if s["order"] == 0)
        self.assertEqual(Path(base["implementation"]).name, "ranker_ds1.py")
        self.assertIn("src/train_line.py", base["upstream"])


if __name__ == "__main__":
    unittest.main()
