# -*- coding: utf-8 -*-
"""Every dataset2 production-chain module must name the LINE run directory the
same way: through ``pipeline_common.line_run_dir``.

The graduated MF basket pack carried a literal ``outputs/dataset2`` from its
original build host. That predates the run-suffix contract, under which the
trainer writes the serve embedding to ``outputs/dataset2-novirt``. The literal
therefore resolved to a directory that exists but holds no export, so the stage
failed on the missing file after the full 244,056-query feature loop had already
run -- about three and a half hours in.

These tests fix the contract at the source level, where it can be checked without
competition data, without local artifacts and without executing the chain.

Scope note: two modules outside the production strategy chain
(``ranker_basket_ab_ds2.py``, ``footprint_ab_probe.py``) still rebuild the name
independently. They carry the same latent defect and are recorded by
``test_out_of_chain_sites_are_a_known_finding`` so the divergence cannot be lost,
but repairing them is not part of this correction.
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

DATASET = "dataset2"

#: Modules named by configs/production.json as the dataset2 strategy chain, plus
#: the trainer that produces the embedding they consume.
PRODUCER = "train_line_jt.py"
CHAIN_CONSUMERS = ("ranker_basket_ds2.py", "ds2_mf_basket_pack.py")

#: Known divergent sites outside the chain. Listed so a reader cannot mistake
#: their absence from the chain tests for their absence from the repository.
OUT_OF_CHAIN = ("ranker_basket_ab_ds2.py", "footprint_ab_probe.py")


def source(name: str) -> str:
    return (REPO / "src" / name).read_text(encoding="utf-8")


#: Suffix tokens that pipeline_common's run-directory namers emit. A literal is a
#: contract violation only when the contract would own that name; sibling output
#: directories such as ``dataset2-ranker`` or ``dataset2-crf`` are not run
#: directories and are legitimately spelled out.
CONTRACT_SUFFIX = re.compile(
    r"^dataset[12]"
    r"(-bpr|-innov|-novirt|-negpop|-holdout|-s\d+|-d\d+|-t[\d.]+|-tmax[\d.e+]+)*$")


def owned_by_the_contract(literal: str) -> bool:
    """True when pipeline_common's run-dir namers can produce this directory name."""
    return bool(CONTRACT_SUFFIX.match(literal))


def literal_path_joins(name: str) -> list[tuple[int, str]]:
    """Every ``<expr> / "<literal>"`` whose literal names a CONTRACT-owned run dir.

    Matches the defect shape exactly: a Div whose right operand is a plain string
    constant that the run-dir namers could have produced. Concatenated forms such
    as ``tdir / ("dataset2-bpr" + suffix)`` are a different expression and are not
    matched -- they are covered by
    :meth:`ProductionChainContractTest.test_bpr_literals_in_the_mf_pack_agree_with_the_bpr_contract`.
    """
    found = []
    for node in ast.walk(ast.parse(source(name))):
        if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                and isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, str)
                and owned_by_the_contract(node.right.value)):
            found.append((node.lineno, node.right.value))
    return found


class DetectorTest(unittest.TestCase):
    """The detector must actually detect. Without this the suite could pass by
    matching nothing at all."""

    def test_detector_classifies_contract_ownership_correctly(self):
        # The exact literal that broke stage 1, plus other names the contract owns.
        for owned in ("dataset2", "dataset1", "dataset2-novirt", "dataset2-bpr",
                      "dataset2-bpr-s31337", "dataset2-novirt-tmax1.26196e+09"):
            with self.subTest(owned=owned):
                self.assertTrue(owned_by_the_contract(owned))
        # Sibling OUTPUT directories are not run directories; flagging them would
        # make the chain test fail for a legitimate literal.
        for free in ("dataset2-ranker", "dataset2-crf", "dataset1-ensemble",
                     "dataset2-submissions"):
            with self.subTest(free=free):
                self.assertFalse(owned_by_the_contract(free))

    def test_detector_still_sees_the_known_out_of_chain_sites(self):
        # If these ever come back clean the detector has silently stopped working,
        # or the sites were fixed -- either way this test must be revisited.
        for name in OUT_OF_CHAIN:
            with self.subTest(module=name):
                self.assertTrue(literal_path_joins(name),
                                f"{name} no longer contains a literal run-dir join; "
                                "update OUT_OF_CHAIN and the module docstring")


class ProductionChainContractTest(unittest.TestCase):
    def test_no_chain_consumer_spells_out_a_run_directory(self):
        for name in CHAIN_CONSUMERS:
            with self.subTest(module=name):
                self.assertEqual(
                    literal_path_joins(name), [],
                    f"{name} rebuilds a run directory from a literal instead of "
                    "calling pipeline_common.line_run_dir/bpr_run_dir")

    def test_mf_pack_calls_the_contract_for_the_line_directory(self):
        text = source("ds2_mf_basket_pack.py")
        self.assertIn('tl.line_run_dir("dataset2")', text)
        self.assertNotIn('tdir / "dataset2"', text)

    def test_producer_and_consumers_resolve_the_same_directory(self):
        # The trainer derives its output directory from the same call, so the
        # contract is what makes producer and consumer agree.
        self.assertIn("pc.line_run_dir(DATASET", source(PRODUCER))
        self.assertIn('tl.line_run_dir("dataset2")', source("ds2_mf_basket_pack.py"))
        self.assertIn('tl.line_run_dir("dataset2")', source("ranker_basket_ds2.py"))

    def test_the_contract_appends_the_novirt_suffix_by_default(self):
        # The property that made the stale literal wrong. Asserted directly so a
        # future change to the suffix rule cannot silently re-break the chain.
        serve = pc.line_run_dir(DATASET)
        self.assertEqual(serve.name, "dataset2-novirt")
        self.assertNotEqual(serve.name, "dataset2")

    def test_serve_and_time_cut_directories_stay_distinct(self):
        # build_features consumes the SERVE embedding (full train). The train-side
        # variant carries a -tmax suffix and must never collide with it.
        serve = pc.line_run_dir(DATASET)
        cut = pc.line_run_dir(DATASET, time_max=1261958400.0)
        self.assertNotEqual(serve, cut)
        self.assertTrue(cut.name.startswith("dataset2-novirt-tmax"))

    def test_bpr_literals_in_the_mf_pack_agree_with_the_bpr_contract(self):
        # These literals are concatenated, so the detector above does not flag
        # them. They are currently harmless and this test is what proves it: if
        # bpr_run_dir's naming ever changes, the MF pack would silently load a
        # different embedding set, and this fails first.
        for seed in (42, 123, 777, 2024, 31337):
            with self.subTest(seed=seed):
                literal = "dataset2-bpr" + (f"-s{seed}" if seed != 42 else "")
                self.assertEqual(pc.bpr_run_dir(DATASET, seed=seed).name, literal)


class OutOfChainFindingTest(unittest.TestCase):
    def test_out_of_chain_sites_are_a_known_finding(self):
        # Documents, without asserting a defect is desirable, that these two
        # modules are outside the strategy chain and were deliberately left
        # unmodified by the path-contract correction.
        import json
        cfg = json.loads((REPO / "configs" / "production.json").read_text(encoding="utf-8"))
        chain = {Path(s["implementation"]).name for s in cfg["dataset2"]["strategy_chain"]}
        for name in OUT_OF_CHAIN:
            with self.subTest(module=name):
                self.assertNotIn(name, chain,
                                 f"{name} is now in the production chain and must "
                                 "be brought onto the path contract")


if __name__ == "__main__":
    unittest.main()
