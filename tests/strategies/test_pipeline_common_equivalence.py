# -*- coding: utf-8 -*-
"""`pipeline_common` must stay byte-for-byte equivalent to its historical source.

`src/pipeline_common.py` was extracted verbatim from `src/train_line.py` so that
the canonical ranking stages could stop importing a PyTorch-coupled module. The
historical trainer keeps its own copies because it mutates several of these
symbols as module globals, so the two definitions could drift apart.

These tests compare the two implementations at the source level, which needs no
torch and therefore runs everywhere, plus a numerical check of the one function
that was deliberately re-implemented (`build_sim_cache`, ported from torch to
NumPy).
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

#: Extracted verbatim; their source must remain identical in both modules.
VERBATIM = [
    "rownorm", "build_history_index", "get_hist_before_time", "count_in_history",
    "build_cooc", "cooc_scores", "cache_dict_to_matrix", "batch_sim_score",
    "split_train_val_by_tail",
]

#: Deliberately re-implemented: torch cosine top-k -> NumPy. Compared numerically.
REIMPLEMENTED = ["build_sim_cache"]

#: Constants that must agree exactly.
CONSTANTS = ["TOP_N_FIRST", "TOP_N_SECOND", "DECAY_W1", "DECAY_W2",
             "RPOP_TIME_QUANTILE", "VAL_PER_SRC_TAIL"]


def function_sources(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    return {n.name: ast.get_source_segment(text, n)
            for n in tree.body if isinstance(n, ast.FunctionDef)}


def constant_values(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = ast.unparse(node.value)
    return out


class PipelineCommonMatchesHistoricalSource(unittest.TestCase):
    def setUp(self):
        self.historical = SRC / "train_line.py"
        self.neutral = SRC / "pipeline_common.py"
        if not self.historical.exists():
            self.skipTest("historical train_line.py not present")

    def test_verbatim_functions_are_identical(self):
        a = function_sources(self.historical)
        b = function_sources(self.neutral)
        drifted = []
        for name in VERBATIM:
            if name not in a or name not in b:
                drifted.append(f"{name}: missing")
                continue
            if a[name] != b[name]:
                drifted.append(name)
        self.assertEqual(drifted, [], f"pipeline_common drifted from train_line: {drifted}")

    def test_constants_agree(self):
        a = constant_values(self.historical)
        b = constant_values(self.neutral)
        for name in CONSTANTS:
            if name in a and name in b:
                self.assertEqual(a[name], b[name], f"constant {name} differs")

    def test_reimplemented_function_is_documented_as_such(self):
        text = (self.neutral).read_text(encoding="utf-8")
        for name in REIMPLEMENTED:
            self.assertIn(name, text)
        self.assertIn("NumPy implementation", text,
                      "the re-implemented build_sim_cache must say so in the source")

    def test_neutral_module_imports_no_framework(self):
        tree = ast.parse(self.neutral.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        for framework in ("torch", "jittor"):
            self.assertNotIn(framework, imported,
                             f"pipeline_common must not import {framework}")


class SimCacheNumericalEquivalence(unittest.TestCase):
    """The NumPy port must reproduce the torch cosine top-k it replaced."""

    def test_numpy_port_matches_torch_reference(self):
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch not installed; source-level checks still apply")

        import numpy as np

        import pipeline_common as pc

        rng = np.random.default_rng(7)
        emb = rng.standard_normal((300, 32)).astype(np.float32)
        targets = rng.choice(300, 40, replace=False)

        saved = (pc.TOP_N_FIRST, pc.TOP_N_SECOND)
        pc.TOP_N_FIRST, pc.TOP_N_SECOND = 5, 20
        try:
            pc.build_sim_cache(emb, targets)
            np_neigh = pc.sim_neigh_arr.copy()
            np_w = pc.sim_weight_arr.copy()

            et = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32))
            en = et / et.norm(dim=1, keepdim=True).clamp_min(1e-8)
            tid = torch.tensor([int(s) for s in targets], dtype=torch.long)
            k = min(pc.TOP_N_SECOND, en.shape[0])
            sim = en[tid] @ en.T
            sim[torch.arange(len(tid)), tid] = 0.0
            vals, idx = torch.topk(sim, k, dim=1)
            t_neigh = idx.numpy().astype(np.int64)
            t_w = vals.numpy()
            mask = np.full(pc.TOP_N_SECOND, pc.DECAY_W2, dtype=np.float32)
            mask[:pc.TOP_N_FIRST] = pc.DECAY_W1
            t_w = t_w * mask
            t_w[t_w < 1e-8] = 0.0
        finally:
            pc.TOP_N_FIRST, pc.TOP_N_SECOND = saved

        # Neighbour ORDER is what the ranking consumes and must match exactly.
        self.assertTrue((np_neigh == t_neigh).all(), "neighbour ids differ from the torch reference")
        # Weights may differ by a single float32 ULP: numpy and torch use
        # different BLAS reduction orders for the matmul.
        self.assertLess(float(np.abs(np_w - t_w.astype(np.float32)).max()), 1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
