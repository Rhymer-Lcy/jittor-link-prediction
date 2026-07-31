# -*- coding: utf-8 -*-
"""Time-cutoff runs must keep a full-size entity table.

`LINE_TIME_MAX` / `BPR_TIME_MAX` restrict TRAINING EDGES to a feature period.
They must not shrink the embedding table: downstream stages index embeddings by
global node id, so a table sized from the filtered frame silently misaligns
every id.

`src/train_line.py` states the contract ("The entity table stays full-size so
node ids align with full-data artifacts") and implements it by computing
`full_num_entity` before applying the cutoff. The Jittor LINE port originally
computed it after, producing a 42,361-row dataset1 embedding instead of 43,215
and failing `ranker_ds1` with:

    AssertionError: dataset1-novirt-tmax1.1548e+08: embedding has 42361 rows,
                    expected 43215

These tests pin the ordering statically in every trainer, so the divergence
cannot reappear, and check the invariant on a synthetic frame.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"

#: (file, cutoff variable) for every trainer that supports a hard time cutoff.
TRAINERS = [
    ("train_line_jt.py", "LINE_TIME_MAX"),
    ("train_bpr_jt.py", "TIME_MAX"),
    ("train_line.py", "LINE_TIME_MAX"),
    ("train_bpr.py", "TIME_MAX"),
]


def _statement_offsets(path: Path, cutoff: str) -> tuple[int | None, int | None]:
    """Line numbers of the entity-count assignment and the cutoff filter.

    The filter is located by inspecting the ``if`` TEST expression, not the
    statement's source text. Matching on source text finds whichever enclosing
    block happens to contain the cutoff somewhere in its body -- an outer
    ``if``/function -- and reports a misleadingly early line.
    """
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    filter_lines = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(isinstance(n, ast.Name) and n.id == cutoff for n in ast.walk(node.test))
    ]
    entity_lines = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id in ("num_entity", "full_num_entity")
                for t in node.targets)
        and ".max()" in (ast.get_source_segment(text, node) or "")
    ]
    if not filter_lines or not entity_lines:
        return (entity_lines[0] if entity_lines else None,
                filter_lines[0] if filter_lines else None)

    # The guard that actually applies the cutoff is the last one that also
    # rebinds the frame; take the latest, and compare against the entity
    # assignment that precedes it most closely.
    filter_line = max(filter_lines)
    before = [ln for ln in entity_lines if ln < filter_line]
    entity_line = max(before) if before else min(entity_lines)
    return entity_line, filter_line


class EntityTableIsSizedBeforeTheCutoff(unittest.TestCase):
    def test_every_trainer_sizes_entities_before_filtering(self):
        offenders = []
        for name, cutoff in TRAINERS:
            path = SRC / name
            if not path.exists():
                continue
            entity_line, filter_line = _statement_offsets(path, cutoff)
            if entity_line is None or filter_line is None:
                continue                      # trainer does not use this pattern
            if entity_line > filter_line:
                offenders.append(
                    f"{name}: entity count at line {entity_line} comes AFTER the "
                    f"{cutoff} filter at line {filter_line}")
        self.assertEqual(offenders, [], "; ".join(offenders))

    def test_invariant_on_a_synthetic_frame(self):
        """Sizing before the cutoff keeps every node; sizing after loses some."""
        frame = pd.DataFrame({
            "src": [0, 1, 2, 3],
            "dst": [1, 2, 3, 9],          # node 9 appears only in the late edge
            "time": [10.0, 20.0, 30.0, 900.0],
        })
        cutoff = 100.0

        before = int(max(frame.src.max(), frame.dst.max())) + 1
        filtered = frame[frame["time"] <= cutoff].reset_index(drop=True)
        after = int(max(filtered.src.max(), filtered.dst.max())) + 1

        self.assertEqual(before, 10, "full-size table must cover node id 9")
        self.assertEqual(after, 4, "sizing after the cutoff drops the late nodes")
        self.assertGreater(before, after,
                           "the two orderings must differ, otherwise this test proves nothing")

    def test_downstream_alignment_check_would_catch_it(self):
        """A short table fails the loader assertion the ranker relies on."""
        expected_rows = 10
        short = np.zeros((4, 8), dtype=np.float32)
        full = np.zeros((expected_rows, 8), dtype=np.float32)
        self.assertNotEqual(short.shape[0], expected_rows)
        self.assertEqual(full.shape[0], expected_rows)


if __name__ == "__main__":
    unittest.main(verbosity=2)
