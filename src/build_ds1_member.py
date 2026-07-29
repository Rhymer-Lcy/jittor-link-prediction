# -*- coding: utf-8 -*-
"""Rebuild the accepted dataset1 submission member from the ranker output.

Runs the ordered, frozen dataset1 postprocessor chain
(``source_slate_recurrence`` -> ``test_graph_reciprocity``) over the LambdaRank
ranker's score CSV and writes the submission member. With ``--verify`` it also
checks the SHA256 of every stage against the accepted anchors in
``configs/production.json``, so a clean checkout can prove it reproduces the
accepted artifact byte for byte.

Usage
-----
    python src/build_ds1_member.py --verify
    python src/build_ds1_member.py --control outputs/dataset1-ensemble/result_ranker.csv \
                                   --output outputs/dataset1-ensemble/dataset1_accepted.csv

The chain is order-sensitive; see :mod:`src.strategies.registry`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategies.registry import DS1_POSTPROCESSOR_CHAIN  # noqa: E402
from strategies.shared.frozen_ops import (  # noqa: E402
    read_score_matrix,
    sha256_file,
    write_score_matrix,
)

REPO = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = REPO / "configs" / "production.json"
DEFAULT_CONTROL = REPO / "outputs" / "dataset1-ensemble" / "result_ranker.csv"
DEFAULT_OUTPUT = REPO / "outputs" / "dataset1-ensemble" / "dataset1_rebuilt.csv"
DEFAULT_TEST = REPO / "data" / "data_A" / "dataset1" / "test.csv"
DEFAULT_TRAIN = REPO / "data" / "data_A" / "dataset1" / "train.csv"


def load_anchors() -> dict:
    with PRODUCTION_CONFIG.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--control", type=Path, default=DEFAULT_CONTROL,
                    help="dataset1 ranker score CSV (chain input)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--test", type=Path, default=DEFAULT_TEST)
    ap.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    ap.add_argument("--verify", action="store_true",
                    help="assert every stage hash against configs/production.json")
    ap.add_argument("--stage-dir", type=Path, default=None,
                    help="optional directory to write each intermediate stage CSV")
    args = ap.parse_args()

    started = time.time()
    anchors = load_anchors() if args.verify else None
    failures: list[str] = []

    def check(name: str, got: str, want: str | None) -> None:
        if want is None:
            return
        status = "OK  " if got == want else "FAIL"
        if got != want:
            failures.append(name)
        print(f"  [{status}] {name}: {got}")

    print(f"control : {args.control}")
    control_sha = sha256_file(args.control)
    if anchors:
        check("control sha256", control_sha,
              anchors["dataset1"]["chain_input"]["sha256"])
    else:
        print(f"  control sha256 {control_sha}")

    test = pd.read_csv(args.test)
    train = pd.read_csv(args.train)
    scores = read_score_matrix(args.control)
    print(f"loaded {scores.shape[0]} x {scores.shape[1]} in {time.time() - started:.1f}s")

    stage_hashes = {}
    for stage_index, (sid, module, apply_fn) in enumerate(DS1_POSTPROCESSOR_CHAIN, start=1):
        scores, info = apply_fn(scores, test, train)
        print(f"[{stage_index}] {sid}: {info['actions']} actions, "
              f"{info['top1_changes']} top-1 changes "
              f"({100 * info['action_coverage']:.2f}% of rows)")
        if anchors:
            want = next((s for s in anchors["dataset1"]["strategy_chain"]
                         if s["strategy_id"] == sid), {})
            if "actions" in want and info["actions"] != want["actions"]:
                failures.append(f"{sid} action count")
                print(f"  [FAIL] {sid} actions: {info['actions']} != {want['actions']}")
            else:
                print(f"  [OK  ] {sid} actions match the accepted anchor")
        if args.stage_dir:
            stage_path = Path(args.stage_dir) / f"stage{stage_index}_{sid}.csv"
            stage_hashes[sid] = write_score_matrix(scores, stage_path)
            print(f"  stage CSV -> {stage_path} ({stage_hashes[sid]})")

    output_sha = write_score_matrix(scores, args.output)
    print(f"output  : {args.output}")
    if anchors:
        check("dataset1 member sha256", output_sha, anchors["dataset1"]["member_sha256"])
    else:
        print(f"  output sha256 {output_sha}")

    print(f"\nelapsed {time.time() - started:.1f}s")
    if failures:
        print("VERIFY FAILED: " + ", ".join(failures))
        return 1
    if anchors:
        print("VERIFY PASSED: the chain reproduces the accepted dataset1 member exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
