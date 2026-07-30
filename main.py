# -*- coding: utf-8 -*-
"""Unified entry point for the Jittor link-prediction submission.

One command surface for both scenarios. Dataset selection is configuration
driven: the same code path serves ``dataset1`` (non-bipartite) and ``dataset2``
(bipartite), and the per-dataset differences are read from
``configs/production.json`` rather than branched on in code.

    python main.py --dataset dataset1 --stage postprocess --verify
    python main.py --dataset dataset2 --stage postprocess --verify
    python main.py --dataset dataset2 --stage train --dry-run
    python main.py --stage package --verify
    python main.py --stage describe

Stages
------
``describe``      print the resolved pipeline for a dataset: every stage, its
                  implementation, its inputs and how it is verified. Runs
                  anywhere, needs no data.
``preprocess``    validate the raw competition data and report graph type.
``train``         train the embedding models for the dataset.
``rank``          build the base score matrix from the trained embeddings.
``postprocess``   apply the frozen postprocessor chain and write the submission
                  member. This is the stage that is proven byte-exact.
``package``       assemble and verify the submission archive.

Every stage prints the exact command it runs. Where a stage is not runnable from
a clean checkout, it says so and names the blocker instead of failing obscurely;
see ``docs/CURRENT_PRODUCTION.md`` and
``docs/maintenance/repository-reorganisation-ambiguities.md``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

PRODUCTION_CONFIG = REPO / "configs" / "production.json"
DATASETS = ("dataset1", "dataset2")


def load_production() -> dict:
    with PRODUCTION_CONFIG.open(encoding="utf-8") as handle:
        return json.load(handle)


def data_dir(dataset: str) -> Path:
    return REPO / "data" / "data_A" / dataset


def run(command: list[str], dry_run: bool) -> int:
    """Echo a command and run it with the current interpreter."""
    printable = " ".join(command)
    print(f"\n$ {printable}")
    if dry_run:
        print("  (dry run: not executed)")
        return 0
    return subprocess.call([sys.executable, *command[1:]] if command[0] == "python"
                           else command, cwd=REPO)


# --------------------------------------------------------------------------
# describe
# --------------------------------------------------------------------------

def stage_describe(config: dict, datasets: tuple[str, ...]) -> int:
    print(f"accepted total score : {config['accepted_total_score']}")
    print(f"score model          : {config['score_model'].split(';')[0]}")
    print(f"accepted archive     : {config['accepted_archive']['path']}")
    print(f"                       sha256 {config['accepted_archive']['sha256']}")
    for dataset in datasets:
        block = config[dataset]
        print(f"\n=== {dataset} -- component {block['component_score']} "
              f"({block['shape'][0]} rows x {block['shape'][1]} candidates, "
              f"{block['line_ending']}, {block['float_format']}) ===")
        for stage in block["strategy_chain"]:
            upstream = stage.get("upstream")
            print(f"  [{stage['order']}] {stage['strategy_id']}")
            print(f"      implementation : {stage['implementation']}")
            if upstream:
                print(f"      upstream       : {', '.join(upstream)}")
            if "actions" in stage:
                print(f"      actions        : {stage['actions']} "
                      f"({100 * stage.get('action_coverage', 0):.2f}% of rows)")
            if "online_delta" in stage:
                print(f"      online delta   : +{stage['online_delta']}")
        decoder = block.get("final_decoder")
        if decoder:
            print(f"  [final] {decoder['strategy_id']}")
            print(f"      implementation : {decoder.get('implementation', 'NOT TRACKED')}")
            print(f"      status         : {decoder['implementation_status']}, "
                  f"{decoder['operational_lifecycle']}, "
                  f"historically {decoder['historical_lifecycle']}")
            print(f"      online gain    : +{decoder['online_observed_gain']} "
                  f"(gate {decoder['original_locked_gate']}, "
                  f"shortfall {decoder['shortfall_from_original_gate']})")
        print(f"  member sha256    : {block['member_sha256']}")
        print(f"  build            : {block['build_command']}")
    return 0


# --------------------------------------------------------------------------
# preprocess
# --------------------------------------------------------------------------

def stage_preprocess(config: dict, datasets: tuple[str, ...], dry_run: bool) -> int:
    import numpy as np
    import pandas as pd

    failures = 0
    for dataset in datasets:
        directory = data_dir(dataset)
        train_path, test_path = directory / "train.csv", directory / "test.csv"
        print(f"\n=== {dataset} ===")
        if not (train_path.exists() and test_path.exists()):
            print(f"  BLOCKED: raw competition data not present under {directory}")
            print("  the competition data is not redistributable and is not tracked")
            failures += 1
            continue
        if dry_run:
            print("  (dry run: not executed)")
            continue
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        src = train["src"].to_numpy(np.int64)
        dst = train["dst"].to_numpy(np.int64)
        overlap = np.intersect1d(np.unique(src), np.unique(dst))
        # Graph type is measured, never assumed: an empty source/destination
        # identity overlap is the bipartite signature. Both scenarios must work.
        bipartite = overlap.size == 0
        print(f"  interactions      : {len(train)}")
        print(f"  distinct sources  : {np.unique(src).size}")
        print(f"  distinct dests    : {np.unique(dst).size}")
        print(f"  role overlap      : {overlap.size}")
        print(f"  self loops        : {int((src == dst).sum())}")
        print(f"  graph type        : {'BIPARTITE' if bipartite else 'NON_BIPARTITE'}")
        print(f"  test queries      : {len(test)}")
        candidates = [c for c in test.columns if c.startswith("c") and c[1:].isdigit()]
        print(f"  candidates/query  : {len(candidates)}")
        print(f"  timestamp range   : {train['time'].min()} .. {train['time'].max()}")
        if "time" in test.columns:
            # The causal boundary: no feature may read an interaction later than
            # the query it scores.
            print(f"  query time range  : {test['time'].min()} .. {test['time'].max()}")
    return 1 if failures else 0


# --------------------------------------------------------------------------
# train / rank
# --------------------------------------------------------------------------

TRAIN_NOTE = """\
  Embedding training is the long stage. The accepted A-board artifacts were
  produced by the PyTorch trainers named in configs/production.json; the Jittor
  ports (src/train_line_jt.py, src/train_bpr_jt.py) implement the same models,
  objectives and knobs and are the framework-compliant path. Seeded numpy owns
  sampling in both, so the two agree statistically, not byte for byte.
  See docs/competition/ab_algorithm_consistency_contract.md."""


def stage_train(config: dict, datasets: tuple[str, ...], dry_run: bool,
                framework: str) -> int:
    print(TRAIN_NOTE)
    line = "src/train_line_jt.py" if framework == "jittor" else "src/train_line.py"
    bpr = "src/train_bpr_jt.py" if framework == "jittor" else "src/train_bpr.py"
    status = 0
    for dataset in datasets:
        print(f"\n=== {dataset} embeddings ({framework}) ===")
        for script in (line, bpr):
            print(f"  DATASET={dataset} python {script}")
        if not dry_run:
            print("  NOT EXECUTED by main.py: each trainer is a multi-hour GPU job "
                  "configured through environment variables (SEED, EMB_DIM, EPOCHS, ...).")
            print("  Run the printed commands directly so the run directory and seed "
                  "are explicit; see docs/architecture/pipeline-overview.md.")
            status = max(status, 0)
    return status


def stage_rank(config: dict, datasets: tuple[str, ...], dry_run: bool) -> int:
    for dataset in datasets:
        block = config[dataset]
        base = block["strategy_chain"][0]
        print(f"\n=== {dataset} base score matrix ===")
        print(f"  strategy       : {base['strategy_id']}")
        print(f"  implementation : {base['implementation']}")
        print(f"  command        : DATASET={dataset} python {base['implementation']}")
        if dataset == "dataset1":
            target = block["chain_input"]
            print(f"  output         : {target['path']}")
            print(f"                   sha256 {target['sha256']}")
        else:
            target = block["decoder_input"]
            print(f"  output         : the base matrix hash-pinned at "
                  f"sha256 {target['sha256']}")
            print("  BLOCKED for byte-exact rebuild: ambiguity A1 -- the dataset2 base "
                  "needs cached train features and a multi-hour LightGBM job, and has "
                  "not been re-executed since 2026-07-28.")
        if not dry_run:
            print("  NOT EXECUTED by main.py: run the printed command directly.")
    return 0


# --------------------------------------------------------------------------
# postprocess -- the stage that is proven byte-exact
# --------------------------------------------------------------------------

def stage_postprocess(datasets: tuple[str, ...], dry_run: bool, verify: bool) -> int:
    status = 0
    builders = {"dataset1": "src/build_ds1_member.py", "dataset2": "src/build_ds2_member.py"}
    for dataset in datasets:
        command = ["python", builders[dataset]] + (["--verify"] if verify else [])
        code = run(command, dry_run)
        if code != 0:
            print(f"  {dataset}: postprocess stage FAILED with exit code {code}")
            status = code
    return status


# --------------------------------------------------------------------------
# package
# --------------------------------------------------------------------------

def stage_package(config: dict, dry_run: bool, verify: bool) -> int:
    archive = config["accepted_archive"]["path"]
    if not verify:
        print("packaging a NEW archive is a submission action and is deliberately not "
              "wired into main.py; see docs/SUBMISSION_PROTOCOL.md")
        return 0
    return run(["python", "tools/submission/package_component.py", "verify",
                "--zip", archive], dry_run)


# --------------------------------------------------------------------------

STAGES = ("describe", "preprocess", "train", "rank", "postprocess", "package")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages: " + ", ".join(STAGES))
    ap.add_argument("--dataset", choices=(*DATASETS, "all"), default="all",
                    help="scenario to operate on (default: all)")
    ap.add_argument("--stage", choices=STAGES, default="describe")
    ap.add_argument("--framework", choices=("jittor", "torch"), default="jittor",
                    help="embedding trainer implementation for the train stage")
    ap.add_argument("--verify", action="store_true",
                    help="assert artifact hashes against configs/production.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command without executing it")
    args = ap.parse_args()

    config = load_production()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    print(f"repository : {REPO.name}")
    print(f"stage      : {args.stage}")
    print(f"datasets   : {', '.join(datasets)}")

    if args.stage == "describe":
        return stage_describe(config, datasets)
    if args.stage == "preprocess":
        return stage_preprocess(config, datasets, args.dry_run)
    if args.stage == "train":
        return stage_train(config, datasets, args.dry_run, args.framework)
    if args.stage == "rank":
        return stage_rank(config, datasets, args.dry_run)
    if args.stage == "postprocess":
        return stage_postprocess(datasets, args.dry_run, args.verify)
    if args.stage == "package":
        return stage_package(config, args.dry_run, args.verify)
    raise AssertionError(f"unhandled stage {args.stage!r}")


if __name__ == "__main__":
    raise SystemExit(main())
