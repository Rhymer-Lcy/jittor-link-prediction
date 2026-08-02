# -*- coding: utf-8 -*-
"""Canonical entry point for the Jittor link-prediction submission.

One command executes the whole production graph for one scenario, from the
official raw competition data to the submission member:

    python main.py --dataset dataset1
    python main.py --dataset dataset2

The official pipeline is Jittor-only. There is no backend argument and no
environment-variable switch: the graph in ``src/canonical_pipeline.py`` names
the Jittor trainers and nothing else can be substituted. No alternative backend
is included or required.

Every stage is gated by a completion record (``<output>.done.json``), not by its
output file existing. A stage is reused only when a record proves it ran to
completion, under this commit, from these inputs, with this configuration, and
produced exactly this artifact. Anything else stops the run with a diagnostic;
see ``docs/architecture/stage-completion-contract.md``.

Stages
------
``run``           execute the canonical graph and write the submission member.
                  The default.
``plan``          print the graph and each stage's current disposition, and
                  execute nothing.
``describe``      print the resolved strategy chain for a dataset from
                  ``configs/production.json``. Runs anywhere, needs no data.
``preprocess``    validate the raw competition data and report graph type.
``postprocess``   re-run only the frozen postprocessor chain over an existing
                  score matrix. This is the stage proven byte-exact.
``package``       verify a submission archive.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

DATASETS = ("dataset1", "dataset2")
STAGES = ("run", "plan", "describe", "preprocess", "postprocess", "package")
DEFAULT_CONFIG = REPO / "configs" / "production.json"


def load_production(path: Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def run(command: list[str]) -> int:
    """Echo a command and run it with the current interpreter."""
    print(f"\n$ {' '.join(command)}")
    return subprocess.call([sys.executable, *command[1:]] if command[0] == "python"
                           else command, cwd=REPO)


# --------------------------------------------------------------------------
# run / plan -- the canonical graph
# --------------------------------------------------------------------------

def contexts(args: argparse.Namespace, datasets: tuple[str, ...]) -> list:
    from canonical_pipeline import RunContext

    return [RunContext(dataset=dataset, data_root=args.data_root,
                       outputs_root=args.output_root, log_dir=args.log_dir,
                       data_pack=args.data_pack, resume=not args.fresh,
                       ds1_member=args.ds1_member, repo=REPO)
            for dataset in datasets]


def stage_run(args: argparse.Namespace, datasets: tuple[str, ...]) -> int:
    from canonical_pipeline import execute

    for ctx in contexts(args, datasets):
        code = execute(ctx, echo=not args.quiet)
        if code != 0:
            return code
    return 0


def stage_plan(args: argparse.Namespace, datasets: tuple[str, ...]) -> int:
    from canonical_pipeline import plan

    for ctx in contexts(args, datasets):
        print(plan(ctx))
        print()
    return 0


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
    print("\nrebuild the whole chain from official raw data with:")
    for dataset in datasets:
        print(f"  python main.py --dataset {dataset}")
    return 0


# --------------------------------------------------------------------------
# preprocess
# --------------------------------------------------------------------------

def stage_preprocess(args: argparse.Namespace, datasets: tuple[str, ...]) -> int:
    import numpy as np
    import pandas as pd

    failures = 0
    for dataset in datasets:
        directory = Path(args.data_root) / args.data_pack / dataset
        train_path, test_path = directory / "train.csv", directory / "test.csv"
        print(f"\n=== {dataset} ===")
        if not (train_path.exists() and test_path.exists()):
            print(f"  BLOCKED: raw competition data not present under {directory}")
            print("  the competition data is not redistributable and is not tracked")
            failures += 1
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
# postprocess -- the stage that is proven byte-exact
# --------------------------------------------------------------------------

def stage_postprocess(datasets: tuple[str, ...], verify: bool) -> int:
    status = 0
    builders = {"dataset1": "src/build_ds1_member.py", "dataset2": "src/build_ds2_member.py"}
    for dataset in datasets:
        command = ["python", builders[dataset]] + (["--verify"] if verify else [])
        code = run(command)
        if code != 0:
            print(f"  {dataset}: postprocess stage FAILED with exit code {code}")
            status = code
    return status


# --------------------------------------------------------------------------
# package
# --------------------------------------------------------------------------

def stage_package(config: dict, verify: bool) -> int:
    archive = config["accepted_archive"]["path"]
    if not verify:
        print("packaging a NEW archive is a submission action and is deliberately not "
              "wired into main.py; see docs/SUBMISSION_PROTOCOL.md")
        return 0
    return run(["python", "tools/submission/package_component.py", "verify",
                "--zip", archive])


# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Stages: " + ", ".join(STAGES)
               + "\nBackend: Jittor, fixed. This entrypoint cannot select another.")
    ap.add_argument("--dataset", choices=(*DATASETS, "all"), default="all",
                    help="scenario to operate on (default: all, dataset1 first)")
    ap.add_argument("--stage", choices=STAGES, default="run",
                    help="what to do (default: run the canonical graph)")
    ap.add_argument("--data-root", type=Path, default=REPO / "data",
                    help="root holding the official data packs (default: <repo>/data)")
    ap.add_argument("--data-pack", default="data_A",
                    help="data pack under --data-root (default: data_A)")
    ap.add_argument("--output-root", type=Path, default=REPO / "outputs",
                    help="root for every run artifact (default: <repo>/outputs)")
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="stage logs (default: <output-root>/_logs)")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                    help="production configuration to describe and verify against")
    ap.add_argument("--fresh", action="store_true",
                    help="refuse to reuse any completed stage; every stage must be "
                         "absent, so an existing run has to be quarantined deliberately")
    ap.add_argument("--ds1-member", type=Path, default=None,
                    help="dataset1 member whose bytes crf_promote carries through; "
                         "must carry a valid completion record")
    ap.add_argument("--verify", action="store_true",
                    help="assert artifact hashes against the production configuration")
    ap.add_argument("--quiet", action="store_true",
                    help="do not echo stage output to the console (logs are still written)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    print(f"repository : {REPO.name}")
    print(f"stage      : {args.stage}")
    print(f"datasets   : {', '.join(datasets)}")
    print("backend    : jittor (fixed)")

    if args.stage == "run":
        return stage_run(args, datasets)
    if args.stage == "plan":
        return stage_plan(args, datasets)
    if args.stage == "describe":
        return stage_describe(load_production(args.config), datasets)
    if args.stage == "preprocess":
        return stage_preprocess(args, datasets)
    if args.stage == "postprocess":
        return stage_postprocess(datasets, args.verify)
    if args.stage == "package":
        return stage_package(load_production(args.config), args.verify)
    raise AssertionError(f"unhandled stage {args.stage!r}")


if __name__ == "__main__":
    raise SystemExit(main())
