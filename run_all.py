# -*- coding: utf-8 -*-
"""One-command reproduction: both datasets, in the required order.

    python run_all.py --data-root /path/to/data --output-root /path/to/outputs

This is an ORCHESTRATION WRAPPER and nothing else. It runs

    python main.py --dataset dataset1
    python main.py --dataset dataset2

as child processes, in that order, and stops at the first failure. It contains
no model, feature, ranking, normalisation, CRF or decoding logic, imports no
training module, and reads no configuration of its own -- every scientific
decision stays where it already lives, in ``main.py`` and the canonical stage
graph. Standard library only.

**The order is required, not a convenience.** ``crf_promote`` writes a
two-member container and therefore needs Dataset-1 member bytes as passthrough
input, so Dataset 2 cannot complete before Dataset 1 has. If Dataset 1 fails,
Dataset 2 is not started.

Behaviour inherited unchanged from ``main.py``: validated resume is the default
(a stage is reused only when its completion record proves it); ``--fresh``
refuses all reuse; the Jittor runtime settings ``conv_opt=1`` and ``use_mkl=0``
are applied and checked by the driver, not here. Nothing is deleted, repaired or
adopted automatically.

    python run_all.py --plan        # print both stage graphs, execute nothing
    python run_all.py --fresh       # refuse to reuse any completed stage
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

#: The packaged ``code/`` directory, resolved from this file rather than from
#: the caller's working directory, so the runner works from anywhere.
CODE_ROOT = Path(__file__).resolve().parent
ENTRYPOINT = CODE_ROOT / "main.py"
DATASETS = ("dataset1", "dataset2")


def default_data_root() -> Path:
    return CODE_ROOT / "data"


def default_output_root() -> Path:
    return CODE_ROOT / "outputs"


def raw_files(data_root: Path, data_pack: str, dataset: str) -> list[Path]:
    base = data_root / data_pack / dataset
    return [base / "train.csv", base / "test.csv"]


def build_command(dataset: str, args: argparse.Namespace) -> list[str]:
    """The exact child command. Kept pure so tests can assert it directly."""
    command = [sys.executable, str(ENTRYPOINT), "--dataset", dataset]
    if args.plan:
        command += ["--stage", "plan"]
    command += ["--data-root", str(args.data_root), "--output-root", str(args.output_root)]
    if args.data_pack != "data_A":
        command += ["--data-pack", args.data_pack]
    if args.log_dir is not None:
        command += ["--log-dir", str(args.log_dir)]
    if args.fresh:
        command += ["--fresh"]
    if args.quiet:
        command += ["--quiet"]
    return command


def preflight(args: argparse.Namespace) -> list[str]:
    """Everything that must hold before the first child starts."""
    problems: list[str] = []
    if not ENTRYPOINT.is_file():
        problems.append(f"the packaged entry point is missing: {ENTRYPOINT}")
    if args.data_root.resolve() == args.output_root.resolve():
        problems.append(
            f"--data-root and --output-root are the same directory "
            f"({args.data_root}); run artifacts would be written into the raw data"
        )
    if not args.plan:
        for dataset in DATASETS:
            for path in raw_files(args.data_root, args.data_pack, dataset):
                if not path.is_file():
                    problems.append(f"missing official raw data: {path}")
    return problems


def run(command: list[str]) -> int:
    """Run a child, streaming its output. Returns its exit code unchanged."""
    print(f"\n$ {' '.join(command)}", flush=True)
    try:
        completed = subprocess.run(command, cwd=str(CODE_ROOT))
    except KeyboardInterrupt:
        print(
            "\nINTERRUPTED: the child process was signalled; stopping.", file=sys.stderr, flush=True
        )
        return 130
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Runs dataset1 then dataset2. Dataset 2 is not started if dataset 1 fails.",
    )
    ap.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="root holding the official data packs (default: <code>/data)",
    )
    ap.add_argument(
        "--data-pack", default="data_A", help="data pack under --data-root (default: data_A)"
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help="root for every run artifact (default: <code>/outputs)",
    )
    ap.add_argument(
        "--log-dir", type=Path, default=None, help="stage logs (default: <output-root>/_logs)"
    )
    ap.add_argument("--fresh", action="store_true", help="refuse to reuse any completed stage")
    ap.add_argument(
        "--plan", action="store_true", help="print both stage graphs and execute nothing"
    )
    ap.add_argument(
        "--quiet", action="store_true", help="do not echo stage output (logs are still written)"
    )
    args = ap.parse_args(argv)

    print("=" * 72)
    print("Jittor link prediction -- full reproduction, both datasets")
    print("=" * 72)
    print(f"code root   : {CODE_ROOT}")
    print(f"data root   : {args.data_root}")
    print(f"output root : {args.output_root}")
    print(f"log dir     : {args.log_dir or (args.output_root / '_logs')}")
    print(
        f"mode        : {'PLAN ONLY (no computation)' if args.plan else ('FRESH (no reuse)' if args.fresh else 'validated resume')}"
    )
    print("order       : dataset1 -> dataset2 (required; dataset2 consumes the dataset1 member)")

    problems = preflight(args)
    if problems:
        print("\nFAILED PREFLIGHT:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print(
            "\nThe competition data is not redistributable and is not bundled; "
            "point --data-root at it.",
            file=sys.stderr,
        )
        return 2

    for index, dataset in enumerate(DATASETS, start=1):
        print(f"\n----- [{index}/{len(DATASETS)}] START {dataset} -----", flush=True)
        code = run(build_command(dataset, args))
        if code != 0:
            print(f"\n----- FAILED {dataset} (exit {code}) -----", file=sys.stderr)
            if dataset == DATASETS[0]:
                print("dataset2 was NOT started: it consumes the dataset1 member.", file=sys.stderr)
            print(
                "No completion record is written for a failed stage; inspect the "
                "stage log named above and re-run.",
                file=sys.stderr,
            )
            return code
        print(f"----- OK {dataset} -----", flush=True)

    print("\n" + "=" * 72)
    if args.plan:
        print("PLAN COMPLETE -- nothing was executed")
    else:
        print("REPRODUCTION COMPLETE")
        print("final members:")
        for dataset in DATASETS:
            print(f"  {args.output_root / 'members' / (dataset + '.csv')}")
        print(f"stage logs   : {args.log_dir or (args.output_root / '_logs')}")
        print(
            "each artifact carries a <artifact>.done.json completion record "
            "recording exactly what produced it"
        )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nINTERRUPTED", file=sys.stderr)
        raise SystemExit(130)
