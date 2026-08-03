# -*- coding: utf-8 -*-
"""LOCAL DIAGNOSTIC ONLY -- side-by-side PyTorch/Jittor embedding comparison.

This is not part of the submission. It exists because the accepted A-board
artifacts were produced by the historical PyTorch trainers and it is sometimes
useful, locally, to train the same model under both frameworks and compare the
resulting embeddings. Nothing here is on the canonical path.

Why it is a separate file
-------------------------
``main.py`` used to carry ``--framework {jittor,torch}``, which selected
``reference/pytorch/train_line.py`` and ``reference/pytorch/train_bpr.py``
instead of the Jittor trainers.
That put a PyTorch execution path inside the organiser-facing interface of a
Jittor-mandated competition -- an ambiguity in the submission, not a feature.
The selection was removed from ``main.py`` rather than deleted from the
repository: the trainers keep their provenance value, and the comparison keeps
working, but it is reachable only from here.

Consequences, all deliberate:

* ``main.py`` has no backend argument and no backend environment variable;
* the canonical import closure contains no torch;
* this file and the two PyTorch trainers are on the official staging exclusion
  list (see ``docs/competition/official_package_inventory.md``);
* PyTorch is not declared in ``environment.yaml`` or ``requirements.txt``, so
  running this tool requires installing it deliberately, with the CPU wheel:
  ``pip install torch --index-url https://download.pytorch.org/whl/cpu``.
  A CUDA-enabled torch ships ``nvidia-*`` packages that shadow the CUDA
  libraries Jittor is built against and break it.

Usage
-----
    python tools/diagnostics/compare_backends.py --dataset dataset1 --model line
    python tools/diagnostics/compare_backends.py --dataset dataset1 --model bpr --dry-run

The two runs write to different directories (``JT_OUT_SUFFIX``), so a diagnostic
comparison can never overwrite a canonical artifact.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: The historical PyTorch trainers. Reachable from here and from nowhere on the
#: canonical path. Kept in the repository for provenance; excluded from the
#: official package.
TORCH_TRAINERS = {
    "line": "reference/pytorch/train_line.py",
    "bpr": "reference/pytorch/train_bpr.py",
}
#: The mandated Jittor trainers -- the canonical implementations.
JITTOR_TRAINERS = {"line": "src/train_line_jt.py", "bpr": "src/train_bpr_jt.py"}

BANNER = """\
================================================================================
DIAGNOSTIC TOOL -- NOT PART OF THE SUBMISSION
The canonical pipeline is Jittor-only and is run with `python main.py --dataset
<dataset>`. This tool trains the same model under both frameworks purely to
compare them, and its artifacts are never canonical inputs.
================================================================================"""


def torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", choices=("dataset1", "dataset2"), required=True)
    ap.add_argument("--model", choices=("line", "bpr"), required=True)
    ap.add_argument(
        "--jittor-suffix",
        default="-jittor-diagnostic",
        help="output-directory suffix for the Jittor arm",
    )
    ap.add_argument(
        "--pytorch-suffix",
        default="-pytorch-diagnostic",
        help="output-directory suffix for the PyTorch arm",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="print both commands and execute neither"
    )
    args = ap.parse_args(argv)

    print(BANNER)
    torch_script = REPO / TORCH_TRAINERS[args.model]
    jittor_script = REPO / JITTOR_TRAINERS[args.model]

    arms = [
        ("jittor", jittor_script, {"JT_OUT_SUFFIX": args.jittor_suffix}),
        ("pytorch", torch_script, {"PYTORCH_OUT_SUFFIX": args.pytorch_suffix}),
    ]
    for name, script, overrides in arms:
        rendered_env = " ".join(f"{key}={value}" for key, value in overrides.items())
        print(
            f"\n[{name}] DATASET={args.dataset} {rendered_env} "
            f"python {script.relative_to(REPO).as_posix()}"
        )

    if args.dry_run:
        print("\n(dry run: neither arm executed)")
        return 0

    if not torch_available():
        print(
            "\nFAIL PyTorch is not installed. It is deliberately absent from the "
            "official environment specification; install the CPU wheel to run this "
            "diagnostic:\n  pip install torch --index-url "
            "https://download.pytorch.org/whl/cpu",
            file=sys.stderr,
        )
        return 2

    status = 0
    for name, script, overrides in arms:
        env = dict(os.environ, DATASET=args.dataset, **overrides)
        print(f"\n===== {name} arm =====")
        code = subprocess.call([sys.executable, str(script)], cwd=str(REPO), env=env)
        if code != 0:
            print(f"{name} arm exited {code}", file=sys.stderr)
            status = code
    return status


if __name__ == "__main__":
    raise SystemExit(main())
