# -*- coding: utf-8 -*-
"""Rebuild the accepted dataset2 submission member by applying the final decoder.

Runs the frozen ``xte_cross_time_exclusivity_decode`` stage over the base
dataset2 score matrix produced by the ranker + basket + row-order CRF chain, and
writes the submission member. With ``--verify`` it asserts the base hash, the
full physical census, the action count and the output hash against
``configs/production.json``, so a checkout with the local artifacts present can
prove it reproduces the accepted member byte for byte.

The base matrix is read from a submission archive by default because that is the
hash-pinned byte form. Rebuilding the base from raw competition data is a
separate, longer operation described in ``docs/CURRENT_PRODUCTION.md``.

Usage
-----
    python src/build_ds2_member.py --verify
    python src/build_ds2_member.py --base outputs/dataset2-ranker/base.csv \
                                   --output outputs/dataset2-ranker/dataset2_rebuilt.csv
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from strategies.ds2 import cross_time_exclusivity as xte  # noqa: E402
from strategies.shared.frozen_ops import (  # noqa: E402
    sha256_file,
    validate_submission_matrix,
)

REPO = Path(__file__).resolve().parents[1]
PRODUCTION_CONFIG = REPO / "configs" / "production.json"
DEFAULT_BASE_ZIP = REPO / "outputs" / "submissions" / "round22_main" / "r22_rgr_main_d9.zip"
DEFAULT_BASE_MEMBER = "dataset2.csv"
DEFAULT_OUTPUT = REPO / "outputs" / "dataset2-ranker" / "dataset2_rebuilt.csv"
DEFAULT_TEST = REPO / "data" / "data_A" / "dataset2" / "test.csv"


def load_anchors() -> dict:
    with PRODUCTION_CONFIG.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_base(args: argparse.Namespace) -> tuple[bytes, str, str]:
    """Base member bytes plus its SHA256 and a human-readable source label."""
    if args.base is not None:
        payload = Path(args.base).read_bytes()
        return payload, hashlib.sha256(payload).hexdigest(), str(args.base)
    with zipfile.ZipFile(args.base_zip) as archive:
        payload = archive.read(args.base_member)
    label = f"{args.base_zip}::{args.base_member}"
    return payload, hashlib.sha256(payload).hexdigest(), label


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    source = ap.add_argument_group("base score matrix (chain input)")
    source.add_argument(
        "--base", type=Path, default=None, help="base dataset2 score CSV; overrides --base-zip"
    )
    source.add_argument(
        "--base-zip",
        type=Path,
        default=DEFAULT_BASE_ZIP,
        help="submission archive holding the hash-pinned base member",
    )
    source.add_argument("--base-member", default=DEFAULT_BASE_MEMBER)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--test", type=Path, default=DEFAULT_TEST)
    ap.add_argument(
        "--verify", action="store_true", help="assert every anchor against configs/production.json"
    )
    ap.add_argument(
        "--audit-only",
        action="store_true",
        help="run the census and action derivation, write no CSV",
    )
    args = ap.parse_args()

    started = time.time()
    anchors = load_anchors() if args.verify else None
    failures: list[str] = []

    def check(name: str, got: object, want: object) -> None:
        if want is None:
            return
        ok = got == want
        if not ok:
            failures.append(name)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}: {got}" + ("" if ok else f" != {want}"))

    payload, base_sha, base_label = read_base(args)
    print(f"base    : {base_label}")
    if anchors:
        check("base sha256", base_sha, anchors["dataset2"]["decoder_input"]["sha256"])
    else:
        print(f"  base sha256 {base_sha}")

    test = pd.read_csv(args.test)
    if anchors:
        check(
            "test.csv sha256",
            sha256_file(args.test),
            anchors["immutable_inputs"]["data/data_A/dataset2/test.csv"],
        )
    scores = pd.read_csv(io.BytesIO(payload), header=None).to_numpy(np.float64)
    print(f"loaded {scores.shape[0]} x {scores.shape[1]} in {time.time() - started:.1f}s")

    treatment, info = xte.apply(scores, test)
    census = info["census"]
    print(
        f"[1] {xte.STRATEGY_ID}: {info['actions']} actions, "
        f"{info['top1_changes']} top-1 changes "
        f"({100 * info['action_coverage']:.2f}% of rows), "
        f"{info['cells_changed']} cells"
    )

    # Structural validation: the census must agree with the test frame it was
    # derived from. True for any correct decode of any input, so it blocks on
    # every run and is not gated by --verify.
    structural = xte.structural_mismatches(census, test)
    if structural:
        print("  [FAIL] census contradicts the physical test file:")
        print(json.dumps(structural, indent=4))
        failures.append("structural census")
    else:
        print(
            f"  [OK  ] all {len(xte.STRUCTURAL_ANCHOR_KEYS)} structural census "
            f"entries agree with {args.test.name}"
        )

    # Frozen-result census: the remaining entries are functions of the served top-1
    # and therefore of the particular model that produced the base. A retrained
    # model legitimately moves them, so they identify the frozen deployment
    # rather than a correctness property. Enforced under --verify, which is the
    # explicit frozen-reference verification mode; recorded otherwise.
    frozen_census = xte.census_mismatches(census, keys=xte.SCORE_DEPENDENT_ANCHOR_KEYS)
    n_score_dep = len(xte.SCORE_DEPENDENT_ANCHOR_KEYS)
    if args.verify:
        if frozen_census:
            print("  [FAIL] score-dependent census differs from the accepted deployment:")
            print(json.dumps(frozen_census, indent=4))
            failures.append("census")
        else:
            print(f"  [OK  ] all {n_score_dep} score-dependent census anchors match")
    elif frozen_census:
        print(
            f"  [NOTE] {len(frozen_census)} of {n_score_dep} score-dependent census "
            f"entries differ from the accepted deployment."
        )
        print(
            "         Expected when the base matrix comes from a retrained model; "
            "run with --verify to"
        )
        print("         assert the frozen values instead. Observed differences:")
        print(json.dumps(frozen_census, indent=4))
    else:
        print(
            f"  [OK  ] all {n_score_dep} score-dependent census anchors match "
            f"the accepted deployment"
        )

    if anchors:
        want = anchors["dataset2"]["final_decoder"]["effect_on_member"]
        check("action rows", info["actions"], want["rows_changed"])
        check("cells changed", info["cells_changed"], want["cells_changed"])
        check("top-1 changes", info["top1_changes"], want["top1_changes"])
        check(
            "pairs collapsed to a tie",
            info["pairs_collapsed_to_tie"],
            want["pairs_collapsed_to_tie"],
        )

    if failures:
        print(f"\nelapsed {time.time() - started:.1f}s")
        print("VERIFY FAILED: " + ", ".join(failures))
        print("no CSV was written")
        return 1

    if args.audit_only:
        print(json.dumps(census, indent=2))
        print("\nAUDIT-ONLY: no CSV written")
        return 0

    # Mandatory final gate on the float path, before any byte is serialised. It
    # reports and aborts; it does not clamp or otherwise repair the values.
    try:
        validate_submission_matrix(
            treatment, tuple(anchors["dataset2"]["shape"]) if anchors else None
        )
    except ValueError as problem:
        print(f"  [FAIL] {problem}")
        print(f"\nelapsed {time.time() - started:.1f}s")
        print("OUTPUT GATE REJECTED the final matrix; no CSV was written")
        return 1
    print("  [OK  ] output gate: shape, finite values and [0, 1] range")

    # Byte-preserving serialisation: swap the two score TOKENS in the base
    # member's own bytes. Writing `treatment` through a float formatter would
    # re-serialise all 15,342,000 values and cannot be trusted to reproduce the
    # accepted bytes.
    rebuilt = xte.swap_score_tokens(
        payload, info["action_rows"], info["c1_col"], info["c2_col"], census["columns"]
    )
    if len(rebuilt) != len(payload):
        print(f"FAIL: rebuilt member is {len(rebuilt)} bytes, base is {len(payload)}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(rebuilt)
    output_sha = hashlib.sha256(rebuilt).hexdigest()
    print(f"output  : {args.output} ({len(rebuilt)} bytes)")

    # The token swap must agree with the float path on every ranking decision.
    reread = pd.read_csv(args.output, header=None).to_numpy(np.float64)
    if not np.array_equal(
        np.argsort(-reread, axis=1, kind="stable"), np.argsort(-treatment, axis=1, kind="stable")
    ):
        print("FAIL: the token swap and the float path disagree on the candidate ranking")
        return 1
    print("  [OK  ] token swap agrees with the float path on every row's ranking")

    if anchors:
        check("dataset2 member sha256", output_sha, anchors["dataset2"]["member_sha256"])
        check("dataset2 member bytes", len(rebuilt), anchors["dataset2"]["member_bytes"])
    else:
        print(f"  output sha256 {output_sha}")

    print(f"\nelapsed {time.time() - started:.1f}s")
    if failures:
        print("VERIFY FAILED: " + ", ".join(failures))
        return 1
    if anchors:
        print("VERIFY PASSED: the tracked decoder reproduces the accepted dataset2 member exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
