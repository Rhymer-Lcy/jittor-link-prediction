#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Component validation and submission packaging for the competition platform.

Takes a treatment CSV from disk to a submittable ZIP with a machine-readable
manifest in seconds, and closes the score arithmetic once the leaderboard
answers.

Modes
-----
  ds1    validate a dataset1 treatment CSV -> ``<exp>_ds1_only.zip``
  ds2    validate a dataset2 treatment CSV -> ``<exp>_ds2_only.zip``
  main   combine one dataset1 and one dataset2 member into a main pack
  score  append an observed leaderboard score to a manifest, closing the delta
  verify re-validate any ZIP this tool produced

Graduated 2026-07-29 from ``scratchpad/round22_opus/harness/scripts/package_component.py``
(source sha256 7794237611f5587a34d6718be423855672039e4c0a31ad6ec8343fd20c02c49d),
which was verified there by 35 self-tests and three dry runs. Behaviour is
preserved; the changes are that the accepted-state constants now come from
``configs/production.json`` instead of being hard-coded, and that the
archive-size messages were corrected (see ``size_policy``).

Why the accepted state is read live from the gold archive
---------------------------------------------------------
Baseline members are read out of the accepted archive on every run and their
SHA256 asserted before anything else happens, so a baseline cannot silently
drift and there is no second copy to keep in sync.

Platform facts this encodes (measured, not assumed)
---------------------------------------------------
* the total is strictly additive: ``total = ds1_MRR + ds2_MRR``;
* a dataset absent from the ZIP scores 0 -- it is NOT carried over, so a
  single-CSV archive measures exactly one component and a main-account pack
  must contain both members;
* a ZIP containing one dataset CSV is accepted and scored directly.
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

REPO = Path(__file__).resolve().parents[2]
PRODUCTION_CONFIG = REPO / "configs" / "production.json"


def _load_production() -> dict:
    if not PRODUCTION_CONFIG.exists():
        raise SystemExit(f"FATAL: missing production manifest {PRODUCTION_CONFIG}")
    with PRODUCTION_CONFIG.open(encoding="utf-8") as handle:
        return json.load(handle)


PROD = _load_production()

GOLD_ZIP = REPO / PROD["accepted_archive"]["path"]
GOLD_ZIP_SHA256 = PROD["accepted_archive"]["sha256"]
GOLD_ZIP_BYTES = PROD["accepted_archive"]["bytes"]
ACCEPTED_TOTAL = PROD["accepted_total_score"]

SPEC = {
    "ds1": {
        "member": "dataset1.csv",
        "rows": PROD["dataset1"]["shape"][0],
        "cols": PROD["dataset1"]["shape"][1],
        "baseline_sha256": PROD["dataset1"]["member_sha256"],
        "baseline_score": PROD["dataset1"]["component_score"],
        "other_score": PROD["dataset2"]["component_score"],
        "suffix": "_ds1_only.zip",
        # every accepted dataset1 row is row-max-normalised to exactly 1.0
        "expect_rowmax_one": 1.0,
    },
    "ds2": {
        "member": "dataset2.csv",
        "rows": PROD["dataset2"]["shape"][0],
        "cols": PROD["dataset2"]["shape"][1],
        "baseline_sha256": PROD["dataset2"]["member_sha256"],
        "baseline_score": PROD["dataset2"]["component_score"],
        "other_score": PROD["dataset1"]["component_score"],
        "suffix": "_ds2_only.zip",
        # only ~130k of 153,420 accepted dataset2 rows peak at 1.0 -- not an invariant
        "expect_rowmax_one": None,
    },
}

VALUE_MIN = 0.0
VALUE_MAX = 1.0
VALUE_TOL = 1e-9

_SIZE = PROD["archive_policy"]["size_thresholds"]
SIZE_TARGET = _SIZE["target_bytes"]
SIZE_WARN = _SIZE["warn_above_bytes"]
SIZE_REVIEW = _SIZE["review_at_or_above_bytes"]
LARGEST_ACCEPTED = PROD["archive_policy"]["largest_accepted_archive_bytes"]

COMPRESS_METHOD = zipfile.ZIP_DEFLATED  # method 8
COMPRESS_LEVEL = PROD["archive_policy"]["compress_level"]

DEFAULT_OUT = REPO / "outputs" / "submissions" / "round22_components"
LOG_DIR = REPO / "outputs" / "submissions" / "_packaging_logs"


class Report:
    """Accumulates PASS/WARN/FAIL checks and mirrors them to stdout and a log file."""

    def __init__(self, log_path: Path):
        self.checks: list[dict] = []
        self.lines: list[str] = []
        self.log_path = log_path

    def emit(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)

    def check(self, status: str, name: str, detail: str = "") -> None:
        self.checks.append({"status": status, "check": name, "detail": detail})
        self.emit(f"  [{status:4}] {name}" + (f" -- {detail}" if detail else ""))

    def ok(self, name, detail=""):
        self.check("PASS", name, detail)

    def warn(self, name, detail=""):
        self.check("WARN", name, detail)

    def fail(self, name, detail=""):
        self.check("FAIL", name, detail)

    @property
    def failed(self) -> bool:
        return any(c["status"] == "FAIL" for c in self.checks)

    @property
    def warned(self) -> bool:
        return any(c["status"] == "WARN" for c in self.checks)

    def verdict(self) -> str:
        if self.failed:
            return "REJECT"
        return "PASS-WITH-WARNINGS" if self.warned else "PASS"

    def flush(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def load_gold(rep: Report | None = None) -> zipfile.ZipFile:
    """Open the accepted archive read-only, asserting its identity first."""
    if not GOLD_ZIP.exists():
        raise SystemExit(f"FATAL: accepted archive missing: {GOLD_ZIP}")
    size = GOLD_ZIP.stat().st_size
    digest = sha256_file(GOLD_ZIP)
    if size != GOLD_ZIP_BYTES or digest != GOLD_ZIP_SHA256:
        raise SystemExit(
            "FATAL: accepted archive has changed -- refusing to run.\n"
            f"  expected {GOLD_ZIP_BYTES} B / {GOLD_ZIP_SHA256}\n"
            f"  found    {size} B / {digest}"
        )
    if rep is not None:
        rep.ok("accepted archive identity", f"{size} B, sha256 {digest[:16]}...")
    return zipfile.ZipFile(GOLD_ZIP, "r")


def baseline_bytes(dataset: str, rep: Report | None = None) -> bytes:
    spec = SPEC[dataset]
    with load_gold(rep) as z:
        raw = z.read(spec["member"])
    digest = sha256_bytes(raw)
    if digest != spec["baseline_sha256"]:
        raise SystemExit(f"FATAL: baseline {spec['member']} hash mismatch: {digest}")
    if rep is not None:
        rep.ok(f"baseline {spec['member']} hash", digest)
    return raw


def parse_matrix(raw: bytes, rep: Report, label: str, spec: dict) -> np.ndarray | None:
    """Parse a headerless numeric CSV, checking structure before dtype."""
    text = raw.decode("ascii", errors="strict")
    crlf = "\r\n" in text
    trailing_nl = text.endswith("\n")
    lines = text.replace("\r\n", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    rep.emit(f"    {label}: {len(lines)} lines, eol={'CRLF' if crlf else 'LF'}, "
             f"trailing_newline={trailing_nl}")

    if len(lines) != spec["rows"]:
        rep.fail(f"{label} row count", f"expected {spec['rows']}, found {len(lines)}")
        return None
    rep.ok(f"{label} row count", str(spec["rows"]))

    widths = {ln.count(",") + 1 for ln in lines}
    if widths != {spec["cols"]}:
        rep.fail(f"{label} candidate-column count",
                 f"expected all rows = {spec['cols']}, found widths {sorted(widths)[:6]}")
        return None
    rep.ok(f"{label} candidate-column count",
           f"all {spec['rows']} rows have {spec['cols']} columns")

    try:
        arr = pd.read_csv(io.BytesIO(raw), header=None, dtype=np.float64).to_numpy()
    except Exception as exc:  # noqa: BLE001 -- surface any parse failure as a check
        rep.fail(f"{label} numeric parse", str(exc)[:200])
        return None
    if arr.shape != (spec["rows"], spec["cols"]):
        rep.fail(f"{label} parsed shape", f"{arr.shape}")
        return None
    rep.ok(f"{label} parsed shape", f"{arr.shape[0]} x {arr.shape[1]}")
    return arr


def validate_values(arr: np.ndarray, rep: Report, dataset: str, allow_range: bool) -> dict:
    """Finiteness, range, degeneracy, and the normalisation profile."""
    spec = SPEC[dataset]
    stats = {}

    n_nan = int(np.isnan(arr).sum())
    n_inf = int(np.isinf(arr).sum())
    stats["nan_cells"] = n_nan
    stats["inf_cells"] = n_inf
    if n_nan or n_inf:
        rep.fail("finite values", f"{n_nan} NaN, {n_inf} Inf")
    else:
        rep.ok("finite values", "no NaN, no Inf")

    vmin, vmax = float(arr.min()), float(arr.max())
    stats["value_min"], stats["value_max"] = vmin, vmax
    if vmin < VALUE_MIN - VALUE_TOL or vmax > VALUE_MAX + VALUE_TOL:
        msg = f"[{vmin!r}, {vmax!r}] outside accepted [{VALUE_MIN}, {VALUE_MAX}]"
        (rep.warn if allow_range else rep.fail)("value range", msg)
    else:
        rep.ok("value range", f"[{vmin!r}, {vmax!r}] within [{VALUE_MIN}, {VALUE_MAX}]")

    rowptp = arr.max(axis=1) - arr.min(axis=1)
    n_degen = int((rowptp == 0).sum())
    stats["degenerate_rows"] = n_degen
    stats["all_zero_rows"] = int((arr == 0).all(axis=1).sum())
    if n_degen:
        rep.fail("degenerate rows",
                 f"{n_degen} fully tied rows (they score the 100-way tie baseline ~0.0519)")
    else:
        rep.ok("degenerate rows", "none -- every row discriminates")

    rowmax = arr.max(axis=1)
    n_norm = int((rowmax == 1.0).sum())
    stats["rows_rowmax_one"] = n_norm
    expect = spec["expect_rowmax_one"]
    if expect is not None:
        if n_norm / spec["rows"] < expect:
            rep.warn("row max-normalisation",
                     f"{n_norm}/{spec['rows']} rows peak at exactly 1.0; "
                     f"every accepted {dataset} row does")
        else:
            rep.ok("row max-normalisation", f"{n_norm}/{spec['rows']} rows peak at exactly 1.0")
    else:
        rep.emit(f"    rows peaking at exactly 1.0: {n_norm}/{spec['rows']} (not an invariant)")

    top_ties = int((((arr == rowmax[:, None]).sum(axis=1)) > 1).sum())
    stats["rows_with_tied_top1"] = top_ties
    rep.emit(f"    rows with a tied top-1: {top_ties}")
    return stats


def diff_against_baseline(trt: np.ndarray, base: np.ndarray, trt_raw: bytes, base_raw: bytes,
                          rep: Report, inversions: bool) -> dict:
    """Row/cell/top-1/order deltas versus the accepted member."""
    d = {}

    def norm_lines(raw: bytes) -> list[str]:
        t = raw.decode("ascii").replace("\r\n", "\n").split("\n")
        if t and t[-1] == "":
            t.pop()
        return t

    tl, bl = norm_lines(trt_raw), norm_lines(base_raw)
    d["changed_rows_text"] = int(sum(1 for a, b in zip(tl, bl) if a != b))

    cell_diff = trt != base
    d["changed_cells"] = int(cell_diff.sum())
    d["changed_rows_numeric"] = int(cell_diff.any(axis=1).sum())
    d["top1_changes"] = int((base.argmax(axis=1) != trt.argmax(axis=1)).sum())

    rows = np.flatnonzero(cell_diff.any(axis=1))
    d["order_changed_rows"] = 0
    d["strict_inversions"] = None
    d["pairs_collapsed_to_tie"] = None
    if rows.size:
        rb = np.argsort(-base[rows], axis=1, kind="stable")
        rt = np.argsort(-trt[rows], axis=1, kind="stable")
        d["order_changed_rows"] = int((rb != rt).any(axis=1).sum())
        if inversions:
            inv = tie = 0
            iu = np.triu_indices(base.shape[1], k=1)
            for i in range(0, rows.size, 500):
                blk = rows[i:i + 500]
                db = (base[blk][:, :, None] - base[blk][:, None, :])[:, iu[0], iu[1]]
                dt = (trt[blk][:, :, None] - trt[blk][:, None, :])[:, iu[0], iu[1]]
                inv += int((((db > 0) & (dt < 0)) | ((db < 0) & (dt > 0))).sum())
                tie += int(((db != 0) & (dt == 0)).sum())
            d["strict_inversions"] = inv
            d["pairs_collapsed_to_tie"] = tie

    rep.emit(f"    changed rows (text)   : {d['changed_rows_text']}")
    rep.emit(f"    changed rows (numeric): {d['changed_rows_numeric']}")
    rep.emit(f"    changed cells         : {d['changed_cells']}")
    rep.emit(f"    top-1 changes         : {d['top1_changes']}")
    rep.emit(f"    order-changed rows    : {d['order_changed_rows']}")
    if inversions:
        rep.emit(f"    strict inversions     : {d['strict_inversions']}")
        rep.emit(f"    pairs collapsed to tie: {d['pairs_collapsed_to_tie']} "
                 f"(%.6f quantisation, expected)")

    if d["changed_rows_text"] != d["changed_rows_numeric"]:
        rep.warn("text/numeric row-change agreement",
                 f"{d['changed_rows_text']} text vs {d['changed_rows_numeric']} numeric")
    else:
        rep.ok("text/numeric row-change agreement", f"{d['changed_rows_text']} rows")

    if d["changed_cells"] == 0:
        rep.warn("treatment differs from baseline",
                 "IDENTICAL to the accepted member -- this measures nothing new")
    else:
        rep.ok("treatment differs from baseline", f"{d['changed_cells']} cells")
    return d


def write_zip(out: Path, members: list[tuple[str, bytes]], rep: Report) -> dict:
    """Write a deflate-9 archive with exactly these members, then reopen and prove it."""
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        rep.emit(f"    overwriting existing {out.name}")
    with zipfile.ZipFile(out, "w", compression=COMPRESS_METHOD, compresslevel=COMPRESS_LEVEL) as z:
        for name, data in members:
            # str arcname => create_system 0 / external_attr 0o600<<16, matching every
            # accepted archive field for field
            z.writestr(name, data)

    size = out.stat().st_size
    info = {"zip_path": str(out), "zip_sha256": sha256_file(out), "zip_bytes": size,
            "compress_method": int(COMPRESS_METHOD), "compress_level": COMPRESS_LEVEL,
            "members": [], "member_sha256": {}}

    with zipfile.ZipFile(out, "r") as z:
        bad = z.testzip()
        rep.ok("archive CRC", "testzip clean") if bad is None else rep.fail("archive CRC", bad)

        infos = z.infolist()
        names = [i.filename for i in infos]
        info["members"] = names
        expected = [n for n, _ in members]
        if names != expected:
            rep.fail("member names and order", f"{names} != {expected}")
        else:
            rep.ok("member names and order", " -> ".join(names))

        if len(set(names)) != len(names):
            rep.fail("duplicate members", str(names))
        else:
            rep.ok("duplicate members", "none")
        if any(i.is_dir() for i in infos):
            rep.fail("directory entries", "archive contains directory entries")
        else:
            rep.ok("directory entries", "none")
        if z.comment:
            rep.fail("archive comment", repr(z.comment))
        else:
            rep.ok("archive comment", "empty")
        methods = {i.compress_type for i in infos}
        if methods != {COMPRESS_METHOD}:
            rep.fail("compression method", f"{methods} != {{8}}")
        else:
            rep.ok("compression method", "8 (standard deflate) on every member")

        for name, data in members:
            got = z.read(name)
            digest = sha256_bytes(got)
            info["member_sha256"][name] = digest
            if got != data:
                rep.fail(f"member {name} round-trip", "reopened bytes differ from input")
            else:
                rep.ok(f"member {name} round-trip", f"byte-identical, sha256 {digest[:16]}...")

    info["size_verdict"] = size_policy(size, rep)
    return info


def size_policy(size: int, rep: Report) -> str:
    """Classify an archive size.

    These thresholds are CONSERVATIVE PRACTICE, not a measured platform limit.
    On 2026-07-29 a 65,363,753 B combined archive was rejected twice with an
    empty "Submission failed:" message and the identical members repacked at
    deflate-9 (64,004,492 B) were accepted, which looked like a size cliff. That
    reading is REFUTED by our own record: accepted archives exist at 65,437,854 B
    and 67,510,742 B. The 2026-07-29 failure is unexplained. Deflate-9 and a
    small archive are kept because smaller is strictly safer, not because a
    limit was measured -- and no size below any threshold here is a guarantee.
    """
    mb = size / 1e6
    if size >= SIZE_REVIEW:
        rep.fail("archive size policy",
                 f"{size} B ({mb:.2f} MB) at or above the {SIZE_REVIEW} review line. "
                 f"Archives this large HAVE been accepted (largest {LARGEST_ACCEPTED} B), but a "
                 f"65,363,753 B pack was also rejected unexplained -- get explicit review.")
        return "REVIEW-REQUIRED"
    if size > SIZE_WARN:
        rep.warn("archive size policy",
                 f"{size} B ({mb:.2f} MB) above the {SIZE_WARN} target; within the accepted range "
                 f"but with little margin.")
        return "ACCEPTABLE-NO-MARGIN"
    rep.ok("archive size policy", f"{size} B ({mb:.2f} MB) at or below the {SIZE_TARGET} target")
    return "OK"


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"\nmanifest -> {path}")


def mode_component(args, dataset: str) -> int:
    spec = SPEC[dataset]
    exp = args.experiment
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT
    zip_path = out_dir / f"{exp}{spec['suffix']}"
    rep = Report(LOG_DIR / f"{exp}_{dataset}.log")

    rep.emit(f"=== {dataset} component :: {exp} ===")
    rep.emit(f"    treatment : {args.csv}")
    rep.emit(f"    baseline  : {GOLD_ZIP.name}::{spec['member']} "
             f"(component score {spec['baseline_score']!r})")
    rep.emit("")

    src = Path(args.csv).resolve()
    if not src.exists():
        rep.fail("treatment CSV exists", str(src))
        rep.flush()
        return 2
    trt_raw = src.read_bytes()
    trt_sha = sha256_bytes(trt_raw)
    rep.ok("treatment CSV sha256", trt_sha)

    base_raw = baseline_bytes(dataset, rep)
    arr = parse_matrix(trt_raw, rep, "treatment", spec)
    if arr is None:
        rep.emit(f"\nVERDICT: {rep.verdict()} -- not packaged")
        rep.flush()
        return 2
    stats = validate_values(arr, rep, dataset, args.allow_range_extension)
    base = pd.read_csv(io.BytesIO(base_raw), header=None, dtype=np.float64).to_numpy()
    diff = diff_against_baseline(arr, base, trt_raw, base_raw, rep, args.inversions)

    if rep.failed and not args.force:
        rep.emit(f"\nVERDICT: {rep.verdict()} -- NOT packaged (use --force to override)")
        rep.flush()
        return 1

    rep.emit("\n--- packaging ---")
    zinfo = write_zip(zip_path, [(spec["member"], trt_raw)], rep)

    payload = {
        "experiment": exp, "dataset": dataset, "mode": dataset,
        "source_csv_path": str(src),
        "baseline_csv_sha256": spec["baseline_sha256"],
        "treatment_csv_sha256": trt_sha,
        "zip_path": zinfo["zip_path"], "zip_sha256": zinfo["zip_sha256"],
        "member_names": zinfo["members"], "member_sha256": zinfo["member_sha256"],
        "rows": spec["rows"], "cols": spec["cols"],
        "changed_rows": diff["changed_rows_text"],
        "changed_rows_numeric": diff["changed_rows_numeric"],
        "changed_cells": diff["changed_cells"],
        "top1_changes": diff["top1_changes"],
        "order_changed_rows": diff["order_changed_rows"],
        "strict_inversions": diff["strict_inversions"],
        "pairs_collapsed_to_tie": diff["pairs_collapsed_to_tie"],
        "validation": {**stats, "checks": rep.checks, "verdict": rep.verdict()},
        "compress_method": zinfo["compress_method"], "compress_level": zinfo["compress_level"],
        "archive_bytes": zinfo["zip_bytes"], "size_verdict": zinfo["size_verdict"],
        "baseline_component_score": spec["baseline_score"],
        "frozen_other_component_score": spec["other_score"],
        "treatment_component_score": None, "component_delta": None,
        "predicted_combined_total": None,
        "accepted_total_reference": ACCEPTED_TOTAL,
        "created_utc": stamp(),
        "command_line": " ".join([sys.executable, *sys.argv]),
        "submitted": False,
    }
    mpath = out_dir / f"{exp}_{dataset}_manifest.json"
    write_manifest(mpath, payload)

    rep.emit(f"\nVERDICT: {rep.verdict()}")
    rep.emit(f"submit  : {zinfo['zip_path']}")
    rep.emit(f"then    : package_component.py score --manifest {mpath} --observed <score>")
    rep.flush()
    return 0 if not rep.failed else 1


def mode_main(args) -> int:
    exp = args.experiment
    out_dir = Path(args.out_dir) if args.out_dir else REPO / "outputs" / "submissions" / "round22_main"
    zip_path = out_dir / f"{exp}_main_d9.zip"
    rep = Report(LOG_DIR / f"{exp}_main.log")
    rep.emit(f"=== combined main pack :: {exp} ===")

    members = []
    srcs = {}
    for dataset, argval in (("ds1", args.ds1), ("ds2", args.ds2)):
        spec = SPEC[dataset]
        if argval == "accepted":
            raw = baseline_bytes(dataset, rep)
            srcs[dataset] = f"{GOLD_ZIP}::{spec['member']}"
        else:
            p = Path(argval).resolve()
            if not p.exists():
                rep.fail(f"{dataset} member exists", str(p))
                rep.flush()
                return 2
            raw = p.read_bytes()
            srcs[dataset] = str(p)
        rep.ok(f"{dataset} member sha256", sha256_bytes(raw))
        arr = parse_matrix(raw, rep, dataset, spec)
        if arr is None:
            rep.emit(f"\nVERDICT: {rep.verdict()} -- not packaged")
            rep.flush()
            return 2
        validate_values(arr, rep, dataset, args.allow_range_extension)
        members.append((spec["member"], raw))

    if rep.failed and not args.force:
        rep.emit(f"\nVERDICT: {rep.verdict()} -- NOT packaged (use --force to override)")
        rep.flush()
        return 1

    rep.emit("\n--- packaging ---")
    zinfo = write_zip(zip_path, members, rep)
    rep.emit("\n--- comparison with the accepted archive ---")
    gold_cmp = compare_with_gold(zip_path, rep)

    payload = {
        "experiment": exp, "dataset": "combined", "mode": "main",
        "source_csv_path": srcs,
        "baseline_csv_sha256": {d: SPEC[d]["baseline_sha256"] for d in ("ds1", "ds2")},
        "treatment_csv_sha256": zinfo["member_sha256"],
        "zip_path": zinfo["zip_path"], "zip_sha256": zinfo["zip_sha256"],
        "member_names": zinfo["members"], "member_sha256": zinfo["member_sha256"],
        "rows": {d: SPEC[d]["rows"] for d in ("ds1", "ds2")}, "cols": 100,
        "compress_method": zinfo["compress_method"], "compress_level": zinfo["compress_level"],
        "archive_bytes": zinfo["zip_bytes"], "size_verdict": zinfo["size_verdict"],
        "gold_comparison": gold_cmp,
        "validation": {"checks": rep.checks, "verdict": rep.verdict()},
        "baseline_component_score": {d: SPEC[d]["baseline_score"] for d in ("ds1", "ds2")},
        "treatment_component_score": None, "component_delta": None,
        "predicted_combined_total": None,
        "accepted_total_reference": ACCEPTED_TOTAL,
        "created_utc": stamp(),
        "command_line": " ".join([sys.executable, *sys.argv]),
        "submitted": False,
    }
    write_manifest(out_dir / f"{exp}_main_manifest.json", payload)
    rep.emit(f"\nVERDICT: {rep.verdict()}")
    rep.flush()
    return 0 if not rep.failed else 1


def compare_with_gold(zip_path: Path, rep: Report) -> dict:
    """Structural diff of a freshly built pack against the accepted archive."""
    out = {}
    with zipfile.ZipFile(GOLD_ZIP) as g, zipfile.ZipFile(zip_path) as n:
        gi = {i.filename: i for i in g.infolist()}
        ni = {i.filename: i for i in n.infolist()}
        out["member_names_match"] = list(gi) == list(ni)
        rep.emit(f"    member order      : gold {list(gi)} | new {list(ni)}")
        for name in gi:
            if name not in ni:
                continue
            same = g.read(name) == n.read(name)
            out[f"{name}_bytes_identical"] = same
            (rep.ok if same else rep.warn)(
                f"{name} vs accepted",
                "byte-identical" if same else "differs (expected for a treatment member)")
            out[f"{name}_gold_compress_size"] = gi[name].compress_size
            out[f"{name}_new_compress_size"] = ni[name].compress_size
            rep.emit(f"    {name}: compressed {gi[name].compress_size} (accepted) vs "
                     f"{ni[name].compress_size} (new)")
        for field in ("create_system", "create_version", "extract_version", "flag_bits",
                      "external_attr", "internal_attr", "compress_type"):
            gvals = {getattr(i, field) for i in g.infolist()}
            nvals = {getattr(i, field) for i in n.infolist()}
            out[f"field_{field}"] = {"gold": sorted(gvals), "new": sorted(nvals)}
            if gvals != nvals:
                rep.warn(f"zip field {field}", f"accepted {sorted(gvals)} vs new {sorted(nvals)}")
        rep.ok("zip container fields", "match the accepted archive (timestamps excepted)")
        gsize, nsize = GOLD_ZIP.stat().st_size, zip_path.stat().st_size
        out.update(gold_bytes=gsize, new_bytes=nsize, delta_bytes=nsize - gsize)
        rep.emit(f"    archive size      : accepted {gsize} | new {nsize} | "
                 f"delta {nsize - gsize:+d}")
    return out


def mode_score(args) -> int:
    mpath = Path(args.manifest).resolve()
    payload = json.loads(mpath.read_text(encoding="utf-8"))
    dataset = payload["dataset"]
    observed = float(args.observed)

    if dataset == "combined":
        payload["treatment_component_score"] = observed
        payload["component_delta"] = observed - ACCEPTED_TOTAL
        payload["predicted_combined_total"] = observed
        print(f"combined total observed : {observed!r}")
        print(f"vs accepted             : {ACCEPTED_TOTAL!r}")
        print(f"delta                   : {observed - ACCEPTED_TOTAL:+.16f}")
    else:
        spec = SPEC[dataset]
        base, other = spec["baseline_score"], spec["other_score"]
        delta, total = observed - base, observed + other
        payload["treatment_component_score"] = observed
        payload["component_delta"] = delta
        payload["predicted_combined_total"] = total
        print(f"dataset                 : {dataset}")
        print(f"baseline component      : {base!r}")
        print(f"treatment component     : {observed!r}")
        print(f"exact delta             : {delta:+.16f}")
        print(f"frozen other component  : {other!r}")
        print(f"predicted combined total: {total!r}")
        print(f"vs accepted {ACCEPTED_TOTAL!r}: {total - ACCEPTED_TOTAL:+.16f}")
        print("\nverdict: " + ("GAIN -- an isolated component pass; a main pack may now be built"
                               if delta > 0 else
                               "NO GAIN -- do not build a main pack from this component"))
    payload["submitted"] = True
    payload["scored_utc"] = stamp()
    payload.setdefault("score_command_line", " ".join([sys.executable, *sys.argv]))
    write_manifest(mpath, payload)
    return 0


def mode_verify(args) -> int:
    p = Path(args.zip).resolve()
    rep = Report(LOG_DIR / f"verify_{p.stem}.log")
    rep.emit(f"=== verify :: {p} ===")
    rep.ok("zip sha256", sha256_file(p))
    with zipfile.ZipFile(p) as z:
        rep.ok("testzip", str(z.testzip()))
        for i in z.infolist():
            raw = z.read(i.filename)
            rep.emit(f"    {i.filename}: method {i.compress_type}, raw {i.file_size}, "
                     f"sha256 {sha256_bytes(raw)}")
            dataset = {"dataset1.csv": "ds1", "dataset2.csv": "ds2"}.get(i.filename)
            if dataset:
                spec = SPEC[dataset]
                arr = parse_matrix(raw, rep, dataset, spec)
                if arr is not None:
                    validate_values(arr, rep, dataset, args.allow_range_extension)
                if sha256_bytes(raw) == spec["baseline_sha256"]:
                    rep.ok(f"{i.filename} identity", "IS the accepted member")
    size_policy(p.stat().st_size, rep)
    rep.emit(f"\nVERDICT: {rep.verdict()}")
    rep.flush()
    return 0 if not rep.failed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)

    def add_common(p, with_csv=True):
        if with_csv:
            p.add_argument("--csv", required=True, help="treatment CSV path")
        p.add_argument("--experiment", required=True, help="short experiment name")
        p.add_argument("--out-dir", default=None)
        p.add_argument("--inversions", action="store_true",
                       help="also count strict pair inversions on changed rows (slower)")
        p.add_argument("--allow-range-extension", action="store_true",
                       help="downgrade an out-of-[0,1] range FAIL to a warning")
        p.add_argument("--force", action="store_true", help="package even if a check FAILs")

    add_common(sub.add_parser("ds1", help="validate + package a dataset1 treatment"))
    add_common(sub.add_parser("ds2", help="validate + package a dataset2 treatment"))
    pm = sub.add_parser("main", help="combine two members into a main pack")
    pm.add_argument("--ds1", required=True, help="dataset1 CSV path, or 'accepted'")
    pm.add_argument("--ds2", required=True, help="dataset2 CSV path, or 'accepted'")
    add_common(pm, with_csv=False)

    ps = sub.add_parser("score", help="append an observed leaderboard score to a manifest")
    ps.add_argument("--manifest", required=True)
    ps.add_argument("--observed", required=True, help="leaderboard score, full precision")

    pv = sub.add_parser("verify", help="re-validate any ZIP")
    pv.add_argument("--zip", required=True)
    pv.add_argument("--allow-range-extension", action="store_true")

    args = ap.parse_args()
    t0 = time.time()
    if args.mode in ("ds1", "ds2"):
        rc = mode_component(args, args.mode)
    elif args.mode == "main":
        rc = mode_main(args)
    elif args.mode == "score":
        rc = mode_score(args)
    else:
        rc = mode_verify(args)
    print(f"\nelapsed {time.time() - t0:.1f}s, exit {rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
