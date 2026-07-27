#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only validator for the dataset2-only footprint A/B probe packs.

The validator deliberately does not create, rewrite, or repair either ZIP.  It
checks the transport contract that can be established without hidden labels:

* exactly one ZIP entry, named ``dataset2.csv``;
* exactly 153,420 headerless rows and 100 finite numeric scores per row;
* identical A/B matrix shape;
* archive and inner-file MD5 hashes;
* score-level and top-1 A/B difference statistics.

Candidate identity/order cannot be recovered from a score-only submission, so
this script reports that limitation explicitly.  Pair it with the builder's
assertion that both arms use the unchanged physical ``test.csv`` row/column
order.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import zipfile
from pathlib import Path
from typing import BinaryIO, Dict, Iterator, Optional, Sequence, TextIO, Tuple

import numpy as np


EXPECTED_ENTRY = "dataset2.csv"
EXPECTED_ROWS = 153_420
EXPECTED_COLUMNS = 100
BUFFER_SIZE = 8 << 20
HERE = Path(__file__).resolve().parent
# Packs are produced by footprint_ab_probe.py under the gitignored repro dir.
WORKDIR = HERE.parent / "outputs" / "dataset2-footprint"


class ValidationError(RuntimeError):
    """Raised when a pack violates the dataset2-only submission contract."""


def _md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            block = handle.read(BUFFER_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _hash_inner(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo
) -> Tuple[str, int, int, bool]:
    digest = hashlib.md5()
    byte_count = 0
    newline_count = 0
    last_byte = b""
    with archive.open(info, "r") as handle:
        while True:
            block = handle.read(BUFFER_SIZE)
            if not block:
                break
            digest.update(block)
            byte_count += len(block)
            newline_count += block.count(b"\n")
            last_byte = block[-1:]
    return digest.hexdigest(), byte_count, newline_count, last_byte == b"\n"


def _open_and_describe(
    path: Path,
) -> Tuple[zipfile.ZipFile, zipfile.ZipInfo, Dict[str, object]]:
    path = path.resolve()
    if not path.is_file():
        raise ValidationError(f"pack does not exist: {path}")
    if not zipfile.is_zipfile(path):
        raise ValidationError(f"not a valid ZIP archive: {path}")

    archive = zipfile.ZipFile(path, "r")
    bad_member = archive.testzip()
    if bad_member is not None:
        archive.close()
        raise ValidationError(f"ZIP CRC failure in {path}: {bad_member}")

    entries = [item for item in archive.infolist() if not item.is_dir()]
    names = [item.filename for item in entries]
    if names != [EXPECTED_ENTRY]:
        archive.close()
        raise ValidationError(
            f"{path} must contain exactly [{EXPECTED_ENTRY!r}], got {names!r}"
        )

    info = entries[0]
    inner_md5, inner_bytes, newline_count, trailing_newline = _hash_inner(
        archive, info
    )
    if inner_bytes != info.file_size:
        archive.close()
        raise ValidationError(
            f"{path}: streamed {inner_bytes} bytes but ZIP records "
            f"{info.file_size}"
        )
    if newline_count != EXPECTED_ROWS:
        archive.close()
        raise ValidationError(
            f"{path}: expected {EXPECTED_ROWS} newline-terminated rows, "
            f"found {newline_count}"
        )
    if not trailing_newline:
        archive.close()
        raise ValidationError(f"{path}: dataset2.csv lacks its final newline")

    description: Dict[str, object] = {
        "path": str(path),
        "archive_bytes": int(path.stat().st_size),
        "archive_md5": _md5_file(path),
        "entry_count": 1,
        "entry": EXPECTED_ENTRY,
        "inner_bytes": int(inner_bytes),
        "inner_md5": inner_md5,
        "compressed_bytes": int(info.compress_size),
        "compression": (
            "deflate"
            if info.compress_type == zipfile.ZIP_DEFLATED
            else str(info.compress_type)
        ),
        "newline_count": int(newline_count),
        "trailing_newline": bool(trailing_newline),
    }
    return archive, info, description


def _parse_score_line(
    line: str, *, arm: str, row_index: int
) -> np.ndarray:
    stripped = line.rstrip("\r\n")
    if not stripped:
        raise ValidationError(f"{arm}: empty row at zero-based index {row_index}")
    comma_count = stripped.count(",")
    if comma_count != EXPECTED_COLUMNS - 1:
        raise ValidationError(
            f"{arm}: row {row_index} has {comma_count + 1} fields, "
            f"expected {EXPECTED_COLUMNS}"
        )
    values = np.fromstring(stripped, dtype=np.float64, sep=",")
    if values.size != EXPECTED_COLUMNS:
        raise ValidationError(
            f"{arm}: row {row_index} parsed as {values.size} numbers, "
            f"expected {EXPECTED_COLUMNS}"
        )
    if not np.isfinite(values).all():
        bad_columns = np.flatnonzero(~np.isfinite(values)).tolist()
        raise ValidationError(
            f"{arm}: non-finite values at row {row_index}, "
            f"columns {bad_columns[:10]}"
        )
    return values


def _text_stream(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo
) -> Tuple[BinaryIO, TextIO]:
    binary = archive.open(info, "r")
    text = io.TextIOWrapper(binary, encoding="ascii", newline="")
    return binary, text


def validate_pair(a_path: Path, b_path: Path) -> Dict[str, object]:
    archive_a, info_a, report_a = _open_and_describe(a_path)
    archive_b: Optional[zipfile.ZipFile] = None
    binary_a: Optional[BinaryIO] = None
    binary_b: Optional[BinaryIO] = None
    text_a: Optional[TextIO] = None
    text_b: Optional[TextIO] = None
    try:
        archive_b, info_b, report_b = _open_and_describe(b_path)
        binary_a, text_a = _text_stream(archive_a, info_a)
        binary_b, text_b = _text_stream(archive_b, info_b)

        row_count = 0
        changed_rows = 0
        changed_cells = 0
        top1_changed_rows = 0
        a_nonconstant_rows = 0
        b_nonconstant_rows = 0
        a_min = math.inf
        a_max = -math.inf
        b_min = math.inf
        b_max = -math.inf
        signed_sum = 0.0
        absolute_sum = 0.0
        square_sum = 0.0
        max_absolute = 0.0
        max_row_absolute = 0.0

        sentinel = object()
        for row_index, pair in enumerate(
            itertools.zip_longest(text_a, text_b, fillvalue=sentinel)
        ):
            line_a, line_b = pair
            if line_a is sentinel or line_b is sentinel:
                raise ValidationError(
                    "A/B row counts differ before reaching the expected size"
                )
            values_a = _parse_score_line(
                line_a, arm="A", row_index=row_index
            )
            values_b = _parse_score_line(
                line_b, arm="B", row_index=row_index
            )

            a_min = min(a_min, float(values_a.min()))
            a_max = max(a_max, float(values_a.max()))
            b_min = min(b_min, float(values_b.min()))
            b_max = max(b_max, float(values_b.max()))
            a_nonconstant_rows += int(float(np.ptp(values_a)) > 0.0)
            b_nonconstant_rows += int(float(np.ptp(values_b)) > 0.0)

            difference = values_b - values_a
            absolute = np.abs(difference)
            changed = difference != 0.0
            n_changed = int(np.count_nonzero(changed))
            changed_cells += n_changed
            changed_rows += int(n_changed > 0)
            top1_changed_rows += int(
                int(np.argmax(values_a)) != int(np.argmax(values_b))
            )
            signed_sum += float(difference.sum(dtype=np.float64))
            absolute_sum += float(absolute.sum(dtype=np.float64))
            square_sum += float(
                np.square(difference, dtype=np.float64).sum(dtype=np.float64)
            )
            row_max = float(absolute.max())
            max_absolute = max(max_absolute, row_max)
            max_row_absolute += row_max
            row_count += 1

        if row_count != EXPECTED_ROWS:
            raise ValidationError(
                f"expected {EXPECTED_ROWS} rows, parsed {row_count}"
            )
        cell_count = row_count * EXPECTED_COLUMNS
        if a_nonconstant_rows != row_count:
            raise ValidationError(
                f"A has {row_count - a_nonconstant_rows} constant-score rows"
            )
        if b_nonconstant_rows != row_count:
            raise ValidationError(
                f"B has {row_count - b_nonconstant_rows} constant-score rows"
            )

        report_a.update(
            {
                "rows": row_count,
                "columns": EXPECTED_COLUMNS,
                "all_finite": True,
                "nonconstant_rows": a_nonconstant_rows,
                "score_min": a_min,
                "score_max": a_max,
            }
        )
        report_b.update(
            {
                "rows": row_count,
                "columns": EXPECTED_COLUMNS,
                "all_finite": True,
                "nonconstant_rows": b_nonconstant_rows,
                "score_min": b_min,
                "score_max": b_max,
            }
        )
        pair_report: Dict[str, object] = {
            "shape_equal": True,
            "rows": row_count,
            "columns": EXPECTED_COLUMNS,
            "cells": cell_count,
            "changed_rows": changed_rows,
            "changed_row_fraction": changed_rows / row_count,
            "changed_cells": changed_cells,
            "changed_cell_fraction": changed_cells / cell_count,
            "top1_changed_rows": top1_changed_rows,
            "top1_changed_fraction": top1_changed_rows / row_count,
            "mean_signed_B_minus_A": signed_sum / cell_count,
            "mean_absolute_difference": absolute_sum / cell_count,
            "root_mean_square_difference": math.sqrt(
                square_sum / cell_count
            ),
            "max_absolute_difference": max_absolute,
            "mean_row_max_absolute_difference": max_row_absolute / row_count,
            "byte_identical_inner_csv": (
                report_a["inner_md5"] == report_b["inner_md5"]
                and report_a["inner_bytes"] == report_b["inner_bytes"]
            ),
            "candidate_order_checked": False,
            "candidate_order_note": (
                "A score-only ZIP cannot establish candidate identity/order; "
                "the common physical test.csv order must be asserted by the "
                "A/B builder."
            ),
        }
        return {
            "status": "ok",
            "contract": {
                "dataset": "dataset2-only",
                "required_entry": EXPECTED_ENTRY,
                "expected_shape": [EXPECTED_ROWS, EXPECTED_COLUMNS],
            },
            "pack_A": report_a,
            "pack_B": report_b,
            "A_vs_B": pair_report,
        }
    finally:
        if text_a is not None:
            text_a.close()
        elif binary_a is not None:
            binary_a.close()
        if text_b is not None:
            text_b.close()
        elif binary_b is not None:
            binary_b.close()
        archive_a.close()
        if archive_b is not None:
            archive_b.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pack-a",
        type=Path,
        default=WORKDIR / "pack_A_ds2only.zip",
    )
    parser.add_argument(
        "--pack-b",
        type=Path,
        default=WORKDIR / "pack_B_ds2only.zip",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="optional report path; never modifies either input ZIP",
    )
    args = parser.parse_args(argv)

    try:
        report = validate_pair(args.pack_a, args.pack_b)
    except (OSError, ValueError, zipfile.BadZipFile, ValidationError) as error:
        print(
            json.dumps(
                {"status": "error", "error": str(error)},
                indent=2,
                ensure_ascii=False,
            )
        )
        return 1

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json_out is not None:
        output = args.json_out.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
