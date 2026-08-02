# -*- coding: utf-8 -*-
"""The stage-completion contract: OUTPUT EXISTS != STAGE COMPLETE.

Every long stage of this pipeline writes one artifact and, historically, the
driver decided whether to re-run it by asking whether that artifact was on disk.
That is not a completeness test. A file is equally present when it was written
by a different commit, from different inputs, under a different configuration,
by an interrupted run that happened to reach a periodic export, or by a crashed
process that got as far as the rename. During the dataset-1 clean reexecution
all sixteen artifacts existed and the maintained driver would have reported a
full success in seconds; they had to be moved aside by hand.

This module replaces that test. A stage is reusable only when a *completion
record* proves it: an atomically published JSON document that binds the
producing commit, the identity of every input, the scientific configuration, the
identity of the output itself and the result of a stage-specific validation. The
record is written last, after the output is final, so its presence is evidence
that the stage ran to completion successfully. Anything else -- a missing
record, a mismatched hash, an unsupported schema, a short epoch count, a
leftover ``.part`` -- stops the run with a diagnostic. Nothing is silently
skipped, silently deleted or silently repaired.

Location policy
---------------
A record lives beside its output: ``<output>.done.json``. The two therefore
travel together. Deleting an output directory deletes its records with it, which
fails in the safe direction: the next run sees "no output, no record" and
re-runs, instead of finding an orphaned record that claims a vanished artifact.

Reuse policy
------------
Reuse is strict by default. Every one of the following must hold, and any single
mismatch is a stop, never a re-run and never a skip:

1. the output exists;
2. a record exists beside it;
3. the record parses and declares a supported ``schema_version``;
4. the record's ``exit_code`` is 0;
5. every declared input still hashes to the recorded value;
6. the configuration digest matches;
7. the relevant-code digest matches AND the producing commit matches;
8. the output's size and SHA256 match the recorded values;
9. ``achieved_units`` equals ``requested_units`` where the stage has units;
10. the recorded stage validation passed.

Both the producing commit and a relevant-code digest are recorded, and both are
compared. The commit is the coarse, always-available identity; the digest is the
precise one, over the exact source files the stage reads. Neither substitutes
for the other and there is no rule that treats a commit difference as ignorable
because "the change was only documentation" -- that judgement is not mechanical
and this module does not make it.

There is deliberately no ``--skip-checks``. The escape hatch is
:func:`quarantine`, which moves an unusable output and its record into
``outputs/_quarantine/NNN/`` and records why, so the evidence survives and the
next run starts clean.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

SCHEMA_VERSION = 1
#: Schema versions this module can read. A record outside this set is a stop,
#: not a re-run: an unknown schema may bind fewer properties than we require.
SUPPORTED_SCHEMA_VERSIONS = (1,)

RECORD_SUFFIX = ".done.json"
PART_SUFFIX = ".part"
QUARANTINE_DIRNAME = "_quarantine"

#: Packages whose versions are recorded as the environment identity. Read from
#: installed metadata, never imported: importing jittor compiles kernels.
ENVIRONMENT_PACKAGES = ("jittor", "numpy", "pandas", "scipy", "scikit-learn",
                        "lightgbm", "tqdm", "jittor_geometric")


class StageContractError(RuntimeError):
    """A stage cannot proceed and the operator must decide what to do."""


# --------------------------------------------------------------------------
# identity primitives
# --------------------------------------------------------------------------

def sha256_file(path: str | Path, *, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(payload) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=_jsonable)


def _jsonable(value):
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"{type(value).__name__} is not serialisable in a completion record")


def digest_of(payload) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_identity(path: str | Path, *, root: Path | None = None) -> dict:
    """Path, size and SHA256 of one file, path expressed relative to ``root``."""
    path = Path(path)
    if not path.is_file():
        raise StageContractError(f"cannot identify a file that does not exist: {path}")
    return {"path": _relative(path, root), "size": path.stat().st_size,
            "sha256": sha256_file(path)}


def _relative(path: Path, root: Path | None) -> str:
    path = Path(path)
    if root is not None:
        try:
            return path.resolve().relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def code_identity(paths: Sequence[str | Path], *, root: Path | None = None) -> dict:
    """A digest over the exact source files a stage's behaviour depends on.

    Finer than a commit: it changes only when one of these files changes. It is
    recorded *in addition* to the commit, never instead of it.
    """
    files = {}
    for path in sorted(Path(p) for p in paths):
        if not path.is_file():
            raise StageContractError(f"relevant-code file is missing: {path}")
        files[_relative(path, root)] = sha256_file(path)
    return {"files": files, "digest": digest_of(files)}


def git_commit(repo: Path) -> str:
    """The producing commit, or an explicit marker when git cannot answer.

    ``UNKNOWN`` never compares equal to a real commit, so a record written
    outside a git checkout can never be reused inside one, or the reverse.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "UNKNOWN"
    if completed.returncode != 0:
        return "UNKNOWN"
    return completed.stdout.strip() or "UNKNOWN"


def environment_identity() -> dict:
    """Interpreter and installed package versions. No package is imported."""
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - Python 3.8+ always has it
        metadata = None
    packages = {}
    for name in ENVIRONMENT_PACKAGES:
        version = "ABSENT"
        if metadata is not None:
            try:
                version = metadata.version(name)
            except Exception:  # noqa: BLE001 - absence is the normal case
                version = "ABSENT"
        packages[name] = version
    return {"python": sys.version.split()[0], "platform": sys.platform,
            "packages": packages}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------
# atomic publication
# --------------------------------------------------------------------------

@contextmanager
def atomic_output(path: str | Path):
    """Yield a temporary sibling path; publish it over ``path`` on clean exit.

    On any exception the temporary file is removed and ``path`` is left exactly
    as it was, so a failed stage cannot leave a half-written artifact under the
    real name. Publication is a rename within one directory, which is atomic.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + PART_SUFFIX)
    if tmp.exists():
        tmp.unlink()
    try:
        yield tmp
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    if not tmp.exists():
        raise StageContractError(f"atomic_output: nothing was written to {tmp}")
    os.replace(tmp, path)


def record_path(output: str | Path) -> Path:
    return Path(output).with_name(Path(output).name + RECORD_SUFFIX)


def part_path(output: str | Path) -> Path:
    return Path(output).with_name(Path(output).name + PART_SUFFIX)


# --------------------------------------------------------------------------
# the stage specification
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StageSpec:
    """Everything needed both to run a stage and to decide whether to reuse it."""

    stage_id: str
    dataset: str
    artifact_kind: str
    command: Sequence[str]
    env: dict
    output: Path
    inputs: Sequence[Path]
    config: dict
    code_files: Sequence[Path]
    seed: int | None = None
    requested_units: int | None = None
    unit_name: str | None = None
    #: Restricts the configuration digest to these keys. Use it when ``config``
    #: also carries geometry that is only measurable *after* the stage runs
    #: (a query count, a row count): such a value cannot take part in a
    #: before-the-fact reuse comparison, but it is still recorded, and the
    #: validator reads it back from the record to re-check the artifact.
    digest_keys: tuple[str, ...] | None = None
    #: Reads the stage's captured log and returns the units actually completed.
    achieved_units_from_log: Callable[[str], int | None] | None = None
    #: Validates the finished output. Returns a JSON-serialisable detail dict;
    #: raises to fail the stage. Runs BEFORE the record is written, always.
    validator: Callable[["StageSpec", Path], dict] | None = None
    #: An in-process step, for the few stages that are pure file assembly and
    #: would gain nothing from a subprocess. Takes the destination path and
    #: returns nothing; raising fails the stage. ``command`` still records what
    #: the step is equivalent to, so the record reads the same either way.
    action: Callable[[Path], None] | None = None
    description: str = ""

    def config_digest(self) -> str:
        if self.digest_keys is None:
            return digest_of(self.config)
        missing = [k for k in self.digest_keys if k not in self.config]
        if missing:
            raise StageContractError(
                f"{self.stage_id}: digest keys absent from config: {missing}")
        return digest_of({k: self.config[k] for k in self.digest_keys})


@dataclass
class Decision:
    """What to do about a stage, and the evidence for it."""

    action: str          # "RUN" | "REUSE" | "STOP"
    code: str            # machine-readable reason slug
    detail: str          # operator-facing explanation
    record: dict | None = None
    mismatches: list = field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - explicitness is the point
        raise TypeError("compare Decision.action explicitly; a Decision is not a bool")


# --------------------------------------------------------------------------
# reuse evaluation
# --------------------------------------------------------------------------

def evaluate(spec: StageSpec, *, repo: Path) -> Decision:
    """Decide whether ``spec`` may be reused, must run, or must stop the pipeline.

    Fail-closed: every inconsistency is ``STOP``. ``RUN`` is returned only for a
    genuinely clean slate -- no output, no record, no leftover part file.
    """
    output = Path(spec.output)
    record_file = record_path(output)
    leftover = part_path(output)

    if leftover.exists():
        return Decision("STOP", "STALE_PART_FILE",
                        f"a leftover partial file is present: {leftover}. A previous "
                        f"run of {spec.stage_id} died mid-write. Quarantine or remove "
                        f"it deliberately before continuing.")

    if not output.exists() and not record_file.exists():
        return Decision("RUN", "CLEAN_SLATE", f"{spec.stage_id} has no output and no record")

    if output.exists() and not record_file.exists():
        return Decision("STOP", "MISSING_COMPLETION_RECORD",
                        f"{output} exists but has no completion record. Its provenance "
                        f"cannot be established, so it is not proof that "
                        f"{spec.stage_id} completed. This is exactly the condition that "
                        f"made the old driver report a false success.")

    if record_file.exists() and not output.exists():
        return Decision("STOP", "MISSING_OUTPUT",
                        f"{record_file} claims {spec.stage_id} completed but its output "
                        f"{output} is absent.")

    try:
        record = json.loads(record_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as problem:
        return Decision("STOP", "MALFORMED_RECORD",
                        f"{record_file} is not readable JSON: {problem}")
    if not isinstance(record, dict):
        return Decision("STOP", "MALFORMED_RECORD",
                        f"{record_file} does not contain a JSON object")

    version = record.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        return Decision("STOP", "UNSUPPORTED_SCHEMA",
                        f"{record_file} declares schema_version {version!r}; this build "
                        f"understands {list(SUPPORTED_SCHEMA_VERSIONS)}.", record)

    missing = [key for key in REQUIRED_RECORD_KEYS if key not in record]
    if missing:
        return Decision("STOP", "INCOMPLETE_RECORD",
                        f"{record_file} is missing required fields: {missing}", record)

    mismatches: list[dict] = []

    def note(what: str, expected, found) -> None:
        mismatches.append({"property": what, "recorded": expected, "current": found})

    if record["stage_id"] != spec.stage_id:
        note("stage_id", record["stage_id"], spec.stage_id)
    if record["dataset"] != spec.dataset:
        note("dataset", record["dataset"], spec.dataset)
    if record["exit_code"] != 0:
        note("exit_code", record["exit_code"], 0)
    if record.get("validation", {}).get("status") != "PASS":
        note("validation.status", record.get("validation", {}).get("status"), "PASS")

    current_commit = git_commit(repo)
    if record["producing_commit"] != current_commit:
        note("producing_commit", record["producing_commit"], current_commit)
    try:
        current_code = code_identity(spec.code_files, root=repo)
    except StageContractError as problem:
        return Decision("STOP", "CODE_IDENTITY_UNAVAILABLE", str(problem), record)
    if record["code_identity"]["digest"] != current_code["digest"]:
        note("code_identity.digest", record["code_identity"]["digest"],
             current_code["digest"])

    if record["config_digest"] != spec.config_digest():
        note("config_digest", record["config_digest"], spec.config_digest())
    if record.get("seed") != spec.seed:
        note("seed", record.get("seed"), spec.seed)
    if record.get("requested_units") != spec.requested_units:
        note("requested_units", record.get("requested_units"), spec.requested_units)
    if spec.requested_units is not None:
        achieved = record.get("achieved_units")
        if achieved != spec.requested_units:
            note("achieved_units", achieved, spec.requested_units)

    recorded_inputs = {entry["path"]: entry for entry in record["inputs"]}
    expected_inputs = {_relative(Path(p), repo) for p in spec.inputs}
    if set(recorded_inputs) != expected_inputs:
        note("input set", sorted(recorded_inputs), sorted(expected_inputs))
    else:
        for path in spec.inputs:
            key = _relative(Path(path), repo)
            entry = recorded_inputs[key]
            if not Path(path).is_file():
                note(f"input {key}", entry["sha256"], "ABSENT")
                continue
            if Path(path).stat().st_size != entry["size"]:
                note(f"input {key} size", entry["size"], Path(path).stat().st_size)
            elif sha256_file(path) != entry["sha256"]:
                note(f"input {key} sha256", entry["sha256"], sha256_file(path))

    actual_size = output.stat().st_size
    if record["output"]["size"] != actual_size:
        note("output size", record["output"]["size"], actual_size)
    else:
        actual_sha = sha256_file(output)
        if record["output"]["sha256"] != actual_sha:
            note("output sha256", record["output"]["sha256"], actual_sha)

    if mismatches:
        return Decision("STOP", "IDENTITY_MISMATCH",
                        f"{spec.stage_id} has a completion record that does not describe "
                        f"the current state; {len(mismatches)} properties differ.",
                        record, mismatches)

    return Decision("REUSE", "VALIDATED_COMPLETION_RECORD",
                    f"{spec.stage_id} was completed by commit {record['producing_commit'][:12]} "
                    f"from identical inputs and configuration.", record)


REQUIRED_RECORD_KEYS = (
    "schema_version", "stage_id", "dataset", "artifact_kind", "producing_commit",
    "code_identity", "command", "environment", "inputs", "config", "config_digest",
    "output", "validation", "exit_code", "started_at", "completed_at",
)


# --------------------------------------------------------------------------
# publication
# --------------------------------------------------------------------------

def build_record(spec: StageSpec, *, repo: Path, exit_code: int, started_at: str,
                 completed_at: str, achieved_units: int | None,
                 validation: dict, output_detail: dict | None = None) -> dict:
    """Assemble the completion document. Never writes anything."""
    output = Path(spec.output)
    record = {
        "schema_version": SCHEMA_VERSION,
        "stage_id": spec.stage_id,
        "dataset": spec.dataset,
        "artifact_kind": spec.artifact_kind,
        "description": spec.description,
        "producing_commit": git_commit(repo),
        "code_identity": code_identity(spec.code_files, root=repo),
        "command": list(spec.command),
        "command_env": dict(spec.env),
        "environment": environment_identity(),
        "inputs": [file_identity(p, root=repo) for p in spec.inputs],
        "config": spec.config,
        "config_digest": spec.config_digest(),
        "seed": spec.seed,
        "unit_name": spec.unit_name,
        "requested_units": spec.requested_units,
        "achieved_units": achieved_units,
        "output": {**file_identity(output, root=repo), **(output_detail or {})},
        "validation": validation,
        "exit_code": exit_code,
        "started_at": started_at,
        "completed_at": completed_at,
    }
    return record


def publish_record(spec: StageSpec, record: dict) -> Path:
    """Write the completion record atomically. This is the LAST step of a stage.

    The document is serialised to ``<output>.done.json.part``, flushed and
    fsynced, and only then renamed into place. Until that rename the stage is
    incomplete by definition, so a crash anywhere earlier leaves no record.
    """
    if record.get("exit_code") != 0:
        raise StageContractError(
            f"refusing to publish a completion record for {spec.stage_id} with "
            f"exit_code {record.get('exit_code')!r}")
    if record.get("validation", {}).get("status") != "PASS":
        raise StageContractError(
            f"refusing to publish a completion record for {spec.stage_id} whose "
            f"validation status is {record.get('validation', {}).get('status')!r}")
    target = record_path(spec.output)
    tmp = target.with_name(target.name + PART_SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, indent=2, sort_keys=True, default=_jsonable) + "\n"
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return target


def complete_stage(spec: StageSpec, *, repo: Path, exit_code: int, started_at: str,
                   completed_at: str, log_text: str = "") -> dict:
    """Validate a finished stage and publish its record, in the fail-closed order.

    Order is load-bearing: a non-zero exit is rejected before anything else; the
    stage validator runs on the published output; the achieved unit count is
    read from the stage's own log and compared with the requested count; only
    then is the record written, and it is written atomically.
    """
    if exit_code != 0:
        raise StageContractError(
            f"{spec.stage_id} exited {exit_code}; no completion record written")
    output = Path(spec.output)
    if not output.is_file():
        raise StageContractError(
            f"{spec.stage_id} exited 0 but produced no output at {output}")
    leftover = part_path(output)
    if leftover.exists():
        raise StageContractError(
            f"{spec.stage_id} exited 0 but left a partial file at {leftover}")

    achieved = None
    if spec.achieved_units_from_log is not None:
        achieved = spec.achieved_units_from_log(log_text)
    if spec.requested_units is not None and achieved != spec.requested_units:
        raise StageContractError(
            f"{spec.stage_id} reported {achieved} of {spec.requested_units} "
            f"{spec.unit_name or 'units'} in its log; the stage is incomplete and no "
            f"completion record was written")

    detail: dict = {}
    if spec.validator is not None:
        detail = spec.validator(spec, output) or {}
    validation = {"status": "PASS", "validator": getattr(spec.validator, "__name__", None),
                  "detail": detail}

    record = build_record(spec, repo=repo, exit_code=exit_code, started_at=started_at,
                          completed_at=completed_at, achieved_units=achieved,
                          validation=validation)
    publish_record(spec, record)
    return record


# --------------------------------------------------------------------------
# the explicit operator escape hatch
# --------------------------------------------------------------------------

def quarantine(paths: Iterable[str | Path], *, outputs_root: Path, reason: str) -> Path:
    """Move unusable artifacts into ``outputs/_quarantine/NNN/`` and say why.

    This is the only sanctioned way past a ``STOP``. It never deletes: the
    evidence that produced the stop is preserved, numbered and annotated, which
    is what makes it safe to restart a stage cleanly.
    """
    root = Path(outputs_root) / QUARANTINE_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    existing = [int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
    slot = root / f"{(max(existing) + 1 if existing else 1):03d}"
    slot.mkdir()
    moved = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        target = slot / path.name
        os.replace(path, target)
        moved.append(target.name)
    (slot / "QUARANTINE.json").write_text(
        json.dumps({"reason": reason, "moved": moved, "quarantined_at": utc_now()},
                   indent=2) + "\n", encoding="utf-8")
    return slot


# --------------------------------------------------------------------------
# achieved-unit extractors and stage validators
# --------------------------------------------------------------------------

def epochs_from_log(pattern: str) -> Callable[[str], int | None]:
    """Build a reader for a trainer's own ``epoch N/M`` progress lines.

    Returns the highest N the stage actually reported. An interrupted trainer
    whose periodic export left a complete-looking artifact reports fewer epochs
    than were requested, and that is the signal ``complete_stage`` refuses on.
    """
    compiled = re.compile(pattern)

    def read(log_text: str) -> int | None:
        seen = [int(m.group(1)) for m in compiled.finditer(log_text or "")]
        return max(seen) if seen else None

    read.__name__ = f"epochs_from_log({pattern!r})"
    return read


LINE_EPOCH_READER = epochs_from_log(r"\[LINE epoch (\d+)/\d+\]")
BPR_EPOCH_READER = epochs_from_log(r"\[BPR epoch (\d+)/\d+\]")


def validate_embedding_csv(spec: StageSpec, path: Path) -> dict:
    """Structural gate on a LINE embedding export: header, width, row count.

    Deliberately structural and streaming. The values themselves are the
    stage's scientific output and are not second-guessed here; what is checked
    is that the file is a complete, well-formed export of the expected shape.
    """
    expected_rows = spec.config.get("num_entity")
    expected_width = int(spec.config.get("emb_dim", 400)) + 1
    with path.open("rb") as handle:
        header = handle.readline()
    if not header.startswith(b"node_id,"):
        raise StageContractError(f"{path} does not start with the node_id header")
    width = header.count(b",") + 1
    if width != expected_width:
        raise StageContractError(
            f"{path} header has {width} fields, expected {expected_width}")
    rows = _count_lines(path) - 1
    if expected_rows is not None and rows != int(expected_rows):
        raise StageContractError(
            f"{path} holds {rows} embedding rows, expected {expected_rows}")
    return {"rows": rows, "width": width}


def validate_embedding_npy(spec: StageSpec, path: Path) -> dict:
    """Gate on a BPR embedding: loadable, 2-D, right dtype, finite, right shape."""
    array = np.load(path, mmap_mode="r")
    if array.ndim != 2:
        raise StageContractError(f"{path} is {array.ndim}-D, expected a 2-D table")
    if array.dtype != np.float32:
        raise StageContractError(f"{path} has dtype {array.dtype}, expected float32")
    expected_rows = spec.config.get("num_entity")
    if expected_rows is not None and array.shape[0] != int(expected_rows):
        raise StageContractError(
            f"{path} has {array.shape[0]} rows, expected {expected_rows}")
    expected_dim = spec.config.get("dim")
    if expected_dim is not None and array.shape[1] != int(expected_dim):
        raise StageContractError(
            f"{path} has width {array.shape[1]}, expected {expected_dim}")
    if not np.isfinite(np.asarray(array)).all():
        raise StageContractError(f"{path} contains non-finite values")
    return {"shape": [int(x) for x in array.shape], "dtype": str(array.dtype)}


def validate_score_matrix_csv(spec: StageSpec, path: Path) -> dict:
    """Gate on a headerless score matrix: shape, finiteness and the [0, 1] domain.

    The domain check is the same property the final submission gate enforces,
    applied one stage earlier so a ranker that drifts out of range is caught
    where it happened rather than at member-build time.
    """
    import pandas as pd

    matrix = pd.read_csv(path, header=None).to_numpy(np.float64)
    rows = spec.config.get("rows")
    columns = int(spec.config.get("columns", 100))
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise StageContractError(
            f"{path} is {matrix.shape}, expected (rows, {columns})")
    if rows is not None and matrix.shape[0] != int(rows):
        raise StageContractError(
            f"{path} has {matrix.shape[0]} rows, expected {rows}")
    if not np.isfinite(matrix).all():
        raise StageContractError(f"{path} contains non-finite scores")
    lo, hi = float(matrix.min()), float(matrix.max())
    if lo < 0.0 or hi > 1.0:
        raise StageContractError(
            f"{path} has scores outside [0, 1]: range [{lo:.6g}, {hi:.6g}]")
    return {"shape": [int(x) for x in matrix.shape], "min": lo, "max": hi}


def validate_member_csv(spec: StageSpec, path: Path) -> dict:
    """Gate on a final submission member: the score-matrix gate plus row/field shape."""
    detail = validate_score_matrix_csv(spec, path)
    from strategies.shared.frozen_ops import verify_submission_file

    verify_submission_file(path, (detail["shape"][0], detail["shape"][1]))
    return detail


def validate_npz_cache(spec: StageSpec, path: Path) -> dict:
    """Gate on the dataset-2 train-feature cache.

    A truncated NPZ is the failure this exists for: ``np.savez`` wrote straight
    to the final name, so an interrupted write left a file that looked like a
    cache and failed only when a later stage tried to read a member out of it.
    Every array is opened, not merely listed, and the internal geometry is
    cross-checked against the query and entity counts the record binds.
    """
    expected_keys = tuple(spec.config.get("npz_keys", ()))
    with np.load(path, allow_pickle=False) as bundle:
        present = tuple(bundle.files)
        missing = [k for k in expected_keys if k not in present]
        if missing:
            raise StageContractError(f"{path} is missing arrays {missing}")
        arrays = {key: bundle[key] for key in present}      # forces a real read

    lens = arrays["lens"]
    queries = int(spec.config["queries"])
    if lens.shape != (queries,):
        raise StageContractError(
            f"{path}: lens is {lens.shape}, expected ({queries},)")
    for key in ("qsrc_tr", "qt_tr", "qorig_tr"):
        if arrays[key].shape != (queries,):
            raise StageContractError(
                f"{path}: {key} is {arrays[key].shape}, expected ({queries},)")
    total = int(lens.sum())
    for key in ("Xf", "yf"):
        if arrays[key].shape[0] != total:
            raise StageContractError(
                f"{path}: {key} has {arrays[key].shape[0]} rows against "
                f"lens.sum() = {total}")
    if arrays["cands_concat"].shape != (total,):
        raise StageContractError(
            f"{path}: cands_concat is {arrays['cands_concat'].shape}, expected ({total},)")
    width = int(spec.config["feature_width"])
    if arrays["Xf"].shape[1] != width:
        raise StageContractError(
            f"{path}: Xf has {arrays['Xf'].shape[1]} columns, expected {width}")
    num_entity = int(np.asarray(arrays["num_entity"]).ravel()[0])
    if num_entity != int(spec.config["num_entity"]):
        raise StageContractError(
            f"{path}: num_entity {num_entity} != {spec.config['num_entity']}")
    offsets = np.concatenate([[0], np.cumsum(lens)])
    positives = np.flatnonzero(arrays["yf"] > 0)
    if not np.array_equal(positives, offsets[1:] - 1):
        raise StageContractError(
            f"{path}: label positives do not land at the last row of every query")
    for key in ("Xf", "yf"):
        if not np.isfinite(arrays[key]).all():
            raise StageContractError(f"{path}: {key} contains non-finite values")
    return {"queries": queries, "rows": total, "feature_width": width,
            "num_entity": num_entity, "keys": sorted(present)}


def _count_lines(path: Path, *, block: int = 8 << 20) -> int:
    count = 0
    trailing_newline = True
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block)
            if not chunk:
                break
            count += chunk.count(b"\n")
            trailing_newline = chunk.endswith(b"\n")
    return count if trailing_newline else count + 1
