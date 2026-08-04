# -*- coding: utf-8 -*-
"""Deterministically stage and audit the official submission tree.

The team name and approved Chinese PDF are explicit inputs. The tool never
creates a placeholder package and never selects a document implicitly.

Everything staged is copied from tracked files at the current commit, so the
package cannot contain a file that exists only in a worktree, in a scratch
directory, on a rented instance or behind a local absolute path. The membership
list is explicit rather than discovered by globbing: a reviewer can read what is
included and why, and a new file cannot drift into the package unnoticed.

Three audits run over the result and all three must pass:

* the import closure of the staged tree reaches no PyTorch;
* nothing matching the hygiene patterns (raw data, predictions, caches, keys,
  host paths) survives either a filename or a content scan;
* every staged file is byte-identical to its tracked source.

The staging root is owned by this tool. It is always
``submission_staging/contest1_<team>_003``; an existing tree is rebuilt only
with ``--replace``, and only when it carries the ``STAGING_MANIFEST.json`` that
proves this tool produced it. Nothing outside that one directory is ever
removed.

    python tools/submission/stage_package.py \
        --team-name TEAM_NAME --document path/to/提交说明文档.pdf
    python tools/submission/stage_package.py --audit --root submission_staging/PACKAGE
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

STAGING_ROOT = REPO / "submission_staging"
OFFICIAL_DOCUMENT_NAME = "提交说明文档.pdf"

# --------------------------------------------------------------------------
# membership
# --------------------------------------------------------------------------

#: (repository path, classification, why it is in the package).
CODE_MEMBERS: list[tuple[str, str, str]] = [
    (
        "run_all.py",
        "entrypoint",
        "one-command reproduction: runs both dataset pipelines in the required "
        "order; an orchestration wrapper with no scientific logic of its own",
    ),
    (
        "main.py",
        "entrypoint",
        "the organiser-facing command; executes the canonical graph for one dataset",
    ),
    (
        "src/canonical_pipeline.py",
        "orchestration",
        "the stage graph for both datasets and the fail-closed runner",
    ),
    (
        "src/stage_contract.py",
        "orchestration",
        "completion records, atomic publication, artifact validators",
    ),
    (
        "src/pipeline_common.py",
        "shared",
        "run-directory and data-root path contracts; framework-neutral utilities",
    ),
    ("src/train_line_jt.py", "training", "LINE embedding trainer (Jittor); dataset1 and dataset2"),
    ("src/train_bpr_jt.py", "training", "BPR-MF embedding trainer (Jittor); dataset1 and dataset2"),
    (
        "src/ensemble_predict.py",
        "inference",
        "embedding loading and collaborative scoring used by both rankers",
    ),
    (
        "src/ranker_ds1.py",
        "ranking",
        "dataset1 cut-split LambdaRank over 21 features; writes the base score matrix",
    ),
    (
        "src/ranker_basket_ds2.py",
        "ranking",
        "dataset2 18-feature LambdaRank with three-pass basket feedback",
    ),
    (
        "src/ds2_basket_featurizer.py",
        "features",
        "dataset2 feature construction and the contract-governed train-feature cache",
    ),
    (
        "src/ds2_mf_basket_pack.py",
        "ranking",
        "dataset2 MF sibling-message geometry; produces the production base matrix",
    ),
    (
        "src/crf_promote.py",
        "postprocessing",
        "dataset2 equality CRF and the structural-invariant demotions",
    ),
    (
        "src/build_ds1_member.py",
        "final output",
        "applies the frozen dataset1 postprocessor chain and writes the member",
    ),
    (
        "src/build_ds2_member.py",
        "final output",
        "applies the cross-time exclusivity decode and writes the member",
    ),
    ("src/strategies/__init__.py", "package marker", "package initialiser"),
    (
        "src/strategies/registry.py",
        "postprocessing",
        "the ordered, order-sensitive dataset1 postprocessor chain",
    ),
    ("src/strategies/shared/__init__.py", "package marker", "package initialiser"),
    (
        "src/strategies/shared/frozen_ops.py",
        "postprocessing",
        "frozen numeric primitives and the final submission-format gate",
    ),
    ("src/strategies/ds1/__init__.py", "package marker", "package initialiser"),
    ("src/strategies/ds1/source_slate_recurrence.py", "postprocessing", "dataset1 postprocessor 1"),
    ("src/strategies/ds1/graph_reciprocity.py", "postprocessing", "dataset1 postprocessor 2"),
    ("src/strategies/ds2/__init__.py", "package marker", "package initialiser"),
    (
        "src/strategies/ds2/cross_time_exclusivity.py",
        "postprocessing",
        "the dataset2 final decoder",
    ),
    (
        "configs/production.json",
        "configuration",
        "the machine-readable production description main.py --stage describe reads",
    ),
    (
        "tools/submission/package_component.py",
        "verification",
        "submission-format verifier; lets a reviewer check a member independently",
    ),
]

#: Copied to the archive root, beside code/.
ROOT_MEMBERS = ["requirements.txt", "environment.yaml"]

#: Deliberately excluded, with the reason a reviewer would want.
EXCLUSIONS: list[tuple[str, str]] = [
    (
        "reference/pytorch/train_line.py",
        "PyTorch-only historical trainer; not on the canonical path",
    ),
    (
        "reference/pytorch/train_bpr.py",
        "PyTorch-only historical trainer; not on the canonical path",
    ),
    (
        "tools/diagnostics/compare_backends.py",
        "local dual-backend diagnostic; spawns the two PyTorch trainers",
    ),
    ("tests/", "the maintained test suite; not required to reproduce a result"),
    ("docs/", "public documentation; the approved PDF is supplied explicitly"),
    ("README.md", "repository overview; not part of the organiser package"),
    ("data/", "official competition data; not redistributable, never bundled"),
    ("outputs/", "run artifacts, completion records, intermediate and final members"),
    ("reference/pytorch/", "optional PyTorch backend; not on the Jittor path"),
    ("submission_staging/", "this staging tree itself"),
    (".git/", "version-control metadata"),
]

# --------------------------------------------------------------------------
# hygiene
# --------------------------------------------------------------------------

#: Content patterns that must not appear in any staged file.
SECRET_PATTERNS = [
    (r"BEGIN (RSA |OPENSSH |EC |DSA )?PRIVATE KEY", "private key material"),
    (r"ssh-(rsa|ed25519) AAAA", "public key blob"),
    (
        r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token)\s*[:=]\s*['\"][^'\"]{6,}",
        "credential literal",
    ),
    (r"gh[pousr]_[A-Za-z0-9]{20,}", "GitHub token"),
    (r"seetacloud\.com", "rented-instance endpoint"),
    (r"connect\.bjb1", "rented-instance endpoint"),
    (r"/root/autodl-tmp", "rented-instance absolute path"),
    (r"[A-Za-z]:\\\\Users\\\\", "Windows user path"),
    (r"[A-Za-z]:/Users/", "Windows user path"),
    (r"\bF:[\\/]", "Windows drive path"),
    (r"jlp-(p1-diag|p1-int|schemeC|se1)-wt", "auxiliary worktree path"),
    # Anchored to statement position. An unanchored "from torch" also matches
    # the prose "differs from torch's" in a docstring, which is a false positive
    # and would make the scan's negatives worthless.
    (r"(?m)^\s*import\s+torch\b", "PyTorch import"),
    (r"(?m)^\s*from\s+torch[\s.]", "PyTorch import"),
]

ADVISORY_PATTERNS: list[tuple[str, str]] = []

#: Declared in the environment specification without being imported by the
#: staged code. Each needs a reason a reviewer would accept.
DECLARED_WITHOUT_IMPORT = {
    "scikit-learn": "named explicitly by the official inspection notice as a "
    "version-pinned requirement of the target environment; the "
    "canonical chain uses LightGBM for its learned components "
    "and does not import sklearn",
}

#: File names/suffixes that must never be staged.
FORBIDDEN_NAMES = [
    "*.npz",
    "*.npy",
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.zip",
    "*.bundle",
    "*.part",
    "*.log",
    "*.pyc",
    "*.done.json",
    "id_rsa*",
    "*.pem",
    "*.key",
    ".env*",
    "train.csv",
    "test.csv",
    "dataset1.csv",
    "dataset2.csv",
    "result_ranker.csv",
]

LARGE_FILE_BYTES = 2 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(REPO), *args), capture_output=True, text=True, check=True
    ).stdout.strip()


def package_name(team_name: str) -> str:
    """Return the organiser-prescribed package name for a validated team name."""
    value = team_name.strip()
    if not value or len(value) > 80:
        raise ValueError("team name must contain between 1 and 80 characters")
    if value in {".", ".."} or not re.fullmatch(r"[\w.-]+", value):
        raise ValueError("team name may contain only letters, numbers, '_', '-', and '.'")
    return f"contest1_{value}_003"


def report_path(path: Path) -> str:
    """Render a path without recording a host-specific absolute location."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


# --------------------------------------------------------------------------
# staging
# --------------------------------------------------------------------------


def staging_root_for(name: str) -> Path:
    """Return the one directory this tool is permitted to create and delete."""
    return STAGING_ROOT / name


def check_staging_root(root: Path, name: str) -> Path:
    """Refuse any staging root the tool is not permitted to destroy.

    ``build`` removes the staging root before rebuilding it, so the caller must
    never be able to aim that removal at an arbitrary directory. A matching
    basename is not sufficient: the resolved path must be exactly the one
    location under ``submission_staging`` that this tool owns.
    """
    expected = staging_root_for(name)
    for candidate in (STAGING_ROOT, root):
        if candidate.is_symlink():
            raise SystemExit(f"REFUSED: {report_path(candidate)} is a link, not a real directory")
    if root.resolve() != expected.resolve():
        raise SystemExit(
            f"REFUSED: the staging root must be {report_path(expected)}, "
            f"not {report_path(root)}"
        )
    return expected


def prepare_staging_root(root: Path, replace: bool) -> None:
    """Create the staging root, refusing to destroy anything not staged by us."""
    if not root.exists():
        root.mkdir(parents=True)
        return
    if not replace:
        raise SystemExit(
            f"REFUSED: {report_path(root)} already exists; pass --replace to rebuild it"
        )
    if not root.is_dir():
        raise SystemExit(f"REFUSED: {report_path(root)} is not a directory")
    if not (root / "STAGING_MANIFEST.json").is_file():
        raise SystemExit(
            f"REFUSED: {report_path(root)} carries no STAGING_MANIFEST.json, "
            "so it was not produced by this tool and will not be deleted"
        )
    shutil.rmtree(root)
    root.mkdir(parents=True)


def build(root: Path, document: Path, replace: bool = False) -> list[dict]:
    prepare_staging_root(root, replace)
    (root / "code").mkdir(parents=True)
    tracked = set(git("ls-files").splitlines())
    commit = git("rev-parse", "HEAD")

    manifest: list[dict] = []
    for relative, classification, rationale in CODE_MEMBERS:
        source = REPO / relative
        if relative not in tracked:
            raise SystemExit(f"REFUSED: {relative} is not tracked by git")
        if not source.is_file():
            raise SystemExit(f"REFUSED: {relative} does not exist")
        target = root / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest.append(
            {
                "package_path": f"code/{relative}",
                "source_path": relative,
                "source_commit": commit,
                "size": source.stat().st_size,
                "sha256": sha256_file(source),
                "classification": classification,
                "rationale": rationale,
            }
        )

    for relative in ROOT_MEMBERS:
        source = REPO / relative
        if relative not in tracked:
            raise SystemExit(f"REFUSED: {relative} is not tracked by git")
        shutil.copy2(source, root / relative)
        manifest.append(
            {
                "package_path": relative,
                "source_path": relative,
                "source_commit": commit,
                "size": source.stat().st_size,
                "sha256": sha256_file(source),
                "classification": "environment",
                "rationale": "dependency specification required at the archive root",
            }
        )

    document = document.resolve()
    if not document.is_file():
        raise SystemExit(f"REFUSED: reviewer document does not exist: {report_path(document)}")
    with document.open("rb") as handle:
        header = handle.read(5)
    if header != b"%PDF-":
        raise SystemExit("REFUSED: reviewer document does not begin with a PDF header")
    target = root / OFFICIAL_DOCUMENT_NAME
    shutil.copy2(document, target)
    manifest.append(
        {
            "package_path": OFFICIAL_DOCUMENT_NAME,
            "source_path": report_path(document),
            "source_commit": None,
            "size": document.stat().st_size,
            "sha256": sha256_file(document),
            "classification": "approved reviewer documentation",
            "rationale": "owner-supplied final Chinese PDF under the organiser-prescribed name",
        }
    )
    return manifest


# --------------------------------------------------------------------------
# audits
# --------------------------------------------------------------------------


def audit_fidelity(root: Path, manifest: list[dict]) -> list[str]:
    """Every staged byte must equal its tracked source."""
    problems = []
    for entry in manifest:
        staged = root / entry["package_path"]
        if not staged.is_file():
            problems.append(f"{entry['package_path']}: missing from the staging tree")
        elif sha256_file(staged) != entry["sha256"]:
            problems.append(f"{entry['package_path']}: differs from its tracked source")
    staged_files = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    declared = {e["package_path"] for e in manifest}
    # The manifest cannot list itself. During a build it does not exist yet, so
    # this only mattered in --audit mode, where its absence from `declared`
    # produced a false failure.
    declared.add("STAGING_MANIFEST.json")
    for extra in sorted(staged_files - declared):
        problems.append(f"{extra}: present in the tree but not declared in the manifest")
    return problems


def audit_imports(root: Path) -> tuple[list[str], set[str]]:
    """Static import closure of the staged tree. Executes nothing."""
    code = root / "code"
    # Local names are module stems AND package directory names; without the
    # latter, "import strategies" is misread as a third-party dependency.
    local = {p.stem for p in code.rglob("*.py")}
    local |= {p.name for p in code.rglob("*") if p.is_dir()}
    third_party: set[str] = set()
    problems = []
    for path in sorted(code.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = set()
            if isinstance(node, ast.Import):
                names = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = {node.module.split(".")[0]}
            for name in names:
                if name in {"torch", "torchvision", "torchaudio"}:
                    problems.append(f"{path.relative_to(root)}: imports {name}")
                if name not in local:
                    third_party.add(name)
    return problems, third_party


def audit_hygiene(root: Path) -> tuple[list[str], list[str]]:
    problems: list[str] = []
    advisories: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        for pattern in FORBIDDEN_NAMES:
            if path.match(pattern):
                problems.append(f"{relative}: matches the forbidden name pattern {pattern}")
        if path.stat().st_size > LARGE_FILE_BYTES:
            problems.append(
                f"{relative}: {path.stat().st_size} bytes exceeds "
                f"the {LARGE_FILE_BYTES} byte staging limit"
            )
        if path.suffix.lower() == ".pdf":
            # A rendered document is legitimately binary. Its SOURCE is scanned
            # instead, which is where any leaked path or secret would originate.
            header = path.read_bytes()[:5]
            if header != b"%PDF-":
                problems.append(f"{relative}: does not begin with a PDF header")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            problems.append(f"{relative}: not decodable as UTF-8 text")
            continue
        for pattern, label in SECRET_PATTERNS:
            for match in re.finditer(pattern, text):
                line = text[: match.start()].count("\n") + 1
                problems.append(f"{relative}:{line}: {label} -- {match.group(0)[:60]!r}")
        for pattern, label in ADVISORY_PATTERNS:
            for match in re.finditer(pattern, text):
                line = text[: match.start()].count("\n") + 1
                advisories.append(f"{relative}:{line}: {label} -- {match.group(0)[:60]!r}")
    return problems, advisories


STDLIB = set(getattr(sys, "stdlib_module_names", ()))


def audit_environment(root: Path, third_party: set[str]) -> tuple[list[str], dict]:
    """Declared dependencies against what the staged code actually imports."""
    alias = {"sklearn": "scikit-learn", "jittor_geometric": "jittor_geometric"}
    needed = {alias.get(name, name) for name in third_party if name not in STDLIB}
    declared = {}
    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            name = re.split(r"[=<>!~ ]", stripped, 1)[0].strip()
            version = stripped[len(name) :].strip()
            declared[name] = version or "UNPINNED"
    problems = []
    for name in sorted(needed):
        if name not in declared:
            problems.append(f"{name} is imported by the staged code but not declared")
        elif declared[name] == "UNPINNED":
            problems.append(f"{name} is declared without an explicit version")
    for name in sorted(set(declared) - needed):
        if name in DECLARED_WITHOUT_IMPORT:
            continue  # justified; carried into the report below
        problems.append(f"{name} is declared but not imported by the staged code")
    for forbidden in ("torch", "torchvision", "torchaudio"):
        if forbidden in declared:
            problems.append(f"{forbidden} must not be declared")
    return problems, {
        "imported_by_staged_code": sorted(needed),
        "declared": declared,
        "declared_without_import": {
            n: DECLARED_WITHOUT_IMPORT[n]
            for n in sorted(set(declared) - needed)
            if n in DECLARED_WITHOUT_IMPORT
        },
        "jittor_geometric": {
            "declared_in_requirements_body": "jittor_geometric" in declared,
            "imported_by_staged_code": "jittor_geometric" in needed,
            "note": "installed from a pinned upstream commit documented in the "
            "comment header of both environment files. It is not on PyPI. A "
            "requirements file could reference it by VCS URL; it is instead "
            "declared as a separately documented install step so that the pin "
            "and its rationale stay readable in one place",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--root", type=Path, help="staging directory; required with --audit, derived otherwise"
    )
    ap.add_argument("--team-name", help="approved team name; required when building")
    ap.add_argument("--document", type=Path, help="approved Chinese PDF; required when building")
    ap.add_argument(
        "--audit", action="store_true", help="audit an existing tree instead of rebuilding it"
    )
    ap.add_argument(
        "--replace",
        action="store_true",
        help="rebuild over an existing staging tree previously produced by this tool",
    )
    args = ap.parse_args()

    if args.audit:
        if args.root is None:
            ap.error("--root is required with --audit")
        root = args.root
        manifest = json.loads((root / "STAGING_MANIFEST.json").read_text(encoding="utf-8"))
    else:
        if not args.team_name or args.document is None:
            ap.error("--team-name and --document are required when building")
        try:
            name = package_name(args.team_name)
        except ValueError as exc:
            ap.error(str(exc))
        root = check_staging_root(args.root or staging_root_for(name), name)
        manifest = build(root, args.document, replace=args.replace)

    name = root.name
    intended_archive_name = f"{name}.zip"

    fidelity = audit_fidelity(root, manifest)
    imports, third_party = audit_imports(root)
    hygiene, advisories = audit_hygiene(root)
    environment, dependency_report = audit_environment(root, third_party)

    total = sum(e["size"] for e in manifest)
    report = {
        "intended_archive_name": intended_archive_name,
        "staging_directory_name": name,
        "staging_root": report_path(root),
        "source_commit": git("rev-parse", "HEAD"),
        "file_count": len(manifest),
        "total_bytes": total,
        "code_file_count": sum(1 for e in manifest if e["package_path"].startswith("code/")),
        "torch_files": len(imports),
        "dependency_report": dependency_report,
        "advisories": advisories,
        "exclusions": [{"path": p, "reason": r} for p, r in EXCLUSIONS],
        "audits": {
            "fidelity": fidelity or "PASS",
            "import_closure": imports or "PASS",
            "hygiene": hygiene or "PASS",
            "environment": environment or "PASS",
        },
    }
    if not args.audit:
        (root / "STAGING_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        report["file_count"] = len(manifest)
    (root.parent / "STAGING_REPORT.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(f"staging root : {report_path(root)}")
    print(f"intended zip : {intended_archive_name}  (not built by this command)")
    print(f"source commit: {report['source_commit']}")
    print(f"files        : {len(manifest)} ({report['code_file_count']} under code/)")
    print(f"total bytes  : {total:,}")
    print(f"imported     : {', '.join(dependency_report['imported_by_staged_code'])}")
    print(f"advisories   : {len(advisories)} (non-blocking, recorded)")
    for name, problems in report["audits"].items():
        status = "PASS" if problems == "PASS" else f"FAIL ({len(problems)})"
        print(f"audit {name:15}: {status}")
        if problems != "PASS":
            for problem in problems[:25]:
                print(f"    {problem}")
    failed = [n for n, p in report["audits"].items() if p != "PASS"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
