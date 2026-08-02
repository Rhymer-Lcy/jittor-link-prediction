# -*- coding: utf-8 -*-
"""The canonical Jittor production graph, and the runner that executes it.

This module holds the one authoritative description of how official raw
competition data becomes a submission member, for both scenarios, and it runs
that description. ``main.py`` is a thin command-line surface over it.

Two properties are deliberate.

**Jittor only.** The graph names ``src/train_line_jt.py`` and
``src/train_bpr_jt.py`` and nothing else can be substituted. There is no
backend argument, no environment-variable switch and no fallback: the historical
PyTorch trainers are unreachable from here, which is what makes the import
closure of a canonical run torch-free. The dual-backend comparison that used to
live behind ``main.py --framework torch`` now lives in
``tools/diagnostics/compare_backends.py``, outside the official package.

**A stage is complete only when its completion record says so.** Every stage is
gated by :mod:`stage_contract`. The historical driver skipped a stage whenever
its output file existed, which during the dataset-1 clean reexecution would have
declared sixteen stale artifacts a successful run in a few seconds. Here an
output without a record, a record that does not match the current inputs, code,
configuration or output, a short epoch count or a leftover partial file all stop
the run with a diagnostic. Nothing is skipped on file existence and nothing is
deleted to make room.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pipeline_common as pc            # noqa: E402  path contract
import stage_contract as sc             # noqa: E402

DATASETS = ("dataset1", "dataset2")
BPR_SEEDS = (42, 123, 777, 2024, 31337)
DS1_CUT = 115480000
DS2_CUT = 1261958400
LINE_EPOCHS = 400
BPR_EPOCHS = 120

#: Jittor runtime settings the canonical run requires. ``conv_opt=1`` and
#: ``use_mkl=0`` are not tuning: the target image ships cuDNN 9, which merged the
#: component libraries Jittor 1.3.10 looks for by name, and without them the
#: first convolution aborts with "libcudnn_ops_infer.so not found". They are set
#: here, printed, and recorded in every completion record, rather than living in
#: an undocumented login shell.
JITTOR_RUNTIME = {"conv_opt": "1", "use_mkl": "0"}

MEMBER_SHAPES = {"dataset1": (61051, 100), "dataset2": (153420, 100)}


class PipelineError(RuntimeError):
    """The run cannot continue. Always carries the operator-facing reason."""


# --------------------------------------------------------------------------
# run context
# --------------------------------------------------------------------------

@dataclass
class RunContext:
    """Everything a run needs that is not intrinsic to the graph."""

    dataset: str
    data_root: Path = field(default_factory=lambda: REPO / "data")
    outputs_root: Path = field(default_factory=lambda: REPO / "outputs")
    log_dir: Path | None = None
    data_pack: str = "data_A"
    resume: bool = True
    ds1_member: Path | None = None
    repo: Path = REPO

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root).resolve()
        self.outputs_root = Path(self.outputs_root).resolve()
        self.log_dir = Path(self.log_dir).resolve() if self.log_dir else self.outputs_root / "_logs"
        if self.ds1_member is not None:
            self.ds1_member = Path(self.ds1_member).resolve()

    @property
    def data_dir(self) -> Path:
        return self.data_root / self.data_pack / self.dataset

    def raw(self, dataset: str) -> tuple[Path, Path]:
        base = self.data_root / self.data_pack / dataset
        return base / "train.csv", base / "test.csv"

    def path_env(self) -> dict:
        """The two variables that put every path helper on this run's roots."""
        return {"DATA_ROOT": str(self.data_root), "OUTPUTS_ROOT": str(self.outputs_root),
                "DATA_PACK": self.data_pack}

    def out(self, *parts: str) -> Path:
        return self.outputs_root.joinpath(*parts)

    def member(self, dataset: str) -> Path:
        return self.out("members", f"{dataset}.csv")


def _with_roots(ctx: RunContext, fn, *args, **kwargs):
    """Call a ``pipeline_common`` path helper under this run's roots.

    The helpers read ``OUTPUTS_ROOT`` so that a subprocess and its parent agree;
    resolving a path in-process therefore has to set the same variable, and put
    it back afterwards so one run's roots never leak into another.
    """
    saved = {k: os.environ.get(k) for k in ("DATA_ROOT", "OUTPUTS_ROOT", "DATA_PACK")}
    os.environ.update(ctx.path_env())
    try:
        return fn(*args, **kwargs)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --------------------------------------------------------------------------
# entity geometry, measured from the official inputs
# --------------------------------------------------------------------------

_GEOMETRY: dict[str, dict] = {}


def geometry(ctx: RunContext, dataset: str) -> dict:
    """Row counts the stage validators check their artifacts against.

    Measured from the official files, never assumed. Returns ``None`` values
    when the raw data is absent so that ``--plan`` still works on a checkout
    without it; a real run cannot get that far, because the inputs gate fails
    first.
    """
    key = f"{ctx.data_root}|{ctx.data_pack}|{dataset}"
    if key in _GEOMETRY:
        return _GEOMETRY[key]
    train_csv, test_csv = ctx.raw(dataset)
    result = {"train_entity": None, "full_entity": None, "test_rows": None}
    if train_csv.is_file() and test_csv.is_file():
        train = pd.read_csv(train_csv, usecols=["src", "dst"])
        test = pd.read_csv(test_csv)
        candidates = test[[c for c in test.columns if c.startswith("c") and c[1:].isdigit()]]
        train_entity = int(max(train["src"].max(), train["dst"].max())) + 1
        result = {
            "train_entity": train_entity,
            "full_entity": int(max(train_entity - 1, int(candidates.to_numpy().max()),
                                   int(test["src"].max()))) + 1,
            "test_rows": int(len(test)),
        }
    _GEOMETRY[key] = result
    return result


# --------------------------------------------------------------------------
# stage-specific validators that need more than one artifact
# --------------------------------------------------------------------------

def validate_crf_archive(spec: sc.StageSpec, path: Path) -> dict:
    """Gate on the CRF intermediate archive: both members present and readable.

    ``dataset2.csv`` is the artifact the member builder consumes, so it gets the
    full score-matrix treatment. ``dataset1.csv`` is carried through the archive
    untouched by ``crf_promote`` and is only checked for presence and size.
    """
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = {"dataset1.csv", "dataset2.csv"} - names
        if missing:
            raise sc.StageContractError(f"{path} is missing archive members {sorted(missing)}")
        payload = archive.read("dataset2.csv")
        passthrough_bytes = archive.getinfo("dataset1.csv").file_size
    import io

    matrix = pd.read_csv(io.BytesIO(payload), header=None).to_numpy(np.float64)
    rows, columns = MEMBER_SHAPES["dataset2"]
    if matrix.shape != (rows, columns):
        raise sc.StageContractError(
            f"{path}::dataset2.csv is {matrix.shape}, expected {(rows, columns)}")
    if not np.isfinite(matrix).all():
        raise sc.StageContractError(f"{path}::dataset2.csv has non-finite scores")
    lo, hi = float(matrix.min()), float(matrix.max())
    if lo < 0.0 or hi > 1.0:
        raise sc.StageContractError(
            f"{path}::dataset2.csv leaves [0, 1]: range [{lo:.6g}, {hi:.6g}]")
    if (matrix.std(axis=1) <= 0).any():
        raise sc.StageContractError(f"{path}::dataset2.csv contains a constant row")
    return {"shape": [rows, columns], "min": lo, "max": hi,
            "passthrough_dataset1_bytes": int(passthrough_bytes)}


def validate_passthrough_archive(spec: sc.StageSpec, path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["dataset1.csv"]:
            raise sc.StageContractError(
                f"{path} should hold exactly dataset1.csv, holds {archive.namelist()}")
        size = archive.getinfo("dataset1.csv").file_size
    if size <= 0:
        raise sc.StageContractError(f"{path} carries an empty dataset1.csv")
    return {"dataset1_bytes": int(size)}


# --------------------------------------------------------------------------
# the graph
# --------------------------------------------------------------------------

def _code(*names: str) -> list[Path]:
    return [SRC / name for name in names]


TRAINER_CODE = _code("train_line_jt.py", "pipeline_common.py", "stage_contract.py")
BPR_CODE = _code("train_bpr_jt.py", "pipeline_common.py", "stage_contract.py")
DS1_RANKER_CODE = _code("ranker_ds1.py", "pipeline_common.py", "ensemble_predict.py",
                        "stage_contract.py")
DS1_MEMBER_CODE = _code("build_ds1_member.py", "strategies/registry.py",
                        "strategies/ds1/source_slate_recurrence.py",
                        "strategies/ds1/test_graph_reciprocity.py",
                        "strategies/shared/frozen_ops.py")
DS2_RANKER_CODE = _code("ranker_basket_ds2.py", "pipeline_common.py",
                        "ensemble_predict.py", "stage_contract.py")
DS2_PACK_CODE = _code("ds2_mf_basket_pack.py", "ds2_basket_featurizer.py",
                      "pipeline_common.py", "ensemble_predict.py", "stage_contract.py")
DS2_CRF_CODE = _code("crf_promote.py")
DS2_MEMBER_CODE = _code("build_ds2_member.py", "strategies/ds2/cross_time_exclusivity.py",
                        "strategies/shared/frozen_ops.py")


def _line_stage(ctx: RunContext, dataset: str, *, time_max: int | None) -> sc.StageSpec:
    geo = geometry(ctx, dataset)
    directory = _with_roots(ctx, pc.line_run_dir, dataset,
                            time_max=float(time_max or 0))
    env = {"DATASET": dataset}
    if time_max:
        env["LINE_TIME_MAX"] = str(time_max)
    return sc.StageSpec(
        stage_id=f"ds{dataset[-1]}_line_{'cut' if time_max else 'full'}",
        dataset=dataset,
        artifact_kind="line_embedding_csv",
        command=[sys.executable, "src/train_line_jt.py"],
        env=env,
        output=directory / "line_latest_emb.csv",
        inputs=[ctx.raw(dataset)[0]],
        config={"model": "LINE", "epochs": LINE_EPOCHS, "emb_dim": 400, "neg_ratio": 5,
                "neg_dist": "uniform", "time_max": time_max or 0, "virtual_edges": False,
                "num_entity": geo["train_entity"]},
        seed=42,
        requested_units=LINE_EPOCHS,
        unit_name="epochs",
        achieved_units_from_log=sc.LINE_EPOCH_READER,
        code_files=TRAINER_CODE,
        validator=sc.validate_embedding_csv,
        description=f"{dataset} LINE embedding"
                    + (f", edges up to t={time_max}" if time_max else ", full train"),
    )


def _bpr_stage(ctx: RunContext, dataset: str, *, seed: int, tau_frac: float,
               time_max: int | None, innov: bool) -> sc.StageSpec:
    geo = geometry(ctx, dataset)
    directory = _with_roots(ctx, pc.bpr_run_dir, dataset, seed=seed, tau_frac=tau_frac,
                            time_max=float(time_max or 0), innov=innov)
    env = {"DATASET": dataset, "SEED": str(seed)}
    if tau_frac:
        env["BPR_TAU_FRAC"] = f"{tau_frac:g}"
    if time_max:
        env["BPR_TIME_MAX"] = str(time_max)
    if innov:
        env["BPR_INNOV"] = "1"
    role = "innov" if innov else ("cut" if time_max else "serve")
    if innov and time_max:
        role = "innov_cut"
    elif innov:
        role = "innov_serve"
    return sc.StageSpec(
        stage_id=f"ds{dataset[-1]}_bpr_{role}" + (f"_s{seed}" if not innov else ""),
        dataset=dataset,
        artifact_kind="bpr_embedding_npy",
        command=[sys.executable, "src/train_bpr_jt.py"],
        env=env,
        output=directory / "bpr_emb.npy",
        inputs=[ctx.raw(dataset)[0]],
        config={"model": "BPR-MF", "epochs": BPR_EPOCHS, "dim": 256,
                "tau_frac": tau_frac, "time_max": time_max or 0, "innovation_only": innov,
                "num_entity": geo["train_entity"]},
        seed=seed,
        requested_units=BPR_EPOCHS,
        unit_name="epochs",
        achieved_units_from_log=sc.BPR_EPOCH_READER,
        code_files=BPR_CODE,
        validator=sc.validate_embedding_npy,
        description=f"{dataset} BPR-MF embedding ({role}, seed {seed})",
    )


def dataset1_stages(ctx: RunContext) -> list[sc.StageSpec]:
    geo = geometry(ctx, "dataset1")
    train_csv, test_csv = ctx.raw("dataset1")
    stages = [
        _line_stage(ctx, "dataset1", time_max=None),
        _line_stage(ctx, "dataset1", time_max=DS1_CUT),
    ]
    for seed in BPR_SEEDS:
        stages.append(_bpr_stage(ctx, "dataset1", seed=seed, tau_frac=0.25,
                                 time_max=None, innov=False))
        stages.append(_bpr_stage(ctx, "dataset1", seed=seed, tau_frac=0.25,
                                 time_max=DS1_CUT, innov=False))
    stages.append(_bpr_stage(ctx, "dataset1", seed=42, tau_frac=0.25,
                             time_max=None, innov=True))
    stages.append(_bpr_stage(ctx, "dataset1", seed=42, tau_frac=0.25,
                             time_max=DS1_CUT, innov=True))

    ranker_out = _with_roots(ctx, pc.ranker_dir, "dataset1") / "result_ranker.csv"
    stages.append(sc.StageSpec(
        stage_id="ds1_ranker",
        dataset="dataset1",
        artifact_kind="score_matrix_csv",
        command=[sys.executable, "src/ranker_ds1.py"],
        env={"DATASET": "dataset1"},
        output=ranker_out,
        inputs=[train_csv, test_csv] + [s.output for s in stages],
        config={"model": "LGBMRanker", "objective": "lambdarank", "n_estimators": 400,
                "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
                "random_state": 42, "features": 21, "cut": DS1_CUT,
                "negatives": 99, "sampling_seed": 20260727,
                "serialisation": "per_row_min_max",
                "rows": geo["test_rows"], "columns": 100},
        seed=42,
        code_files=DS1_RANKER_CODE,
        validator=sc.validate_score_matrix_csv,
        description="dataset1 cut-split LambdaRank score matrix",
    ))

    member_out = ctx.member("dataset1")
    stages.append(sc.StageSpec(
        stage_id="ds1_member",
        dataset="dataset1",
        artifact_kind="submission_member_csv",
        command=[sys.executable, "src/build_ds1_member.py",
                 "--control", str(ranker_out), "--test", str(test_csv),
                 "--train", str(train_csv), "--output", "<staged output>"],
        env={},
        output=member_out,
        inputs=[ranker_out, train_csv, test_csv],
        config={"chain": ["ds1_source_slate_recurrence", "ds1_test_graph_reciprocity"],
                "order_sensitive": True, "rows": geo["test_rows"], "columns": 100},
        code_files=DS1_MEMBER_CODE,
        validator=sc.validate_member_csv,
        description="dataset1 submission member (frozen postprocessor chain)",
    ))
    return stages


def dataset2_stages(ctx: RunContext) -> list[sc.StageSpec]:
    geo = geometry(ctx, "dataset2")
    train_csv, test_csv = ctx.raw("dataset2")
    stages = [
        _line_stage(ctx, "dataset2", time_max=None),
        _line_stage(ctx, "dataset2", time_max=DS2_CUT),
    ]
    for seed in BPR_SEEDS:
        stages.append(_bpr_stage(ctx, "dataset2", seed=seed, tau_frac=0.0,
                                 time_max=None, innov=False))
        stages.append(_bpr_stage(ctx, "dataset2", seed=seed, tau_frac=0.0,
                                 time_max=DS2_CUT, innov=False))

    ranker_dir = _with_roots(ctx, pc.ranker_dir, "dataset2")
    ranker_out = ranker_dir / "ranker_basket3_dataset2.csv"
    embeddings = [s.output for s in stages]
    stages.append(sc.StageSpec(
        stage_id="ds2_ranker",
        dataset="dataset2",
        artifact_kind="score_matrix_csv",
        command=[sys.executable, "src/ranker_basket_ds2.py"],
        env={"DATASET": "dataset2"},
        output=ranker_out,
        inputs=[train_csv, test_csv] + embeddings,
        config={"model": "LGBMRanker", "objective": "lambdarank", "features": 18,
                "passes": 3, "cut": DS2_CUT, "seeds": list(BPR_SEEDS),
                "rows": geo["test_rows"], "columns": 100},
        seed=42,
        code_files=DS2_RANKER_CODE,
        validator=sc.validate_score_matrix_csv,
        description="dataset2 three-pass basket-feedback LambdaRank score matrix",
    ))

    pack_out = ranker_dir / "mf_basket3_dataset2.csv"
    stages.append(sc.StageSpec(
        stage_id="ds2_mf_pack",
        dataset="dataset2",
        artifact_kind="score_matrix_csv",
        command=[sys.executable, "src/ds2_mf_basket_pack.py", "--geom", "MF",
                 "--out", "<staged output>"],
        env={"DATASET": "dataset2"},
        output=pack_out,
        inputs=[train_csv, test_csv] + embeddings,
        config={"geometry": "MF", "svd_rank": 128, "passes": 3, "cut": DS2_CUT,
                "rows": geo["test_rows"], "columns": 100},
        seed=42,
        code_files=DS2_PACK_CODE,
        validator=sc.validate_score_matrix_csv,
        description="dataset2 production base matrix (MF sibling-message geometry)",
    ))

    crf_dir = ctx.out("dataset2-crf")
    passthrough = crf_dir / "ds1_passthrough.zip"
    stages.append(sc.StageSpec(
        stage_id="ds2_ds1_passthrough",
        dataset="dataset2",
        artifact_kind="passthrough_archive",
        command=["<in-process>", "package the dataset1 member crf_promote copies through"],
        env={},
        output=passthrough,
        inputs=[],                      # resolved at run time; see _passthrough_source
        config={"member": "dataset1.csv", "role": "carried through untouched"},
        code_files=_code("canonical_pipeline.py"),
        validator=validate_passthrough_archive,
        action=lambda destination, ctx=ctx: _write_passthrough(ctx, destination),
        description="archive supplying the dataset1 bytes crf_promote requires",
    ))

    crf_out = crf_dir / "ds2_crf_intermediate.zip"
    stages.append(sc.StageSpec(
        stage_id="ds2_crf",
        dataset="dataset2",
        artifact_kind="crf_intermediate_archive",
        command=[sys.executable, "src/crf_promote.py", "--base", str(pack_out),
                 "--ds1-from", str(passthrough), "--out", "<staged output>",
                 "--tau", "0.20", "--B", "70", "--W", "1",
                 "--zr-exclude", "--demote-dups", "--st-exclude", "--no-pair",
                 "--data", str(ctx.data_root / ctx.data_pack / "dataset2")],
        env={},
        output=crf_out,
        inputs=[pack_out, passthrough, train_csv, test_csv],
        config={"tau": 0.20, "B": 70.0, "W": 1, "p": 1.0, "eta": 0.0,
                "triple": True, "pair": False, "zr_exclude": True,
                "demote_dups": True, "st_exclude": True},
        code_files=DS2_CRF_CODE,
        validator=validate_crf_archive,
        description="dataset2 equality-CRF promotion and invariant demotions",
    ))

    member_out = ctx.member("dataset2")
    stages.append(sc.StageSpec(
        stage_id="ds2_member",
        dataset="dataset2",
        artifact_kind="submission_member_csv",
        command=[sys.executable, "src/build_ds2_member.py", "--base-zip", str(crf_out),
                 "--base-member", "dataset2.csv", "--test", str(test_csv),
                 "--output", "<staged output>"],
        env={},
        output=member_out,
        inputs=[crf_out, test_csv],
        config={"decoder": "xte_cross_time_exclusivity_decode",
                "serialisation": "byte_preserving_token_swap",
                "rows": geo["test_rows"], "columns": 100},
        code_files=DS2_MEMBER_CODE,
        validator=sc.validate_member_csv,
        description="dataset2 submission member (cross-time exclusivity decode)",
    ))
    return stages


def build_stages(ctx: RunContext) -> list[sc.StageSpec]:
    if ctx.dataset == "dataset1":
        return dataset1_stages(ctx)
    if ctx.dataset == "dataset2":
        return dataset2_stages(ctx)
    raise PipelineError(f"unknown dataset {ctx.dataset!r}; expected one of {DATASETS}")


# --------------------------------------------------------------------------
# the dataset1 passthrough archive
# --------------------------------------------------------------------------

def _passthrough_source(ctx: RunContext) -> Path:
    """The dataset1 member whose bytes ``crf_promote`` carries through.

    ``crf_promote`` writes a two-member archive, so it needs dataset1 bytes even
    when only dataset2 is being produced. They are never read as scores: the
    dataset2 member builder extracts ``dataset2.csv`` alone. Even so, the source
    must be a member this repository produced under a validated completion
    record -- a frozen accepted member is not an acceptable input to a canonical
    run, and no member is fabricated to stand in for one.
    """
    candidate = ctx.ds1_member or ctx.member("dataset1")
    record = sc.record_path(candidate)
    if not candidate.is_file() or not record.is_file():
        raise PipelineError(
            f"the dataset2 CRF stage needs dataset1 member bytes to package, and\n"
            f"  {candidate}\n"
            f"is not a completed artifact of this run. Run `--dataset dataset1` first, or\n"
            f"pass --ds1-member with a member that carries a valid completion record.\n"
            f"No placeholder is generated and no frozen member is substituted.")
    return candidate


def _write_passthrough(ctx: RunContext, destination: Path) -> None:
    source = _passthrough_source(ctx)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("dataset1.csv", source.read_bytes())


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

def resolve_inputs(spec: sc.StageSpec, ctx: RunContext) -> sc.StageSpec:
    """Fill in inputs that are only known once earlier stages have run."""
    if spec.stage_id == "ds2_ds1_passthrough":
        from dataclasses import replace

        return replace(spec, inputs=[_passthrough_source(ctx)])
    return spec


def stage_environment(spec: sc.StageSpec, ctx: RunContext) -> dict:
    """The child environment: inherited, plus this run's roots and Jittor settings.

    A conflicting inherited value is a stop, not an override. Silently winning
    over an operator's explicit ``use_mkl=1`` would be exactly the kind of hidden
    behaviour this entrypoint exists to remove.
    """
    env = dict(os.environ)
    for key, value in JITTOR_RUNTIME.items():
        # Case-insensitive, because Windows folds environment names and the guard
        # has to see an operator's setting on either platform.
        variants = [name for name in env if name.lower() == key.lower()]
        for name in variants:
            if env[name] != value:
                raise PipelineError(
                    f"the environment sets {name}={env[name]!r} but the canonical "
                    f"Jittor runtime requires {key}={value!r}. Unset it or correct it; "
                    f"this entrypoint will not silently override an explicit setting.")
            del env[name]
        env[key] = value
    env.update(ctx.path_env())
    env.update({k: str(v) for k, v in spec.env.items()})
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_subprocess(command: list[str], env: dict, cwd: Path, log_path: Path,
                    echo: bool) -> tuple[int, str]:
    """Run a stage, streaming its output to both the log file and the console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    captured: list[str] = []
    with log_path.open("w", encoding="utf-8", errors="replace", newline="\n") as handle:
        process = subprocess.Popen(command, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors="replace",
                                   bufsize=1)
        for line in process.stdout:                      # type: ignore[union-attr]
            handle.write(line)
            captured.append(line)
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
        code = process.wait()
    return code, "".join(captured)


def run_stage(spec: sc.StageSpec, ctx: RunContext, *, echo: bool = True) -> str:
    """Execute one stage under the completion contract. Returns its disposition.

    Order, and it is load-bearing:

    1. decide, from the completion record alone, whether the stage may be reused;
    2. run it, writing into an isolated staged path wherever the stage takes its
       output as an argument;
    3. validate the staged artifact;
    4. publish the artifact atomically;
    5. publish the completion record atomically, last.

    A failure at any point leaves no completion record, so the stage remains
    incomplete and the next run refuses to treat its leftovers as finished work.
    """
    spec = resolve_inputs(spec, ctx)
    decision = sc.evaluate(spec, repo=ctx.repo)
    if decision.action == "REUSE":
        if not ctx.resume:
            raise PipelineError(
                f"{spec.stage_id} is already complete but this is a --fresh run. "
                f"Quarantine {spec.output} and its record deliberately, or re-run "
                f"with --resume.")
        print(f"  REUSE {spec.stage_id}: {decision.detail}")
        return "REUSE"
    if decision.action == "STOP":
        detail = decision.detail
        if decision.mismatches:
            detail += "\n" + "\n".join(
                f"      {m['property']}: recorded {m['recorded']!r}, current {m['current']!r}"
                for m in decision.mismatches)
        raise PipelineError(f"{spec.stage_id} [{decision.code}]\n    {detail}")

    missing = [str(p) for p in spec.inputs if not Path(p).is_file()]
    if missing:
        raise PipelineError(
            f"{spec.stage_id} cannot run: required inputs are absent: {missing}")

    staged = sc.part_path(spec.output)
    spec.output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(part).replace("<staged output>", str(staged)) for part in spec.command]
    started = sc.utc_now()
    clock = time.time()
    print(f"  START {spec.stage_id}: {spec.description}")

    if spec.action is not None:
        log_text = ""
        try:
            spec.action(staged)
        except Exception as problem:                      # noqa: BLE001 - reported
            raise PipelineError(f"{spec.stage_id} failed: {problem}") from problem
        code = 0
    else:
        env = stage_environment(spec, ctx)
        log_path = ctx.log_dir / f"{spec.stage_id}.log"
        code, log_text = _run_subprocess(command, env, ctx.repo, log_path, echo)
        if code != 0:
            tail = "".join(log_text.splitlines(keepends=True)[-15:])
            raise PipelineError(
                f"{spec.stage_id} exited {code} after {time.time() - clock:.0f}s; "
                f"no completion record was written.\n"
                f"    log: {log_path}\n{tail}")

    produced = staged if staged.exists() else spec.output
    if not produced.is_file():
        raise PipelineError(
            f"{spec.stage_id} exited 0 but produced neither {staged} nor {spec.output}")

    # Validate the artifact BEFORE it takes the final name, wherever the stage
    # let us stage it. A stage that publishes its own output atomically (the
    # trainers, the rankers) is validated in place, one step later; the record
    # is still written only after the validation passes.
    if produced != spec.output:
        from dataclasses import replace

        if spec.validator is not None:
            spec.validator(replace(spec, output=produced), produced)
        os.replace(produced, spec.output)

    record = sc.complete_stage(spec, repo=ctx.repo, exit_code=code, started_at=started,
                               completed_at=sc.utc_now(), log_text=log_text)
    print(f"  OK    {spec.stage_id}  {time.time() - clock:.0f}s  "
          f"sha256 {record['output']['sha256'][:16]}")
    return "RAN"


def plan(ctx: RunContext) -> str:
    """The stage graph as text, with each stage's current disposition."""
    stages = build_stages(ctx)
    lines = [f"canonical Jittor pipeline: {ctx.dataset}",
             f"  repository   : {ctx.repo}",
             f"  data root    : {ctx.data_root} (pack {ctx.data_pack})",
             f"  outputs root : {ctx.outputs_root}",
             f"  log directory: {ctx.log_dir}",
             f"  jittor runtime: " + " ".join(f"{k}={v}" for k, v in JITTOR_RUNTIME.items()),
             f"  mode         : {'validated resume' if ctx.resume else 'fresh'}",
             f"  stages       : {len(stages)}", ""]
    for index, spec in enumerate(stages, start=1):
        try:
            resolved = resolve_inputs(spec, ctx)
            decision = sc.evaluate(resolved, repo=ctx.repo)
            status = f"{decision.action} ({decision.code})"
        except Exception as problem:                      # noqa: BLE001 - shown, not raised
            status = f"UNRESOLVED ({type(problem).__name__})"
        lines.append(f"[{index:2d}] {spec.stage_id}")
        lines.append(f"     {spec.description}")
        lines.append(f"     output : {spec.output}")
        lines.append(f"     status : {status}")
    return "\n".join(lines)


def execute(ctx: RunContext, *, echo: bool = True) -> int:
    """Run the whole graph in order. Returns a process exit code."""
    stages = build_stages(ctx)
    train_csv, test_csv = ctx.raw(ctx.dataset)
    absent = [str(p) for p in (train_csv, test_csv) if not p.is_file()]
    if absent:
        print(f"FAIL official raw data is absent: {absent}", file=sys.stderr)
        print("     the competition data is not redistributable and is not tracked; "
              "point --data-root at it.", file=sys.stderr)
        return 2

    ctx.log_dir.mkdir(parents=True, exist_ok=True)
    print(f"===== CANONICAL RUN: {ctx.dataset} =====")
    print(f"commit  {sc.git_commit(ctx.repo)}")
    print(f"backend jittor (fixed; this entrypoint cannot select another)")
    print(f"runtime " + " ".join(f"{k}={v}" for k, v in JITTOR_RUNTIME.items()))
    print(f"stages  {len(stages)}")
    clock = time.time()
    for index, spec in enumerate(stages, start=1):
        print(f"[{index}/{len(stages)}] {spec.stage_id}")
        try:
            run_stage(spec, ctx, echo=echo)
        except (PipelineError, sc.StageContractError) as problem:
            print(f"\nFAIL {problem}", file=sys.stderr)
            print(f"\n===== RUN ABORTED after {time.time() - clock:.0f}s =====",
                  file=sys.stderr)
            return 1

    member = ctx.member(ctx.dataset)
    print(f"\n===== {ctx.dataset} COMPLETE in {time.time() - clock:.0f}s =====")
    print(f"member  {member}")
    print(f"sha256  {sc.sha256_file(member)}")
    print(f"bytes   {member.stat().st_size}")
    return 0
