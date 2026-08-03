# Optional PyTorch reference backend

The production and submission pipelines use Jittor exclusively. This directory
retains the earlier PyTorch implementations for research, comparison, and future
experimentation. No production entry point imports or selects these files.

Use a separate virtual environment. Mixing a CUDA-enabled PyTorch installation
with the pinned Jittor environment can replace CUDA libraries required by
Jittor. The reference dependency file therefore selects the PyTorch CPU wheel.

```bash
python -m venv .venv-pytorch
python -m pip install --upgrade pip
python -m pip install -r reference/pytorch/requirements.txt
python -m pip install torch==2.12.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

The PyTorch version and CPU index follow the official installation matrix. A
CUDA build may be installed in a dedicated environment by selecting the
matching official index for the host driver; it must not be installed into the
canonical Jittor environment.

Run a reference trainer explicitly:

```bash
DATASET=dataset1 python reference/pytorch/train_line.py
DATASET=dataset1 BPR_TAU_FRAC=0.25 python reference/pytorch/train_bpr.py
```

PyTorch outputs always receive the `-pytorch` directory suffix by default.
Set `PYTORCH_OUT_SUFFIX` to another safe suffix when isolating an experiment.
The suffix cannot be empty, which prevents a reference run from overwriting a
canonical Jittor artifact.

For a dry-run comparison of both framework commands:

```bash
python tools/diagnostics/compare_backends.py --dataset dataset1 --model line --dry-run
```

The implementations share objectives and parameter conventions, but framework,
optimizer, and random-number-stream differences mean that their output bytes are
not expected to match.
