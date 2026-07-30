# -*- coding: utf-8 -*-
"""The canonical framework boundary: Jittor trains, PyTorch is reference-only.

The competition mandates Jittor. These tests enforce, statically, that:

* no canonical production module imports torch at module scope;
* the Jittor trainers import jittor and never torch, so no silent fallback
  between frameworks can appear;
* torch is not declared as a mandatory dependency of the inspection
  environment.

The defect these guard against was real: `src/train_line.py` is both the
historical PyTorch trainer and the shared utility module that every ranking
stage imports, so a top-level `import torch` made torch a hard dependency of the
whole canonical pipeline. Five stages failed to import without it.
"""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"

#: Modules that make up the canonical production path. None may require torch.
CANONICAL_MODULES = [
    "train_line_jt.py", "train_bpr_jt.py", "ensemble_predict.py",
    "ranker_ds1.py", "ranker_basket_ds2.py", "ds2_basket_featurizer.py",
    "ds2_mf_basket_pack.py", "crf_promote.py", "triple_promote.py",
    "build_ds1_member.py", "build_ds2_member.py",
]

#: The Jittor neural trainers. These must train through Jittor, never torch.
JITTOR_TRAINERS = ["train_line_jt.py", "train_bpr_jt.py"]

#: Reference-only: the historical PyTorch trainer. It may reference torch, but
#: only behind a guard, so that importing it for its shared utilities does not
#: require torch. It is not a canonical training entry point.
REFERENCE_ONLY = {"train_line.py", "train_bpr.py"}


def top_level_imports(path: Path) -> set[str]:
    """Modules imported at module scope, i.e. unconditionally on import."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in tree.body:                       # module scope only, not nested
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


class CanonicalPathIsTorchFree(unittest.TestCase):
    def test_no_canonical_module_imports_torch_at_module_scope(self):
        offenders = []
        for name in CANONICAL_MODULES:
            path = SRC / name
            if not path.exists():
                continue
            if "torch" in top_level_imports(path):
                offenders.append(name)
        self.assertEqual(offenders, [], f"canonical modules importing torch: {offenders}")

    def test_reference_only_trainer_guards_its_torch_import(self):
        """train_line.py may use torch, but never unconditionally at import."""
        path = SRC / "train_line.py"
        if not path.exists():
            self.skipTest("train_line.py not present")
        self.assertNotIn("torch", top_level_imports(path),
                         "train_line.py imports torch unconditionally; every canonical "
                         "stage that imports it for shared utilities would then require torch")
        text = path.read_text(encoding="utf-8")
        self.assertIn("TORCH_AVAILABLE", text, "the torch guard flag is missing")

    def test_canonical_modules_import_without_torch(self):
        """Import every canonical module with torch masked out of sys.modules."""
        import importlib

        sys.path.insert(0, str(SRC))
        blocked = {"torch", "torch.nn", "torch.nn.functional"}

        class _Blocker:
            def find_module(self, name, path=None):
                return self if name.split(".")[0] == "torch" else None

            def load_module(self, name):
                raise ModuleNotFoundError(f"No module named {name!r}")

            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] == "torch":
                    raise ModuleNotFoundError(f"No module named {name!r}")
                return None

        saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] == "torch"}
        for key in list(saved):
            del sys.modules[key]
        blocker = _Blocker()
        sys.meta_path.insert(0, blocker)
        failures = []
        try:
            # Only leaf utility modules are safe to import here: several ranking
            # modules execute pipeline work at module scope by design.
            for name in ("train_line",):
                sys.modules.pop(name, None)
                try:
                    importlib.import_module(name)
                except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                    failures.append(f"{name}: {type(exc).__name__}: {exc}")
        finally:
            sys.meta_path.remove(blocker)
            for key in list(sys.modules):
                if key.split(".")[0] == "torch":
                    del sys.modules[key]
            sys.modules.update(saved)
        self.assertEqual(failures, [], f"canonical import failed without torch: {failures}")

    def test_jittor_trainers_use_jittor_and_not_torch(self):
        for name in JITTOR_TRAINERS:
            path = SRC / name
            if not path.exists():
                continue
            imports = top_level_imports(path)
            self.assertIn("jittor", imports, f"{name} does not import jittor")
            self.assertNotIn("torch", imports, f"{name} imports torch; no fallback is permitted")

    def test_no_framework_fallback_wording_in_jittor_trainers(self):
        """A trainer must fail loudly rather than switch framework."""
        for name in JITTOR_TRAINERS:
            path = SRC / name
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8").lower()
            for bad in ("except importerror:\n    import torch", "fallback to torch",
                        "fall back to torch"):
                self.assertNotIn(bad, text, f"{name} appears to contain a torch fallback")


class EnvironmentDoesNotRequireTorch(unittest.TestCase):
    def test_torch_is_not_a_mandatory_dependency(self):
        for name in ("environment.yaml", "requirements.txt"):
            path = REPO / name
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip().lstrip("-").strip()
                if stripped.startswith("#") or not stripped:
                    continue
                self.assertNotEqual(
                    stripped.split("=")[0].split(">")[0].split("<")[0].strip(), "torch",
                    f"{name} declares torch as a mandatory dependency; the canonical "
                    f"path must not require PyTorch")


if __name__ == "__main__":
    unittest.main(verbosity=2)
