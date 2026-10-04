"""The analysis service is CPU-only and torch-free (plan §13.1): production modules must not import
torch, even indirectly. Runs in a subprocess so other tests' imports do not leak in."""

import subprocess
import sys

MODULES = [
    "vramforge_estimator.schemas",
    "vramforge_estimator.pipeline",
    "vramforge_estimator.sources",
    "vramforge_estimator.inspection",
    "vramforge_estimator.preprocessing",
    "vramforge_estimator.scan",
    "vramforge_estimator.batching",
    "vramforge_estimator.architectures",
    "vramforge_estimator.trainers",
    "vramforge_estimator.memory",
    "vramforge_estimator.compatibility",
    "vramforge_estimator.exports",
    "vramforge_api.app",
    "vramforge_worker.main",
    "vramforge_worker.tasks",
]


def test_production_modules_do_not_import_torch() -> None:
    code = (
        "import importlib, sys\n"
        f"for m in {MODULES!r}: importlib.import_module(m)\n"
        "bad = sorted(k for k in sys.modules if k == 'torch' or k.startswith('torch.'))\n"
        "print(','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"torch imported by production modules: {out.stdout.strip()}"
