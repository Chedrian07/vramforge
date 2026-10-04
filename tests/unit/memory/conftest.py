"""Make the shared test doubles in tests/unit/trainers importable (no packages under tests/)."""

import sys
from pathlib import Path

_FAKES = str(Path(__file__).resolve().parents[1] / "trainers")
if _FAKES not in sys.path:
    sys.path.insert(0, _FAKES)
