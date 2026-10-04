"""Write the API's OpenAPI document (the web app generates TypeScript types from it).

Usage: uv run python scripts/export_openapi.py [output path]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from vramforge_api.app import create_app

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "docs" / "api" / "openapi.json"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    spec = create_app().openapi()
    out.write_text(
        json.dumps(spec, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {out} ({len(spec.get('paths', {}))} paths)")


if __name__ == "__main__":
    main()
