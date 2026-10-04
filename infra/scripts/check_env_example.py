"""Check that .env.example documents every variable compose.yaml reads, with the same default.

- "${NAME:-default}" interpolations must appear as an active ``NAME=default`` line.
- Value-less environment entries (pass-through, e.g. ``HF_TOKEN:``) must appear, usually
  commented out (``# NAME=...``), because an empty value would still be passed to containers.
- .env.example must not mention variables that compose.yaml does not use.

Usage: python infra/scripts/check_env_example.py [compose.yaml] [.env.example]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
# "$${VAR}" is an escape for the container shell, not a compose interpolation.
INTERPOLATION = re.compile(r"(?<!\$)\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}")
ENV_LINE = re.compile(r"^(#\s*)?([A-Z][A-Z0-9_]*)=(.*)$")


def compose_variables(text: str) -> tuple[dict[str, str | None], set[str]]:
    """Return ({interpolated name: default}, {pass-through names}).

    Only parsed YAML values are scanned, so examples inside comments are ignored.
    """
    interpolated: dict[str, str | None] = {}
    passthrough: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            env = node.get("environment")
            if isinstance(env, dict):
                passthrough.update(k for k, v in env.items() if v is None)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            for name, default in INTERPOLATION.findall(node):
                previous = interpolated.get(name)
                if previous and default and previous != default:
                    raise ValueError(f"{name} has two defaults: {previous!r} and {default!r}")
                interpolated[name] = default or previous

    walk(yaml.safe_load(text))
    return interpolated, passthrough


def env_example_entries(text: str) -> dict[str, tuple[bool, str]]:
    """Return {name: (active, value)} for ``NAME=value`` and ``# NAME=value`` lines."""
    entries: dict[str, tuple[bool, str]] = {}
    for raw in text.splitlines():
        match = ENV_LINE.match(raw.strip())
        if match:
            commented, name, value = match.groups()
            entries[name] = (commented is None, value.strip())
    return entries


def check(compose_path: Path, example_path: Path) -> list[str]:
    interpolated, passthrough = compose_variables(compose_path.read_text(encoding="utf-8"))
    entries = env_example_entries(example_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for name, default in sorted(interpolated.items()):
        if name not in entries:
            errors.append(f"{name}: missing from {example_path.name}")
            continue
        active, value = entries[name]
        if default is not None and (not active or value != default):
            errors.append(f"{name}: expected active line {name}={default}, found {value!r}")
    for name in sorted(passthrough - interpolated.keys()):
        if name not in entries:
            errors.append(f"{name}: pass-through variable missing from {example_path.name}")
        elif entries[name][0] and entries[name][1] == "":
            errors.append(f"{name}: comment it out; an empty value would still be passed through")
    for name in sorted(entries.keys() - interpolated.keys() - passthrough):
        errors.append(f"{name}: documented in {example_path.name} but unused in compose.yaml")
    return errors


def main(argv: list[str]) -> int:
    compose_path = Path(argv[1]) if len(argv) > 1 else ROOT / "compose.yaml"
    example_path = Path(argv[2]) if len(argv) > 2 else ROOT / ".env.example"
    errors = check(compose_path, example_path)
    for error in errors:
        print(f"env-example: {error}", file=sys.stderr)
    if not errors:
        print(f"env-example: {example_path.name} matches {compose_path.name}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
