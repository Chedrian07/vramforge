"""Static checks of the Dockerfiles and build contexts (no Docker needed).

Recipe and reasons: docs/research/stack-compat.md §5.1 (no parity group in images), §7.5 (no
"# syntax=" pull), §10.7 (non-root user, /data ownership, non-editable members), §1.5 and W3
(Next.js standalone: HOSTNAME=0.0.0.0, static assets copied).
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKER = ROOT / "infra" / "docker"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def instructions(dockerfile: str) -> list[str]:
    """Logical instructions: comments dropped, backslash continuations joined."""
    lines = [line for line in dockerfile.splitlines() if not line.lstrip().startswith("#")]
    joined = re.sub(r"\\\n", " ", "\n".join(lines))
    return [" ".join(line.split()) for line in joined.splitlines() if line.strip()]


def workspace_members() -> list[str]:
    data = tomllib.loads(read(ROOT / "pyproject.toml"))
    members: list[str] = data["tool"]["uv"]["workspace"]["members"]
    return members


def test_python_image_installs_only_the_runtime_packages() -> None:
    syncs = [i for i in instructions(read(DOCKER / "python.Dockerfile")) if "uv sync" in i]
    assert len(syncs) == 2
    for sync in syncs:
        # --no-default-groups keeps the "dev" and "parity" groups (torch, trl, peft) out.
        for flag in ("--frozen", "--no-dev", "--no-default-groups", '--package "${PACKAGE}"'):
            assert flag in sync, (flag, sync)
    assert "--no-install-workspace" in syncs[0]
    assert "--no-editable" in syncs[1]
    assert "ARG PACKAGE=vramforge-worker" in instructions(read(DOCKER / "python.Dockerfile"))


def test_python_image_mounts_every_workspace_member_manifest() -> None:
    """The dependency layer binds each member's pyproject.toml; a new member must be added."""
    dockerfile = read(DOCKER / "python.Dockerfile")
    for member in workspace_members():
        manifest = f"{member}/pyproject.toml"
        assert f"source={manifest},target={manifest}" in dockerfile, member
        assert f"COPY {member.split('/')[0]} " in dockerfile, member


def test_python_image_runs_as_the_app_user_with_owned_data_dirs() -> None:
    steps = instructions(read(DOCKER / "python.Dockerfile"))
    setup = next(step for step in steps if "useradd" in step)
    assert "--uid 10001" in setup
    assert "mkdir -p /data/artifacts /data/uploads /data/hf" in setup
    assert "chown -R app:app /data" in setup
    assert steps.index(setup) < steps.index("USER 10001:10001")
    env = " ".join(step for step in steps if step.startswith("ENV "))
    for setting in (
        "HF_HOME=/data/hf",
        "VRAMFORGE_DATA_DIR=/data",
        "VRAMFORGE_PROFILES_DIR=/app/profiles",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS=1",
        "PYTHONUNBUFFERED=1",
    ):
        assert setting in env, setting
    assert steps[-1] == 'CMD ["vramforge-api"]'


def test_web_image_is_a_non_root_standalone_server() -> None:
    steps = instructions(read(DOCKER / "web.Dockerfile"))
    runner = steps[next(i for i, s in enumerate(steps) if s.endswith(" AS runner")) :]
    env = " ".join(step for step in runner if step.startswith("ENV "))
    for setting in ("HOSTNAME=0.0.0.0", "PORT=3000", "NEXT_TELEMETRY_DISABLED=1"):
        assert setting in env, setting
    copies = [step for step in runner if step.startswith("COPY ")]
    assert any(step.endswith(" ./.next/static") for step in copies)
    assert any(step.endswith(" ./public") for step in copies)
    assert "USER node" in runner
    assert any("pnpm install --frozen-lockfile" in step for step in steps)
    assert any("corepack" in step for step in steps)


def test_dockerfiles_do_not_pull_a_syntax_frontend() -> None:
    for path in DOCKER.glob("*.Dockerfile"):
        assert not read(path).lstrip().lower().startswith("# syntax="), path.name


def test_python_build_context_is_an_allowlist() -> None:
    rules = [
        line.strip()
        for line in read(ROOT / ".dockerignore").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert rules[0] == "*"
    allowed = {rule[1:].rstrip("/") for rule in rules if rule.startswith("!")}
    tops = {member.split("/")[0] for member in workspace_members()}
    assert allowed == {"pyproject.toml", "uv.lock", "profiles", *tops}
    for secret in ("**/.env", "**/.env.*"):
        assert secret in rules


def test_web_build_context_ignores_host_outputs() -> None:
    rules = set(read(DOCKER / "web.Dockerfile.dockerignore").split())
    assert {"node_modules", ".next", ".env", ".env.*"} <= rules
