"""Static checks of compose.yaml against its contract (no Docker needed).

Sources: plan.md §13 and §18 (localhost-only default, least-privilege secrets), docs/architecture.md
§1 (services, ports, paths), docs/research/stack-compat.md §7.3 and §10.7-10.8 (verified details).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SERVICES = {"proxy", "web", "api", "worker", "migrate", "postgres", "redis"}
LONG_RUNNING = SERVICES - {"migrate"}
PYTHON_SERVICES = ("api", "worker", "migrate")
INTERPOLATION = re.compile(r"(?<!\$)\$\{([A-Za-z_][A-Za-z0-9_]*)([^}]*)\}")


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def services(compose: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return compose["services"]


def strings(node: Any) -> list[str]:
    if isinstance(node, dict):
        return [s for value in node.values() for s in strings(value)]
    if isinstance(node, list):
        return [s for value in node for s in strings(value)]
    return [node] if isinstance(node, str) else []


def test_project_and_services(compose: dict[str, Any]) -> None:
    assert compose["name"] == "vramforge"
    assert set(compose["services"]) == SERVICES
    assert set(compose["volumes"]) == {"pgdata", "redisdata", "vfdata"}


def test_only_the_proxy_publishes_a_port_on_localhost_by_default(
    services: dict[str, dict[str, Any]],
) -> None:
    published = {name for name, svc in services.items() if svc.get("ports")}
    assert published == {"proxy"}
    assert services["proxy"]["ports"] == ["${VRAMFORGE_BIND:-127.0.0.1}:${VRAMFORGE_PORT:-8080}:80"]
    assert not any(svc.get("network_mode") == "host" for svc in services.values())


def test_every_interpolation_has_a_default(compose: dict[str, Any]) -> None:
    """Zero-.env start: "${VAR:-default}" everywhere, no required "${VAR}" or "${VAR:?}"."""
    found = [m for s in strings(compose) for m in INTERPOLATION.findall(s)]
    assert found
    missing = sorted({name for name, rest in found if not rest.startswith(":-")})
    assert missing == []


def test_health_and_restart_policies(services: dict[str, dict[str, Any]]) -> None:
    for name in LONG_RUNNING:
        assert services[name]["restart"] == "unless-stopped", name
        assert services[name]["healthcheck"]["test"], name
    assert services["migrate"]["restart"] == "no"
    assert "healthcheck" not in services["migrate"]
    assert services["worker"]["healthcheck"]["test"] == ["CMD", "vramforge-worker", "healthcheck"]
    api_check = " ".join(services["api"]["healthcheck"]["test"])
    assert "http://127.0.0.1:8000/api/v1/health" in api_check


def test_startup_order(services: dict[str, dict[str, Any]]) -> None:
    def conditions(name: str) -> dict[str, str]:
        return {dep: spec["condition"] for dep, spec in services[name]["depends_on"].items()}

    assert conditions("migrate") == {"postgres": "service_healthy"}
    for name in ("api", "worker"):
        assert conditions(name) == {
            "migrate": "service_completed_successfully",
            "postgres": "service_healthy",
            "redis": "service_healthy",
        }
    assert conditions("proxy") == {"api": "service_healthy", "web": "service_healthy"}


def test_python_services_share_one_image_definition(services: dict[str, dict[str, Any]]) -> None:
    builds = [services[name]["build"] for name in PYTHON_SERVICES]
    assert all(build == builds[0] for build in builds)
    assert builds[0]["dockerfile"] == "infra/docker/python.Dockerfile"
    assert services["api"]["command"] == ["vramforge-api"]
    assert services["worker"]["command"] == ["vramforge-worker"]
    assert services["migrate"]["command"] == ["vramforge-api", "migrate"]


def test_volumes_and_read_only_local_sources(services: dict[str, dict[str, Any]]) -> None:
    local = "${VRAMFORGE_LOCAL_SOURCES_DIR:-./local-sources}:/sources/local:ro"
    for name in ("api", "worker"):
        assert services[name]["volumes"] == ["vfdata:/data", local]
    # migrate initializes the shared data volume alone, before api and worker start (avoids the
    # concurrent copy-up race seen on Linux CI: "mkdir .../uploads: file exists").
    assert services["migrate"]["volumes"] == ["vfdata:/data"]
    for name in ("api", "worker"):
        assert (
            services[name]["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
        )
    # PostgreSQL 18 keeps PGDATA under /var/lib/postgresql/18/docker (stack-compat.md §7.3).
    assert services["postgres"]["volumes"] == ["pgdata:/var/lib/postgresql"]
    assert services["redis"]["volumes"] == ["redisdata:/data"]


def test_datastore_checks_follow_the_verified_recipe(services: dict[str, dict[str, Any]]) -> None:
    # Without -h the probe reports healthy while initdb's socket-only server runs (§7.3 V24).
    assert "pg_isready -h 127.0.0.1" in " ".join(services["postgres"]["healthcheck"]["test"])
    assert services["redis"]["command"] == ["redis-server", "--appendonly", "yes"]


def test_datastores_sit_on_the_internal_network_only(
    compose: dict[str, Any], services: dict[str, dict[str, Any]]
) -> None:
    assert compose["networks"]["backend"]["internal"] is True
    for name in ("postgres", "redis", "migrate"):
        assert services[name]["networks"] == ["backend"], name
    for name in ("api", "worker"):
        assert set(services[name]["networks"]) == {"default", "backend"}, name
    for name in ("proxy", "web"):
        assert "backend" not in services[name].get("networks", ["default"]), name


def test_secrets_reach_only_the_services_that_use_them(
    services: dict[str, dict[str, Any]],
) -> None:
    env = {name: services[name]["environment"] for name in PYTHON_SERVICES}
    assert set(env["migrate"]) == {"VRAMFORGE_DATABASE_URL", "VRAMFORGE_LOG_LEVEL"}
    assert "VRAMFORGE_ACCESS_TOKEN" not in env["worker"]
    assert "HF_TOKEN" in env["worker"]
    assert {"HF_TOKEN", "VRAMFORGE_ACCESS_TOKEN"} <= set(env["api"])
    # Value-less pass-through: never a default token, never an empty string in the container.
    for name, svc in services.items():
        for key in ("HF_TOKEN", "VRAMFORGE_ACCESS_TOKEN"):
            assert (svc.get("environment") or {}).get(key) is None, (name, key)


def test_web_keeps_the_image_hostname(services: dict[str, dict[str, Any]]) -> None:
    """The standalone server needs HOSTNAME=0.0.0.0 from the image (stack-compat.md W3)."""
    assert "HOSTNAME" not in (services["web"].get("environment") or {})
    assert "hostname" not in services["web"]


def test_proxy_hardening(services: dict[str, dict[str, Any]]) -> None:
    proxy = services["proxy"]
    assert proxy["read_only"] is True
    assert proxy["cap_drop"] == ["ALL"]
    assert proxy["cap_add"] == ["NET_BIND_SERVICE"]
    for name, svc in services.items():
        assert "no-new-privileges:true" in svc["security_opt"], name


def test_base_images_are_pinned_and_overridable(compose: dict[str, Any]) -> None:
    values = strings(compose)
    images = {
        "PYTHON_IMAGE": "python:3.12.15-slim-trixie",
        "UV_IMAGE": "ghcr.io/astral-sh/uv:0.12.23",
        "NODE_IMAGE": "node:24.21.0-trixie-slim",
        "CADDY_IMAGE": "caddy:2.11.6-alpine",
        "POSTGRES_IMAGE": "postgres:18.6-alpine",
        "REDIS_IMAGE": "redis:8.10.2-alpine",
    }
    for variable, default in images.items():
        assert f"${{{variable}:-{default}}}" in values, variable
    assert not any(":latest" in value for value in values)


def test_no_gpu_is_required_by_default(services: dict[str, dict[str, Any]]) -> None:
    for name, svc in services.items():
        assert "deploy" not in svc, name
        assert "profiles" not in svc, name
        assert "devices" not in svc, name
