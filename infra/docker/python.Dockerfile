# VRAMForge Python image: one image for the api, worker and migrate services.
#
#   api      vramforge-api              (default CMD, uvicorn on :8000)
#   worker   vramforge-worker           (healthcheck: vramforge-worker healthcheck)
#   migrate  vramforge-api migrate      (one-shot: alembic upgrade head)
#
# Build context: repository root (the root .dockerignore is an allowlist).
# Recipe: docs/research/stack-compat.md §10.7 (verified E11, E17, E21).
# There is deliberately no "# syntax=" line: the BuildKit built-in frontend handles RUN --mount
# and this avoids one docker.io pull per build (stack-compat.md §7.5, D2).

ARG PYTHON_IMAGE=python:3.12.15-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.23

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# The worker package pulls vramforge-api and vramforge-estimator[analysis]; never torch/trl/peft
# (the "parity" group is not a default group, stack-compat.md §5.1 E10).
ARG PACKAGE=vramforge-worker

# 1) Third-party dependencies only; this layer is reused until uv.lock or a pyproject changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/estimator/pyproject.toml,target=packages/estimator/pyproject.toml \
    --mount=type=bind,source=services/api/pyproject.toml,target=services/api/pyproject.toml \
    --mount=type=bind,source=services/worker_cpu/pyproject.toml,target=services/worker_cpu/pyproject.toml \
    uv sync --frozen --no-dev --no-default-groups --no-install-workspace --package "${PACKAGE}"

# 2) Workspace members as regular wheels (non-editable), so the runtime needs no /src.
COPY pyproject.toml uv.lock ./
COPY packages packages
COPY services services
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-default-groups --no-editable --package "${PACKAGE}"

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="vramforge-python" \
      org.opencontainers.image.description="VRAMForge API, CPU worker and migrations" \
      org.opencontainers.image.source="https://github.com/Chedrian07/vramforge" \
      org.opencontainers.image.licenses="NOASSERTION"

# Named volumes inherit ownership from the image on first mount, so /data must exist and belong
# to the app user before switching to it; otherwise uid 10001 gets "Permission denied"
# (stack-compat.md §10.7, E18).
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --create-home --home-dir /home/app \
        --shell /usr/sbin/nologin app \
 && mkdir -p /data/artifacts /data/uploads /data/hf /app/profiles \
 && chown -R app:app /data

COPY --from=build /opt/venv /opt/venv
# Backend profile registry, root-owned and read-only for the app. The [s] glob copies the
# contents of profiles/ and still builds while that directory does not exist (BuildKit allows
# a COPY wildcard that matches nothing); /app/profiles is created above either way.
COPY profile[s] /app/profiles/

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    HF_HOME=/data/hf \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VRAMFORGE_DATA_DIR=/data \
    VRAMFORGE_PROFILES_DIR=/app/profiles

WORKDIR /app
# Numeric uid:gid so orchestrators can verify runAsNonRoot without reading /etc/passwd.
USER 10001:10001
EXPOSE 8000
CMD ["vramforge-api"]
