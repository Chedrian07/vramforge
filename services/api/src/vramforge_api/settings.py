"""Runtime settings shared by the API and the worker (environment prefix ``VRAMFORGE_``).

Every default works inside the compose stack without a `.env` file (docs/architecture.md §1).
Secrets are `SecretStr` and never logged or serialized into results.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vramforge_estimator.units import GiB

_ROOT_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# How the server's Hugging Face token is used for source access (never its value).
HfTokenMode = Literal["not_configured", "configured_not_shared", "shared"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VRAMFORGE_",
        extra="ignore",
        populate_by_name=True,
        # compose passes "VAR=" through when a .env line is empty: treat it as unset.
        env_ignore_empty=True,
    )

    database_url: str = "postgresql+psycopg://vramforge:vramforge@postgres:5432/vramforge"
    redis_url: str = "redis://redis:6379/0"
    data_dir: Path = Path("/data")
    # None => the compatibility registry's own default (repo/profiles or /app/profiles).
    profiles_dir: Path | None = None
    # "name=/abs/path" entries separated by "," or ";"; referenced as "local:<name>/<rel>".
    local_roots: str = "local=/sources/local"
    hf_token: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("VRAMFORGE_HF_TOKEN", "HF_TOKEN", "hf_token")
    )
    # plan §18: the service's own Hugging Face account never stands in for a user's access. Only
    # the operator of a single-user deployment opts in to using the server token for every
    # owner's sources; otherwise it is never sent and private/gated sources are denied.
    share_server_hf_token: bool = False
    access_token: SecretStr | None = None
    cookie_secure: bool = False

    max_concurrent_jobs_per_owner: int = Field(default=2, ge=1)
    job_timeout_s: int = Field(default=21_600, ge=60)
    max_upload_bytes: int = Field(default=2 * GiB, ge=1)
    # Total size of one owner's unexpired uploads (all files together).
    max_upload_bytes_per_owner: int = Field(default=10 * GiB, ge=1)
    max_json_body_bytes: int = Field(default=1024 * 1024, ge=1024)
    retention_days: int = Field(default=7, ge=1)
    inspect_timeout_s: float = Field(default=60.0, gt=0)
    allow_private_network: bool = False

    queue_name: str = "analyses"
    # RQ job id is "<analysis_id>-a<attempt>"; the reaper re-runs expired leases up to this many
    # attempts (docs/research/stack-compat.md §6, R1/R4).
    max_attempts: int = Field(default=3, ge=1)
    lease_ttl_s: int = Field(default=60, ge=5)
    cancel_poll_s: float = Field(default=2.0, gt=0)
    cancel_grace_s: float = Field(default=20.0, ge=0)
    queued_requeue_after_s: int = Field(default=120, ge=1)
    worker_ttl_s: int = Field(default=60, ge=20)
    maintenance_interval_s: int = Field(default=60, ge=1)
    # SSE: DB polling cadence, keepalive comment interval and a hard cap per connection.
    sse_poll_interval_s: float = Field(default=0.5, gt=0)
    sse_keepalive_s: float = Field(default=15.0, gt=0)
    sse_max_duration_s: float = Field(default=6 * 3600.0, gt=0)
    # Progress events are coalesced to at most one per interval (the final one always goes out).
    progress_event_interval_s: float = Field(default=1.0, ge=0)

    api_host: str = "0.0.0.0"  # noqa: S104 - container bind; the proxy publishes on localhost
    api_port: int = 8000
    log_level: str = "INFO"

    @field_validator("local_roots")
    @classmethod
    def _check_roots(cls, value: str) -> str:
        parse_local_roots(value)
        return value

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def hf_home(self) -> Path:
        return self.data_dir / "hf"

    @property
    def local_root_map(self) -> dict[str, Path]:
        return parse_local_roots(self.local_roots)

    @property
    def retention_seconds(self) -> int:
        return self.retention_days * 86_400

    def hf_token_value(self) -> str | None:
        """The configured server token (None when unset or blank). Use `source_hf_token` for
        source access."""
        if self.hf_token is None:
            return None
        value = self.hf_token.get_secret_value().strip()
        return value or None

    def source_hf_token(self) -> str | None:
        """The token sent for source access: the server token only when it is shared."""
        return self.hf_token_value() if self.share_server_hf_token else None

    @property
    def hf_token_mode(self) -> HfTokenMode:
        if self.hf_token_value() is None:
            return "not_configured"
        return "shared" if self.share_server_hf_token else "configured_not_shared"

    def access_token_value(self) -> str | None:
        if self.access_token is None:
            return None
        value = self.access_token.get_secret_value()
        return value or None


def parse_local_roots(spec: str) -> dict[str, Path]:
    """Parse ``"name=/abs/path,other=/abs/other"`` into a name → absolute path map."""
    roots: dict[str, Path] = {}
    for raw in re.split(r"[;,]", spec or ""):
        entry = raw.strip()
        if not entry:
            continue
        name, sep, path = entry.partition("=")
        name, path = name.strip(), path.strip()
        if not sep or not _ROOT_NAME.match(name):
            raise ValueError(f"invalid local root entry (expected name=/abs/path): {name!r}")
        if not path.startswith("/"):
            raise ValueError(f"local root {name!r} must be an absolute container path")
        if name in roots:
            raise ValueError(f"duplicate local root name: {name!r}")
        roots[name] = Path(path)
    return roots


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings (read once from the environment)."""
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()


__all__ = ["HfTokenMode", "Settings", "get_settings", "parse_local_roots", "reset_settings_cache"]
