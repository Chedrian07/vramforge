"""HTTP-only models that are not part of the estimator contract."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SessionStatus(BaseModel):
    """Whether the deployment requires an access token and whether this browser has one."""

    model_config = ConfigDict(extra="forbid")

    auth_required: bool
    authenticated: bool


class SessionLogin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=1, max_length=512)


__all__ = ["SessionLogin", "SessionStatus"]
