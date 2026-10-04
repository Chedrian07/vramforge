"""HTTP-only models that are not part of the estimator contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vramforge_estimator.schemas import AnalysisRequest

ExportFormat = Literal["json", "yaml", "md", "trainer-config"]


class SessionStatus(BaseModel):
    """Whether the deployment requires an access token and whether this browser has one."""

    model_config = ConfigDict(extra="forbid")

    auth_required: bool
    authenticated: bool


class SessionLogin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token: str = Field(min_length=1, max_length=512)


class ScenarioExportRequest(BaseModel):
    """`POST /analyses/{id}/scenarios/export`: the form state of `/scenarios` plus the file format.

    The server recomputes exactly like `/scenarios` and exports that result, so the file matches
    the scenario shown on screen.
    """

    model_config = ConfigDict(extra="forbid")

    request: AnalysisRequest
    format: ExportFormat = "json"
    client_fingerprint: str | None = Field(default=None, max_length=256)


__all__ = ["ExportFormat", "ScenarioExportRequest", "SessionLogin", "SessionStatus"]
