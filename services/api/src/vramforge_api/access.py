"""Server-side source access (credentials, roots) for one owner. Never serialized.

The server's Hugging Face token is used only when the operator shares it
(`VRAMFORGE_SHARE_SERVER_HF_TOKEN=true`, single-user deployments): plan §18 forbids letting the
service's global account stand in for a user's access. Access-denied issues get a Korean hint that
says which of the three server-token modes applies, so the operator knows what to change.
"""

from __future__ import annotations

from vramforge_estimator.schemas import AnalysisResult, ErrorCode, InspectResponse, Issue
from vramforge_estimator.sources import SourceAccess

from .settings import HfTokenMode, Settings
from .store import owner_uploads_dir

# SOURCE_ACCESS_DENIED reasons a Hugging Face token cannot change (local file permissions, a
# disabled repository, a network failure while downloading).
NON_TOKEN_REASONS = frozenset({"permission_denied", "disabled", "download_failed"})
SHARE_HINT_KEY = "server_hf_token"

ACCESS_HINTS: dict[HfTokenMode, str] = {
    "not_configured": (
        "이 서버에는 Hugging Face 토큰이 설정되어 있지 않아 비공개·gated 저장소에 접근할 수 "
        "없습니다. 혼자 쓰는 배포라면 운영자가 VRAMFORGE_HF_TOKEN(또는 HF_TOKEN)을 설정하고 "
        "VRAMFORGE_SHARE_SERVER_HF_TOKEN=true로 서버 토큰 사용을 허용할 수 있습니다."
    ),
    "configured_not_shared": (
        "서버에 설정된 Hugging Face 토큰은 사용자의 접근 권한을 대신하지 않도록 쓰지 않습니다. "
        "혼자 쓰는 배포라면 운영자가 VRAMFORGE_SHARE_SERVER_HF_TOKEN=true로 서버 토큰 사용을 "
        "허용할 수 있습니다."
    ),
    "shared": (
        "서버에 설정된 Hugging Face 토큰으로도 접근할 수 없습니다. 토큰 계정의 저장소 권한과 "
        "gated 저장소의 이용 조건 동의를 확인하세요."
    ),
}


def source_access(
    settings: Settings, owner_key: str, *, http_timeout_s: float = 30.0
) -> SourceAccess:
    """`uploads_dir` is the owner's own upload directory, so a resolver can only ever see
    uploads of the requesting owner (`upload:<id>` → `<uploads>/<owner>/<id>/`)."""
    return SourceAccess(
        hf_token=settings.source_hf_token(),
        local_roots=settings.local_root_map,
        uploads_dir=owner_uploads_dir(settings, owner_key),
        hf_home=settings.hf_home,
        allow_private_network=settings.allow_private_network,
        http_timeout_s=http_timeout_s,
    )


def _token_related(issue: Issue) -> bool:
    reason = issue.details.get("reason")
    if issue.code is ErrorCode.SOURCE_ACCESS_DENIED:
        return reason not in NON_TOKEN_REASONS
    # The Hub answers "not found" for private repositories the caller cannot see.
    return issue.code is ErrorCode.SOURCE_NOT_FOUND and reason == "repository_not_found"


def with_access_hint(issue: Issue, settings: Settings) -> Issue:
    """`issue` plus the server-token hint when a Hugging Face token could change the outcome."""
    if not _token_related(issue) or SHARE_HINT_KEY in issue.details:
        return issue
    mode = settings.hf_token_mode
    return issue.model_copy(
        update={
            "user_message": f"{issue.user_message} {ACCESS_HINTS[mode]}",
            "details": {**issue.details, SHARE_HINT_KEY: mode},
        }
    )


def hint_issues(issues: list[Issue], settings: Settings) -> list[Issue]:
    return [with_access_hint(issue, settings) for issue in issues]


def hint_result(result: AnalysisResult, settings: Settings) -> AnalysisResult:
    return result.model_copy(
        update={
            "errors": hint_issues(result.errors, settings),
            "warnings": hint_issues(result.warnings, settings),
        }
    )


def hint_inspection(response: InspectResponse, settings: Settings) -> InspectResponse:
    update = {}
    for part in ("model", "dataset"):
        found = getattr(response, part)
        if found is not None:
            update[part] = found.model_copy(update={"issues": hint_issues(found.issues, settings)})
    return response.model_copy(update=update)


__all__ = [
    "ACCESS_HINTS",
    "hint_inspection",
    "hint_issues",
    "hint_result",
    "source_access",
    "with_access_hint",
]
