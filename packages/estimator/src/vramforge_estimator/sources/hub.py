"""Hugging Face Hub access bound to the server-configured endpoint (plan.md §6.2, §18).

Only ``huggingface_hub`` talks to the network, always against ``HF_ENDPOINT`` (server
configuration, never user input) and always at a pinned commit after resolution. The token comes
from ``SourceAccess.hf_token`` only; when it is absent ``token=False`` stops huggingface_hub from
falling back to a token saved on the server (plan §18: no global account in place of the user's).
Tokens never appear in issues, logs or manifests.

Tests replace `get_hub_client` (always looked up through this module) with a fake.
"""

from __future__ import annotations

import ipaddress
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from vramforge_estimator import __version__
from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import ErrorCode, FileEntry, Stage

from .base import SourceAccess
from .issues import blocking_error
from .references import Kind, configured_hf_endpoint, is_commit_sha

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HubRepoInfo:
    repo_id: str  # canonical id returned by the Hub (redirects resolved)
    sha: str  # immutable commit
    files: tuple[FileEntry, ...]
    private: bool | None
    gated: bool | None
    last_modified: str | None


class HubClient(Protocol):
    def repo_info(self, kind: Kind, repo_id: str, revision: str | None) -> HubRepoInfo: ...

    def check_file_access(self, kind: Kind, repo_id: str, revision: str, filename: str) -> None:
        """Raise when `filename` is not readable with the current credentials (gated repos)."""
        ...

    def download_file(self, kind: Kind, repo_id: str, revision: str, filename: str) -> Path:
        """Download one (small, metadata) file at a pinned commit into the HF cache."""
        ...

    def read_safetensors_header(self, repo_id: str, revision: str, filename: str) -> dict[str, Any]:
        """The raw header mapping of a remote ``.safetensors`` file (ranged requests only)."""
        ...


def get_hub_client(access: SourceAccess) -> HubClient:
    return HfHubClient(access)


def hub_cache_dir(access: SourceAccess) -> Path | None:
    return access.hf_home / "hub" if access.hf_home is not None else None


class HfHubClient:
    """`HubClient` backed by huggingface_hub (imported lazily: analysis extra)."""

    def __init__(self, access: SourceAccess) -> None:
        from huggingface_hub import HfApi

        self._endpoint = _checked_endpoint(access)
        self._token: str | bool = access.hf_token or False
        self._timeout = access.http_timeout_s
        self._cache_dir = hub_cache_dir(access)
        self._api = HfApi(
            endpoint=self._endpoint,
            token=self._token,
            library_name="vramforge-estimator",
            library_version=__version__,
        )

    def repo_info(self, kind: Kind, repo_id: str, revision: str | None) -> HubRepoInfo:
        fetch = self._api.model_info if kind == "model" else self._api.dataset_info
        info = fetch(
            repo_id,
            revision=revision,
            files_metadata=True,
            timeout=self._timeout,
            token=self._token,
        )
        files = sorted(
            (
                FileEntry(
                    path=sibling.rfilename,
                    size=sibling.size,
                    sha256=sibling.lfs.sha256 if sibling.lfs is not None else None,
                    blob_id=sibling.blob_id,
                )
                for sibling in info.siblings or []
            ),
            key=lambda entry: entry.path,
        )
        sha = info.sha or ""
        if not is_commit_sha(sha):
            raise metadata_error(kind, "missing_commit_sha", retryable=True)
        return HubRepoInfo(
            repo_id=info.id,
            sha=sha,
            files=tuple(files),
            private=info.private,
            gated=None if info.gated is None else bool(info.gated),
            last_modified=info.last_modified.isoformat() if info.last_modified else None,
        )

    def check_file_access(self, kind: Kind, repo_id: str, revision: str, filename: str) -> None:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        url = hf_hub_url(
            repo_id, filename, repo_type=kind, revision=revision, endpoint=self._endpoint
        )
        get_hf_file_metadata(url, token=self._token, timeout=self._timeout, endpoint=self._endpoint)

    def download_file(self, kind: Kind, repo_id: str, revision: str, filename: str) -> Path:
        path = self._api.hf_hub_download(
            repo_id,
            filename,
            repo_type=kind,
            revision=revision,
            cache_dir=self._cache_dir,
            token=self._token,
            etag_timeout=self._timeout,
        )
        return Path(str(path))

    def read_safetensors_header(self, repo_id: str, revision: str, filename: str) -> dict[str, Any]:
        from huggingface_hub import errors as hf_errors

        try:
            meta = self._api.parse_safetensors_file_metadata(
                repo_id,
                filename,
                repo_type="model",
                revision=revision,
                token=self._token,
                timeout=self._timeout,
            )
            header: dict[str, Any] = {
                name: {
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "data_offsets": list(tensor.data_offsets),
                }
                for name, tensor in meta.tensors.items()
            }
            if meta.metadata:
                header["__metadata__"] = dict(meta.metadata)
        except hf_errors.HFValidationError:
            raise
        except (TypeError, AttributeError, ValueError):
            # huggingface_hub==1.33.0 hf_api.py:2205-2218 (_parse_safetensors_header) converts only
            # KeyError/IndexError: a header that is a JSON array, or has non-object entries or
            # scalar offsets, escapes as TypeError/AttributeError (ValueError from dict()).
            raise hf_errors.SafetensorsParsingError("malformed safetensors header") from None
        return header


def _checked_endpoint(access: SourceAccess) -> str:
    """The configured endpoint must be https on a public host unless private networks are on."""
    endpoint = configured_hf_endpoint()
    parts = urlsplit(endpoint)
    host = (parts.hostname or "").lower()
    private = host in ("localhost", "metadata.google.internal") or host.endswith(".localhost")
    try:
        address = ipaddress.ip_address(host)
        private = private or not address.is_global
    except ValueError:
        pass
    if parts.scheme != "https" and not access.allow_private_network:
        private = True
    if not host or (private and not access.allow_private_network):
        raise blocking_error(
            ErrorCode.SOURCE_URL_NOT_ALLOWED,
            "서버에 설정된 Hugging Face endpoint가 허용되지 않습니다"
            "(https가 아니거나 사설 주소). 관리자 설정을 확인하세요.",
            stage=Stage.RESOLVING,
            details={"reason": "endpoint_not_allowed"},
        )
    return endpoint


# ---------------------------------------------------------------- error mapping


def _label(kind: Kind) -> str:
    return "모델" if kind == "model" else "데이터셋"


def metadata_error(
    kind: Kind,
    reason: str,
    *,
    retryable: bool = False,
    stage: Stage = Stage.RESOLVING,
    message: str | None = None,
    **details: object,
) -> EstimatorError:
    return blocking_error(
        ErrorCode.MODEL_METADATA_UNAVAILABLE,
        message or f"Hugging Face에서 {_label(kind)} 메타데이터를 가져오지 못했습니다.",
        stage=stage,
        retryable=retryable,
        component=kind,
        details={"reason": reason, **details},
    )


def hub_error(exc: BaseException, *, kind: Kind, stage: Stage = Stage.RESOLVING) -> EstimatorError:
    """Map a huggingface_hub / httpx exception to a contract issue (no message text copied)."""
    import httpx
    from huggingface_hub import errors as hf_errors

    status = _status(exc)
    details: dict[str, object] = {"error_type": type(exc).__name__}
    if status is not None:
        details["http_status"] = status
    logger.info("hub request failed: %s (status=%s)", type(exc).__name__, status)

    def issue(
        code: ErrorCode, message: str, reason: str, retryable: bool = False
    ) -> EstimatorError:
        return blocking_error(
            code,
            message,
            stage=stage,
            retryable=retryable,
            component=kind,
            details={"reason": reason, **details},
        )

    if isinstance(exc, hf_errors.GatedRepoError):
        return issue(
            ErrorCode.SOURCE_ACCESS_DENIED,
            "접근 승인이 필요한 저장소(gated)입니다. "
            "승인받은 계정의 Hugging Face 토큰으로만 읽을 수 있습니다.",
            "gated",
        )
    if isinstance(exc, hf_errors.DisabledRepoError):
        return issue(ErrorCode.SOURCE_ACCESS_DENIED, "비활성화된 저장소입니다.", "disabled")
    if isinstance(exc, hf_errors.RepositoryNotFoundError):
        return issue(
            ErrorCode.SOURCE_NOT_FOUND,
            f"Hugging Face에서 {_label(kind)} 저장소를 찾을 수 없습니다. 이름을 확인하세요. "
            "비공개 저장소는 접근 권한이 있는 토큰으로만 보입니다.",
            "repository_not_found",
        )
    if isinstance(exc, hf_errors.RevisionNotFoundError):
        return issue(
            ErrorCode.SOURCE_NOT_FOUND,
            "지정한 revision을 저장소에서 찾을 수 없습니다.",
            "revision_not_found",
        )
    if isinstance(exc, hf_errors.RemoteEntryNotFoundError):
        return issue(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            "고정한 revision에서 필요한 파일을 찾을 수 없습니다.",
            "file_not_found",
        )
    if isinstance(exc, hf_errors.SafetensorsParsingError):
        return issue(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            "safetensors header를 해석할 수 없습니다.",
            "malformed_safetensors_header",
        )
    if isinstance(exc, hf_errors.OfflineModeIsEnabled):
        return issue(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            "서버가 오프라인 모드로 설정되어 Hugging Face에 접근할 수 없습니다.",
            "offline_mode",
        )
    if isinstance(exc, hf_errors.LocalEntryNotFoundError | httpx.TransportError | TimeoutError):
        return issue(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            "Hugging Face에 연결하지 못했습니다(네트워크 오류 또는 시간 초과). "
            "잠시 후 다시 시도하세요.",
            "network",
            retryable=True,
        )
    if isinstance(exc, hf_errors.HFValidationError):
        return issue(
            ErrorCode.INVALID_REQUEST,
            "Hugging Face 저장소 ID 형식이 올바르지 않습니다.",
            "invalid_id",
        )
    if status in (401, 403):
        return issue(
            ErrorCode.SOURCE_ACCESS_DENIED,
            f"이 {_label(kind)} 저장소에 접근할 권한이 없습니다. 토큰 권한을 확인하세요.",
            "forbidden",
        )
    if status == 404:
        return issue(
            ErrorCode.SOURCE_NOT_FOUND,
            f"Hugging Face에서 {_label(kind)}을(를) 찾을 수 없습니다.",
            "not_found",
        )
    if status is not None and (status == 429 or status >= 500):
        return issue(
            ErrorCode.MODEL_METADATA_UNAVAILABLE,
            f"Hugging Face 서버가 요청을 처리하지 못했습니다(HTTP {status}). "
            "잠시 후 다시 시도하세요.",
            "server_unavailable",
            retryable=True,
        )
    return issue(
        ErrorCode.MODEL_METADATA_UNAVAILABLE,
        f"Hugging Face에서 {_label(kind)} 메타데이터를 가져오지 못했습니다.",
        "hub_error",
        retryable=isinstance(exc, OSError) and status is None,
    )


def _status(exc: BaseException) -> int | None:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def is_hub_exception(exc: BaseException) -> bool:
    """Exceptions that `hub_error` knows how to translate."""
    import httpx
    from huggingface_hub import errors as hf_errors

    return isinstance(
        exc,
        hf_errors.HfHubHTTPError
        | hf_errors.EntryNotFoundError
        | hf_errors.SafetensorsParsingError
        | hf_errors.HFValidationError
        | hf_errors.OfflineModeIsEnabled
        | httpx.HTTPError
        | TimeoutError
        | OSError,
    )
