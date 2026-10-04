"""Pure normalization of model/dataset references (plan.md §6.1, §18).

No network or filesystem access happens here. Every accepted input maps to exactly one of:
a Hugging Face repo id (plus an optional revision), ``local:<root>/<relative>`` or
``upload:<id>``. Anything else is rejected with a stable error code. URLs are only parsed for
their path; the host is never contacted (only the server-configured HF endpoint is, see
``sources.hub``), so user input can never choose where requests go.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import parse_qsl, unquote, urlsplit

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.schemas import (
    DatasetSourceRef,
    ErrorCode,
    ModelSourceRef,
    SourceType,
    Stage,
)

from .issues import blocking_error

Kind = Literal["model", "dataset"]

DEFAULT_HF_ENDPOINT = "https://huggingface.co"
HF_HOSTS = frozenset({"huggingface.co", "www.huggingface.co", "hf.co"})

# First path segments of huggingface.co that are site pages, not repositories.
_RESERVED_FIRST_SEGMENTS = frozenset(
    {
        "api",
        "blog",
        "buckets",
        "chat",
        "collections",
        "datasets",
        "docs",
        "join",
        "learn",
        "login",
        "models",
        "new",
        "organizations",
        "papers",
        "posts",
        "pricing",
        "settings",
        "spaces",
        "tasks",
    }
)
_FILE_ACTIONS = frozenset({"blob", "resolve", "raw"})

# huggingface_hub validate_repo_id: word chars, "-", "." ; no "--"/".."; max 96 chars.
_REPO_PART = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_])?$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+-]{0,255}$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CONTENT_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_SPLIT_OR_CONFIG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,255}$")
_ROOT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# Upload ids are server-generated (uuid / ulid / token_urlsafe): no dots, no separators.
_UPLOAD_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$")
_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_CONTROL = re.compile(r"[\x00-\x20\x7f]")

_EXAMPLE_FORMS = (
    "Hugging Face 저장소 ID(예: org/name), https://huggingface.co 주소, "
    "local:<root>/<상대 경로> 또는 upload:<id> 형식만 사용할 수 있습니다."
)


@dataclass(frozen=True)
class NormalizedReference:
    """Display-safe, validated form of a user reference."""

    kind: Kind
    source_type: SourceType
    display: str  # "hf:org/name", "hf-dataset:org/name", "local:<root>/<rel>", "upload:<id>"
    repo_id: str | None = None
    revision: str | None = None  # effective requested revision (URL or field); None = default
    local_root: str | None = None
    local_relative: str | None = None  # "" means the root directory itself
    upload_id: str | None = None
    path_in_repo: str | None = None  # file/dir path of /tree|blob|resolve URLs (informational)
    config_hint: str | None = None  # dataset viewer URL: /viewer/<config>/<split>
    split_hint: str | None = None
    notes: tuple[str, ...] = ()


def normalize_model_reference(ref: ModelSourceRef) -> NormalizedReference:
    return _normalize(
        "model",
        ref.reference,
        ref.source_type,
        explicit_type="source_type" in ref.model_fields_set,
        revision=ref.revision,
    )


def normalize_dataset_reference(ref: DatasetSourceRef) -> NormalizedReference:
    """Normalize a dataset reference; viewer URL config/split become `config_hint`/`split_hint`.

    The URL query (e.g. ``?row=0``) never restricts the analysed range (plan §6.1).
    """
    return _normalize(
        "dataset",
        ref.reference,
        ref.source_type,
        explicit_type="source_type" in ref.model_fields_set,
        revision=ref.revision,
        config=ref.config,
        split=ref.split,
    )


def configured_hf_endpoint() -> str:
    """The server-side HF endpoint (``HF_ENDPOINT``); never taken from user input."""
    try:
        from huggingface_hub import constants
    except ImportError:  # analysis extra not installed
        return os.environ.get("HF_ENDPOINT", DEFAULT_HF_ENDPOINT).rstrip("/")
    return str(constants.ENDPOINT).rstrip("/")


def allowed_hf_hosts() -> frozenset[str]:
    host = urlsplit(configured_hf_endpoint()).hostname
    return HF_HOSTS | ({host.lower()} if host else set())


def is_commit_sha(revision: str | None) -> bool:
    return bool(revision and _COMMIT_SHA.match(revision))


# ---------------------------------------------------------------- internals


def _error(code: ErrorCode, message: str, kind: Kind, **details: object) -> EstimatorError:
    return blocking_error(code, message, stage=Stage.RESOLVING, component=kind, details=details)


def _normalize(
    kind: Kind,
    reference: str,
    source_type: SourceType,
    *,
    explicit_type: bool,
    revision: str | None,
    config: str | None = None,
    split: str | None = None,
) -> NormalizedReference:
    raw = reference.strip()
    if not raw or _CONTROL.search(raw):
        raise _error(
            ErrorCode.INVALID_REQUEST,
            "참조에 공백이나 제어 문자를 넣을 수 없습니다. " + _EXAMPLE_FORMS,
            kind,
            reason="invalid_characters",
        )
    field_revision = _clean_optional(revision)

    scheme_match = _SCHEME.match(raw)
    if _WINDOWS_DRIVE.match(raw) or raw.startswith(("/", "\\", "~", "./", "../")) or "\\" in raw:
        raise _error(
            ErrorCode.LOCAL_PATH_NOT_ALLOWED,
            "서버의 절대 경로나 상대 경로는 직접 사용할 수 없습니다. "
            "관리자가 등록한 root를 기준으로 local:<root>/<상대 경로> 형식으로 입력하세요.",
            kind,
            reason="host_path",
        )

    if scheme_match:
        scheme = scheme_match.group(1).lower()
        rest = raw[scheme_match.end() :]
        if scheme == "local":
            _check_type(kind, SourceType.LOCAL, source_type, explicit_type)
            return _local(kind, rest, field_revision)
        if scheme == "upload":
            _check_type(kind, SourceType.UPLOAD, source_type, explicit_type)
            return _upload(kind, rest, field_revision)
        if scheme in ("hf", "hf-dataset"):
            _check_type(kind, SourceType.HUGGINGFACE, source_type, explicit_type)
            id_kind: Kind = "model" if scheme == "hf" else "dataset"
            if id_kind != kind:
                raise _kind_mismatch(kind)
            return _hf(kind, _repo_id(rest, kind), None, field_revision, None, (), config, split)
        if scheme in ("https", "http"):
            _check_type(kind, SourceType.HUGGINGFACE, source_type, explicit_type)
            return _hf_url(kind, raw, field_revision, config, split)
        raise _error(
            ErrorCode.SOURCE_URL_NOT_ALLOWED,
            "허용되지 않는 주소 형식입니다. " + _EXAMPLE_FORMS,
            kind,
            reason="scheme_not_allowed",
        )

    if source_type is SourceType.LOCAL:
        return _local(kind, raw, field_revision)
    if source_type is SourceType.UPLOAD:
        return _upload(kind, raw, field_revision)
    return _hf(kind, _repo_id(raw, kind), None, field_revision, None, (), config, split)


def _clean_optional(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _check_type(kind: Kind, implied: SourceType, given: SourceType, explicit_type: bool) -> None:
    if explicit_type and given is not implied:
        raise _error(
            ErrorCode.CONFLICTING_OPTIONS,
            "참조 형식(Hugging Face / local: / upload:)과 source_type 값이 다릅니다. "
            "둘 중 하나를 고쳐 주세요.",
            kind,
            reason="source_type_conflict",
            source_type=given.value,
            implied_source_type=implied.value,
        )


def _kind_mismatch(kind: Kind) -> EstimatorError:
    if kind == "dataset":
        message = (
            "데이터셋 입력에 모델 참조가 들어왔습니다. "
            "https://huggingface.co/datasets/<org>/<name> 형식이나 데이터셋 ID를 입력하세요."
        )
    else:
        message = (
            "모델 입력에 데이터셋 참조가 들어왔습니다. "
            "https://huggingface.co/<org>/<name> 형식이나 모델 ID를 입력하세요."
        )
    return _error(ErrorCode.SOURCE_URL_NOT_ALLOWED, message, kind, reason="kind_mismatch")


def _repo_id(value: str, kind: Kind) -> str:
    parts = value.split("/")
    valid = (
        1 <= len(parts) <= 2
        and len(value) <= 96
        and all(_REPO_PART.match(p) for p in parts)
        and "--" not in value
        and ".." not in value
        and not value.endswith(".git")
    )
    if not valid:
        raise _error(
            ErrorCode.INVALID_REQUEST,
            "Hugging Face 저장소 ID 형식이 올바르지 않습니다(예: org/name). " + _EXAMPLE_FORMS,
            kind,
            reason="invalid_repo_id",
        )
    return value


def _validate_revision(value: str, kind: Kind) -> str:
    valid = (
        _REVISION.match(value)
        and ".." not in value
        and "//" not in value
        and not value.endswith(("/", ".lock"))
    )
    if not valid:
        raise _error(
            ErrorCode.INVALID_REQUEST,
            "revision 형식이 올바르지 않습니다. 브랜치·태그 이름 또는 commit sha를 입력하세요.",
            kind,
            reason="invalid_revision",
        )
    return value


def _pick(kind: Kind, field: str, from_url: str | None, from_field: str | None) -> str | None:
    """URL path values and explicit fields must agree; a conflict is never resolved silently."""
    if from_url is not None and from_field is not None and from_url != from_field:
        raise _error(
            ErrorCode.CONFLICTING_OPTIONS,
            f"주소에 들어 있는 {field} 값과 {field} 입력값이 다릅니다. "
            "하나만 지정하거나 같게 맞춰 주세요.",
            kind,
            reason=f"{field}_conflict",
            url_value=from_url,
            field_value=from_field,
        )
    return from_url if from_url is not None else from_field


def _hf(
    kind: Kind,
    repo_id: str,
    url_revision: str | None,
    field_revision: str | None,
    path_in_repo: str | None,
    notes: tuple[str, ...],
    config: str | None,
    split: str | None,
    config_hint: str | None = None,
    split_hint: str | None = None,
) -> NormalizedReference:
    if url_revision is not None:
        _validate_revision(url_revision, kind)
    if field_revision is not None:
        _validate_revision(field_revision, kind)
    revision = _pick(kind, "revision", url_revision, field_revision)
    if kind == "dataset":
        _pick(kind, "config", config_hint, _clean_optional(config))
        _pick(kind, "split", split_hint, _clean_optional(split))
    prefix = "hf" if kind == "model" else "hf-dataset"
    return NormalizedReference(
        kind=kind,
        source_type=SourceType.HUGGINGFACE,
        display=f"{prefix}:{repo_id}",
        repo_id=repo_id,
        revision=revision,
        path_in_repo=path_in_repo,
        config_hint=config_hint,
        split_hint=split_hint,
        notes=notes,
    )


def _hf_url(
    kind: Kind,
    raw: str,
    field_revision: str | None,
    config: str | None,
    split: str | None,
) -> NormalizedReference:
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:
        raise _error(
            ErrorCode.SOURCE_URL_NOT_ALLOWED,
            "주소를 해석할 수 없습니다.",
            kind,
            reason="malformed_url",
        ) from None
    if parts.scheme.lower() != "https":
        raise _error(
            ErrorCode.SOURCE_URL_NOT_ALLOWED,
            "https 주소만 허용합니다.",
            kind,
            reason="scheme_not_allowed",
        )
    if parts.username is not None or parts.password is not None:
        raise _error(
            ErrorCode.SOURCE_URL_NOT_ALLOWED,
            "주소에 사용자 정보(아이디·비밀번호·토큰)를 넣을 수 없습니다. "
            "접근 토큰은 별도 입력란을 사용하세요.",
            kind,
            reason="credentials_in_url",
        )
    host = (parts.hostname or "").lower()
    if host not in allowed_hf_hosts() or port is not None:
        raise _error(
            ErrorCode.SOURCE_URL_NOT_ALLOWED,
            "Hugging Face(huggingface.co) 주소만 허용합니다.",
            kind,
            reason="host_not_allowed",
        )

    segments = [unquote(s) for s in parts.path.split("/") if s]
    if any(s in (".", "..") or _CONTROL.search(s) for s in segments):
        raise _unsupported_path(kind)

    notes: list[str] = []
    query = parse_qsl(parts.query, keep_blank_values=True)
    if query:
        if any(name == "row" for name, _ in query):
            notes.append(
                "주소의 row 값은 미리보기 위치일 뿐이라 무시했습니다. "
                "분석은 선택한 split의 모든 row를 대상으로 합니다."
            )
        if any(name != "row" for name, _ in query):
            notes.append(
                "주소의 조회 매개변수(검색·필터·페이지)는 분석 범위를 바꾸지 않아 무시했습니다."
            )

    if not segments:
        raise _unsupported_path(kind)
    is_dataset_url = segments[0] == "datasets"
    if (kind == "dataset") != is_dataset_url:
        if segments[0] in _RESERVED_FIRST_SEGMENTS and not is_dataset_url:
            raise _unsupported_path(kind)
        raise _kind_mismatch(kind)
    if is_dataset_url:
        segments = segments[1:]
    if not segments or segments[0] in _RESERVED_FIRST_SEGMENTS:
        raise _unsupported_path(kind)

    # org/name or a legacy single-segment id (only when nothing follows it).
    if len(segments) == 1:
        repo_parts, rest = segments, []
    else:
        repo_parts, rest = segments[:2], segments[2:]
    repo_id = _repo_id("/".join(repo_parts), kind)

    url_revision: str | None = None
    path_in_repo: str | None = None
    config_hint: str | None = None
    split_hint: str | None = None
    if rest:
        action = rest[0]
        if action in ("tree", "commit", *_FILE_ACTIONS):
            if len(rest) < 2:
                raise _unsupported_path(kind)
            url_revision = rest[1]
            if action == "commit" and len(rest) > 2:
                raise _unsupported_path(kind)
            if action in _FILE_ACTIONS and len(rest) < 3:
                raise _unsupported_path(kind)
            if len(rest) > 2:
                path_in_repo = "/".join(rest[2:])
                if kind == "model":
                    notes.append(
                        "주소의 파일 경로는 저장소 위치 확인에만 사용했습니다. "
                        "모델은 저장소 전체를 기준으로 분석합니다."
                    )
                else:
                    notes.append(
                        "주소의 파일 경로는 분석 범위를 제한하지 않습니다. "
                        "범위는 config/split 선택으로 정합니다."
                    )
        elif action == "viewer" and kind == "dataset":
            if len(rest) > 3:
                raise _unsupported_path(kind)
            if len(rest) >= 2:
                config_hint = _hint(rest[1], kind, "config")
            if len(rest) == 3:
                split_hint = _hint(rest[2], kind, "split")
        else:
            raise _unsupported_path(kind)

    if config_hint is not None or split_hint is not None:
        selected = ", ".join(
            f"{name} '{value}'"
            for name, value in (("config", config_hint), ("split", split_hint))
            if value is not None
        )
        notes.append(f"데이터셋 viewer 주소의 경로에 {selected}이(가) 들어 있습니다.")

    return _hf(
        kind,
        repo_id,
        url_revision,
        field_revision,
        path_in_repo,
        tuple(notes),
        config,
        split,
        config_hint=config_hint,
        split_hint=split_hint,
    )


def _hint(value: str, kind: Kind, field: str) -> str:
    if not _SPLIT_OR_CONFIG.match(value):
        raise _error(
            ErrorCode.INVALID_REQUEST,
            f"주소의 {field} 이름 형식이 올바르지 않습니다.",
            kind,
            reason=f"invalid_{field}",
        )
    return value


def _unsupported_path(kind: Kind) -> EstimatorError:
    forms = "저장소 메인, tree/<revision>, blob·resolve 파일 주소"
    if kind == "dataset":
        forms += ", viewer/<config>/<split>"
    return _error(
        ErrorCode.SOURCE_URL_NOT_ALLOWED,
        f"지원하지 않는 Hugging Face 주소 형식입니다. {forms}만 지원합니다.",
        kind,
        reason="unsupported_url_path",
    )


def _local_revision(kind: Kind, revision: str | None) -> str | None:
    if revision is None:
        return None
    if not _CONTENT_DIGEST.match(revision):
        raise _error(
            ErrorCode.CONFLICTING_OPTIONS,
            "로컬·업로드 source에는 revision 대신 "
            "내용 digest(sha256:<64자리 hex>)만 지정할 수 있습니다.",
            kind,
            reason="revision_not_applicable",
        )
    return revision


def _local(kind: Kind, rest: str, revision: str | None) -> NormalizedReference:
    root, sep, relative = rest.partition("/")
    relative = relative.rstrip("/") if sep else ""
    segments = relative.split("/") if relative else []
    bad = (
        not _ROOT_NAME.match(root)
        or any(s in ("", ".", "..") for s in segments)
        or any(":" in s or _CONTROL.search(s) for s in segments)
    )
    if bad:
        raise _error(
            ErrorCode.LOCAL_PATH_NOT_ALLOWED,
            "로컬 경로는 local:<root>/<상대 경로> 형식이어야 하며 '..', '.', 빈 경로 요소, "
            "절대 경로를 쓸 수 없습니다.",
            kind,
            reason="invalid_local_reference",
        )
    return NormalizedReference(
        kind=kind,
        source_type=SourceType.LOCAL,
        display=f"local:{root}/{relative}",
        revision=_local_revision(kind, revision),
        local_root=root,
        local_relative=relative,
    )


def _upload(kind: Kind, upload_id: str, revision: str | None) -> NormalizedReference:
    if not _UPLOAD_ID.match(upload_id):
        raise _error(
            ErrorCode.LOCAL_PATH_NOT_ALLOWED,
            "업로드 ID 형식이 올바르지 않습니다.",
            kind,
            reason="invalid_upload_id",
        )
    return NormalizedReference(
        kind=kind,
        source_type=SourceType.UPLOAD,
        display=f"upload:{upload_id}",
        revision=_local_revision(kind, revision),
        upload_id=upload_id,
    )
