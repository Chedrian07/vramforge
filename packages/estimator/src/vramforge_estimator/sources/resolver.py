"""SourceResolver: reference → immutable `SourceManifest` (plan.md §6.1, §16.2, §18).

HF sources are pinned to the commit sha returned by the Hub; local roots and uploads to a sha256
content manifest. Nothing here downloads weight files.

Dataset viewer URLs carry config/split hints in their path (``/viewer/<config>/<split>``); the
dataset inspector reads them with ``sources.references.normalize_dataset_reference(ref)``.
"""

from __future__ import annotations

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.keys import source_key
from vramforge_estimator.schemas import (
    DatasetSourceRef,
    ErrorCode,
    ModelSourceRef,
    SourceManifest,
    SourceType,
    Stage,
)

from . import hub
from .base import ResolvedSource, SourceAccess
from .local import build_local_source
from .references import (
    NormalizedReference,
    normalize_dataset_reference,
    normalize_model_reference,
)


def resolve_model(ref: ModelSourceRef, access: SourceAccess) -> ResolvedSource:
    """Normalize the reference, pin the revision to an immutable identity and list files.

    Raises `EstimatorError` with SOURCE_NOT_FOUND / SOURCE_ACCESS_DENIED / LOCAL_PATH_NOT_ALLOWED /
    SOURCE_URL_NOT_ALLOWED. Never downloads weight files.
    """
    return _resolve(normalize_model_reference(ref), access)


def resolve_dataset(ref: DatasetSourceRef, access: SourceAccess) -> ResolvedSource:
    """Same as `resolve_model` for datasets (HF datasets, local files, uploads)."""
    return _resolve(normalize_dataset_reference(ref), access)


def _resolve(norm: NormalizedReference, access: SourceAccess) -> ResolvedSource:
    if norm.source_type is SourceType.HUGGINGFACE:
        return _resolve_hf(norm, access)
    return build_local_source(norm, access)


def _resolve_hf(norm: NormalizedReference, access: SourceAccess) -> ResolvedSource:
    kind = norm.kind
    assert norm.repo_id is not None
    client = hub.get_hub_client(access)
    try:
        info = client.repo_info(kind, norm.repo_id, norm.revision)
        if info.gated:
            _check_gated_access(client, norm, info, access)
    except EstimatorError:
        raise
    except Exception as exc:
        if hub.is_hub_exception(exc):
            raise hub.hub_error(exc, kind=kind, stage=Stage.RESOLVING) from None
        raise

    notes = list(norm.notes)
    if info.repo_id != norm.repo_id:
        notes.append(
            f"저장소 이름 '{norm.repo_id}'이(가) '{info.repo_id}'(으)로 연결되어 "
            "그 기준으로 분석합니다."
        )
    if norm.revision is None:
        notes.append(
            f"revision을 지정하지 않아 기본 브랜치의 현재 commit {info.sha[:12]}로 고정했습니다."
        )
    elif norm.revision != info.sha:
        notes.append(f"revision '{norm.revision}'을(를) commit {info.sha[:12]}로 고정했습니다.")

    manifest = SourceManifest(
        kind=kind,
        source_type=SourceType.HUGGINGFACE,
        reference=f"{'hf' if kind == 'model' else 'hf-dataset'}:{info.repo_id}",
        repo_id=info.repo_id,
        requested_revision=norm.revision,
        resolved_revision=info.sha,
        files=list(info.files),
        private=info.private,
        gated=info.gated,
        last_modified=info.last_modified,
        fingerprint=source_key(SourceType.HUGGINGFACE.value, f"{kind}:{info.repo_id}@{info.sha}"),
        notes=notes,
    )
    return ResolvedSource(kind=kind, manifest=manifest, repo_id=info.repo_id, revision=info.sha)


def _check_gated_access(
    client: hub.HubClient, norm: NormalizedReference, info: hub.HubRepoInfo, access: SourceAccess
) -> None:
    """Gated repos expose metadata to everyone; probe one file to fail early without access."""
    if access.hf_token is None:
        raise EstimatorError(
            make_issue(
                ErrorCode.SOURCE_ACCESS_DENIED,
                "접근 승인이 필요한 저장소(gated)입니다. "
                "승인받은 계정의 Hugging Face 토큰을 입력하세요.",
                stage=Stage.RESOLVING,
                component=norm.kind,
                reason="gated_without_token",
            )
        )
    # README.md stays public on gated repos, so probe something else.
    probe = next(
        (f.path for f in info.files if f.path == "config.json"),
        next((f.path for f in info.files if f.path != "README.md"), None),
    )
    if probe is not None:
        client.check_file_access(norm.kind, info.repo_id, info.sha, probe)
