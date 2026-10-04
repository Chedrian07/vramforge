"""Server-side source access (credentials, roots) for one owner. Never serialized."""

from __future__ import annotations

from vramforge_estimator.sources import SourceAccess

from .settings import Settings
from .store import owner_uploads_dir


def source_access(
    settings: Settings, owner_key: str, *, http_timeout_s: float = 30.0
) -> SourceAccess:
    """`uploads_dir` is the owner's own upload directory, so a resolver can only ever see
    uploads of the requesting owner (`upload:<id>` → `<uploads>/<owner>/<id>/`)."""
    return SourceAccess(
        hf_token=settings.hf_token_value(),
        local_roots=settings.local_root_map,
        uploads_dir=owner_uploads_dir(settings, owner_key),
        hf_home=settings.hf_home,
        allow_private_network=settings.allow_private_network,
        http_timeout_s=http_timeout_s,
    )


__all__ = ["source_access"]
