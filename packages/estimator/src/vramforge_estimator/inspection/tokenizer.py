"""Implementation module (stub until implemented by the owning agent)."""

from __future__ import annotations

from vramforge_estimator.sources import ResolvedSource, SourceAccess

from .base import TokenizerHandle


def load_tokenizer(source: ResolvedSource, access: SourceAccess) -> TokenizerHandle:
    """Load the model's own tokenizer and chat template (trust_remote_code=False).

    Raises `EstimatorError` with TOKENIZER_REQUIRED / TEMPLATE_REQUIRED / REMOTE_CODE_REQUIRED.
    Never substitutes another model's tokenizer or template.
    """
    raise NotImplementedError
