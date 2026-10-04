"""Tokenizer/template facts for `TokenizerManifest` (plan.md §7.3, §7.5, §16.3).

The template is analysed, never rendered here: sha256 of the exact template string the tokenizer
uses, ``{% generation %}`` markers (assistant-only loss), and the variables it reads that callers
may pass as template kwargs (e.g. ``enable_thinking``). Parsing uses a sandboxed Jinja
environment configured like transformers' chat-template environment.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable
from typing import Any

from vramforge_estimator.keys import stable_hash

# transformers==5.18.0 utils/chat_template_utils.py (render_jinja_template) uses the same pattern.
GENERATION_TAG = re.compile(r"\{%-?\s*generation\s*-?%\}")
# Variables apply_chat_template always provides (besides special tokens).
STANDARD_TEMPLATE_VARIABLES = frozenset(
    {"messages", "tools", "documents", "add_generation_prompt", "raise_exception", "strftime_now"}
)
STANDARD_SPECIAL_TOKENS = frozenset(
    {
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    }
)
# model_max_length at or above this is a "no limit" placeholder (transformers VERY_LARGE_INTEGER
# is int(1e30)); real context lengths are far below (MiMo: 262,144).
SENTINEL_THRESHOLD = 1_000_000_000
PROBE_TEXT = "VRAMForge probe 123"
FINGERPRINT_VERSION = "vramforge.tokenizer.v1"


def template_sha256(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def has_generation_markers(template: str) -> bool:
    return bool(GENERATION_TAG.search(template))


def _environment() -> Any:
    import jinja2
    import jinja2.ext
    from jinja2 import nodes
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class _GenerationTag(jinja2.ext.Extension):
        tags = {"generation"}  # noqa: RUF012 - jinja2 API

        def parse(self, parser: Any) -> Any:
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            return nodes.Scope(body, lineno=lineno)

    return ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[_GenerationTag, jinja2.ext.loopcontrols],
    )


def template_kwargs(template: str, special_tokens: Iterable[str] = ()) -> list[str] | None:
    """Variables the template reads that are neither standard, special tokens nor assigned.

    None when the template does not parse (rendering will then fail per row in the scan).
    """
    import jinja2
    from jinja2 import meta, nodes

    env = _environment()
    try:
        ast = env.parse(template)
    except jinja2.TemplateSyntaxError:
        return None
    undeclared = meta.find_undeclared_variables(ast)
    assigned = {n.name for n in ast.find_all(nodes.Name) if n.ctx in ("store", "param")}
    assigned |= {m.name for m in ast.find_all(nodes.Macro)}
    excluded = STANDARD_TEMPLATE_VARIABLES | STANDARD_SPECIAL_TOKENS | set(env.globals)
    return sorted(undeclared - assigned - excluded - set(special_tokens))


def max_length(value: object) -> tuple[int | None, bool]:
    """(model_max_length, is_sentinel); sentinels are reported as None (plan §7.5)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None, False
    if isinstance(value, float) and not math.isfinite(value):
        return None, True
    if value >= SENTINEL_THRESHOLD:
        return None, True
    return int(value), False


def special_token_defaults(tokenizer: Any) -> tuple[bool | None, bool | None]:
    """Whether a plain ``tokenizer(text)`` call adds tokens before/after the text."""
    try:
        with_special = list(tokenizer(PROBE_TEXT, add_special_tokens=True)["input_ids"])
        plain = list(tokenizer(PROBE_TEXT, add_special_tokens=False)["input_ids"])
    except Exception:
        return None, None
    if not plain:
        return None, None
    for start in range(len(with_special) - len(plain) + 1):
        if with_special[start : start + len(plain)] == plain:
            return start > 0, start + len(plain) < len(with_special)
    return None, None


def tokenizer_fingerprint(
    *,
    tokenizer_class: str,
    files_sha256: dict[str, str],
    chat_template_sha256: str | None,
    model_type: str | None,
) -> str:
    """Files alone are not enough: part of the pipeline lives in library code (research R1)."""
    import tokenizers
    import transformers

    return stable_hash(
        FINGERPRINT_VERSION,
        {
            "tokenizer_class": tokenizer_class,
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "files_sha256": files_sha256,
            "chat_template_sha256": chat_template_sha256,
            "model_type": model_type,
        },
        prefix="tok_",
    )
