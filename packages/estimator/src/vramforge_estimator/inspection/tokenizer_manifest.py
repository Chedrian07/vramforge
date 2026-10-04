"""Tokenizer/template facts for `TokenizerManifest` (plan.md §7.3, §7.5, §16.3).

The template is analysed, never rendered here: sha256 of the exact template string the tokenizer
uses, ``{% generation %}`` markers (assistant-only loss), whether it compiles at all, and the
variables it reads that callers may pass as template kwargs (e.g. ``enable_thinking``). Parsing
uses a sandboxed Jinja environment configured like transformers' chat-template environment.
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
    """The environment transformers compiles chat templates in.

    transformers==5.18.0 utils/chat_template_utils.py:431-501 (_cached_compile_jinja_template):
    same sandbox options and extensions, and ``{% generation %}`` becomes the same CallBlock as in
    its AssistantTracker, so what fails to compile here fails there. Its other changes (a
    ``tojson`` filter replacing the builtin one, ``raise_exception``/``strftime_now`` globals)
    only matter when rendering.
    """
    import jinja2
    import jinja2.ext
    from jinja2 import nodes
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    class _GenerationTag(jinja2.ext.Extension):
        tags = {"generation"}  # noqa: RUF012 - jinja2 API

        def parse(self, parser: Any) -> Any:
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            call = self.call_method("_generation_support")
            return nodes.CallBlock(call, [], [], body).set_lineno(lineno)

        def _generation_support(self, caller: Any) -> Any:  # templates are never rendered here
            return caller()

    return ImmutableSandboxedEnvironment(
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=[_GenerationTag, jinja2.ext.loopcontrols],
    )


def template_parse_error(template: str) -> str | None:
    """Display-safe reason the template does not compile like transformers compiles it, else None.

    transformers compiles the template (``from_string``) before rendering any conversation, so a
    template that fails here fails for every row. Jinja's own message can quote the template (tag,
    filter and variable names, the offending line), so only the error class and, for Jinja
    errors, the template line are reported.
    """
    import jinja2

    try:
        _environment().from_string(template)
    except jinja2.TemplateSyntaxError as exc:  # parser errors and TemplateAssertionError
        action = "컴파일할" if isinstance(exc, jinja2.TemplateAssertionError) else "해석할"
        line = f", {exc.lineno}번째 줄" if isinstance(exc.lineno, int) and exc.lineno > 0 else ""
        cause = f"{type(exc).__name__}{line}"
    except Exception as exc:  # e.g. SyntaxError/RecursionError compiling the generated code
        action, cause = "컴파일할", type(exc).__name__  # its line is not a template line
    else:
        return None
    return (
        f"chat template을 Jinja로 {action} 수 없습니다({cause}). "
        "transformers도 이 template으로 대화를 렌더링할 수 없습니다."
    )


def _probed_names(ast: Any) -> set[str]:
    """Names tested with ``is defined``/``is undefined`` or read through ``| default``.

    Templates declare optional caller kwargs this way before assigning a fallback, e.g.
    ``{% set enable_thinking = enable_thinking if enable_thinking is defined else true %}``.
    """
    from jinja2 import nodes

    names = {
        test.node.name
        for test in ast.find_all(nodes.Test)
        if test.name in ("defined", "undefined") and isinstance(test.node, nodes.Name)
    }
    names |= {
        flt.node.name
        for flt in ast.find_all(nodes.Filter)
        if flt.name in ("default", "d") and isinstance(flt.node, nodes.Name)
    }
    return names


def template_kwargs(template: str, special_tokens: Iterable[str] = ()) -> list[str] | None:
    """Variables the template reads from the caller (besides standard names and special tokens).

    A name read before any assignment is a kwarg. Names that are also assigned somewhere (loop- or
    branch-scoped ``set``) count only when the template probes them as optional input
    (``is defined`` / ``| default``), which keeps defaults like ``enable_thinking`` or
    ``date_string`` while dropping scoping artefacts. None when the template does not parse
    (`template_parse_error` says why; rendering then fails per row in the scan).
    """
    import jinja2
    from jinja2 import meta, nodes

    env = _environment()
    try:
        ast = env.parse(template)
        # runs the code generator: unknown filters/tests raise TemplateAssertionError here
        undeclared = meta.find_undeclared_variables(ast)
    except jinja2.TemplateSyntaxError:
        return None
    assigned = {n.name for n in ast.find_all(nodes.Name) if n.ctx in ("store", "param")}
    assigned |= {m.name for m in ast.find_all(nodes.Macro)}
    excluded = STANDARD_TEMPLATE_VARIABLES | STANDARD_SPECIAL_TOKENS | set(env.globals)
    local_only = assigned - _probed_names(ast)
    return sorted(undeclared - local_only - excluded - set(special_tokens))


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
