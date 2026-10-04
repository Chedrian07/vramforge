"""Module-name matching rules of the pinned libraries.

Every rule mirrors the cited source so that "which modules are adapted / quantized / saved" is
decided exactly like the trainer will (docs/research/loading-quantization-peft.md §2.2, §5.4,
§5.5). Patterns come from user requests, so regexes are size-limited and nested quantifiers are
rejected before compiling (Python `re` has no timeout).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from functools import lru_cache

from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import ErrorCode, Stage

MAX_PATTERN_LENGTH = 512
# `(...+)+`, `(...*)*`, `(...+){2,}` style nesting: classic catastrophic backtracking shapes.
_NESTED_QUANTIFIER = re.compile(r"\((?:[^()\\]|\\.)*[+*}](?:[^()\\]|\\.)*\)\s*[+*{]")


def pattern_error(pattern: str, reason: str) -> EstimatorError:
    return EstimatorError(
        make_issue(
            ErrorCode.INVALID_REQUEST,
            "모듈 이름 패턴을 사용할 수 없습니다: " + reason,
            stage=Stage.REQUEST,
            component="lora",
            pattern=pattern[:80],
        )
    )


@lru_cache(maxsize=1024)
def compile_user_regex(pattern: str) -> re.Pattern[str]:
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise pattern_error(pattern, f"{MAX_PATTERN_LENGTH}자를 넘습니다.")
    if _NESTED_QUANTIFIER.search(pattern):
        raise pattern_error(pattern, "중첩 반복(예: (a+)+)은 허용하지 않습니다.")
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise pattern_error(pattern, f"정규식 오류 ({exc.msg}).") from exc


def peft_name_match(name: str, names: Iterable[str]) -> bool:
    """PEFT list semantics for target_modules / exclude_modules: exact or `.suffix`
    (peft==0.21.2 tuners/tuners_utils.py:2350-2376, check_target_module_exists)."""
    return any(name == n or name.endswith("." + n) for n in names)


def peft_regex_match(name: str, pattern: str) -> bool:
    """PEFT string semantics: `re.fullmatch(pattern, name)` (utils/other.py:1516-1521)."""
    return compile_user_regex(pattern).fullmatch(name) is not None


def inside_modules_to_save(name: str, modules_to_save: Iterable[str]) -> bool:
    """Adapters never wrap modules under a modules_to_save module:
    `re.match(rf"(^|.*\\.){m}($|\\..*)", key)` (tuners_utils.py:2361-2364)."""
    return any(compile_user_regex(rf"(^|.*\.){m}($|\..*)").match(name) for m in modules_to_save)


def modules_to_save_match(name: str, modules_to_save: Iterable[str]) -> bool:
    """ModulesToSaveWrapper selection: `key.endswith(target)` without a dot boundary
    (peft utils/other.py:1087-1088, _set_trainable)."""
    return any(name.endswith(m) for m in modules_to_save)


def rank_for(name: str, rank_pattern: Mapping[str, int], default: int) -> int:
    """LoRA rank of one module: first `rank_pattern` key with
    `re.match(rf"(.*\\.)?({key})$", name)` (utils/other.py:1524-1532, lora/model.py:309-311)."""
    for key, rank in rank_pattern.items():
        if compile_user_regex(rf"(.*\.)?({key})$").match(name):
            return rank
    return default


def bnb_skip_match(name: str, patterns: Iterable[str]) -> bool:
    """bitsandbytes skip list: `re.match(f"{key}\\.", n) or re.match(key, n) or n.endswith(key)`
    (transformers==5.18.0 quantizers/quantizers_utils.py:26-42, should_convert_module)."""
    return any(compile_user_regex(p).match(name) or name.endswith(p) for p in patterns)


def name_matches_any(name: str, patterns: Iterable[str]) -> bool:
    """Profile patterns (e.g. upcast_to_fp32_patterns): "*" = all, else PEFT list semantics."""
    return any(p == "*" or name == p or name.endswith("." + p) for p in patterns)
