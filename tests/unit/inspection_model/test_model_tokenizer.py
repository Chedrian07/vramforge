"""Tokenizer manifest on the tiny fixture (plan.md §6.2, §7.3, §7.5); local and fake-HF sources."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.inspection import TokenizerHandle, load_tokenizer
from vramforge_estimator.inspection.tokenizer_manifest import (
    max_length,
    template_kwargs,
    template_parse_error,
    template_sha256,
)
from vramforge_estimator.schemas import ErrorCode, ModelSourceRef
from vramforge_estimator.sources import SourceAccess, hub, resolve_model

FIXTURES = Path(__file__).parents[2] / "fixtures"
TOKENIZER = FIXTURES / "tokenizers" / "tiny-chat"
MODEL = FIXTURES / "models" / "tiny-dense-decoder"
TEMPLATE_SHA = "f527a96140fe38e00e4bc7bb59d634d9a652fbc078f640198dc4ecda351b9da0"
SHA = "1111111111111111111111111111111111111111"


def fixture_files() -> dict[str, bytes]:
    files = {
        name: (TOKENIZER / name).read_bytes()
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    }
    files["config.json"] = (MODEL / "config.json").read_bytes()
    return files


Loader = Callable[..., TokenizerHandle]


@pytest.fixture
def load_local(tmp_path: Path) -> Loader:
    counter = iter(range(1000))

    def load(
        files: dict[str, bytes] | None = None, *, drop: tuple[str, ...] = ()
    ) -> TokenizerHandle:
        root = tmp_path / "models"
        target = root / f"m{next(counter)}"
        target.mkdir(parents=True)
        for name, data in {**fixture_files(), **(files or {})}.items():
            if name not in drop:
                (target / name).parent.mkdir(parents=True, exist_ok=True)
                (target / name).write_bytes(data)
        access = SourceAccess(local_roots={"models": root})
        source = resolve_model(ModelSourceRef(reference=f"local:models/{target.name}"), access)
        return load_tokenizer(source, access)

    return load


def _config(**changes: Any) -> bytes:
    config = json.loads((TOKENIZER / "tokenizer_config.json").read_text(encoding="utf-8"))
    config.update(changes)
    return json.dumps(config).encode()


def test_tiny_tokenizer_manifest(load_local: Loader) -> None:
    handle = load_local()
    m = handle.manifest
    assert m.tokenizer_class == "TokenizersBackend"
    assert (m.vocab_size, m.config_vocab_size) == (386, 512)
    assert (m.bos_token, m.eos_token, m.pad_token) == (None, "<|im_end|>", "<|endoftext|>")
    assert (m.adds_bos_by_default, m.adds_eos_by_default) == (False, False)
    assert (m.model_max_length, m.model_max_length_is_sentinel) == (4096, False)
    assert m.chat_template_present is True
    assert m.chat_template_source == "chat_template.jinja"
    assert m.chat_template_sha256 == TEMPLATE_SHA
    assert m.has_generation_markers is True
    assert m.template_kwargs == ["enable_thinking"]
    assert m.template_parse_error is None
    assert m.files_sha256 == {
        "chat_template.jinja": TEMPLATE_SHA,
        "tokenizer.json": "0c6e3a3520fc6f70906a5a38ca9983b8034ce0ac57324189dfa1e5e3cc8c1e8e",
        "tokenizer_config.json": "e0bd31f8b43a38882462e32700399976462775a2128982de2d43d20573e3b257",
    }
    assert m.fingerprint.startswith("tok_")
    assert load_local().manifest.fingerprint == m.fingerprint  # deterministic
    assert handle.tokenizer.name_or_path.startswith("local:models/")

    # the handle works after the private staging directory is gone (no torch, plain lists)
    tok = handle.tokenizer
    out = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        return_assistant_tokens_mask=True,
        return_dict=True,
    )
    assert isinstance(out["input_ids"], list)
    assert sum(out["assistant_masks"]) > 0
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    assert rendered.endswith("<|im_start|>assistant\n<think></think>")


def test_fingerprint_follows_tokenizer_files(load_local: Loader) -> None:
    base = load_local().manifest
    changed_template = load_local({"chat_template.jinja": b"{{ messages[0].content }}"}).manifest
    assert changed_template.fingerprint != base.fingerprint
    assert changed_template.chat_template_sha256 != base.chat_template_sha256
    assert changed_template.has_generation_markers is False
    assert changed_template.template_kwargs == []
    changed_config = load_local({"tokenizer_config.json": _config(model_max_length=2048)}).manifest
    assert changed_config.fingerprint != base.fingerprint


@pytest.mark.parametrize("value", [None, 10**30, 1e30])
def test_model_max_length_sentinel(load_local: Loader, value: object) -> None:
    config = json.loads(_config())
    if value is None:
        config.pop("model_max_length")  # transformers then uses VERY_LARGE_INTEGER
    else:
        config["model_max_length"] = value
    m = load_local({"tokenizer_config.json": json.dumps(config).encode()}).manifest
    assert m.model_max_length is None
    assert m.model_max_length_is_sentinel is True


def test_max_length_helper() -> None:
    assert max_length(262144) == (262144, False)
    assert max_length(int(1e30)) == (None, True)
    assert max_length(float("inf")) == (None, True)
    assert max_length(None) == (None, False)
    assert max_length(True) == (None, False)


def test_template_in_tokenizer_config(load_local: Loader) -> None:
    template = (TOKENIZER / "chat_template.jinja").read_text(encoding="utf-8")
    m = load_local(
        {"tokenizer_config.json": _config(chat_template=template)}, drop=("chat_template.jinja",)
    ).manifest
    assert m.chat_template_source == "tokenizer_config.json"
    assert m.chat_template_sha256 == TEMPLATE_SHA


def test_jinja_file_wins_over_tokenizer_config(load_local: Loader) -> None:
    m = load_local({"tokenizer_config.json": _config(chat_template="{{ 'other' }}")}).manifest
    assert m.chat_template_source == "chat_template.jinja"
    assert m.chat_template_sha256 == TEMPLATE_SHA


def test_missing_template_is_reported_not_substituted(load_local: Loader) -> None:
    handle = load_local(drop=("chat_template.jinja",))
    m = handle.manifest
    assert m.chat_template_present is False
    assert (m.chat_template_source, m.chat_template_sha256) == ("none", None)
    assert (m.has_generation_markers, m.template_kwargs) == (False, [])
    assert handle.tokenizer.chat_template is None


def test_processor_held_template(load_local: Loader) -> None:
    template = (TOKENIZER / "chat_template.jinja").read_text(encoding="utf-8")
    handle = load_local(
        {"chat_template.json": json.dumps({"chat_template": template}).encode()},
        drop=("chat_template.jinja",),
    )
    m = handle.manifest
    assert m.chat_template_source == "processor"
    assert m.chat_template_sha256 == TEMPLATE_SHA
    assert "chat_template.json" in m.files_sha256
    assert handle.tokenizer.chat_template == template


def test_eos_added_by_post_processor(load_local: Loader, tmp_path: Path) -> None:
    from tokenizers import Tokenizer, processors

    tok = Tokenizer.from_file(str(TOKENIZER / "tokenizer.json"))
    eos_id = tok.token_to_id("<|endoftext|>")
    tok.post_processor = processors.TemplateProcessing(
        single="$A <|endoftext|>", special_tokens=[("<|endoftext|>", eos_id)]
    )
    path = tmp_path / "with_eos.json"
    tok.save(str(path), pretty=False)
    m = load_local({"tokenizer.json": path.read_bytes()}).manifest
    assert (m.adds_bos_by_default, m.adds_eos_by_default) == (False, True)


def test_no_tokenizer_files(load_local: Loader) -> None:
    with pytest.raises(EstimatorError) as exc:
        load_local(drop=("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"))
    assert exc.value.issue.code is ErrorCode.TOKENIZER_REQUIRED
    with pytest.raises(EstimatorError) as exc:
        load_local(drop=("tokenizer.json",))  # config alone is not a tokenizer
    assert exc.value.issue.code is ErrorCode.TOKENIZER_REQUIRED


def test_broken_tokenizer_file(load_local: Loader) -> None:
    with pytest.raises(EstimatorError) as exc:
        load_local({"tokenizer.json": b'{"version": "1.0", "model": 42}'})
    issue = exc.value.issue
    assert issue.code is ErrorCode.TOKENIZER_REQUIRED
    assert issue.details["reason"] == "tokenizer_load_failed"


def test_remote_code_tokenizer_is_refused(load_local: Loader, tmp_path: Path) -> None:
    marker = tmp_path / "executed"
    code = f"open({str(marker)!r}, 'w').write('x')\n".encode()
    config = _config(
        tokenizer_class="VramforgeCustomTokenizer",
        auto_map={"AutoTokenizer": ["tokenization_custom.VramforgeCustomTokenizer", None]},
    )
    with pytest.raises(EstimatorError) as exc:
        load_local({"tokenizer_config.json": config, "tokenization_custom.py": code})
    assert exc.value.issue.code is ErrorCode.REMOTE_CODE_REQUIRED
    assert not marker.exists()


def test_auto_map_with_native_class_loads_natively(load_local: Loader) -> None:
    config = _config(auto_map={"AutoTokenizer": ["tok.Custom", None]})
    assert (
        load_local({"tokenizer_config.json": config}).manifest.tokenizer_class
        == "TokenizersBackend"
    )


def test_template_kwargs_ignore_assigned_and_standard_names() -> None:
    template = (
        "{% set body = messages[0].content %}{{ bos_token }}{{ body }}"
        "{% for m in messages %}{{ m.role }}{% endfor %}"
        "{% macro f(x) %}{{ x }}{% endmacro %}{{ f(1) }}"
        "{% if enable_thinking is false %}x{% endif %}{{ date_string }}{{ custom_flag }}"
        "{{ raise_exception('no') if false }}{{ image_token }}"
    )
    assert template_kwargs(template) == [
        "custom_flag",
        "date_string",
        "enable_thinking",
        "image_token",
    ]
    assert template_kwargs(template, ["image_token"]) == [
        "custom_flag",
        "date_string",
        "enable_thinking",
    ]
    assert template_kwargs("{% if %}") is None
    # unknown filters fail in the code generator, not the parser: still None, never an exception
    assert template_kwargs("{{ messages | no_such_filter }}") is None


SECRET = "vf_secret_marker"
# (template, error class, template line or None): every one fails transformers' compile step.
BROKEN_TEMPLATES = [
    (f"{{% for m in messages %}}{{{{ {SECRET} }}}}", "TemplateSyntaxError", 1),  # unclosed for
    (f"{{{{ messages[0].content }}}}\n{{% {SECRET} %}}", "TemplateSyntaxError", 2),  # unknown tag
    (
        f"{{%- for m in messages -%}}\n\n{{{{ m.{SECRET} }}\n{{%- endfor -%}}",
        "TemplateSyntaxError",
        3,
    ),
    (f"{{{{ messages | {SECRET} }}}}", "TemplateAssertionError", 1),  # unknown filter (compile)
    (f"{{{{ messages is {SECRET} }}}}", "TemplateAssertionError", 1),  # unknown test (compile)
    # transformers' {% generation %} is a CallBlock: a loop control inside it is not in a loop
    (
        "{% for m in messages %}{% generation %}{% break %}{% endgeneration %}{% endfor %}",
        "SyntaxError",
        None,
    ),
    # too many statically nested blocks for Python's compiler
    (
        "".join(f"{{% for x{i} in messages %}}" for i in range(25)) + "{% endfor %}" * 25,
        "SyntaxError",
        None,
    ),
    ("{{ " + "(" * 400 + "1" + ")" * 400 + " }}", "RecursionError", None),
]
COMPILING_TEMPLATES = [
    (TOKENIZER / "chat_template.jinja").read_text(encoding="utf-8"),
    "{% for m in messages %}{% if loop.index > 2 %}{% break %}{% endif %}{{ m }}{% endfor %}",
    "{% for m in messages %}{% generation %}{{ m.content }}{% endgeneration %}{% endfor %}",
    "{{ messages | tojson(indent=2) }}{{ raise_exception('x') if not messages }}",
    # unknown filters/tests in a condition or branch only fail if rendered there (Jinja 3.x)
    "{% if not messages %}{{ messages | no_such_filter }}{% endif %}",
    "{% if messages is no_such_test %}x{% endif %}",
    "{% set ns = namespace(n=0) %}{% for m in messages %}{% set ns.n = ns.n + 1 %}{% endfor %}",
]


def _transformers_compiles(template: str) -> bool:
    from transformers.utils.chat_template_utils import _compile_jinja_template

    try:
        _compile_jinja_template(template)
    except Exception:
        return False
    return True


@pytest.mark.parametrize(("template", "error", "line"), BROKEN_TEMPLATES)
def test_template_parse_error_is_display_safe(template: str, error: str, line: int | None) -> None:
    message = template_parse_error(template)
    assert message is not None
    assert not _transformers_compiles(template)  # same verdict as the renderer
    assert f"({error}" in message
    if line is None:
        assert "번째 줄" not in message  # a Python line of generated code is not a template line
    else:
        assert f"{line}번째 줄" in message
    assert message.startswith("chat template을 Jinja로")
    # Jinja's own message quotes the template; none of it may reach the manifest
    for fragment in (SECRET, "{%", "{{", "messages", "for m", "'"):
        assert fragment not in message


@pytest.mark.parametrize("template", COMPILING_TEMPLATES)
def test_compiling_templates_have_no_parse_error(template: str) -> None:
    assert _transformers_compiles(template)
    assert template_parse_error(template) is None
    assert template_kwargs(template) is not None


@pytest.mark.parametrize(("template", "error", "line"), BROKEN_TEMPLATES[:5])
def test_uncompilable_template_is_reported_not_dropped(
    load_local: Loader, template: str, error: str, line: int
) -> None:
    handle = load_local({"chat_template.jinja": template.encode()})
    m = handle.manifest
    # still the model's own template: present, hashed and kept on the tokenizer
    assert m.chat_template_present is True
    assert m.chat_template_source == "chat_template.jinja"
    assert m.chat_template_sha256 == template_sha256(template)
    assert m.files_sha256["chat_template.jinja"] == template_sha256(template)
    assert handle.tokenizer.chat_template == template
    # no kwargs are guessed from a template that cannot be compiled
    assert m.template_kwargs == []
    assert m.template_parse_error is not None
    assert f"({error}, {line}번째 줄)" in m.template_parse_error
    assert SECRET not in m.model_dump_json()


@pytest.mark.parametrize(
    "template",
    ["{%- for m in messages -%}{{ m.content }}", "{{ messages | tojson | vf_custom_filter }}"],
)
def test_uncompilable_template_in_tokenizer_config(load_local: Loader, template: str) -> None:
    m = load_local(
        {"tokenizer_config.json": _config(chat_template=template)}, drop=("chat_template.jinja",)
    ).manifest
    assert (m.chat_template_present, m.chat_template_source) == (True, "tokenizer_config.json")
    assert m.template_parse_error is not None
    assert "vf_custom_filter" not in m.template_parse_error


@pytest.mark.parametrize(
    ("template", "expected"),
    [
        # optional kwargs that receive a fallback inside the template are still kwargs
        (
            "{%- set enable_thinking = enable_thinking if enable_thinking is defined else true %}"
            "{% if enable_thinking %}<think>{% endif %}",
            ["enable_thinking"],
        ),
        (
            "{%- if not date_string is defined %}"
            "{%- set date_string = strftime_now('%d %b %Y') %}{%- endif %}{{ date_string }}",
            ["date_string"],
        ),
        (
            "{%- if tools_in_user_message is undefined %}"
            "{%- set tools_in_user_message = true %}{%- endif %}",
            ["tools_in_user_message"],
        ),
        ("{%- set thinking = thinking | default(false) %}{{ thinking }}", ["thinking"]),
        ("{%- set thinking = thinking | d(false) %}{{ thinking }}", ["thinking"]),
        # scoping artefacts of template-local names are not kwargs
        ("{% for m in messages %}{% set last = m %}{% endfor %}{{ last }}", []),
        ("{% if messages %}{% set sys = messages[0] %}{% endif %}{{ sys }}", []),
        ("{% set x = 1 %}{% if x is defined %}{{ x }}{% endif %}", []),
        ("{% for m in messages %}{% if m is defined %}{{ m }}{% endif %}{% endfor %}", []),
    ],
)
def test_template_kwargs_with_fallback_defaults(template: str, expected: list[str]) -> None:
    assert template_kwargs(template) == expected


def test_kwarg_with_template_default_reaches_the_manifest(load_local: Loader) -> None:
    # compatibility only forwards enable_thinking when the manifest lists it (plan §7.3)
    template = (
        "{%- set enable_thinking = enable_thinking if enable_thinking is defined else true -%}"
        "{%- for message in messages -%}{{ message.content }}{%- endfor -%}"
        "{%- if add_generation_prompt and not enable_thinking -%}<think></think>{%- endif -%}"
    )
    m = load_local({"chat_template.jinja": template.encode()}).manifest
    assert m.template_kwargs == ["enable_thinking"]


# ---------------------------------------------------------------- HF path through the fake hub


def _hub(
    fake_hub: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    repo_id: str,
    files: dict[str, bytes],
) -> Any:
    repo = fake_hub.FakeRepo(
        repo_id=repo_id, sha=SHA, files=files, lfs=frozenset({"tokenizer.json"})
    )
    client = fake_hub.FakeHub(tmp_path / "hub", {("model", repo_id): repo})
    monkeypatch.setattr(hub, "get_hub_client", lambda access: client)
    return client


def test_hf_tokenizer_matches_local(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_local: Loader
) -> None:
    files = {**fixture_files(), "README.md": b"# card"}
    client = _hub(fake_hub, tmp_path, monkeypatch, "org/tiny", files)
    access = SourceAccess(hf_home=tmp_path / "hf")
    source = resolve_model(ModelSourceRef(reference="org/tiny"), access)
    handle = load_tokenizer(source, access)
    assert handle.manifest == load_local().manifest
    assert handle.tokenizer.name_or_path == "org/tiny"
    downloads = {name for call, name in client.calls if call == "download_file"}
    assert downloads == {
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "config.json",
    }


def test_hf_tokenizer_integrity(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _hub(fake_hub, tmp_path, monkeypatch, "org/tiny", fixture_files())
    client.tamper["chat_template.jinja"] = b"{{ 'swapped' }}"
    access = SourceAccess(hf_home=tmp_path / "hf")
    source = resolve_model(ModelSourceRef(reference="org/tiny"), access)
    with pytest.raises(EstimatorError) as exc:
        load_tokenizer(source, access)
    assert exc.value.issue.code is ErrorCode.MODEL_METADATA_UNAVAILABLE
    assert exc.value.issue.details["reason"] == "integrity_mismatch"


def test_hf_class_selection_matches_loading_by_repo_id(
    fake_hub: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, load_local: Loader
) -> None:
    # transformers forces TokenizersBackend for some repo ids (MODEL_IDS_TO_TOKENIZERS_BACKEND);
    # the staged copy must behave like AutoTokenizer.from_pretrained("<repo id>").
    files = {
        **fixture_files(),
        "tokenizer_config.json": _config(tokenizer_class="LlamaTokenizerFast"),
    }
    _hub(fake_hub, tmp_path, monkeypatch, "deepseek-ai/deepseek-coder-tiny-test", files)
    access = SourceAccess(hf_home=tmp_path / "hf")
    source = resolve_model(ModelSourceRef(reference="deepseek-ai/deepseek-coder-tiny-test"), access)
    assert load_tokenizer(source, access).manifest.tokenizer_class == "TokenizersBackend"
    local = load_local({"tokenizer_config.json": _config(tokenizer_class="LlamaTokenizerFast")})
    assert local.manifest.tokenizer_class == "LlamaTokenizer"
