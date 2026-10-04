"""Parity between OUR torch-free adapters and the REAL TRL 1.14.1 trainers (CPU, offline).

Recipe: docs/research/trl-sft-dpo.md §11 - byte-level tiny tokenizer with the real MiMo chat
template (tests/fixtures/golden/tokenizers) and a tiny ``Qwen3_5ForCausalLM`` saved to a temp dir,
passed to the trainers as a path so TRL's own model loading runs. The TRL-side datasets are built
independently from the raw rows (TRL format: prompt/completion, messages, text, preference), so the
comparison checks our mapping too. Lengths, loss masks and collator shapes must be identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

for _module in ("torch", "trl", "peft", "datasets"):  # collect cleanly without the parity group
    pytest.importorskip(_module)

import datasets  # noqa: E402
import torch  # noqa: E402
from transformers import (  # noqa: E402
    PreTrainedTokenizerFast,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)
from trl import (  # noqa: E402
    DPOConfig,
    DPOTrainer,
    GRPOConfig,
    GRPOTrainer,
    SFTConfig,
    SFTTrainer,
)

from vramforge_estimator.batching import plan_batches  # noqa: E402
from vramforge_estimator.inspection import TokenizerHandle  # noqa: E402
from vramforge_estimator.preprocessing import TokenizedRecord, get_adapter  # noqa: E402
from vramforge_estimator.preprocessing.trl_sft import TrlSftAdapter  # noqa: E402
from vramforge_estimator.scan import LengthTable  # noqa: E402
from vramforge_estimator.schemas import (  # noqa: E402
    ColumnMapping,
    EffectiveDtypes,
    EmptySystemPolicy,
    GrpoResolved,
    Objective,
    OptimizerResolved,
    QuantizationResolved,
    ResolvedConfig,
    RewardKind,
    RolloutBackend,
    Strategy,
    TokenizerManifest,
    WorkspaceAssumptions,
)

pytestmark = pytest.mark.parity

TOKENIZERS = Path(__file__).parents[1] / "fixtures" / "golden" / "tokenizers"
OMIT, KEEP = EmptySystemPolicy.OMIT, EmptySystemPolicy.KEEP

RAW = [  # CyberNative-like rows: empty/blank/real system, Korean, code fences, long outliers
    {
        "system": "",
        "question": "Write a C function.",
        "chosen": "```c\nint f(void){return 0;}\n```",
        "rejected": "```c\nint f(){}\n```",
        "lang": "c",
    },
    {
        "system": "You are a security reviewer.",
        "question": "파이썬 eval 위험성을 설명해줘",
        "chosen": "eval은 위험합니다.\n```python\nimport ast\nast.literal_eval(x)\n```",
        "rejected": "괜찮아요",
        "lang": "python",
    },
    {
        "system": "  ",
        "question": "Long answer please.",
        "chosen": "A" * 1500,
        "rejected": "B" * 3000,
        "lang": "go",
    },
    {
        "system": "",
        "question": "SQL 인젝션 예제와 수정 코드",
        "chosen": '```go\ndb.Query("SELECT * FROM t WHERE id = ?", id)\n```',
        "rejected": '```go\ndb.Query("SELECT * FROM t WHERE id = " + id)\n```',
        "lang": "go",
    },
]
PREF = ColumnMapping(system="system", prompt="question", chosen="chosen", rejected="rejected")


def u(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


def a(text: str) -> dict[str, str]:
    return {"role": "assistant", "content": text}


def s(text: str) -> dict[str, str]:
    return {"role": "system", "content": text}


CONVS = [
    [u("2+2는?"), a("4"), u("코드로?"), a("```py\nprint(2 + 2)\n```")],
    [s("짧게 답해"), u("hello"), a("hi")],
    [s(""), u("빈 system 메시지"), a("ok")],
]


def system_part(text: str, policy: EmptySystemPolicy) -> list[dict[str, str]]:
    """The product mapping rule, written independently of the adapters."""
    if policy is OMIT and not text.strip():
        return []
    return [s(text)]


def trl_prompt(row: dict[str, str], policy: EmptySystemPolicy) -> list[dict[str, str]]:
    return [*system_part(row["system"], policy), u(row["question"])]


def drop_blank_system(
    conv: list[dict[str, str]], policy: EmptySystemPolicy
) -> list[dict[str, str]]:
    if policy is KEEP:
        return conv
    return [m for m in conv if not (m["role"] == "system" and not m["content"].strip())]


# ---------------------------------------------------------------- fixtures


def _tokenizer(name: str) -> PreTrainedTokenizerFast:
    return PreTrainedTokenizerFast.from_pretrained(TOKENIZERS / name)


@pytest.fixture(scope="module")
def mimo() -> PreTrainedTokenizerFast:
    return _tokenizer("mimo_bytelevel")


@pytest.fixture(scope="module")
def plain() -> PreTrainedTokenizerFast:
    return _tokenizer("plain_bytelevel")


def handle(tok: PreTrainedTokenizerFast) -> TokenizerHandle:
    template = tok.chat_template or ""
    manifest = TokenizerManifest(
        tokenizer_class=type(tok).__name__,
        vocab_size=len(tok),
        eos_token=tok.eos_token,
        chat_template_present=bool(template),
        has_generation_markers="generation -%}" in template,
        fingerprint="parity",
    )
    return TokenizerHandle(tokenizer=tok, manifest=manifest)


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory, mimo: PreTrainedTokenizerFast) -> str:
    path = tmp_path_factory.mktemp("tiny-qwen3_5")
    config = Qwen3_5TextConfig(
        vocab_size=len(mimo),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        eos_token_id=mimo.eos_token_id,
        pad_token_id=mimo.pad_token_id,
    )
    torch.manual_seed(0)
    Qwen3_5ForCausalLM(config).save_pretrained(path)  # architectures -> create_model_from_path
    return str(path)


def args(cls: type, out: Path, **kw: Any) -> Any:
    base = {
        "output_dir": str(out),
        "report_to": "none",
        "use_cpu": True,
        "bf16": False,
        "per_device_train_batch_size": 2,
        "save_strategy": "no",
    }
    return cls(**{**base, **kw})


def resolved(objective: Objective, microbatch: int, pad: int | None) -> ResolvedConfig:
    dt = "float32"
    return ResolvedConfig(
        objective=objective,
        strategy=Strategy.FULL,
        loading_scope="text_only",
        load_dtype=dt,
        effective_dtypes=EffectiveDtypes(
            weights_nonquantized=dt,
            compute=dt,
            adapter=dt,
            gradient=dt,
            optimizer_state=dt,
            logits=dt,
            loss=dt,
            kv_cache=dt,
            recurrent_state=dt,
        ),
        quantization=QuantizationResolved(enabled=False),
        optimizer=OptimizerResolved(name="adamw_torch_fused", states_per_param=2, state_dtype=dt),
        microbatch=microbatch,
        accumulation=1,
        pad_to_multiple_of=pad,
        gradient_checkpointing=True,
        checkpointing_granularity="per_decoder_layer",
        loss_path="chunked_nll",
        workspace=WorkspaceAssumptions(
            cuda_context_bytes=(1, 1),
            library_workspace_bytes=(1, 1),
            allocator_slack_fraction=(0.0, 0.0),
        ),
    )


def table(records: list[TokenizedRecord]) -> LengthTable:
    return LengthTable(
        row_ids=[r.row_id for r in records],
        prompt_tokens=[r.prompt_tokens for r in records],
        completion_tokens=[r.completion_tokens for r in records],
        sequence_tokens=[r.sequence_tokens for r in records],
        loss_token_count=[r.loss_token_count for r in records],
        chosen_total_tokens=[r.chosen_total_tokens for r in records],
        rejected_total_tokens=[r.rejected_total_tokens for r in records],
    )


def assert_sft_matches(trainer: SFTTrainer, ours: list[TokenizedRecord]) -> None:
    assert len(trainer.train_dataset) == len(ours)  # nothing dropped
    for i, rec in enumerate(ours):
        assert rec.ok, rec.error_message
        ids = trainer.train_dataset[i]["input_ids"]
        labels = trainer.train_dataset[i]["labels"]
        assert rec.sequence_tokens == len(ids)
        assert rec.loss_token_count == sum(1 for t in labels[1:] if t != -100)
        if rec.completion_tokens is not None:  # prompt-completion: completion_only_loss
            assert rec.completion_tokens == sum(1 for t in labels if t != -100)
            assert rec.prompt_tokens == len(ids) - rec.completion_tokens


# ---------------------------------------------------------------- SFT


@pytest.mark.parametrize("policy", [OMIT, KEEP])
def test_sft_conversational_prompt_completion(tiny_dir, mimo, tmp_path, policy) -> None:
    rows = [{"prompt": trl_prompt(r, policy), "completion": [a(r["chosen"])]} for r in RAW]
    trainer = SFTTrainer(
        model=tiny_dir,
        args=args(SFTConfig, tmp_path, max_length=None),
        train_dataset=datasets.Dataset.from_list(rows),
        processing_class=mimo,
    )
    mapping = PREF.model_copy(update={"empty_system_policy": policy})
    adapter = get_adapter(Objective.SFT, handle(mimo), mapping, empty_system_policy=policy)
    ours = [adapter.process(r, f"train:{i}") for i, r in enumerate(RAW)]
    assert_sft_matches(trainer, ours)
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(RAW))])
    plan = plan_batches(table(ours), resolved(Objective.SFT, len(RAW), None), seed=42)
    assert tuple(batch["input_ids"].shape) == (
        plan.worst_case.sequences_per_forward,
        plan.worst_case.padded_length,
    )


@pytest.mark.parametrize("policy", [OMIT, KEEP])
@pytest.mark.parametrize("assistant_only_loss", [False, True])
def test_sft_messages(tiny_dir, mimo, tmp_path, policy, assistant_only_loss) -> None:
    rows = [{"messages": drop_blank_system(c, policy)} for c in CONVS]
    trainer = SFTTrainer(
        model=tiny_dir,
        args=args(SFTConfig, tmp_path, max_length=None, assistant_only_loss=assistant_only_loss),
        train_dataset=datasets.Dataset.from_list(rows),
        processing_class=mimo,
    )
    mapping = ColumnMapping(messages="conv", empty_system_policy=policy)
    adapter = TrlSftAdapter(
        handle(mimo),
        mapping,
        empty_system_policy=policy,
        assistant_only_loss=assistant_only_loss,
    )
    ours = [adapter.process({"conv": c, "id": i}, f"train:{i}") for i, c in enumerate(CONVS)]
    assert_sft_matches(trainer, ours)


def test_sft_plain_prompt_completion_and_text(tiny_dir, plain, tmp_path) -> None:
    raw = [
        {"p": "Q: 2+2?", "c": " 4"},
        {"p": "질문: ", "c": "답변</s>"},
        {"p": "x" * 900, "c": "y"},
    ]
    pc = SFTTrainer(
        model=tiny_dir,
        args=args(SFTConfig, tmp_path / "pc", max_length=None),
        train_dataset=datasets.Dataset.from_list(
            [{"prompt": r["p"], "completion": r["c"]} for r in raw]
        ),
        processing_class=plain,
    )
    adapter = get_adapter(Objective.SFT, handle(plain), ColumnMapping(prompt="p", completion="c"))
    assert_sft_matches(pc, [adapter.process(r, f"train:{i}") for i, r in enumerate(raw)])

    texts = ["hello 세계", "already ends</s>", "```py\nprint(1)\n```"]
    lm = SFTTrainer(
        model=tiny_dir,
        args=args(SFTConfig, tmp_path / "lm", max_length=None),
        train_dataset=datasets.Dataset.from_list([{"text": t} for t in texts]),
        processing_class=plain,
    )
    text_adapter = get_adapter(Objective.SFT, handle(plain), ColumnMapping(text="t"))
    assert_sft_matches(lm, [text_adapter.process({"t": t}, f"r{i}") for i, t in enumerate(texts)])


def test_sft_pad_to_multiple_of_matches_the_planner(tiny_dir, mimo, tmp_path) -> None:
    rows = [{"prompt": trl_prompt(r, OMIT), "completion": [a(r["chosen"])]} for r in RAW]
    trainer = SFTTrainer(
        model=tiny_dir,
        args=args(SFTConfig, tmp_path, max_length=None, pad_to_multiple_of=64),
        train_dataset=datasets.Dataset.from_list(rows),
        processing_class=mimo,
    )
    ours = [
        get_adapter(Objective.SFT, handle(mimo), PREF).process(r, f"t:{i}")
        for i, r in enumerate(RAW)
    ]
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(RAW))])
    plan = plan_batches(table(ours), resolved(Objective.SFT, len(RAW), 64), seed=1)
    assert tuple(batch["input_ids"].shape) == (len(RAW), plan.worst_case.padded_length)
    assert plan.worst_case.padded_length % 64 == 0


# ---------------------------------------------------------------- DPO


def assert_dpo_matches(trainer: DPOTrainer, ours: list[TokenizedRecord]) -> None:
    assert len(trainer.train_dataset) == len(ours)
    for i, rec in enumerate(ours):
        assert rec.ok, rec.error_message
        got = trainer.train_dataset[i]
        p = len(got["prompt_ids"])
        assert rec.prompt_tokens == p
        assert rec.chosen_completion_tokens == len(got["chosen_ids"])
        assert rec.rejected_completion_tokens == len(got["rejected_ids"])
        assert rec.chosen_total_tokens == p + len(got["chosen_ids"])
        assert rec.rejected_total_tokens == p + len(got["rejected_ids"])


@pytest.mark.parametrize("policy", [OMIT, KEEP])
def test_dpo_explicit_prompt_and_collator_shape(tiny_dir, mimo, tmp_path, policy) -> None:
    rows = [
        {
            "prompt": trl_prompt(r, policy),
            "chosen": [a(r["chosen"])],
            "rejected": [a(r["rejected"])],
        }
        for r in RAW
    ]
    trainer = DPOTrainer(
        model=tiny_dir,
        args=args(DPOConfig, tmp_path, max_length=None),
        train_dataset=datasets.Dataset.from_list(rows),
        processing_class=mimo,
    )
    mapping = PREF.model_copy(update={"empty_system_policy": policy})
    adapter = get_adapter(Objective.DPO, handle(mimo), mapping, empty_system_policy=policy)
    ours = [adapter.process(r, f"train:{i}") for i, r in enumerate(RAW)]
    assert_dpo_matches(trainer, ours)
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(len(RAW))])
    plan = plan_batches(table(ours), resolved(Objective.DPO, len(RAW), None), seed=42)
    worst = plan.worst_case
    assert tuple(batch["input_ids"].shape) == (worst.sequences_per_forward, worst.padded_length)
    assert worst.token_slots == batch["input_ids"].numel()  # 2B x max over both branches


def test_dpo_implicit_prompt_message_lists(tiny_dir, mimo, tmp_path) -> None:
    raw = [
        {"c": [u("질문"), a("좋은 답")], "r": [u("질문"), a("나쁜 답 " * 30)]},
        {"c": [s("규칙"), u("q"), a("x")], "r": [s("규칙"), u("q"), a("y")]},
        {"c": [u("same"), a("same")], "r": [u("same"), a("same")]},  # extract_prompt quirk
    ]
    trainer = DPOTrainer(
        model=tiny_dir,
        args=args(DPOConfig, tmp_path, max_length=None),
        train_dataset=datasets.Dataset.from_list(
            [{"chosen": r["c"], "rejected": r["r"]} for r in raw]
        ),
        processing_class=mimo,
    )
    adapter = get_adapter(Objective.DPO, handle(mimo), ColumnMapping(chosen="c", rejected="r"))
    assert_dpo_matches(trainer, [adapter.process(r, f"train:{i}") for i, r in enumerate(raw)])


def test_dpo_plain_strings(tiny_dir, plain, tmp_path) -> None:
    raw = [{"p": "Q:", "c": " yes", "r": " no</s>"}, {"p": "코드?", "c": "```x```", "r": "z" * 700}]
    trainer = DPOTrainer(
        model=tiny_dir,
        args=args(DPOConfig, tmp_path, max_length=None),
        train_dataset=datasets.Dataset.from_list(
            [{"prompt": r["p"], "chosen": r["c"], "rejected": r["r"]} for r in raw]
        ),
        processing_class=plain,
    )
    adapter = get_adapter(
        Objective.DPO, handle(plain), ColumnMapping(prompt="p", chosen="c", rejected="r")
    )
    assert_dpo_matches(trainer, [adapter.process(r, f"train:{i}") for i, r in enumerate(raw)])


@pytest.mark.parametrize("objective", [Objective.SFT, Objective.DPO])
def test_one_step_trains_on_the_planned_untruncated_shape(
    tiny_dir, mimo, tmp_path, objective
) -> None:
    """One real optimizer step with max_length=None (trl-sft-dpo.md §11.2): the batch the model
    receives is exactly our worst-case shape (SFT B x T, DPO 2B x T) and longer than TRL's
    default max_length of 1024, so nothing was truncated on the way."""
    kw = {
        "max_length": None,
        "max_steps": 1,
        "per_device_train_batch_size": len(RAW),
        "train_sampling_strategy": "sequential",
    }
    prompts = [trl_prompt(r, OMIT) for r in RAW]
    trainer: SFTTrainer | DPOTrainer
    if objective is Objective.SFT:
        rows = [
            {"prompt": p, "completion": [a(r["chosen"])]} for p, r in zip(prompts, RAW, strict=True)
        ]
        trainer = SFTTrainer(
            model=tiny_dir,
            args=args(SFTConfig, tmp_path, **kw),
            train_dataset=datasets.Dataset.from_list(rows),
            processing_class=mimo,
        )
    else:
        rows = [
            {"prompt": p, "chosen": [a(r["chosen"])], "rejected": [a(r["rejected"])]}
            for p, r in zip(prompts, RAW, strict=True)
        ]
        trainer = DPOTrainer(
            model=tiny_dir,
            args=args(DPOConfig, tmp_path, **kw),
            train_dataset=datasets.Dataset.from_list(rows),
            processing_class=mimo,
        )
    seen: list[tuple[int, ...]] = []
    original = trainer.compute_loss

    def spy(model: Any, inputs: dict[str, Any], *rest: Any, **kwargs: Any) -> Any:
        seen.append(tuple(inputs["input_ids"].shape))
        return original(model, inputs, *rest, **kwargs)

    trainer.compute_loss = spy  # type: ignore[method-assign]
    trainer.train()
    adapter = get_adapter(objective, handle(mimo), PREF)
    ours = [adapter.process(r, f"train:{i}") for i, r in enumerate(RAW)]
    worst = plan_batches(table(ours), resolved(objective, len(RAW), None), seed=42).worst_case
    assert seen == [(worst.sequences_per_forward, worst.padded_length)]
    assert worst.padded_length > 1024


# ---------------------------------------------------------------- GRPO


def grpo_trainer(tiny_dir: str, tok: Any, out: Path, prompts: list[Any], **kw: Any) -> GRPOTrainer:
    return GRPOTrainer(
        model=tiny_dir,
        reward_funcs=lambda prompts, completions, **_: [0.0] * len(prompts),
        args=args(GRPOConfig, out, num_generations=2, max_completion_length=8, **kw),
        train_dataset=datasets.Dataset.from_list([{"prompt": p} for p in prompts]),
        processing_class=tok,
    )


@pytest.mark.parametrize("policy", [OMIT, KEEP])
@pytest.mark.parametrize("template_kwargs", [{}, {"enable_thinking": False}])
def test_grpo_prompt_tokenization(tiny_dir, mimo, tmp_path, policy, template_kwargs) -> None:
    prompts = [trl_prompt(r, policy) for r in RAW]
    trainer = grpo_trainer(tiny_dir, mimo, tmp_path, prompts, chat_template_kwargs=template_kwargs)
    trl_ids, _, _ = trainer._tokenize_prompts(prompts)
    mapping = PREF.model_copy(update={"empty_system_policy": policy})
    adapter = get_adapter(
        Objective.GRPO,
        handle(mimo),
        mapping,
        template_kwargs=template_kwargs,
        empty_system_policy=policy,
    )
    ours = [adapter.process(r, f"train:{i}") for i, r in enumerate(RAW)]
    assert [r.prompt_tokens for r in ours] == [len(ids) for ids in trl_ids]


def test_grpo_plain_prompts(tiny_dir, plain, tmp_path) -> None:
    prompts = ["Q: 2+2?", "코드 작성", "x" * 500]
    trainer = grpo_trainer(tiny_dir, plain, tmp_path, prompts)
    trl_ids, _, _ = trainer._tokenize_prompts(prompts)
    adapter = get_adapter(Objective.GRPO, handle(plain), ColumnMapping(prompt="p"))
    ours = [adapter.process({"p": p}, f"train:{i}") for i, p in enumerate(prompts)]
    assert [r.prompt_tokens for r in ours] == [len(ids) for ids in trl_ids]


def grpo_resolved(
    *, update: int, spg: int, accumulation: int, budget: int, pad: int | None
) -> ResolvedConfig:
    grpo = GrpoResolved(
        num_generations=2,
        generation_batch_size=update * spg,
        steps_per_generation=spg,
        num_iterations=1,
        completion_budgets=[budget],
        budget_explicit=True,
        beta=0.0,
        reference_needed=False,
        reward_kind=RewardKind.CPU_RULE,
        rollout_backend=RolloutBackend.TRANSFORMERS_SHARED_POLICY,
        live_sequences=update * spg,
        update_microbatch=update,
        accumulation=accumulation,
    )
    base = resolved(Objective.GRPO, update, pad)
    return base.model_copy(update={"accumulation": accumulation, "grpo": grpo})


@pytest.mark.parametrize("pad", [None, 64])
def test_grpo_update_forward_matches_the_planned_scenario(tiny_dir, mimo, tmp_path, pad) -> None:
    """Two prompts of different lengths in one generation batch (G=2, B=2, spg=2 -> U=2): every
    update microbatch is as wide as the generation-batch-wide prompt maximum plus the completion
    width, both rounded by pad_to_multiple_of, and the LM head keeps L + 1 positions per sequence
    (docs/research/trl-grpo.md §6.4, R2). min_new_tokens pins L to the budget."""
    budget = 8
    raw = [RAW[0], RAW[3]]
    prompts = [trl_prompt(r, OMIT) for r in raw]
    trainer = grpo_trainer(
        tiny_dir,
        mimo,
        tmp_path,
        prompts,
        steps_per_generation=2,
        gradient_accumulation_steps=2,
        max_steps=1,
        pad_to_multiple_of=pad,
        generation_kwargs={"min_new_tokens": budget},
    )
    calls: list[dict[str, tuple[int, ...]]] = []
    original = trainer._compute_loss

    def spy(model: Any, inputs: dict[str, Any]) -> Any:
        call = {
            "prompt": tuple(inputs["prompt_ids"].shape),
            "completion": tuple(inputs["completion_ids"].shape),
        }

        def hook(module: Any, args_: Any, kwargs: Any, output: Any) -> None:
            call["logits"] = tuple(output.logits.shape)

        handle_ = model.register_forward_hook(hook, with_kwargs=True)
        try:
            return original(model, inputs)
        finally:
            handle_.remove()
            calls.append(call)

    trainer._compute_loss = spy  # type: ignore[method-assign]
    trainer.train()
    adapter = get_adapter(Objective.GRPO, handle(mimo), PREF)
    ours = [adapter.process(r, f"train:{i}") for i, r in enumerate(raw)]
    assert len({r.prompt_tokens for r in ours}) == 2  # the shorter prompt is padded up
    config = grpo_resolved(update=2, spg=2, accumulation=2, budget=budget, pad=pad)
    plan = plan_batches(table(ours), config, seed=42)
    assert plan.grpo is not None and plan.grpo.unique_prompts_per_generation == 2
    shape = plan.scenarios[0]
    assert shape.prompt_length is not None and shape.completion_length is not None
    vocab = len(mimo)
    expected = {
        "prompt": (shape.sequences_per_forward, shape.prompt_length),
        "completion": (shape.sequences_per_forward, shape.completion_length),
        "logits": (shape.sequences_per_forward, shape.completion_length + 1, vocab),
    }
    assert len(calls) == 2 and all(call == expected for call in calls)
    assert shape.logits_positions == shape.sequences_per_forward * (shape.completion_length + 1)
    assert shape.padded_length == shape.prompt_length + shape.completion_length


# ---------------------------------------------------------------- TRL defaults (regression)


def test_trl_defaults_truncate_and_drop(tiny_dir, mimo, tmp_path) -> None:
    """Documents what our exported config must override (fails loudly on a TRL upgrade)."""
    long_prompt = {"prompt": [u("x" * 1200)], "chosen": [a("a")], "rejected": [a("b")]}
    rows = [long_prompt] + [
        {"prompt": trl_prompt(r, OMIT), "chosen": [a(r["chosen"])], "rejected": [a(r["rejected"])]}
        for r in RAW
    ]
    dpo = DPOTrainer(
        model=tiny_dir,
        args=args(DPOConfig, tmp_path / "dpo"),
        train_dataset=datasets.Dataset.from_list(rows),
        processing_class=mimo,
    )
    assert dpo.args.max_length == 1024 and len(dpo.train_dataset) == len(RAW)  # silently dropped
    longest = max(range(len(RAW)), key=lambda i: len(dpo.train_dataset[i]["rejected_ids"]))
    assert dpo.data_collator([dpo.train_dataset[longest]])["input_ids"].shape[1] == 1024
    sft_rows = [{"messages": [u("x" * 1200), a("a")]}] + [{"messages": c} for c in CONVS]
    sft = SFTTrainer(
        model=tiny_dir,
        args=args(SFTConfig, tmp_path / "sft", assistant_only_loss=True),
        train_dataset=datasets.Dataset.from_list(sft_rows),
        processing_class=mimo,
    )
    assert len(sft.train_dataset) == len(CONVS)  # fully masked after truncation -> dropped
