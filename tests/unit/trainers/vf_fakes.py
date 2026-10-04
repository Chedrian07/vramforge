"""Test doubles for the trainer, memory and compatibility tests.

`FakeArch` implements the ArchitectureAdapter protocol with small, hand-checkable sizes so the
tests exercise the trainer procedure (phases, lifetimes, LM-head/loss buffers) independently of
the real architecture adapters. Sizes are synthetic and never real model numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from vramforge_estimator.architectures import (
    GenerationTimepoints,
    StepTimepoints,
    TrainableGroup,
)
from vramforge_estimator.schemas import (
    AllocationCategory,
    AllocationSpec,
    ArchitectureFacts,
    BatchPlan,
    BatchShape,
    ComponentParams,
    DpoResolved,
    EffectiveDtypes,
    Evidence,
    GrpoBatchPlan,
    GrpoResolved,
    LinearModule,
    LoraResolved,
    ModelComponent,
    ModelInventory,
    Objective,
    OptimizerResolved,
    QuantizationResolved,
    ReferenceStrategy,
    ResolvedConfig,
    RewardKind,
    RolloutBackend,
    SamplerPlan,
    SequenceShape,
    Strategy,
    TensorInfo,
    TensorRole,
    WorkspaceAssumptions,
)
from vramforge_estimator.units import tensor_bytes

H = 64  # hidden size
V = 1000  # vocab rows of the LM head
LAYERS = 2
WEIGHTS = 1_000_000
LOAD_TRANSIENT = 100_000

TEXT_KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
DIMS = {
    "q_proj": (H, H),
    "k_proj": (H, 32),
    "v_proj": (H, 32),
    "o_proj": (H, H),
    "gate_proj": (H, 128),
    "up_proj": (H, 128),
    "down_proj": (128, H),
}


def _linear(
    name: str, kind: str, fan_in: int, fan_out: int, comp: ModelComponent, layer: int | None
):
    return LinearModule(
        name=name,
        kind=kind,
        in_features=fan_in,
        out_features=fan_out,
        has_bias=False,
        component=comp,
        layer_index=layer,
        layer_type="full_attention" if comp is ModelComponent.TEXT else "vision",
        dtype="bfloat16",
    )


def _tensor(name: str, shape: list[int], comp: ModelComponent, role: TensorRole, module: str):
    numel = 1
    for d in shape:
        numel *= d
    return TensorInfo(
        name=name,
        dtype="bfloat16",
        shape=shape,
        numel=numel,
        nbytes=numel * 2,
        component=comp,
        role=role,
        module=module,
    )


def make_inventory(*, vision: bool = False) -> ModelInventory:
    """Two text decoder layers (+ an optional one-block vision tower)."""
    text = ModelComponent.TEXT
    linear: list[LinearModule] = []
    tensors = [
        _tensor(
            "model.embed_tokens.weight", [V, H], text, TensorRole.EMBEDDING, "model.embed_tokens"
        )
    ]
    for layer in range(LAYERS):
        tensors.append(
            _tensor(
                f"model.layers.{layer}.input_layernorm.weight",
                [H],
                text,
                TensorRole.NORM,
                f"model.layers.{layer}.input_layernorm",
            )
        )
        for kind in TEXT_KINDS:
            fan_in, fan_out = DIMS[kind]
            group = (
                "self_attn" if kind.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else "mlp"
            )
            name = f"model.layers.{layer}.{group}.{kind}"
            linear.append(_linear(name, kind, fan_in, fan_out, text, layer))
            tensors.append(
                _tensor(f"{name}.weight", [fan_out, fan_in], text, TensorRole.LINEAR_WEIGHT, name)
            )
    tensors.append(_tensor("model.norm.weight", [H], text, TensorRole.NORM, "model.norm"))
    tensors.append(_tensor("lm_head.weight", [V, H], text, TensorRole.LM_HEAD, "lm_head"))
    if vision:
        vis = ModelComponent.VISION
        for kind, (fan_in, fan_out) in {"qkv": (H, 3 * H), "proj": (H, H)}.items():
            name = f"model.visual.blocks.0.attn.{kind}"
            linear.append(_linear(name, kind, fan_in, fan_out, vis, None))
            tensors.append(
                _tensor(f"{name}.weight", [fan_out, fan_in], vis, TensorRole.LINEAR_WEIGHT, name)
            )
    params = sum(t.numel for t in tensors)
    by_comp: dict[ModelComponent, int] = {}
    for t in tensors:
        by_comp[t.component] = by_comp.get(t.component, 0) + t.numel
    facts = ArchitectureFacts(
        architectures=["TinyForConditionalGeneration" if vision else "TinyForCausalLM"],
        model_type="tiny",
        num_hidden_layers=LAYERS,
        hidden_size=H,
        intermediate_size=128,
        vocab_size=V,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        head_dim_source="explicit",
        layer_types=["full_attention"] * LAYERS,
        has_vision=vision,
        max_position_embeddings=4096,
        config_sha256="0" * 64,
    )
    return ModelInventory(
        facts=facts,
        tensors=tensors,
        linear_modules=linear,
        params_total=params,
        bytes_serialized_total=2 * params,
        by_component=[
            ComponentParams(component=c, params=n, bytes_serialized=2 * n)
            for c, n in by_comp.items()
        ],
        inventory_hash="inv_test",
    )


def text_targets(inventory: ModelInventory) -> list[str]:
    return [m.name for m in inventory.linear_modules if m.component is ModelComponent.TEXT]


def all_targets(inventory: ModelInventory) -> list[str]:
    return [m.name for m in inventory.linear_modules]


def lora_numel(inventory: ModelInventory, names: Sequence[str], r: int = 16) -> int:
    by = {m.name: m for m in inventory.linear_modules}
    return sum(r * (by[n].in_features + by[n].out_features) for n in names)


# ---------------------------------------------------------------- resolved config


def make_cfg(
    objective: Objective = Objective.SFT,
    strategy: Strategy = Strategy.QLORA,
    *,
    inventory: ModelInventory | None = None,
    targets: list[str] | None = None,
    modules_to_save: list[str] | None = None,
    load_dtype: str = "bfloat16",
    optimizer: str = "adamw_torch",
    microbatch: int = 1,
    accumulation: int = 8,
    loss_path: str | None = None,
    reference: ReferenceStrategy = ReferenceStrategy.FROZEN_BASE_SWITCH,
    precompute_batch_size: int | None = None,
    num_generations: int = 4,
    generation_batch_size: int = 4,
    steps_per_generation: int | None = None,
    num_iterations: int = 1,
    beta: float = 0.0,
    reward: RewardKind = RewardKind.UNSPECIFIED,
    budgets: tuple[int, ...] = (1024,),
    resolutions: list | None = None,
) -> ResolvedConfig:
    inv = inventory or make_inventory()
    quantized = strategy is Strategy.QLORA
    if strategy is Strategy.FULL:
        adapter = gradient = load_dtype
    elif quantized:
        adapter = gradient = "bfloat16"
    else:
        adapter = gradient = "float32"
    eight_bit = optimizer in ("adamw_8bit", "paged_adamw_8bit")
    lora = None
    if strategy is not Strategy.FULL:
        names = targets if targets is not None else text_targets(inv)
        lora = LoraResolved(
            r=16,
            alpha=32,
            dropout=0.0,
            target_module_patterns=sorted({n.rsplit(".", 1)[-1] for n in names}),
            target_modules=names,
            modules_to_save=modules_to_save or [],
        )
    if loss_path is None:
        loss_path = "trl_chunked_nll" if objective is Objective.SFT else "trl_fused_logprob"
    dpo = None
    if objective is Objective.DPO:
        dpo = DpoResolved(
            reference_strategy=reference,
            beta=0.1,
            loss_type="sigmoid",
            precompute_batch_size=precompute_batch_size,
        )
    grpo = None
    if objective is Objective.GRPO:
        spg = steps_per_generation or generation_batch_size // microbatch
        gbs = microbatch * spg
        grpo = GrpoResolved(
            num_generations=num_generations,
            generation_batch_size=gbs,
            steps_per_generation=spg,
            num_iterations=num_iterations,
            completion_budgets=list(budgets),
            budget_explicit=len(budgets) == 1,
            beta=beta,
            reference_needed=beta != 0,
            reward_kind=reward,
            rollout_backend=RolloutBackend.TRANSFORMERS_SHARED_POLICY,
            live_sequences=microbatch * spg,
            update_microbatch=microbatch,
            accumulation=accumulation,
        )
    return ResolvedConfig(
        profile_id="fake-profile",
        profile_version="0",
        environment_id="cuda-trl-1.14.1",
        architecture_adapter="fake",
        trainer_adapter=f"trl-1.14.1-{objective.value}",
        objective=objective,
        strategy=strategy,
        loading_scope="full_checkpoint",
        load_dtype=load_dtype,
        effective_dtypes=EffectiveDtypes(
            weights_nonquantized=load_dtype,
            compute="bfloat16",
            adapter=adapter,
            gradient=gradient,
            optimizer_state="uint8" if eight_bit else gradient,
            logits="float32",
            loss="float32",
            kv_cache=load_dtype,
            recurrent_state="float32",
        ),
        quantization=QuantizationResolved(
            enabled=quantized,
            method="bnb_nf4" if quantized else None,
            double_quant=quantized,
            compute_dtype="bfloat16" if quantized else None,
            skip_module_patterns=["lm_head"] if quantized else [],
        ),
        lora=lora,
        trainable_full_patterns=[".*"] if strategy is Strategy.FULL else (modules_to_save or []),
        optimizer=OptimizerResolved(
            name=optimizer,
            states_per_param=2,
            state_dtype="uint8" if eight_bit else gradient,
            eight_bit=eight_bit,
            block_size=256 if eight_bit else None,
            min_8bit_size=4096 if eight_bit else None,
            paged=optimizer.startswith("paged"),
            fused=optimizer == "adamw_torch_fused",
        ),
        microbatch=microbatch,
        accumulation=accumulation,
        gradient_checkpointing=True,
        checkpointing_granularity="per_decoder_layer",
        attention_path_by_layer_type={"full_attention": "sdpa"},
        loss_path=loss_path,
        dpo=dpo,
        grpo=grpo,
        workspace=WorkspaceAssumptions(
            cuda_context_bytes=(300, 1000),
            library_workspace_bytes=(10, 100),
            allocator_slack_fraction=(0.05, 0.15),
        ),
        resolutions=resolutions or [],
    )


# ---------------------------------------------------------------- batch plans


def sft_shape(rows: int = 1, length: int = 100) -> BatchShape:
    return BatchShape(
        name="worst_case",
        objective=Objective.SFT,
        rows_per_microbatch=rows,
        sequences_per_forward=rows,
        padded_length=length,
        token_slots=rows * length,
        logits_positions=rows * length,
        description="test",
    )


def dpo_shape(pairs: int = 1, length: int = 100) -> BatchShape:
    return BatchShape(
        name="worst_case",
        objective=Objective.DPO,
        rows_per_microbatch=pairs,
        sequences_per_forward=2 * pairs,
        padded_length=length,
        token_slots=2 * pairs * length,
        logits_positions=2 * pairs * length,
        description="test",
    )


def grpo_shape(prompt: int = 50, budget: int = 100, rows: int = 1) -> BatchShape:
    return BatchShape(
        name=f"budget_{budget}",
        objective=Objective.GRPO,
        rows_per_microbatch=rows,
        sequences_per_forward=rows,
        padded_length=prompt + budget,
        token_slots=rows * (prompt + budget),
        logits_positions=rows * (budget + 1),
        prompt_length=prompt,
        completion_length=budget,
        description="test",
    )


def make_plan(cfg: ResolvedConfig, shapes: list[BatchShape]) -> BatchPlan:
    unit = {"sft": "samples", "dpo": "pairs", "grpo": "completions"}[cfg.objective.value]
    grpo = None
    if cfg.grpo is not None:
        g = cfg.grpo
        grpo = GrpoBatchPlan(
            unique_prompts_per_generation=g.generation_batch_size // g.num_generations,
            num_generations=g.num_generations,
            live_sequences=g.live_sequences,
            update_microbatch=g.update_microbatch,
            accumulation=g.accumulation,
            generation_batch_size=g.generation_batch_size,
            steps_per_generation=g.steps_per_generation,
            num_iterations=g.num_iterations,
            completion_budgets=g.completion_budgets,
            max_prompt_length=shapes[0].prompt_length or 0,
        )
    return BatchPlan(
        objective=cfg.objective,
        unit=unit,
        microbatch=cfg.microbatch,
        accumulation=cfg.accumulation,
        effective_batch=cfg.microbatch * cfg.accumulation,
        sampler=SamplerPlan(kind="random", covers_all_rows=True),
        worst_case=shapes[0],
        scenarios=shapes if cfg.objective is Objective.GRPO else [],
        grpo=grpo,
        batch_key="bat_test",
    )


# ---------------------------------------------------------------- fake architecture adapter


def _alloc(name, category, size, live_at, *, saved=False, note=None) -> AllocationSpec:
    return AllocationSpec(
        name=name,
        category=category,
        bytes_low=size,
        bytes_high=size,
        live_at=list(live_at),
        saved_for_backward=saved,
        evidence=Evidence.ANALYTIC if size is not None else Evidence.UNKNOWN,
        note=note,
    )


@dataclass(frozen=True)
class FlaggedGroup(TrainableGroup):
    """Mimics the repository adapters: one group per tensor shape plus `receives_grad`."""

    receives_grad: bool = True
    component: str = "text"


def _flagged(groups: dict[tuple[str, str, bool], list[int]], kind: str, dtype: str):
    return [
        FlaggedGroup(name, kind, sum(sizes), dtype, len(sizes), "", executed, comp)
        for (name, comp, executed), sizes in groups.items()
    ]


@dataclass
class FakeArch:
    """Activations scale with batch x seq: saved 10 B/token, recompute 3, no-grad 7,
    KV 2 B/token/sequence, prefill working set 5."""

    adapter_id: str = "fake"
    unknown_activations: bool = False
    vision_lora_group: bool = False  # split LoRA into text + vision groups
    flagged_groups: bool = False  # per-shape groups with receives_grad (repository adapters)
    load_budget: int | None = None  # S_load reported by `loading_budget_bytes` (None: absent)
    final_hidden: bool = False  # emit the final-norm output under the "<prefix>.final_hidden" alias

    def supports(self, facts: ArchitectureFacts) -> bool:
        return True

    def loading_budget_bytes(self, inventory, cfg) -> int | None:
        return self.load_budget

    def lora_target_modules(self, inventory, target, exclude):
        if target == "all-linear":
            mods = [m for m in inventory.linear_modules if m.kind != "lm_head"]
        elif target == "auto_verified":
            mods = [m for m in inventory.linear_modules if m.component is ModelComponent.TEXT]
        else:
            names = list(target)
            mods = [
                m
                for m in inventory.linear_modules
                if m.name in names or any(m.name.endswith("." + n) for n in names)
            ]
        return [m for m in mods if not any(m.name.endswith(e) for e in exclude)]

    def resident_weights(self, inventory, cfg, live_at):
        return [_alloc("weights.base", AllocationCategory.WEIGHTS_BASE, WEIGHTS, live_at)]

    def load_transient(self, inventory, cfg, live_at):
        return [
            _alloc(
                "weights.quantize_transient",
                AllocationCategory.LOAD_TRANSIENT,
                LOAD_TRANSIENT,
                live_at,
            )
        ]

    def _flagged_groups(self, inventory, cfg):
        shapes: dict[tuple[str, str, bool], list[int]] = {}
        if cfg.strategy is Strategy.FULL:
            for t in inventory.tensors:
                comp = t.component.value
                role = "embedding:" if t.role is TensorRole.EMBEDDING else ""
                key = (f"full:{comp}:{role}{'x'.join(map(str, t.shape))}", comp, comp == "text")
                shapes.setdefault(key, []).append(t.numel)
            return _flagged(shapes, "full", cfg.load_dtype)
        by = {m.name: m for m in inventory.linear_modules}
        for n in cfg.lora.target_modules:
            m = by[n]
            comp = m.component.value
            for part, size in (("A", f"16x{m.in_features}"), ("B", f"{m.out_features}x16")):
                numel = 16 * (m.in_features if part == "A" else m.out_features)
                shapes.setdefault((f"lora:{part}:{comp}:{size}", comp, comp == "text"), []).append(
                    numel
                )
        return _flagged(shapes, "lora", cfg.effective_dtypes.adapter)

    def trainable_groups(self, inventory, cfg):
        if self.flagged_groups:
            return self._flagged_groups(inventory, cfg)
        groups: list[TrainableGroup] = []
        if cfg.strategy is Strategy.FULL:
            numel = sum(t.numel for t in inventory.tensors)
            groups.append(
                TrainableGroup("full", "full", numel, cfg.load_dtype, len(inventory.tensors))
            )
            return groups
        names = cfg.lora.target_modules
        by = {m.name: m for m in inventory.linear_modules}
        if self.vision_lora_group:
            text = [n for n in names if by[n].component is ModelComponent.TEXT]
            vision = [n for n in names if by[n].component is not ModelComponent.TEXT]
            groups.append(
                TrainableGroup(
                    "lora:text",
                    "lora",
                    lora_numel(inventory, text),
                    cfg.effective_dtypes.adapter,
                    2 * len(text),
                )
            )
            if vision:
                groups.append(
                    TrainableGroup(
                        "lora:vision",
                        "lora",
                        lora_numel(inventory, vision),
                        cfg.effective_dtypes.adapter,
                        2 * len(vision),
                    )
                )
        else:
            groups.append(
                TrainableGroup(
                    "lora",
                    "lora",
                    lora_numel(inventory, names),
                    cfg.effective_dtypes.adapter,
                    2 * len(names),
                )
            )
        for mod in cfg.lora.modules_to_save:
            ts = [t for t in inventory.tensors if t.module.endswith(mod)]
            dtype = "bfloat16" if cfg.quantization.enabled else cfg.load_dtype
            groups.append(
                TrainableGroup(
                    f"modules_to_save:{mod}",
                    "modules_to_save",
                    sum(t.numel for t in ts),
                    dtype,
                    len(ts),
                )
            )
        return groups

    def train_step_ledger(self, inventory, cfg, shape: SequenceShape, tps: StepTimepoints, prefix):
        tokens = shape.batch * shape.seq_len
        saved = None if self.unknown_activations else 10 * tokens
        extra = []
        if self.final_hidden:
            hidden = _alloc(
                f"{prefix}.final_hidden",
                AllocationCategory.SAVED_ACTIVATIONS,
                tokens * H * 2,
                [tps.forward, tps.loss],
            )
            extra.append(
                hidden.model_copy(update={"storage_alias_group": f"{prefix}.final_hidden"})
            )
        return [
            *extra,
            _alloc(
                f"{prefix}.saved",
                AllocationCategory.SAVED_ACTIVATIONS,
                saved,
                [tps.forward, tps.loss, tps.backward],
                saved=True,
                note="fake: activation profile 없음" if saved is None else None,
            ),
            _alloc(
                f"{prefix}.forward_transient",
                AllocationCategory.RECOMPUTE_WORKING_SET,
                2 * tokens,
                [tps.forward],
            ),
            _alloc(
                f"{prefix}.recompute",
                AllocationCategory.RECOMPUTE_WORKING_SET,
                3 * tokens,
                [tps.backward],
            ),
        ]

    def no_grad_forward_ledger(self, inventory, cfg, shape, live_at, prefix):
        return [
            _alloc(
                f"{prefix}.no_grad",
                AllocationCategory.RECOMPUTE_WORKING_SET,
                7 * shape.batch * shape.seq_len,
                live_at,
            )
        ]

    def generation_ledger(
        self,
        inventory,
        cfg,
        num_sequences,
        prompt_len,
        new_tokens,
        tps: GenerationTimepoints,
        prefix,
    ):
        return [
            _alloc(
                f"{prefix}.kv_prefill",
                AllocationCategory.GENERATION_CACHE,
                2 * num_sequences * prompt_len,
                [tps.prefill],
            ),
            _alloc(
                f"{prefix}.kv_final",
                AllocationCategory.GENERATION_CACHE,
                2 * num_sequences * (prompt_len + new_tokens - 1),
                [tps.decode],
            ),
            _alloc(
                f"{prefix}.prefill_working_set",
                AllocationCategory.RECOMPUTE_WORKING_SET,
                5 * num_sequences * prompt_len,
                [tps.prefill],
            ),
        ]

    def lm_head_dims(self, inventory):
        return H, V


def by_name(schedule, name: str) -> AllocationSpec:
    return next(a for a in schedule.allocations if a.name == name)


def names(schedule) -> set[str]:
    return {a.name for a in schedule.allocations}


def tp_ids(schedule) -> list[str]:
    return [t.id for t in sorted(schedule.timepoints, key=lambda t: t.order)]


__all__ = [
    "LOAD_TRANSIENT",
    "WEIGHTS",
    "FakeArch",
    "FlaggedGroup",
    "H",
    "V",
    "all_targets",
    "by_name",
    "dpo_shape",
    "grpo_shape",
    "lora_numel",
    "make_cfg",
    "make_inventory",
    "make_plan",
    "names",
    "sft_shape",
    "tensor_bytes",
    "text_targets",
    "tp_ids",
]
