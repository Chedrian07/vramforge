"""Plain helpers for the architecture adapter tests (import as `arch_helpers`)."""

from __future__ import annotations

from typing import Any

from vramforge_estimator.schemas import (
    EffectiveDtypes,
    LoraResolved,
    Objective,
    OptimizerResolved,
    QuantizationResolved,
    ResolvedConfig,
    Strategy,
    WorkspaceAssumptions,
)


def make_cfg(
    *,
    objective: Objective = Objective.SFT,
    strategy: Strategy = Strategy.QLORA,
    scope: str = "full_checkpoint",
    load: str = "bfloat16",
    compute: str = "bfloat16",
    adapter: str | None = None,
    quant: bool | None = None,
    double_quant: bool = True,
    skip: list[str] | None = None,
    targets: list[str] | None = None,
    r: int = 16,
    alpha: float = 32,
    dropout: float = 0.0,
    rank_pattern: dict[str, int] | None = None,
    modules_to_save: list[str] | None = None,
    bias: str = "none",
    dora: bool = False,
    gc: bool = True,
    paths: dict[str, str] | None = None,
    pad_to_multiple_of: int | None = None,
    use_cache: bool = False,
    upcast: list[str] | None = None,
    **extra: Any,
) -> ResolvedConfig:
    """A ResolvedConfig like the resolver would produce for TRL 1.14.1 (bf16 load preset)."""
    quant = strategy is Strategy.QLORA if quant is None else quant
    if adapter is None:
        adapter = "bfloat16" if quant else ("float32" if strategy is Strategy.LORA else load)
    lora = None
    if strategy is not Strategy.FULL:
        lora = LoraResolved(
            r=r,
            alpha=alpha,
            dropout=dropout,
            target_module_patterns=[],
            target_modules=list(targets or []),
            modules_to_save=list(modules_to_save or []),
            rank_pattern=dict(rank_pattern or {}),
            bias=bias,
            use_dora=dora,
        )
    return ResolvedConfig(
        objective=objective,
        strategy=strategy,
        loading_scope=scope,  # type: ignore[arg-type]
        load_dtype=load,
        effective_dtypes=EffectiveDtypes(
            weights_nonquantized=load,
            compute=compute,
            adapter=adapter,
            gradient=adapter,
            optimizer_state=adapter,
            logits="float32",
            loss="float32",
            kv_cache=load,
            recurrent_state="float32",
        ),
        quantization=QuantizationResolved(
            enabled=quant,
            method="bnb_nf4" if quant else None,
            double_quant=double_quant,
            blocksize=64 if quant else None,
            nested_blocksize=256 if quant else None,
            quant_storage_dtype="uint8" if quant else None,
            compute_dtype="bfloat16" if quant else None,
            skip_module_patterns=list(skip if skip is not None else (["lm_head"] if quant else [])),
        ),
        upcast_to_fp32_patterns=list(upcast or []),
        lora=lora,
        optimizer=OptimizerResolved(
            name="adamw_torch_fused", states_per_param=2, state_dtype=adapter, fused=True
        ),
        microbatch=1,
        accumulation=1,
        pad_to_multiple_of=pad_to_multiple_of,
        gradient_checkpointing=gc,
        checkpointing_granularity="per_decoder_layer" if gc else "none",
        attention_path_by_layer_type=dict(
            paths
            if paths is not None
            else {"full_attention": "sdpa", "linear_attention": "torch_fallback"}
        ),
        loss_path="chunked_nll",
        use_cache_during_training=use_cache,
        workspace=WorkspaceAssumptions(
            cuda_context_bytes=(300 * 2**20, 2**30),
            library_workspace_bytes=(0, 2**28),
            allocator_slack_fraction=(0.0, 0.1),
        ),
        **extra,
    )


def by_name(allocs: list[Any]) -> dict[str, Any]:
    return {a.name: a for a in allocs}
