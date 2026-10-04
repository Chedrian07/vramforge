<!-- Generated from the profile registry. Do not edit by hand. Regenerate with:
     uv run --no-sync python -m vramforge_estimator.compatibility.support_matrix --write docs/support-matrix.md
     tests/unit/compatibility/test_support_matrix.py fails when this file is stale. -->

# 지원 매트릭스 (Support matrix)

objective × architecture × strategy × backend 조합의 지원 등급입니다 (plan.md §11.4, §20.1). 값은 `profiles/`의 registry에서 생성하며, 계산 방법은 [methodology.md](methodology.md)에 있습니다.

- `analytic`: 명시적 allocation·workspace 가정을 가진 정적 추정 (GPU 실측 아님).
- `calibrated`, `measured`: 아직 등록된 profile이 없습니다 (GPU 검증 M5, `profiles/calibrated/README.md`).
- `unsupported`: 지원하지 않는 조합. 숫자 대신 원인을 반환합니다.
- readiness `ready`는 backend 조합 기준입니다. 요청 단위 조건(예: GRPO reward 미지정)은 결과를 `conditional`로 바꿉니다.
- 등록되지 않은 구조(adapter 없음)는 metadata만 반환하며 메모리 수치를 만들지 않습니다.

## 조합별 등급

셀 값: `등급 / readiness`.

| objective | architecture adapter | profile | environment | full | lora | qlora |
|---|---|---|---|---|---|---|
| sft | `dense_decoder` | dense-decoder@1.0.0 | cuda-trl-1.14.1 | analytic / ready | analytic / ready | analytic / ready |
| sft | `qwen3_5_hybrid` | qwen3_5-hybrid@1.0.0 | cuda-trl-1.14.1 | analytic / ready | analytic / ready | analytic / ready |
| dpo | `dense_decoder` | dense-decoder@1.0.0 | cuda-trl-1.14.1 | analytic / ready | analytic / ready | analytic / ready |
| dpo | `qwen3_5_hybrid` | qwen3_5-hybrid@1.0.0 | cuda-trl-1.14.1 | analytic / ready | analytic / ready | analytic / ready |
| grpo | `dense_decoder` | dense-decoder@1.0.0 | cuda-trl-1.14.1 | analytic / ready | analytic / ready | analytic / ready |
| grpo | `qwen3_5_hybrid` | qwen3_5-hybrid@1.0.0 | cuda-trl-1.14.1 | analytic / ready | analytic / ready | analytic / ready |

## 학습 환경

### `cuda-trl-1.14.1` (version 1.0.0)

단일 NVIDIA GPU, PyTorch 2.14.1 + Transformers 5.18.0 + TRL 1.14.1 학습 환경

| 패키지 | 버전 |
|---|---|
| accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |
| datasets | 5.0.1 |
| huggingface-hub | 1.33.0 |
| peft | 0.21.2 |
| safetensors | 0.8.0 |
| tokenizers | 0.23.2 |
| torch | 2.14.1 |
| transformers | 5.18.0 |
| trl | 1.14.1 |

- Python: 3.12, platform: linux-cuda, CUDA: 미고정 (GPU 보정 M5에서 고정), NVIDIA driver: 미고정 (GPU 보정 M5에서 고정)
- `dependency_lock_digest`: `sha256:ccd37fedde9c120a68463ded39d0925353779d48665408c54869ea67b210cc87`
- 학습 컨테이너 image digest: 없음 (미배포)
- 설치된 선택 kernel: 없음
- 설치되지 않은 패키지: flash-linear-attention, causal-conv1d, liger-kernel, flash-attn, kernels, vllm, pillow, torchvision
- linear attention 경로: `torch_fallback`, log-prob kernel: `trl_triton_fused`
- liger-kernel 미설치: use_liger_kernel=True는 TRL에서 ImportError (docs/research/trl-sft-dpo.md §1.6).
- Pillow/torchvision 미설치: VLM checkpoint는 processing_class로 AutoTokenizer를 명시해야 Trainer가 생성됨 (docs/research/trl-sft-dpo.md §3).

## Profile 상세

### dense-decoder (1.0.0) — `dense_decoder`

Dense decoder (self-attention + MLP), TRL 1.14.1 SFT/DPO/GRPO 정적(analytic) profile

- 환경: `cuda-trl-1.14.1`
- model types (참고용, 판정은 구조 기반): llama, mistral, qwen2, qwen3
- architectures (참고용): LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM
- trainer adapters: dpo → `trl-1.14.1-dpo`, grpo → `trl-1.14.1-grpo`, sft → `trl-1.14.1-sft`
- 로딩: `trl_model_id`, 클래스 = `config.architectures[0]`, 기본 범위 `full_checkpoint` (TRL은 config.architectures[0] 클래스로 checkpoint 전체를 로드합니다.)
  - text_only 검증 여부: 아니오 — 비전 모듈이 있는 checkpoint의 text-only 로딩은 이 profile에서 검증하지 않았습니다. 비전 모듈이 없으면 full_checkpoint와 같습니다.
- load dtype 기본값: `bfloat16`
  - `bfloat16`: 제품 기본값. 내보내는 설정에 model_init_kwargs.dtype=bfloat16을 고정합니다.
  - `float32`: TRL 1.14.1 기본값(model_init_kwargs.dtype 미지정). 비양자화 모듈·residual·rollout KV cache가 fp32가 됩니다.
- device_map `auto` 예산 계수: quantized 0.81, dense 0.9. 단일 GPU, max_memory 미지정: get_balanced_memory 0.9 x bnb 4-bit 0.90 (docs/research/loading-quantization-peft.md §4.5).
- 4-bit preset: bitsandbytes nf4/fp4 (기본 nf4), double quant on, compute `bfloat16`, storage `uint8`, blocksize 64/256, 제외 모듈 lm_head. 정확히 nn.Linear인 모듈만 4-bit로 바뀌고 embedding·norm·lm_head는 load dtype을 유지합니다 (docs/research/loading-quantization-peft.md §Q2).
- attention 경로: `full_attention` → `sdpa` (지원: full_attention: sdpa, eager)
- loss 경로 dpo: 기본 `trl_fused_logprob`, 지원 trl_fused_logprob. 전 위치 fp32 logits + TRL Triton fused log-prob kernel (chunked 경로는 liger-kernel 필요).
- loss 경로 grpo: 기본 `trl_fused_logprob`, 지원 trl_fused_logprob. completion 위치(L+1) fp32 logits + TRL Triton fused log-prob kernel.
- loss 경로 sft: 기본 `trl_chunked_nll`, 지원 trl_chunked_nll, hf_ce. chunked_nll: lm_head를 256개 valid 위치씩 계산해 (B,T,V) logits가 없습니다. hf_ce(loss_type=nll)는 fp32 전체 logits를 만듭니다.
- gradient checkpointing: 기본 on, `per_decoder_layer`, use_reentrant=False. TRL 기본 gradient_checkpointing=True, transformers 5.18 기본 use_reentrant=False, every_n_layers=1.
- DPO reference 자동 선택: peft → `frozen_base_switch`, full → `standalone_model`
- GRPO rollout backend: transformers_shared_policy
- batch preset (microbatch / accumulation): dpo 1/8, grpo 1/4, sft 1/8
- calibration: 없음 (analytic만)
- 지원 하드웨어: compute capability ≥ 8.0. bf16 autocast와 flash/memory-efficient SDPA 경로를 가정합니다. compute capability 8.0 미만 GPU는 SDPA math backend(O(T²))로 바뀔 수 있어 이 profile로 계산하지 않습니다 (docs/research/architecture-memory.md 구현 시사점 G).

| optimizer | state | 비고 |
|---|---|---|
| `adamw_torch` | 2 × param dtype | foreach 경로: step에서 2차 moment 크기의 임시 목록(_foreach_sqrt)을 만듭니다. |
| `adamw_torch_fused` | 2 × param dtype (fused) | transformers 5.18 기본 optim. tensor마다 device step scalar. |
| `adamw_8bit` | 2 × 8-bit | numel >= 4096이면 2.03125 B/param, 작은 tensor와 nn.Embedding은 fp32 8 B/param. |
| `paged_adamw_8bit` | 2 × 8-bit (paged) | paged managed memory도 VRAM에 포함합니다 (paging 절감은 확정치로 쓰지 않음). |

| objective | strategy | 등급 | readiness | 비고 |
|---|---|---|---|---|
| sft | full | analytic | ready | lm_head를 학습하면 chunked_nll backward에서 [V,H] weight grad 3벌이 공존합니다. |
| sft | lora | analytic | ready | 비양자화 LoRA adapter는 fp32 (PEFT autocast_adapter_dtype). |
| sft | qlora | analytic | ready | NF4 + double quant, compute bf16, lm_head 비양자화, adapter bf16 (TRL QLoRA 준비). |
| dpo | full | analytic | ready | reference 기본값은 두 번째 전체 모델(standalone_model)입니다. |
| dpo | lora | analytic | ready | reference 기본값은 adapter를 끈 같은 모델(frozen_base_switch)입니다. |
| dpo | qlora | analytic | ready | reference 기본값은 adapter를 끈 같은 4-bit 모델(frozen_base_switch)입니다. |
| grpo | full | analytic | ready | beta != 0이면 reference 전체 모델이 추가로 상주합니다. reward가 없으면 조건부 결과입니다. |
| grpo | lora | analytic | ready | Transformers 공유 policy rollout만 지원합니다. reward가 없으면 조건부 결과입니다. |
| grpo | qlora | analytic | ready | Transformers 공유 policy rollout만 지원합니다. reward가 없으면 조건부 결과입니다. |

| 지원하지 않는 옵션 | 이유 |
|---|---|
| `training.packing` | packing은 max_length 정수가 필수이고 bfd(절단)·bfd_split(분할)·wrapped(중간 절단) 모두 엄격 무절단 계약을 위반합니다. |
| `training.offload` | 단일 GPU TRL 경로에는 parameter/optimizer offload가 없고 activation offloading 메모리 모델은 구현되지 않았습니다. |
| `training.compile` | torch.compile 메모리 profile이 없습니다. |
| `training.loss_kernel=liger` | liger-kernel이 학습 환경에 없습니다 (TRL ImportError). |
| `training.loss_kernel=chunked (dpo, grpo)` | DPO/GRPO의 chunked log-prob 경로는 use_liger_kernel=True와 liger-kernel이 필요합니다. |
| `training.attention_backend=flash_attention_2` | flash-attn이 학습 환경에 없습니다. |
| `training.lora.use_dora` | DoRA의 forward별 dequant·eye(in) 임시값 메모리 모델이 없습니다. |
| `training.precision=fp16\|fp32` | bf16 native AMP 이외 경로의 dtype 규칙은 모델링하지 않았습니다. |
| `training.load_dtype=float16` | float16 로드는 bf16 autocast와 섞이는 경로를 검증하지 않았습니다. |
| `training.quantization.compute_dtype=float32\|float16` | bnb compute fp32는 Linear4bit 임시 메모리가 약 3배인 별도 profile입니다 (미구현). |
| `grpo.rollout_backend=vllm_colocate\|vllm_server` | 별도 vLLM 엔진의 가중치·KV pool·CUDA graph 메모리를 모델링하지 않았습니다 (M6). |
| `dpo.loss_type=sft` | chosen 위치 cross-entropy의 추가 logits·log_softmax 버퍼를 모델링하지 않았습니다. |
| `training.num_devices>1` | 분산 topology adapter가 없습니다 (총 용량을 GPU 수로 나누지 않습니다). |
| `model.loading_scope=text_only` | 비전 타워가 있는 checkpoint의 text-only 로딩은 검증하지 않았습니다. |

| fallback 규칙 | 계산에 쓰는 경로 |
|---|---|
| `grpo.max_live_sequences` | Transformers 생성 경로는 device의 generation batch 전체(C)를 한 번에 생성하므로 C로 계산하고 요청 무효로 기록합니다. |
| `training.lora.dropout (dpo)` | TRL DPOConfig.disable_dropout=True가 LoRA dropout을 포함한 모든 dropout을 0으로 만들므로 0으로 계산하고 요청 무효로 기록합니다. |
| `training.template` | chat template이 쓰지 않는 template 옵션은 무시되는 그대로 계산하고 요청 무효로 기록합니다. |
| `training.loss_kernel (sft)` | chunked_nll을 쓸 수 없는 조합(lm_head LoRA)도 nll로 자동 전환하지 않고 차단합니다 (logits 메모리가 크게 늘어남). |
| `unsupported_options` | 실행 경로나 메모리 모델이 없는 옵션을 조용히 무시한 채 절감량을 유지하지 않고, 수치 대신 차단 사유를 반환합니다 (plan §11.3). |

| workspace 가정 (ASSUMPTION) | low | high | 근거 |
|---|---|---|---|
| CUDA context | 307 MiB (322,122,547 B) | 1.00 GiB (1,073,741,824 B) | docs/research/loading-quantization-peft.md §Q10.2 |
| library workspace | 4 MiB (4,194,304 B) | 128 MiB (134,217,728 B) | docs/research/loading-quantization-peft.md §Q10.2 |
| allocator slack (할당량 대비) | 5% | 15% | 가정 (docs/research/loading-quantization-peft.md §Q10.2: 단편화 미정) |

### qwen3_5-hybrid (1.0.0) — `qwen3_5_hybrid`

Qwen3.5 hybrid decoder (linear attention + full attention), TRL 1.14.1 SFT/DPO/GRPO 정적(analytic) profile

- 환경: `cuda-trl-1.14.1`
- model types (참고용, 판정은 구조 기반): qwen3_5, qwen3_5_text
- architectures (참고용): Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM
- trainer adapters: dpo → `trl-1.14.1-dpo`, grpo → `trl-1.14.1-grpo`, sft → `trl-1.14.1-sft`
- 로딩: `trl_model_id`, 클래스 = `config.architectures[0]`, 기본 범위 `full_checkpoint` (TRL은 config.architectures[0] 클래스로 checkpoint 전체를 로드합니다.)
  - `Qwen3_5ForConditionalGeneration` → `full_checkpoint`: TRL create_model_from_path가 config.architectures[0] 클래스로 로드하므로 텍스트만 학습해도 비전 타워가 상주합니다 (docs/research/loading-quantization-peft.md §Q1.3).
  - `Qwen3_5ForCausalLM` → `full_checkpoint`: 텍스트 전용 checkpoint는 전체가 텍스트 디코더입니다.
  - text_only 검증 여부: 아니오 — text-only loader(Qwen3_5ForCausalLM 객체를 직접 넘기는 경로)는 TRL 문자열 진입 경로가 아니며 이 profile에서 검증하지 않았습니다.
- load dtype 기본값: `bfloat16`
  - `bfloat16`: 제품 기본값. 내보내는 설정에 model_init_kwargs.dtype=bfloat16을 고정합니다.
  - `float32`: TRL 1.14.1 기본값(model_init_kwargs.dtype 미지정). 비양자화 모듈·residual·rollout cache가 fp32가 됩니다.
- device_map `auto` 예산 계수: quantized 0.81, dense 0.9. 단일 GPU, max_memory 미지정: get_balanced_memory 0.9 x bnb 4-bit 0.90 (docs/research/loading-quantization-peft.md §4.5).
- 4-bit preset: bitsandbytes nf4/fp4 (기본 nf4), double quant on, compute `bfloat16`, storage `uint8`, blocksize 64/256, 제외 모듈 lm_head. nn.Linear만 4-bit로 바뀌고(비전 Linear 포함) embedding·norm·conv1d·A_log·dt_bias·lm_head는 load dtype을 유지합니다 (docs/research/loading-quantization-peft.md §Q2).
- attention 경로: `full_attention` → `sdpa`, `linear_attention` → `torch_fallback` (지원: full_attention: sdpa, eager; linear_attention: torch_fallback)
- loss 경로 dpo: 기본 `trl_fused_logprob`, 지원 trl_fused_logprob. 전 위치 fp32 logits + TRL Triton fused log-prob kernel (chunked 경로는 liger-kernel 필요).
- loss 경로 grpo: 기본 `trl_fused_logprob`, 지원 trl_fused_logprob. completion 위치(L+1) fp32 logits + TRL Triton fused log-prob kernel.
- loss 경로 sft: 기본 `trl_chunked_nll`, 지원 trl_chunked_nll, hf_ce. chunked_nll: lm_head를 256개 valid 위치씩 계산해 (B,T,V) logits가 없습니다. hf_ce(loss_type=nll)는 fp32 전체 logits를 만듭니다.
- gradient checkpointing: 기본 on, `per_decoder_layer`, use_reentrant=False. TRL 기본 gradient_checkpointing=True, transformers 5.18 기본 use_reentrant=False, every_n_layers=1.
- DPO reference 자동 선택: peft → `frozen_base_switch`, full → `standalone_model`
- GRPO rollout backend: transformers_shared_policy
- batch preset (microbatch / accumulation): dpo 1/8, grpo 1/4, sft 1/8
- calibration: 없음 (analytic만)
- 지원 하드웨어: compute capability ≥ 8.0. bf16 autocast와 flash/memory-efficient SDPA 경로를 가정합니다. compute capability 8.0 미만 GPU는 SDPA math backend(O(T²))로 바뀔 수 있어 이 profile로 계산하지 않습니다 (docs/research/architecture-memory.md 구현 시사점 G).

| optimizer | state | 비고 |
|---|---|---|
| `adamw_torch` | 2 × param dtype | foreach 경로: step에서 2차 moment 크기의 임시 목록(_foreach_sqrt)을 만듭니다. |
| `adamw_torch_fused` | 2 × param dtype (fused) | transformers 5.18 기본 optim. tensor마다 device step scalar. |
| `adamw_8bit` | 2 × 8-bit | numel >= 4096이면 2.03125 B/param, 작은 tensor와 nn.Embedding은 fp32 8 B/param. |
| `paged_adamw_8bit` | 2 × 8-bit (paged) | paged managed memory도 VRAM에 포함합니다 (paging 절감은 확정치로 쓰지 않음). |

| objective | strategy | 등급 | readiness | 비고 |
|---|---|---|---|---|
| sft | full | analytic | ready | lm_head를 학습하면 chunked_nll backward에서 [V,H] weight grad 3벌이 공존합니다. 비전 타워는 학습 파라미터지만 텍스트 데이터에서는 gradient·optimizer state가 없습니다. |
| sft | lora | analytic | ready | 비양자화 LoRA adapter는 fp32 (PEFT autocast_adapter_dtype). |
| sft | qlora | analytic | ready | NF4 + double quant, compute bf16, lm_head 비양자화, adapter bf16 (TRL QLoRA 준비). |
| dpo | full | analytic | ready | reference 기본값은 두 번째 전체 모델(standalone_model)입니다. |
| dpo | lora | analytic | ready | reference 기본값은 adapter를 끈 같은 모델(frozen_base_switch)입니다. |
| dpo | qlora | analytic | ready | reference 기본값은 adapter를 끈 같은 4-bit 모델(frozen_base_switch)입니다. |
| grpo | full | analytic | ready | beta != 0이면 reference 전체 모델이 추가로 상주합니다. reward가 없으면 조건부 결과입니다. |
| grpo | lora | analytic | ready | Transformers 공유 policy rollout만 지원합니다. reward가 없으면 조건부 결과입니다. |
| grpo | qlora | analytic | ready | Transformers 공유 policy rollout만 지원합니다. reward가 없으면 조건부 결과입니다. |

| 지원하지 않는 옵션 | 이유 |
|---|---|
| `training.packing` | packing은 max_length 정수가 필수이고 bfd(절단)·bfd_split(분할)·wrapped(중간 절단) 모두 엄격 무절단 계약을 위반합니다. linear attention의 packed 경계 처리도 미확인입니다. |
| `training.offload` | 단일 GPU TRL 경로에는 parameter/optimizer offload가 없고 activation offloading 메모리 모델은 구현되지 않았습니다. |
| `training.compile` | torch.compile 메모리 profile이 없습니다. |
| `training.loss_kernel=liger` | liger-kernel이 학습 환경에 없습니다 (TRL ImportError). |
| `training.loss_kernel=chunked (dpo, grpo)` | DPO/GRPO의 chunked log-prob 경로는 use_liger_kernel=True와 liger-kernel이 필요합니다. |
| `training.attention_backend=flash_attention_2` | flash-attn이 학습 환경에 없습니다. |
| `training.linear_attention_kernel=fla` | flash-linear-attention·causal-conv1d가 없어 torch fallback으로 실행됩니다 (요청 무효로 계산). |
| `training.lora.use_dora` | DoRA의 forward별 dequant·eye(in) 임시값 메모리 모델이 없습니다. |
| `training.precision=fp16\|fp32` | bf16 native AMP 이외 경로의 dtype 규칙은 모델링하지 않았습니다. |
| `training.load_dtype=float16` | float16 로드는 bf16 autocast와 섞이는 경로를 검증하지 않았습니다. |
| `training.quantization.compute_dtype=float32\|float16` | bnb compute fp32는 Linear4bit 임시 메모리가 약 3배인 별도 profile입니다 (미구현). |
| `grpo.rollout_backend=vllm_colocate\|vllm_server` | 별도 vLLM 엔진의 가중치·KV pool·CUDA graph 메모리를 모델링하지 않았습니다 (M6). |
| `dpo.loss_type=sft` | chosen 위치 cross-entropy의 추가 logits·log_softmax 버퍼를 모델링하지 않았습니다. |
| `training.num_devices>1` | 분산 topology adapter가 없습니다 (총 용량을 GPU 수로 나누지 않습니다). |
| `model.loading_scope=text_only` | 비전 타워가 있는 checkpoint의 text-only 로딩은 검증하지 않았습니다. |

| fallback 규칙 | 계산에 쓰는 경로 |
|---|---|
| `training.linear_attention_kernel` | 설치되지 않은 kernel(fla) 요청은 실제로 실행되는 torch fallback 경로로 계산하고 요청 무효로 기록합니다. |
| `grpo.max_live_sequences` | Transformers 생성 경로는 device의 generation batch 전체(C)를 한 번에 생성하므로 C로 계산하고 요청 무효로 기록합니다. |
| `training.lora.dropout (dpo)` | TRL DPOConfig.disable_dropout=True가 LoRA dropout을 포함한 모든 dropout을 0으로 만들므로 0으로 계산하고 요청 무효로 기록합니다. |
| `training.template` | chat template이 쓰지 않는 template 옵션은 무시되는 그대로 계산하고 요청 무효로 기록합니다. |
| `training.loss_kernel (sft)` | chunked_nll을 쓸 수 없는 조합(lm_head LoRA)도 nll로 자동 전환하지 않고 차단합니다 (logits 메모리가 크게 늘어남). |
| `unsupported_options` | 실행 경로나 메모리 모델이 없는 옵션을 조용히 무시한 채 절감량을 유지하지 않고, 수치 대신 차단 사유를 반환합니다 (plan §11.3). |

| workspace 가정 (ASSUMPTION) | low | high | 근거 |
|---|---|---|---|
| CUDA context | 307 MiB (322,122,547 B) | 1.00 GiB (1,073,741,824 B) | docs/research/loading-quantization-peft.md §Q10.2 |
| library workspace | 4 MiB (4,194,304 B) | 128 MiB (134,217,728 B) | docs/research/loading-quantization-peft.md §Q10.2 |
| allocator slack (할당량 대비) | 5% | 15% | 가정 (docs/research/loading-quantization-peft.md §Q10.2: 단편화 미정) |

## GPU preset

공칭 용량입니다. nvidia-smi가 보고하는 총량과 실제 학습에 쓸 수 있는 용량(ECC·드라이버 예약·CUDA context·다른 프로세스 제외)은 더 작으므로, 가능하면 사용 가능 용량을 직접 입력하세요.

| id | 이름 | 공칭 용량 |
|---|---|---|
| `rtx-3060-12gb` | NVIDIA GeForce RTX 3060 12GB | 12 GiB |
| `rtx-3090-24gb` | NVIDIA GeForce RTX 3090 24GB | 24 GiB |
| `rtx-4060-ti-16gb` | NVIDIA GeForce RTX 4060 Ti 16GB | 16 GiB |
| `rtx-4080-16gb` | NVIDIA GeForce RTX 4080 16GB | 16 GiB |
| `rtx-4090-24gb` | NVIDIA GeForce RTX 4090 24GB | 24 GiB |
| `rtx-5090-32gb` | NVIDIA GeForce RTX 5090 32GB | 32 GiB |
| `a10-24gb` | NVIDIA A10 24GB | 24 GiB |
| `a10g-24gb` | NVIDIA A10G 24GB | 24 GiB |
| `l4-24gb` | NVIDIA L4 24GB | 24 GiB |
| `rtx-a6000-48gb` | NVIDIA RTX A6000 48GB | 48 GiB |
| `rtx-6000-ada-48gb` | NVIDIA RTX 6000 Ada 48GB | 48 GiB |
| `l40s-48gb` | NVIDIA L40S 48GB | 48 GiB |
| `a100-40gb` | NVIDIA A100 40GB | 40 GiB |
| `a100-80gb` | NVIDIA A100 80GB | 80 GiB |
| `h100-80gb` | NVIDIA H100 80GB | 80 GiB |
