# 메모리 산정 방법론

> 계산 코어(`packages/estimator`)가 실제로 쓰는 수식, 단계(phase)와 timepoint, 미상(unknown)과 제외(excluded)의 구분, 여유 정책과 적합 판정을 설명합니다.
> 구조별 가중치·activation·generation cache 식은 architecture adapter 문서 [methodology-architectures.md](methodology-architectures.md)에 있습니다.
> 지원 조합은 [support-matrix.md](support-matrix.md)(registry에서 생성), 사양은 [plan.md](../plan.md) §9–§12, 근거는 [docs/research/](research/)입니다.

| 항목 | 값 |
|---|---|
| 대상 환경 | 단일 NVIDIA GPU, `profiles/environments/cuda-trl-1.14.1.yaml` (torch 2.14.1, transformers 5.18.0, trl 1.14.1, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0) |
| trainer adapter | `trl-1.14.1-sft`, `trl-1.14.1-dpo`, `trl-1.14.1-grpo` (`vramforge_estimator.trainers`) |
| 피크 엔진 | `vramforge_estimator.memory.evaluate` |
| 근거 등급 | 모든 수치는 `analytic` (보정·실측 profile 없음, GPU 검증은 M5) |

`AllocationSpec.formula_ref`는 이 문서의 anchor(`methodology.md#<id>`)를 가리키며, 테스트가 anchor 존재를 확인합니다.

## 1. 단위와 원칙

- 내부 수치는 모두 **정수 byte**입니다. 화면의 GiB는 1 GiB = 1,073,741,824 bytes이고 계산에 다시 쓰지 않습니다.
- 계수가 소수인 식(예: 16.7 × 256 × V)은 정확한 10진 분수로 곱한 뒤 올림합니다(`ceil`).
- 크기를 정할 근거가 없으면 `null` + 사유입니다. 0으로 채우거나 임의 배율을 곱하지 않습니다.
- 모든 항목은 근거(evidence)를 가집니다: `analytic`(소스·CPU 측정으로 확인한 식), `assumption`(profile에 버전 관리되는 가정 범위), `unknown`.
- `bytes_low`/`bytes_high`는 통상/보수 시나리오입니다. 결정적인 항목은 low = high입니다. 통계적 신뢰구간이 아닙니다.

<a id="peak-evaluation"></a>
## 2. 시간축 모델과 피크

학습 step을 시간 순서의 timepoint로 나누고, 각 allocation이 살아 있는 timepoint 목록(`live_at`)을 가집니다.

```text
M(t)       = Σ_{a가 t에 생존} size(a)          # 같은 storage_alias_group은 한 번만 (max low, max high)
M_peak     = max_t M(t)                        # 서로 다른 단계의 최대값을 더하지 않음
floor(t)   = Σ 크기가 확정된 resident 항목의 low  # weights_base/adapter/other_models, gradients,
                                               # optimizer_states, master_weights
known_floor_bytes   = max_t floor(t)
scenario_low_bytes  = max_t low(t)     (미상 timepoint가 하나라도 있으면 null)
scenario_high_bytes = max_t high(t)    (미상 timepoint가 하나라도 있으면 null)
```

- `live_at`에는 timepoint id 또는 `"<PHASE>:*"`(그 단계의 모든 timepoint)를 씁니다. 없는 id를 가리키면 오류입니다.
- 피크 timepoint는 high가 가장 큰 timepoint(동률이면 앞선 것)입니다. 미상 timepoint가 있으면, 크기를 아는 부분의 합이 가장 큰 **미상 timepoint**를 피크 후보로 보여 줍니다. 미상 부분에는 상한이 없으므로 어느 미상 timepoint든 실제 피크일 수 있고, 그래서 범위는 `null`입니다.
- `peak_breakdown`은 피크 timepoint에 살아 있는 allocation **그 자체**입니다(alias group은 대표 1개). 항목 합 = `total_high` = `scenario_high_bytes`이며, 서로 다른 시점의 최대값을 쌓지 않습니다(plan §12.2).
- 단계별 피크(`PhasePeak`)는 9개 단계 모두를 보고합니다. timepoint가 없는 단계는 제외 사유(같은 이름의 `ExcludedComponent`) 또는 "해당 없음"으로 표시합니다.
- 예: 동시에 존재하지 않는 10 GiB 단계와 12 GiB 단계의 피크는 22 GiB가 아니라 12 GiB입니다(plan §19.1).

<a id="phases"></a>
## 3. 단계와 timepoint

| 단계 | timepoint | 조건 |
|---|---|---|
| `MODEL_LOAD_AND_QUANTIZE` | `policy_device_map_check`, `policy_load_peak` | 항상 |
| | `reference_device_map_check`, `reference_load_peak` | DPO `standalone_model`, GRPO β≠0이고 PEFT가 아님 |
| `REFERENCE_PRECOMPUTE` | `forward` | DPO `precomputed_log_probs` |
| `ROLLOUT_PREFILL_AND_DECODE` | `prefill`, `decode` | GRPO |
| `POLICY_FORWARD_BACKWARD` | `old_logprob` | GRPO, K mod (spg × num_iterations) ≠ 0 |
| | `reference_logprob` | GRPO, β ≠ 0 |
| `REWARD` | `score` | GRPO local reward 모델이 학습 GPU에 있을 때 |
| `POLICY_FORWARD_BACKWARD` | `forward`, (`reference_forward`), `loss`, `loss_backward`, `backward` | 항상 (`reference_forward`는 DPO `frozen_base_switch`·`standalone_model`) |
| `OPTIMIZER_STEP` | `step` | 항상 |
| `EVALUATION` | `forward` | `scope.include_evaluation` |
| `CHECKPOINT_SAVE_OR_CONSOLIDATE` | `save` | `scope.include_checkpoint_save` |
| `WEIGHT_SYNC` | — | Transformers 공유 policy 경로에는 별도 동기화가 없음 (해당 없음) |

- architecture adapter에는 `forward`(forward 끝), `loss`(LM head·loss), `backward`(첫 layer 재계산) 세 timepoint를 넘깁니다. trainer는 그 사이에 `reference_forward`와 `loss_backward`를 둡니다. adapter의 `loss` 시점에 살아 있는 allocation(저장된 activation)은 두 timepoint에서도 살아 있는 것으로 확장합니다.
- 순서는 TRL 1.14.1 실행 순서입니다. GRPO는 generate → old/ref log-prob → reward → 학습 micro-step 순입니다(docs/research/trl-grpo.md §10 검증 수정).

<a id="sft"></a><a id="dpo"></a><a id="grpo"></a>
### 3.1 수명 규칙

| allocation | 생존 구간 | 근거 |
|---|---|---|
| 정책 가중치 | 정책 load peak 이후 모든 timepoint | plan §9.1 |
| load transient | load peak만 | loading-quantization-peft §3.5 |
| adapter 가중치 (LoRA, `modules_to_save` 사본) | 정책 load peak 다음부터 끝까지 | PEFT는 Trainer `__init__`에서 로드 직후 생성 |
| gradient | accumulation > 1: `POLICY_FORWARD_BACKWARD` 전체 + `step`; = 1: `backward` + `step`. GRPO는 (spg × num_iterations) mod K ≠ 0이면 rollout·log-prob·reward 시점에도 | `zero_grad(set_to_none=True)`로 step 뒤 해제, trl-grpo V5-S |
| optimizer state | 정상 상태 step의 모든 timepoint (rollout, update, step, eval, save) — load와 precompute 제외 | 첫 step에 지연 생성 |
| activation | architecture adapter가 지정 (`forward`/`loss`/`backward`) | **accumulation 횟수를 곱하지 않음** (plan §9.4) |
| rollout cache | `prefill`, `decode` | generate가 끝나면 해제 |
| rollout buffer | 이전 generation: rollout~reward, 현재: log-prob/reward~step | trl-grpo §5.4 |
| reference 모델 가중치 | reference load peak 이후 | 별도 모델 |
| CUDA context / library workspace / allocator slack | 모든 timepoint / load 제외 / 각 timepoint | §7 |

<a id="load-phase"></a>
## 4. 모델 로딩

- 상주 가중치와 load transient는 architecture adapter가 계산합니다(4-bit payload·metadata, 비양자화 모듈, 비전 타워 포함 여부; [methodology-architectures.md](methodology-architectures.md)).
- **device_map 예산 검사**: TRL은 단일 GPU에서 `device_map="auto"`로 로드하고, `max_memory`가 없으면 transformers가 가용 메모리의 0.9배, bitsandbytes 4-bit가 다시 0.90배를 예산으로 씁니다(docs/research/loading-quantization-peft.md §4.5, V9).

```text
device_map_budget = ceil(W / f),   f = 0.81 (bnb 4-bit) | 0.9 (비양자화)
W = 상주 가중치 bytes (compute_module_sizes의 S_load 이상이라 보수적)
```

  이 값은 실제 할당이 아니라 로딩 직전 필요한 **가용 메모리 요구량**입니다. `*_device_map_check` timepoint에만 있고, 그 시점에 이미 상주한 모델(예: reference를 로드할 때의 정책)과 함께 합산됩니다. 4-bit는 부족하면 로딩이 `ValueError`로 실패하고, 비양자화 모델은 CPU로 offload됩니다.
- `standalone_model` reference와 GRPO reference 모델은 정책과 같은 `model_init_kwargs`·`quantization_config`로 정책 다음에 로드합니다.

<a id="trainable-state"></a>
## 5. 학습 상태

- architecture adapter의 `TrainableGroup`(LoRA, `modules_to_save`, full, bias)이 크기와 dtype을 정합니다. LoRA와 `modules_to_save` 사본은 adapter 가중치로 상주하고, full FT와 bias는 base 가중치 자체입니다.
- **실행되지 않는 학습 파라미터**: TRL은 비전 타워를 freeze하지 않지만 텍스트 데이터에서는 실행되지 않으므로, 그 파라미터는 가중치만 상주하고 gradient·optimizer state가 없습니다(loading-quantization-peft §Q8.4, trl-sft-dpo V-19, trl-grpo #15). 그룹을 inventory(실제 모듈 차원, component)와 대응시켜 실행 부분을 구하고, 대응하지 못하면 그룹 전체를 실행된 것으로 계산합니다(보수적, 가정으로 표시).
- LoRA 파라미터 수: `P = Σ_j r_j × (in_j + out_j)` (r_j는 PEFT `rank_pattern` 규칙, alpha와 무관).

<a id="gradients"></a>
### 5.1 Gradient

```text
G = n_exec × bytes(param dtype)
param dtype: QLoRA = bf16 (TRL이 학습 파라미터를 bf16으로 변환), LoRA = fp32 (PEFT autocast_adapter_dtype),
             Full = load dtype (master copy 없음)
```

<a id="optimizer-states"></a>
### 5.2 Optimizer state

```text
AdamW (adamw_torch, adamw_torch_fused):  S = 2 × n_exec × bytes(param dtype)
                                          fused는 tensor마다 device fp32 step scalar → + 512 B × tensors
                                          (caching allocator kMinBlockSize)
bnb 8-bit (adamw_8bit, paged_adamw_8bit): tensor별  n ≥ 4096이고 nn.Embedding이 아니면 2n + 8 × ceil(n/256)
                                          그 외 8n (fp32 state 2개); optimizer당 qmap 2,048 B
                                          tensor 구성을 모르면 [2n + 8·ceil(n/256), 8n] 범위
```

- paged state는 caching allocator 밖 managed memory일 수 있지만 VRAM에 포함합니다(paging 절감은 확정치로 쓰지 않음, loading-quantization-peft 미확정 4).

<a id="optimizer-step"></a>
### 5.3 Optimizer step 임시 메모리

- `adamw_torch`(foreach)는 step마다 `exp_avg_sq_sqrt = _foreach_sqrt(exp_avg_sq)` 목록을 만듭니다: `n_exec × bytes(param dtype)`, `OPTIMIZER_STEP`에만 생존(torch 2.14.1 `torch/optim/adam.py` `_multi_tensor_adam`).
- fused AdamW와 bnb 8-bit는 in-place로 갱신하므로 별도 목록을 두지 않습니다. gradient clipping의 tensor별 norm은 작아서 넣지 않습니다.

<a id="lm-head-input"></a>
## 6. LM head와 loss

lm_head가 학습되면(full FT, `modules_to_save`에 lm_head) 입력(최종 norm 출력)을 weight gradient용으로 저장합니다: `positions × H × bytes(residual)`. frozen lm_head는 입력을 저장하지 않습니다(architecture-memory §10.1 행 27). GRPO는 `logits_to_keep`이 hidden의 view라 `(B, P+L, H)` 전체 storage가 남습니다.

residual dtype = load dtype입니다(embedding은 autocast 대상이 아니고 Linear4bit는 입력 dtype으로 되돌림, loading-quantization-peft §Q9.3).

<a id="sft-chunked-nll"></a>
### 6.1 SFT `chunked_nll` (TRL 기본)

lm_head를 256개 valid 위치씩 checkpoint 안에서 계산하므로 `(B, T, V)` logits가 없고, loss 피크는 길이와 무관합니다(trl-sft-dpo §5.1, architecture-memory §6.3). C = 256.

| allocation | 크기 | timepoint |
|---|---|---|
| 정렬된 hidden 복사 | B·(T−1)·H·bytes(residual) | `loss`, `loss_backward` |
| reshape 복사 (B > 1) | B·(T−1)·H·bytes(residual) | `loss` |
| 정렬 index·label | 16·B·(T−1) | `loss`, `loss_backward` |
| chunk forward | [15.0, 16.7] × C × V | `loss` |
| chunk backward (lm_head frozen) | [16.0, 18.0] × C × V | `loss_backward` |
| lm_head weight-grad 누적 (lm_head 학습) | [3·b_w·V·H, 3·b_w·V·H + 18·C·V] | `loss_backward` |
| lm_head weight bf16 사본 (fp32 로드) | V·H·2 | `loss`, `loss_backward` |

- 학습되는 lm_head는 chunk마다 `[V, H]` weight grad가 out-of-place로 누적되어 3벌이 공존합니다(CPU 검증, 예시 5.75 GiB). accumulation 창의 두 번째 micro-step부터는 기존 `.grad`가 따로 있으므로 3벌 전부를 임시값으로 둡니다.
- 계수는 CPU 측정값(15.0–16.7, 16.0–17.7)과 문서의 보수 상한 18입니다. CUDA 커널 workspace는 §8의 가정에 포함됩니다.

<a id="sft-nll"></a>
### 6.2 SFT `nll` (`hf_ce`)

N = B·T (prompt·padding 위치 포함).

```text
loss          = [10, 12] × N × V   # bf16 logits 2 + fp32 복사 4 + log_softmax 4 (+ metric·출력 복사)
loss_backward = [12, 14] × N × V   # 저장된 log_softmax 4 + grad 8 (logits 참조가 남으면 14)
```

<a id="dpo-logits"></a>
### 6.3 DPO logits

collator는 chosen B행 뒤에 rejected B행을 붙여 N = 2B행을 가장 긴 branch 길이 T로 padding하고, 한 번의 forward가 모든 위치의 logits를 만듭니다. accelerate native AMP가 출력을 fp32로 바꾸고 TRL fused log-prob kernel이 이를 backward용으로 저장합니다(trl-sft-dpo §8, 구현 시사점 D).

| allocation | 크기 | timepoint |
|---|---|---|
| 정책 lm_head 출력 (bf16) | N·T·V·2 | `forward` |
| 정책 fp32 logits | N·T·V·4 | `forward`, `reference_forward`, `loss`, `loss_backward` |
| reference lm_head 출력 (bf16) | N·T·V·2 | `reference_forward` |
| reference fp32 logits | N·T·V·4 | `reference_forward`, `loss` (`_compute_loss`가 끝날 때까지 정책 logits와 공존) |
| metric boolean-index 복사 | B·(T−1)·V·4 (branch별 순차, 상한) | `loss` |
| kernel 행별 출력 | 4 × 4 × N·(T−1) | `loss`, `loss_backward` |
| kernel `grad_logits` | N·(T−1)·V·4 | `loss_backward` |

결과적으로 `reference_forward` ≈ 10·N·T·V, `loss` ≈ 8·N·T·V + metric 복사, `loss_backward` ≈ 8·N·T·V입니다.

<a id="dpo-reference"></a>
### 6.4 DPO reference 전략

| 전략 | 자동 선택 | 메모리 |
|---|---|---|
| `frozen_base_switch` | PEFT (LoRA·QLoRA) | 추가 가중치 없음. adapter를 끈 같은 모델의 no-grad forward (`reference_forward`) |
| `standalone_model` | Full FT, 또는 별도 reference 모델 지정 | 두 번째 전체 모델(정책과 같은 dtype·양자화)을 정책 다음에 로드해 계속 상주 |
| `precomputed_log_probs` | 요청할 때만 | `REFERENCE_PRECOMPUTE`: Trainer `__init__`에서 2·B_pre행 × T no-grad forward, logits는 모델 dtype(정책 wrapper 이전). optimizer state·gradient 없음. 학습 중 reference forward 없음 |

- B_pre = `precompute_batch_size` 또는 microbatch(TRL 기본), T = 최악 padding 길이(가장 긴 row가 든 batch).
- 정책과 다른 reference checkpoint는 inventory를 분석하기 전까지 크기를 알 수 없으므로 `null`입니다.
- precompute와 `sync_ref_model`, PEFT와 `sync_ref_model`은 함께 쓸 수 없습니다(요청 단계에서 차단).

<a id="grpo-rollout"></a>
### 6.5 GRPO rollout

device의 generation batch C = B_update × spg개 sequence를 `generate` 한 번으로 만들고, 같은 prompt의 G개 사본도 각자 cache를 가집니다(prefix sharing 없음, trl-grpo §4.2).

- KV cache(full attention 층)와 conv/recurrent state(linear attention 층), prefill working set은 architecture adapter의 generation ledger가 C, P, L로 계산합니다(T_kv = P + L − 1).
- step logits: `C·V·b` (b = 4 full FT(accelerate 출력 변환), PEFT는 load dtype) + 마지막 위치 fp32 복사 `C·V·4`를 `prefill`, `decode`에 둡니다.

<a id="grpo-logprob-passes"></a>
### 6.6 GRPO old/reference log-prob pass

```text
old log-prob pass  ⇔  K mod (spg × num_iterations) ≠ 0       (vLLM importance sampling은 미지원)
ref log-prob pass  ⇔  β ≠ 0   (PEFT: adapter off, full FT: reference 모델)
E = B_update × (L + 1) × V
no-grad pass logits = 10·E (C / B_update ≥ 2개 chunk) | 6·E (chunk 1개)
```

no-grad forward working set은 architecture adapter의 `no_grad_forward_ledger(B_update행, P + L)`입니다.

<a id="grpo-policy-logits"></a>
### 6.7 GRPO 정책 update

```text
forward       = 2E (bf16 lm_head 출력) + 4E (fp32)
loss          = 4E (fused kernel이 저장)
loss_backward = 4E + 4 × B_update × L × V (grad_logits) ≈ 8E
```

CUDA Triton fused kernel 경로 기준입니다(Linux torch 2.14.1은 triton을 포함, trl-grpo R4). micro-batch 폭은 generation batch 전체의 최대 P, L입니다.

<a id="grpo-rollout-buffers"></a>
### 6.8 GRPO rollout buffer

```text
buffer = C × (16·P + L × (16 + 4 × n_logp)) + 4·C     # id·mask int64, log-prob fp32, advantage
n_logp = (old pass) + (ref pass)
```

이전 generation의 `_buffered_inputs`는 다음 rollout이 끝날 때까지 남으므로 rollout~reward 구간에는 두 벌입니다.

<a id="grpo-reward"></a>
### 6.9 GRPO reward

| reward | 처리 |
|---|---|
| 미지정 | `REWARD` 제외(`GRPO_REWARD_UNSPECIFIED`), readiness `conditional`, 실행용 설정 없음 |
| CPU 규칙 / 원격 | 학습 GPU 밖으로 보고 제외. 그 자원이 0이라는 뜻이 아님. readiness `conditional` |
| local 모델, 학습 GPU | reward 가중치(전 구간)와 `C × T_reward` forward가 그 모델 inventory 없이는 `null` → 범위 `null`, 적합 판정 보류 |
| local 모델, 다른 장치 | 이 GPU 계산에서 제외, readiness `conditional` |

<a id="workspace-assumptions"></a>
## 7. Workspace 가정

수식이 없는 구성 요소는 profile에 버전 관리되는 `assumption` 범위입니다(loading-quantization-peft §Q10.2). GPU 보정(M5) 전까지 실측값이 아닙니다.

| 항목 | low | high | 생존 |
|---|---|---|---|
| CUDA context + 모듈 로드 | 0.3 GiB | 1.0 GiB | 모든 timepoint |
| cuBLAS(+Lt)·cuDNN workspace | 4 MiB | 128 MiB | load를 제외한 timepoint |
| caching allocator slack | 같은 timepoint 할당량(context 제외)의 5% | 15% | 각 timepoint (크기 미상 timepoint에는 두지 않음) |

profile의 allocator slack과 운영 margin(§9)은 별도 항목입니다(plan §10.2).

<a id="unknown-vs-excluded"></a>
## 8. 미상, 제외, 해당 없음

| 구분 | 의미 | 결과 |
|---|---|---|
| 제외 (excluded) | 범위에서 의도적으로 뺀 항목 (reward, eval, save). `ExcludedComponent`에 사유 | 단계는 `included=false`. reward 제외와, 평가 split이 있는데 평가를 뺀 경우는 readiness `conditional` |
| 미상 (unknown) | 범위 안이지만 산정 근거가 없는 항목 | `null` + 사유, `unknown_components`, 범위 `null`, 적합 판정 보류 |
| 해당 없음 | 그 학습 방식에 없는 단계 (예: SFT의 rollout) | 단계 `included=false`, "해당 없음" |

- 평가·checkpoint 저장은 기본 제외입니다. 범위에 넣으면 평가 batch 계획이 없고 저장 경로의 GPU 임시값이 조사되지 않았으므로 미상으로 계산합니다.
- 결과를 "전체 수명주기 검증 완료"로 표현하지 않습니다(plan §9.2).

<a id="margin-policy"></a>
## 9. 여유와 권장 용량

```text
planning_margin                 = max(min_bytes, ceil(scenario_high × fraction))   # 기본 2 GiB, 0.15
recommended_application_capacity = scenario_high + planning_margin
required_total_device_capacity   = recommended_application_capacity + external_reserved_bytes
```

- 정확한 10진 분수로 계산합니다(20 GiB × 0.15 = 정확히 3 GiB).
- `scenario_high`가 `null`이면 권장 용량도 `null`입니다. floor에 배율을 곱해 권장 용량을 만들지 않습니다(plan §10.2).
- 이 margin은 운영 여유 정책이지 오차 보증이 아닙니다.

<a id="hardware-fit"></a>
## 10. 적합 판정

```text
capacity = usable_bytes  또는  (device_total_bytes | GPU preset 공칭 용량) − external_reserved_bytes
utilization = (scenario_high + margin) / capacity      # 1을 넘을 수 있음
```

| 순서 | 조건 | 상태 / reason |
|---|---|---|
| 1 | 하드웨어 미선택 (`capacity_only`) 또는 용량 정보 없음 | `not_evaluated` / `not_evaluated` |
| 2 | readiness `unsupported` | `unknown` / `unsupported` |
| 3 | known floor > capacity (미상 항목이 있어도 판정) | `exceeds` / `floor_exceeds_capacity` |
| 4 | 미상 항목 때문에 high가 `null` | `unknown` / `unknown_components` |
| 5 | high > capacity | `exceeds` / `high_exceeds_capacity` |
| 6 | high ≤ capacity < high + margin | `low_margin` / `margin_insufficient` |
| 7 | high + margin ≤ capacity | `expected_fit` / `fits_with_margin` (readiness `conditional`이면 조건부라고 표시) |

GPU preset의 공칭 용량은 nvidia-smi 총량·실사용 가능량보다 큽니다. 가능하면 사용 가능 용량을 직접 입력합니다.

<a id="evidence"></a>
## 11. 근거 등급

| 수준 | 의미 | 현재 |
|---|---|---|
| `metadata_only` | 구조·inventory만, 메모리 수치 없음 | 등록되지 않은 구조 |
| `analytic` | 명시적 allocation·workspace 가정을 가진 정적 시나리오 | 등록된 모든 profile |
| `calibrated` | 등록 GPU·버전 영역에서 보정된 값 | 없음 (`profiles/calibrated/` 비어 있음) |
| `measured` | 해당 job의 단계별 관측 peak | 없음 (GPU worker 미연결) |

개별 항목의 `analytic`은 설치된 소스로 확인했거나 CPU에서 측정한 식입니다. CUDA 전용 경로(Triton kernel, caching allocator 반올림)의 차이는 §7 가정과 M5 보정의 대상입니다.

<a id="scenarios"></a>
## 12. 시나리오

- SFT·DPO: batch 계획의 구조적 최악 shape(가장 긴 row들이 같은 batch) 하나, id `default`.
- GRPO: completion budget을 지정하지 않으면 후보(기본 1,024 / 2,048 / 4,096 / 8,192)마다 시나리오를 만들고 `primary_scenario_id`는 `null`입니다. 지정하면 그 budget 하나입니다.
- 각 시나리오는 schedule의 timepoint와 allocation 전체, 장치별 결과, 권장 용량, 적합 판정, 제외 항목을 담습니다.

<a id="host-ram"></a>
## 13. RAM과 디스크

RAM·디스크도 GPU와 같은 규칙입니다: 근거 없는 부분은 `null`입니다.

| 출력 | 항목 | 식 |
|---|---|---|
| 학습 노드 RAM | 로딩 staging | 범위 안 가장 큰 tensor의 직렬화 크기 (+ load dtype 변환이 host에서 일어나면 변환본) |
| | DPO precompute fingerprint | 가장 큰 tensor fp32 크기 × 2 (`hash_module`, CPU 실측 2.00배) |
| | runtime 기본 RSS | `null` (Python·PyTorch·CUDA 라이브러리, 측정 근거 없음) |
| 분석 서버 RAM | tokenizer 파일, row별 길이 값 (8 B × row) | 하한 |
| 디스크 | 모델·데이터 다운로드(manifest 파일 크기), row-length artifact, checkpoint 1회분(학습 가중치 + optimizer state), 학습 노드 datasets cache(`null`) | 미상 항목이 있으면 total `null` |

RAM 결과의 `bytes_low`는 단계별 최대값으로 본 **확인된 하한**이고, runtime 기본 RSS가 미상이므로 `bytes_high`는 `null`입니다. 원본 데이터셋 파일 크기를 그대로 RAM 요구량으로 쓰지 않습니다(plan §10.1).

<a id="compatibility-resolution"></a>
## 14. 호환성 해석

`compatibility.resolve`는 구조로 고른 architecture adapter의 analytic profile을 적용하고, 모든 설정을 `requested → resolved (+사유)`로 기록합니다(plan §11.3). 품질에 영향을 주는 값(rank, β, generation 수, budget)은 메모리를 위해 바꾸지 않습니다.

- **load dtype**: 기본 `bfloat16`(내보내는 설정에 고정). `float32`는 TRL 기본값이며 명시적 선택지입니다.
- **로딩 범위**: TRL은 `config.architectures[0]` 클래스로 로드하므로 Qwen3.5 VLM checkpoint는 비전 타워까지 상주(`full_checkpoint`)합니다. text_only는 검증되지 않아 차단합니다.
- **effective dtype**: compute bf16, adapter·gradient·optimizer state는 위 §5, logits·loss fp32(accelerate), KV cache = load dtype, recurrent state fp32.
- **4-bit preset**: NF4 + double quant(blocksize 64/256), compute bf16, `lm_head` 제외.
- **preset**: microbatch/accumulation SFT·DPO 1/8, GRPO 1/4.
- **GRPO batch**(TRL `GRPOConfig.__post_init__`): spg = K (둘 다 미지정), spg = gbs / (B_update × W) (gbs 지정, 나누어떨어져야 함), gbs = B_update × W × spg; gbs mod G = 0, G ≥ 2, gbs와 spg 동시 지정 불가; U = gbs / G, C = B_update × spg. 예: G=4, gbs=4, B=1, K=4 → spg=4, C=4, U=1.
- **차단(오류)**: 4-bit와 strategy 모순, 다중 GPU, 엄격 모드의 packing, 실행 경로나 메모리 모델이 없는 옵션(offload, compile, liger, flash-attn, vLLM rollout, DoRA, bf16 외 precision), DPO reference 충돌, 정보가 빠진 local reward.
- **요청 무효(경고)**: 설치되지 않은 linear-attention kernel 요청(실제 실행 경로인 torch fallback으로 계산), `max_live_sequences`, template이 쓰지 않는 template 옵션.
- readiness: 차단 항목이 있으면 `unsupported`, reward가 GPU 밖이거나 미지정이면 `conditional`, 평가 split을 지정했는데 평가 단계를 계산 범위에서 뺐으면 `conditional`, 그 외 `ready`.

## 15. 참고

- [docs/research/trl-sft-dpo.md](research/trl-sft-dpo.md) — SFT·DPO 전처리, loss·logits 경로, reference
- [docs/research/trl-grpo.md](research/trl-grpo.md) — GRPO batch 규칙, rollout, log-prob, reward, phase 순서
- [docs/research/loading-quantization-peft.md](research/loading-quantization-peft.md) — 로딩, 4-bit, PEFT, optimizer, workspace 가정
- [docs/research/architecture-memory.md](research/architecture-memory.md) — activation·cache 식, LM head/loss 규칙, phase 조합
- [support-matrix.md](support-matrix.md) — profile registry에서 생성한 지원 등급
- [methodology-architectures.md](methodology-architectures.md) — architecture adapter의 가중치·activation·cache 식
