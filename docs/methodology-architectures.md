# Architecture adapter 계산 방법론 (`dense_decoder`, `qwen3_5_hybrid`)

> `packages/estimator/src/vramforge_estimator/architectures/`가 내는 모든 수치의 식, 근거, 증거 수준을 정리한다.
> `AllocationSpec.formula_ref`는 이 문서의 anchor(`methodology-architectures.md#<id>`)를 가리킨다.
> 사양은 [`plan.md`](../plan.md) §6.3, §6.4, §9.3–9.5, §9.7, §11.4, §19.1, §19.3이고, 라이브러리 동작의 근거는
> [`docs/research/architecture-memory.md`](research/architecture-memory.md)(이하 **AM**),
> [`docs/research/loading-quantization-peft.md`](research/loading-quantization-peft.md)(이하 **LQ**),
> [`docs/research/trl-grpo.md`](research/trl-grpo.md)(이하 **GR**)다.

| 항목 | 값 |
|---|---|
| 대상 스택 | torch 2.14.1, transformers 5.18.0, trl 1.14.1, peft 0.21.2, bitsandbytes 0.50.2, accelerate 1.15.0 (단일 CUDA GPU, sm80 이상 가정) |
| 단위 | 정수 byte. GiB 변환은 화면 전용 |
| 범위 | 모델 구조만: 상주 가중치, 학습 가능 파라미터, 학습 step activation(embedding → final norm), no-grad forward, generation cache. LM head·loss·optimizer·gradient·batch tensor는 trainer/memory 모듈 담당 |
| 코드 | `structure.py`(구조 해석·검증), `matching.py`(PEFT/bnb 이름 규칙), `weights.py`, `trainable.py`, `activations.py`(층별 식), `ledger.py`(timepoint 조립), `decoder.py`(adapter), `registry.py` |

## 0. 표기와 증거 수준

| 기호 | 의미 | MiMo-V2.6-Distill-Qwen-9B |
|---|---|---|
| `B`, `T`, `N = B·T` | batch, padding 포함 길이, token slot | |
| `H`, `I`, `V` | hidden, intermediate, vocab | 4096, 12288, 248320 |
| `nq`, `nkv`, `d`, `r` | query head, KV head, head_dim, rotary 차원 | 16, 4, 256, 64 |
| `Hk`, `Hv`, `dk`, `dv`, `K` | Gated DeltaNet key/value head 수·차원, conv kernel | 16, 32, 128, 128, 4 |
| `C = 2·Hk·dk + Hv·dv`, `Vd = Hv·dv` | conv 채널, value 폭 | 8192, 4096 |
| `nc = ceil(T/64)`, `Np = B·64·nc` | delta-rule chunk 수, chunk padding 포함 slot | |
| `b`, `f` | 16-bit activation byte(2), fp32 byte(4) | |
| `L` | generation cache에 있는 position 수 | |

| `Evidence` | 언제 쓰는가 |
|---|---|
| `analytic` | 소스·측정으로 검증된 닫힌식, 또는 소스에서 직접 읽은 규칙(INFERRED이면 note에 표시) |
| `assumption` | CPU 측정 계수나 INFERRED 대안을 `bytes_low`/`bytes_high` 범위로 표현한 값 |
| `unknown` | 검증된 식이 없는 경로. `bytes_low = bytes_high = None` + 사유(0으로 채우지 않음, plan §0-12) |

골든 값 규약: autograd가 `tensor * python_float`에 저장하는 0-dim float64 wrapped scalar(8 B)는 세지 않는다(AM §F).

## 1. Adapter 선택과 구조 검증

`registry.match_adapter(facts)`는 config 사실만으로 고르고, 모델 이름 문자열은 보지 않는다.

| adapter | 조건 (`ArchitectureFacts`) |
|---|---|
| `qwen3_5_hybrid` | `layer_types`에 `linear_attention`이 있고 나머지는 `full_attention`, `linear_attention` 차원 5개(`num_key_heads`, `num_value_heads`, `key_head_dim`, `value_head_dim`, `conv_kernel_dim`)가 모두 양수 |
| `dense_decoder` | 모든 layer가 `full_attention`/`sliding_attention`(`layer_types`가 없으면 transformers `get_layer_types_and_kwargs`처럼 `sliding_window` 유무로 추론), linear-attention 차원 없음 |
| 공통 거절 | MoE 표지(`num_experts`, `num_local_experts`, `n_routed_experts`, `moe_intermediate_size`, `num_experts_per_tok`, `n_shared_experts`, `moe_layer_freq`, `decoder_sparse_step` > 0), `is_encoder_decoder`, `hidden_act`가 SiLU 아님, `rms_norm_eps` 없이 `layer_norm_eps`/`layer_norm_epsilon`만 있음(nn.LayerNorm decoder) |

모든 adapter 메서드는 먼저 inventory의 **모듈 shape**을 검증한다(`ModelStructure`). 맞지 않으면
`UNSUPPORTED_ARCHITECTURE`(한국어 메시지 + `details`)다.

- 공통: 각 text layer에 `gate/up/down_proj`(`H→I`, `I→H`), `input_layernorm`, `post_attention_layernorm`. 그 밖의 Linear(예: MoE expert, `shared_expert_gate`)가 있으면 거절.
  실행되는 text norm에 `bias` tensor가 있으면 nn.LayerNorm(예: Llama 이름을 쓰는 StableLM)이라 저장 tensor가 RMSNorm 식(AM §5)과 달라 거절한다. 실행되지 않는 vision tower의 LayerNorm bias는 상관없다.
- Qwen3.5 full layer: `q_proj: H→2·nq·d`(query + output gate, AM §1.2), `k/v_proj: H→nkv·d`, `o_proj: nq·d→H`, head 단위 `q_norm/k_norm [d]`.
- Qwen3.5 linear layer: `in_proj_qkv: H→C`, `in_proj_z: H→Vd`, `in_proj_a/b: H→Hv`, `out_proj: Vd→H`, gated `norm`.
- dense: `q_proj: H→nq·d`, `k/v_proj: H→nkv·d`, `o_proj: nq·d→H`; norm은 위 두 개와 선택적 `q_norm/k_norm`(Qwen3)만.

Inventory 관례(입력 계약 해석): `component`는 `text`/`vision`/`projector`/`audio`/`mtp`, `role`은 embedding/lm_head/norm/conv/parameter/linear_weight/linear_bias,
`linear_modules`는 2-D Linear마다 1개(lm_head는 있어도 없어도 된다), text Linear는 `layer_index`(없으면 이름의 `layers.<i>.`)를 쓴다.
`tests/fixtures/inventories/inventory_builder.py`가 같은 관례로 fixture를 만든다.

<a id="w-scope"></a>
## 2. Loading scope

| scope | 상주 component | 근거 |
|---|---|---|
| `full_checkpoint` | text + vision + projector + audio (MTP 제외) | TRL은 `config.architectures[0]`(예: `Qwen3_5ForConditionalGeneration`)으로 로드한다 (LQ §1.3) |
| `text_only` | text (+`other`) | `AutoModelForCausalLM` → `Qwen3_5ForCausalLM`, `model.visual.*` 무시 (LQ §1.4) |

- MTP tensor는 어느 scope에서도 로드되지 않는다(`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`).
- vision tower는 텍스트 전용 데이터에서 **실행되지 않는다**(AM §9). 가중치만 상주하고 activation·gradient·optimizer state는 없다.
- `other` component는 실행된다고 보수적으로 가정한다.
- tied group(`inventory.tied_groups`, 없으면 `tie_word_embeddings`)은 embedding tensor 1개만 센다.

<a id="w-dense"></a>
## 3. 상주 가중치

```text
W_dense = Σ_{비양자화 tensor} numel × bytes(load_dtype)        # upcast_to_fp32_patterns에 걸리면 fp32
W_q4    = Σ_{양자화 Linear}  B_q4(numel(weight))              # bias는 W_dense (load dtype)
```

비양자화 tensor는 모두 load dtype이다. Qwen3.5에는 `_keep_in_fp32_modules`가 없고 TRL은 norm·embedding·lm_head를
올리지 않는다(LQ §3.1-3, §Q6.2). TRL 문자열 진입의 기본 load dtype은 fp32이므로 resolver가 정한 `load_dtype`을 그대로 쓴다.

allocation은 따로 낸다: `weights.base.q4_payload`, `weights.base.q4_metadata`, `weights.base.dense`, `weights.vision_tower`.

<a id="w-bnb-4bit"></a>
### 3.1 bitsandbytes 4-bit (LQ §3.1-3.5, CPU에서 byte 단위 검증 E3/V2)

```text
nb = ceil(n/64)                                              # blocksize 64 고정, 부분 block도 absmax 1개
B_q4_noDQ(n) = ceil(n/2) + 4·nb + 64                          # packed + fp32 absmax + 16-entry code
B_q4_DQ(n)   = ceil(n/2) + nb + 4·ceil(nb/256) + 4 + 64 + 1024  # uint8 absmax + nested absmax/offset/code
```

- 양자화 대상: `type(module) is nn.Linear`인 모듈만(conv1d, `A_log`, `dt_bias`, norm, embedding은 load dtype). vision Linear도 4-bit다.
- 제외: `quantization.skip_module_patterns`는 trainer config의 `llm_int8_skip_modules`로 그대로 나가고, transformers는 이 목록이 있으면
  기본 skip을 **대체**한다(`get_modules_to_not_convert`, `add_default_skips=False`, LQ §2.2). 그래서 목록이 비어 있을 때만 기본 skip
  (출력 embedding = `lm_head`, tied 모듈)을 쓰고, 목록이 있으면 그 목록만 쓴다. 매칭은 `re.match(key, name) or name.endswith(key)`.
  목록이 출력 embedding을 빼서 `lm_head`가 4-bit가 되는 구성은 LM head·loss 쪽 메모리 모델이 없어 `UNSUPPORTED_BACKEND_COMBINATION`으로 거절한다.
- `quant_storage`는 byte 수를 바꾸지 않는다. CUDA allocator의 512 B 반올림(MiMo +422,544 B)은 allocator slack 쪽이다.

| MiMo (NF4) | DQ | no-DQ |
|---|---:|---:|
| full_checkpoint (Linear4bit 358) | 3,802,183,024 B | 4,145,469,568 B |
| text_only (Linear4bit 248) | 3,569,313,760 B | 3,891,674,624 B |
| 비양자화 text (bf16 / fp32) | 4,070,597,632 B / 8,141,195,264 B | |
| vision 비양자화 (bf16) | 9,663,968 B | |

<a id="load-transient"></a>
## 4. 로딩 단계 (LQ §3.5, §Q4)

```text
quantized:   low  = max_q [ n_q·bytes(load) + (DQ ? 8·ceil(n_q/64) : 0) ]
             high = max_t [ quantized ? n_t·bytes(load) + max(conv_t, DQ_t) : conv_t ],  conv_t = n_t·bytes(ckpt) if ckpt ≠ load
unquantized: ckpt == load → transient 없음 (allocation 없음)
             ckpt ≠ load  → low 0 (host 변환), high = Σ 상위 4개 tensor의 checkpoint-dtype byte (device 변환, worker 4개)
```

- on-the-fly 양자화는 thread pool 없이 tensor 1개씩 처리한다(검증 V8). low가 `analytic`, device 변환 상한은 INFERRED(LQ 미확정 1)라 `assumption`.
- MiMo bf16 load: 106,954,752 B(gate_proj). fp32 load: 207,618,048 B, device 변환이면 embed_tokens bf16 원본 2,034,237,440 B.
- `caching_allocator_warmup`은 `S_load = Σ_q 0.5·n + Σ_nq n·bytes(load)`(MiMo bf16 7,765,103,072 B) 크기의 tensor를 잠시 할당했다가
  reserved로 남긴다. 이 크기는 최종 가중치보다 작고 가중치 로드 전에 해제되므로 allocated 피크를 올리지 않는다(allocation 없음).
  같은 `S_load`는 단일 GPU·`device_map="auto"`·`max_memory` 미지정에서 **free × 0.81 ≥ S_load**여야 로딩된다는 조건에 쓰인다(LQ §4.5).
  adapter의 `loading_budget_bytes()`가 이 값을 돌려준다.

<a id="trainable"></a>
## 5. 학습 가능 파라미터 (LQ §5, §Q6, §Q8)

LoRA 대상 해석(`lora_target_modules`)은 inventory 전체(full-checkpoint 이름)에 대해 한다. loading scope 적용은 `trainable_groups`가 한다.

| target | 의미 |
|---|---|
| `"auto_verified"` | adapter가 검증한 text decoder Linear 종류 전부. hybrid: q/k/v/o, in_proj_qkv/z/a/b, out_proj, gate/up/down (MiMo 248개). dense: q/k/v/o, gate/up/down. lm_head·embedding·vision 제외 |
| `"all-linear"` | PEFT 의미: 출력 embedding을 뺀 모든 Linear, vision 포함 (MiMo 358개) |
| 그 밖의 문자열 | 정규식 1개, `re.fullmatch(pattern, module_name)` |
| 리스트 | 이름 일치 또는 `name.endswith("." + entry)` |
| 원소 1개 리스트 + 정규식 전용 문자(`*+?[](){}\|^$\`) | 그 정규식 1개 (요청 스키마가 "이름 목록 또는 정규식 1개"를 같은 리스트 필드로 받음). 점(`.`)만 있으면 이름 의미 |
| `exclude` | 리스트 의미로 마지막에 적용 |

- Linear가 아닌 모듈(embedding, conv, container)이 걸리면 `CONFLICTING_OPTIONS`로 거절한다. 아무것도 안 걸려도 오류다.
- 사용자 정규식은 512자 이하, 중첩 반복(`(a+)+` 등) 금지(ReDoS 방지). PEFT는 이 경우 학습 중에도 멈출 수 있다.
- 이름은 checkpoint(CondGen) 기준이다. `text_only` scope의 실제 모듈 이름은 `model.language_model.` → `model.`로 바뀌므로 그 prefix에 의존하는 정규식은 결과가 다를 수 있다.

```text
P_lora = Σ_targets r_m·(in_m + out_m)          # in/out = 모듈 속성, r_m = rank_pattern 첫 일치(re.match(rf"(.*\.)?({key})$")) 또는 r
P_dora = Σ_targets out_m                       # lora_magnitude_vector
alpha, rsLoRA, dropout은 파라미터 수를 바꾸지 않는다
```

| 그룹 kind | 저장소 | dtype |
|---|---|---|
| `lora` (A/B, DoRA magnitude) | **새 파라미터** | `effective_dtypes.adapter` (TRL QLoRA bf16, 비양자화 LoRA fp32 — `autocast_adapter_dtype`) |
| `modules_to_save` | **새 deep copy** (원본은 frozen으로 base에 남음) | QLoRA면 bf16(TRL이 trainable 전부 cast), 아니면 원본 dtype |
| `bias` (`all`: 이름이 `.bias`로 끝나는 모든 파라미터, `lora_only`: LoRA 모듈의 bias) | **base 가중치** (gradient·state만 추가) | 위와 같음 |
| `full` | **base 가중치** | load dtype |

- 그룹은 (이름, dtype, 실행 여부)와 tensor shape별로 나눈다. `numel / tensor_count`가 tensor 하나의 크기라서 8-bit optimizer의 크기 기준(4096)과 step scalar를 정확히 셀 수 있다. full FT의 `nn.Embedding`은 `:embedding:` 그룹으로 분리한다(8-bit optimizer가 32-bit state를 쓰는 대상, LQ §8.3).
- `ArchTrainableGroup.receives_grad=False`: vision tower처럼 텍스트 전용 데이터에서 실행되지 않는 파라미터. 가중치는 상주하지만 gradient와 optimizer state는 생기지 않는다(LQ §Q8.4, E5d/V6). MiMo all-linear: 51,265,024 중 43,278,336만 gradient를 받는다.
- `modules_to_save`는 점 경계 없는 `endswith`로 고르고(예: `"norm"`은 `input_layernorm`, `q_norm`도 고름), 그 안의 모듈은 LoRA 대상에서 빠진다(점 경계 정규식). 서로 포함되는 선택, 4-bit 모듈을 가리키는 선택(LQ 미확정 7)은 거절한다. tied `lm_head`는 자기 tensor가 없어도 embedding shape의 복사본을 만든다.
- full fine-tune에서 `trainable_full_patterns`는 모듈 이름 정규식(`re.fullmatch`)이다. 비어 있거나 `.*`/`*`면 로드된 모든 파라미터가 학습된다(resolver 기본 `[".*"]`).
- 4-bit + full fine-tune은 거절한다(`validate_quantization_for_training`).

<a id="act-layers"></a>
## 6. 학습 step activation (`train_step_ledger`)

범위는 embedding 출력부터 final norm까지다. LM head 입력이 lm_head에 저장되는지, logits·loss는 trainer adapter가 센다.
층별 saved byte는 아래 식을 항목(term) 단위로 옮겼고 각 항목은 ledger group(`norms`, `mlp`, `attention`, `linear_attention`, `lora`, `mask`)을 가진다.
층 하나의 saved 합계 `S_layer = Σ terms`.

**검증**: AM §F 골든 표(3행 × 5열), fixture (a), §10.2 실제 차원 값(MiB 소수 둘째 자리)을 그대로 재현한다(`tests/unit/architectures/test_arch_formulas.py`).
추가로 CPU에서 실제 PEFT 모델의 autograd graph를 순회해 147개 점(Qwen3.5 linear/full, Llama, Qwen3 × frozen/full FT/full FT+autocast/LoRA bf16·fp32 ± autocast × 3 shape)이 byte 단위로 일치함을 확인했다(`test_arch_parity.py`, parity marker).

**16-bit 전용**: 모든 식은 bf16/fp16 load(`b = 2`)에서 검증되었다. fp32 load에서는 RMSNorm의 `x.float()`가 복사하지 않는 등 saved set이 달라진다(AM §5, INFERRED). 따라서 fp32 load의 층별 saved set, final norm, 이를 기반으로 한 transient는 `unknown`이다. 경계 hidden state, cos/sin, mask는 load dtype으로 계산된다.

<a id="act-norm"></a>
### 6.1 RMSNorm, MLP (AM §5)

```text
RMSNorm_q35(rows, w)   = f·rows·w + f·rows + f·w + [f·rows·w  if weight trains]     # Qwen3.5 zero-centered, fp32 (1+w)
RMSNorm_llama(rows, w) = f·rows·w + f·rows       + [b·rows·w  if weight trains]     # Llama/Qwen3
MLP = [b·N·H if gate/up 입력 저장] + 3·b·N·I (SiLU 입력, SiLU 출력, up 출력) + [b·N·I if down 입력 저장]
```

final norm(`act.final_norm`)은 checkpoint 밖이라 backward까지 산다. LoRA에서는 norm weight가 frozen이다(`modules_to_save`로 고르면 trainable).

<a id="act-attention"></a>
### 6.2 Full attention

Qwen3.5 (`q_proj`가 query + gate, partial rotary, 항상 output gate):

```text
S_fullmix = [qkv 입력] + RMSNorm_q35(N·nq, d) + RMSNorm_q35(N·nkv, d)
          + b·N·nq·d (q) + 2·b·N·kv·d (k, v; kv = nq if expand else nkv)
          + 경로별 항 (아래 표) + b·N·nq·d (sigmoid(gate)) + [o_proj 입력]
```

dense (Llama/Qwen3, AM §8):

```text
S_attn = [qkv 입력] + [RMSNorm_llama(N·nq, d) + RMSNorm_llama(N·nkv, d)  if Qwen3 q/k norm]
       + b·N·nq·d + 2·b·N·kv·d + 경로별 항 (SDPA 출력 = o_proj 입력, 복사 없음)
```

cos/sin은 층마다 세지 않고 모델당 1번(§6.5).

**경로표** (`attention_path_by_layer_type`; 값이 없으면 transformers 기본 `sdpa`로 계산하고 note를 단다):

| resolved 값 | CUDA 실제 경로 (torch 2.14.1, sm80+) | 경로별 항 | 근거 |
|---|---|---|---|
| `sdpa`, mask 없음, `d ≤ 256` | flash (Hopper/Blackwell cuDNN도 같은 byte) | lse `f·B·nq·T`, out `b·N·nq·d`, Qwen3.5만 contiguous 복사 `b·N·nq·d` | AM §3.2-3.4, flash/cuDNN 출력이 q layout을 따름 |
| `sdpa`, mask 있음 또는 `d > 256` | mem-efficient | lse `f·B·nq·ceil32(T)`, out, **복사 없음**, K/V는 `repeat_kv`로 nq head (`expand`), mask면 층마다 additive mask `b·B·T·ceil8(T)` | AM §3.1, §3.3; torch `attention.cpp:579-639`(8 정렬 padding) |
| `sdpa_mem_efficient` | mem-efficient | 위와 같음 | |
| `eager` | eager | K/V 확장, softmax fp32 + probs `(f+b)·B·nq·T²`, Qwen3.5는 contiguous 출력이 gate mul에 저장, dense는 o_proj가 학습될 때만 | AM §3.3, fit 계수 `6·nq` |
| `flash_attention_2`, `sdpa_math`, 그 밖 | — | **unknown** | AM G, 미확정 4 |

mask가 생기는 조건(AM §3.1, transformers `masking_utils._ignore_causal_mask_sdpa`): batch에 padding이 있거나, sliding layer에서 `T ≥ sliding_window`. eager는 항상 float mask를 만든다.

<a id="act-mask"></a>
**padding 판정**: `SequenceShape`에는 padding 여부가 없다. `B > 1`, `pad_to_multiple_of > 1`, GRPO(왼쪽 padding된 prompt + 오른쪽 padding된 completion)이면 padding **가능**으로 보고
mask 없는 경로와 mask 경로를 둘 다 계산해 group마다 `low = min`, `high = max`로 낸다(어느 경로든 범위 안). `B = 1`이고 padding이 없으면 `enable_gqa` + `is_causal`로 mask가 없다.
모델 수준 mask(`act.attn_mask.<layer_type>`): SDPA bool `[B,1,T,T]` 1 B/원소(padding이면), eager float `[B,1,T,T]` load dtype(항상). GC면 checkpoint kwargs로 backward까지, GC가 없으면 forward 동안만 산다.
linear-attention layer의 padding mask 곱(`apply_mask_to_padding_states`)은 Linear 입력을 masked tensor로 바꿀 뿐 크기가 같고, 추가되는 `[B,T,1]` mask는 무시할 만하다.

<a id="act-linear"></a>
### 6.3 Gated DeltaNet linear attention (AM §1.3, §2, §4.2)

```text
S_linmix = [in_proj 입력] + b·N·C (conv 입력: frozen conv1d도 저장) + [b·B·C·(T+K−1) SiLU 입력: torch만]
         + b·N·Hv (beta) + f·N·Hv (softplus 입력) + [f·N·Hv softplus 출력 if A_log 학습]
         + f·Hv·(2 if A_log 학습 else 1)                      # -exp(A_log) [Hv]: 아래 정정 참고
         + delta rule (kernel별) + gated RMSNorm + [out_proj 입력]
gated RMSNorm = f·N·Vd + f·N·Hv + [b·N·Vd if norm 학습] + 2·f·N·Vd + b·N·Vd
```

| kernel (`linear_attention` 값) | delta rule 항 | 증거 |
|---|---|---|
| `torch_fallback` (고정 환경 기본, fla/causal-conv1d 미설치) | autocast 없음: `f·Np·Hv·(1+dv+dk) + 5·f·Np·Hv·64 + 3·f·Np·Hv·dk + [c]·f·Np·Hv·dk + [nc≥2]·f·Np·Hv·dk + (2+[c])·f·Np·Hv + [c]·f·B·Hv·nc + 2·f·Np·Hv·dv + nc·f·B·Hv·dk·dv + 2·f·N·Hv·(dk+1) (l2norm) + 64·64`, `c = nc≥2 or cache` | analytic (216점 + 독립 재검증) |
| `torch_fallback` + bf16 autocast (TRL `bf16=True`) | fp32 matmul 피연산자를 사용마다 bf16 cast로 저장: AM `q35_linear_attn_torch_autocast` 형식, nc=1 정정 포함 | analytic |
| `fla` | causal-conv1d는 x만, fla는 정규화 q/k(Hv head) `2·b·N·Hv·dk` + rstd `2·f·N·Hv` + v 복사 `b·N·Vd` + g cumsum `f·N·Hv` + A `b·N·Hv·64` | analytic (INFERRED: CUDA 미실행, 소스 정독) |
| hub kernels, fla 대체 backend, 그 밖 | — | unknown |

- **정정(측정)**: AM §4.3은 frozen `A_log`면 `[Hv]` tensor 2개가 빠진다고 적었지만, CPU graph walk(tiny config A)에서 mul이 `-exp(A_log)` `[Hv]` fp32 1개를 계속 저장한다. 이 1개를 넣어야 fixture (a)(3,909,948 / 3,619,548)와 frozen L1(3,482,748)이 맞는다. 실제 차원에서 128 B라 §10.2 값은 그대로다. 이 항은 modeling 코드에서 계산되므로 fla 경로에도 넣었다(AM의 fla 식에는 없음, 256 B).
- **autocast 판정**: `ResolvedConfig`에 mixed precision 플래그가 없어 `effective_dtypes.compute`가 bf16/fp16이면 accelerate native AMP(autocast)로 본다(TRL 기본 `bf16=True`, AM §2.4).
- torch fallback + packing/padding-free는 sequence 경계에서 상태가 섞여 지원 불가다(AM G). `ResolvedConfig`에 packing이 없으므로 resolver가 막아야 한다.

<a id="act-lora"></a>
### 6.4 LoRA (PEFT 0.21.2, AM §4.3, LQ §5.9)

같은 tensor를 읽는 Linear 묶음(`qkv`, `o_proj`, `in_proj`(4개), `out_proj`, `gate_up`, `down`)마다:

| 상황 | 저장 |
|---|---|
| frozen base, LoRA 없음 | 없음 |
| base weight 학습(full FT, modules_to_save) | 공유 입력 `b·N·in` 1번 |
| bf16 adapter (TRL QLoRA) | 공유 입력 1번 + 모듈마다 `b·N·r` (lora_B 입력) |
| fp32 adapter + bf16 autocast (비양자화 LoRA, CUDA 기본) | 공유 입력 없음, 모듈마다 cast 사본 `b·N·in` + `b·N·r` + cast weight `b·(in·r + r·out)` |
| fp32 adapter, autocast 없음 | 모듈마다 `f·N·in` + `f·N·r` |
| `lora_dropout > 0` | 모듈마다 mask `1·N·in`(CUDA bool, INFERRED), bf16 adapter면 dropout 출력 `b·N·in`이 공유 입력을 대신함 |
| DoRA | saved set·임시값 미검증 → `act.dora_extra` **unknown** |

dense SDPA의 o_proj 입력은 SDPA 출력과 같은 storage라 공유 입력 비용이 0이다. QLoRA 기반 Linear4bit는 activation을 저장하지 않는다(`ctx.tensors = (None, B)`).
첫 layer의 input_layernorm(PEFT + GC 없음 + `enable_input_require_grads` 없음)은 실제로 저장하지 않지만 보수적으로 센다.

<a id="act-kwargs"></a>
### 6.5 모델 수준 tensor

| allocation | 크기 | 수명 |
|---|---|---|
| `act.rope_cos_sin` | Qwen3.5 mrope `2·bytes(load)·B·T·r`; dense `2·bytes(load)·T·r` (position_ids 없음) ~ `·B` (trainer가 position_ids를 넘기면) | forward·loss·backward (RoPE mul이 저장, GC면 kwargs) |
| `act.attn_mask.<type>` | §6.2 | GC: 전체, no-GC: forward |
| `act.final_norm` | RMSNorm(N, H) | forward·loss·backward |

<a id="act-boundary"></a>
`act.final_hidden`(`[B,T,H]`, load dtype)은 final norm 출력 = LM head 입력이다. forward·loss에만 살고 `storage_alias_group = "<prefix>.final_hidden"`이다. LM head ledger가 lm_head 학습 때문에 이 tensor를 저장분으로 셀 때 같은 alias group을 쓰면 중복되지 않는다.
batch tensor(input_ids, labels, mask)와 embedding이 저장하는 index는 같은 storage라 trainer 쪽에서 센다.

<a id="act-gc"></a>
### 6.6 Gradient checkpointing과 timepoint (AM §4.4, 구현 시사점 C)

`StepTimepoints(forward, loss, backward)`에 붙이는 방식:

| allocation | GC (`per_decoder_layer`, non-reentrant) | GC 없음 |
|---|---|---|
| `act.ckpt_boundaries` = layers × `[B,T,H]` × bytes(load) | forward·loss·backward | — |
| 층별 saved set `act.<type>.<group>` (count = 해당 type의 layer 수) | — (backward 재계산 안에 포함) | forward·loss·backward |
| `act.layer_transient.forward` = `k_fwd × S_max` | forward | forward |
| `act.recompute.backward` = `k_bwd × S_max` | backward (`RECOMPUTE_WORKING_SET`) | — |
| `act.layer_transient.backward` = `k_nogc × S_max` | — | backward |

`S_max` = 가장 큰 layer의 saved set(같은 학습 mode, padding 범위 포함). backward 중에도 모든 경계 입력이 산다고 보는 것은 보수적이다(실제로는 뒤쪽 layer부터 해제).
transformers 5.18의 `every_n_layers`, `offload`는 `ResolvedConfig`에 없고 GRPO에서는 무시된다(GR §4.3). checkpoint 단위는 decoder layer다.

<a id="act-transient"></a>
### 6.7 Transient 계수 (CPU 측정, GPU 보정 전 → `assumption`)

`(peak − retained) / S_max`. low는 AM §4.4의 가장 작은 측정값(CPU allocator 또는 실제 차원 tensor 수준), high는 AM 기본값(측정이 더 크면 그 값)이다.

| mode | GC forward `k_fwd` | GC backward `k_bwd` | no-GC forward | no-GC backward `k_nogc` |
|---|---|---|---|---|
| full FT (param grad 제외) | 0.26 – 0.45 | 0.60 – 1.35 | 0.04 – 0.06 | 0 – 0.15 |
| LoRA bf16 adapter (QLoRA) | 0.32 – 0.45 | 1.05 – 1.35 | 0.04 – 0.09 | 0.12 – 0.16 |
| LoRA fp32 adapter + autocast | 0.37 – 0.45 | 1.14 – 1.36 | 0.19 – 0.20 | 0.12 – 0.16 |

full FT의 parameter gradient와 GA 누적은 gradient ledger(trainer/memory) 몫이다. CUDA caching allocator의 block 반올림·단편화는 포함하지 않는다(allocator slack).

<a id="act-q4"></a>
### 6.8 bitsandbytes 실행 중 임시값 (LQ §3.6, CUDA 경로 INFERRED)

```text
forward  dequant = n_max·bytes(compute) + (DQ ? 8·ceil(n_max/64) : 0)   # M = tokens > 1536; M ≤ 4는 fused(0); 그 사이는 [0, dequant]
LoRA clone       = M × out_max × bytes(load)                            # PEFT bnb 래퍼의 result.clone()
backward dequant = n_max·bytes(load) + (load ≠ compute ? n_max·bytes(compute) : 0) + (DQ ? 4·ceil(n_max/64) : 0)
```

`act.q4_dequant.forward`는 max(dequant, clone), backward는 GC면 재계산 forward까지 포함한 max, GC가 없으면 backward dequant다. 실행되는 text 모듈만(vision 제외) 본다.
MiMo(T=4096): forward 106,954,752 B, backward 103,809,024 B(bf16 load) / 305,135,616 B(fp32 load).

### 6.9 학습 중 `use_cache=True`

TRL은 `use_cache=False`를 넘기고 GC도 cache를 끈다. resolver가 `use_cache_during_training`을 켜고 GC가 없으면 cache가 linear layer의 마지막 state-update branch를 붙잡는다(AM §4.2 cache 변형).
측정상 이 branch는 backward에 필요 없고 cache가 살아 있는 동안만 남으므로 `act.cache_branch`와 `act.cache_states`(conv + recurrent state)를 forward·loss에만 둔다. full-attention K/V는 SDPA가 저장하는 k/v를 대신할 뿐이라 추가분이 없다.

<a id="act-nograd"></a>
## 7. No-grad forward (`no_grad_forward_ledger`)

reference/old-policy log-prob 같은 inference forward. saved tensor가 없고 한 번에 한 layer만 실행된다(AM §7).

```text
nograd.hidden            = B·T·H·bytes(load)                  # layer 사이 hidden state 1개
nograd.rope_cos_sin, nograd.attn_mask.<type>                  # §6.5와 같은 크기
nograd.layer_working_set = k_fwd × S_max   (GC forward 계수: checkpoint 구간 forward도 saved를 버린다)
nograd.q4_dequant        = forward dequant / clone
```

LM head·logits는 제외한다.

<a id="gen-kv"></a>
## 8. Generation cache (`generation_ledger`, AM §7, 구현 시사점 E, GR R3)

`generate()`의 `DynamicCache`를 `num_sequences = C`개 sequence에 대해 센다. 마지막 생성 token은 다시 넣지 않으므로 decode 시점 position 수는 `L = P + new − 1`이다.

```text
KV(full)    = 2 · n_full · nkv · d · bytes(load) · C · L          # prefill: L = P
KV(sliding) = 같은 식, decode L = min(P + new − 1, W) (new ≥ 2), prefill L = P   # DynamicSlidingWindowLayer는 cat 결과의 view를 남김
cat 성장     = low: layer 1개의 새 K(또는 V), high: K+V                       # decode
```

- K/V dtype = load dtype: PEFT generate는 autocast가 없고, full FT는 autocast여도 RoPE type promotion으로 K가, lazy init으로 V가 load dtype이 된다(GR R3 표 A·C·F·G·Q·R). resolver의 `effective_dtypes.kv_cache`가 다르면 note에 표시하고 이 규칙을 쓴다.
- KV는 attention layer에만 붙는다. hybrid에서 32개 전부에 KV 식을 쓰면 4배 과대다(plan §19.3).
- prefill 작업 집합은 §7의 no-grad ledger(`gen.prefill.nograd.*`)를 `B = C`, `T = P`로 쓴다(PEFT면 autocast 없음). C > 1 또는 GRPO면 왼쪽 padding mask 범위가 붙는다.
- decode 한 step의 작업 집합(token 1개)과 `(C, V)` logits는 무시할 만하거나 trainer 쪽이다. bnb decode(M = C ≤ 4)는 fused kernel이라 clone만 남는다.

<a id="gen-linear"></a>
### 8.1 Linear-attention state

```text
conv state      = n_lin · C · conv_dim · K · bytes(conv dtype)     # K 위치(K−1 아님), L과 무관
recurrent state = n_lin · C · Hv · dk · dv · 4                     # 항상 fp32 (mamba_ssm_dtype 무시)
conv dtype      = compute dtype if (full FT and autocast) else load dtype   # GR R3
```

MiMo bf16 load, sequence 1개: `51,904,512 + 32,768·L` B(linear 24층 49.5 MiB + full 8층 32 KiB/position).
C = 4, P = 272, L = 1,295: KV 161.875 MiB, linear state 198 MiB.

## 9. INFERRED / UNKNOWN 정리

| 항목 | 상태 | 처리 |
|---|---|---|
| fp32 load의 층별 saved set | 식 없음 (AM §5) | unknown |
| FA2, SDPA math, flex, hub kernels, fla 대체 backend | saved set 미확인 | unknown |
| DoRA activation·임시값 | 미검증 | unknown |
| fla + causal-conv1d saved set | INFERRED (CUDA 미실행) | analytic + note |
| mem-efficient mask 경로(복사 없음, 8/32 정렬) | INFERRED (CUDA 소스) | analytic, padding 범위의 high |
| SDPA backend 선택 | sm80+ 가정 (hardware가 `ResolvedConfig`에 없음) | flash/mem-efficient만 |
| transient 계수 | CPU 측정, CUDA allocator 미포함 | assumption 범위 |
| device-side dtype 변환 | INFERRED | load transient high |
| dropout mask dtype(CUDA bool) | INFERRED | analytic + note |
| vision tower 미실행 | VERIFIED (텍스트 전용 데이터) | 가중치만, `receives_grad=False` |

## 10. 다른 모듈과의 계약 메모

- `TrainableGroup.kind`가 `lora`/`modules_to_save`면 새 저장소(가중치 allocation 필요), `full`/`bias`면 base 가중치(gradient·optimizer만). `receives_grad=False` 그룹은 gradient·state를 만들지 않는다.
- `effective_dtypes.adapter`(LoRA dtype)는 resolver가 정한 값을 쓴다. generation cache dtype은 위 구조 규칙을 쓴다.
- trainer adapter는 `act.final_hidden` alias group, `SequenceShape.batch`(DPO는 2 × pairs), 생성 timepoint를 넘긴다.
