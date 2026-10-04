# Activation·cache 메모리 회계: Qwen3.5 hybrid decoder와 dense decoder

| 항목 | 값 |
|---|---|
| 주제 | 학습 activation(saved tensors)·transient·logits/loss·generation cache의 shape ledger 근거. plan.md §9.5 (AllocationSpec), §9.7 (rollout cache), §19.3 (메모리 모델 테스트) |
| 대상 버전 | torch==2.14.1 (설치본 `torch.version.git_version` = `5c4886908584029761b579af026dcfb627c84070` = GitHub tag `v2.14.1`), transformers==5.18.0, trl==1.14.1, peft==0.21.2, accelerate==1.15.0, bitsandbytes==0.50.2, datasets==5.0.1, huggingface_hub==1.33.0, tokenizers==0.23.2, safetensors==0.8.0 |
| 참고(미설치, 소스만 열람) | `fla-core==0.5.2`/`flash-linear-attention==0.5.2` (PyPI wheel, 2026-07-27), `causal-conv1d==1.7.0` (PyPI sdist, 2026-08-20). 공유 venv에는 `kernels`, `fla`, `causal_conv1d`, `flash_attn`, `liger_kernel`이 **없다** (importlib 확인) |
| 예시 모델 | `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` @ `2367e865d009c13ac81713a2878291d33ab28177` (config/header만 사용, weight 미다운로드) |
| 작성일 | 2026-10-04 |
| 작성 | research-arch-memory (Milestone M0) |
| 방법 | (1) 설치된 site-packages 소스 정독, (2) CUDA 전용 C++/Triton 경로는 tag `v2.14.1`의 torch 소스와 PyPI 배포본 소스 정독, (3) macOS arm64 CPU에서 tiny random-init 모델로 `torch.autograd.graph.saved_tensors_hooks` 기반 saved-tensor 측정과 `torch.profiler` `MemoryProfile` 기반 live-bytes timeline 측정, (4) `huggingface_hub.get_safetensors_metadata` header 조회와 meta-device 인스턴스화 |
| 실험 코드 | `/tmp/vf-research/scratch/arch-memory/` (저장소 밖 scratch). 핵심: `vfmem.py`(측정 harness), `formulas.py`(검증된 닫힌식), `e1`–`e13*.py` |

증거 태그: **VERIFIED** = 설치본 소스에서 읽었거나 실행·측정함, **INFERRED** = 소스로 추론했으나 실행하지 못함(CUDA/Triton 전용 등), **UNKNOWN** = 근거 부족.
인용 형식: `<pkg>==<ver> <site-packages 기준 경로>:<행> (<함수/클래스>)`. torch C++ 소스는 `torch v2.14.1 <repo 경로>:<행>`으로 적는다.

---

## 0. 표기법

| 기호 | 의미 | 예시 모델 값 |
|---|---|---|
| `B`, `T`, `N=B·T` | batch, padding 포함 sequence 길이, token slot 수 | |
| `H`, `I` | `hidden_size`, `intermediate_size` | 4096, 12288 |
| `nq`, `nkv`, `d` | full-attn query head 수, KV head 수, `head_dim` | 16, 4, 256 |
| `r` | rotary 차원 = `int(d · partial_rotary_factor)` | 64 |
| `Hk`, `Hv`, `dk`, `dv` | `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim` | 16, 32, 128, 128 |
| `Kd=Hk·dk`, `Vd=Hv·dv`, `C=2·Kd+Vd` | linear-attn key/value 폭, conv 채널(`conv_dim`) | 2048, 4096, 8192 |
| `K` | `linear_conv_kernel_dim` | 4 |
| `c=64`, `nc=ceil(T/64)`, `Np=B·64·nc` | delta-rule chunk 크기, chunk 수, chunk padding 포함 slot | `T=4096` → `nc=64` |
| `V` | `vocab_size` | 248,320 |
| `b`, `f` | compute dtype byte(bf16=2), fp32 byte(4) | |
| `L` | generation cache의 cached position 수 | |

이하 모든 byte 식은 "bf16으로 로드한 모델(또는 bf16 autocast)"을 기본으로 한다. 레이어 바깥 항목(embedding, final norm, lm_head, loss)은 따로 적는다.

---

## 1. Qwen3.5 text decoder layer 구조 (transformers 5.18)

### 1.1 레이어 골격과 공통 규칙

- 각 레이어는 pre-norm residual 구조다: `residual = x; x = input_layernorm(x); x = token_mixer(x); x = residual + x; residual = x; x = post_attention_layernorm(x); x = mlp(x); x = residual + x`. `layer_types[i]`가 `"linear_attention"`이면 mixer가 `Qwen3_5GatedDeltaNet`, `"full_attention"`이면 `Qwen3_5Attention`이다. 두 타입 모두 같은 `Qwen3_5MLP`를 가진다. — transformers==5.18.0 `models/qwen3_5/modeling_qwen3_5.py:860-913 (Qwen3_5DecoderLayer)`. VERIFIED
- 예시 config의 `layer_types`는 `[linear×3, full]×8` (full 8개, linear 24개)이고 meta-device 인스턴스화로 확인했다. config 클래스의 기본값은 `full_attention_interval=4`, `partial_rotary_factor=0.25`이다 — `models/qwen3_5/configuration_qwen3_5.py:110-117 (Qwen3_5TextConfig.__post_init__)`. VERIFIED
- config.json의 `attn_output_gate`, `mamba_ssm_dtype`, `mtp_*`는 config 객체에 attribute로만 남고 modeling 코드는 읽지 않는다(`grep` 결과 0건). output gate는 config와 무관하게 **항상** 적용된다. VERIFIED
- 기본 attention 구현은 `sdpa`다: `get_correct_attn_implementation`이 `requested_attention is None`이면 `"sdpa"`를 고르고, SDPA dispatch가 불가능할 때만 eager로 내려간다 — `modeling_utils.py:1851-1887`. 예시 config를 meta로 만들었을 때 `config._attn_implementation == "sdpa"`. VERIFIED

**`Qwen3_5RMSNorm` (zero-centered, fp32 upcast)** — `modeling_qwen3_5.py:839-854`

```python
def _norm(self, x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
def forward(self, x):
    output = self._norm(x.float())
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)
```

입력을 fp32로 복사한 뒤 정규화와 `(1+w)` 곱을 fp32로 하고 마지막에만 입력 dtype으로 돌린다. Llama/Qwen3의 `RMSNorm`은 `self.weight * hidden_states.to(input_dtype)`로 곱셈을 bf16에서 한다 (`models/qwen3/modeling_qwen3.py:50-64`). 이 차이가 saved tensor dtype을 바꾼다 (§5). VERIFIED

### 1.2 Full-attention block (`Qwen3_5Attention`) — `modeling_qwen3_5.py:748-820`

```python
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
)
gate = gate.reshape(*input_shape, -1)
query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
...
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = attn_output * torch.sigmoid(gate)
attn_output = self.o_proj(attn_output)
```

| 단계 | shape | dtype | 비고 |
|---|---|---|---|
| 입력 `x` (input_layernorm 출력) | `[B,T,H]` | b | q/k/v_proj 공용 입력 |
| `q_proj(x)` | `[B,T,2·nq·d]` | b | **query와 gate를 함께** 낸다 (`out_features = nq·d·2`, 예시 header `[8192, 4096]`) |
| `query_states`, `gate` | 각 `[B,T,nq,d]` view | b | `torch.chunk`의 view. `gate.reshape(B,T,nq·d)`는 non-contiguous라 **복사**된다 |
| `q_norm`/`k_norm` | `[B,T,nq,d]`, `[B,T,nkv,d]` | 입력 b, 내부 fp32 | head_dim 축 RMSNorm (`Qwen3_5RMSNorm(d)`) |
| `k_proj`, `v_proj` 출력 | `[B,T,nkv·d]` | b | |
| RoPE (`apply_rotary_pos_emb`, `:673-708`) | q `[B,nq,T,d]`, k `[B,nkv,T,d]` | b | **partial rotary**: 앞 `r` 차원만 회전 후 `torch.cat([rot, pass])`로 새 tensor. cos/sin은 `[B,T,r]` b (`Qwen3_5TextRotaryEmbedding.forward`, `:185-201`, mrope interleave `[11,11,10]`) |
| attention | 출력 `[B,nq,T,d]` | b | `ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, eager_attention_forward)` (`:801-814`), scale `d^-0.5`, GQA 비율 `nq/nkv` |
| `transpose(1,2).contiguous()` | `[B,T,nq,d]` | b | SDPA 출력이 q의 물리 layout(`[B,nq,T,d]`)을 따르므로 **복사**가 발생한다 (§3.4) |
| output gate | `[B,T,nq·d]` | b | `attn_output * sigmoid(gate)` |
| `o_proj` | `[B,T,H]` | b | |

`attention_bias=False`, `attention_dropout=0.0`(예시 config). VERIFIED

### 1.3 Linear-attention block (`Qwen3_5GatedDeltaNet`, Gated DeltaNet) — `modeling_qwen3_5.py:499-662`

```python
mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)        # [B, C, T]
z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
b = self.in_proj_b(hidden_states); a = self.in_proj_a(hidden_states)
mixed_qkv = causal_conv1d_fn(mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation)
query, key, value = torch.split(mixed_qkv.transpose(1, 2), [self.key_dim, self.key_dim, self.value_dim], dim=-1)
beta = b.sigmoid()
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
if self.num_v_heads // self.num_k_heads > 1:
    query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
    key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
core_attn_out, last_recurrent_state = torch_chunk_gated_delta_rule(query, key, value, g=g, beta=beta, ..., use_qk_l2norm_in_kernel=True)
core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
output = self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))
```
(발췌·축약, 실제 `:564-661`)

| 단계 | shape | dtype | 비고 |
|---|---|---|---|
| padding mask 곱 (`apply_mask_to_padding_states`, `:236-245`) | `[B,T,H]` | b | 2D mask가 주어지고 padding이 있을 때만 (`create_recurrent_attention_mask`, `masking_utils.py:1474-1504`는 all-ones이면 `None`) |
| `in_proj_qkv` | `[B,T,C]` → transpose `[B,C,T]` | b | 예시 header `[8192, 4096]` |
| `in_proj_z` | `[B,T,Vd]` → `[B,T,Hv,dv]` | b | gated norm의 gate |
| `in_proj_b`, `in_proj_a` | `[B,T,Hv]` | b | |
| depthwise causal conv1d (`conv1d`: `groups=C`, kernel `K`, padding `K−1`) + SiLU | 출력 `[B,C,T]` (torch 경로 내부 `[B,C,T+K−1]` 후 slice) | b | `causal_conv1d_fn` (`:268-288`) |
| split → q, k `[B,T,Hk,dk]`, v `[B,T,Hv,dv]` | view | b | |
| `beta = sigmoid(b)` | `[B,T,Hv]` | b | |
| `g = -exp(A_log)·softplus(a + dt_bias)` | `[B,T,Hv]` | **fp32** | `a.float()`로 fp32 승격 |
| `repeat_interleave(Hv/Hk)` (예시 2) | q, k `[B,T,Hv,dk]` | b | **새 tensor**. transformers가 kernel 호출 전에 head를 맞춘다 |
| delta rule (chunk, 64) | 출력 `[B,T,Hv,dv]` | b(출력), 내부 fp32 | `use_qk_l2norm_in_kernel=True`. 경로별 차이는 §2 |
| gated RMSNorm (`Qwen3_5RMSNormGated`, `:216-233`) | `[N·Hv, dv]` | 내부 fp32, 출력 b | 정규화 → `weight * x.to(b)` → `* silu(z.float())` → `.to(b)` |
| `out_proj` | `[B,T,H]` | b | |

`Qwen3_5GatedDeltaNet`은 `@use_kernel_forward_from_hub("Qwen3_5GatedDeltaNet")`와 `@use_kernelized_func([...])`로 장식되어 있다 (`:499-502`). VERIFIED

### 1.4 MLP — `modeling_qwen3_5.py:823-836`

`down_proj(act_fn(gate_proj(x)) * up_proj(x))`, `act_fn = ACT2FN["silu"]`. `gate/up` 출력 `[B,T,I]` b, 곱 `[B,T,I]` b, 출력 `[B,T,H]`. VERIFIED

### 1.5 모델 수준

- `Qwen3_5TextModel.forward` (`:1236-1306`): embedding `[B,T,H]` b → position_ids가 없으면 `arange(T)`를 `[4,B,T]`로 expand (text 1 + mrope 3) → mask dict `{"full_attention": create_causal_mask(...), "linear_attention": create_recurrent_attention_mask(...)}` → 레이어 → final `norm`. VERIFIED
- **`use_cache` 기본값 함정**: `use_cache`가 `True`이고 `past_key_values`가 없으면 학습 forward에서도 `DynamicCache(config=self.config)`를 만든다 (`:1255-1256`). config의 `use_cache=True`이므로 호출자가 `use_cache=False`를 넘기지 않으면 학습 중에도 cache가 생긴다. TRL SFT는 `inputs["use_cache"] = False`를 넣는다 (trl==1.14.1 `trainer/sft_trainer.py:1795`). gradient checkpointing이 켜지면 `GradientCheckpointingLayer.__call__`이 레이어에 `use_cache=False`, `past_key_values=None`을 넘긴다 (`modeling_layers.py:53-109`). VERIFIED
- `Qwen3_5ForConditionalGeneration.forward` (`:1803-1894`)는 text 경로가 동일하고 `logits = self.lm_head(hidden_states[:, slice_indices, :])`로 logits를 만든다. `logits_to_keep=0`(기본)이면 전 위치. VERIFIED
- vision tower는 `pixel_values`/`pixel_values_videos`/`mm_encoder_outputs`가 있을 때만 호출된다 (`:1611-1623`, §9). VERIFIED

---

## 2. Kernel 선택 (fla / causal-conv1d / torch fallback)과 경로별 메모리

### 2.1 선택 메커니즘

linear-attn의 네 함수는 `use_kernel_func_from_hub_with_fallback`으로 장식되어 있다.

| 함수 (transformers 이름) | fallback package | 내부 경로 | hub kernel (`kernels`) | 위치 |
|---|---|---|---|---|
| `causal_conv1d_fn` | `causal_conv1d` | `causal_conv1d.causal_conv1d_fn` | `kernels-community/mamba-ssm` v3 (cuda/xpu, TRAINING·INFERENCE) | `modeling_qwen3_5.py:268`, `hub_kernels.py:185-210` |
| `causal_conv1d_update` | `causal_conv1d` | 동일 | 동일 | `:248`, `hub_kernels.py:211-236` |
| `torch_chunk_gated_delta_rule` → `chunk_gated_delta_rule` | `fla` | `fla.ops.gated_delta_rule.chunk_gated_delta_rule` | `kernels-community/fla` v1 (cuda/xpu, TRAINING·INFERENCE) | `:299`, `hub_kernels.py:67-76, 237-262` |
| `torch_recurrent_gated_delta_rule` → `fused_recurrent_gated_delta_rule` | `fla` | `fla.ops.gated_delta_rule.fused_recurrent_gated_delta_rule` | `kernels-community/fla` (cuda/xpu, **INFERENCE만**: "no backward implementation") | `:436`, `hub_kernels.py:263-279` |
| `Qwen3_5RMSNormGated` (class) | — | — | `RMSNormGated` → `kernels-community/fla` `FusedRMSNormGated` | `:216`, `hub_kernels.py:511-537` |
| `Qwen3_5GatedDeltaNet` (class 전체) | — | — | `Atlas-Inference/gdn` — **CUDA capability 12.1(GB10) 또는 ROCm gfx1151에서만** | `:499`, `hub_kernels.py:162-183` |

결정 규칙 — transformers==5.18.0 `integrations/hub_kernels.py:984-1037 (use_kernel_func_from_hub_with_fallback)`:

```python
try:
    module = importlib.import_module(package)
    implementation = resolve_internal_import(module, full_func_path)
    if implementation is None and full_module_path != package:
        module = importlib.import_module(full_module_path)
        implementation = getattr(module, func_name, None)
except Exception:
    implementation = torch_function
...
kwargs = {k: v for k, v in kwargs.items() if k in applicable_params}
return implementation(*args, **kwargs)
```

- 우선순위는 docstring대로 "1. Hf kernels (if requested) 2. Original package 3. Torch only path"다. package 해석은 **modeling 모듈 import 시점**에 한 번 일어나며 device를 보지 않는다. `fla`(=`fla-core`)나 `causal_conv1d`가 import되면 그것을 쓰고, 아니면 torch 함수를 쓴다(이때 "falling back to its reference PyTorch implementation" 경고를 한 번 남김, CPU 실행에서 확인). VERIFIED
- hub kernel은 `kernels` 패키지(버전 `0.17.0 ≤ v < 0.18.0`)가 있고 `USE_HUB_KERNELS`가 참이며 사용자가 `from_pretrained(..., use_kernels=True)`(또는 `kernel_config`)로 kernelize를 요청할 때만 적용된다 (`hub_kernels.py:61-62, 83-140`; `modeling_utils.py:4095, 4237-4342`). `kernels`가 없으면 데코레이터는 identity stub이다 (`hub_kernels.py:740-760`). VERIFIED(소스), 실제 교체 동작은 INFERRED(`kernels` 미설치)
- 학습 forward(`use_cache=False` 또는 첫 forward)는 항상 chunk 경로를 탄다. recurrent(`fused_recurrent`/torch recurrent) 경로는 cache에 이전 상태가 있고 `seq_len == 1`인 decode에서만 쓴다 (`modeling_qwen3_5.py:560-562, 573-582, 624-649`). VERIFIED
- **packing 주의**: transformers는 `cu_seqlens=kwargs.pop("cu_seq_lens_q", None)`을 넘기지만 torch fallback 함수 시그니처에는 `cu_seqlens`가 없어 wrapper가 이를 버린다 (`applicable_params` 필터). torch `causal_conv1d_fn`도 `seq_idx`를 쓰지 않는다. 따라서 padding-free/packing 입력에서 torch 경로는 sequence 경계에서 conv·recurrent 상태가 섞인다. 메모리 문제는 아니지만 "torch fallback + packing" 조합은 지원 불가로 표시해야 한다. 소스 VERIFIED, 수치적 영향은 INFERRED

### 2.2 torch fallback chunk 경로 (`torch_chunk_gated_delta_rule`, `modeling_qwen3_5.py:299-433`)

- q, k, v, beta, g를 `[B,heads,T,·]`로 transpose하며 **fp32 contiguous로 변환**하고(`:337-340`), q/k에 fp32 l2norm, q에 `dk^-0.5`를 곱한 뒤, 길이를 64의 배수로 **zero-padding**한다(`:350-352`, `pad_size=0`이어도 `F.pad`가 새 tensor를 만든다는 것이 측정식과 일치).
- chunk 내부 양: `pairwise_decay`, `k_beta@kᵀ`, `q@kᵀ`, `ut_system`, `intra_chunk_attn`이 모두 `[B,Hv,nc,64,64]` fp32, `torch.linalg.solve_triangular` 두 번(`:393-395`).
- chunk 간 순차 scan(`:417-424`)에서 매 chunk의 recurrent state `S_i` `[B,Hv,dk,dv]` fp32가 matmul에 의해 **saved**된다. 즉 chunk 수만큼 state가 쌓인다: `nc · f · B·Hv·dk·dv` (예시 `T=4096`: 64 × 2 MiB = 128 MiB/레이어).
- 마지막 상태는 `output_final_state=False`(cache 없음)이면 버려지고, 그 branch만 저장하던 tensor(예: chunk가 1개일 때 `key_decayed`)는 forward 중에 해제된다. cache가 있으면(`use_cache=True`) 마지막 state-update branch가 cache buffer의 graph로 남아 그 tensor들이 유지된다.
- 측정으로 검증한 레이어 saved byte 식(§4.2)에서 linear-attn mixer는 예시 모델 `B=1,T=4096`에서 **1,444.8 MiB**(autocast 시 1,616.8 MiB)로, full-attn mixer(369.6 MiB)의 약 4배다. VERIFIED(CPU 측정, dtype·shape 규칙은 device 무관)

### 2.3 fla + causal-conv1d 경로 (CUDA 전용, INFERRED)

`fla-core==0.5.2` `fla/ops/gated_delta_rule/chunk.py:253-338 (ChunkGatedDeltaRuleFunction)`:

```python
if use_qk_l2norm_in_kernel:
    q, q_rstd = l2norm_fwd(q)
    k, k_rstd = l2norm_fwd(k)
...
g, o, A, final_state, initial_state, g_input = chunk_gated_delta_rule_fwd(...)
ctx.save_for_backward(q, q_rstd, k, k_rstd, v, g, beta_raw, beta, A, initial_state,
                      cu_seqlens, chunk_indices, g_input, A_log, dt_bias)
```

- saved: l2-normalized q, k (`l2norm_fwd`의 `y = torch.empty_like(x)` → 입력 dtype b, `[B,T,Hv,dk]` — transformers가 `repeat_interleave`로 Hv head를 만든 뒤 호출하므로 Hk가 아니라 **Hv** 기준), `rstd` fp32 `[B·T·Hv]` 두 개 (`fla/modules/l2norm.py:157-168`), v, chunk-local cumsum `g` fp32 `[B,T,Hv]`(`chunk_local_cumsum(..., output_dtype=torch.float)`), beta, `A = torch.zeros(B,T,HV,64, dtype=k.dtype)` (`fla/ops/gated_delta_rule/chunk_fwd.py:382`). INFERRED
- 저장하지 않는 것: chunk state `h` `[B,NT,HV,K,V]`(k dtype, `fla/ops/common/chunk_delta_h.py:703-708`), `w`, `u`, `v_new`, 출력 `o`. backward에서 다시 계산한다(`chunk_gated_delta_rule_bwd`). 따라서 chunk-state 비용은 saved가 아니라 **transient**다 (`T=4096`: `64·32·128·128·2` B = 64 MiB). INFERRED
- `@input_guard`가 모든 tensor 인자를 contiguous로 만든다 (`fla/utils/_decorators.py:97-140`). transformers가 넘기는 v는 conv 출력의 split view라 non-contiguous이므로 **`[B,T,Hv,dv]` b 복사본**이 저장된다. INFERRED
- `causal-conv1d==1.7.0` `causal_conv1d/causal_conv1d_interface.py:9-61 (CausalConv1dFn)`: `ctx.save_for_backward(x, weight, bias, seq_idx, initial_states)`, SiLU는 kernel 안에서 융합하고 출력은 저장하지 않는다. torch 경로가 저장하던 conv 출력(`b·B·C·(T+K−1)`)이 사라진다. INFERRED
- `fla`는 `@dispatch('gated_delta_rule')`로 다른 backend(`flash_qla`, `triton_ascend`)를 고를 수 있다 (`fla/ops/backends/__init__.py:160-190`). 기본 Triton 경로가 아닌 backend의 saved set은 UNKNOWN.
- 결과: 예시 모델 linear-attn mixer saved는 **499.2 MiB**(`T=4096`, full FT)로 torch 경로의 약 1/3이다. 남는 큰 항목은 torch로 남아 있는 gated RMSNorm(256.5 MiB)이다. INFERRED

### 2.4 bf16 autocast가 torch 경로를 바꾼다

- TRL 기본 `bf16=True`(trl==1.14.1 `trainer/base_config.py:67-73, 105` — `fp16`이 아니면 `bf16=True`) → `Accelerator(mixed_precision="bf16")` → non-DeepSpeed이면 CUDA에서 `native_amp=True` (accelerate==1.15.0 `accelerator.py:586-597`) → `prepare_model`이 `model.forward`를 `torch.autocast`와 `convert_outputs_to_fp32`로 감싼다 (`accelerator.py:1824-1835`). VERIFIED(소스), CUDA 실행 INFERRED
- torch v2.14.1 autocast 목록에서 `matmul`, `mm`, `bmm`, `linear`, `conv1d`, `scaled_dot_product_attention`은 CUDA·CPU 모두 lower-precision이다 (`aten/src/ATen/autocast_mode.h:819-852`, `autocast_mode.cpp:338-363`). 이 모델에서 fp32-list 연산(`exp`, `pow`, `rsqrt`, `softplus`, `log_softmax`, `nll_loss`, `cumsum`, `sum`)은 이미 fp32 입력을 받으므로 CPU autocast가 CUDA autocast와 같은 dtype 결정을 낸다. VERIFIED(소스 대조)
- 그 결과 torch delta-rule의 fp32 matmul 피연산자가 **사용마다 bf16 cast 복사본**으로 저장된다(state는 chunk마다 2번 cast, `v_new`·`key` slice도 사용마다 cast). CPU autocast 측정으로 식을 따로 세웠고 216개 점에서 byte 단위로 일치했다 (§4.2). 예시 모델에서는 chunk가 많아(64) autocast 쪽이 오히려 172 MiB 크다. VERIFIED(CPU autocast 측정)
- full-attn·MLP·norm의 saved set은 autocast에서 변하지 않았다(tiny 측정 L2 872,288 B 동일). VERIFIED

### 2.5 decode 경로

- `causal_conv1d_update`(torch, `:248-265`): `cat([conv_state, x])` → `[B,C,K+1]` transient 후 `conv_state.copy_`. torch recurrent(`:436-496`)는 q/k/v를 fp32로 만들고 state `[B,Hv,dk,dv]` fp32 연산을 토큰마다 한다. fla가 있으면 `fused_recurrent_gated_delta_rule`, causal_conv1d가 있으면 그 `causal_conv1d_update`를 쓴다(package fallback은 학습/추론을 구분하지 않음, hub kernel만 INFERENCE 제한). VERIFIED(소스)
- fla의 final state는 `dtype=torch.float32`로 할당된다 (`fla/ops/common/chunk_delta_h.py:704-707`, `fused_recurrent.py:211-213`). torch 경로도 fp32 state를 낸다. 따라서 cache의 recurrent state는 어느 경로든 fp32다 (§7). torch 경로 VERIFIED, fla INFERRED

---

## 3. Full attention: SDPA backend와 eager

### 3.1 transformers의 SDPA wrapper — `integrations/sdpa_attention.py:29-36, 79-170`

```python
def use_gqa_in_sdpa(attention_mask, key, value) -> bool:
    if _is_torch_xpu_available:
        return _is_torch_greater_or_equal_than_2_8
    elif _is_torch_mps_available:
        return _is_torch_greater_or_equal_than_2_13
    return attention_mask is None and key.shape[-1] == value.shape[-1] <= 256
...
if hasattr(module, "num_key_value_groups") and module.num_key_value_groups > 1:
    if not use_gqa_in_sdpa(attention_mask, key, value):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)
    else:
        sdpa_kwargs = {"enable_gqa": True}
is_causal = q_length > 1 and attention_mask is None and is_causal
```

- CUDA(Linux)에서는 **mask가 없을 때만** `enable_gqa=True`이고, mask가 있으면 `repeat_kv`로 K/V를 `[B,nq,T,d]`로 **실제 복사**한다(`reshape` of expand). VERIFIED
- macOS 함정: `_is_torch_mps_available`은 tensor device가 아니라 **머신**의 MPS 가용성으로 결정된다. 이 Mac에서는 CPU tensor라도 mask가 있을 때 `enable_gqa=True`가 된다(측정에서 K/V가 nkv head로 저장됨). CUDA 규칙을 재현하려면 이 flag를 False로 바꿔야 했다(`e3b_sdpa_cuda_emul.py`). VERIFIED
- causal mask 생성(`masking_utils.py:237-284, 378-541, 870-1001`): SDPA는 padding이 없고(`fast_all(padding_mask)`) packed sequence가 감지되지 않으면 mask를 **만들지 않고** `is_causal=True`를 쓴다. padding이나 packing이 있으면 `[B,1,T,T]` **bool**(1 byte/원소) mask를 한 번 만들어 모든 full 레이어가 공유한다. eager는 항상 `[B,1,T,T]` 부동소수(입력 dtype, b)로 만든다(`eager_mask`, `:544-610`). VERIFIED

### 3.2 torch 2.14.1 CUDA backend 선택 (소스, INFERRED)

- 기본 순서는 flash → mem-efficient → math이며, `check_prefer_cudnn_attention()`이 참(cuDNN > 9.15, major 9/10 = Hopper/Blackwell 데이터센터, env `TORCH_CUDNN_SDPA_DEPRIORITIZED` 미설정)이면 cuDNN을 맨 앞에 둔다 — torch v2.14.1 `aten/src/ATen/native/transformers/cuda/sdp_utils.cpp:94-135, 1184-1247`.
- flash: mask 불가(`check_for_attn_mask`), head_dim ≤ 256, **GQA 지원**(`backend_supports_grouped_query_attention = true`, `:1082-1095`). sm86/89/120/121에서 학습 시 head_dim ∈ (192,224] 또는 (head_dim > 224 이고 dropout > 0)만 금지한다 (`:530-563`). 따라서 Qwen3.5의 `d=256`, `attention_dropout=0`은 RTX 30/40 계열에서도 flash로 학습된다. bf16 flash는 sm80 이상 필요.
- mem-efficient: mask 지원, **CUDA에서 GQA 지원**(`supports_gqa = true`, ROCm은 false, `:1146-1156`). sm80 미만은 bf16 불가(`:1162-1183`) → 그 경우 math backend로 떨어져 O(T²)가 된다.
- dispatch(`attention.cpp:816-890`): bool mask는 `convert_boolean_attn_mask`로 **query dtype의 additive mask**(`where(mask, 0, -inf)`)가 된다(`:579-592`). mem-efficient는 이를 정렬 padding 후 `[B,nq,T,T]`로 expand(view)한다(`:624-639`).
- 이 판정은 torch 소스 정독이며, 설치본 git commit이 tag `v2.14.1`과 같음을 확인했다(GitHub API ref → `5c48869…`).

### 3.3 backend별 saved tensor

derivatives: torch v2.14.1 `tools/autograd/derivatives.yaml:2931-2941, 2955-2957`. output shape: torch==2.14.1 `_meta_registrations.py:6268-6324 (flash), 6381-6422 (cudnn), 6537-6563 (cpu flash), 6657-6700 (efficient)`.

| backend | saved (bf16 입력, causal, mask 없음) | logsumexp | T² 항 | 근거 |
|---|---|---|---|---|
| CUDA flash | q, k, v(GQA면 nkv head 그대로), out, lse, rng_state | `[B,nq,T]` fp32 | 없음 | INFERRED(소스) |
| CUDA cuDNN | q, k, v, out, lse, (attn_bias) | `[B,nq,T,1]` fp32 | 없음 | INFERRED(소스) |
| CUDA mem-efficient | q, k, v, **attn_bias**, out, lse, philox | `[B,nq,ceil(T/32)·32]` fp32 | mask가 있으면 additive mask `b·B·T²`(레이어마다 새로 변환·저장) | INFERRED(소스) |
| CPU flash (이 측정 환경) | q, k, v, out, lse, attn_mask | `[B,nq,T]` fp32 | mask가 있으면 `b·B·T²` | VERIFIED(측정: `ScaledDotProductFlashAttentionForCpuBackward0`, `_fused_sdp_choice=1`) |
| math (CPU·CUDA 공통 composite) | fp32 q(scaled), fp32 expand된 K, V, fp32 softmax | 없음 | `f·B·nq·T²` | VERIFIED(CPU에서 `sdpa_kernel([SDPBackend.MATH])` 측정) |
| eager | q, repeat_kv된 K, V (`b·N·nq·d` 각), softmax fp32, probs b | 없음 | `(f+b)·B·nq·T²` = `6·B·nq·T²` | VERIFIED(측정, 최소제곱 계수 b=36=6·nq) |

- mask가 있는 CUDA 경로(배치 padding, DPO처럼 chosen/rejected를 합친 배치)는 레이어마다: K/V가 nq head로 확장되어 `2·b·N·(nq−nkv)·d` 증가 + additive mask `b·B·T²` 저장. 공유 bool mask `B·T²` 1개는 forward 동안(GC면 checkpoint kwargs로 backward까지) 산다. tiny 측정: B=2,T=100에서 +40,000 B(=2·B·T²) + K/V 확장 2×64,000 B. VERIFIED(CPU, CUDA 규칙 재현)
- 예시 모델 `B=2,T=4096` padding 배치: full 레이어당 additive mask 64 MiB + K/V 확장 96 MiB, 공유 bool mask 32 MiB. 산술

### 3.4 SDPA 출력 layout과 "contiguous 복사"

- Qwen3.5는 partial rotary의 `torch.cat`이 q를 `[B,nq,T,d]` 물리 layout으로 만든다. SDPA 출력은 q의 stride를 따르므로(CPU flash `empty_like(query)`, CUDA flash `out = at::empty_like(q_padded)` 후 `output.transpose(1,2)` — `flash_api.cpp:480-501`, `attention.cu:942-972`) `transpose(1,2).contiguous()`가 **복사본 `b·N·nq·d`**를 만들고, 이 복사본을 output gate의 `mul`이 저장한다. VERIFIED(CPU), CUDA INFERRED
- Llama/Qwen3는 q가 `[B,T,nq,d]` layout을 유지해 복사가 없고 `o_proj` 입력이 SDPA 출력과 storage를 공유한다(dense 측정에서 `o_proj` 별도 항목 없음). VERIFIED

---

## 4. 실험: tiny random-init 모델의 saved-tensor·live-memory 측정

### 4.1 방법

- harness `vfmem.py`: (a) 모든 module에 pre/post forward hook을 걸어 module stack을 유지하고, `TorchFunctionMode`로 현재 torch 함수 이름을 기록하며, `saved_tensors_hooks`의 pack hook에서 `(module, func, shape, dtype, untyped_storage().nbytes(), data_ptr, weakref)`를 남긴다. (b) **forward가 끝난 시점에 살아 있는** tensor만 세고(죽은 branch 제외), parameter/buffer storage는 제외하고, `data_ptr`로 dedup한다. 모두 동시에 살아 있으므로 주소 재사용 오류가 없다(weakref 필터 없이 `use_cache=False`로 셀 때는 forward 중 해제된 tensor의 주소 재사용으로 9건이 틀렸다. 필터 후 남은 18건은 단일 chunk에서 `key_decayed`가 해제되는 것을 식에 반영해 0건이 되었다). (c) `torch.profiler.profile(profile_memory=True, record_shapes=True, with_stack=True)`의 `MemoryProfile.timeline`으로 CPU allocator 수준 live bytes를 재구성하고, 고유 크기 sentinel 할당으로 구간(forward/backward)을 표시한다. profile 이전에 만든 tensor의 해제는 무시한다.
- 모델: `Qwen3_5ForCausalLM`(text) tiny config A `{H=96, I=176, nq=6, nkv=2, d=40, r=10, Hk=3, Hv=9, dk=24, dv=28, K=4, V=1000}`, config B `{H=112, I=200, nq=4, nkv=1, d=48, Hk=2, Hv=6, dk=20, dv=36, K=3, 4 layers}`, dense Qwen3/Llama tiny 2종, 그리고 **¼ 스케일 실측 비율 config** `{H=1024, I=3072, nq=4, nkv=1, d=256, Hk=4, Hv=8, dk=dv=128, K=4}`(모든 항이 정확히 1/4로 줄어드는 config, 측정값이 식과 일치하고 실제 차원 식/¼식 = 3.9995·3.9878). 전부 bf16(`model.to(torch.bfloat16)`), `attn_implementation="sdpa"`(CPU flash), 별도 표기 없으면 `use_cache=False`, full fine-tune.

### 4.2 검증된 레이어별 saved byte 식 (gradient checkpointing 없음)

검증 범위: Qwen3.5 torch 경로 216점(2 config × B∈{1,2,3} × T∈{40,64,100,128,150,200} × layer × `use_cache`∈{F,T}) 일치, autocast 216점 일치, dense 120점 일치, ¼ 스케일 2점 일치. 오차 0 byte. VERIFIED

**공통 부품**

```
RMSNorm_q35(rows, w)   = f·rows·w + f·rows + f·w + [f·rows·w  if norm weight trainable]
RMSNorm_llama(rows, w) = f·rows·w + f·rows       + [b·rows·w  if norm weight trainable]
MLP(N)                 = b·N·H (gate/up 입력) + 4·b·N·I   (SiLU 입력, SiLU 출력, up 출력, 곱=down 입력)
```

**Qwen3.5 full-attention mixer** (SDPA flash류, mask 없음; cos/sin `2·b·N·r`은 모든 full 레이어가 같은 storage를 공유하므로 모델당 1번)

```
S_fullmix = b·N·H
          + RMSNorm_q35(N·nq, d) + RMSNorm_q35(N·nkv, d)
          + b·N·nq·d (q) + 2·b·N·nkv·d (k, v) + f·N·nq (lse) + b·N·nq·d (out)
          + b·N·nq·d (sigmoid(gate)) + b·N·nq·d (contiguous 복사) + b·N·nq·d (o_proj 입력)
```

**Qwen3.5 linear-attention mixer, torch fallback, autocast 없음**

```
S_linmix = b·N·H + b·N·C + b·B·C·(T+K−1) + b·N·Hv + 2·f·N·Hv
         + 2·(f·N·Hv·dk + f·N·Hv)                                   # fp32 l2norm q,k
         + f·N·Vd + f·N·Hv + [b·N·Vd if trainable] + 2·f·N·Vd + b·N·Vd  # gated RMSNorm
         + b·N·Vd                                                   # out_proj 입력
         + f·Np·Hv + f·Np·Hv·dv + f·Np·Hv·dk                        # padded beta, v, k
         + 5·f·Np·Hv·64                                             # [.,64,64] 5개
         + 3·f·Np·Hv·dk + [f·Np·Hv·dk if nc≥2 or cache] + [f·Np·Hv·dk if nc≥2]
         + 3·f·Np·Hv + f·B·Hv·nc + 2·f·Np·Hv·dv
         + nc·f·B·Hv·dk·dv                                          # chunk state S_0..S_{nc-1}
         + 64·64 + 2·f·Hv
```

autocast 변형(`q35_linear_attn_torch_autocast`)과 fla 변형(`q35_linear_attn_fla`, INFERRED)은 `formulas.py`에 같은 형식으로 있다. 레이어 합계는 `2·RMSNorm_q35(N,H) + MLP(N) + mixer`.

**최소제곱 적합으로 본 구조**(`e13_gc_and_fit.py`, basis `[B·T, B·T², B·Tp, B·nc, B, 1]`, config A, B∈1..4 × T∈{70..300}, max 오차 1e-4 B):

| 레이어 | a (B·T 계수, byte/token) | b (B·T²) | padded chunk 계수 (64c+d per chunk) | 상수 |
|---|---|---|---|---|
| full, sdpa | 8,672 | 0 | 0 | 1,088 (`(1+w)` fp32 상수) |
| full, eager | 8,808 | **36 = 6·nq** | 0 | 1,088 |
| linear, sdpa(무관) | 11,382 (식과 정확히 일치) | 0 | 1,296,036/chunk = 20,250.6 byte/padded token | 4,936 (= 64·64 mask + 상수) |

### 4.3 LoRA(PEFT 0.21.2)와 frozen base

`peft/tuners/lora/layer.py:1074-1116 (Linear.forward)`: `result = base_layer(x)`, 이어서 `x = self._cast_input_dtype(x, lora_A.weight.dtype)` 후 `result + lora_B(lora_A(dropout(x))) * scaling`. `get_peft_model`의 `autocast_adapter_dtype=True`(기본, `peft/mapping_func.py:110`)가 bf16 adapter를 fp32로 올린다(`tuners_utils.py:2705-2750`). TRL SFT는 비양자화 모델에 이 기본값을 쓰고(ZeRO-3만 예외), 양자화 모델은 wrapping 뒤 학습 파라미터를 bf16으로 되돌린다 (trl==1.14.1 `trainer/sft_trainer.py:1120-1166`). VERIFIED. (adapter dtype 결정의 상세는 `docs/research/loading-quantization-peft.md` 참고)

측정 규칙(B=1,T=100, r=8, `all-linear`, 레이어 1·2 기준, `e4_lora.py`, `e4b_lora_diff.py`, `e11_autocast.py`):

| 상황 | Linear 입력 저장 | lora_B 입력 | 기타 | 근거 |
|---|---|---|---|---|
| frozen base Linear, LoRA 없음 | **저장 안 함**(weight만 참조) | — | 입력이 다른 소비자에게도 필요 없으면 해제 | VERIFIED |
| LoRA, adapter bf16 (TRL QLoRA 경로) | base 입력 storage를 **공유**(같은 입력을 읽는 모듈끼리 1개) | `b·N·r`/모듈 | | VERIFIED (L1 diff +12,800−3,636 B 정확히 설명) |
| LoRA, adapter fp32, autocast 없음 | 모듈마다 `x.to(fp32)` 복사 **`f·N·in`/모듈**(공유 안 됨) | `f·N·r` | | VERIFIED (L1 +290,400 B 예측=측정) |
| LoRA, adapter fp32, **bf16 autocast** (TRL 비양자화 LoRA의 CUDA 기본) | autocast cast 복사 **`b·N·in`/모듈**(공유 안 됨), bf16 base 입력은 저장 안 됨 | `b·N·r` | cast된 weight 복사 `b·(in·r + r·out)`/모듈 저장 | VERIFIED(CPU autocast) |
| `lora_dropout>0` | 모듈마다 dropout 출력 + mask | | CPU는 mask가 입력 dtype(`_dropout_impl`), CUDA `native_dropout`은 bool 1 byte | CPU VERIFIED, CUDA INFERRED |

- "all-linear"는 conv1d를 대상으로 하지 않지만 **frozen conv1d도 입력을 저장**한다(ConvolutionBackward). frozen `A_log/dt_bias`이면 softplus 출력(`f·N·Hv`)과 `[Hv]` tensor 2개가 빠진다. frozen RMSNorm은 fp32 normalized 항이 빠진다. VERIFIED
- gradient checkpointing 없이 LoRA를 쓰면 첫 레이어의 input_layernorm은 입력이 grad를 요구하지 않아 아무것도 저장하지 않는다(L0가 L1보다 39,184 B 작음). TRL은 GC+PEFT일 때 `enable_input_require_grads()`를 호출해 이 차이를 없앤다 (`sft_trainer.py:1157-1158`). VERIFIED
- fp32 weight + bf16 autocast(마스터 fp32 full FT): Linear/conv마다 **weight의 bf16 cast 복사본이 saved**된다(tiny 측정 합계 743,136 B = 2 byte × 행렬 파라미터 수, lm_head 포함). 예시 모델에서는 lm_head만 1.89 GiB다. 입력도 Linear마다 따로 cast 저장된다. VERIFIED(CPU autocast)

### 4.4 Gradient checkpointing

- transformers 5.18 기본은 `use_reentrant=False`이고(`modeling_utils.py:3138-3139`), 새 옵션 `every_n_layers`, `offload`(saved input을 pinned host로, `save_on_cpu`)가 있다 (`:3113-3179`). Trainer는 `gradient_checkpointing_kwargs`에서 두 키를 꺼내 전달한다 (`trainer.py:1489-1501`). TRL 모든 config의 기본은 `gradient_checkpointing=True` (`trl/trainer/base_config.py:61-66`). VERIFIED
- checkpoint 단위는 `GradientCheckpointingLayer`(=decoder layer)다. non-reentrant checkpoint는 레이어 입력 hidden state를 `save_for_backward`로 보관하므로 saved-tensor hook에서도 보인다. 측정: GC 시 hook 합계 = `Σ_ckpt layers b·N·H` + final norm + lm_head 입력 + `f·N·V`(log_softmax) + labels/indices (`e13_gc_and_fit.py`, label 복사 8·B byte 차이만 존재). position_embeddings(cos/sin)와 4D mask 같은 kwargs는 recompute closure(`partial(super().__call__, **kwargs)`, `modeling_layers.py:109`)가 잡아 둔다. VERIFIED
- `every_n_layers` 카운터는 모델의 **모든** `GradientCheckpointingLayer`를 `self.modules()` 순서로 센다(`modeling_utils.py:3181-3214`). `Qwen3_5ForConditionalGeneration`에서는 vision block(예시 27개)이 먼저 세어지므로 text layer `i`는 `(27 + i) % n == 0`일 때 checkpoint된다(tiny vision depth 3으로 확인: text layer0=False, layer1=True). VERIFIED
- 측정한 transient 비율(`e7_layer_peaks.py` tiny, `e12_quarter_profiles.py` ¼ 스케일, CPU allocator, `S`=해당 레이어 saved 식):

| 조건 | forward 중 (peak − retained) | backward 중 (peak − retained) | 비고 |
|---|---|---|---|
| GC 없음, full FT/LoRA | 0.04–0.06 S (fp32 adapter+autocast 0.19 S) | LoRA 0.12 S; full FT는 grad 할당이 activation 해제와 겹침 | |
| GC, LoRA bf16 adapter, autocast, T=256 / 1024 | 0.39 S / 0.32 S | **1.14 S / 1.17 S** | 1개 레이어 recompute + backward 임시값 |
| GC, LoRA fp32 adapter, autocast, T=256 | 0.42 S | **1.36 S** | per-module cast 때문 |
| GC, full FT, autocast, T=1024 | 0.26 S | 1.51 S (grad 포함), 0.95 S(grad 제외 추정) | full FT는 param grad가 누적 |

CUDA caching allocator의 block 반올림·단편화는 포함되지 않는다. VERIFIED(CPU 측정), CUDA 비율은 UNKNOWN(보정 필요)

### 4.5 CPU 측정이 CUDA와 다른 지점

1. SDPA backend: CPU는 flash-for-CPU. CUDA flash/cuDNN과 saved set이 같은 구조(q,k,v,out,lse)이고 lse shape만 다르다(mem-efficient는 32 배수 padding). INFERRED
2. MPS flag 때문에 이 Mac에서는 mask가 있어도 GQA가 확장되지 않는다. Linux CUDA 규칙은 확장한다(§3.1). VERIFIED
3. torch delta-rule 대신 fla/causal-conv1d가 설치된 CUDA 환경에서는 §2.3의 식을 쓴다. INFERRED
4. dropout mask dtype(CPU 입력 dtype vs CUDA bool). INFERRED
5. transient 비율에 allocator overhead가 없다. UNKNOWN
6. autocast: 이 모델의 관련 op는 CPU/CUDA 목록이 같다(§2.4). VERIFIED

---

## 5. RMSNorm / SiLU / gated MLP / residual의 saved tensor

| 모듈 | saved | dtype | 조건 | 근거 |
|---|---|---|---|---|
| `Qwen3_5RMSNorm` | `x.float()` 복사(pow·mul이 공유), `rsqrt` `[rows,1]`, `(1+w).float()` `[w]` | fp32 | 항상(입력이 grad를 요구할 때) | VERIFIED |
| | 정규화 결과(곱하기 전) | fp32 | norm weight가 trainable일 때만 | VERIFIED |
| Llama/Qwen3 `RMSNorm` | `x.float()` 복사, `rsqrt` | fp32 | 항상 | VERIFIED |
| | `hidden_states.to(input_dtype)` | b | weight trainable일 때만 | VERIFIED |
| `Qwen3_5RMSNormGated` | fp32 입력 복사, rsqrt, `z.float()`(SiLU 입력), `silu(z)` fp32, `weight·x̂` | fp32, fp32, fp32, fp32, b | 항상 | VERIFIED |
| | `x̂.to(b)`(weight 곱 전) | b | weight trainable일 때만 | VERIFIED |
| SiLU (`ACT2FN["silu"]`) | 입력 | 입력 dtype(b) | 항상 | VERIFIED |
| gated MLP `act(g)*u` | 두 피연산자(SiLU 출력, up 출력) | b | 항상 | VERIFIED |
| `down_proj` 입력(곱) | | b | down이 trainable/LoRA일 때만 | VERIFIED |
| residual add | 없음 | | | VERIFIED |
| `sigmoid` (gate, beta) | 출력 | b | | VERIFIED |
| `F.softplus` (g 경로) | 입력, (A_log trainable이면) 출력 | fp32 | | VERIFIED |

fp32로 학습(모델 fp32)하면 `x.float()`가 복사하지 않고 입력 자체를 저장하므로 RMSNorm 항이 달라진다(이 보고서의 식은 bf16 전용). INFERRED

---

## 6. LM head와 loss

### 6.1 logits

- `logits = self.lm_head(hidden_states[:, slice_indices, :])`, `slice_indices = slice(-logits_to_keep, None)`(정수) 또는 index tensor. `logits_to_keep=0`이면 전 위치 `[B,T,V]`. hidden slice는 view라 복사가 없다 (`modeling_qwen3_5.py:1875-1879`, CausalLM `:1731-1734`). VERIFIED
- dtype: bf16 모델이면 bf16. bf16 autocast(accelerate native_amp)에서도 `linear`가 lower-precision이므로 bf16이다(측정: autocast 하에서 logits bf16). 그 뒤 accelerate의 `convert_outputs_to_fp32`가 출력 안의 bf16 tensor를 `.float()`로 바꾼 **fp32 logits `f·N·V`**를 output에 넣는다 (accelerate==1.15.0 `utils/operations.py:889-910`, `accelerator.py:1829-1835`). VERIFIED(소스), CUDA INFERRED
- generate prefill은 `logits_to_keep=1`을 넣는다 (`generation/utils.py:2920-2924`). VERIFIED

### 6.2 transformers `ForCausalLMLoss` — `loss/loss_utils.py:32-70`

```python
logits = logits.float()
if shift_labels is None:
    labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
    shift_labels = labels[..., 1:].contiguous()
logits = logits.view(-1, vocab_size)
shift_labels = shift_labels.view(-1)
loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
```

- logits를 **항상 fp32로 복사**하고, shift는 logits가 아니라 labels를 pad해서 하므로 logits slice 복사는 없다. flatten은 view다. `num_items_in_batch`가 있으면 reduction `sum` 후 나눗셈(메모리 동일). VERIFIED
- `cross_entropy` = `log_softmax` + `nll_loss`. saved: log_softmax 출력 `[N,V]` fp32, target int64, total_weight. VERIFIED(측정)
- 측정(`e6_loss.py`, CPU, H=64, V∈{4000,6000,8000}, N∈{512..2000}; 비율이 모든 점에서 같음):

| 구간 | 크기 | 구성 |
|---|---|---|
| forward 피크 | **10·N·V** byte | bf16 logits 2 + fp32 복사 4 + log_softmax 4 |
| forward 후 유지 | 4·N·V (+2 bf16 logits를 output이 들고 있으면 6) | log_softmax(saved) |
| backward 피크 | **12·N·V** (logits 유지 시 14) | saved 4 + nll grad 4 + log_softmax grad 4, 이후 bf16 grad 2로 축소 |

accelerate fp32 변환이 있으면 forward 직후 `compute_loss`가 끝날 때까지 4(lsm)+4(fp32 logits)=8·N·V가 유지되고(forward 피크는 10 그대로), Trainer가 output을 버린 뒤(`trainer.py:2004-2045`, `compute_loss(..., return_outputs=False)`) 4·N·V만 남는다. 연산 op가 같으므로 CUDA에서도 같은 tensor들이 할당된다고 본다. CPU VERIFIED, CUDA INFERRED

### 6.3 TRL SFT `chunked_nll` (SFT 기본)

- `SFTConfig.loss_type` 기본이 `None → "chunked_nll"`(liger 미사용 시) (`trl/trainer/sft_config.py:281, 333-334`). patched forward가 lm_head를 부르지 않고, valid token을 앞으로 모은 뒤 256개씩 `torch.utils.checkpoint`로 감싼 `_chunk`에서 `(h @ w.to(h.dtype).t()).float()` → `log_softmax` → `nll_loss`, argmax, entropy를 계산한다 (`sft_trainer.py:87, 100-232, 235-387`). VERIFIED (상세 수명은 `docs/research/trl-sft-dpo.md` §5)
- 측정(`e6b_chunked.py`, chunk C=256): forward 피크 ≈ **16.1–16.7·C·V**, backward 피크 ≈ **16.7–17.7·C·V**(N과 무관), forward 후 유지 = gather된 hidden 복사 `b·N'·H` + 정렬 index/labels(int64). 예시 V=248,320에서 0.95–1.05 GiB. VERIFIED(CPU)
- 측정 함정: `correct`/`entropy_sum` 출력을 backward 동안 잡고 있으면 non-reentrant checkpoint holder가 살아 recompute된 entropy용 tensor(`8·C·V`/chunk)가 backward 끝까지 누적된다(N=4096에서 137·C·V). TRL `compute_loss`는 이 값들을 `.item()`으로 소비하고 output을 반환하지 않으므로 해당하지 않는다. VERIFIED(CPU, 재현과 해소 모두 측정)

### 6.4 그 밖의 logits 경로 (다른 조사 범위, 포인터만)

- TRL SFT `loss_type="nll"`의 metric 계산은 `outputs.logits[..., :-1, :]`를 `entropy_from_logits`(128행 chunk)로 처리한다(`trl/trainer/utils.py:601-640`). B>1이면 slice가 non-contiguous라 `reshape(-1,V)`가 전체 복사를 만든다(accelerate fp32 logits이면 `4·N·V`). INFERRED
- DPO(비 Liger)는 전 위치 logits를 만들고 `selective_log_softmax_and_entropy`를 쓴다. CUDA에서는 TRL Triton fused kernel(`trl/kernels`)을 쓸 수 있어 saved set이 UNKNOWN이다 (`trl/trainer/utils.py:501-532, 535-597`). → `docs/research/trl-sft-dpo.md`
- GRPO의 `logits_to_keep`(completion만) 사용과 rollout 수명 → `docs/research/trl-grpo.md`

---

## 7. Generation cache (Qwen3.5, transformers 5.18)

- `generate()`는 `cache_implementation`이 없으면 `DynamicCache(config=self.config.get_text_config(decoder=True))`를 만든다 (`generation/utils.py:2261, 2298-2300`). `DynamicCache.__init__`은 `layer_types`를 읽어 `DYNAMIC_LAYER_TYPE_MAPPING`으로 레이어를 만든다: `"full_attention" → DynamicLayer`, `"linear_attention" → LinearAttentionLayer` (`cache_utils.py:1249-1268, 1728-1761, 1807-1849`). `number_of_states = getattr(config, "number_of_conv_states", 1)`. VERIFIED
- `DynamicLayer.update`는 `torch.cat([self.keys, key_states], dim=-2)`로 자란다 (`cache_utils.py:129-148`). 저장되는 K는 k_norm·RoPE를 거친 값이다. decode step마다 해당 레이어 K/V의 이전 버전과 새 버전이 잠깐 공존한다. VERIFIED(소스)
- `LinearAttentionLayer.update_conv_state`는 prefill 입력(conv 이전 `mixed_qkv`, `[B,C,T]`)의 마지막 `conv_kernel_size`(=K, **K−1 아님**) 위치를 `[B,C,K]` buffer에 `copy_`한다. recurrent state는 `zeros_like(recurrent_states)`로 lazy 초기화 후 `copy_` (`cache_utils.py:1029-1117`). VERIFIED
- 실측(`e7_generate_cache.py`, tiny 4-layer `[lin,lin,lin,full]`, greedy):

| B, prompt, new | full layer K/V | linear layer conv_states | recurrent_states |
|---|---|---|---|
| 1, 37, 5 | `keys/values (1,2,41,40)` bf16 | `(1,396,4)` bf16 | `(1,9,24,28)` **float32** |
| 2, 37, 12 | `(2,2,48,40)` bf16 | `(2,396,4)` bf16 | `(2,9,24,28)` float32 |
| 3, 100, 20 | `(3,2,119,40)` bf16 | `(3,396,4)` bf16 | `(3,9,24,28)` float32 |

  cached 길이는 `L = prompt + generated − 1`(마지막 토큰은 다시 넣지 않음). recurrent state dtype은 config의 `mamba_ssm_dtype`과 무관하게 delta-rule이 내는 fp32이다(transformers 5.18은 `mamba_ssm_dtype`을 읽지 않음). VERIFIED
- 예시 모델 sequence 1개당:

```
linear layer : b·C·K + f·Hv·dk·dv = 65,536 + 2,097,152 = 2,162,688 B   (L과 무관)
               × 24 layers = 51,904,512 B = 49.5 MiB
full layer   : 2·nkv·d·b = 4,096 B / token,  × 8 layers = 32,768 B / token
M_cache(L)   = 51,904,512 + 32,768 · L   [byte/sequence]
```

  L=1,024 → 81.5 MiB, 4,096 → 177.5 MiB, 8,192 → 305.5 MiB, 32,768 → 1,073.5 MiB, 262,144 → 8,241.5 MiB. plan §9.7의 `M_KV` 식은 full 8개 레이어에만 적용하고, linear 24개는 상수 state로 따로 더한다(plan §19.3 hybrid fixture와 일치). 산술(식 VERIFIED)
- prefill은 `torch.no_grad`이므로 saved tensor가 없고, transient는 한 레이어 forward 크기(torch 경로면 §4.4의 forward transient) + `[B,1,V]` logits다. INFERRED

---

## 8. Dense decoder 기준 (Qwen3 / Llama, transformers 5.18)

- 구조: q/k/v/o_proj, Qwen3는 head 단위 `q_norm/k_norm`(`Qwen3RMSNorm(d)`), Llama는 없음. RoPE는 full head_dim, cos/sin은 position_ids가 없으면 `[1,T,d]`(B와 무관). `apply_rotary_pos_emb`가 elementwise라 q가 `[B,T,nq,d]` layout을 유지하고 SDPA 출력이 `o_proj` 입력과 storage를 공유한다 (`models/qwen3/modeling_qwen3.py:50-64, 126-170, 211-325`). VERIFIED
- 검증 식(120점 일치, `e8b_dense_grid.py`):

```
S_dense_layer = 2·RMSNorm_llama(N,H) + MLP(N)
              + b·N·H + [RMSNorm_llama(N·nq,d) + RMSNorm_llama(N·nkv,d)  (Qwen3만)]
              + b·N·nq·d + 2·b·N·nkv·d + f·N·nq + b·N·nq·d      # q, k, v, lse, out(=o_proj 입력)
cos/sin       = 2·b·T·d  (모델당 1번, position_ids 미지정 시)
```

- tied embedding(`tie_word_embeddings=True`, `_tied_weights_keys`)은 activation에 영향이 없다. full FT에서 lm_head·embedding grad가 같은 parameter에 누적될 뿐이다(예시 모델은 `false`). VERIFIED(소스)
- dense KV cache: `M_KV = 2 · n_layers · nkv · d · b · L` per sequence (`DynamicLayer`, 같은 cat 성장). plan §19.1의 KV fixture(2 layers, 4 heads, d=128, 2 byte, 1024) = 4,194,304 B와 같은 식이다. VERIFIED

---

## 9. 예시 checkpoint의 vision tower

- `get_safetensors_metadata(REPO, revision=REV)` header만 조회(`e9_vision_params.py`): 4 shard, 760 tensor, 전부 BF16.

| group | params | bytes (BF16) |
|---|---|---|
| `model.visual.*` | **456,010,480** | 912,020,960 |
| `model.language_model.*` (embed 1,017,118,720 포함) | 7,936,684,544 | 15,873,369,088 |
| `lm_head` | 1,017,118,720 | 2,034,237,440 |
| 합계 | 9,409,813,744 | 18,819,627,488 |

  config로 계산한 vision 파라미터(27 block × 15,239,504 + patch_embed 1,770,624 + pos_embed 2,654,208 + merger 40,119,040)와 정확히 같고, meta-device `Qwen3_5ForConditionalGeneration`의 집계와도 같다. text-only `Qwen3_5ForCausalLM`은 8,953,803,264. VERIFIED
- text-only 입력에서 vision tower는 **실행되지 않는다**(tiny CondGen에 pre-hook: 호출 0회, saved 0 byte). activation 0, 상주 weight는 로드했을 때만. VERIFIED
- 로딩 클래스: TRL `create_model_from_path`는 `config.architectures[0]`(`Qwen3_5ForConditionalGeneration`)으로 로드하므로 vision weight가 상주한다 (trl==1.14.1 `trainer/utils.py:1306-1320`). `AutoModelForCausalLM`은 `qwen3_5 → Qwen3_5ForCausalLM`("VLM compatibility")이고 `model.visual.*`를 무시한다 (`models/auto/modeling_auto.py:854`, `modeling_qwen3_5.py:1679`). VERIFIED (상세 로딩 범위는 `docs/research/loading-quantization-peft.md`)

---

## 10. Shape ledger

### 10.1 AllocationSpec 표

열 설명: `saved`=autograd가 backward까지 보관, `lifetime`=생성~해제 구간과 recompute group(`layer[i]` = 해당 decoder layer의 checkpoint 단위), `path`=적용 profile. 예시 열은 B=1, T=4096, full FT(norm weight trainable), bf16.

| # | allocation | shape | dtype | saved | lifetime / recompute group | path | 예시 B=1,T=4096 |
|---|---|---|---|---|---|---|---|
| 1 | input_ids, labels, attention_mask | `[B,T]` | int64 | embedding이 indices 저장 | step | 공통 | 32 KiB 각 |
| 2 | embedding 출력 | `[B,T,H]` | b | 아니오 (GC면 layer0 ckpt 입력) | forward 동안 residual로 전달 | 공통 | 32 MiB |
| 3 | cos, sin | Qwen3.5 `[B,T,r]`×2, dense `[1,T,d]`×2 | b | 예(full 레이어 RoPE mul, storage 공유) | forward~backward, 모델당 1개 | full-attn | 1 MiB |
| 4 | 4D causal mask (SDPA) | `[B,1,T,T]` 또는 None | bool | CPU flash/mem-eff는 변환본 저장 | forward(+GC kwargs) | padding/packing 시만 | None (B=1) |
| 4b | 4D mask (eager) | `[B,1,T,T]` | b | 아니오(add) | forward | eager | (32 MiB) |
| 5 | input/post RMSNorm | x fp32 `[N,H]`, rstd `[N,1]`, `(1+w)` `[H]`, normalized `[N,H]` | f | 예(normalized는 trainable만) | layer[i] | 전 레이어 2개 | 128.03 MiB/norm (frozen 64.03) |
| 6 | norm 출력 = Linear 입력 | `[N,H]` | b | trainable/LoRA 소비자가 있으면 1개(공유) | layer[i] | 전 레이어 | 32 MiB |
| 7 | MLP | gate·up 출력, SiLU 출력, 곱 각 `[N,I]` | b | 예(곱은 down이 학습될 때) | layer[i] | 전 레이어 | 416 MiB (입력 포함) |
| 8 | q_proj 출력(q+gate) | `[N,2·nq·d]` | b | 아니오 | transient | full | 64 MiB |
| 9 | gate reshape 복사 → sigmoid 출력 | `[N,nq·d]` | b | sigmoid 출력 예 | layer[i] | full | 32 MiB |
| 10 | q_norm / k_norm | `[N,nq,d]`/`[N,nkv,d]` fp32 + rstd + normalized | f | 예 | layer[i] | full | 128.25 / 32.06 MiB |
| 11 | RoPE q, k / v | `[B,nq,T,d]`, `[B,nkv,T,d]` / `[B,nkv,T,d]` | b | 예(attention 입력) | layer[i] | full | 32 + 8 + 8 MiB |
| 12 | attention out, lse | `[B,nq,T,d]`, `[B,nq,T]` | b, f | 예 | layer[i] | flash/cuDNN/CPU-flash | 32 MiB + 256 KiB |
| 12m | (mask 시) K/V 확장, additive mask | `[B,nq,T,d]`×2, `[B,1,T,T]` | b | 예 | layer[i] | CUDA mem-eff | (B=2: 96 + 64 MiB) |
| 12e | (eager) softmax, probs | `[B,nq,T,T]` | f, b | 예 | layer[i] | eager | 1,536 MiB |
| 13 | contiguous 복사, o_proj 입력 | `[N,nq·d]` 각 | b | 예 | layer[i] | Qwen3.5 full | 32 + 32 MiB |
| 14 | in_proj_* 입력 | `[N,H]` | b | 예(공유) | layer[i] | linear | 32 MiB |
| 15 | in_proj_qkv 출력 = conv 입력 | `[B,C,T]` view | b | 예(torch conv·causal_conv1d 모두) | layer[i] | linear | 64 MiB |
| 16 | conv 출력(+K−1) | `[B,C,T+K−1]` | b | 예(SiLU 입력) | layer[i] | **torch만** | 64.05 MiB |
| 17 | z | `[N,Vd]` | b | 아니오(fp32 복사본이 저장) | transient | linear | 32 MiB |
| 18 | beta, softplus 입출력 | `[N,Hv]` | b, f | 예 | layer[i] | linear | 0.25 + 1 MiB |
| 19 | repeat_interleave q, k | `[N,Hv,dk]`×2 | b | torch 아니오 / fla는 정규화본 저장 | transient | linear | 64 MiB |
| 20 | delta-rule 내부 (torch) | §4.2 식 (`[Np,Hv,64]` f×5, `[Np,Hv,dk]` f×5, states `nc×[B,Hv,dk,dv]` f ...) | f (+autocast b cast) | 예 | layer[i] | torch | 995.0 MiB (autocast 1,167.0 MiB) |
| 20f | delta-rule (fla) | q,k 정규화 `[N,Hv,dk]` b ×2, rstd, v 복사 `[N,Vd]` b, g f, A `[N,Hv,64]` b | b/f | 예 | layer[i] | fla | 64 + 1 + 32 + 0.5 + 16 = 113.5 MiB |
| 20h | fla chunk state h | `[B,nc,Hv,dk,dv]` | b | 아니오(backward 재계산) | transient | fla | 64 MiB |
| 21 | gated RMSNorm | fp32 입력, rstd, (trainable) b normalized, z fp32, silu f, 곱 b | f/b | 예 | layer[i] | linear | 256.5 MiB |
| 22 | out_proj 입력 | `[N,Vd]` | b | 예 | layer[i] | linear | 32 MiB |
| 23 | LoRA 추가 | §4.3 표 (`b·N·r`/모듈, fp32 adapter면 모듈별 입력 복사) | adapter/autocast dtype | 예 | layer[i] | LoRA | r=16: 1 MiB/layer (+bf16 cast 96–128 MiB/layer for fp32+autocast) |
| 24 | ckpt 입력 | `[N,H]` | b | 예(`save_for_backward`) | forward~해당 layer backward | GC | 32 MiB/layer, 1 GiB/32 layers |
| 25 | recompute 피크 | `≈ k_bwd · S_layer` | | (재계산) | backward 중 1 layer | GC | k=1.14–1.36 측정 |
| 26 | final norm | 행 5와 같음 | f | 예 | backward 끝까지 | 공통 | 128.03 MiB |
| 27 | lm_head 입력 | `[N,H]` | b | lm_head trainable일 때만 | | HF loss | 32 MiB |
| 28 | logits | `[N_k,V]` | b (+accelerate fp32 복사) | 아니오 | output 수명(Trainer가 backward 전 해제) | HF loss | 1,940 MiB (+3,880) |
| 29 | loss fp32 복사, log_softmax | `[N,V]` 각 | f | log_softmax 예 | forward 피크 10NV / backward 12NV | HF loss | 9.47 / 11.37 GiB 피크 |
| 30 | chunked CE | `[256,V]` 단위 | b→f | gather hidden만 | chunk별 (checkpoint) | TRL chunked_nll | 0.95–1.05 GiB 피크 |
| 31 | gen KV (full) | `[B,nkv,L,d]`×2 | b | — | rollout 단계 | generate | 32 KiB/token/seq (8 layers) |
| 32 | gen conv / recurrent (linear) | `[B,C,K]` / `[B,Hv,dk,dv]` | b / **f** | — | rollout 단계, L 무관 | generate | 49.5 MiB/seq |

### 10.2 예시 모델 sanity (B=1, T=4096, layer 합계 = 2 norm + MLP + mixer)

`calc_sanity.py`. "Σ no GC" = 24·S_lin + 8·S_full + final norm + (full FT면 lm_head 입력) + cos/sin + indices. GC retained = 32·`b·N·H` + 같은 바깥 항목. LoRA는 r=16, all-linear(text 8개/7개 모듈).

| mode | linear-attn 경로 | S_lin / layer | S_full / layer | Σ saved, no GC | GC retained | max S_layer |
|---|---|---|---|---|---|---|
| full FT | torch | 2,116.9 MiB | 1,040.6 MiB | 57.90 GiB | 1,185.1 MiB | 2,116.9 MiB |
| full FT | torch + autocast | 2,288.9 MiB | 1,040.6 MiB | 61.93 GiB | 1,185.1 MiB | 2,288.9 MiB |
| full FT | fla (INFERRED) | 1,171.3 MiB | 1,040.6 MiB | 35.74 GiB | 1,185.1 MiB | 1,171.3 MiB |
| LoRA bf16 adapter | torch | 1,957.4 MiB | 833.5 MiB | 52.45 GiB | 1,089.1 MiB | 1,957.4 MiB |
| LoRA bf16 adapter | torch + autocast | 2,129.4 MiB | 833.5 MiB | 56.48 GiB | 1,089.1 MiB | 2,129.4 MiB |
| LoRA bf16 adapter | fla (INFERRED) | 1,011.8 MiB | 833.5 MiB | 30.29 GiB | 1,089.1 MiB | 1,011.8 MiB |
| LoRA fp32 adapter + autocast | torch + autocast | 2,257.4 MiB | 929.5 MiB | 60.23 GiB | 1,089.1 MiB | 2,257.4 MiB |
| LoRA fp32 adapter + autocast | fla (INFERRED) | 1,139.8 MiB | 929.5 MiB | 34.04 GiB | 1,089.1 MiB | 1,139.8 MiB |

조합 예(활성값·loss 부분만, weight·grad·optimizer 제외, allocator 여유 제외):
- TRL SFT 기본(GC + chunked_nll), LoRA fp32 adapter + autocast: torch 경로 ≈ 1.06 GiB + max(chunk CE ≈1.0 GiB, 1.36 × 2.20 GiB) ≈ **4.1 GiB**, fla 경로 ≈ 1.06 + 1.36 × 1.11 ≈ **2.6 GiB**. calibrated(CPU)
- full FT, GC 없음, HF loss(`nll`), autocast, torch 경로: 61.93 GiB + 12·N·V(11.37 GiB, backward 시작 시점, log_softmax 포함) ≈ **73 GiB**. analytic

---

## 구현 시사점 (Implementation implications)

### A. Architecture adapter가 노출할 profile 키

| 키 | 값 | 결정 근거(정적 분석으로 resolve) |
|---|---|---|
| `attn_impl` | `sdpa` (기본) / `eager` / `flash_attention_2`(UNKNOWN profile) | 요청값 → transformers 5.18 기본 `sdpa` (§1.1) |
| `sdpa_backend` | `flash` / `cudnn` / `mem_efficient` / `math` | GPU arch·dtype·mask 유무로 §3.2 규칙 적용. 결과를 requested/resolved로 함께 기록 |
| `mask_materialized` | bool | batch에 padding이 있거나 packing(position_ids 리셋)이 있으면 true (§3.1) |
| `linear_attn_kernel` | `torch` / `fla` / `hub_kernels`(UNKNOWN) | 학습 환경에 `fla`(fla-core)·`causal_conv1d` import 가능 여부. 둘은 따로 판단(conv만 있을 수도 있음) |
| `autocast` | bool | `bf16=True` 또는 `fp16=True`이고 non-DeepSpeed이면 CUDA에서 true (§2.4) |
| `adapter_dtype` | `bf16`/`fp32` | TRL: 양자화 모델 bf16, 비양자화 fp32(ZeRO-3 제외) (§4.3) |
| `lora_dropout` | float | >0이면 모듈별 dropout 출력+mask 항 추가 |
| `grad_ckpt` | `none` / `every_layer` / `every_n` / `offload` | TRL 기본 `every_layer`, `use_reentrant=False` |
| `loss_path` | `hf_ce` / `trl_chunked_nll` / `liger`(UNKNOWN) | TRL SFT 기본 `chunked_nll`, DPO/GRPO는 해당 조사 문서 |
| `outputs_fp32` | bool | accelerate native_amp이면 true → fp32 logits 복사 |
| `use_cache_in_train` | bool | TRL은 false. 다른 Trainer로 직접 호출하면 config 기본 true일 수 있음 |

### B. AllocationSpec 생성 규칙

1. **모든 식은 `formulas.py`(scratch)의 검증된 형태를 그대로 옮긴다.** 레이어 saved `S_layer = 2·RMSNorm(N,H) + MLP(N) + mixer(path)`. Linear 입력 저장 여부는 "그 Linear의 weight가 학습되거나 LoRA 대상인가"로만 결정하고, 같은 tensor를 읽는 Linear들은 `storage_alias_group`으로 1번만 센다. 단 dtype cast가 끼면(fp32 adapter, autocast의 fp32→bf16) alias가 깨져 모듈마다 센다.
2. `evidence` 값: Qwen3.5 torch 경로·dense·SDPA(CPU flash와 동형인 flash/cuDNN)·loss는 `analytic`(CPU 측정으로 byte 단위 검증), fla/causal-conv1d·mem-efficient·cuDNN의 lse 크기는 `analytic`이지만 출처를 "source-inferred"로 표기, transient 계수는 `calibrated`(CPU), hub kernels·FA2 varlen·liger·fla 대체 backend는 `unknown`(수치 대신 null, plan §19.1의 Unknown 규칙).
3. Hybrid 모델은 full-attn 레이어 수(예시 8)에만 attention 식과 KV 식을 적용하고, linear 레이어(24)는 별도 식을 호출한다. 32개 전부에 일반 KV/attention 식을 적용하면 테스트가 실패해야 한다(plan §19.3).
4. cos/sin은 모델당 1번(Qwen3.5 `2·b·B·T·r`, dense `2·b·T·d`).
5. `T`는 padding 포함 길이, linear 레이어는 추가로 64 배수 `Tp`를 쓴다(`nc = ceil(T/64)`). chunk가 1개(T ≤ 64)이면 `query_decayed` 항이 항상 빠지고, 거기에 cache까지 없으면 `key_decayed` 항도 빠진다.

### C. Phase별 피크 조합 (plan §9.1, §9.2)

```
GC 없음:
  M_fwd_end   = Σ_layers S_layer + S_outside + logits_live
  M_fwd_peak  = M_fwd_end_without_loss + loss_fwd_peak            # hf_ce: 10·N·V, chunked: ~16.7·256·V
  M_bwd_peak  ≈ Σ S + S_outside + max(loss_bwd_extra, k_nogc · S_max)   # hf_ce: +8·N·V (12NV 총), k_nogc ≈ 0.15
GC (every_layer):
  M_retained  = n_ckpt · b·N·H + S_outside + kwargs(cos/sin, bool mask)
  M_fwd_peak  = M_retained + max(k_fwd · S_max, loss_fwd_peak)     # k_fwd ≈ 0.45 (측정 0.26–0.42)
  M_bwd_peak  = M_retained + max(loss_bwd_peak, k_bwd · S_max)      # k_bwd 기본 1.35 (측정 1.14–1.36)
  (full FT는 여기에 그 시점까지 누적된 param grad를 더한다 — grad 담당 문서)
every_n: checkpoint 안 된 레이어는 S_layer 전체를 retained에 더한다(vision block 선카운트 규칙 주의).
offload: ckpt 입력을 host pinned memory로 옮긴다 → device retained에서 n_ckpt·b·N·H를 빼고 host RAM에 더한다.
```

S_outside = final norm(`RMSNorm_q35(N,H)`) + lm_head 입력(hf_ce이고 lm_head trainable) + chunked면 gather hidden `b·N'·H` + index/labels int64.

기본 상수(CPU calibrated, GPU 보정 전까지): `k_bwd = 1.35`, `k_fwd = 0.45`, `k_nogc = 0.15`. 결과 화면에는 이 계수가 "CPU 보정값"임을 표시한다.

### D. LM head / loss 규칙

- `hf_ce`: `M_logits = b·N_k·V`(N_k = `logits_to_keep` 반영). forward 피크 `10·N·V`, 유지 `4·N·V`(+`2·N·V` bf16 logits가 살아 있으면, +`4·N·V` accelerate fp32 복사가 살아 있으면), backward 피크 `12·N·V`. plan §9.5의 `M_logits_tensor` 하나로 대체하지 않는다.
- `trl_chunked_nll`: peak `≈ 17.7·256·V` byte(보수값), 유지 `b·N'·H + 16·N'`. full logits 없음. lm_head가 LoRA 대상이면 TRL이 거부하므로 profile을 `unsupported`로.
- label이 `-100`인 prompt 위치: `hf_ce`는 메모리가 줄지 않는다(전 위치 logits). chunked는 valid token chunk만 돈다(최소 1 chunk). activation은 두 경우 모두 전 위치(plan §8.1).

### E. Generation cache 규칙 (plan §9.7)

```
M_gen_cache(seq, L) = Σ_full 2·nkv·d·b·L + Σ_linear (b·C·K + 4·Hv·dk·dv)
예시: 51,904,512 + 32,768·L  byte / sequence
```

- recurrent state는 config `mamba_ssm_dtype`과 무관하게 fp32(transformers 5.18 경로). conv state는 K(=4) 위치. DynamicCache 성장은 `torch.cat`이므로 step마다 레이어 1개분 K/V transient를 더하는 것이 보수적이다(INFERRED).
- vLLM 등 별도 backend는 이 식을 쓰지 않는다(다른 문서).

### F. 테스트 fixture로 쓸 golden 값 (CPU에서 재현 가능)

tiny config A(`vfmem.TINY_Q35`), bf16, full FT, SDPA, `use_cache=False`, no GC:

| B, T | Qwen3.5 linear(torch) | linear(torch+autocast) | full(첫 full, cos/sin 포함) | Qwen3 layer(비첫) | Llama layer(비첫) |
|---|---|---|---|---|---|
| 1, 100 | 3,737,584 | 3,370,096 | 872,288 | 620,800 | 425,600 |
| 2, 200 | 14,930,776 | 13,739,608 | 3,485,888 | 2,483,200 | 1,702,400 |
| 1, 40 | 1,648,036 | 1,505,188 | 349,568 | 248,320 | 170,240 |

추가 fixture 제안: (a) LoRA fp32 adapter(autocast 없음) L1 = 3,909,948 / bf16 adapter L1 = 3,619,548 (3-layer `[lin,lin,full]`, r=8, B=1,T=100), (b) eager full 레이어의 `B·T²` 계수 = `6·nq`, (c) generate 후 cache dtype/shape(§7 표), (d) vision tower 파라미터 456,010,480.

### G. 지원 범위 판정

- `torch` linear-attn 경로 + packing/padding-free: 정확성 문제 → `unsupported`.
- sm80 미만 GPU + bf16: SDPA math backend(O(T²)) profile로 계산하거나 미지원 처리.
- `use_kernels=True`, liger, FA2(`flash_attention_2` varlen), fla 대체 backend: saved set 미확인 → `unknown`.

---

## 미확정 사항 (Open questions)

1. **GPU 보정**: CPU allocator로 잰 transient 계수(`k_bwd` 1.14–1.36, `k_fwd` 0.26–0.42)가 CUDA caching allocator(블록 반올림, stream별 pool, 단편화)에서 얼마나 커지는지. plan §17 측정으로 `allocator_slack`과 함께 확정해야 한다. UNKNOWN
2. **fla 경로 실측**: fla-core 0.5.2 + causal-conv1d 1.7.0 saved set은 소스 추론이다. CUDA에서 hook으로 실측하고, `chunk_gated_delta_rule_bwd`의 transient(h 재계산 `[B,nc,Hv,dk,dv]`, dq/dk/dv 등)를 재야 한다. INFERRED
3. `fla`의 `@dispatch` 대체 backend(`flash_qla` 등)와 hub kernel(`kernels-community/fla` v1, `FusedRMSNormGated`, GB10 전용 `Atlas-Inference/gdn`)의 saved set. UNKNOWN
4. `flash_attention_2`(varlen, unpad/pad) 경로의 index·unpadded q/k/v 복사본 크기. 이 문서는 SDPA 기본만 다룬다. UNKNOWN
5. cuDNN SDPA가 Hopper/Blackwell에서 실제로 선택되는지(cuDNN 버전·head_dim 256 bprop 조건 `check_cudnn_d256_bprop_head_dim`)와 그때 attn_bias/workspace 크기. INFERRED/UNKNOWN
6. DPO/GRPO의 TRL Triton fused logprob kernel(`trl/kernels`)의 saved set과 피크. → trainer 조사 문서와 GPU 보정. UNKNOWN
7. DynamicCache의 `torch.cat` 성장이 긴 rollout에서 만드는 단편화 크기. INFERRED
8. 학습 중 `use_cache=True`(TRL 밖 사용)일 때 cache 객체가 graph를 잡아 두는 추가 수명(최종 state branch, K/V 복사본)을 profile로 넣을지. 측정상 byte 차이는 작지만 수명이 길어진다. INFERRED
9. fp32 master weight + autocast의 weight cast 복사본(`2·P` byte, backward까지 saved)을 weight 담당 ledger와 어느 쪽에 둘지(중복 계산 방지). VERIFIED(CPU)지만 소유권 미정
10. 예시 모델 실제 차원의 직접 측정은 CPU bf16 GEMM이 너무 느려 ¼ 스케일로 대체했다(식이 정확히 선형이라 대체 가능하나, 전체 32층·실제 vocab의 end-to-end 피크는 GPU 측정 대상). INFERRED
