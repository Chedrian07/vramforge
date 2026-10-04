# 조사: 모델 로딩 · bitsandbytes 4-bit · PEFT LoRA/QLoRA · Gradient checkpointing · Optimizer

## 1. 개요 (Header)

| 항목 | 값 |
|---|---|
| 주제 | 로딩 범위(loading scope)와 effective dtype, bitsandbytes 4-bit 저장 형식과 byte 공식, TRL의 QLoRA 준비 절차, PEFT LoRA 파라미터·dtype 규칙, gradient checkpointing, optimizer state, 런타임 임시 메모리, 비-PyTorch 오버헤드 |
| 대상 버전 | torch==2.14.1, transformers==5.18.0, trl==1.14.1, peft==0.21.2, bitsandbytes==0.50.2, accelerate==1.15.0 (부수: datasets==5.0.1, huggingface_hub==1.33.0, tokenizers==0.23.2, safetensors==0.8.0) |
| 작성일 | 2026-10-04 |
| 작성 | research-loading-quant-peft (Milestone M0) |
| 검증 | verify-loading-quant-peft (2026-10-04, 적대적 재검증). 소스 재정독 + 독립 실험(V1–V13, scratch `/tmp/vf-research/scratch/verify-loading-quant-peft/`)으로 핵심 주장 22개를 다시 확인했다. 수정한 곳은 본문에 **[검증 보정]** 으로 표시했고, 전체 내역은 문서 끝 "검증 로그"에 있다 |
| 방법 | ① 설치된 site-packages 소스 정독 ② CPU 실험 E1–E13 (macOS arm64, CUDA 없음, bitsandbytes CPU backend) ③ HF Hub safetensors **header 메타데이터만** 조회 (가중치 미다운로드) |
| 예시 모델 | `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` @ `2367e865d009c13ac81713a2878291d33ab28177` — 760 tensors, 전부 BF16, 9,409,813,744 params (header의 `parameter_count` 확인) |
| 근거 태그 | **VERIFIED** = 소스에서 확인했거나 실측함 / **INFERRED** = 소스 기반 추론(CUDA 전용 경로 등 이 머신에서 실행 불가) / **UNKNOWN** |
| 인용 형식 | `<pkg>==<ver> <site-packages 상대 경로>:<줄 범위> (<함수/클래스>)` |

### 실험 목록 (method detail)

스크립트는 로컬 scratch(`/tmp/vf-research/scratch/loading-quant-peft/`, 커밋하지 않음)에 있다. 모두 `/tmp/vf-research/.venv/bin/python` + `HF_HOME=/tmp/vf-research/hf` + `HF_HUB_OFFLINE=1`로 실행했다.

| ID | 스크립트 | 내용 | 핵심 결과 |
|---|---|---|---|
| E1 | `e1_make_tiny.py`, `e1_load_paths.py`, `e1_trl_paths.py`, `e1_quant_paths.py` | 예시 모델과 **같은 모듈 구조**(linear_attention 3 + full_attention 1, vision depth 2)의 tiny random `Qwen3_5ForConditionalGeneration`을 BF16으로 저장. 실제 config처럼 top-level `dtype`은 없애고 `text_config.dtype=bfloat16`만 둠. 로딩 경로별 class·dtype·키·양자화 대상 확인 | §Q1, §Q2 표 |
| E2 | `e2_peft.py` | tiny 4-bit 모델에 PEFT LoRA 적용: `target_modules=None`/`all-linear`/regex/`exclude_modules`/`modules_to_save`/DoRA/bias | §Q5 |
| E3 | `e3_bnb_storage.py` | `bitsandbytes.functional.quantize_4bit` (CUDA와 같은 op 경로 `torch.ops.bitsandbytes.quantize_4bit`)의 tensor별 byte 실측 | §Q3 표: 공식과 byte 단위 일치 |
| E4 / E4b | `e4_fullsize_meta.py`, `e4b_rounding.py` | **실제 크기** 예시 모델을 meta device로 생성 → transformers와 같은 `replace_with_bnb_linear` 적용 → 양자화/비양자화 inventory와 LoRA 파라미터 수. E4b는 header inventory로 4-bit byte를 tensor별로 다시 계산(+512 B 반올림) | §Q2, §Q3, §Q5 수치 |
| E5 | `e5_sft_qlora_cpu.py` (+`e5c`, `e5d`) | **실제 `trl.SFTTrainer`** 로 tiny 모델 QLoRA/LoRA 2 step 학습 (`use_cpu=True`, GA=2) → 학습 후 dtype, checkpointing kwargs, optimizer state, grad 수명, autocast 활성 여부 | §Q6–Q8 |
| E6 | `e6_act_dtypes.py` | bf16 autocast에서 residual stream·layer 경계 tensor의 dtype (fp32 로드 vs bf16 로드) | §Q9 |
| E7 | `e7_load_transient.py` | `from_pretrained` 중 on-the-fly 양자화 계측: 입력 dtype/device, 동시에 살아 있는 full-precision tensor 수 | §Q4 |
| E8 | `e8_bnb_optim.py` | bnb AdamW 8-bit/32-bit/paged, torch fused AdamW의 state byte 실측 | §Q8 |
| E9 | `e9_skip_modules.py` | `llm_int8_skip_modules` 지정 시 기본 skip 목록 동작 | §Q2 |
| E10 | `e10_saved_tensors.py` | `saved_tensors_hooks`로 LoRA 래핑 layer가 backward용으로 저장하는 tensor 기록 | §Q5, §Q9 |
| E11 | `e11_profiles.py` | 예시 모델의 profile별 정적 상주 byte 합산 | §구현 시사점 |
| E12 | `e12_compute_dtype.py` | `Linear4bit` compute dtype(fp32 기본값 vs bf16)과 autocast | §Q9 |
| E13 | `e13_keymap.py` | **실제 checkpoint header의 키 집합**이 두 loader의 state_dict와 1:1 대응하는지 | §Q1 |

---

## 2. 조사 결과 (Findings)

### Q1. 기본 dtype, TRL `model_init_kwargs`, Auto class, loading scope

**1.1 transformers 5.18 `from_pretrained`의 기본 dtype은 `"auto"`다.** [VERIFIED]
`dtype` 인자를 주지 않으면 `"auto"`로 바뀐다 — `transformers==5.18.0 transformers/modeling_utils.py:4105-4107 (PreTrainedModel.from_pretrained)`. `"auto"`는 `config.dtype` → (sharded metadata의 `dtype`) → checkpoint 첫 파일을 meta로 읽어 첫 floating tensor의 dtype 순으로 결정한다 — `transformers/modeling_utils.py:815-896 (_get_dtype)`.

```python
# transformers/modeling_utils.py:833-849 (_get_dtype, 발췌)
if dtype == "auto":
    if hasattr(config, "dtype") and config.dtype is not None:
        dtype = config.dtype
    else:
        if is_sharded and "dtype" in sharded_metadata: ...
        elif state_dict is not None: ...
        else:
            state_dict = load_state_dict(checkpoint_files[0], map_location="meta", ...)
            dtype = get_state_dict_dtype(state_dict)
```

- 예시 모델의 top-level `Qwen3_5Config.dtype`은 `None`, `text_config.dtype`은 `torch.bfloat16`, `vision_config.dtype`은 `None`이다 [VERIFIED, `AutoConfig.from_pretrained` 출력]. 따라서 `Qwen3_5ForConditionalGeneration.from_pretrained(id)`는 **weight dtype(BF16)** 으로, `AutoModelForCausalLM`은 `text_config.dtype`(BF16)으로 로드된다 [VERIFIED E1: 두 경로 모두 전 파라미터 `torch.bfloat16`].
- bnb 4-bit quantizer는 `update_dtype`을 재정의하지 않으므로 dtype을 바꾸지 않는다 (base는 identity) — `transformers/quantizers/base.py:99-110 (HfQuantizer.update_dtype)`, `transformers/quantizers/quantizer_bnb_4bit.py:45-187`에 override 없음 [VERIFIED].

**1.2 TRL 1.14.1은 모델 ID 문자열을 받으면 기본 dtype을 `float32`로 강제한다.** [VERIFIED]
`SFTConfig/DPOConfig/GRPOConfig.model_init_kwargs`의 기본값은 `None`이고(`trl==1.14.1 trl/trainer/sft_config.py:147-153`, `trl/trainer/dpo_config.py:161`, `trl/trainer/grpo_config.py:428`), 트레이너는 이를 `{}`로 바꿔 `create_model_from_path`에 넘긴다(`trl/trainer/sft_trainer.py:972-986`, `trl/trainer/dpo_trainer.py:554-567`, `trl/trainer/grpo_trainer.py:334-347`).

```python
# trl==1.14.1 trl/trainer/utils.py:1292-1309 (create_model_from_path, 발췌)
dtype = kwargs.get("dtype", "float32")
if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
    pass
elif isinstance(dtype, str) and dtype in ["bfloat16", "float16", "float32"]:
    kwargs["dtype"] = getattr(torch, dtype)          # 미지정이면 여기서 torch.float32
...
if "device_map" not in kwargs:
    kwargs["device_map"] = None if PartialState().device.type == "cpu" else "auto"
if architecture is None:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=...)
    architecture = getattr(transformers, config.architectures[0], None)
```

- 트레이너 docstring도 같은 내용을 명시한다: "If `dtype` is not specified in `args.model_init_kwargs`, it defaults to `float32`" — `trl/trainer/sft_trainer.py:837-840` [VERIFIED].
- 실측 [VERIFIED E1/E5]: tiny clone을 TRL 경로로 로드하면 기본값은 **모든 파라미터 fp32**, `dtype="bfloat16"` 또는 `"auto"`는 BF16.
- TRL CLI의 `ModelConfig.dtype` 기본값도 `"float32"`다 — `trl/trainer/model_config.py:89-92` [VERIFIED].
- `device_map`: CUDA가 있으면 `"auto"`, `MULTI_GPU`/`DEEPSPEED` 분산이면 `None` — `trl/trainer/sft_trainer.py:981-983` [VERIFIED].
- TRL 1.14.1 트레이너는 `quantization_config` 인자를 직접 받는다(trainer 인자와 `model_init_kwargs`에 동시에 주면 오류) — `trl/trainer/sft_trainer.py:973-980` [VERIFIED].

**1.3 TRL이 쓰는 클래스는 `config.architectures[0]`의 구체 클래스다 → 예시 모델은 `Qwen3_5ForConditionalGeneration`(vision tower 포함).** [VERIFIED]
위 코드처럼 `getattr(transformers, "Qwen3_5ForConditionalGeneration")`이 존재하므로 Auto class를 거치지 않는다. `AutoModelForImageTextToText`/`AutoModelForCausalLM` fallback은 remote-code checkpoint에만 쓴다 — `trl/trainer/utils.py:1306-1320`. E1 실측: TRL 기본 경로의 class는 `Qwen3_5ForConditionalGeneration`, vision 파라미터가 포함된다. DPO의 별도 reference(`dpo_trainer.py:911-930`)와 GRPO의 `beta≠0` reference(`grpo_trainer.py:967-985`)도 같은 함수(같은 dtype 기본값·같은 `quantization_config`)로 로드한다 [VERIFIED].
- 주의: architecture 추론용 `AutoConfig.from_pretrained`에는 `revision`이 전달되지 않는다(`utils.py:1308`). 가중치는 `model_init_kwargs["revision"]`로 로드되지만 class 결정은 기본 revision의 config를 읽는다 [VERIFIED, 재현성 리스크].

**1.4 같은 checkpoint를 `AutoModelForCausalLM`으로 읽으면 text-only `Qwen3_5ForCausalLM`이 된다.** [VERIFIED]
- 매핑: `("qwen3_5", "Qwen3_5ForCausalLM"),  # VLM compatibility` — `transformers/models/auto/modeling_auto.py:854`. `AutoModelForImageTextToText`는 `Qwen3_5ForConditionalGeneration` — `modeling_auto.py:1173`.
- composite config → `text_config` 교체(부모 `quantization_config`도 복사): `transformers/models/auto/auto_factory.py:395-410 (_BaseAutoModelClass.from_pretrained)`.
- vision/MTP 가중치는 **경고 없이 버려진다**: `_keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]` — `transformers/models/qwen3_5/modeling_qwen3_5.py:1673-1679 (Qwen3_5ForCausalLM)`.
- prefix 재매핑: `"qwen3_5_text": [PrefixChange(prefix_to_remove="language_model", model_prefix="model")]` — `transformers/conversion_mapping.py:1054`; 정규식 `^model\.language_model\.(.+)$ → model.\1` — `transformers/core_model_loading.py:1098-1128 (PrefixChange)`.
- E1(tiny): class `Qwen3_5ForCausalLM`, missing 0, unexpected 0, 로드된 `up_proj` 값이 checkpoint와 동일. E13(**실제 header 760 tensors**): `Qwen3_5ForConditionalGeneration` missing 0/unexpected 0; `Qwen3_5ForCausalLM`도 renaming 후 missing 0/unexpected 0, 버려지는 키 333개(`model.visual.*`) = 456,010,480 params.

| loading scope | 진입 경로 | class | 총 params | vision |
|---|---|---|---:|---|
| CondGen (TRL 기본) | `SFT/DPO/GRPOTrainer(model="<id>")`, `AutoModelForImageTextToText` | `Qwen3_5ForConditionalGeneration` | 9,409,813,744 | 포함 (상주) |
| CausalLM (text-only) | 사용자가 `AutoModelForCausalLM`으로 만든 객체를 트레이너에 전달 | `Qwen3_5ForCausalLM` | 8,953,803,264 | 제외 |

- config의 `mtp_num_hidden_layers: 1`, `mamba_ssm_dtype: "float32"`는 transformers 5.18 어디에서도 참조하지 않는다 (`grep -rn mamba_ssm_dtype transformers/` 결과 없음). checkpoint에도 `mtp.*` tensor가 없다 [VERIFIED].
- `Qwen3_5PreTrainedModel`에는 `_keep_in_fp32_modules`/`_keep_in_fp32_modules_strict`가 없다 (`modeling_qwen3_5.py:916-930`; E9에서 `model._keep_in_fp32_modules == set()`) [VERIFIED].

### Q2. BitsAndBytesConfig 4-bit 변환 대상

**2.1 교체 규칙: `type(module) is nn.Linear`(정확한 타입) 또는 HF `Conv1D`만 `Linear4bit`로 바뀐다.** [VERIFIED]

```python
# transformers==5.18.0 transformers/integrations/bitsandbytes.py:184-215 (replace_with_bnb_linear, 발췌)
for module_name, module in model.named_modules():
    if not should_convert_module(module_name, modules_to_not_convert):
        continue
    with torch.device("meta"):
        if isinstance(module, Conv1D) or type(module) is nn.Linear:
            ...
            new_module = bnb.nn.Linear4bit(in_features, out_features, module.bias is not None,
                quantization_config.bnb_4bit_compute_dtype,
                compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                quant_type=quantization_config.bnb_4bit_quant_type,
                quant_storage=quantization_config.bnb_4bit_quant_storage)
```

- `nn.Linear`의 하위 클래스, `nn.Conv1d/Conv3d`, `nn.Embedding`, norm, raw `nn.Parameter`는 변환되지 않는다. bias는 양자화하지 않는다(`param_needs_quantization`은 `name != "bias"`) — `transformers/quantizers/quantizer_bnb_4bit.py:91-95`.
- 호출 위치: `Bnb4BitHfQuantizer._process_model_before_weight_loading` — `quantizer_bnb_4bit.py:121-143`.

**2.2 기본 skip 목록 = tied 키 ∪ 마지막 파라미터의 모듈 ∪ output embedding.** [VERIFIED]
`get_keys_to_not_convert` — `transformers/quantizers/base.py:38-62`; `get_modules_to_not_convert` — `base.py:238-258`; 패턴 매칭은 `re.match(key+"\.")`(prefix), `re.match(key)`, `endswith(key)` — `transformers/quantizers/quantizers_utils.py:26-42 (should_convert_module)`.
- 예시 모델(`tie_word_embeddings=false`): `all_tied_weights_keys == {}`, 결과 skip 목록은 `['lm_head']` [VERIFIED E4/E9].
- **함정**: 사용자가 `llm_int8_skip_modules`를 주면 기본 목록을 **대체**한다(`add_default_skips=False`). `llm_int8_skip_modules=["model.visual"]`이면 `lm_head`가 `Linear4bit`가 된다 [VERIFIED E9: `lm_head=Linear4bit`, Linear4bit 32개]. `["model.visual","lm_head"]`처럼 다시 넣어야 한다.

**2.3 BitsAndBytesConfig 기본값은 QLoRA 설정이 아니다.** [VERIFIED]
`bnb_4bit_quant_type="fp4"`, `bnb_4bit_use_double_quant=False`, `bnb_4bit_compute_dtype=None→torch.float32`, `bnb_4bit_quant_storage=None→torch.uint8` — `transformers/utils/quantization_config.py:440-478 (BitsAndBytesConfig.__init__)`. E1d에서 출력 확인(`fp4 False torch.float32 torch.uint8 None`). blocksize 필드는 없다(§3.1). TRL CLI `get_quantization_config`는 `bnb_4bit_compute_dtype=model_args.dtype`(기본 `"float32"`), `nf4`, nested quant `False`를 쓴다 — `trl/trainer/utils.py:252-268`, `trl/trainer/model_config.py:156-171`.

**2.4 Qwen3.5 모듈별 결과** [VERIFIED E1d(tiny, CPU 실제 로드) + E4(실제 크기, meta)]

| 모듈 (CondGen 기준 이름) | 타입 | 4-bit? | 로드 후 dtype |
|---|---|---|---|
| `language_model.layers.*.linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}` | `nn.Linear` | 예 (`in_proj_a/b` 32×4096 포함) | `Params4bit` uint8 |
| `language_model.layers.*.self_attn.{q,k,v,o}_proj` | `nn.Linear` | 예 | uint8 |
| `language_model.layers.*.mlp.{gate,up,down}_proj` | `nn.Linear` | 예 | uint8 |
| `linear_attn.conv1d.weight` [8192,1,4] | `nn.Conv1d` | 아니오 | load dtype |
| `linear_attn.A_log` [32], `linear_attn.dt_bias` [32] | `nn.Parameter` | 아니오 | load dtype (`mamba_ssm_dtype` 무시) |
| `linear_attn.norm` (`Qwen3_5RMSNormGated` [128]), `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `language_model.norm` | norm | 아니오 | load dtype |
| `language_model.embed_tokens` [248320,4096] | `nn.Embedding` | 아니오 | load dtype |
| `lm_head` [248320,4096] | `nn.Linear` | **아니오 (기본 skip)** | load dtype |
| `visual.blocks.*.attn.{qkv,proj}`, `visual.blocks.*.mlp.{linear_fc1,linear_fc2}`, `visual.merger.{linear_fc1,linear_fc2}` | `nn.Linear` (bias 있음) | **예 (CondGen scope에서)** | weight uint8, **bias는 load dtype** |
| `visual.patch_embed.proj` (`nn.Conv3d`), `visual.pos_embed` (`nn.Embedding`), `norm1/norm2/merger.norm` (`nn.LayerNorm`) | 기타 | 아니오 | load dtype |

모듈 정의 근거: `transformers/models/qwen3_5/modeling_qwen3_5.py:503-548 (Qwen3_5GatedDeltaNet.__init__)`, `:947-1008 (VisionMLP/PatchEmbed/PatchMerger)`, `:1010-1022 (VisionAttention)`, `:1221-1234 (Qwen3_5TextModel.__init__)`, `:1765-1774 (Qwen3_5ForConditionalGeneration.__init__)`.

| 수량 (E4, 실제 크기) | CondGen (TRL 기본) | CausalLM (text-only) |
|---|---:|---:|
| `Linear4bit` 수 | 358 (text 248 + vision 110) | 248 |
| 양자화 원소 수 | 7,369,682,944 (vision 451,178,496) | 6,918,504,448 |
| 비양자화 원소 수 | 2,040,130,800 | 2,035,298,816 |
| └ embed_tokens / lm_head | 1,017,118,720 / 1,017,118,720 | 동일 |
| └ conv1d / RMSNorm / RMSNormGated / A_log+dt_bias | 786,432 / 270,336 / 3,072 / 1,536 | 동일 |
| └ vision pos_embed / Conv3d / Linear4bit bias / LayerNorm | 2,654,208 / 1,770,624 / 280,432 / 126,720 | — |
| 가장 큰 양자화 weight | `mlp.gate_proj` 12288×4096 = 50,331,648 | 동일 |

**2.5 `Linear4bit`의 bias는 첫 forward에서 compute dtype으로 in-place 변환된다.** [VERIFIED 소스]
`bias.data = bias.data.to(x.dtype)` — `bitsandbytes==0.50.2 bitsandbytes/nn/modules.py:631-635 (Linear4bit.forward)`. text-only 학습에서는 vision이 실행되지 않으므로 vision bias는 load dtype 그대로 남는다 [VERIFIED E6: fp32 로드 시 vision bias fp32].

### Q3. bitsandbytes 0.50.2 4-bit 저장 형식과 byte 공식

**3.1 기본값** [VERIFIED]
- blocksize: `Params4bit`의 `blocksize=None → 64` — `bitsandbytes/nn/modules.py:230-231 (Params4bit.__new__)`. transformers는 `Linear4bit` 생성 시 blocksize를 넘기지 않으므로(`transformers/integrations/bitsandbytes.py:208-215`) **항상 64**이고 설정 경로가 없다. `quantize_4bit` 자체 기본값도 64 — `bitsandbytes/functional.py:917-918`.
- `quant_storage` 기본 `torch.uint8` — `functional.py:884-892`, `nn/modules.py:222`, `quantization_config.py:476-478`.
- nested(double quant) blocksize **256 고정** — `functional.py:939-940`. gemm 경로는 `state2.blocksize != 256`이면 `NotImplementedError` — `bitsandbytes/autograd/_functions.py:333-347 (MatMul4Bit.forward)`.

**3.2 구성 요소** [VERIFIED 소스 + E3 실측]

```python
# bitsandbytes==0.50.2 bitsandbytes/functional.py:929-960 (quantize_4bit, 발췌)
_out, _absmax = torch.ops.bitsandbytes.quantize_4bit.default(A, blocksize, quant_type, quant_storage)
code = get_4bit_type(quant_type, device=A.device)          # 16 x fp32, tensor마다 새로 생성
if compress_statistics:
    offset = _absmax.mean()                                 # fp32 scalar
    qabsmax, state2 = quantize_blockwise(_absmax - offset, blocksize=256)
    del _absmax
    state = QuantState(absmax=qabsmax, shape=input_shape, dtype=A.dtype, blocksize=blocksize,
                       code=code, quant_type=quant_type, offset=offset, state2=state2)
else:
    state = QuantState(absmax=_absmax, shape=input_shape, dtype=A.dtype, ...)
```

| 구성 | shape / dtype | byte | 근거 |
|---|---|---|---|
| packed weight | `((n+1)//(2·s), 1)` of `quant_storage` (s = itemsize) | uint8: `ceil(n/2)` | `bitsandbytes/backends/cuda/ops.py:384-420`, `bitsandbytes/_ops.py:225-236` |
| absmax (DQ 없음) | `(ceil(n/64),)` fp32 | `4·ceil(n/64)` | 동일 |
| code (4-bit 표) | `(16,)` fp32, tensor마다 | 64 | `functional.py:772-859 (get_4bit_type)` |
| absmax (DQ) | `(ceil(n/64),)` uint8 | `ceil(n/64)` | `functional.py:613-686 (quantize_blockwise)` |
| nested absmax (`state2.absmax`) | `(ceil(nb/256),)` fp32 | `4·ceil(nb/256)` | `backends/cuda/ops.py:299-332` |
| offset | 0-dim fp32 | 4 | `functional.py:939` |
| nested code (`state2.code`) | `(256,)` fp32, `copy=True`로 tensor마다 복사 | 1024 | `functional.py:654-680` |

- `QuantState.dtype` = 양자화 시점 입력 dtype = **load dtype** (TRL 기본이면 fp32) [VERIFIED E1d: `qs.dtype torch.float32` vs `torch.bfloat16`]. 이 값이 backward dequant 출력 dtype을 정한다(§Q9).
- `n % 64 != 0`이어도 블록 정렬 padding은 없다. 마지막 부분 블록도 absmax 1개를 갖고, packed는 nibble 반올림만 한다 [VERIFIED E3: 4097×3 → n=12,291, absmax 193개, packed 6,146 B].
- `quant_storage=bfloat16`(FSDP-QLoRA용)도 byte 수는 같다. 다만 packed byte가 짝수여야 하며 CPU default 구현은 홀수에서 실패했다 [VERIFIED E3: 3×7에서 오류]. CUDA op는 `(n+1)//(2·s)`로 내림하므로 `n % 4 != 0`이면 정보 손실 가능성이 있다 [INFERRED].

**3.3 공식** (n = 원소 수, b = 64, nb = ceil(n/64))

```text
B_q4_noDQ(n) = ceil(n/2) + 4·nb + 64                              ≈ 0.5625 B/param
B_q4_DQ(n)   = ceil(n/2) + nb + 4·ceil(nb/256) + 4 + 64 + 1024    ≈ 0.51587 B/param
```

**3.4 계산 예와 실측** [VERIFIED E3 — CPU에서 `F.quantize_4bit(W_bf16, blocksize=64, quant_type="nf4")`로 측정한 byte가 공식과 정확히 일치]

| weight | n | packed | DQ 없음 total | DQ total | B/param (DQ) |
|---|---:|---:|---:|---:|---:|
| 4096×4096 | 16,777,216 | 8,388,608 | 9,437,248 (absmax 1,048,576 + code 64) | 8,655,940 (absmax 262,144 + nested 4,096 + offset 4 + code 64 + nested code 1,024) | 0.51593 |
| 12288×4096 | 50,331,648 | 25,165,824 | 28,311,616 (absmax 3,145,728 + 64) | 25,965,636 (786,432 + 12,288 + 4 + 64 + 1,024) | 0.51589 |
| 8192×4096 (`in_proj_qkv`, `q_proj`) | 33,554,432 | 16,777,216 | 18,874,432 | 17,310,788 | 0.51590 |
| 32×4096 (`in_proj_a/b`) | 131,072 | 65,536 | 73,792 | 68,708 | 0.52420 |

**3.5 예시 모델 합계** [VERIFIED E4 = E4b, 두 방법이 byte 단위로 일치]

| scope | 4-bit tensor 수 | DQ (NF4, bs64) | DQ 없음 |
|---|---:|---:|---:|
| CondGen | 358 | 3,802,183,024 B (3.5411 GiB) | 4,145,469,568 B (3.8608 GiB) |
| CausalLM | 248 | 3,569,313,760 B (3.3242 GiB) | 3,891,674,624 B (3.6244 GiB) |

CUDA caching allocator는 모든 할당을 최소 512 B로 반올림한다(`kMinBlockSize = 512` — `torch==2.14.1 torch/include/c10/core/AllocatorConfig.h:18-19`). tensor별로 반올림해도 CondGen DQ에서 +422,544 B뿐이다 [VERIFIED E4b]. 무시 가능하지만 정수 byte 계약을 위해 공식에 넣을 수 있다.

### Q4. 로딩 중 transient (materialize·quantize 순서)

**4.1 모델 골격은 meta device에서 만든다.** `get_init_context`가 `torch.device("meta")`를 넣는다 — `transformers/modeling_utils.py:3736-3765`. quantizer는 이 meta 골격에서 모듈을 교체한다(`hf_quantizer.preprocess_model`, `modeling_utils.py:4296-4302`, `transformers/quantizers/base.py:159-175`). CPU에 전체 모델을 만들지 않는다 [VERIFIED].

**4.2 on-the-fly 양자화는 thread pool 없이 tensor 하나씩 처리한다.** [VERIFIED 소스 + E7]

```python
# transformers/core_model_loading.py:1626-1636 (convert_and_load_state_dict_in_model)
# When doing on-the-fly quantization, we also use sync loading to avoid worker threads loading full-precision
# tensors to GPU faster than the main thread can quantize them, which would cause a large memory spike.
has_on_the_fly_quantization = hf_quantizer is not None and not hf_quantizer.pre_quantized
if (is_env_variable_true("HF_DEACTIVATE_ASYNC_LOAD") or "disk" in device_map.values()
        or has_on_the_fly_quantization):
    thread_pool = None
else:
    thread_pool = ThreadPoolExecutor(max_workers=GLOBAL_WORKERS)
```

순서: safetensors slice `tensor[...]`(CPU) → `.to(device=param_device, dtype=_dtype)` (`core_model_loading.py:1236-1241 _materialize_copy`) → `Bnb4bitQuantize.convert`가 `Params4bit(value).to(value.device)` 호출 (`transformers/integrations/bitsandbytes.py:30-61`) → `Params4bit._quantize`가 그 device에서 `quantize_4bit` 실행 후 `self.data`를 packed로 교체 (`bitsandbytes/nn/modules.py:381-395, 424-428`) → full-precision tensor는 `materialize_tensors`의 pop과 `del realized_value`로 해제 (`core_model_loading.py:953-979, 1784`).
- `_dtype`: 양자화 대상 weight도 **load dtype**으로 materialize한다(meta `Params4bit`가 load dtype을 가짐, `core_model_loading.py:1672-1714`).
- E7 실측: 41번의 quantize 호출 모두 입력 dtype = load dtype(TRL 기본 fp32 / `dtype=bf16`이면 bf16). 각 호출 시점에 이전 full-precision tensor가 살아 있는 경우는 0번이다. [검증 재현 V8: 별도 tiny clone, weakref 계측 — 41회 모두 `MainThread`, 이전 입력 생존 0, 입력 dtype = load dtype]
- `param_device`: TRL은 `device_map="auto"` → GPU. bnb quantizer는 `device_map=None`이면 `{"": torch.cuda.current_device()}`로 바꾼다 — `quantizer_bnb_4bit.py:102-119`. 따라서 양자화는 **GPU에서** 일어난다 [INFERRED: CUDA kernel 경로 미실행].

**4.3 가장 큰 transient** [INFERRED from 소스, 수치는 공식]
`(가장 큰 양자화 weight 1개 × load dtype) + 출력 packed/absmax (+DQ 임시)`. 예시 모델 `gate_proj` 12288×4096 기준:
- fp32 로드: 201,326,592 + 25,165,824 + 3,145,728(+DQ 임시 `absmax-offset` 3,145,728, qabsmax 786,432) ≈ 233.6 MB (0.218 GiB)
- bf16 로드: 100,663,296 + 같은 출력 ≈ 132.9 MB (0.124 GiB)

비양자화 파라미터(embed/lm_head 등)는 `.to(device, dtype)` 결과가 곧 최종 파라미터이므로 GPU에 추가 복사본이 없다 [INFERRED]. **Host RAM**은 tensor마다 safetensors slice를 CPU로 읽는다. 가장 큰 `embed_tokens`/`lm_head`는 BF16 2,034,237,440 B이고, fp32 변환이 CPU 쪽에서 일어나면 4,068,474,880 B의 임시 tensor가 추가된다 [INFERRED: CPU→CUDA dtype 변환 위치는 ATen copy 구현에 따름, wheel에 C++ 소스 없음].

**4.4 `caching_allocator_warmup`이 로딩 전에 모델 크기만큼 예약한다.** [VERIFIED 소스]
accelerator device마다 `Σ(numel × element_size)`를 계산한다. 양자화 weight는 0.5 B/param으로 센다(`quantizer_bnb_4bit.py:83-89 param_element_size`, `modeling_utils.py:4991-5022 get_total_byte_count`). 이 크기의 fp16 tensor 하나를 할당했다가 즉시 버린다(상한 `total_device_memory − 1.2 GiB`) — `modeling_utils.py:5025-5106 (caching_allocator_warmup)`, 호출 `:4417-4419`. 해제된 블록은 caching allocator에 **reserved로 남는다** [INFERRED]. 예시 모델 CondGen 4-bit 요청량: fp32 로드 11.032 GiB, bf16 로드 7.232 GiB [E11 계산]. [검증 재계산: 7,369,682,944×0.5 + 2,040,130,800×4 = 11,845,364,672 B, ×2이면 7,765,103,072 B. 호출 조건은 `device_map is not None` — bnb는 None을 `{"": cuda}`로 바꾸므로 단일 GPU에서는 항상 실행된다(`modeling_utils.py:4417-4419`). 기존 reserved−allocated가 있으면 요청량이 줄어든다(`:5068-5088`)]

**4.5 device_map 제약** [VERIFIED 소스 + CPU 시뮬레이션 V9 / free memory 값 자체는 INFERRED(CUDA 미실행)] **[검증 보정: 실효 계수 0.90 → 0.81]**
`device_map`이 문자열(`"auto"` 등)이면 `_get_device_map`이 다음 순서로 처리한다 — `transformers==5.18.0 transformers/integrations/accelerate.py:338-376 (_get_device_map)`.
1. `get_balanced_memory`로 GPU별 예산을 만든다. 기본값은 accelerate `get_max_memory`의 `torch.cuda.mem_get_info(i)[0]`(free)에 `memory_reserved − memory_allocated`를 더한 값이다(`accelerate==1.15.0 accelerate/utils/modeling.py:819`, `transformers/integrations/accelerate.py:202-237`). **GPU가 1개이고 사용자가 `max_memory`를 주지 않았으면 여기서 먼저 0.9배**를 한다("90% is a good compromise", `integrations/accelerate.py:283-292`).
2. bnb quantizer가 예산을 다시 **0.90배**로 줄인다(`quantizer_bnb_4bit.py:97-100`, 호출 `integrations/accelerate.py:363`).
3. `infer_auto_device_map`으로 배치한다. GPU 0에서는 "남은 모듈 중 가장 큰 layer" 크기를 예약한다(`integrations/accelerate.py:752`). 예시 모델은 가장 큰 leaf인 `lm_head`가 마지막에 배치되므로 이 예약은 결과를 바꾸지 않는다 [VERIFIED V9: 전부 GPU 0에 배치되는 최소 `max_memory[0]`이 모듈 합계 S보다 0.0003 GiB만 크다].
4. 추론된 dict로 `validate_environment`를 **다시** 호출하고, cpu/disk가 섞이면 `ValueError`를 낸다(`integrations/accelerate.py:374`, `quantizer_bnb_4bit.py:70-81`, `llm_int8_enable_fp32_cpu_offload=True`가 아니면).

따라서 단일 GPU에서 `max_memory`를 지정하지 않으면 실효 예산은 **(free + reserved − allocated) × 0.9 × 0.90 = × 0.81**이다. 사용자가 `max_memory`를 주면 1단계의 0.9는 빠지고 `min(user, free)` × 0.90이 된다. TRL은 `MULTI_GPU`/`DEEPSPEED`에서 `device_map=None`을 넘기며, 이때 bnb는 `{"": current_device}`로 바꾼다(`quantizer_bnb_4bit.py:102-119`). 이 경로에는 사전 적합 검사가 없어 넘치면 그대로 OOM이 난다. 비양자화 모델은 `"auto"`에서 CPU offload가 가능하다(`modeling_utils.py:4357-4359 accelerate_dispatch`) [INFERRED: 이 경우 VRAM 적합 판정 의미가 달라짐].

**4.6 로드 후 비양자화 모듈 dtype = load dtype.** transformers는 Qwen3.5에 대해 fp32로 올리는 규칙이 없다(§1.4) [VERIFIED E1d: TRL 기본이면 embed/lm_head/norm/conv1d/A_log/dt_bias/vision 비양자화 전부 fp32, `dtype=bf16`이면 전부 bf16].

**4.7 비양자화 로딩은 비동기다.** `ThreadPoolExecutor(max_workers=min(4, cpu_count))`(`core_model_loading.py:1232-1234, 1636`)가 최종 device·dtype으로 바로 materialize한다. 최종 가중치 외 GPU transient는 없다 [INFERRED].

### Q5. PEFT 0.21.2 LoRA

**5.1 A/B shape와 in/out 결정** [VERIFIED]
`lora_A = nn.Linear(in_features, r, bias=False)` → weight `(r, in)`; `lora_B = nn.Linear(r, out_features, bias=lora_bias)` → weight `(out, r)` (+ `lora_bias`면 bias `(out,)`) — `peft==0.21.2 peft/tuners/lora/layer.py:263-264 (LoraLayer.update_layer)`. in/out은 packed weight shape가 아니라 **모듈 속성**에서 가져온다. `Linear4bit`는 `nn.Linear` 하위 클래스이므로 `module.in_features/out_features`를 쓴다 — `peft/tuners/tuners_utils.py:167-188 (_get_in_out_features)`. E2: vision `qkv`(packed weight `(6144,1)`)에서 `in/out=64/192`, A `(16,64)`, B `(192,16)`. 학습 가능 파라미터 수 = Σ r(in+out) = 209,280으로 일치 [VERIFIED E2]. scaling = `alpha/r` (rsLoRA면 `alpha/√r`, `layer.py:278-281`)이며 메모리와 무관하다.

**5.2 adapter dtype 결정 순서** [VERIFIED 소스 + E2/E5]
1. 기본 dtype(fp32)으로 생성
2. `_move_adapter_to_device_of_base_layer`가 base dtype으로 cast: `nn.Linear`는 `weight.dtype`, `Linear4bit`는 `compute_dtype`(uint8 packed가 아님) — `peft/tuners/tuners_utils.py:2149-2212`
3. `get_peft_model(..., autocast_adapter_dtype=True)`(기본, `peft/mapping_func.py:105-110`)가 **BaseTunerLayer 안의** fp16/bf16(및 fp8) adapter 파라미터를 fp32로 올림 — `peft/tuners/tuners_utils.py:2705-2763 (cast_adapter_dtype)`

```python
# peft/tuners/tuners_utils.py:2722-2746 (cast_adapter_dtype, 발췌)
dtypes_to_convert_to_fp32 = {torch.float16, torch.bfloat16}
for module in model.modules():
    if not isinstance(module, BaseTunerLayer):
        continue
    ...  # lora_A / lora_B / lora_magnitude_vector 의 파라미터를 .to(torch.float32)
```

| 상황 | adapter dtype | 근거 |
|---|---|---|
| bf16 base, LoRA (PEFT 기본) | **fp32** | E2 `[bf16 base, LoRA] float32`, E5 `lora bfloat16` |
| bf16 base, `autocast_adapter_dtype=False` | bf16 | E2 |
| 4-bit base, `get_peft_model` 직후 | fp32 | E2 |
| 4-bit base, **TRL 트레이너 이후** | **bf16** (§Q6) | E2·E5 |
| TRL + DeepSpeed ZeRO-3 + 비양자화 | base dtype (`autocast_adapter_dtype=False`) | `trl/trainer/sft_trainer.py:1132-1133` |
| `modules_to_save` 복사본, trainable bias | 원래 모듈 dtype (fp32로 올리지 않음) | E2: lm_head 복사본 bf16, bias bf16 |

**5.3 `target_modules="all-linear"`** [VERIFIED 소스 + E2/E4]
`isinstance(module, (nn.Linear, Conv1D))`인 모든 모듈(그래서 `Linear4bit` 포함)에서 `model.get_output_embeddings()`(= `lm_head`)를 뺀다. SEQ_CLS면 분류 head를 뺀다 — `peft/tuners/tuners_utils.py:2430-2492 (_maybe_include_all_linear_layers)`. **VLM 예외 처리는 없으므로 vision tower Linear도 포함된다.** 결과 이름 집합은 `_find_minimal_target_modules`로 suffix 형태로 압축될 수 있다(`tuners_utils.py:909-930`).

| 예시 모델, r=16 (E4) | LoRA layer 수 | 학습 파라미터 | bf16 / fp32 |
|---|---:|---:|---:|
| `all-linear` (CondGen) | 358 (vision 110) | 51,265,024 (vision 7,986,688) | 97.8 / 195.6 MiB |
| `all-linear` + `exclude_modules=r".*visual.*"` | 248 | 43,278,336 | 82.5 / 165.1 MiB |
| 이름 목록 `[q,k,v,o,gate,up,down,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj]` | 248 | 43,278,336 | 82.5 / 165.1 MiB |

- `target_modules=None`이면 model_type 기본 매핑을 찾는다. `qwen3_5`는 매핑에 없으므로 `ValueError: Please specify target_modules ...` — `peft/tuners/lora/model.py:695-705`, `peft/utils/constants.py`에 `qwen3_5` 없음 [VERIFIED E2].

**5.4 매칭 의미론** [VERIFIED 소스 + E2]
- `target_modules`가 문자열이면 `re.fullmatch(pattern, 전체_모듈_이름)` — `peft/utils/other.py:1516-1521`. 리스트면 전체 이름 일치 또는 `endswith("." + name)` — `peft/tuners/tuners_utils.py:2370-2376`. `layers_to_transform`/`layers_pattern`은 첫 번째 숫자 index를 쓴다(`:2378-2410`).
- `exclude_modules`도 문자열은 `fullmatch`, 리스트는 일치 또는 `.name` suffix다 — `:2350-2357`. `modules_to_save`에 걸리는 모듈은 LoRA 대상에서 빠진다(`:2361-2364`).
- E2: text-only regex → 31 layer, `all-linear + exclude_modules=".*visual.*"` → 31 layer (tiny, 같은 결과).

**5.5 `modules_to_save`: 원본은 frozen으로 남고 trainable deep copy가 추가된다.** [VERIFIED]
`self.modules_to_save[adapter_name] = copy.deepcopy(self.original_module)` — `peft/utils/other.py:653 (ModulesToSaveWrapper.update)`; 원본 `requires_grad_(False)` — `:661`. E2: 원본·복사본 storage가 다름. 모듈 선택은 **점 경계 없는 `key.endswith(target_key)`** 다 — `peft/utils/other.py:1087-1088 (_set_trainable)`. 예를 들어 `"norm"`은 `input_layernorm`, `q_norm`까지 잡는다 [VERIFIED 소스].
- 예시: `modules_to_save=["lm_head"]`는 lm_head 1,017,118,720 params의 **복사본 + grad + AdamW state 2개**를 추가한다. 2 B/elem이면 7.578 GiB, fp32이면 15.156 GiB [E11 계산].
- **[검증 추가]** 사용자가 지정하지 않아도 `modules_to_save`에 `lm_head`가 들어가는 경로가 있다. SFT에서 `chat_template_path`가 `.jinja`/`.j2` 파일이 아닌 모델 ID이고 `clone_chat_template`이 새 token을 추가하면, TRL이 `peft_config.trainable_token_indices={"embed_tokens": added_tokens}`를 넣고 `"lm_head"`를 `modules_to_save`에 append한다(`trl/trainer/sft_trainer.py:1038-1048, 1099-1119`) [VERIFIED 소스, 미실행]. 기본값 `chat_template_path=None`이면 일어나지 않는다. resolver는 이 경우 위 lm_head 항목을 자동으로 더해야 한다.
- 참고: LoRA 대상에서 `modules_to_save`를 빼는 검사는 점 경계 정규식 `(^|.*\.){m}($|\..*)`를 쓴다(`peft/tuners/tuners_utils.py:2361-2364`). 반면 wrapper를 씌우는 `_set_trainable`은 점 경계 없는 `endswith`를 쓴다. 두 규칙이 다르므로 `"proj"`처럼 짧은 이름은 LoRA가 걸린 `q_proj` 등을 다시 감쌀 수 있다 [VERIFIED 소스, INFERRED 결과].

**5.6 `lora_dropout`** [VERIFIED 소스 + E10]
`> 0`이면 `nn.Dropout(p)`, 아니면 `nn.Identity()` — `peft/tuners/lora/layer.py:255-258`. 드롭아웃 출력은 `lora_A`의 입력으로 저장되고(x 대신 새 tensor), dropout도 backward용 tensor를 저장한다. CPU에서는 입력 dtype의 noise tensor를 저장했다(E10). CUDA fused `native_dropout`은 bool mask(1 B/elem)를 저장한다 [INFERRED]. `p=0`이면 입력을 그대로 반환한다(복사 없음) [VERIFIED: `F.dropout(x, 0.0, True) is x`].
TRL DPO는 `disable_dropout=True`가 기본이라 모든 `nn.Dropout.p=0`이 된다 — `trl/trainer/dpo_config.py:184-187`, `trl/trainer/utils.py:185-188`. GRPO는 기본 `False`(`grpo_config.py:452-455`), SFT에는 이 옵션이 없다 [VERIFIED].

**5.7 `bias`** [VERIFIED 소스 + E2]
`"none"`(기본, `peft/tuners/lora/config.py:658-660`) / `"all"`(이름이 `bias`로 끝나는 모든 파라미터, vision LayerNorm bias 포함) / `"lora_only"`(LoRA 래핑 base layer의 bias) — `peft/tuners/tuners_utils.py:525-539`. 학습되는 bias는 load dtype이다(QLoRA+TRL이면 bf16).

**5.8 DoRA** [VERIFIED 파라미터, INFERRED 런타임]
targeted Linear마다 `lora_magnitude_vector` weight `(out_features,)`가 추가된다(E2: `(384,)` fp32, TRL QLoRA에서는 bf16) — `peft/tuners/lora/variants.py:141-167`, `peft/tuners/lora/dora.py:101-130 (DoraLinearLayer.update_layer)`. forward마다 base weight 전체를 dequant하고(`dequantize_module_weight`), `eye(in_features)`와 `lora_weight (out×in)`을 만들며, `weight + scaling·lora_weight`의 열 norm을 계산한다 — `dora.py:94-99 (get_lora_weight), 132-165 (DoraLinearLayer.forward)`. layer당 O(out·in + in²) 크기의 임시 tensor가 생긴다. 예: `down_proj` in=12288이면 `eye`만 151M 원소(bf16 302 MB). **[검증 보정]** `eye`의 dtype은 `lora_A.weight.dtype`이다(`dora.py:97`). 그래서 TRL QLoRA(bf16 adapter)에서는 302 MB이고, 비양자화 LoRA(fp32 adapter)에서는 604 MB다.

**5.9 LoRA 래퍼가 만드는 추가 tensor** [VERIFIED E10 + 소스]
- 비양자화 `lora.Linear.forward`는 autocast 여부와 상관없이 항상 `x = self._cast_input_dtype(x, lora_A.weight.dtype)`를 호출한다 — `peft/tuners/lora/layer.py:1100`, `tuners_utils.py:2220-2235`. **fp32 adapter**면 x의 fp32 복사본(transient, 4 B/elem)이 생긴다. autocast가 이를 bf16으로 다시 cast하고 그 **bf16 복사본(2 B/elem)을 backward용으로 저장**한다. adapter가 bf16이면 x 자체를 저장한다(복사 없음). E10의 저장 목록이 이 차이를 보인다(`'new'` vs `'input_x'`).
- bnb `Linear4bit` LoRA 래퍼: base 출력을 `result.clone()`으로 복사한다(`peft/tuners/lora/bnb.py:533`). 입력 변환은 `not torch.is_autocast_enabled()`일 때만 한다(`:543`). 이 무인자 호출은 **CUDA autocast 상태만** 보고한다 [VERIFIED: CPU autocast 안에서 `torch.is_autocast_enabled()=False`, `('cpu')=True`]. 따라서 CUDA autocast에서는 x를 fp32로 바꾸지 않는다 [INFERRED].
- 모든 경우 `lora_B`는 r 차원 중간값(tokens × r)을 저장한다 [VERIFIED E10].

### Q6. TRL 1.14.1이 실제로 수행하는 QLoRA 준비

**6.1 SFT/DPO/GRPO 트레이너는 `peft.prepare_model_for_kbit_training`을 호출하지 않는다.** [VERIFIED]
grep 결과 이 함수는 `trl/experimental/*`(cpo, orpo, `experimental/utils.py:794`)에만 있다. 실제 순서(`trl/trainer/sft_trainer.py:1119-1167`; DPO `dpo_trainer.py:640-708`; GRPO `grpo_trainer.py:440-513` 동일):
1. `get_peft_model(model, peft_config, **kw)` — `autocast_adapter_dtype` 기본 True. ZeRO-3이면서 비양자화일 때만 False (`sft_trainer.py:1131-1134`)
2. PEFT + ZeRO-3 + GC이면 `use_reentrant=True` 강제 (`:1136-1153`)
3. PEFT + `gradient_checkpointing`이면 `model.enable_input_require_grads()` (`:1157-1158`) — input embedding 출력에 `requires_grad_(True)` forward hook을 단다. embedding weight의 grad는 만들지 않는다 (`transformers/modeling_utils.py:2141-2185`)
4. **양자화 모델이면 `requires_grad`인 모든 파라미터를 bf16으로 cast**

```python
# trl==1.14.1 trl/trainer/sft_trainer.py:1160-1167
# When using QLoRA, the PEFT adapter weights are converted to bf16 to follow the recommendations from the
# original paper ... this option is not yet supported for quantized models. See: https://github.com/huggingface/peft/issues/2889
if _is_quantized_model:
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)
```

`_is_quantized_model`은 `is_loaded_in_4bit or is_loaded_in_8bit`이다 — `sft_trainer.py:1000`, `dpo_trainer.py:582`, `grpo_trainer.py:361`.
- E5 실측(SFTTrainer, QLoRA, `dtype=bf16`): trainer 생성 직후 frozen = `Parameter bf16` + `Params4bit uint8`, trainable = `Parameter bf16` 209,280. TRL 기본 dtype이면 frozen 비양자화는 **fp32**이고 trainable은 bf16이다.

**6.2 TRL은 norm/embedding/lm_head를 fp32로 올리지 않는다. 하지만 기본 load dtype이 fp32라서 결과는 같다.** [VERIFIED E1d/E5]
`model_init_kwargs={"dtype": "bfloat16"}`(또는 `"auto"`)를 주지 않으면 비양자화 2,040,130,800 원소가 모두 fp32로 상주한다.

| 예시 모델 (각 248320×4096 = 1,017,118,720 원소) | bf16 | fp32 |
|---|---:|---:|
| `embed_tokens` 1개 | 2,034,237,440 B (1.894 GiB) | 4,068,474,880 B (3.789 GiB) |
| `embed_tokens` + `lm_head` | 4,068,474,880 B (3.789 GiB) | 8,136,949,760 B (7.578 GiB) |
| CondGen 비양자화 전체 | 4,080,261,600 B (3.800 GiB) | 8,160,523,200 B (7.600 GiB) |

**6.3 사용자가 직접 `prepare_model_for_kbit_training`을 부를 때** [VERIFIED 소스]
모든 파라미터를 freeze하고, `Params4bit`가 아닌 fp16/bf16 파라미터를 **전부 fp32로** 올린 뒤 `torch.cuda.empty_cache()`를 부른다 — `peft==0.21.2 peft/utils/other.py:193-216`. GC kwargs 기본값 `{}`가 transformers에 그대로 넘어가 `use_reentrant` 없이 `checkpoint`가 호출되고, torch는 이를 **True**로 처리한다(`other.py:190-191, 253-258`; `torch/utils/checkpoint.py:635-645`). 다만 `Trainer.train()`이 시작할 때 자기 kwargs로 GC를 다시 켜므로(`transformers/trainer.py:1489-1500`, `train()` 내부) 실제 학습은 Trainer 설정을 따른다 [INFERRED].

**6.4 그 밖의 TRL 동작**
- 사용자가 `PeftModel`을 넘기고 `ref_model=None`이면 DPO는 `"ref"` adapter를 하나 더 만든다(`dpo_trainer.py:647-674`). GRPO는 `beta != 0`일 때만 만든다(`grpo_trainer.py:452-480`, `beta` 기본값 0.0 — `grpo_config.py:676-677`). `add_adapter`의 `autocast_adapter_dtype=True` 기본값 때문에 이 adapter는 fp32로 생기고, frozen이라 TRL의 bf16 cast(`requires_grad`만 대상)를 피한다. **[검증 보정: INFERRED → VERIFIED(DPO)]** V12: tiny 4-bit + bf16 compute에서 사용자가 만든 `PeftModel`을 `DPOTrainer`에 넘기면 `default`는 bf16(trainable, 8,448)이고 `ref`는 **fp32(frozen, 8,448)** 다. 따라서 이 경로는 `P_lora × 4 B`를 추가로 상주시킨다. GRPO는 같은 코드이므로 INFERRED다.
- 양자화 모델을 adapter 없이 학습하면 `ValueError("You cannot perform fine-tuning on purely quantized models...")` — `transformers/trainer_utils.py:100-140 (validate_quantization_for_training)` [VERIFIED 소스]. plan §5.1의 "Full + 4-bit 비지원"과 일치한다.
- TRL은 vision tower를 freeze하지 않는다(트레이너에 freeze 코드 없음, grep). Full FT에서 vision 파라미터는 trainable이지만 text-only 데이터에서는 grad가 생기지 않는다(§Q8.4).

### Q7. Gradient checkpointing (transformers 5.18)

**7.1 TRL 기본값으로 켜져 있다.** `_BaseConfig.gradient_checkpointing=True`, `bf16=None → not fp16` — `trl/trainer/base_config.py:61-72, 104-105` [VERIFIED; E5 출력 `gradient_checkpointing=True gc_kwargs=None mixed_precision=bf16`].

**7.2 기본은 non-reentrant다.** [VERIFIED 소스 + E5]

```python
# transformers==5.18.0 transformers/modeling_utils.py:3138-3152 (gradient_checkpointing_enable)
if gradient_checkpointing_kwargs is None:
    gradient_checkpointing_kwargs = {"use_reentrant": False}
if offload:
    def checkpoint_func(function, *args, **kwargs):
        with save_on_cpu(pin_memory=True, device_type=device_type):
            return checkpoint(function, *args, **kwargs)
else:
    checkpoint_func = checkpoint
gradient_checkpointing_func = functools.partial(checkpoint_func, **gradient_checkpointing_kwargs)
```

Trainer는 `train()` 안에서 `args.gradient_checkpointing_kwargs`에서 `every_n_layers`(기본 1)와 `offload`(기본 False)를 꺼내고 나머지를 넘긴다 — `transformers/trainer.py:1489-1500`. E5: decoder layer의 `_gradient_checkpointing_func.keywords == {'use_reentrant': False}`. 참고로 torch 2.14.1의 `checkpoint`는 `use_reentrant=None`이면 경고 후 **True**를 쓴다(`torch/utils/checkpoint.py:635-645`). kwargs를 `{}`로 넘기는 경로(§6.3)는 reentrant가 된다.

**7.3 단위: `GradientCheckpointingLayer` 인스턴스 하나(디코더 layer 1개).** [VERIFIED]
`Qwen3_5DecoderLayer(GradientCheckpointingLayer)` — `modeling_qwen3_5.py:860`; `Qwen3_5VisionBlock(GradientCheckpointingLayer)` — `:1093`. `every_n_layers`는 이 인스턴스들에만 적용된다(`modeling_utils.py:3200-3210`). E5: decoder·vision block 모두 `gradient_checkpointing=True`.

```python
# transformers/modeling_layers.py:81-110 (GradientCheckpointingLayer.__call__, 발췌)
if self.gradient_checkpointing and self.training:
    if "use_cache" in kwargs and kwargs["use_cache"]:
        kwargs["use_cache"] = False
    if not self._can_checkpoint_with_cache:
        ... kwargs["past_key_values"] = None
    return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
```

**7.4 경계에 남는 것** [VERIFIED 구조 / INFERRED 수명]
non-reentrant checkpoint는 구간 안의 saved tensor를 버리고, 재계산 closure가 **입력**을 붙잡는다. `Qwen3_5TextModel.forward`는 `decoder_layer(hidden_states, position_embeddings=..., attention_mask=..., position_ids=..., ...)`로 호출하므로(`modeling_qwen3_5.py:1290-1299`) layer마다 남는 것은 입력 `hidden_states` `[B, T, 4096]` 1개다. `position_embeddings`(cos/sin)와 `position_ids`는 모든 layer가 같은 객체를 공유한다. mask는 layer type별로 1개씩, 즉 `full_attention`용 causal mask와 `linear_attention`용 recurrent mask가 같은 type의 layer끼리 공유된다(`modeling_qwen3_5.py:1282-1285`) **[검증 보정: mask 2종]**. 재계산은 forward 때의 autocast 상태로 실행된다 — `torch/utils/checkpoint.py:1897, 1944-1960 (_checkpoint_without_reentrant_generator.recompute_fn)`. RNG state는 기본 보존된다(`preserve_rng_state=True`, CPU/CUDA RNG state 소량).
- 경계 tensor dtype = residual stream dtype = **load dtype** (§Q9.3; TRL 기본이면 fp32 4 B/elem).
- `offload=True`면 경계 tensor는 pinned host memory로 간다(device→host 복사). `every_n_layers=k`이면 일부 layer만 checkpoint하고 나머지는 전체 activation을 유지한다. **[검증 보정]** 선택 규칙은 "디코더 layer 번호 % k"가 아니다. `self.modules()` 순서로 센 **모든 `GradientCheckpointingLayer`의 일련번호** `layer_index % k == 0`이다(`modeling_utils.py:3181-3207 (_set_gradient_checkpointing)`; docstring은 `:3123-3131`). CondGen은 `model.visual`이 `language_model`보다 먼저 생성되므로(`modeling_qwen3_5.py:1318-1319`) vision block 27개가 앞 번호를 차지한다. 그래서 디코더 layer j는 `(27 + j) % k == 0`일 때 checkpoint된다. CausalLM은 `j % k == 0`이다 [VERIFIED V10: tiny(vision 2, decoder 4), k=3에서 CondGen decoder `[F,T,F,F]`, CausalLM `[T,F,F,T]`]. layer type(linear/full attention)마다 activation 크기가 다르므로 activation ledger는 이 번호 규칙으로 checkpoint 대상 layer를 골라야 한다.
- 학습 중 cache: TRL은 forward에 `use_cache=False`를 명시하고(`sft_trainer.py:314, 1795`, `dpo_trainer.py:1279, 1332`, `grpo_trainer.py:1423`), Trainer는 `config.use_cache = args.use_cache`(기본 False)로 둔다(`trainer.py:633-634`, `training_args.py:953-958`) [VERIFIED].

### Q8. Optimizer와 gradient

**8.1 기본 optimizer = `adamw_torch_fused`** (torch ≥ 2.8). `training_args.py:803-813`; `_get_adamw_torch`가 `fused=True`를 추가한다 — `transformers/trainer_optimizer.py:201-208` [VERIFIED; E5: `torch.optim.adamw.AdamW {'fused': True}`].

**8.2 torch AdamW state** [VERIFIED 소스 + E5/E8]
첫 `optimizer.step()`에서 `p.grad is not None`인 파라미터에 대해서만 지연 생성된다. `exp_avg`/`exp_avg_sq`는 `zeros_like(p)`이므로 **파라미터 dtype을 따른다**. `step`은 fused/capturable이면 device 위 fp32 scalar다 — `torch==2.14.1 torch/optim/adam.py:150-189 (Adam._init_group)`, `torch/optim/optimizer.py:221-226 (_get_scalar_dtype)`.
- E5/E8: bf16 파라미터 → state bf16 (4.0001 B/param + scalar), fp32 → state fp32 (8 B/param + scalar). TRL QLoRA(adapter bf16)는 **4 B/param**, 비양자화 LoRA(adapter fp32)는 **8 B/param**이다.
- non-fused foreach 경로(`adamw_torch`)는 그룹 전체 크기의 중간 tensor 목록을 만들 수 있다 [INFERRED, 미측정].

**8.3 bnb 8-bit / paged** [VERIFIED 소스 + E8]
- `numel ≥ min_8bit_size(4096)`: state1/state2 uint8 + absmax1/absmax2 fp32 `ceil(n/256)`개 → `2 + 8/256 = 2.03125 B/param`. 그보다 작으면 fp32 state 2개(8 B/param) — `bitsandbytes/optim/optimizer.py:491-532 (Optimizer2State.init_state)`, 기본값 `bitsandbytes/optim/adamw.py:9-20 (AdamW.__init__)`. qmap(256 fp32 × 2)은 optimizer당 공유된다.
- bnb 32-bit state는 파라미터 dtype과 관계없이 fp32(8 B/param)다 [E8].
- paged: `numel ≥ 1e5`인 state만 `F.get_paged`(`lib.cget_managed_ptr`)로 할당한다 — `optimizer.py:374-392`, `functional.py:91-100`. torch caching allocator **밖**의 managed memory라 `torch.cuda.memory_allocated`에 잡히지 않는다 [INFERRED: C 구현 미확인]. r=16 LoRA에서 `(16×4096)=65,536` 원소 tensor는 paging 대상이 아니고, `(12288×16)=196,608`은 대상이다.
- 8-bit bnb optimizer를 쓰면 Trainer가 모든 `nn.Embedding` 파라미터를 32-bit state로 등록한다 — `transformers/trainer.py:1314-1326` [VERIFIED 소스]. `modules_to_save=["embed_tokens"]`나 Full FT에 해당한다.

**8.4 grad dtype과 수명** [VERIFIED E5]
- grad dtype = 파라미터 dtype. fp32 LoRA는 bf16 autocast 아래에서도 **fp32 grad**(E5 `lora bfloat16`), TRL QLoRA bf16 adapter는 **bf16 grad**다.
- gradient accumulation: post-accumulate-grad hook이 2 step × GA 2 = 4번 호출됐다. grad는 microbatch 사이에 유지·누적되고 optimizer step 뒤 `model.zero_grad()`(`transformers/trainer.py:1908`; `Module.zero_grad(set_to_none=True)` 기본 — `torch/nn/modules/module.py:2957`)로 `None`이 된다. E5 학습 후 grad는 `None`이다.
- 실행되지 않은 trainable 파라미터는 grad도 state도 생기지 않는다. E5(text-only 데이터, all-linear): vision LoRA 20개 tensor는 optimizer state 0개, text LoRA 62개는 state를 가졌다 [VERIFIED E5d]. bnb도 `p.grad is None`이면 건너뛴다(`optimizer.py:325-333`).

### Q9. 혼합 정밀도 런타임 임시 메모리

**9.1 `Linear4bit` forward** [VERIFIED 소스, CUDA 분기는 INFERRED]
`x = x.to(self.compute_dtype)` 후 `bnb.matmul_4bit(...)`를 호출하고 결과를 `.to(inp_dtype)`로 되돌린다 — `bitsandbytes/nn/modules.py:609-637`. grad가 필요하면 `MatMul4Bit.apply` → `torch.ops.bitsandbytes.gemm_4bit`로 간다(`autograd/_functions.py:303-362, 491`). CUDA 커널 선택 — `bitsandbytes/backends/cuda/ops.py:934-982`:
- `M(=토큰 수) > 1536` 또는 `K % blocksize != 0` → **dequant + `F.linear` fallback**: `B_dq = torch.empty(shapeB, dtype=A.dtype)`로 **weight 전체를 compute dtype으로 복원**한다. DQ면 fp32 `nb` 원소 tensor **2개**가 함께 살아 있다(`absmax_dq`와 `absmax = absmax_dq + absmax_offset`, 둘 다 함수 끝까지 참조됨) — `ops.py:903-916` **[검증 보정: 4·nb → 8·nb]**.
- `M ≤ 4` → fused kernel. `5 ≤ M ≤ 1536` → GPU 아키텍처별 heuristic(`_gemm_4bit_use_custom_cuda`, `ops.py:584-812`). fp32 A는 `M < 8`일 때만 fused kernel.
- 학습 시퀀스는 보통 M > 1536이므로 **layer 호출마다 `n × bytes(compute)` transient**가 생긴다. 예: 12288×4096 bf16 = 100,663,296 B.
- `ctx.tensors = (None, B)`: activation A를 저장하지 않고 packed weight만 참조한다(`_functions.py:357-360`).
- **[검증 추가]** `Linear4bit.forward`는 입력이 compute dtype이 아니면 `x.to(compute_dtype)` 사본(tokens × in × bytes(compute))을 만들고, 출력은 `.to(inp_dtype)`로 되돌린다(`bitsandbytes/nn/modules.py:626-637`). TRL 기본(fp32 residual, §9.3) + bf16 compute에서는 layer 호출마다 bf16 입력 사본(transient)과 fp32 출력(tokens × out × 4 B)이 생긴다 [VERIFIED 소스, 크기 INFERRED].

**9.2 `Linear4bit` backward** [VERIFIED 소스, INFERRED 크기]

```python
# bitsandbytes==0.50.2 bitsandbytes/autograd/_functions.py:381-384 (MatMul4Bit.backward)
if req_gradA:
    # dequantize returns [N, K]; matmul(grad_output[M,N], [N,K]) = grad_A[M,K].
    grad_A = torch.matmul(grad_output, F.dequantize_4bit(B, ctx.state).to(grad_output.dtype))
```

`dequantize_4bit` 출력 dtype = `quant_state.dtype` = **load dtype**이다(`functional.py:992-1077`). transient = `n·bytes(load)` + (load≠grad dtype이면) `n·bytes(grad)` + DQ absmax `4·nb`. 예: 12288×4096에서 TRL 기본 fp32 로드는 201,326,592 + 100,663,296 = 301,989,888 B, bf16 로드는 100,663,296 B.

**9.3 activation dtype은 load dtype을 따른다.** [VERIFIED E6 + E5c(실제 SFTTrainer step)]
embedding은 autocast 대상이 아니라 출력이 weight dtype이다. residual add는 type promotion을 하고, `Linear4bit`는 출력을 입력 dtype으로 되돌린다. 그래서 **TRL 기본(fp32 로드)에서는 bf16 autocast가 켜져 있어도 layer 입출력과 `Linear4bit+LoRA` 입출력이 fp32**다. E5c에서 SFTTrainer 학습 step 중 `torch.is_autocast_enabled('cpu')=True`였고, layer0 in/out은 fp32/fp32(기본)와 bf16/bf16(`dtype=bf16`)이었다. Trainer 자체는 autocast를 걸지 않고(`transformers/trainer.py:2178-2183`), accelerate가 `model.forward`를 autocast로 감싼 뒤 출력의 fp16/bf16 tensor를 **fp32로 변환**한다 — `accelerate==1.15.0 accelerate/accelerator.py:1824-1835`, `accelerate/utils/operations.py:901-948 (ConvertOutputsToFp32)`. 반환되는 logits가 있으면 fp32 사본이 생긴다(TRL loss 경로별 반환 여부는 loss 조사 범위).

**9.4 `compute_dtype=fp32`(BitsAndBytesConfig 기본)** [VERIFIED CPU E12, INFERRED CUDA]
입력 x의 fp32 복사본(4 B/elem)과 fp32 `B_dq`(4 B/elem)가 생긴다. 내부 `F.linear`는 autocast 아래에서 bf16으로 다시 cast된다(CPU에서 matmul 출력 bf16 확인). bf16 compute 대비 transient가 약 3배(2 → 6 B/elem)다.

**9.5 autocast weight cache** [VERIFIED 소스 / INFERRED 크기]
autocast는 가장 바깥 context를 나갈 때 cast cache를 비운다 — `torch/amp/autocast_mode.py:342-351`. fp32 LoRA weight(비양자화 LoRA 기본)는 forward 동안 bf16 사본이 cache에 남는다(`실행 경로의 P_lora × 2 B`. 예시 모델에 text-only 데이터면 text LoRA 43,278,336 × 2 = 82.5 MiB, vision LoRA는 실행되지 않음). TRL QLoRA는 adapter가 이미 bf16이라 0이다. checkpoint 재계산은 자체 autocast context 안에서 돌기 때문에 layer 단위로 생겼다 사라진다.

**9.6 LoRA 경로** — §5.9 참조(`result.clone()` 출력 크기 1개, fp32 adapter의 입력 cast 사본, r 차원 중간값).

### Q10. 비-PyTorch 오버헤드와 caching allocator

**10.1 소스로 확인한 사실** [VERIFIED]
- allocator 상수: `kMinBlockSize = 512`(모든 크기를 512 B 이상으로 반올림), `kSmallSize = 1 MiB`(small pool 상한), `kSmallBuffer = 2 MiB`(small pool segment), `kMinLargeAlloc = 10 MiB`, `kRoundLarge = 2 MiB` — `torch==2.14.1 torch/include/c10/core/AllocatorConfig.h:16-25`. 1–10 MiB 할당이 쓰는 `kLargeBuffer` 값은 헤더에 없다 [UNKNOWN; 일반적으로 20 MiB로 알려져 있으나 미확인].
- `expandable_segments` 기본 false(`AllocatorConfig.h:353` `use_expandable_segments_{false}`; 접근자는 `:213-215`) **[검증 보정: 인용 줄]**. 환경변수는 `PYTORCH_ALLOC_CONF`(구 `PYTORCH_CUDA_ALLOC_CONF`도 지원, `:158-159`).
- `memory_allocated`는 tensor 점유만 센다. "unused memory can be held by the caching allocator and some context needs to be created on GPU" — `torch/cuda/memory.py:525-540`. CUDA context와 driver 예약은 `memory_reserved`에도 잡히지 않는다 [INFERRED: reserved는 caching allocator segment만 센다].
- cuBLAS/cuBLASLt workspace 크기는 런타임 API로 조회할 수 있다: `torch.backends.cuda.cublas_workspace_size()`, `cublaslt_workspace_size()`. 기본 빌드에서는 `TORCH_CUBLASLT_UNIFIED_WORKSPACE`로 cuBLASLt가 cuBLAS workspace를 재사용한다 — `torch/backends/cuda/__init__.py:340-405`. 기본 크기 값 자체는 C++에 있어 이 머신에서 확인할 수 없다 [UNKNOWN].
- transformers 로딩의 `caching_allocator_warmup`(§4.4)은 로드 직후 reserved를 모델 크기 이상으로 만든다. TRL은 `torch_empty_cache_steps=None`(기본)이므로 학습 중 `empty_cache`를 부르지 않는다(`trl/trainer/base_config.py:47-50, 97-102`). PEFT `prepare_model_for_kbit_training`은 upcast 후 `empty_cache`를 부른다(`peft/utils/other.py:211-216`).
- bnb paged optimizer buffer는 allocator 밖 managed memory로 보인다(§8.3) [INFERRED].

**10.2 가정 범위 (calibration 필요)** [INFERRED/ASSUMPTION — 이 머신에서 측정 불가]

| 항목 | 가정 범위 | 근거·비고 |
|---|---|---|
| CUDA context + 모듈 로드 (driver, cuBLAS handle 포함) | 0.3–1.0 GiB / GPU | PyTorch 문서(R15/R16)가 "context" 존재만 명시. GPU 세대·driver·lazy loading에 따라 다름 |
| cuBLAS(+Lt) workspace | (handle, stream)마다 수 MiB–수십 MiB | 런타임 API로 측정 가능. 크기 기본값 미확인 |
| cuDNN workspace | 이 모델에서는 conv1d/conv3d fallback 경로만 해당 | GatedDeltaNet의 causal conv1d fallback과 vision Conv3d. 크기 미확인 |
| allocator 단편화(reserved − allocated, 피크 시점) | 미정 | GPU 검증의 `max_memory_reserved − max_memory_allocated`로 보정 |

측정 방법(GPU 검증 단계): `torch.cuda.init()` 직후와 첫 matmul 후에 `total − free(mem_get_info) − memory_reserved()`를 기록하면 비-PyTorch 점유를 얻는다. 학습 피크는 `max_memory_allocated`/`max_memory_reserved`로 분리 기록한다.

---

## 3. 구현 시사점 (Implementation implications)

### 3.1 loading scope / dtype resolver (ArchitectureAdapter + compatibility registry)

```yaml
# registry profile 필드 제안 (값은 예시 모델 + TRL 1.14.1 문자열 진입 기준)
loading:
  entrypoint: trl_model_id            # trl_model_id | user_model_object(AutoModelForCausalLM)
  architecture_class: Qwen3_5ForConditionalGeneration   # getattr(transformers, config.architectures[0])
  scope: conditional_generation        # vision tower 상주(456,010,480 params) | text_only
  dropped_key_patterns: []             # text_only: ["^model\\.visual\\.", "^mtp\\."]
  key_prefix_remap: {}                 # text_only: {"^model\\.language_model\\.": "model."}
  load_dtype: float32                  # TRL: model_init_kwargs.dtype 미지정 시 float32 (VERIFIED)
                                       # "auto" → config.dtype → checkpoint 첫 float dtype (예시: bfloat16)
  device_map: auto                     # 분산(MULTI_GPU/DEEPSPEED)이면 None
quantization:                          # BitsAndBytesConfig 기본값은 fp4/noDQ/fp32 → 전부 명시해서 resolve
  quant_type: nf4
  double_quant: true
  compute_dtype: bfloat16
  quant_storage: uint8
  blocksize: 64                        # transformers에서 바꿀 수 없음
  nested_blocksize: 256
  skip_modules_effective: ["lm_head"]  # 사용자 목록은 기본값을 대체(lm_head 다시 포함 여부 확인 필수)
peft:
  adapter_dtype: bf16                  # 양자화+TRL. 비양자화 LoRA는 fp32. ZeRO-3 비양자화는 base dtype
  target_resolution: all-linear        # vision Linear 포함, lm_head 제외. qwen3_5는 None 불가
optimizer:
  name: adamw_torch_fused              # state dtype = param dtype
gradient_checkpointing: {enabled: true, use_reentrant: false, granularity: decoder_layer, every_n_layers: 1, offload: false}
  # [검증 보정] every_n_layers>1이면 vision block을 포함한 GradientCheckpointingLayer 일련번호로 선택(CondGen: (27+j)%k==0, §7.4)
```

1. **기본 product preset은 `model_init_kwargs.dtype="bfloat16"`을 명시한 profile로 계산하고, TRL 기본(fp32) profile은 별도로 표시한다.** 예시 모델에서 둘의 상주 차이는 3.800 GiB다(아래 표). `requested`에 dtype이 없으면 `resolved.load_dtype = float32`로 기록한다(plan §11.3 requested/resolved).
2. "텍스트 전용 데이터"라도 TRL 문자열 진입이면 **vision tower를 상주 가중치에 포함**한다(4-bit vision 451,178,496 원소 + 비양자화 4,831,984 원소). text-only loader(`Qwen3_5ForCausalLM`)는 E13에서 header 키가 1:1 대응함을 확인했으므로 `user_model_object` profile로 선택 가능하다. 다만 사용자가 모델 객체를 만들어 넘기는 실행 경로라는 점을 표시한다.
3. 비양자화 파라미터 dtype = `load_dtype`. Qwen3.5에는 `_keep_in_fp32_modules`가 없다. `A_log`/`dt_bias`/conv1d/norm/embedding/lm_head 모두 동일 규칙이다.

### 3.2 가중치 byte 공식 (AllocationSpec: `weights.base_q4`, `weights.base_dense`)

```text
quantized(m)  ⇔  module_type is exactly nn.Linear (or HF Conv1D)  ∧  m ∉ skip_effective  ∧  m in loading scope
W_q4  = Σ_quantized B_q4_DQ(n)       # DQ off: B_q4_noDQ
B_q4_DQ(n)   = ceil(n/2) + ceil(n/64) + 4·ceil(ceil(n/64)/256) + 4 + 64 + 1024
B_q4_noDQ(n) = ceil(n/2) + 4·ceil(n/64) + 64
(정수 byte 정밀도가 필요하면 각 구성 tensor를 512 B 배수로 올림: 예시 모델 +0.42 MB)
W_nonq = Σ_nonquantized numel × bytes(load_dtype)     # Linear4bit bias 포함
W_dense (비양자화 LoRA/Full) = Σ_all numel × bytes(load_dtype)
```

### 3.3 학습 상태 byte (AllocationSpec: `adapter.weights`, `adapter.grads`, `optimizer.state`)

```text
P_lora = Σ_targets r·(in + out)  [+ out if lora_bias] [+ out if use_dora]   # in/out = 모듈 속성
P_mts  = Σ_{modules_to_save} numel(module)                                   # 원본은 W_*에 그대로 남고 복사본 추가
b_ad   = 2 if (quantized ∧ TRL) else (4 if autocast_adapter_dtype else bytes(base))
b_mts  = 2 if (quantized ∧ TRL) else bytes(load_dtype)
W_adapter = P_lora·b_ad + P_mts·b_mts (+ trainable bias)
G         = 같은 집합·같은 dtype (첫 backward부터 optimizer step 후 zero_grad까지 상주)
S_adamw_torch(_fused) = Σ_tensors 2·n·b_param  + N_tensors × 4 B (step scalar: fused/capturable이면 device 위 512 B 블록, 아니면 CPU)
S_bnb_8bit = Σ_tensors (n ≥ 4096 ? 2n + 8·ceil(n/256) : 8n)   # nn.Embedding 파라미터는 항상 8n
S_bnb_32bit = Σ 8n
실행 경로에 없는 trainable 파라미터(예: text-only 데이터의 vision LoRA)는 G = S = 0, 가중치만 상주.
```

timepoint 규칙(TrainerAdapter):
- `S`는 **첫 `OPTIMIZER_STEP` timepoint에서 처음 생긴다**. 첫 step 피크 = W + G + S(새로 할당).
- GA > 1이면 두 번째 microbatch forward부터 G 전체가 상주한다. GA = 1이면 backward 중 G가 점진적으로 생긴다.
- `set_to_none=True`이므로 step 뒤 G는 0이 된다.

### 3.4 예시 모델 profile별 정적 상주량 (activation 제외, r=16, E11 계산)

| profile | W_4bit | W_nonq | P_lora | W_lora | grad | optim | 합계 (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| QLoRA, TRL 기본 dtype(fp32), CondGen, all-linear | 3.541 | 7.600 | 51,265,024 | 0.0955 | 0.0955 | 0.1913 | **11.523** |
| QLoRA, dtype=bf16, CondGen, all-linear | 3.541 | 3.800 | 51,265,024 | 0.0955 | 0.0955 | 0.1913 | **7.723** |
| QLoRA, dtype=bf16, CondGen, text-only targets | 3.541 | 3.800 | 43,278,336 | 0.0806 | 0.0806 | 0.1615 | 7.664 |
| QLoRA, dtype=bf16, CausalLM(text-only 객체) | 3.324 | 3.791 | 43,278,336 | 0.0806 | 0.0806 | 0.1615 | 7.438 |
| QLoRA, dtype=bf16, CondGen, all-linear, paged_adamw_8bit | 3.541 | 3.800 | 51,265,024 | 0.0955 | 0.0955 | 0.0971 | 7.629 |
| LoRA, dtype=bf16, CondGen, all-linear | 0 | 17.527 | 51,265,024 | 0.1910 | 0.1910 | 0.3823 | 18.291 |
| LoRA, TRL 기본 dtype(fp32), CondGen, all-linear | 0 | 35.054 | 51,265,024 | 0.1910 | 0.1910 | 0.3823 | 35.819 |

- all-linear 기준 grad/optim은 상한이다. text-only 데이터면 vision LoRA 7,986,688 params에는 grad/state가 생기지 않는다(§8.4). 예: 첫 행의 grad+optim은 실제로 text 부분(43,278,336)만큼만 생긴다.
- `modules_to_save=["lm_head"]` 추가 시 위 합계에 +7.578 GiB(2 B/elem: 복사본+grad+state 2개)다.
- paged_adamw_8bit 행의 optim에는 `numel ≥ 1e5` state의 managed memory가 포함돼 있다. 이것을 VRAM에 넣을지는 미확정 4번에서 정한다.

### 3.5 로딩 phase (`MODEL_LOAD_AND_QUANTIZE`)

```text
peak_allocated_load ≤ W_final(device) + max_q [ n_q·bytes(load_dtype) + (DQ ? 8·ceil(n_q/64) : 0) ]   # W_q 원본 + (_absmax, _absmax−offset) fp32 임시
                      # [검증 보정] 비양자화 tensor의 bf16→fp32 변환이 host에서 일어난다는 가정(INFERRED, 미확정 1)이 붙은 상한이다.
                      # device에서 변환된다면 가장 큰 비양자화 tensor의 원본 사본(embed_tokens bf16 2,034,237,440 B)이 순간적으로 더해진다.
reserved_after_load ≥ min( Σ_q 0.5·n + Σ_nq n·bytes(load_dtype),  total_device_memory − 1.2 GiB )   # caching_allocator_warmup
host_RAM_transient ≈ 가장 큰 단일 tensor (예: embed_tokens bf16 2.03 GB, CPU 쪽 fp32 변환이면 +4.07 GB)  [INFERRED]
```

- 예시 모델(`gate_proj` 기준): W_final 대비 초과분 ≤ 207,618,048 B (0.193 GiB, fp32 로드) / 106,954,752 B (0.100 GiB, bf16 로드). §4.3의 0.218/0.124 GiB는 이 순간 materialize된 전체(출력 포함)다. 로딩 피크는 학습 피크보다 낮을 가능성이 크지만, warmup으로 reserved가 먼저 커지는 점을 `allocator_slack`의 근거로 기록한다.
- fit 판정 **[검증 보정]**: 단일 GPU, bnb 4-bit, `device_map="auto"`, `max_memory` 미지정이면 `S_load ≤ (free + reserved − allocated) × 0.81`이어야 로딩된다(§4.5). 넘으면 `ValueError`다. `S_load = Σ_q 0.5·n + Σ_nq n·bytes(load_dtype)`(`compute_module_sizes`; warmup 합계와 1 MB 이내로 같다)이다. 예시 CondGen은 S = 11.032 GiB(fp32 로드)이므로 **free ≥ 13.62 GiB**, S = 7.232 GiB(bf16 로드)이므로 **free ≥ 8.93 GiB**가 필요하다 [VERIFIED V9 CPU 시뮬레이션 + 소스, CUDA 미실행]. 사용자가 `max_memory`를 주면 계수는 0.90이다. 분산(`device_map=None`)은 사전 검사 없이 OOM으로 실패한다. 비양자화 + `"auto"`는 CPU offload로 "로딩 성공"할 수 있으므로 VRAM-only 판정에서는 `unsupported/offload` 경고를 낸다.

### 3.6 런타임 transient (AllocationSpec, evidence=analytic, 수명 = 해당 matmul 호출)

```text
q4_forward_dequant  = n_max_q · bytes(compute_dtype) [+ 8·ceil(n/64) if DQ]     # tokens(M) > 1536 (CUDA) 또는 compute fp32  [검증 보정: DQ 항 4→8·nb, ops.py:911-914]
q4_backward_dequant = n_max_q · bytes(load_dtype) + (load_dtype ≠ grad_dtype ? n_max_q · bytes(grad_dtype) : 0) [+ 4·ceil(n/64)]
lora4bit_clone      = tokens × out_features × bytes(base_out_dtype)                 # PEFT bnb 래퍼의 result.clone()
lora_fp32_input_copy= tokens × in_features × (4 transient + 2 saved)                 # 비양자화 lora.Linear + fp32 adapter
dora_extra          = in²·bytes(adapter) + O(3·out·in)·bytes(compute) per targeted layer   # use_dora; [검증 보정] eye는 adapter dtype(dora.py:97)
autocast_cache      = P_lora × 2  (fp32 adapter일 때만, forward 구간)
```

- 예시 모델 최댓값(12288×4096, nb = 786,432): DQ 항을 빼면 forward 100.7 MB, backward 100.7 MB(bf16 로드) / 302.0 MB(fp32 로드)다. DQ를 포함하면 forward 106,954,752 B(+8·nb), backward 103,809,024 B(bf16 로드) / 305,135,616 B(fp32 로드)(+4·nb)다 [검증 보정].
- `5 ≤ M ≤ 1536` 구간의 fused/dequant 선택은 GPU 아키텍처별 heuristic이다. 보수적으로 dequant 경로(최대 transient)를 `bytes_high`, fused 경로(0)를 `bytes_low`로 둔다.

### 3.7 activation dtype 규칙 (activation ledger 담당자에게 전달)

- residual stream·checkpoint 경계 tensor dtype = `embed_tokens.weight.dtype` = `load_dtype`. **TRL 기본은 fp32(4 B)** 이고 `dtype=bf16`이면 2 B다(E5c/E6). `Linear4bit` 출력도 입력 dtype으로 되돌려진다.
- checkpoint 경계: layer마다 `[B, T, hidden]` 1개 × `bytes(load_dtype)`, 공유 position embedding/mask 1벌. 재계산 중에는 그 layer 하나의 full activation + 위 transient가 살아 있다.
- LoRA saved tensor: fp32 adapter(비양자화)는 layer마다 bf16 입력 사본, bf16 adapter(QLoRA)는 입력 공유. 모든 경우 r 차원 중간값을 저장한다. dropout>0이면 mask(CUDA bool 1 B/elem, INFERRED)와 dropped 입력 사본이 추가된다. DPO 기본 `disable_dropout=True`면 p=0 → 추가 없음.
- accelerate는 forward 출력의 bf16 tensor를 fp32로 변환한다. logits를 반환하는 loss 경로면 `tokens × vocab × 4 B` 사본이 생긴다(loss 경로별 확인 필요).

### 3.8 입력 검증 규칙 (compatibility resolver)

- `objective ∈ {SFT, DPO, GRPO}` × `QLoRA`: adapter 없는 4-bit는 거절한다(`validate_quantization_for_training`).
- `target_modules` 미지정 + `model_type=qwen3_5` → 사용자 입력 단계에서 preset을 강제한다(PEFT가 `ValueError`).
- `llm_int8_skip_modules`를 사용자가 바꾸면 `lm_head` 포함 여부를 다시 계산한다. 빠지면 `lm_head`가 4-bit가 되어 W와 PEFT `all-linear` 결과가 모두 바뀐다.
- `modules_to_save` 이름은 점 경계 없는 suffix 매칭이므로 resolve된 실제 모듈 목록을 결과에 표시한다(plan §5.2 "해석한 모듈 목록 표시").
- `compute_dtype=float32`(bnb 기본)는 별도 profile로 취급한다(§9.4 transient 증가). 권장 profile은 bf16이다.

---

## 4. 미확정 사항 (Open questions)

1. **CUDA 실행 경로 미검증** [INFERRED]: `gemm_4bit`의 fused/dequant 선택(M ≤ 1536 구간), GPU 위 양자화 kernel의 임시 버퍼, `.to(cuda, dtype)`에서 dtype 변환이 host와 device 중 어디서 일어나는지(host RAM transient 크기에 영향)를 GPU 검증(plan §17)에서 `torch.cuda.memory_stats`로 확인해야 한다.
2. **custom op 안의 autocast**: `compute_dtype=fp32`일 때 bnb fallback의 `F.linear`가 autocast로 bf16 cast되는 것은 CPU default 구현에서만 확인했다(E12). CUDA dispatch에서도 같은지 미확인.
3. **CUDA context / cuBLAS·cuDNN workspace / `kLargeBuffer` 실제 값**: 이 머신(CUDA 없음)에서 측정 불가. 대상 GPU별 calibration profile 값으로 채워야 한다(§10.2).
4. **bnb paged optimizer의 managed memory**가 `nvidia-smi`·`mem_get_info`에 어떻게 잡히는지, 그리고 압박 시 host로 paging되어 VRAM 피크에서 빠지는 조건. 이 정의가 정해지기 전에는 paged 절감량을 확정치로 쓰지 않는다.
5. **TRL `create_model_from_path`의 architecture 추론이 revision을 무시**(§1.3)한다. 고정 revision과 `main`의 `config.architectures`가 다르면 다른 class가 로드될 수 있다. registry는 class를 고정 revision의 config로 결정하고, 실행 시 `model_init_kwargs.revision`과 함께 경고를 띄울지 결정이 필요하다.
6. **AutoProcessor 의존성**: 이 venv에서는 Pillow/torchvision이 없어 MiMo의 `AutoProcessor.from_pretrained`가 실패했다. TRL 트레이너는 `processing_class=None`이면 AutoProcessor를 호출하므로(`sft_trainer.py:1003-1006`) 실제 학습 환경 lock에 Pillow 포함 여부, 또는 tokenizer를 명시적으로 넘기는 preset이 필요하다(전처리 조사와 조율).
7. **`modules_to_save`가 4-bit 모듈을 가리키는 경우**(예: skip에서 빠진 `lm_head`): `Linear4bit` deep copy가 학습 가능한지는 확인하지 않았다. 지원 조합에서 차단하는 편이 안전하다.
8. **PeftModel 입력 + DPO/GRPO의 `"ref"` adapter** — **[검증으로 일부 해소]** DPO는 V12에서 fp32 frozen 사본(`P_lora × 4 B`)을 실측했다(§6.4). 남은 것은 GRPO(`beta≠0`)의 실측과, ref adapter를 forward할 때 생기는 activation/transient(adapter 전환 비용)다.
9. **vision tower가 실행되지 않는 trainable 파라미터**에 대해 DDP/FSDP(다중 GPU)에서도 grad/state가 생기지 않는지는 단일 장치에서만 확인했다(E5d, V6). 다중 GPU 확장(plan §9.8) 때 재확인이 필요하다. [검증 추가] `PeftModel`은 `PreTrainedModel`이 아니므로 Trainer는 `ddp_find_unused_parameters=None`일 때 `find_unused_parameters=True`로 DDP를 감싼다(`transformers/trainer.py:737-746`). 그래서 미사용 파라미터 때문에 DDP가 멈추지는 않을 것으로 보인다. 다만 모든 rank에서 쓰이지 않은 파라미터의 grad가 None으로 남는지는 DDP reducer(C++) 동작이라 미확인이다.

---

## 검증 로그 (Verification log)

| 항목 | 값 |
|---|---|
| 검증자 | verify-loading-quant-peft (적대적 재검증, 원 작성자 아님) |
| 날짜 | 2026-10-04 |
| 환경 | 같은 공유 venv(torch 2.14.1 / transformers 5.18.0 / trl 1.14.1 / peft 0.21.2 / bitsandbytes 0.50.2 / accelerate 1.15.0, 버전을 import로 재확인), macOS arm64 CPU(MPS가 있으므로 `use_cpu=True`로 CPU를 강제하고 파라미터 device가 전부 `cpu`인지 확인) |
| 원칙 | 원 문서의 실험 산출물(`tiny_qwen35`, `tensor_inventory.json`)은 재사용하지 않았다. header를 다시 받고, tiny clone을 새로 만들고, 스크립트도 새로 작성했다 |

### 재현 실험 (scratch `/tmp/vf-research/scratch/verify-loading-quant-peft/`, 커밋하지 않음)

| ID | 스크립트 | 내용과 결과 |
|---|---|---|
| V1 | `v_headers.py` | `huggingface_hub.get_safetensors_metadata`(header만)로 받음: 760 tensors 전부 BF16, 9,409,813,744 params. `model.visual.*` 333개 = 456,010,480, 나머지 427개 = 8,953,803,264. index `total_size` 18,819,627,488 = 2 × params |
| V2 | `v_q4bytes.py` | CPU `F.quantize_4bit`(nf4, bs64)에서 QuantState 구성 tensor의 byte를 직접 합산했다. 4096², 12288×4096, 8192×4096, 32×4096, 1152×3456, 4608², 4097×3, 3×7 × DQ on/off **16개 경우 모두 공식과 byte 단위 일치** |
| V3 | `v_meta_inventory.py` | 실제 config로 full-size meta 모델을 만들고 `Bnb4BitHfQuantizer._process_model_before_weight_loading`(transformers와 같은 경로)을 적용했다. CondGen: skip `['lm_head']`, Linear4bit 358(vision 110), 양자화 7,369,682,944, 비양자화 2,040,130,800, DQ 3,802,183,024 B, noDQ 4,145,469,568 B, 512 B 반올림 +422,544 B. CausalLM: 248, 3,569,313,760 B. `llm_int8_skip_modules=["model.visual"]`이면 `lm_head`가 Linear4bit가 된다 |
| V4 | `v_peft_meta.py` | 위 meta 모델에 실제 `get_peft_model(..., low_cpu_mem_usage=True)` 적용: all-linear 358 layer / 51,265,024(vision 7,986,688). `exclude_modules=".*visual.*"`, CausalLM, 이름 목록 세 경우 모두 248 / 43,278,336 |
| V5 | `v_target_none.py` | `LoraConfig(r=16)`(target 미지정) → `ValueError: Please specify target_modules ...` |
| V6 | `v_make_tiny.py`, `v_sft_tiny.py` ×4 | 새 tiny CondGen clone(linear 3 + full 1, vision depth 2, vocab 248320, BF16, top-level dtype 제거)으로 실제 `SFTTrainer`(GA 2, 2 step) 실행. 4개 경우(QLoRA·LoRA × TRL 기본·`dtype=bf16`) 모두 원 문서 표와 일치. 상세는 아래 표 1·8·9·12–15번 |
| V7 | `v_keymap.py`, `v_auto_causal.py` | V1 header의 key와 shape가 CondGen state_dict(760)와 일치하고, `^model\.language_model\.` 재매핑 후 CausalLM state_dict(427)와도 일치(missing/unexpected/shape 불일치 0). tiny를 실제 `AutoModelForCausalLM`으로 로드하면 `Qwen3_5ForCausalLM`, missing 0 / unexpected 0, `up_proj` 값이 checkpoint와 같다. dtype 없이 `from_pretrained`하면 bf16 |
| V8 | `v_load_transient.py` | `F.quantize_4bit`를 weakref로 계측: 41회 호출 모두 `MainThread`, 입력 dtype = load dtype(fp32/bf16), 이전 입력이 살아 있던 경우 0 |
| V9 | `v_devmap.py` | full-size 4-bit meta 모델로 `infer_auto_device_map`을 이분 탐색: 전부 GPU 0에 놓이는 최소 `max_memory[0]` = S + 0.0003 GiB(S = 11.032 / 7.232 GiB, fp32 / bf16 로드). 주의: 이 시뮬레이션은 모듈 교체를 dtype context 밖에서 해서 bf16 경우 Linear4bit bias 280,432개를 fp32로 셌다(+560,864 B). 실제 `from_pretrained`는 `preprocess_model`을 `local_torch_dtype(dtype)` 안에서 호출한다(`modeling_utils.py:4290-4302, 3740`). 결론 수치(8.93 GiB)에는 영향이 없다 |
| V10 | `v_every_n.py` | `gradient_checkpointing_enable(every_n_layers=3)`: CondGen vision `[T,F]`, decoder `[F,T,F,F]` / CausalLM decoder `[T,F,F,T]` |
| V11 | `v_bnb_optim.py` | CPU bnb optimizer state byte: AdamW8bit·PagedAdamW8bit n=65,536 → tensor별 133,120 B(2.03125 B/param) + optimizer 공유 qmap 2,048 B. n=512 → fp32 8 B/param. AdamW32bit + bf16 param → 8 B/param. CPU에서는 paged가 꺼지므로 paging 자체는 검증하지 못했다 |
| V12 | `v_dpo_ref.py` | 4-bit tiny + 사용자 `PeftModel` → `DPOTrainer`: `default` bf16 trainable 8,448, `ref` **fp32 frozen** 8,448 |
| V13 | `v_saved.py` | `saved_tensors_hooks`로 기록, CPU bf16 autocast: fp32 adapter면 `lora_A`가 저장하는 활성값이 x의 새 bf16 사본(`is_x=False`)이고, bf16 adapter면 x 자체(`is_x=True`)다. r 차원 중간값은 두 경우 모두 저장된다 |

### 확인한 주장과 판정

| # | 주장 (문서 위치) | 판정 | 근거 (한 줄) |
|---|---|---|---|
| 1 | TRL `create_model_from_path`는 dtype 미지정 시 fp32로 로드하고, `from_pretrained` 단독은 `"auto"`(예시 BF16)다 (§1.1–1.2) | VERIFIED | `trl/trainer/utils.py:1292-1300`, `transformers/modeling_utils.py:4106-4107, 815-896`. V6: TRL 기본이면 frozen 비양자화 전부 fp32. V7: 무인자 `from_pretrained` → bf16 |
| 2 | 문자열 진입 시 `getattr(transformers, architectures[0])` → CondGen, vision 상주. `AutoConfig`에 revision을 넘기지 않는다 (§1.3) | VERIFIED | `utils.py:1308-1309`(`trust_remote_code`만 전달). V6: class CondGen, vision LoRA tensor 20개 존재. V1: 9,409,813,744 |
| 3 | `AutoModelForCausalLM` → `Qwen3_5ForCausalLM`, visual/mtp를 경고 없이 버리고 prefix를 재매핑한다. 333 tensors / 456,010,480 (§1.4) | VERIFIED | `modeling_auto.py:854`, `modeling_qwen3_5.py:1679`, `conversion_mapping.py:1054`, `auto_factory.py:395-410`. V1·V7 |
| 4 | 변환 규칙은 `type(module) is nn.Linear` 또는 Conv1D, 기본 skip은 `['lm_head']`, 사용자 skip 목록은 기본값을 대체한다. 358 = 248 + 110 (§2.1–2.4) | VERIFIED | `integrations/bitsandbytes.py:189`, `quantizer_bnb_4bit.py:129-131` + `quantizers/base.py:238-258`(`add_default_skips=False`). V3 |
| 5 | `BitsAndBytesConfig` 기본값은 fp4 / DQ 없음 / fp32 / uint8, blocksize 64 고정, nested 256 (§2.3, §3.1) | VERIFIED | `quantization_config.py:440-493`. `Linear4bit.__init__`에는 blocksize 인자가 아예 없다(`nn/modules.py:537-567`). `Params4bit` None→64(`:230-231`). `functional.py:940` |
| 6 | `B_q4_DQ`/`B_q4_noDQ` 공식과 예시 4개 값, block padding 없음 (§3.3–3.4) | VERIFIED | V2(16개 경우 byte 일치). CUDA op 할당 shape가 default op와 같다(`_ops.py:225-236`, `backends/cuda/ops.py:384-392`, nested `:299-311`) |
| 7 | 예시 합계 3,802,183,024 / 3,569,313,760 B, 비양자화 2,040,130,800, embed+lm_head 3.789 / 7.578 GiB, 반올림 +422,544 B (§3.5, §6.2) | VERIFIED | V3. §3.4 profile 표 7행도 이 값으로 다시 합산해 일치(step scalar를 512 B 블록으로 셈) |
| 8 | TRL SFT/DPO/GRPO는 `prepare_model_for_kbit_training`을 호출하지 않는다. 순서는 get_peft_model → (ZeRO-3 reentrant) → `enable_input_require_grads` → 양자화면 trainable 전부 bf16 (§6.1) | VERIFIED | 함수는 `trl/experimental/*`에만 있다(grep). `sft_trainer.py:1129-1167`, `dpo_trainer.py:640-712`, `grpo_trainer.py:440-516`. V6: QLoRA trainable 전부 bf16 |
| 9 | adapter는 fp32로 생성 → base dtype(Linear4bit는 `compute_dtype`) → `autocast_adapter_dtype`로 fp32. 비양자화 LoRA는 fp32, TRL QLoRA는 bf16, `modules_to_save` 복사본은 upcast하지 않는다 (§5.2) | VERIFIED | `tuners_utils.py:2149-2165, 2705-2763`. `ModulesToSaveWrapper`는 `BaseTunerLayer`가 아니다(`other.py:303, 598`). V6 `lora_bf16`: base bf16 + adapter·state fp32 |
| 10 | all-linear 358 / 51,265,024(vision 7,986,688), text-only 248 / 43,278,336. `qwen3_5`에서 target 미지정이면 `ValueError` (§5.3) | VERIFIED | V4(실제 PEFT 경로), V5. `peft/utils/constants.py`에 `qwen3_5` 없음 |
| 11 | `modules_to_save` = frozen 원본 + trainable deepcopy, 점 경계 없는 `endswith`, lm_head +7.578 GiB (§5.5) | VERIFIED (+보완) | `other.py:653, 661, 1088`. 1,017,118,720 × 2 B × 4 = 8,136,949,760 B. SFT `chat_template_path` 자동 추가 경로와 LoRA 제외 규칙(점 경계 정규식)과의 불일치를 본문에 추가 |
| 12 | GC는 TRL 기본 on, transformers 기본 `use_reentrant=False`, 단위는 `GradientCheckpointingLayer`, torch는 None을 True로 처리 (§7) | VERIFIED / CORRECTED(세부) | `base_config.py:61`, `modeling_utils.py:3138-3139`, `trainer.py:1494-1498`(`gc_kwargs or None`), `torch/utils/checkpoint.py:635-645`. V6 keywords `{'use_reentrant': False}`. **보정**: `every_n_layers` 선택은 vision block을 포함한 일련번호 기준(V10, §7.4) |
| 13 | 기본 optim `adamw_torch_fused`. state는 지연 생성되고 param dtype을 따르며 step은 fp32 scalar. bnb 8-bit 2.03125 / 8, 32-bit 8, paged는 ≥1e5, Embedding은 32-bit 강제 (§8.1–8.3) | VERIFIED | `training_args.py:803-810`, `trainer_optimizer.py:206-207`, `torch/optim/adam.py:150-189`, bnb `optimizer.py:374-392, 491-532`, `trainer.py:1315-1326`. V6, V11. paged의 managed memory 할당은 CPU에서 확인 불가(미확정 4 유지) |
| 14 | 실행되지 않은 trainable은 grad·state가 없다. GA microbatch 사이에 grad가 유지되고 `zero_grad` 뒤 None이 된다 (§8.4) | VERIFIED | V6: text state 62/62, vision 0/20, post-accumulate hook 4회(2 step × GA 2), substep 끝에 grad 62개 존재, 학습 후 전부 None. `trainer.py:1908`, `module.py:2957` |
| 15 | autocast 하에서 residual·checkpoint 경계 dtype = load dtype, Linear4bit 출력 = 입력 dtype (§9.3) | VERIFIED | V6: layer0 (in, out) = fp32/fp32(TRL 기본) vs bf16/bf16(`dtype=bf16`), `is_autocast_enabled('cpu')=True`. `bitsandbytes/nn/modules.py:637`. accelerate의 fp32 출력 변환 `accelerator.py:1824-1835` |
| 16 | on-the-fly 양자화는 동기·tensor 1개씩. warmup 요청량 11.03 / 7.23 GiB (§4.2, §4.4) | VERIFIED | `core_model_loading.py:1626-1636`. V8. warmup 공식 재계산 일치(`modeling_utils.py:4991-5106`, 상한 `:5096`) |
| 17 | Linear4bit forward는 M > 1536이면 전체 dequant, backward는 `quant_state.dtype`으로 dequant 후 grad dtype으로 cast. 100.7 / 302 MB (§9.1–9.2) | VERIFIED(소스) / CORRECTED(DQ 항) | `backends/cuda/ops.py:926, 954, 903-916`, `_functions.py:384`, CUDA `dequantize_4bit` 출력 `torch.empty(shape, dtype=dtype)`(`ops.py:432`). **보정**: forward의 DQ 임시는 8·nb다(`absmax_dq`와 그 합이 동시에 존재). CUDA 실행 자체는 INFERRED |
| 18 | `lora.Linear`는 항상 adapter dtype으로 입력을 cast하고, bnb wrapper는 base 출력을 clone한다. DoRA는 magnitude `(out,)`과 `eye(in)` 임시를 만든다 (§5.8–5.9) | VERIFIED | `lora/layer.py:1100`, `lora/bnb.py:533, 543`, `lora/dora.py:97`(eye의 dtype = adapter dtype, fp32 adapter면 302 MB가 아니라 604 MB). V13 |
| 19 | bnb 4-bit + `device_map="auto"` 적합 조건 = free × 0.90 (§3.5, §4.5) | **CORRECTED** | 단일 GPU이고 `max_memory`를 주지 않으면 `get_balanced_memory`의 0.9 × bnb 0.90 = **0.81**이다(`integrations/accelerate.py:283-292, 363, 374`). V9: 필요 free 13.62 GiB(fp32 로드) / 8.93 GiB(bf16 로드) |
| 20 | DPO/GRPO `"ref"` adapter는 fp32로 유지될 수 있다 [원 INFERRED] (§6.4, 미확정 8) | **CORRECTED** (INFERRED → VERIFIED, DPO) | V12: ref fp32 frozen, default bf16. GRPO는 같은 코드이지만 `beta` 기본 0.0이라 기본 설정에서는 생성하지 않는다 |
| 21 | allocator 상수(`kMinBlockSize=512` 등), `kLargeBuffer`가 헤더에 없음, `expandable_segments` 기본 false (§10.1) | VERIFIED (인용 보정) | `torch/include/c10/core/AllocatorConfig.h:16-25`. 기본 false는 `:353`에 있다(원 인용 `:209-215`는 접근자) |
| 22 | host·device dtype 변환 위치, CUDA context·workspace 크기, paged managed memory의 계측 방식 (§4.3, §10, 미확정 1·3·4) | UNVERIFIABLE | wheel에 C++ 소스가 없고 이 머신에는 CUDA가 없다. 원 문서의 INFERRED/UNKNOWN 태그를 유지했고, §3.5 상한에는 이 가정이 붙어 있다고 명시했다 |

### 본문 수정 요약

- §4.5, §3.5: bnb 4-bit 로딩 적합 계수를 0.90에서 **0.81**로 바꿨다(단일 GPU, `max_memory` 미지정). 사용자가 `max_memory`를 준 경우와 분산(`device_map=None`, 사전 검사 없음)을 구분했다. 예시 모델의 필요 free memory를 넣었다.
- §3.5: `peak_allocated_load` 상한이 "host에서 dtype 변환"이라는 INFERRED 가정 위에 있음을 명시했다.
- §9.1, §3.6: forward dequant의 DQ 임시를 4·nb에서 **8·nb**로 고쳤다. DQ를 포함한 byte 값을 추가했다. fp32 residual에서 `Linear4bit` 입출력 cast 사본이 생긴다는 점을 추가했다.
- §7.4, §3.1 YAML: `every_n_layers`의 선택 번호 규칙(CondGen `(27+j)%k`)과 mask 2종 공유를 보정했다.
- §6.4, 미확정 8: DPO ref adapter fp32를 VERIFIED로 올렸다.
- §5.8, §3.6: DoRA `eye(in)` 임시의 dtype이 adapter dtype이라는 점을 반영했다(fp32 adapter면 2배).
- §5.5: SFT `chat_template_path`의 `lm_head` 자동 `modules_to_save` 경로와 매칭 규칙 불일치를 추가했다.
- §10.1 인용 줄을 고쳤다. 미확정 9에 DDP `find_unused_parameters` 기본값을 보충했다.
- 원 문서의 나머지 수치(§3.4 profile 표, §6.2 표, §4.3 transient, LoRA 파라미터 수)는 다시 계산해 바꿀 것이 없었다.
