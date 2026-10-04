# TRL 1.14.1 SFTTrainer·DPOTrainer 조사 — 전처리, 절단, collation, loss/메모리 경로

| 항목 | 값 |
|---|---|
| 주제 | TRL 1.14.1 `SFTTrainer`/`DPOTrainer`의 전처리, truncation, collation, loss·logits 메모리 경로, 데이터 coverage |
| 문서 | `docs/research/trl-sft-dpo.md` |
| 작성일 | 2026-10-04 |
| 작성 | `research-trl-sft-dpo` (Milestone M0) |
| 검증 | `verify-trl-sft-dpo` (2026-10-04). 핵심 주장 20개를 설치된 소스와 독립 실험으로 다시 확인했다. 틀렸거나 과장된 부분은 본문에서 고쳤고, 판정과 근거는 문서 끝 "검증 로그"에 있다 |
| 대상 환경 (pinned) | torch==2.14.1, transformers==5.18.0, trl==1.14.1, peft==0.21.2, bitsandbytes==0.50.2, accelerate==1.15.0, datasets==5.0.1, huggingface_hub==1.33.0, tokenizers==0.23.2, safetensors==0.8.0 |
| 예시 모델 | `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` @ `2367e865d009c13ac81713a2878291d33ab28177` (config·tokenizer·processor 파일만 사용, 가중치 미다운로드) |
| 예시 데이터 | `CyberNative/Code_Vulnerability_Security_DPO` @ `81aeacf06cf43b16d7278a3a01f019a496a53c51` (`secure_programming_dpo.json`, JSON Lines 4,656 rows) |
| 방법 | ① 설치된 site-packages 소스 정독 ② macOS arm64 CPU 실험: 실제 MiMo tokenizer/processor와 로컬에서 만든 tiny random-init `Qwen3_5ForConditionalGeneration`(vocab 248,320 유지)으로 **실제** `SFTTrainer`/`DPOTrainer`를 생성, 전체 4,656 row 전처리 결과를 독립 재구현과 대조, 1 optimizer step smoke ③ CUDA 전용 경로(Triton fused kernel, bitsandbytes, FlashAttention)는 소스 기반 추론 |
| 실험 환경 | 공용 `/tmp/vf-research/.venv`(변경 없음) + 오버레이 venv `/tmp/vf-research/venv-trl-sft-dpo` (`.pth`로 공용 site-packages를 참조, 추가 설치: Pillow 12.3.0, torchvision 0.29.1 `--no-deps`, pytest 9.1.1) + torch 없는 venv `/tmp/vf-research/venv-trl-sft-dpo-notorch` (transformers 5.18.0, tokenizers 0.23.2, huggingface_hub 1.33.0, jinja2, pytest) |
| 실험 코드 | `/tmp/vf-research/scratch/trl-sft-dpo/` (`exp01`–`exp13`, `build_tiny.py`, `parity/test_trl_parity_doc.py`). 저장소에는 넣지 않음. 결과는 부록 A |
| 증거 태그 | **VERIFIED** = 소스에서 확인했거나 실측함 / **INFERRED** = 소스 기반 추론(CUDA 전용 등, 미실행) / **UNKNOWN** = 확인 불가 |

인용 형식은 `<pkg>==<ver> <site-packages 기준 경로>:<줄 범위> (<함수/클래스>)`이다. 같은 패키지를 연속 인용할 때는 버전을 생략한 곳이 있다(버전은 항상 위 pinned 값).

---

## 0. 핵심 요약

1. **두 Trainer 모두 `max_length` 기본값이 1024**이다. 기본값 그대로 예시 데이터를 넣으면 SFT(chosen) 17 rows, DPO 28 pairs가 잘린다. `max_length=None`은 두 Trainer 모두 end-to-end로 동작한다(전체 4,656 rows 입력=출력, 1 step 학습에서 T>1024 배치 확인). VERIFIED
2. 절단이 켜져 있으면 **경고 없이 row가 사라진다.** SFT는 절단 후 label이 모두 `-100`인 row를 `filter`로 지우고, DPO는 `len(prompt_ids) >= max_length`인 pair를 지운다. VERIFIED (§1.2, §7.4)
3. SFT의 기본 `loss_type`은 **`"chunked_nll"`**이다. lm_head를 `labels != -100` 위치에만, 256 token chunk 단위로 계산하므로 `(B,T,V)` logits가 생기지 않는다(`outputs.logits is None` 실측). DPO(비 Liger)는 policy와 reference 각각 **전 위치 `(2B,T,V)` logits**를 만들고, bf16 mixed precision에서는 accelerate가 이를 **fp32로 변환한 텐서**를 받는다. CPU 실측 VERIFIED, CUDA 수명·피크 INFERRED (§5, §8)
4. 모델을 문자열로 넘기면 TRL은 `dtype`이 없을 때 **float32**로 로드하고 GPU에서는 `device_map="auto"`를 쓴다. TRL CLI의 `ModelConfig.dtype` 기본값도 `"float32"`다. VERIFIED (§1.7)
5. 예시 모델(VLM, `Qwen3_5ForConditionalGeneration`)에서 `processing_class`를 주지 않으면 `AutoProcessor` → `Qwen3VLProcessor`가 로드되고 `_is_vlm=True`가 된다. 이미지 키가 없으므로 데이터는 텍스트 경로로 흐르며, **processor 경로와 tokenizer 경로의 token ids는 4,656 rows 전부 동일**하다. 단, processor 로드에는 Pillow와 torchvision이 필요하다(없으면 Trainer 생성 단계에서 실패). VERIFIED (§3)
6. DPO는 prompt, prompt+chosen, prompt+rejected를 **각각 전체 대화로 chat template 렌더링·토큰화**한 뒤 prompt 길이로 자른다. collator는 chosen B행 뒤에 rejected B행을 붙인 **2B행**을 전체 최대 길이로 right-pad한다: `T_dpo = 2B × round_up(max_i max(L_chosen_i, L_rejected_i), m)`. VERIFIED (§7, §8)
7. `DPOConfig`에 `max_prompt_length`, `max_completion_length`, `use_logits_to_keep`, `model_adapter_name`, `ref_adapter_name`, `force_use_ref_model`, `label_pad_token_id`, `padding_value`, `rpo_alpha`, `reference_free`는 **없다**(생성 시 `TypeError`). `padding_free=True`는 경고 후 `False`로 바뀐다. VERIFIED (§6)
8. 원본 CyberNative 컬럼을 매핑 없이 넣으면 DPO는 `extract_prompt`가 chosen/rejected **코드의 공통 접두사**를 prompt로 잡고 `question`을 버린 채 오류 없이 진행한다. SFT는 `KeyError: 'text'`. 내보내기는 매핑된 데이터를 전제로 해야 한다. VERIFIED (§7.1)
9. torch 없는 환경에서 `AutoTokenizer` + `apply_chat_template`만으로 만든 재구현이 TRL 결과와 같은 golden 값을 재현한다(DPO 최대 branch 2,272 tokens, Σ(prompt+chosen)=1,007,173, assistant-only loss token 20–2,007). VERIFIED (§11, 부록 A)
10. Parity 테스트 레시피: 바이트 단위 tiny tokenizer + MiMo chat template fixture + tiny `Qwen3_5ForCausalLM`으로 실제 Trainer를 CPU·오프라인에서 띄워 ids·labels·collator shape·1 step 배치 shape를 비교한다. 6 tests가 약 5.6초에 통과하고, parity 의존성이 없으면 깨끗하게 skip된다. VERIFIED (§11)

---

## 1. SFTConfig 기본값과 절단·packing·loss 옵션

### 1.1 기본값 표

아래 "기본값"은 `SFTConfig(output_dir=..., report_to="none")`를 실제로 생성해 `__post_init__` 이후 값을 읽은 결과다. VERIFIED (exp: 설정 덤프, 부록 A-1)

| 필드 | 기본값 (해석 후) | 근거 | 메모 |
|---|---|---|---|
| `max_length` | `1024` | trl==1.14.1 trl/trainer/sft_config.py:203-210 | `None` 허용, `None`이면 절단 없음 |
| `truncation_mode` | `"keep_start"` | sft_config.py:211-218, 316-322 | `"keep_end"`는 deprecated 경고 |
| `packing` | `False` | sft_config.py:223-229 | |
| `packing_strategy` | `"bfd"` | sft_config.py:230-238, 323-330 | choices `bfd`/`bfd_split`/`wrapped`. `"bfd-requeue"`는 `"bfd_split"`으로 rename |
| `padding_free` | `False` | sft_config.py:239-248 | 실효값 = `padding_free or (packing and strategy in {bfd, bfd_split})` (sft_trainer.py:1180) |
| `pad_to_multiple_of` | `None` | sft_config.py:249-252 | |
| `eval_packing` | `None` (= `packing`) | sft_config.py:253-256 | |
| `completion_only_loss` | `None` → prompt-completion 데이터면 `True` | sft_config.py:259-270; sft_trainer.py:1213-1216 | |
| `assistant_only_loss` | `False` | sft_config.py:271-280 | conversational 전용 |
| `dataset_text_field` | `"text"` | sft_config.py:181-184 | |
| `eos_token` | `None` → `processing_class.eos_token` | sft_config.py:197-202; sft_trainer.py:1018-1036 | 지정하면 tokenizer·model config·generation config까지 바뀜 |
| `loss_type` | `None` → **`"chunked_nll"`** (`use_liger_kernel=True`면 `"nll"`) | sft_config.py:281-293, 332-334 | 값: `nll`, `dft`, `chunked_nll` |
| `activation_offloading` | `False` | sft_config.py:294-297 | |
| `use_liger_kernel` | `False` | trl/trainer/base_config.py:90-95 | |
| `gradient_checkpointing` | **`True`** | base_config.py:61-66 | TrainingArguments 기본(False)과 다름 |
| `bf16` | `None` → `not fp16` → **`True`** | base_config.py:67-74, 104-105 | |
| `model_init_kwargs` | `None` → `dtype="float32"`, `device_map="auto"`(GPU) | trl/trainer/utils.py:1292-1305 (`create_model_from_path`) | §1.7 |
| `learning_rate` | `2e-5` | sft_config.py:141-144 | |
| `shuffle_dataset` | `False` | sft_config.py:219-222 | |
| `dataset_kwargs` | `None` | sft_config.py:185-192 | 유일한 key: `skip_prepare_dataset` |
| `per_device_train_batch_size` | `8` | transformers TrainingArguments | 계획(microbatch 1)과 다르므로 명시 필요 |
| `optim` | `"adamw_torch_fused"` | transformers==5.18.0 transformers/training_args.py:804-811 | torch>=2.8이면 fused AdamW |
| `dataloader_drop_last` | `False` | TrainingArguments | §9 |
| `remove_unused_columns` | `True` | TrainingArguments | SFT signature: `input_ids`, `labels`, `seq_lengths` (sft_trainer.py:1712-1721) |
| `train_sampling_strategy` | `"random"` | training_args.py:1362-1367 | 값: `random`/`sequential`/`group_by_length`/`batch_rebalance` (어느 버전에서 도입됐는지는 설치본만으로 확인할 수 없음, UNKNOWN) |

### 1.2 `max_length`: 절단 위치와 `None` 지원

텍스트 데이터에서 절단은 **`_prepare_dataset` 한 곳**에서만 일어난다. 기본 collator(`DataCollatorForLanguageModeling`)는 자르지 않는다. 절단 직후 label이 모두 `-100`이 된 row는 **삭제**된다. VERIFIED

```python
# trl==1.14.1 trl/trainer/sft_trainer.py:1659-1683 (SFTTrainer._prepare_dataset), 발췌
if args.max_length is not None and not packing:
    if args.truncation_mode == "keep_start":
        sl = slice(None, args.max_length)
    ...
    def truncate(example, sl):
        return {"input_ids": example["input_ids"][sl], "labels": example["labels"][sl]}
    dataset = dataset.map(truncate, fn_kwargs={"sl": sl}, **map_kwargs)
    dataset = dataset.filter(
        lambda example: any(label != -100 for label in example["labels"]), **map_kwargs
    )
```

- `max_length=None`이면 위 블록 전체를 건너뛴다. collator에도 절단이 없으므로 **end-to-end 무절단**이다. VERIFIED: 전체 4,656 rows in=out(exp04), tiny 모델 1 step 학습에서 `(3, 1540)` 배치가 그대로 모델에 들어감(§11).
- 기본값(1024)에서 assistant-only loss와 긴 prompt가 겹치면 row가 사라진다. 실측: 1,500단어 user + 짧은 답 row가 `assistant_only_loss=True, max_length=1024`에서 2→1 rows, `max_length=None`에서 유지(1,513 tokens). VERIFIED (exp05)
- 이미 토큰화된 데이터(`input_ids` 컬럼 존재)도 `max_length`가 정수면 같은 절단·삭제를 거친다. `dataset_kwargs={"skip_prepare_dataset": True}`면 준비 단계 전체(절단 포함)를 건너뛴다. VERIFIED (exp08: 기본값 → 1 row·1,024 tokens, `None` 또는 skip → 2 rows·1,200/10 tokens)
- `max_length=None`을 **거부하는 검증은 packing뿐**이다: `packing=True`이고 `max_length is None`이면 `ValueError("When packing is enabled, \`max_length\` can't be \`None\`.")` (sft_trainer.py:1686-1688). 반대로 `padding_free=True`이고 packing이 꺼져 있으면 **`max_length`가 반드시 `None`이어야** 한다 (sft_trainer.py:1284-1289). VERIFIED (exp05)
- 비전 데이터 경로에서는 collator가 `truncation=self.max_length is not None`으로 processor를 호출하므로 절단 위치가 collator로 옮겨 간다 (sft_trainer.py:676-686). VERIFIED(소스). 텍스트 전용 데이터에는 해당하지 않는다.

### 1.3 packing 전략

| `packing_strategy` | 긴 샘플(> `max_length`) | 토큰 보존 | 샘플 경계 | 실효 padding-free |
|---|---|---|---|---|
| `bfd` (기본) | 초과분 **절단·폐기** | 아니오 | `seq_lengths`로 보존 | 예 |
| `bfd_split` | 초과분을 **다른 bin으로 분할** | 예 | 원래 문맥이 **분할**됨 | 예 |
| `wrapped` | 경계를 무시하고 **중간 절단** | 예 | 없음(`seq_lengths` 없음) | 아니오 |

근거: trl==1.14.1 trl/data_utils.py:739-826 (`_pack_bfd`), 829-843 (`_pack_wrapped`), 846-933 (`pack_dataset`). 실측(exp05, `max_length=32`, 토큰 길이 32/7/14/52의 4 rows, 합 105): `bfd` → 3 rows·85 tokens(52→32 절단), `bfd_split` → 4 rows·105 tokens(`seq_lengths=[[32],[32],[20,7],[14]]`), `wrapped` → 4 rows·105 tokens·`seq_lengths` 없음. VERIFIED

- `_pack_bfd`는 길이 0인 시퀀스를 **조용히 제거**한다 (data_utils.py:749-752). `bfd`와 `bfd_split`이 같은 함수를 쓰므로 두 전략 모두 해당한다. VERIFIED(소스)
- packing은 `dataset.map(..., batched=True)` 안에서 수행되므로 bin은 map 배치(datasets 기본 1,000 rows) 안에서만 만들어진다. VERIFIED: `pack_dataset`이 `batch_size`를 넘기지 않는다(trl/data_utils.py:910-925). datasets==5.0.1 datasets/arrow_dataset.py:3221 (`Dataset.map`)의 기본값은 `batch_size=1000`이다. 실험(검증 로그 V-15): 1-token row 1,500개를 `seq_length=4000`으로 packing하면 세 전략 모두 2 rows(`[1000, 500]`)가 나온다.
- bfd 계열 packing에서 attention 구현이 FlashAttention 계열이 아니면 "cross-contamination" 경고만 내고 진행한다 (sft_trainer.py:1248-1256). Qwen3.5 hybrid의 linear-attention 층이 packed `position_ids`/`seq_lengths` 경계를 지키는지는 UNKNOWN.
- 결론: **무절단·무분할·무연결**을 동시에 만족하는 packing 설정은 없다. `bfd` + `max_length ≥ 전체 최대 길이`는 절단·분할은 없지만 서로 다른 샘플을 한 행에 잇는다(plan §7.4 엄격 모드 위반).

### 1.4 `padding_free`

- 실효값: `args.padding_free or (args.packing and args.packing_strategy in {"bfd", "bfd_split"})` (sft_trainer.py:1180). VERIFIED
- 켜지면 custom collator 금지, FlashAttention 계열이 아니면 경고, batch size 1이면 경고 (sft_trainer.py:1185-1209). VERIFIED(소스)
- collator는 배치 전체를 한 행으로 이어 붙이고 `position_ids`를 시퀀스마다 0부터 다시 매긴다. `attention_mask`는 만들지 않는다. VERIFIED (§4)

### 1.5 loss 마스크·텍스트 필드·EOS

- `completion_only_loss=None`이면 첫 example에 `prompt`와 `completion`이 모두 있을 때만 `True`다 (sft_trainer.py:1213-1216). VERIFIED (exp04: prompt-completion → `True`, messages → `False`)
- `assistant_only_loss=True`는 conversational 데이터만 허용한다 (sft_trainer.py:1257-1261). 템플릿에 `{% generation %}` 마커가 없으면 TRL 동봉 training template로 바꾸는데, **문자열이 정확히 일치하는 알려진 템플릿**일 때만 가능하고 아니면 `ValueError` (sft_trainer.py:1263-1268; trl/chat_template_utils.py:1032-1199). VERIFIED (exp10)
- EOS: non-conversational 데이터에만 붙인다. `text`(또는 `completion`) 문자열이 `eos_token`으로 끝나지 않으면 문자열로 덧붙인다 (sft_trainer.py:1518-1536). Conversational 데이터는 템플릿이 만든 종료 토큰을 그대로 쓴다. VERIFIED (exp05: text 끝 `<|im_end|>` 추가, exp04: messages 마지막 label = 248046 `<|im_end|>`)

### 1.6 `loss_type`, Liger, chunked/fused loss와 호환 제약

| 경로 | 선택 조건 | (B,T,V) logits | 호환 제약 (모두 VERIFIED) |
|---|---|---|---|
| `chunked_nll` (기본) | `loss_type` 미지정·`use_liger_kernel=False` | 만들지 않음 (`outputs.logits=None`) | `use_liger_kernel=True`와 동시 지정 시 `ValueError` (sft_trainer.py:1365-1366). PEFT가 `lm_head`를 감싸면 `ValueError` (sft_trainer.py:1349-1358, exp13). `target_modules="all-linear"`는 lm_head를 제외하므로 통과(exp13). VLM은 top-level `lm_head`가 있으면 지원 (sft_config.py:286-291) — `Qwen3_5ForConditionalGeneration`에서 1 step 학습 성공(exp06) |
| `nll` | 명시 지정 또는 Liger | 만든다 | 일반 transformers loss 경로 |
| `dft` | 명시 지정 | 만든다 | `compute_loss_func`와 동시 지정 불가 (sft_trainer.py:1330-1337) |
| Liger | `use_liger_kernel=True` | Liger 구현에 따름 | pinned 환경에 liger-kernel이 없어 `train()`에서 `ImportError` (transformers==5.18.0 transformers/integrations/liger.py:39-43; exp13). TRL 최소 버전 0.8.2 (trl/import_utils.py:24) |

- chunk 크기는 상수 `_CHUNKED_LM_HEAD_CHUNK_SIZE = 256`이며 설정 필드가 없다 (sft_trainer.py:87). VERIFIED
- 패치는 `super().__init__` 전에 `model.forward`를 교체한다. PEFT면 `get_base_model()`의 forward를 바꾼다 (sft_trainer.py:1338-1359). VERIFIED(소스)

### 1.7 gradient checkpointing, bf16, dtype, activation offloading

- `create_model_from_path`는 `dtype` 미지정 시 `"float32"`, `device_map` 미지정 시 CPU면 `None`, 아니면 `"auto"`를 쓴다. 클래스는 `config.architectures[0]`이다. VERIFIED. 실측(V-07, V-12): bf16으로 저장한 tiny 체크포인트를 `model_init_kwargs` 없이 넘기면 policy와 자동 생성된 reference가 모두 `float32`로 로드된다. 분산 실행(`MULTI_GPU`/`DEEPSPEED`)에서는 Trainer가 `device_map=None`으로 덮어쓴다 (sft_trainer.py:981-983; dpo_trainer.py:562-564, 922-924). VERIFIED(소스)

```python
# trl==1.14.1 trl/trainer/utils.py:1292-1309 (create_model_from_path), 발췌
dtype = kwargs.get("dtype", "float32")
...
if "device_map" not in kwargs:
    kwargs["device_map"] = None if PartialState().device.type == "cpu" else "auto"
if architecture is None:
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=kwargs.get("trust_remote_code", False))
    architecture = getattr(transformers, config.architectures[0], None)
```

- TRL CLI의 `ModelConfig.dtype` 기본값도 `"float32"`이고 (trl/trainer/model_config.py:89-95), QLoRA의 `bnb_4bit_compute_dtype`은 이 `dtype`을 그대로 쓴다 (trl/trainer/utils.py:252-268). VERIFIED(소스)
- 양자화 모델이면 학습 가능한 파라미터(LoRA 등)를 bf16으로 바꾼다 (sft_trainer.py:1160-1167; dpo_trainer.py:705-708). VERIFIED(소스)
- (검증 시 추가) **양자화하지 않은 모델 + LoRA**에서는 TRL이 adapter dtype을 바꾸지 않는다. 따라서 PEFT 기본값(`autocast_adapter_dtype=True`)대로 adapter 파라미터는 **fp32**다. 실측(V-17): bf16 base에서 trainable 파라미터는 `float32`, frozen 파라미터는 `bfloat16`이었다. 예외로 DeepSpeed ZeRO-3 + 비양자화이면 TRL이 `autocast_adapter_dtype=False`를 넘겨 adapter가 base dtype을 따른다 (sft_trainer.py:1131-1134; dpo_trainer.py:642-645). VERIFIED(실측 + 소스)
- gradient checkpointing은 `Trainer.train()`에서 켠다. transformers 5.18 기본은 `use_reentrant=False`, `every_n_layers=1`, `offload=False`이다. `every_n_layers`와 `offload` 두 key는 `gradient_checkpointing_kwargs`에 넣어 전달한다(도입 버전은 확인하지 않음) (transformers/trainer.py:1490-1501; transformers/modeling_utils.py:3113-3179). VERIFIED(소스 + 실측 V-16: `__init__` 직후 `is_gradient_checkpointing=False`, `train()` 후 `True`, checkpoint 함수의 kwargs는 `{'use_reentrant': False}`. checkpoint 대상은 `Qwen3_5DecoderLayer`와 비전 타워의 `Qwen3_5VisionBlock` 모두다). PEFT + checkpointing이면 TRL이 `enable_input_require_grads()`를 호출한다 (sft_trainer.py:1155-1158). PEFT + DeepSpeed ZeRO-3 + checkpointing에서는 TRL이 `use_reentrant=True`로 강제한다 (sft_trainer.py:1136-1153; dpo_trainer.py:677-694). VERIFIED(소스)
- `activation_offloading=True`면 `training_step`을 `OffloadActivations`(saved-tensor hook)로 감싼다. 기본 인자: `use_pin_memory=True`, `use_streams=True`, `min_offload_size=1024` bytes, `max_fwd_stash_size=5`. `lm_head` forward 동안은 offload하지 않는다 (trl/models/activation_offloading.py:662-742; sft_trainer.py:1421-1425, 1924-1927). VERIFIED(소스). chunked_nll은 `lm_head` 모듈을 호출하지 않으므로 이 예외가 적용되지 않는다. INFERRED

---

## 2. SFT 데이터셋 준비 파이프라인

### 2.1 단계 순서

trl==1.14.1 trl/trainer/sft_trainer.py:1460-1710 (`SFTTrainer._prepare_dataset`). VERIFIED(소스) + 실측(exp04, exp05, exp08)

1. `Dataset.with_transform()` 형식이면 `ValueError` (1469-1476).
2. `input_ids` 컬럼이 있으면 **이미 처리됨**으로 보고 3–5단계를 건너뛴다 (1479-1480).
3. `formatting_func` → `{"text": ...}` (1496-1503). `from`/`value` 대화 → ChatML (1505-1516).
4. EOS 추가: non-conversational만 (1518-1536).
5. 토큰화 `tokenize_fn` (1542-1625): 형식별 규칙은 §2.2.
6. `labels` 생성 (1627-1654): `labels` 컬럼이 이미 있으면 그대로 둔다. 없으면 `assistant_masks`(있으면 항상)와 `completion_mask`(`completion_only_loss`일 때만)를 **AND**해서 0인 위치를 `-100`으로 만들고 mask 컬럼을 지운다.
7. 절단과 fully-masked row 삭제 (1656-1683, §1.2).
8. packing (1685-1700): `max_length` 필수, `input_ids`/`labels`만 남기고 `pack_dataset`.
9. Liger면 컬럼 정리 (1701-1705), `shuffle_dataset`이면 shuffle (1707-1708).

### 2.2 형식별 토큰화 규칙

| 형식 | 토큰화 | mask |
|---|---|---|
| LM text (`dataset_text_field`) | `processing_class(text=text+eos)` — `add_special_tokens` 기본(True) | 모든 위치 loss |
| LM conversational (`messages`) | `apply_chat_template(messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=aol, tools=..., **chat_template_kwargs)` — **대화 전체를 한 번** 렌더링 | `aol`이면 `assistant_masks` |
| prompt-completion (string) | `prompt_ids = processing_class(text=prompt)`, 전체 = `processing_class(text=prompt+completion+eos)` | `completion_mask = [0]*len(prompt_ids) + [1]*(나머지)` |
| prompt-completion (conversational) | `prompt_ids = apply_chat_template(prompt, add_generation_prompt=True)`, 전체 = `apply_chat_template(prompt+completion, return_assistant_tokens_mask=aol)` | `completion_mask` (+ `aol`이면 AND) |

근거: sft_trainer.py:1542-1614, trl/data_utils.py:401-437 (`_tokenize`). VERIFIED

- prompt가 전체의 접두사가 아니면 **경고만** 하고 `len(prompt_ids)` 위치로 mask를 만든다 (sft_trainer.py:1578-1587). 예시 데이터에서는 위반 0건. VERIFIED
- `aol=True`일 때 **한 row라도** assistant token이 0개면 `RuntimeError`가 난다. 그 row만 지우는 것이 아니라 전처리 전체가 실패한다 (sft_trainer.py:1607-1613, example 단위 검사). VERIFIED(소스)
- row별 `tools`(JSON 문자열 허용)와 `chat_template_kwargs`가 템플릿에 전달된다 (sft_trainer.py:1543-1549). VERIFIED(소스)

### 2.3 chat template 적용과 special token (BOS 중복 위험)

- tokenizer 경로: `apply_chat_template(tokenize=True)`는 문자열을 렌더링한 뒤 **`add_special_tokens=False`**, `truncation=False`로 토큰화한다. 템플릿이 넣은 BOS 외에 BOS가 추가되지 않는다. VERIFIED: transformers==5.18.0 transformers/tokenization_utils_base.py:3123-3132 (`PreTrainedTokenizerBase.apply_chat_template`)
- processor 경로: 렌더링 결과가 `bos_token`으로 시작할 때만 `add_special_tokens=False`를 넘기고, 아니면 tokenizer 기본값(True)을 쓴다. VERIFIED(소스)

```python
# transformers==5.18.0 transformers/processing_utils.py:2225-2227 (ProcessorMixin.apply_chat_template)
single_prompt = prompt[0] if is_batched else prompt
if self.tokenizer.bos_token is not None and single_prompt.startswith(self.tokenizer.bos_token):
    processor_kwargs["add_special_tokens"] = False
```

- 예시 tokenizer는 `bos_token=None`이고 `add_special_tokens=True/False`의 결과가 같다(`[14556, 1814]` 동일). 따라서 두 경로가 같은 ids를 낸다. VERIFIED (exp01, exp02, exp04). post-processor로 BOS를 붙이면서 템플릿이 BOS를 렌더링하지 않는 모델에서는 processor 경로에만 BOS가 붙을 수 있다. INFERRED
- non-conversational 문자열은 `processing_class(text=...)`(add_special_tokens=True)로 토큰화하므로, 문자열 안에 이미 BOS 텍스트가 들어 있으면 BOS가 두 번 들어갈 수 있다. INFERRED

### 2.4 `assistant_masks`와 `{% generation %}`

- transformers의 Jinja 확장 `AssistantTracker`가 `{% generation %}` 블록의 문자 구간을 기록한다 (transformers/utils/chat_template_utils.py:437-477). tokenizer 경로는 `char_to_token(start)`–`char_to_token(end-1)` 구간을 1로 (tokenization_utils_base.py:3134-3155), processor 경로는 offset이 겹치는 token을 1로 만든다 (processing_utils.py:2259-2284). VERIFIED(소스). 예시 데이터에서는 두 방식의 mask가 4,656 rows 모두 같다. VERIFIED (exp04)
- **assistant-only loss에는 `{% generation %}` 마커가 필수**다. 없으면 TRL이 알려진 템플릿만 교체하고, 그 외는 `ValueError`. VERIFIED (exp10)
- TRL은 assistant 종료 토큰이 mask 안에 있는지 검사해 없으면 경고한다 (sft_trainer.py:1270-1281; chat_template_utils.py:886-952). 예시 템플릿은 통과(`True`). VERIFIED (exp10)

### 2.5 예시 모델 템플릿의 특이사항 (MiMo, sha256 `59a64ebb…ff63`)

- 템플릿에 이미 `{%- generation -%}` 마커가 있으며 **매크로 `render_assistant_message` 안**에 있다. 마커가 assistant 턴 전체(`<|im_start|>assistant\n<think>{reasoning}</think>{content}<|im_end|>`)를 감싼다. 그래서 `aol=True`면 **assistant 헤더(`<|im_start|>`, `assistant`, `\n`)와 `<think></think>`가 loss에 포함**된다. VERIFIED (exp04: 첫 loss token = `<|im_start|>assistant\n<think>`)
- prompt-completion 형식에서는 generation prompt가 `<|im_start|>assistant\n`(`enable_thinking`이 정의되지 않음)이므로 completion이 `<think></think>`부터 시작한다. 같은 대화에서 loss token 수가 messages+aol보다 정확히 3개 적다(최소 17 vs 20). VERIFIED (exp04)
- 턴 사이에 줄바꿈이 없고(`<|im_end|><|im_start|>`), 마지막 `<|im_end|>` 뒤에도 줄바꿈이 없다. VERIFIED (exp02: chosen tail `\n```<|im_end|>`)
- `content`가 문자열이든 `[{"type": "text", ...}]` 목록이든 같은 텍스트로 렌더링한다(`render_content`). processor 경로가 동일 ids를 내는 이유다. VERIFIED (exp02/exp04)
- 빈 `system`도 `<|im_start|>system\n<|im_end|>`로 렌더링되어 row마다 4 tokens가 늘어난다. 예시 데이터는 `system`이 4,656 rows 모두 빈 문자열이다. VERIFIED (exp09)

### 2.6 SFT에서 row가 줄거나 잘리는 지점 (전부)

| 지점 | 조건 | 동작 | 근거 |
|---|---|---|---|
| 절단 | `max_length` 정수 & packing 꺼짐 | 앞 `max_length` tokens만 유지 | sft_trainer.py:1659-1674 |
| fully-masked 삭제 | 위 절단과 같은 조건 | label 전부 -100인 row 삭제 (조용히) | sft_trainer.py:1676-1683 |
| packing `bfd` | packing 켜짐 | 초과분 폐기 + 빈 시퀀스 삭제 | data_utils.py:749-755 |
| packing `bfd_split` | packing 켜짐 | 초과분 분할 + 빈 시퀀스 삭제 | data_utils.py:749-752, 756-769 |
| packing `wrapped` | packing 켜짐 | 경계 무시 중간 절단 | data_utils.py:829-843 |
| aol 마스크 없음 | `aol=True` | 삭제가 아니라 `RuntimeError` (한 row라도 assistant token이 없으면) | sft_trainer.py:1607-1613 |

`max_length=None`, `packing=False`이면 이 중 어느 것도 일어나지 않는다. VERIFIED (exp04: 6가지 조합 모두 4,656→4,656)

---

## 3. VLM 체크포인트 + text-only 데이터의 `processing_class`

- 두 Trainer 모두 `processing_class=None`이면 `AutoProcessor.from_pretrained(model.config._name_or_path, revision=model_init_kwargs["revision"], trust_remote_code=...)`를 호출한다. 결과가 `ProcessorMixin`이면 `_is_vlm=True`, `_tokenizer = processor.tokenizer`. VERIFIED: trl==1.14.1 trl/trainer/sft_trainer.py:1002-1016; trl/trainer/dpo_trainer.py:589-603
- 예시 모델의 결과 (exp01, VERIFIED):

| 항목 | 값 |
|---|---|
| `AutoProcessor` | `Qwen3VLProcessor` (`ProcessorMixin`) → `_is_vlm=True` |
| tokenizer | `Qwen3_5Tokenizer`, backend `tokenizers` |
| BOS / EOS / PAD | `None` / `<\|im_end\|>`=248046 / `<\|endoftext\|>`=248044 |
| `padding_side` / `model_max_length` | `right` / 262,144 |
| `len(tokenizer)` vs config `vocab_size` | 248,077 vs 248,320 |
| `<think>`, `</think>`, `<\|im_start\|>` | added tokens 248068, 248069, 248045 |
| model config `text_config.eos_token_id` | 248044 (tokenizer EOS와 다름 → 학습 시 transformers가 248046으로 정렬한다는 로그 확인) |

- **의존성 함정**: Pillow·torchvision이 없는 공용 venv에서 `AutoProcessor`는 image processor(`Qwen2VLImageProcessor[Pil]`) 단계에서 `ValueError`, Pillow만 있으면 `Qwen3VLVideoProcessor` 단계에서 torchvision `ImportError`로 실패했다. 즉 TRL CLI처럼 `processing_class`를 주지 않는 경로는 학습 환경에 **Pillow와 torchvision이 있어야** Trainer가 생성된다. VERIFIED (exp01)
- 비전 경로 분기는 **첫 example**에 `image`/`images` key가 있는지로만 정한다 (sft_trainer.py:1050-1052; dpo_trainer.py:719-720). text-only면 일반 collator와 사전 전처리를 그대로 쓴다. 비전 경로가 켜지면 SFT는 packing·padding_free·aol을 거부하고 전처리를 collator로 미루며(sft_trainer.py:1059-1079, 1290-1296), DPO는 precompute를 거부한다(dpo_trainer.py:733-739). VERIFIED(소스)
- text-only + processor일 때 `_tokenize`는 대화를 `prepare_multimodal_messages`로 바꿔(`content` → `[{"type":"text","text":...}]`) processor의 `apply_chat_template`에 넘기고, 출력의 batch 차원을 벗긴다 (trl/data_utils.py:425-437, 33-124). 예시 템플릿에서는 결과 ids가 tokenizer 경로와 **완전히 같다**: DPO·SFT(4가지 형식/옵션) 모두 4,656 rows 불일치 0. VERIFIED (exp02, exp04)
- `processing_class=AutoTokenizer...`를 직접 넘기면 `_is_vlm=False`가 되고 같은 ids를 내며 Pillow/torchvision이 필요 없다. 단 TRL CLI 스크립트는 `processing_class`를 넘기지 않는다 (trl/scripts/sft.py:101-108, trl/scripts/dpo.py:99-106). VERIFIED
- 모델 클래스는 `config.architectures[0]` = `Qwen3_5ForConditionalGeneration`이다(ref model 타입으로 확인). 텍스트만 학습해도 **비전 타워가 생성·로드된다**. VERIFIED(클래스) / 비전 타워 가중치가 GPU에 상주한다는 메모리 영향은 INFERRED
- `Qwen3_5ForConditionalGeneration.accepts_loss_kwargs = False`라서 `Trainer.model_accepts_loss_kwargs=False`이고 `num_items_in_batch`가 모델에 전달되지 않는다 (transformers==5.18.0 transformers/models/qwen3_5/modeling_qwen3_5.py:1768; transformers/trainer.py:503-511). VERIFIED (exp04/exp06 출력)
- `peft==0.21.2`에서 이 VLM 클래스에 `target_modules="all-linear"`를 주면 **비전 타워 linear에도 LoRA가 붙는다**. 대상은 block마다 `attn.qkv`, `attn.proj`, `mlp.linear_fc1`, `mlp.linear_fc2`이고, `visual.merger.linear_fc1/fc2`도 포함된다. `lm_head`는 제외된다. VERIFIED (exp13 부속 실험, 재실험 V-08: depth 1인 tiny 모델에서 21개 중 6개가 `model.visual.*`였고 그중 2개가 merger). 실제 MiMo(depth 27, `deepstack_visual_indexes=[]`)에서는 27×4+2 = 110개 비전 모듈이 대상이 될 것으로 본다. 이 숫자는 모듈 구조에서 셈한 값이다(INFERRED). 상세는 PEFT 조사 범위다.

---

## 4. SFT collator와 토큰 슬롯 공식

`DataCollatorForLanguageModeling` (trl==1.14.1 trl/trainer/sft_trainer.py:402-525). VERIFIED(소스 + exp collator 실험)

- `input_ids`는 `pad_token_id`로, `labels`는 `-100`으로, `attention_mask`는 0으로 **오른쪽** padding한다(`pad(..., padding_side="right")`). `pad_to_multiple_of`는 sequence 차원을 올림한다 (trl/trainer/utils.py:157-164).
- 데이터셋에 `labels`가 이미 있으므로 collator는 label을 새로 만들지 않는다. 텍스트 경로 collator에는 **절단이 없다**.
- pad token = `args.pad_token or tokenizer.pad_token or tokenizer.eos_token` (sft_trainer.py:1221). 예시 모델은 `<|endoftext|>`(248044).
- padding-free: 모든 example을 한 행으로 이어 붙이고 `position_ids`를 example마다 0부터(packed면 `seq_lengths`로) 다시 매기며, `position_ids == 0`인 위치(각 시퀀스 첫 token과 pad)의 label을 `-100`으로 만든다. `attention_mask`는 없다 (sft_trainer.py:485-516).

| 모드 | 출력 shape | 실측 (길이 100, 300) |
|---|---|---|
| 기본 | `(B, round_up(max_i L_i, m))` — `input_ids`, `labels`, `attention_mask` | m=None → `(2, 300)`, m=64 → `(2, 320)` |
| padding-free | `(1, round_up(Σ_i L_i, m))` — `input_ids`, `labels`, `position_ids` | m=None → `(1, 400)`, m=64 → `(1, 448)` |

```text
T_sft_slots      = B × round_up(max_i L_i, m)        # m = pad_to_multiple_of (None이면 1)
T_sft_slots(pf)  = round_up(Σ_i L_i, m)               # padding_free (1행)
L_i              = len(input_ids_i)  (§2.2 규칙으로 계산한 전체 길이, EOS/템플릿 토큰 포함)
```

plan §19.1의 "B=2, 길이 100/300, pad_multiple=1 → 600 slots"와 일치한다.

---

## 5. SFT `compute_loss`의 부가 계산과 메모리

trl==1.14.1 trl/trainer/sft_trainer.py:1785-1917 (`SFTTrainer.compute_loss`). VERIFIED(소스) + exp06(CPU)

- 항상 `inputs["use_cache"]=False` (1795). MoE면 `output_router_logits=True` (1799-1800).
- 로깅 지표: `entropy`, `mean_token_accuracy`, `num_tokens`(attention_mask 합 또는 padding-free면 `position_ids` 길이), MoE면 `aux_loss`.

### 5.1 `chunked_nll` (기본) — (B,T,V) 없음

```python
# trl==1.14.1 trl/trainer/sft_trainer.py:197-220 (_chunked_cross_entropy_loss), 발췌
order = valid.to(torch.int8).argsort(descending=True, stable=True)
hidden = hidden[order]
labels = labels[order]
n_padded = (n_valid_tensor / chunk_size).ceil().clamp(min=1).to(torch.int64) * chunk_size
for start in range(0, n_padded, chunk_size):
    h_chunk = hidden[start : start + chunk_size]
    lbl_chunk = labels[start : start + chunk_size]
    chunk_loss, chunk_correct, chunk_entropy = torch.utils.checkpoint.checkpoint(
        _chunk, h_chunk, lm_head_weight, lm_head_bias, lbl_chunk, logit_scale,
        final_logit_softcapping, use_reentrant=False,
    )
```

- 패치된 forward는 backbone(`self.base_model`, VLM이면 멀티모달 wrapper)만 돌려 `last_hidden_state`를 얻고 lm_head 모듈을 호출하지 않는다. 반환 `logits=None` (sft_trainer.py:275-381). VERIFIED (exp06: hidden `(2,58,64)` bf16, `logits=None`)
- **lm_head 투영은 shift 후 `labels != -100`인 위치에만** 256개씩 수행한다. 마지막 chunk는 -100으로 채워 버린다. 최소 1 chunk는 항상 돈다. entropy와 accuracy도 chunk 안에서 계산한다 (`_chunk`, sft_trainer.py:100-118). VERIFIED(소스)
- 따라서 padding·prompt(completion-only/assistant-only일 때)는 lm_head 메모리에 들어가지 않지만 **backbone activation에는 그대로 들어간다**(plan §8.1과 일치).
- 메모리 성격: `hidden[order]` 복사본 `B·(T−1)·H·bytes(hidden)`가 backward까지 남는다(INFERRED, 소스 sft_trainer.py:186-199). chunk 하나의 순간 텐서는 `256×V` 크기의 bf16 matmul 결과, fp32 logits, fp32 log-softmax, entropy용 fp32 임시값 정도다. 원저자는 이를 ≈ 256·V·18 bytes로 잡았다. **검증 실측(V-14e, CPU, storage 단위 `MemTracker`)**: 입력과 grad를 뺀 peak가 `256·V` 원소당 forward 15.0–15.5 B, forward+backward 16.0–16.5 B였다. V=248,320이면 0.92–0.98 GiB다. 따라서 18 B(≈1.07 GiB)는 약간 보수적인 상한으로 쓸 수 있다. 이 peak는 chunk 수(1–8)와 무관하게 일정했다(chunk마다 해제됨). 단 이는 `SFTTrainer`처럼 backward 전에 `outputs`(`entropy_sum`의 graph)를 버릴 때에 한한다. `entropy_sum`을 backward까지 살려 두면 peak가 chunk 하나마다 `8·256·V` bytes씩 늘어났다(V-14c, 격리 실험: 1/2/4/8 chunk에서 15.5/24/40/72 B). 원인은 재계산된 entropy 경로의 saved tensor가 소비되지 않고 남기 때문으로 보인다(INFERRED). CUDA 커널 workspace와 caching allocator 반올림은 GPU 보정이 필요하다(INFERRED).
- `_chunk`는 `w.to(h.dtype)`로 lm_head weight를 hidden dtype에 맞춘다. 모델을 bf16으로 로드하면 복사가 없다(hidden도 bf16, exp06·V-07). 모델을 fp32(TRL 기본)로 로드하면 autocast가 켜져 있어도 hidden이 **fp32**로 들어온다(V-07 실측: `chunked_ce.hidden torch.float32`, autocast True). 이때 bf16 autocast matmul을 위해 `V×H` weight의 bf16 사본이 생길 수 있다(예시 모델 ≈ 2.0 GB). INFERRED

### 5.2 `nll` / `dft` — (B,T,V) 여러 개

- 모델 내부 `ForCausalLMLoss`가 `logits.float()`로 올린 뒤 cross-entropy를 계산한다 (transformers==5.18.0 transformers/loss/loss_utils.py:49-71). VERIFIED(소스)
- accelerate bf16 mixed precision은 모델 출력을 fp32로 바꾼다(§8.4). CPU 실측: `outputs.logits`가 `float32 (2, 58, 248320)`. VERIFIED (exp06 `sft_nll`)
- TRL은 이어서 `no_grad`로 `entropy_from_logits(shift_logits)`와 `argmax`를 계산한다 (sft_trainer.py:1847-1882). CUDA에서는 Triton fused kernel(행별 출력만 할당), 그 외에는 128행 chunk (trl/trainer/utils.py:601-640). VERIFIED(소스) / CUDA 동작 INFERRED
- `logits_to_keep`은 SFT에서 쓰지 않는다. `Qwen3_5ForConditionalGeneration.forward`의 기본 `logits_to_keep=0`은 모든 위치다 (modeling_qwen3_5.py:1818, 1877-1879). VERIFIED

### 5.3 `num_items_in_batch`

- Trainer는 gradient accumulation 창의 배치를 **항상 미리 모두 가져온다**(`get_batch_samples`). `labels[..., 1:] != -100` 개수는 `model_accepts_loss_kwargs` 또는 `compute_loss_func`가 있을 때만 센다 (transformers/trainer.py:2236-2290). 미리 가져온 배치(정수 텐서, 크기는 작다)는 device에 올라가 있다. VERIFIED(소스)/INFERRED(상주 위치)
- 예시 모델은 `accepts_loss_kwargs=False`라 `num_items_in_batch=None`이다(exp06 로그). chunked loss는 로컬 평균을 내고 Trainer가 accumulation 횟수로 나눈다 (trainer.py:2064-2066; sft_trainer.py:225-231). 메모리와는 무관하다. VERIFIED
- (검증 시 추가) 같은 Qwen3.5라도 text 전용 `Qwen3_5ForCausalLM`은 `accepts_loss_kwargs`를 선언하지 않는다(modeling_qwen3_5.py에서 선언은 1313 `Qwen3_5Model`과 1768 `Qwen3_5ForConditionalGeneration`뿐). 그래서 Trainer가 forward의 `**kwargs` 존재로 판정하고, SFT에서 `model_accepts_loss_kwargs=True`가 되며 `num_items_in_batch`가 전달된다(V-21 실측: `tensor(22)`). §11 parity 테스트의 tiny 모델이 이 클래스이므로, loss 정규화는 실제 MiMo 경로와 다르다. DPOTrainer는 모델과 무관하게 `False`로 고정한다 (dpo_trainer.py:949-952). VERIFIED

---

## 6. DPOConfig 기본값과 제거된 필드

`DPOConfig(output_dir=..., report_to="none")` 실제 생성값. VERIFIED (부록 A-1)

| 필드 | 기본값 | 근거 | 메모 |
|---|---|---|---|
| `max_length` | `1024` | trl==1.14.1 trl/trainer/dpo_config.py:194-200 | prompt+completion **전체** 길이 상한. `None` 허용 |
| `truncation_mode` | `"keep_start"` | dpo_config.py:201-208, 368-374 | `keep_end` deprecated |
| `padding_free` | `False` | dpo_config.py:209-218 | `True`여도 경고 후 `False` (dpo_trainer.py:711-718, exp13) |
| `pad_to_multiple_of` | `None` | dpo_config.py:219-222 | |
| `precompute_ref_log_probs` | `False` | dpo_config.py:223-230 | |
| `precompute_ref_batch_size` | `None` → train은 `per_device_train_batch_size`, eval은 `per_device_eval_batch_size` | dpo_config.py:231-238; dpo_trainer.py:986-1005 | |
| `loss_type` | `["sigmoid"]` | dpo_config.py:241-249 | 문자열이면 list로 감쌈 |
| `beta` | `0.1` | dpo_config.py:293-300 | |
| `label_smoothing` / `use_weighting` / `ld_alpha` / `f_divergence_type` | `0.0` / `False` / `None` / `"reverse_kl"` | dpo_config.py:258-307 | |
| `sync_ref_model` / `ref_model_mixup_alpha` / `ref_model_sync_steps` | `False` / `0.6` / `512` | dpo_config.py:320-343 | PEFT·precompute와 비호환 |
| `disable_dropout` | `True` | dpo_config.py:184-187 | policy·ref 모두 |
| `gradient_checkpointing` / `bf16` | `True` / `True` | base_config.py | SFT와 같음 |
| `learning_rate` | `1e-6` | dpo_config.py:155-158 | |
| `use_liger_kernel` | `False` | base_config.py:90-95 | DPO에서는 TRL 자체 chunked log-prob 경로 선택 + liger 설치 요구 |

**존재하지 않는 필드** (1.14.1에서 `DPOConfig(...)`에 넘기면 `TypeError: unexpected keyword argument`): `max_prompt_length`, `max_completion_length`, `use_logits_to_keep`, `model_adapter_name`, `ref_adapter_name`, `force_use_ref_model`, `label_pad_token_id`, `padding_value`, `rpo_alpha`, `reference_free`. SFT의 `max_seq_length`, `dataset_batch_size`, `num_of_sequences`, `chars_per_token`도 같다. VERIFIED (exp: 제거 필드 검사, 부록 A-1)

- prompt/completion 별도 상한이 없으므로 DPO 절단은 `max_length` 하나로 prompt+completion 시퀀스 전체에 적용된다(§7.3).
- adapter 이름은 하드코딩이다: 기존 adapter가 붙은 `PeftModel`을 `ref_model` 없이 넘기면 `"default"`를 복사한 `"ref"` adapter를 만든다 (dpo_trainer.py:647-675). VERIFIED(소스)
- label tensor가 없다. pad 값은 `input_ids`에 pad token, `attention_mask`·`completion_mask`에 0이다 (dpo_trainer.py:184-201). VERIFIED

---

## 7. DPO 전처리

trl==1.14.1 trl/trainer/dpo_trainer.py:1007-1108 (`DPOTrainer._prepare_dataset`). VERIFIED(소스) + exp02/03/07

### 7.1 prompt 추출

- **첫 example에 `prompt` key가 없을 때만** 전체에 `extract_prompt`를 적용한다 (1020-1025). `maybe_extract_prompt`는 쓰지 않는다. VERIFIED(소스)
- `extract_prompt`는 chosen과 rejected의 **공통 접두사**(대화면 공통 턴, 문자열이면 공통 문자)를 prompt로 잡는다 (trl/data_utils.py:557-641). 한쪽이 다른 쪽의 접두사이면 루프가 끝까지 돌아 인덱스가 하나 모자란다(`chosen="abc", rejected="abcdef"` → prompt `"ab"`). VERIFIED (실측)
- 예시 데이터를 매핑 없이 넣으면 prompt가 ```` ```c++\n#include <cstring>\n\nvoid copyString(char* dest, const char* src) {\n    while ( ```` 같은 **두 답변 코드의 공통 접두사**가 되고 `question`은 사라진다. prompt/completion 경계 불일치 경고만 나오고 학습은 진행된다. SFT는 `KeyError: 'text'`. VERIFIED (exp07)

### 7.2 EOS·토큰화·분할

- non-conversational이면 `chosen`, `rejected` 문자열 끝에 `eos_token`을 붙인다 (1027-1040). conversational이면 붙이지 않는다. VERIFIED
- 토큰화는 세 번의 **전체 렌더링**이다: `prompt`(conversational이면 `add_generation_prompt=True`), `prompt+chosen`, `prompt+rejected`. 그 뒤 `chosen_ids = prompt_chosen_ids[len(prompt_ids):]`로 자른다. 접두사 불일치는 경고만 하고 그대로 자른다 (1046-1092). VERIFIED
- non-conversational은 `processing_class(text=...)`(add_special_tokens=True)라 BOS를 붙이는 tokenizer면 prompt와 prompt+chosen 모두 앞에 BOS 1개가 붙는다(일관됨). INFERRED
- row별 `tools`, `chat_template_kwargs` 지원 (1047-1049). DPO에는 `chat_template_path`나 training-template 교체가 없다. VERIFIED(소스)
- 저장 컬럼: `prompt_ids`, `chosen_ids`, `rejected_ids`. 학습 signature: 이 셋 + `ref_chosen_logps`, `ref_rejected_logps` (1110-1132). VERIFIED (exp02)
- 예시 데이터(빈 system 생략, user=`question`): 독립 재구현과 **불일치 0, 접두사 위반 0** (processor 경로·tokenizer 경로 모두). chosen은 `<think></think>```c++\n…`로 시작해 `\n```<|im_end|>`로 끝난다. VERIFIED (exp02)

### 7.3 절단 위치

- 절단은 **collator**에서 시퀀스별로 한다: `prompt+chosen`, `prompt+rejected`를 각각 `[:max_length]`(`keep_start`) (dpo_trainer.py:151-164). VERIFIED
- 실측(기본값 1024): 가장 긴 row 2355(2,272 tokens)는 `(2, 1024)`로 잘리고 completion 2,004/2,002 tokens 중 756만 남는다. VERIFIED (exp03)
- 사용자 지정 collator는 Trainer가 사후 절단을 하지 않으므로 스스로 잘라야 한다(docstring, dpo_trainer.py:446-450). VERIFIED(소스)

### 7.4 row 삭제

```python
# trl==1.14.1 trl/trainer/dpo_trainer.py:1103-1106 (DPOTrainer._prepare_dataset)
if args.max_length is not None and args.truncation_mode == "keep_start":
    ...
    dataset = dataset.filter(lambda example: len(example["prompt_ids"]) < args.max_length, **map_kwargs)
```

- prompt만으로 `max_length`를 채우는 pair는 **조용히 삭제**된다(경고 없음). 실측: 1,500단어 prompt pair가 2→1 rows, `max_length=None`이면 유지(prompt 1,508 tokens). VERIFIED (exp03)
- 예시 데이터는 prompt 최대 268 tokens라 기본값에서도 삭제는 0건, 절단은 28 pairs. VERIFIED (exp02/09)
- DPO는 사전 토큰화 데이터(`prompt_ids` 등만 있는 데이터)를 받지 않는다(`KeyError: 'chosen'`). VERIFIED (exp08)

---

## 8. DPO collation, forward, reference

### 8.1 collator: 2B행, 전체 최대 길이 padding

```python
# trl==1.14.1 trl/trainer/dpo_trainer.py:146-170 (DataCollatorForPreference.torch_call), 발췌
prompt_chosen_ids = [example["prompt_ids"] + example["chosen_ids"] for example in examples]
prompt_rejected_ids = [example["prompt_ids"] + example["rejected_ids"] for example in examples]
chosen_mask = [[0] * len(example["prompt_ids"]) + [1] * len(example["chosen_ids"]) for example in examples]
rejected_mask = [[0] * len(example["prompt_ids"]) + [1] * len(example["rejected_ids"]) for example in examples]
...
input_ids = prompt_chosen_ids + prompt_rejected_ids
attention_mask = chosen_attention_mask + rejected_attention_mask
completion_mask = chosen_mask + rejected_mask
```

- 행마다 prompt+completion을 잇고, **chosen B행 다음 rejected B행**을 batch 차원으로 이어 2B행을 만든다. 2B행 전체의 최대 길이로 오른쪽 padding한다(`pad_to_multiple_of` 적용). prompt는 두 번 들어간다. VERIFIED
- 실측: 2 pairs, (prompt, chosen, rejected) = (100, 1, 600)/(50, 150, 100) → `(4, 700)`, `m=64` → `(4, 704)`; 예시 데이터 rows 0–1 → `(4, 555)`. VERIFIED (collator 실험, exp02)

```text
L_c_i = len(prompt_ids_i) + len(chosen_ids_i)      L_r_i = len(prompt_ids_i) + len(rejected_ids_i)
T_dpo_slots = 2B × round_up(max_i max(L_c_i, L_r_i), m)
```

plan §19.1의 "2 pairs, branch 100/150/700/200 → 2,800 slots"와 일치한다.

### 8.2 policy forward와 log-prob

```python
# trl==1.14.1 trl/trainer/dpo_trainer.py:1346-1350 (DPOTrainer._compute_loss)
outputs = model(**model_kwargs)
shift_logits = outputs.logits[..., :-1, :]
per_token_logps, per_token_entropies = selective_log_softmax_and_entropy(
    shift_logits, shift_labels, entropy_requires_grad=False, row_mask=shift_completion_mask
)
```

- **한 번의 forward**로 2B행을 처리한다. `logits_to_keep`을 넘기지 않으므로 prompt·pad를 포함한 **모든 위치**의 logits를 만든다. VERIFIED (exp06: `(4, 57, 248320)`, B=2·T=58)
- `selective_log_softmax_and_entropy`는 CUDA/XPU이고 dtype이 fp16/bf16/fp32이면 TRL Triton kernel을 쓴다. kernel은 forward에서 행별 fp32 벡터 4개만 할당하고 **입력 logits를 backward용으로 저장**하며, backward에서 `grad_logits = torch.empty_like(logits)`(같은 dtype, 연속 메모리)를 만든다 (trl/trainer/utils.py:501-532, 643-683; trl/kernels/logprob_entropy.py:146-248). VERIFIED(소스) / CUDA 실행 INFERRED (이 호스트는 triton 없음 → fallback 경로 실측)
- 그 밖의 지표 계산이 policy graph가 살아 있는 동안 추가 텐서를 만든다: `chosen_logits[chosen_mask.bool()]`, `rejected_logits[...]`의 boolean-index 복사(`n_completion_tokens × V`)와 `argmax` (dpo_trainer.py:1641-1669). `use_weighting=True`면 `logsumexp(2.0 * shift_logits)` 임시, `loss_type`에 `"sft"`가 있으면 chosen 위치 cross-entropy가 추가된다 (1570-1610). VERIFIED(소스) / 크기 INFERRED

### 8.3 reference 경로

| 경우 | 조건 | 동작 | 근거 |
|---|---|---|---|
| 별도 reference 모델 | `ref_model=None`, PEFT 아님, precompute 아님 | `create_model_from_path(policy 경로, **model_init_kwargs + quantization_config)`로 **두 번째 전체 모델**을 로드하고 `accelerator.prepare_model(..., evaluation_mode=True)` | dpo_trainer.py:911-927, 957-963 |
| 사용자 지정 `ref_model` | `ref_model` 전달 | 그대로 사용 (PEFT여도 사용) | dpo_trainer.py:928-929 |
| PEFT adapter 끄기 | policy가 PEFT, `ref_model=None` | `use_adapter(model, None)` → `model.disable_adapter()`로 같은 모델 forward. 기존 adapter를 재학습하는 경우 `"ref"` adapter 사용 | dpo_trainer.py:1376-1387; trl/trainer/utils.py:1365-1402 |
| precompute | `precompute_ref_log_probs=True` | 학습 중 reference forward 없음. 배치의 `ref_chosen_logps`/`ref_rejected_logps` 사용 | dpo_trainer.py:1366-1367 |

- 실측: full FT → `ref_model=Qwen3_5ForConditionalGeneration`, LoRA → `ref_model=None`, precompute → `ref_model=None`. VERIFIED (exp06)
- reference forward는 `torch.no_grad()`와 `disable_gradient_checkpointing` 안에서 돈다. 실행 시점은 **policy forward가 끝나 graph가 살아 있는 상태**다. `ref_outputs`는 `_compute_loss`가 끝날 때까지 지역 변수로 남는다 (dpo_trainer.py:1366-1422). VERIFIED(소스). 따라서 policy logits와 reference logits가 동시에 존재한다. 검증 실측(V-07, CPU): reference의 `selective_log_softmax`가 호출되는 순간, policy의 `shift_logits` 텐서(fp32, `requires_grad=True`)가 weakref 기준으로 살아 있었다(standalone ref와 LoRA 두 경로 모두). 텐서 수명은 VERIFIED이고, CUDA에서의 byte peak는 INFERRED다.
- 비호환 (VERIFIED, exp13): `sync_ref_model` + PEFT → `NotImplementedError`, `sync_ref_model` + precompute → `ValueError`, `use_liger_kernel=True` → liger 미설치로 `ImportError`, `model is ref_model` → `ValueError` (dpo_trainer.py:583-587).

### 8.4 logits dtype: accelerate가 fp32로 바꾼다

```python
# accelerate==1.15.0 accelerate/accelerator.py:1824-1835 (Accelerator.prepare_model), 발췌
if self.native_amp:
    model._original_forward = model.forward
    autocast_context = get_mixed_precision_context_manager(self.native_amp, self.autocast_handler)
    ...
        model.forward = MethodType(autocast_context(model_forward_func), model)
        model.forward = MethodType(convert_outputs_to_fp32(model.forward.__func__), model)
```

- bf16이면 CUDA에서 `native_amp = is_bf16_available(True)`, CPU에서 `True`다. DeepSpeed·Megatron-LM이면 이 분기를 타지 않아 `native_amp=False`이고 이 변환도 없다 (accelerator.py:586-593). `convert_to_fp32`는 출력의 fp16/bf16 텐서를 모두 `.float()`로 바꾼다 (accelerate/utils/operations.py:889-943). VERIFIED(소스)
- **policy** 모델은 `train()` 안의 `_prepare_for_training`에서 prepare된다 (transformers/trainer.py:1712-1743). 반면 **자동 생성되거나 사용자가 넘긴 별도 reference 모델**은 `DPOTrainer.__init__`에서 `accelerator.prepare_model(..., evaluation_mode=True)`로 바로 prepare되므로, wrapper가 `__init__`부터 붙어 있다 (dpo_trainer.py:957-963). 따라서 **학습 중** policy·reference logits는 PEFT 끄기든 별도 모델이든 fp32다. `__init__`에서 도는 precompute는 `self.ref_model or self.model`을 쓴다. 그래서 `ref_model=None`(full FT·PEFT 기본)이면 wrapper가 없는 policy가 돌아 **모델 dtype** logits가 나온다. 반대로 **사용자가 `ref_model`을 넘기면** wrapper가 붙은 ref가 돌아 **fp32** logits가 나온다. VERIFIED (exp06, CPU, bf16 모델: 학습 policy/ref `float32 (4,57,V)`, precompute `bfloat16 (4,55,V)`/`(4,57,V)`, `no_grad`). 검증 재실험(V-07, V-12): 학습 중 policy/ref `float32 (4,49,V)`. precompute는 `ref_model=None`일 때 `bfloat16`(fp32 로드 시 `float32`), bf16 `ref_model`을 넘겼을 때 `float32`였다. `__init__` 직후 `hasattr(ref_model, "_original_forward")=True`, policy는 `False`였다.

### 8.5 precompute: 언제, 무엇을, 어디에

- 시점: `DPOTrainer.__init__` 끝, `super().__init__()` 뒤 (dpo_trainer.py:986-1005). gradient checkpointing은 `train()`에서 켜지므로 이 시점에는 꺼져 있다. VERIFIED(소스)
- 모델: `self.ref_model or self.model`. full FT + `ref_model=None`이면 **학습 전 policy 자체**를 reference로 쓴다(그래서 reference 모델을 로드하지 않는다). PEFT면 adapter를 끄고 돈다 (1164-1169, 1281-1286). VERIFIED(소스)
- 데이터: `DataLoader(batch_size=precompute_ref_batch_size or per_device_train_batch_size, collate_fn=self.data_collator, shuffle=False)`(drop_last 기본 False → 전 row), 배치마다 `(2B_pre, T, V)` logits를 만든 뒤 행별 합만 CPU로 옮긴다 (1151-1186). VERIFIED(소스)
- 저장: 두 컬럼 `ref_chosen_logps`, `ref_rejected_logps`를 Arrow 파일 `cache-<fingerprint>.arrow`로 데이터셋 캐시 디렉터리에 쓰고 `concatenate_datasets(axis=1)`로 붙인다. fingerprint = hash(dataset fingerprint, `hash_module(ref or policy)`) (1140-1149, 1188-1207). 메모리 데이터셋이면 임시 디렉터리에 무작위 이름으로 쓴다 (datasets==5.0.1 datasets/arrow_dataset.py:3203-3211). VERIFIED(소스) + exp06(컬럼 추가 확인)
- `hash_module`은 `state_dict()`의 모든 텐서를 하나씩 CPU로 옮기고, bf16을 fp32로 바꾼 뒤 `tensor.numpy().tobytes()`로 hash한다 (trl/trainer/utils.py:1324-1332). VERIFIED(소스). **정정(검증)**: host RAM 순간 사용량은 가장 큰 텐서의 fp32 크기 1배가 아니라 **약 2배**다. `tobytes()`가 fp32 텐서와 같은 크기의 bytes 사본을 하나 더 만들기 때문이다. 실측(V-13, CPU): bf16 268,435,456원소 파라미터 하나로 peak RSS가 fp32 크기의 **2.00배**(2,147,876,864 bytes) 늘었다. 예시 모델의 embed/lm_head(248,320×4,096)는 fp32로 4,068,474,880 bytes이므로 ≈ 8.14 GB(7.58 GiB)가 순간적으로 필요하다. GPU에 상주하는 모델에서는 `.cpu()`가 bf16 host 사본을 먼저 만든다. 다만 `.to(float32)` 뒤에 해제되므로 peak는 그대로 2×fp32다(INFERRED).
- `IterableDataset`과 비전 데이터는 precompute 불가 (1135-1139, 733-739). 학습 시작 후 full FT + precompute로 새 eval 데이터를 넘기면 `ValueError` (1715-1727). VERIFIED(소스)

### 8.6 Liger 선택 시(참고)

`use_liger_kernel=True`는 TRL 자체 `_ChunkedLogProbFunction`을 써서 completion 위치 hidden만 모아 vocab 8,192 × token 2,048 타일로 계산한다 (dpo_trainer.py:81, 1209-1259; trl/trainer/utils.py:1457-1664). 그러나 liger-kernel 설치를 요구하고(`is_liger_kernel_available()`, dpo_trainer.py:806-811) pinned 환경에는 없다. `use_weighting`, `compute_metrics`, lm_head LoRA, prompt-learning PEFT, MoE aux loss와 비호환이다 (812-847, 904-909). VERIFIED(소스+exp13)

---

## 9. Dataloader·sampler coverage

| 항목 | 1.14.1 / 5.18 동작 | 근거 |
|---|---|---|
| `dataloader_drop_last` | 기본 `False` → 마지막 부분 배치 포함 | transformers/trainer.py:1032 |
| sampler | 기본 `RandomSampler`(`train_sampling_strategy="random"`), `sequential` 선택 가능 | trainer.py:1058-1132 |
| 마지막 accumulation 창 | 남은 배치 수(remainder)만큼 그대로 처리 | trainer.py:1826-1842 (`_run_epoch`) |
| 길이 기반 row 삭제 | SFT: 절단 시 fully-masked 삭제, `bfd` packing 빈 시퀀스 삭제. DPO: `prompt_ids ≥ max_length` 삭제. **`max_length=None`·packing 꺼짐이면 0건** | §2.6, §7.4 |
| `max_steps` | 기본 -1. 양수면 `num_train_epochs`보다 우선 → 전체 coverage 전에 종료 가능 | transformers/training_args.py:201-205 (문서) |
| `IterableDataset` | `max_steps` 필수, `dispatch_batches=False` 강제 | sft_trainer.py:956-965; dpo_trainer.py:537-546 |
| 다중 GPU | accelerate `even_batches=True`(기본)는 부족한 마지막 배치를 **데이터 앞쪽 샘플을 복제**해 채운다(삭제 아님) | transformers/training_args.py:587-588 (문서) — INFERRED |
| `group_by_length` / `batch_rebalance` | `input_ids` 또는 `length` 컬럼이 필요. DPO 데이터에는 `input_ids`가 없어 `length` 컬럼 없이는 사용 불가 | trainer.py:1075-1128 — INFERRED(DPO 적용) |
| DPO precompute loader | `shuffle=False`, drop_last 기본 False → 전 row | dpo_trainer.py:1151-1158 |

실측: `max_length=None`에서 SFT 6개 조합과 DPO 2개 경로 모두 4,656→4,656 rows. VERIFIED (exp02, exp04)

---

## 10. 무절단·무분할 보장 설정과 `max_length=None` 거부 검증

### 10.1 SFT

| 필드 | 값 | 이유 |
|---|---|---|
| `max_length` | `null` | `_prepare_dataset` 절단과 fully-masked 삭제를 끈다 |
| `packing` / `eval_packing` | `false` / `false` | packing은 `max_length` 정수 필수, bfd 절단·bfd_split 분할·wrapped 중간 절단·샘플 연결 |
| `padding_free` | `false` | 켜려면 `max_length: null` 필수 + FlashAttention 계열 필요 (엄격 기본에서는 끔) |
| `dataset_kwargs` | 미지정 | `skip_prepare_dataset: true`는 사전 토큰화 + `labels` 데이터일 때만 |
| `truncation_mode` | `keep_start`(기본) | `max_length=null`이면 무의미, `keep_end`는 deprecated |
| `dataloader_drop_last` | `false` | 마지막 배치 포함 |
| `max_steps` | `-1` | epoch 기준으로 전체 coverage |

거부 검증: `packing=True` + `max_length=None` → `ValueError` (sft_trainer.py:1686-1688). `padding_free=True` + packing 꺼짐 + `max_length` 정수 → `ValueError` (1284-1289). 그 외 `None` 거부 없음. VERIFIED (exp05)

### 10.2 DPO

| 필드 | 값 | 이유 |
|---|---|---|
| `max_length` | `null` | collator 절단과 prompt 길이 row 삭제를 끈다 |
| `padding_free` | `false` | 어차피 강제로 꺼진다 |
| `pad_to_multiple_of` | `null` 또는 정수 | padding만 늘린다(절단 없음) |
| `dataloader_drop_last` | `false` | |
| `max_steps` | `-1` | |

거부 검증: 없음. `max_length`를 쓰는 곳은 collator 절단, prompt 삭제 filter, 비전 데이터의 `keep_end` 금지, 이미지 토큰 오류 메시지뿐이다 (dpo_trainer.py:726-731, 758-764, 1103-1106, 1778-1783). VERIFIED (소스 grep + exp02/03, §11 1 step 학습에서 `(6, 3040)` 배치가 그대로 모델에 들어감)

### 10.3 YAML 파서의 함정

- TRL 스크립트의 `__main__`은 `parse_args_and_config(fail_with_unknown_args=False)`라 YAML의 **알 수 없는 key를 조용히 무시**한다. 실측: `max_prompt_length: 512`가 들어간 YAML이 DPO parser에서 오류 없이 파싱되고 값은 사라짐, `max_length: null` → `None`. `fail_with_unknown_args=True`면 `ValueError`. VERIFIED (trl/scripts/utils.py:303-359, exp11)
- **CLI 경로는 다르다(검증 시 확인, 미확정 사항 7 해소)**: `trl sft|dpo --config x.yaml`은 `parse_args_and_config(..., return_remaining_strings=True, separate_remaining_strings=True)`로 YAML의 알 수 없는 key를 `["--max_prompt_length", "512"]` 같은 문자열로 모은다. 이 문자열은 `accelerate launch` parser 인자 앞부분에 붙는다 (trl/cli/commands/training.py:51-72; trl/cli/accelerate_launcher.py:22-47; trl/scripts/utils.py:361-388 `set_defaults_with_config`). 그 결과 accelerate parser가 `error: unrecognized arguments: --max_prompt_length`로 종료한다(`SystemExit 2`). VERIFIED (V-10: `TrainingCommand.run`의 parse 단계를 그대로 재현하고 launch는 하지 않음). 정리하면, 스크립트를 직접 실행하면(`python trl/scripts/dpo.py --config ...` 또는 `accelerate launch .../dpo.py --config ...`) **조용히 무시**되고, `trl` CLI로 실행하면 **실패**한다. 이름이 accelerate launch 옵션과 겹치는 key(예: `num_processes`, `mixed_precision`)는 오류 없이 accelerate가 받아 버린다(INFERRED, parser 정의 기준).

---

## 11. Parity 테스트 레시피 (실행 검증됨)

### 11.1 설계

- **목적**: 우리 PreprocessingAdapter(torch 없음)가 만든 token ids·labels·길이와 실제 TRL 1.14.1 Trainer 출력이 같은지, `max_length=None` 설정에서 Trainer가 실제로 자르지 않는지 CPU·오프라인으로 확인한다.
- **tiny tokenizer**: `tokenizers`의 ByteLevel BPE(merge 없음, 256 byte token)에 `<|im_start|>`, `<think>`, `</think>`를 added token으로, `<|im_end|>`를 EOS, `<|endoftext|>`를 PAD로 둔다. HF cache가 필요 없고 vocab이 261이라 빠르다. 템플릿은 실제 MiMo `chat_template.jinja`(3,916 bytes)를 fixture로 둔다(출처 revision과 sha256 기록, docs/git-conventions.md §10의 fixture 규칙 충족).
- **tiny model**: `Qwen3_5ForCausalLM(Qwen3_5TextConfig(hidden 32, 2 layers = linear+full, vocab=len(tok)))`를 `save_pretrained`한 디렉터리 경로를 `model=`에 문자열로 넘긴다. 이렇게 하면 `create_model_from_path`(dtype·device_map 기본값 포함)까지 실제 경로를 탄다. linear-attention은 CPU에서 PyTorch fallback으로 돈다.
- **비교 대상**: `trainer.train_dataset[i]`의 `input_ids`/`labels`(SFT), `prompt_ids`/`chosen_ids`/`rejected_ids`(DPO), `trainer.data_collator([...])`의 shape, 1 step 학습 중 `compute_loss`가 받은 `input_ids` shape.
- **회귀 방지**: TRL 기본값(1024)에서 일어나는 절단·삭제를 그대로 assert하는 테스트를 둔다. TRL을 올릴 때 동작이 바뀌면 이 테스트가 깨져서 알려 준다.
- **저장소 배치 제안**: `tests/preprocessing_parity/test_trl_parity.py` + 템플릿 fixture(`tests/fixtures/...`, 파일명·위치는 impl-scan 소유 범위에서 결정). `pytestmark = pytest.mark.parity`와 `pytest.importorskip`으로 parity 의존성이 없으면 수집 오류 없이 skip한다(저장소 `tests/conftest.py`의 marker 정책과 `--strict-markers`에 맞춤).
- 테스트의 `ref_sft`/`ref_dpo`는 **우리 어댑터가 지켜야 할 계약**을 그대로 적은 참조 구현이다. 실제 테스트에서는 이 자리를 `vramforge_estimator.preprocessing`의 함수로 바꾸고, 참조 구현은 TRL 동작 문서화용으로 남긴다.

### 11.2 코드 (그대로 실행해 통과한 버전)

실행: `pytest -q -p no:cacheprovider test_trl_parity_doc.py` → `6 passed in 5.62s` (overlay venv, `--strict-markers` + `parity` marker 등록). torch 없는 venv에서는 `1 skipped` (수집 오류 없음). VERIFIED

```python
"""Parity contract between our preprocessor and TRL 1.14.1 SFTTrainer/DPOTrainer (CPU, offline, no HF cache).

Fixture provenance: mimo_chat_template.jinja = XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B@2367e865d009c13ac81713a2878291d33ab28177
chat_template.jinja, sha256 59a64ebb4df6d1489d09a91267cf3ceb106162d4a893c4f84833cfb8c897ff63.
"""
import os
from pathlib import Path

import pytest

os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false", TQDM_DISABLE="1")
for _mod in ("torch", "trl", "peft", "datasets"):  # collection must not fail without the 'parity' group
    pytest.importorskip(_mod)
import datasets  # noqa: E402
import torch  # noqa: E402
from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig  # noqa: E402
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer  # noqa: E402

pytestmark = pytest.mark.parity

TEMPLATE = (Path(__file__).parent / "mimo_chat_template.jinja").read_text(encoding="utf-8")

@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny-qwen3_5-bytelevel")
    vocab = {ch: i for i, ch in enumerate(pre_tokenizers.ByteLevel.alphabet())}  # 256 byte tokens, no merges
    core = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    core.pre_tokenizer, core.decoder = pre_tokenizers.ByteLevel(add_prefix_space=False), decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    tok.add_tokens(["<|im_start|>", "<think>", "</think>"])
    tok.chat_template = TEMPLATE
    tok.save_pretrained(d)
    cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                            layer_types=["linear_attention", "full_attention"], num_attention_heads=2,
                            num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                            linear_key_head_dim=8, linear_value_head_dim=8, eos_token_id=tok.eos_token_id,
                            pad_token_id=tok.pad_token_id)
    torch.manual_seed(0)
    Qwen3_5ForCausalLM(cfg).save_pretrained(d)  # architectures=["Qwen3_5ForCausalLM"] -> TRL create_model_from_path
    return str(d)

ROWS = [  # (system, question, chosen, rejected); covers empty system, unicode, code fences, a rejected-only outlier
    ("", "Write a C function.", "```c\nint f(void){return 0;}\n```", "```c\nint f(){}\n```"),
    ("You are a security reviewer.", "파이썬 eval 위험성을 설명해줘", "eval은 위험합니다.\n```python\nast.literal_eval(x)\n```", "괜찮아요"),
    ("", "Long answer please.", "A" * 1500, "B" * 3000),
]
def _sys(s): return [{"role": "system", "content": s}] if s else []  # product mapping: omit empty system
def sft_rows(): return [{"messages": _sys(s) + [{"role": "user", "content": q}, {"role": "assistant", "content": c}]} for s, q, c, _ in ROWS]
def dpo_rows(): return [{"prompt": _sys(s) + [{"role": "user", "content": q}], "chosen": [{"role": "assistant", "content": c}],
                         "rejected": [{"role": "assistant", "content": r}]} for s, q, c, r in ROWS]

# ---- reference re-implementation = the contract our preprocessor must satisfy ----
def ref_sft(tok, messages, assistant_only_loss):
    out = tok.apply_chat_template(messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=assistant_only_loss)
    mask = out["assistant_masks"] if assistant_only_loss else [1] * len(out["input_ids"])
    return out["input_ids"], [t if m else -100 for t, m in zip(out["input_ids"], mask)]

def ref_dpo(tok, prompt, chosen, rejected):
    enc = lambda msgs, gen: tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen), add_special_tokens=False)["input_ids"]
    p, pc, pr = enc(prompt, True), enc(prompt + chosen, False), enc(prompt + rejected, False)
    assert pc[:len(p)] == p and pr[:len(p)] == p  # prefix property (TRL only warns)
    return p, pc[len(p):], pr[len(p):]

_OUT = {}

@pytest.fixture(autouse=True)
def _out_dir(tmp_path):
    _OUT["dir"] = str(tmp_path / "trainer-out")

def _args(cls, **kw):
    base = dict(output_dir=_OUT["dir"], report_to="none", use_cpu=True, bf16=False,
                per_device_train_batch_size=2, save_strategy="no")
    return cls(**{**base, **kw})

@pytest.mark.parametrize("aol", [False, True])
def test_sft_messages_parity(tiny_dir, aol):
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    tr = SFTTrainer(model=tiny_dir, args=_args(SFTConfig, max_length=None, assistant_only_loss=aol),
                    train_dataset=datasets.Dataset.from_list(sft_rows()), processing_class=tok)
    assert len(tr.train_dataset) == len(ROWS)  # nothing dropped
    for i, row in enumerate(sft_rows()):
        ids, labels = ref_sft(tok, row["messages"], aol)
        assert tr.train_dataset[i]["input_ids"] == ids and tr.train_dataset[i]["labels"] == labels
    b = tr.data_collator([tr.train_dataset[i] for i in range(len(ROWS))])
    assert b["input_ids"].shape == (len(ROWS), max(len(r["input_ids"]) for r in tr.train_dataset))  # B x max_len

def test_dpo_parity_and_shape(tiny_dir):
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    tr = DPOTrainer(model=tiny_dir, args=_args(DPOConfig, max_length=None),
                    train_dataset=datasets.Dataset.from_list(dpo_rows()), processing_class=tok)
    assert len(tr.train_dataset) == len(ROWS)
    lens = []
    for i, row in enumerate(dpo_rows()):
        p, c, r = ref_dpo(tok, row["prompt"], row["chosen"], row["rejected"])
        got = tr.train_dataset[i]
        assert (got["prompt_ids"], got["chosen_ids"], got["rejected_ids"]) == (p, c, r)
        lens += [len(p) + len(c), len(p) + len(r)]
    b = tr.data_collator([tr.train_dataset[i] for i in range(len(ROWS))])
    assert b["input_ids"].shape == (2 * len(ROWS), max(lens))  # 2B x max(all chosen & rejected lengths)

def test_trl_defaults_truncate_and_drop(tiny_dir):
    """Documents TRL behavior our 'strict_no_truncation' export must override (fails loudly on a TRL upgrade)."""
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    long_prompt = [{"prompt": [{"role": "user", "content": "x" * 1200}], "chosen": [{"role": "assistant", "content": "a"}],
                    "rejected": [{"role": "assistant", "content": "b"}]}] + dpo_rows()
    tr = DPOTrainer(model=tiny_dir, args=_args(DPOConfig), train_dataset=datasets.Dataset.from_list(long_prompt), processing_class=tok)
    assert tr.args.max_length == 1024 and len(tr.train_dataset) == len(ROWS)  # prompt >= 1024 row silently dropped
    assert tr.data_collator([tr.train_dataset[2]])["input_ids"].shape[1] == 1024  # long pair truncated in collator
    sft = SFTTrainer(model=tiny_dir, args=_args(SFTConfig, assistant_only_loss=True),
                     train_dataset=datasets.Dataset.from_list([{"messages": [{"role": "user", "content": "x" * 1200},
                     {"role": "assistant", "content": "a"}]}] + sft_rows()), processing_class=tok)
    assert len(sft.train_dataset) == len(ROWS) and max(len(r["input_ids"]) for r in sft.train_dataset) <= 1024

@pytest.mark.parametrize("trainer", ["sft", "dpo"])
def test_one_step_no_truncation(tiny_dir, trainer):
    """One real optimizer step: the model must see the full, untruncated (B, T) / (2B, T) batch."""
    tok = AutoTokenizer.from_pretrained(tiny_dir)
    kw = dict(max_length=None, max_steps=1, per_device_train_batch_size=len(ROWS), train_sampling_strategy="sequential")
    if trainer == "sft":
        tr = SFTTrainer(model=tiny_dir, args=_args(SFTConfig, **kw), train_dataset=datasets.Dataset.from_list(sft_rows()),
                        processing_class=tok)
        expected = (len(ROWS), max(len(ref_sft(tok, r["messages"], False)[0]) for r in sft_rows()))
    else:
        tr = DPOTrainer(model=tiny_dir, args=_args(DPOConfig, **kw), train_dataset=datasets.Dataset.from_list(dpo_rows()),
                        processing_class=tok)
        expected = (2 * len(ROWS), max(len(p) + max(len(c), len(r)) for p, c, r in
                                       (ref_dpo(tok, x["prompt"], x["chosen"], x["rejected"]) for x in dpo_rows())))
    seen, orig = [], tr.compute_loss
    def spy(model, inputs, *a, **k):
        seen.append(tuple(inputs["input_ids"].shape))
        return orig(model, inputs, *a, **k)
    tr.compute_loss = spy
    tr.train()
    assert seen == [expected] and expected[1] > 1024, (seen, expected)
```

### 11.3 실제 MiMo tokenizer·processor로 하는 확장 (선택, network/HF cache 필요)

같은 비교를 실제 체크포인트로 4,656 rows 전체에 돌린 스크립트가 `exp02_dpo_prep.py`, `exp04_sft_prep.py`다(부록 A). 아래는 그 핵심을 옮긴 **개념 코드**다(`our_adapter`는 앞으로 만들 어댑터 자리이고, 그대로 실행한 코드는 exp02/exp04). 저장소에 넣을 때는 `@pytest.mark.network`와 `pytest.importorskip("torchvision")`, `pytest.importorskip("PIL")`을 붙인다(processor 경로는 둘 다 필요).

```python
trainer = DPOTrainer(model=TINY_DIR_WITH_REAL_PROCESSOR_FILES, args=DPOConfig(..., max_length=None),
                     train_dataset=mapped_ds, processing_class=None)   # None → AutoProcessor (TRL CLI와 같은 경로)
assert type(trainer.processing_class).__name__ == "Qwen3VLProcessor" and trainer._is_vlm
for i, row in enumerate(mapped_ds):
    assert (trainer.train_dataset[i]["prompt_ids"], trainer.train_dataset[i]["chosen_ids"],
            trainer.train_dataset[i]["rejected_ids"]) == our_adapter.dpo_ids(row)   # 4,656 rows 모두 일치 확인됨
```

`TINY_DIR_WITH_REAL_PROCESSOR_FILES`는 `build_tiny.py`처럼 실제 config에서 차원만 줄인 tiny `Qwen3_5ForConditionalGeneration`(vocab 248,320 유지, 32.0M params, 64 MB)과 `AutoProcessor.from_pretrained(MiMo).save_pretrained(dir)` 결과를 한 디렉터리에 둔 것이다. VERIFIED (exp02/04/06)

---

## 구현 시사점 (Implementation implications)

### A. PreprocessingAdapter (torch 없는 재구현) 규칙

1. **tokenizer는 `AutoTokenizer`로 로드한다**(processor 아님). 예시 모델에서 processor 경로(TRL 기본)와 결과가 같고, torch·torchvision·Pillow가 필요 없다. torch 없는 venv에서 golden 값을 재현함. VERIFIED (exp12). 다른 profile은 processor/tokenizer 동등성을 parity 테스트로 따로 확인한다(BOS를 post-processor로 붙이는 모델은 다를 수 있음, §2.3).
2. **대화 데이터는 전체 대화를 한 번 렌더링하고 `add_special_tokens=False`로 토큰화**한다. prompt와 completion을 따로 토큰화해 더하지 않는다.
   - DPO: `p = enc(prompt, add_generation_prompt=True)`, `pc = enc(prompt+chosen)`, `pr = enc(prompt+rejected)`, `chosen_ids = pc[len(p):]`, `rejected_ids = pr[len(p):]`.
   - SFT `messages`: `apply_chat_template(messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=aol)`. `labels = input_ids`, `aol`이면 mask 0 위치를 `-100`.
   - SFT 대화형 prompt-completion: `p = enc(prompt, add_generation_prompt=True)`, 전체 = `apply_chat_template(prompt+completion, ...)`, `completion_mask = [0]*len(p) + [1]*(len−len(p))`, `completion_only_loss`(기본 True)와 `aol`을 AND.
   - non-conversational: 문자열 끝에 EOS 문자열을 붙이고(이미 끝나면 생략) `tokenizer(text)`(`add_special_tokens=True`). DPO는 chosen·rejected 각각에 붙인다.
   - row별 `tools`, `chat_template_kwargs`를 템플릿에 넘기고 둘 다 `preprocess_key`에 포함한다.
3. **접두사 검사**: `pc[:len(p)] != p`이면 TRL은 경고만 하고 그대로 자른다(SFT·DPO 모두). 우리는 row 단위 `PREFIX_MISMATCH` 경고로 기록하고 TRL과 같은 방식으로 길이를 계산한다(같은 숫자를 내야 parity가 맞는다).
4. **assistant-only loss**: 템플릿에 `{% generation %}`이 없으면 TRL은 `trl/chat_templates/*_training.jinja` 중 **문자열이 정확히 일치하는** 템플릿만 교체한다. production은 trl을 import할 수 없으므로(torch 의존) ① 필요한 training template을 출처·license(Apache-2.0)와 함께 vendoring하거나 ② 해당 조합을 unsupported로 보고한다. MiMo 템플릿은 마커가 있어 교체가 없고, **assistant 헤더(`<|im_start|>assistant\n`)와 `<think></think>`가 loss에 포함**된다.
5. **row 레코드에 남길 값**(plan §7.6 보강):
   - DPO: `prompt_tokens`, `chosen_completion_tokens`, `rejected_completion_tokens`, `chosen_total_tokens = prompt+chosen`, `rejected_total_tokens = prompt+rejected`, `prefix_ok`.
   - SFT: `sft_total_tokens`, `loss_token_count = Σ(labels ≠ −100)`, `loss_positions = Σ(labels[1:] ≠ −100)`(chunked_nll의 lm_head 처리 위치 수), `first_loss_pos`.
6. **CyberNative 매핑**(제품 결정, TRL 동작 아님): DPO `prompt = [system(비어 있지 않을 때)] + [user: question]`, `chosen = [assistant: chosen]`, `rejected = [assistant: rejected]`. SFT는 같은 prompt + chosen. `lang`, `vulnerability`는 사용하지 않는다. 빈 system을 포함하면 MiMo에서 row당 정확히 +4 tokens다(이미 `preprocess_key`의 "empty-system 정책" 항목, docs/architecture.md §3.1).
7. **golden 값**(빈 system 생략, 위 매핑, MiMo @2367e86, 데이터 @81aeacf): rows 4,656 / prompt 최대 268 / prompt+chosen 최대 2,272 (row 2355) / prompt+rejected 최대 2,272 (row 3169) / DPO 최대 branch 2,272 / Σ(prompt+chosen) 1,007,173 / Σ(prompt+rejected) 841,963 / rejected가 더 긴 row 411 / SFT(chosen) 길이 p50 207·p90 316·p99 453.35·최대 2,272 / assistant-only loss tokens 20–2,007 / prompt-completion loss tokens 17–2,004 / 1,024 초과: SFT 17 rows, DPO 28 pairs. 빈 system 포함 시 각 길이 +4 (최대 2,276). VERIFIED (exp02/04/09/12)
8. **loss token 0인 row**: `max_length=None`이면 TRL은 지우지 않고 loss 0으로 학습 계산만 소비한다. 우리는 `NO_LOSS_TOKENS` 경고로 표시한다. `aol=True`인데 assistant token이 0인 row는 TRL이 `RuntimeError`로 실패하므로 `training readiness = unsupported` 사유로 보고한다.

### B. `trainer-config.yaml` 내보내기 규칙 (`strict_no_truncation`)

1. SFT 필수 필드 (TRL CLI의 `ModelConfig`+`SFTConfig` key 이름 기준):

```yaml
model_name_or_path: XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
model_revision: 2367e865d009c13ac81713a2878291d33ab28177
dtype: bfloat16                 # TRL 기본 float32
max_length: null                # 절단·fully-masked 삭제 끔
packing: false
eval_packing: false
padding_free: false
loss_type: chunked_nll          # 기본값이지만 resolved 값을 명시
use_liger_kernel: false         # pinned 환경에 liger-kernel 없음
assistant_only_loss: <resolved> # completion_only_loss는 prompt-completion이면 null(=true)
gradient_checkpointing: true
bf16: true
per_device_train_batch_size: 1  # TrainingArguments 기본 8
gradient_accumulation_steps: 8
dataloader_drop_last: false
max_steps: -1
num_train_epochs: <resolved>
```

2. DPO 필수 필드:

```yaml
model_name_or_path: XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
model_revision: 2367e865d009c13ac81713a2878291d33ab28177
dtype: bfloat16
max_length: null                # collator 절단·prompt 길이 삭제 끔
padding_free: false             # TRL이 어차피 끈다
loss_type: [sigmoid]
beta: 0.1
precompute_ref_log_probs: <reference_strategy == precomputed_log_probs>
precompute_ref_batch_size: <null 또는 값>
sync_ref_model: false
use_liger_kernel: false
disable_dropout: true
gradient_checkpointing: true
bf16: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
dataloader_drop_last: false
max_steps: -1
```

3. **제거된 필드를 내보내지 않는다**: `max_prompt_length`, `max_completion_length`, `use_logits_to_keep`, `model_adapter_name`, `ref_adapter_name`, `force_use_ref_model`, `label_pad_token_id`, `padding_value`, `rpo_alpha`, `reference_free`, `max_seq_length`, `dataset_batch_size`, `num_of_sequences`, `chars_per_token`. TRL 스크립트를 직접 실행하면 YAML의 알 수 없는 key가 **조용히 무시**된다. `trl` CLI로 실행하면 그 key가 accelerate launch 인자로 넘어가 **실행 시작 전에 실패**한다(§10.3). 어느 쪽이든 우리 쪽에서 미리 검증해야 한다. parity 테스트에 내보낸 YAML을 `TrlParser(...).parse_args_and_config(fail_with_unknown_args=True)`로 파싱하는 테스트를 둔다(exp11 방식). production은 1.14.1 필드 목록을 profile에 버전 고정해 검사한다. 검증 시 위 B-1/B-2 YAML(placeholder를 실제 값으로 바꾸고 `dataset_name`, `output_dir`만 추가)을 SFT·DPO parser에 `fail_with_unknown_args=True`로 넣었고, 오류 없이 파싱되며 CLI 경로의 `config_remaining`도 `[]`였다. VERIFIED (V-20)
4. **데이터 참조**: 내보내기는 매핑이 끝난 데이터(`prompt/chosen/rejected` 또는 `messages`)를 가리켜야 한다. `dataset_name: CyberNative/...` 원본을 TRL CLI에 그대로 주면 DPO는 잘못된 prompt로 조용히 학습하고 SFT는 실패한다(§7.1). 매핑 스크립트나 매핑된 데이터 artifact를 함께 내보내고, 매핑이 없으면 `ready`로 표시하지 않는다.
5. **실행 환경 요구**: `processing_class`를 넘기지 않는 TRL CLI 경로에서 Qwen3.5 VLM 체크포인트는 Pillow와 torchvision이 있어야 Trainer가 생성된다. environment profile에 두 패키지를 넣거나, `AutoTokenizer`를 `processing_class`로 넘기는 launcher를 내보낸다(두 경로의 ids는 동일 검증).
6. **PEFT target**: SFT `chunked_nll`에서는 `lm_head`를 target에 넣으면 실패한다. `all-linear`는 lm_head를 빼지만 VLM 비전 타워 linear는 포함하므로, text-only preset은 `language_model` 범위로 한정하거나 `exclude_modules`로 `visual`을 뺀다(PEFT 조사와 교차 확인).

### C. BatchPlanner 공식

```text
SFT (padding)       slots = B × round_up(max_i L_i, m)
SFT (padding_free)  slots = round_up(Σ_i L_i, m)                       # 엄격 기본에서는 사용 안 함
DPO                 slots = 2B × round_up(max_i max(Lc_i, Lr_i), m)    # Lc = prompt+chosen, Lr = prompt+rejected
m = pad_to_multiple_of (null이면 1)
```

lm_head가 처리하는 위치 수(logits 크기 결정):

| 경로 | lm_head 처리 위치 | 비고 |
|---|---|---|
| SFT `chunked_nll` | `Σ (labels[1:] ≠ −100)` (padding·마스크 위치 제외), 256개 chunk 단위 | 순간 logits는 `256 × V` |
| SFT `nll`/`dft` | `B × T` 전부 | |
| DPO (비 Liger) | `2B × T` 전부 (prompt·pad 포함) | policy와 reference 각각 |
| DPO precompute | `2B_pre × T_pre` 전부 | `B_pre = precompute_ref_batch_size or per_device_train_batch_size` |

예시 최악 microbatch(B=1): SFT 2,272 slots, DPO 2 × 2,272 = 4,544 slots (row 2355가 chosen 최대, row 3169가 rejected 최대이며 서로 다른 row다). VERIFIED (exp09)

### D. TrainerAdapter용 logits·loss allocation (timepoint 후보)

bf16 mixed precision, 모델 dtype bf16, `V = vocab_size`(lm_head 출력 차원), `N = 2B`(DPO) 기준. "dtype" 열은 CPU에서 실측했고, 수명·동시 존재는 소스 기반 추론이다.

| allocation | shape × bytes | 살아 있는 구간 | 근거 |
|---|---|---|---|
| SFT chunked: 정렬된 hidden 복사 | `B(T−1) × H × 2` | loss → backward(loss) | INFERRED (sft_trainer.py:197-199) |
| SFT chunked: chunk 순간값 | `256 × V × 16` (CPU 실측 15.0–16.5 B, 보수적 상한 18 B) | loss chunk 순간, backward 재계산 순간 (chunk 수와 무관) | CPU 실측 VERIFIED (V-14e), CUDA INFERRED (sft_trainer.py:100-118) |
| SFT chunked: lm_head weight cast | `V × H × 2` | 모델이 fp32로 로드된 경우만 | INFERRED |
| SFT nll: 출력 logits | `B × T × V × 4` (fp32) | forward 끝 → compute_loss 반환 | dtype VERIFIED (exp06), 수명 INFERRED |
| SFT nll: CE log-softmax 저장 | `B·T × V × 4` | forward → backward | INFERRED (loss_utils.py:59-70) |
| DPO policy logits bf16 | `N × T × V × 2` | lm_head 직후 순간 | INFERRED |
| DPO policy logits fp32 | `N × T × V × 4` | forward → fused kernel backward | dtype VERIFIED (exp06), 수명 INFERRED (logprob_entropy.py:192) |
| DPO 지표 boolean-index 복사 | `n_completion × V × 4` | compute_loss 안 순간 | INFERRED (dpo_trainer.py:1643-1647) |
| DPO reference logits fp32 | `N × T × V × 4` | ref forward → compute_loss 반환 (policy logits와 **동시**) | dtype VERIFIED, policy logits와 동시 생존 CPU VERIFIED (V-07), CUDA byte peak INFERRED (dpo_trainer.py:1366-1422) |
| DPO backward grad_logits | `N × (T−1) × V × 4` → slice backward `N × T × V × 4` → bf16 cast `N × T × V × 2` | backward 순차 | INFERRED (logprob_entropy.py:210) |
| DPO precompute logits | `2B_pre × T_pre × V × bytes(모델 dtype)`. 사용자가 `ref_model`을 넘기면 `× 4`(fp32) | precompute 배치 순간 (no_grad, activation 저장 없음) | dtype VERIFIED (exp06, V-07, V-12) |

크기 감각(텐서 크기 산술일 뿐 VRAM 실측이 아님): V = 248,320에서 DPO 최악 row(N=2, T=2,272)의 fp32 logits 1개 = 4,513,464,320 bytes(4.20 GiB)이고, policy·reference fp32가 겹치면 8.41 GiB다. SFT `nll` B=1·T=2,272의 fp32 logits 1개 = 2.10 GiB, SFT `chunked_nll`의 `256 × V` fp32 버퍼 1개 = 254,279,680 bytes(0.24 GiB). 실제 피크는 plan §17 GPU 보정으로 확정한다.

### E. reference 전략 매핑 (plan §5.3)

| plan 전략 | TRL 1.14.1 조건 | 메모리 의미 |
|---|---|---|
| `frozen_base_switch` | PEFT policy + `ref_model=None` | 추가 가중치 없음. 같은 모델을 adapter 끄고 no-grad forward. 기존 adapter가 붙은 `PeftModel`을 넘기면 `"ref"` adapter 사본이 추가로 상주 |
| `standalone_model` | PEFT 아님 + precompute 아님 (또는 `ref_model` 지정) | **두 번째 전체 모델**. dtype·quantization은 policy와 같은 `model_init_kwargs`(기본 float32!, V-12 실측). `__init__`에서 바로 prepare되므로 policy보다 먼저 device에 올라가고 autocast·fp32 출력 wrapper가 붙는다 (dpo_trainer.py:957-963) |
| `precomputed_log_probs` | `precompute_ref_log_probs=True` | 학습 phase에 reference 없음. `REFERENCE_PRECOMPUTE` phase가 `__init__`에서 먼저 실행된다: no-grad forward를 돌리고, logits는 모델 dtype이다(`ref_model`을 직접 넘기면 fp32). 결과는 row당 scalar 2개로 Arrow cache에 저장한다. full FT면 policy 자체를 reference로 쓴다. host RAM은 `hash_module` 때문에 가장 큰 텐서 fp32 크기의 약 2배가 순간적으로 필요하다(§8.5) |

차단 규칙: `sync_ref_model`은 PEFT·precompute와 함께 쓸 수 없다. precompute는 `IterableDataset`·비전 데이터와 함께 쓸 수 없다. VERIFIED

### F. Compatibility registry 항목

- `trainer=sft`: `loss_path=chunked_nll` (기본) — 조건: lm_head가 PEFT 대상이 아님, liger 꺼짐, top-level `lm_head` 존재. 위반 시 `nll` 경로로 바꾸는 것은 메모리가 크게 늘므로 **자동 fallback 금지**, requested/resolved를 분리해 기록한다.
- `use_liger_kernel=true`: pinned 환경에서 unsupported(SFT `train()`·DPO `__init__`에서 ImportError).
- DPO `padding_free`: requested가 true여도 resolved는 false.
- SFT `packing`: 엄격 모드 unsupported. 허용 모드라도 `max_length` 정수 필수, `bfd`(절단)·`bfd_split`(분할)·`wrapped`(중간 절단)를 각각 다른 데이터 보존 상태로 표시한다.
- SFT `padding_free`: `max_length=null` + FlashAttention 계열 필요. Qwen3.5 hybrid의 linear-attention 경계 처리는 UNKNOWN → 미검증 표시.
- `assistant_only_loss`: `{% generation %}` 마커 또는 TRL 동봉 템플릿 일치가 필요하다.
- QLoRA: TRL이 학습 가능한 파라미터를 bf16으로 바꾼다. CLI 경로의 `bnb_4bit_compute_dtype`은 `dtype`을 따른다.
- 비양자화 LoRA: adapter 파라미터는 fp32다(PEFT 기본 `autocast_adapter_dtype=True`, V-17 실측). 예외는 ZeRO-3로, 이때는 base dtype을 따른다. adapter weight·grad·optimizer state bytes는 이 dtype으로 계산한다.

### G. resolved config에 기록할 기본 동작

- gradient checkpointing: TRL 기본으로 켜져 있다. transformers 5.18에서 `use_reentrant=False`, `every_n_layers=1`(decoder layer마다, 비전 block 포함), `offload=False`다. `every_n_layers`/`offload`는 `gradient_checkpointing_kwargs`의 key로 activation ledger에 영향을 준다. PEFT + ZeRO-3이면 TRL이 `use_reentrant=True`로 강제한다.
- SFT·DPO forward 모두 `use_cache=False`.
- DPO는 policy·reference의 dropout을 0으로 만든다(`disable_dropout=True`).
- 기본 optimizer는 `adamw_torch_fused`.
- 예시 모델은 `accepts_loss_kwargs=False`라 loss가 microbatch 평균 후 accumulation 횟수로 나뉜다(메모리 무관, 로그 해석에만 영향).

---

## 미확정 사항 (Open questions)

1. **CUDA fused kernel과 autocast의 실제 수명·피크**: DPO logits(fp32 변환본, grad_logits, slice backward)와 SFT chunk 순간값은 소스 추론이다. GPU 보정(plan §17)에서 `torch.cuda.memory_stats`로 timepoint별 확인이 필요하다.
2. **fp32 로드(TRL 기본) + bf16 autocast의 weight cast cache**: 학습 가능한 weight의 bf16 사본이 autocast 구간 동안 유지되는지, LoRA의 frozen weight가 op마다 다시 cast되는지 GPU에서 확인해야 한다. 내보내기는 bf16을 명시하므로 기본 시나리오에는 영향이 없다.
3. **Qwen3.5 hybrid + padding_free/packing**: GatedDeltaNet(linear attention)이 packed `position_ids`/`seq_lengths` 경계를 지키는지 UNKNOWN. 확인 전까지 해당 조합은 unsupported.
4. **비전 타워 상주 크기**: TRL이 `Qwen3_5ForConditionalGeneration`을 로드한다는 것은 확인했지만 비전 타워 파라미터 수·4-bit 양자화 제외 여부는 model-inspection/quantization 조사 범위다.
5. **`all-linear` LoRA가 비전 타워에 붙은 경우의 optimizer state**: 검증 시 CPU에서 **해소**했다(V-19). text-only SFT 1 step 뒤 fused `AdamW`를 보면, 비전 LoRA 파라미터 12개는 optimizer param group에 등록돼 있지만 state가 0개였다. text LoRA 파라미터는 30개 모두 state가 있었다. 따라서 비전 adapter는 weight(fp32)만 상주하고 grad와 optimizer state는 생기지 않는다. CUDA fused 경로도 같은 `grad is None` skip 로직을 쓸 것으로 본다(INFERRED). 남은 일은 PEFT 조사와 weight bytes를 교차 확인하는 것이다.
6. **다중 GPU coverage**: accelerate `even_batches=True`의 샘플 복제가 "전체 row 사용"을 어떻게 바꾸는지(중복 포함) 실행 검증 필요. 현재 범위는 단일 GPU.
7. **`trl <cmd> --config` CLI의 알 수 없는 key 처리**: 검증 시 **해소**했다(V-10, §10.3). 알 수 없는 key는 accelerate launch parser로 넘어가 `unrecognized arguments`로 실패한다. accelerate launch 옵션과 이름이 겹치는 key는 accelerate가 받아 들인다(INFERRED). 우리 내보내기는 알 수 없는 key를 아예 만들지 않으므로 영향은 작다.
8. **parity 의존성 그룹**: 루트 `pyproject.toml`의 `parity` 그룹은 trl/torch/peft/accelerate만 고정한다. 학습 환경과 같게 transformers==5.18.0, datasets==5.0.1, tokenizers==0.23.2, huggingface_hub==1.33.0도 고정할지, processor 경로 테스트용 Pillow·torchvision을 넣을지는 오케스트레이터 결정이 필요하다(이 문서는 파일을 바꾸지 않음).
9. **다른 모델의 processor/tokenizer 동등성**: BOS를 post-processor로 붙이는 tokenizer나 문자열 content만 받는 템플릿에서는 두 경로가 다를 수 있다. profile마다 parity로 확인한다.
10. **liger-kernel 도입 여부**: 나중에 환경에 넣으면 SFT Liger 경로와 DPO chunked log-prob 경로의 메모리·호환성을 다시 조사해야 한다.

---

## 부록 A. 실험 목록과 주요 출력

모든 스크립트는 `/tmp/vf-research/scratch/trl-sft-dpo/`에 있고 공통 설정은 `common.py`(HF_HOME, `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`)다. 실행 인터프리터는 `/tmp/vf-research/venv-trl-sft-dpo/bin/python`(exp12만 torch 없는 venv).

| ID | 명령/스크립트 | 주요 출력 |
|---|---|---|
| A-1 | `SFTConfig`/`DPOConfig` 생성 후 필드 덤프, 제거 필드로 생성 시도 (인라인 python, 공용 venv) | §1.1·§6 표의 값. 제거 필드 14개 모두 `TypeError: ... unexpected keyword argument`. `max_length=None` 두 config 모두 허용 |
| A-2 | `exp01_processor.py` | 공용 venv: image processor 단계 `ValueError`(Pillow·torchvision 없음). Pillow만: `Qwen3VLVideoProcessor requires the Torchvision library`. 둘 다 있을 때: `AutoProcessor -> Qwen3VLProcessor`, tokenizer `Qwen3_5Tokenizer`, bos `None`, eos 248046, pad 248044, `add_special_tokens True vs False: [14556, 1814] [14556, 1814] True` |
| A-3 | `build_tiny.py` | tiny `Qwen3_5ForConditionalGeneration` 32,024,600 params, vocab 248,320, layer_types `['linear_attention','full_attention']`, 실제 processor 파일과 함께 저장 |
| A-4 | `exp02_dpo_prep.py processor None` / `tokenizer None` | 두 경로 모두 `rows_in=4656 rows_out=4656`, `reimpl mismatches: 0 prefix violations: 0`, prompt 최대 268, branch 최대 2,272, 1,024 초과 28, collated `(4, 555)` |
| A-5 | `exp03_dpo_default_trunc.py` | 기본 `max_length=1024`, row 2355 → `(2, 1024)`, completion 2,004/2,002 → 756/756. prompt 1,500단어 pair: 2→1 rows. `None`: 2 rows, prompt 1,508 |
| A-6 | `exp04_sft_prep.py {messages,pc} {processor,tokenizer} {noaol,aol} None` (6조합) | 모두 4,656→4,656, `reimpl mismatches: 0`, `model_accepts_loss_kwargs=False`, `swapped_template=False`. aol 첫 loss token `<\|im_start\|>assistant\n<think>`(pos 95), pc 첫 loss token `<think></think>```c`(pos 98), loss tokens: no-aol 51–2,272 / aol 20–2,007 / pc 17–2,004 |
| A-7 | `exp05_sft_validations.py` | §1.2·§1.3 실측값. `padding_free` + 기본 `max_length` → `ValueError`, `packing` + `None` → `ValueError`, padding-free collated `(1, 1525)` + `position_ids` |
| A-8 | `exp06_train_smoke.py {sft,sft_nll,dpo,dpo_lora,dpo_pre}` (CPU, `bf16=True`, `dtype=bfloat16`, 1 step) | sft: chunked loss hidden `(2,58,64) bf16`, `logits=None`. sft_nll: `logits=float32(2,58,248320)`. dpo: ref `Qwen3_5ForConditionalGeneration`, policy·ref `float32 (4,57,248320)`. dpo_lora: ref `None`, `float32`. dpo_pre: precompute `bfloat16 (4,55,V)`·`(4,57,V)` no-grad, 학습 중 ref forward 없음, 컬럼 `ref_chosen_logps`, `ref_rejected_logps` 추가 |
| A-9 | `exp07_raw_unmapped.py` | DPO prompt = 두 코드 답변의 공통 접두사, `question` 미포함, 경계 불일치 경고. SFT `KeyError 'text'` |
| A-10 | `exp08_pretokenized.py` | SFT 사전 토큰화: 기본 → 1 row·1,024 / `None` → 2 rows·1,200·10 / `skip_prepare_dataset` → 2 rows·1,200·10. DPO 사전 토큰화: `KeyError 'chosen'` |
| A-11 | `exp09_golden_stats.py` | 구현 시사점 A-7의 golden 값 (빈 system 생략/포함 두 경우) |
| A-12 | `exp10_aol_template.py` | MiMo `has_generation_markers=True`, stop token trained `True`. 마커 없는 미등록 템플릿 + aol → `ValueError` |
| A-13 | `exp11_cli_yaml.yaml` + DPO `make_parser()` | `fail_with_unknown_args=False`: 파싱 성공, `max_length=None`, `max_prompt_length` 무시. `True`: `ValueError: Unknown arguments from config file` |
| A-14 | `exp12_torchfree_reimpl.py` (torch 없는 venv) | `torch imported: False`, branch 최대 2,272, Σ(prompt+chosen) 1,007,173, aol loss tokens 20–2,007 |
| A-15 | `exp13_incompat.py` + LoRA `all-linear` 확인 | §1.6·§8.3의 오류 메시지 그대로. `all-linear`: 21개 중 비전 6개, lm_head 미포함 |
| A-16 | `parity/test_trl_parity_doc.py` | `6 passed in 5.62s` (overlay venv), torch 없는 venv에서 `1 skipped` |
| A-17 | collator 단독 실험 (인라인 python) | §4·§8.1의 shape 표 |

---

## 검증 로그 (Verification log)

| 항목 | 값 |
|---|---|
| 검증자 | `verify-trl-sft-dpo` (적대적 사실 확인, Milestone M0) |
| 날짜 | 2026-10-04 |
| 방법 | 원저자의 실험 스크립트(exp01–exp13)는 재사용하지 않았다. 설치된 site-packages 소스를 다시 읽고 독립 스크립트 V-01–V-22를 새로 작성해 실행했다. tiny 모델도 따로 만들었다(`v02_build_tiny.py`). 하나는 실제 MiMo config에서 차원만 줄인 `Qwen3_5ForConditionalGeneration`(32,018,068 params, vocab 248,320)에 실제 processor 파일을 붙인 것이다. 다른 하나는 text 전용 `Qwen3_5ForCausalLM`에 실제 tokenizer를 붙인 것이다 |
| 환경 | 공용 `/tmp/vf-research/.venv`는 바꾸지 않았다. 자체 오버레이 venv 3개를 썼다. `venv-verify-trl-sft-dpo`: Pillow 12.3.0, torchvision 0.29.1 `--no-deps`, pytest 9.1.1. `venv-verify-trl-sft-dpo-pilonly`: Pillow만. `venv-verify-trl-sft-dpo-notorch`: torch 미설치에 transformers 5.18.0, tokenizers 0.23.2, huggingface_hub 1.33.0, jinja2, numpy, pytest |
| 스크립트 | `/tmp/vf-research/scratch/verify-trl-sft-dpo/` (저장소에는 넣지 않음) |
| 한계 | CUDA 전용 경로는 이 호스트에서 실행할 수 없어 INFERRED로 남긴다. 해당 경로는 Triton fused kernel의 `grad_logits`, autocast weight cast cache, CUDA caching allocator, FlashAttention이다. 다중 GPU `even_batches`도 실행하지 않았다 |

### 원저자 핵심 주장 20개

| # | 주장 (원문 요약) | 판정 | 근거 (한 줄) |
|---|---|---|---|
| C1 | 두 config의 `max_length` 기본값은 1024이고, `None`이면 두 Trainer 모두 end-to-end로 자르지 않는다 | VERIFIED | sft_config.py:203-210, dpo_config.py:194-200. V-01 덤프(1024/1024, `None` 허용). V-04에서 8개 조합 모두 4,656→4,656. V-22에서 parity 1-step 테스트(T>1024 배치)가 통과했다 |
| C2 | SFT 절단은 `_prepare_dataset`에서만 일어나고, 절단 뒤 label이 전부 -100인 row를 경고 없이 지운다 | VERIFIED | sft_trainer.py:1659-1683. V-06: aol + 1024에서 2→1 rows였고 drop/trunc 관련 경고는 0건이었다. aol을 끈 기본값은 지우지 않고 1024로 자른다 |
| C3 | DPO는 collator에서 시퀀스별로 자르고, keep_start에서 `len(prompt_ids) >= max_length`인 pair를 조용히 지운다 | VERIFIED | dpo_trainer.py:151-164, 1103-1106. V-06: 2→1 rows, 경고 없음. 저장된 ids는 그대로이고 collator 출력만 `(2,1024)`다. V-18: row 2355의 completion 2,004/2,002가 756/756으로 줄었다 |
| C4 | packing+`None`은 ValueError, bfd는 절단, bfd_split은 분할, wrapped는 중간 절단이다. bfd는 빈 시퀀스를 지운다. packing 없는 padding_free는 `None`이 필수다 | VERIFIED (보완) | sft_trainer.py:1686-1688, 1284-1289; data_utils.py:739-843. V-06: 길이 32/7/14/52의 4 rows가 bfd 3 rows·85 tok, bfd_split 4·105, wrapped 4·105(`seq_lengths` 없음)가 됐다. 두 ValueError 메시지도 확인했다. 보완: 빈 시퀀스 삭제는 같은 `_pack_bfd`를 쓰는 bfd_split에도 적용된다 |
| C5 | SFT 기본 `loss_type`은 `"chunked_nll"`이다. `labels != -100` 위치만 256 단위 chunk로 lm_head를 돌리고 `logits=None`을 반환한다. LoRA-lm_head나 liger와 함께 쓰면 ValueError다 | VERIFIED | sft_config.py:332-334; sft_trainer.py:87, 121-232, 1338-1366. V-07: hidden `bf16 (2,50,64)`, `outputs.logits=None`. V-08에서 두 ValueError 메시지가 원문 그대로 나왔다 |
| C6 | DPO는 chosen B행 뒤에 rejected B행을 붙여 2B행을 만들고, 최대 길이로 right-pad한 뒤 한 번의 forward로 모든 위치의 logits를 만든다 | VERIFIED | dpo_trainer.py:146-201, 1346-1350. V-07: B=2·T=50에서 policy `shift_logits (4,49,248320)`. V-04: row 2355 단독 collate가 `(2,2272)` |
| C7 | bf16에서 accelerate가 autocast와 fp32 변환을 건다. 학습 중 policy/ref logits는 fp32다. 모델은 `train()`에서만 prepare되므로 `__init__` precompute는 모델 dtype이다 | CORRECTED | policy에 대한 부분은 맞다(accelerator.py:1824-1835; V-07 `float32 (4,49,V)`). 그러나 별도 reference 모델은 `__init__`에서 prepare된다(dpo_trainer.py:957-963; V-12에서 `_original_forward` 확인). 그래서 사용자가 `ref_model`을 주고 precompute하면 logits가 fp32다(V-12). §8.4, D 표, E 표를 고쳤다 |
| C8 | full FT는 두 번째 전체 모델을 쓰고, PEFT는 `disable_adapter`(또는 `"ref"` adapter 사본)를 쓴다. precompute는 `__init__`에서 policy로 돌고 row당 scalar 2개를 Arrow cache에 쓴다 | VERIFIED | dpo_trainer.py:647-675, 911-927, 986-1005, 1134-1207, 1376-1387. V-07: ref 타입은 full FT에서 `Qwen3_5ForConditionalGeneration`, LoRA에서 `None`이고, precompute는 컬럼 2개를 추가했다. V-08: 기존 `PeftModel`이면 adapters가 `['default','ref']`. V-12: ref dtype이 policy dtype과 같다 |
| C9 | 문자열로 넘긴 모델은 `dtype`이 없으면 float32로, GPU면 `device_map="auto"`로 로드한다. CLI `ModelConfig.dtype` 기본값은 `"float32"`이고 QLoRA compute dtype이 이를 따른다 | VERIFIED (보완) | utils.py:1292-1305, 252-268; model_config.py:89-95. V-01 `ModelConfig().dtype='float32'`, V-07/V-12 기본 로드 `float32`. 보완: 분산 실행에서는 `device_map=None`이다 (sft_trainer.py:981-983; dpo_trainer.py:562-564, 922-924) |
| C10 | `processing_class=None`이면 `Qwen3VLProcessor`·`_is_vlm=True`가 되고 text 경로로 처리된다. processor와 tokenizer의 ids가 4,656 rows에서 같다. Pillow·torchvision이 없으면 실패한다 | VERIFIED | sft_trainer.py:1002-1016; data_utils.py:401-437. V-04: DPO와 SFT(messages, messages+aol, prompt-completion) 모두 두 경로에서 불일치 0. V-05: 공용 venv는 `ValueError`(image processor), Pillow만 있으면 `ImportError`(Qwen3VLVideoProcessor requires Torchvision), 둘 다 있으면 `Qwen3VLProcessor` |
| C11 | tokenizer의 `apply_chat_template(tokenize=True)`는 항상 `add_special_tokens=False`를 쓰고, processor는 BOS로 시작할 때만 그렇게 한다. MiMo는 BOS가 없다 | VERIFIED | tokenization_utils_base.py:3123-3132; processing_utils.py:2225-2227. V-03: `bos_token=None`, `"Hello world"`는 True/False 모두 `[9419, 1814]` |
| C12 | MiMo 템플릿은 macro 안의 `{% generation %}`이 assistant 턴 전체를 감싼다. 그래서 aol에 `<\|im_start\|>assistant\n<think></think>`가 들어가고 pc는 3 token 뒤부터 시작한다. 마커가 없고 TRL에 등록되지 않은 템플릿 + aol은 ValueError다 | VERIFIED | 템플릿 sha256 `59a64ebb…ff63`을 직접 확인했다. V-03: row 0의 첫 aol 위치는 95 `['<\|im_start\|>','assistant','Ċ','<think>']`, pc 첫 loss 위치는 98 `<think>`. V-09: 마커를 제거한 템플릿 + aol → `ValueError`, stop token trained `True`. 비교는 문자열 일치다(chat_template_utils.py:1032-1199) |
| C13 | 제거된 DPO/SFT 필드는 TypeError, DPO `padding_free=True`는 False로 바뀐다. TRL 스크립트는 YAML의 알 수 없는 key를 무시한다 | VERIFIED (보완) | V-01: 14개 모두 `TypeError`. V-06: `trainer.padding_free=False` + 경고. V-10: 스크립트 경로에서 무시됨을 확인했다. 보완: `trl` CLI 경로는 accelerate launch parser에서 `unrecognized arguments`로 실패한다(§10.3) |
| C14 | 매핑하지 않은 CyberNative를 넣으면 DPO는 코드 공통 접두사를 prompt로 잡고 `question`을 버린다. SFT는 `KeyError 'text'`다 | VERIFIED | dpo_trainer.py:1020-1025; data_utils.py:557-641. V-09: prompt가 ```` ```c++\n#include <cstring>\n\nvoid copyString(... while ( ````였고 `question`은 없었다. SFT는 `KeyError: 'text'`. `extract_prompt('abc','abcdef')`의 prompt는 `'ab'` |
| C15 | golden 값(최대 branch 2,272 등, 빈 system +4)과 torch 없는 재현 | VERIFIED | V-03(tokenizer만), V-03b(torch 미설치 venv, `torch imported: False`), V-04(TRL 출력)가 모두 같았다. prompt 최대 268, row 2355/3169 2,272, Σ 1,007,173/841,963, rejected가 더 긴 row 411, 1024 초과 SFT 17·DPO 28, p50/p90/p99 207/316/453.35, aol 20–2,007, pc 17–2,004. 빈 system을 넣으면 모든 row가 +4 |
| C16 | `dataloader_drop_last=False`, RandomSampler, 마지막 accumulation 나머지도 처리한다. `max_length=None`에 packing을 끄면 길이 기반 삭제가 없다 | VERIFIED | trainer.py:1032, 1058-1132, 1826-1842, 2468-2475. V-11: 5 rows·bs 2·GA 2에서 배치 `[2,2,1]`, `current_gradient_accumulation_steps` `[2,2,1]`, optimizer step 2, random·sequential 모두 전 row 사용 |
| C17 | TRL 기본 `gradient_checkpointing=True`·`bf16=True`, TrainingArguments 기본 `per_device_train_batch_size=8`·`adamw_torch_fused`. GC는 `train()`에서 `use_reentrant=False`·`every_n_layers=1`·`offload=False`로 켜진다 | VERIFIED (보완) | base_config.py:61-74, 104-105; training_args.py:775, 804-811; trainer.py:1490-1501; modeling_utils.py:3138-3139. V-01·V-16: `__init__` 뒤 False, `train()` 뒤 True, kwargs `{'use_reentrant': False}`. 보완: PEFT + ZeRO-3 + GC는 `use_reentrant=True`로 강제된다 |
| C18 | `Qwen3_5ForConditionalGeneration.accepts_loss_kwargs=False`라 `num_items_in_batch`가 넘어가지 않고, loss는 로컬 평균을 GA로 나눈 값이다 | VERIFIED (보완) | modeling_qwen3_5.py:1768; trainer.py:503-511, 2064-2066. V-04 `model_accepts_loss_kwargs False`, V-07 `num_items_in_batch None`. 보완: `Qwen3_5ForCausalLM`은 True다(V-21). DPO는 항상 False다(dpo_trainer.py:949-952) |
| C19 | liger-kernel이 없어 SFT는 `train()`, DPO는 `__init__`에서 ImportError가 난다 | VERIFIED | transformers/integrations/liger.py:39-43 (trainer.py:1481-1482에서 호출); dpo_trainer.py:806-811. V-08: SFT는 생성까지 OK(`loss_type=nll`)이고 `train()`에서 ImportError, DPO는 생성 시 ImportError |
| C20 | `all-linear`은 비전 linear를 감싸고 lm_head는 뺀다 | VERIFIED (보완) | V-08: 21개 중 비전 6개, `lm_head` 미포함. 보완: merger `linear_fc1/fc2`도 포함된다. 실제 모델은 110개로 추정한다(INFERRED) |

### 추가 확인 항목

| # | 항목 | 판정 | 근거 |
|---|---|---|---|
| X1 | §8.5 `hash_module` host RAM "≈ 4.07 GB" | CORRECTED | `tobytes()` 사본 때문에 가장 큰 텐서 fp32 크기의 약 2배가 필요하다. V-13 실측은 2.00배였다. 예시 모델에서는 ≈ 8.14 GB(7.58 GiB)다 |
| X2 | §5.1 chunk 순간값 "≈ 256·V·18 bytes" | CORRECTED (정밀화) | V-14e CPU `MemTracker`: 256·V 원소당 15.0–16.5 B였고 chunk 수와 무관했다. 18 B는 보수적 상한으로 써도 된다. `entropy_sum`을 backward까지 붙잡으면 chunk마다 8 B씩 쌓인다(V-14c). `SFTTrainer`는 backward 전에 `outputs`를 버리므로 해당하지 않는다 |
| X3 | §1.3 packing bin이 1,000-row map 배치 단위로 만들어진다 (원문 INFERRED) | VERIFIED | datasets/arrow_dataset.py:3221 `batch_size=1000`. V-15: 1,500 rows가 `[1000, 500]`으로 packing됐다 |
| X4 | 미확정 사항 7 (CLI의 알 수 없는 key) | VERIFIED (해소) | V-10: `--max_prompt_length 512`가 accelerate launch 인자로 넘어가 `SystemExit 2` |
| X5 | 미확정 사항 5 (비전 LoRA optimizer state) | VERIFIED (CPU, 해소) | V-19: fused AdamW에서 비전 LoRA 12개는 state가 0개, text 30개는 모두 state가 있었다 |
| X6 | (신규) 비양자화 LoRA의 adapter dtype | VERIFIED | V-17: bf16 base에서 trainable 파라미터는 `float32`, frozen 파라미터는 `bfloat16`이었다. ZeRO-3 예외는 소스로 확인했다(sft_trainer.py:1131-1134) |
| X7 | §11.2 parity 테스트 코드를 그대로 실행 | VERIFIED | 문서의 코드 블록을 그대로 추출해 실행했다(V-22): `6 passed in 5.98s`. torch 없는 venv에서는 수집 오류 없이 `1 skipped` |
| X8 | 구현 시사점 B-1/B-2 YAML의 key가 유효한가 | VERIFIED | V-20: placeholder를 치환한 뒤 SFT·DPO parser를 `fail_with_unknown_args=True`로 통과했고, CLI `config_remaining=[]`이었다 |
| X9 | `train_sampling_strategy`, `every_n_layers`/`offload`가 "5.18 신규"라는 서술 | UNVERIFIABLE | 설치본에는 이전 버전 이력이 없다. 문구를 "도입 버전 미확인"으로 낮췄다 |
| X10 | §7.3 row 2355 절단 756/756, §8.1 rows 0–1 `(4, 555)` | VERIFIED | V-18 |

### 검증 실험 목록

| ID | 스크립트 | 주요 출력 |
|---|---|---|
| V-01 | `v01_configs.py` | 기본값 덤프(§1.1, §6), 제거 필드 14개 `TypeError`, `ModelConfig().dtype='float32'` |
| V-02 | `v02_build_tiny.py --with-processor` | `tiny_vlm`(VLM 32,018,068 params + processor), `tiny_lm`(`Qwen3_5ForCausalLM` + tokenizer) |
| V-03 / V-03b | `v03_golden_independent.py` / `v03b_golden_notorch.py` | golden 값 일체. torch 미설치 venv에서도 같은 값 |
| V-04 | `v04_trl_prep_parity.py {dpo,sft_msg,sft_msg_aol,sft_pc} {processor,tokenizer}` | 8개 조합 모두 4,656→4,656, 독립 재구현과 불일치 0 |
| V-05 | `v05_autoprocessor_deps.py` (3개 venv) | `ValueError` / `ImportError`(torchvision) / `Qwen3VLProcessor` |
| V-06 | `v06_truncation_packing.py` | SFT·DPO 기본값 절단·삭제, packing 3전략, `ValueError` 2종, DPO `padding_free` 강제 off |
| V-07 | `v07_train_smoke.py {sft,sft_nll,dpo,dpo_lora,dpo_pre} {bfloat16,default}` | logits dtype·shape, `logits=None`, policy logits 생존, precompute dtype |
| V-08 | `v08_incompat_lora.py` | 비호환 오류 메시지, `all-linear` 대상 목록, `"ref"` adapter |
| V-09 | `v09_misc.py` | 마커 제거 템플릿 → `ValueError`, 매핑 없는 CyberNative, `extract_prompt` 경계 사례 |
| V-10 | `v10_cli.py` + `v10_cli.yaml` | 스크립트 경로 무시, `fail_with_unknown_args=True` 오류, CLI 경로 `SystemExit 2` |
| V-11 | `v11_coverage.py` | sampler, drop_last, GA 나머지 처리 |
| V-12 | `v12_ref_variants.py` | ref dtype = policy dtype, ref는 `__init__`에서 wrap, `ref_model`을 직접 넘기면 precompute fp32 |
| V-13 | `v13_hash_module_rss.py` | peak RSS +2.00 × fp32 크기 |
| V-14 | `v14b_chunk_rss.py`, `v14c_…`, `v14d_…`, `v14e_chunk_phases.py` | chunked CE peak 15.0–16.5 B/(256·V). entropy graph를 유지하면 chunk당 +8 B |
| V-15 | `v15_pack_batches.py` | packing bin이 map 배치(1,000 rows) 단위 |
| V-16 | `v16_gc.py` | GC 켜지는 시점과 kwargs |
| V-17 | `v17_lora_dtype.py` | 비양자화 LoRA adapter `float32` |
| V-18 | `v18_rows.py` | row 2355/3169/0/1 길이와 1024 절단 결과 |
| V-19 | `v19_vision_lora_optstate.py` | 비전 LoRA optimizer state 0 |
| V-20 | `yaml/check.py` + `yaml/{sft,dpo}.yaml` | B-1/B-2 YAML strict parse 통과 |
| V-21 | `v21_loss_kwargs_lm.py` | `Qwen3_5ForCausalLM`에서 SFT `model_accepts_loss_kwargs=True`, `num_items_in_batch=tensor(22)`. DPO `False` |
| V-22 | `parity/test_trl_parity_doc.py` (§11.2를 그대로 추출) | `6 passed in 5.98s`, torch 없는 venv에서 `1 skipped` |
