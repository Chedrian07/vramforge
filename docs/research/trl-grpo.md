# TRL 1.14.1 GRPOTrainer 조사 — 설정 기본값, 배치 제약, sampler coverage, 생성·logprob 메모리 경로

| 항목 | 값 |
|---|---|
| 주제 | TRL 1.14.1 `GRPOConfig`/`GRPOTrainer`의 설정 기본값, batch·generation 검증 규칙, `RepeatSampler` coverage, Transformers 생성(rollout) 경로, old/ref log-prob 계산, loss 메모리, reward·reference 모델, optimizer step 단위의 phase 타임라인 |
| 문서 | `docs/research/trl-grpo.md` |
| 작성일 | 2026-10-04 |
| 작성 | `research-trl-grpo` (Milestone M0) |
| 대상 환경 (pinned) | torch==2.14.1, transformers==5.18.0, trl==1.14.1, peft==0.21.2, bitsandbytes==0.50.2, accelerate==1.15.0, datasets==5.0.1, huggingface_hub==1.33.0, tokenizers==0.23.2, safetensors==0.8.0 |
| 예시 모델 | `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` @ `2367e865d009c13ac81713a2878291d33ab28177` (config·tokenizer·processor 파일만 사용, 가중치 미다운로드) |
| 예시 데이터 | `CyberNative/Code_Vulnerability_Security_DPO` @ `81aeacf06cf43b16d7278a3a01f019a496a53c51`, split `train`, 4,656 rows |
| 방법 | ① 설치된 site-packages 소스 정독 ② macOS arm64 CPU 실험 E1–E5, E7(부록 A): 실제 MiMo processor/tokenizer와 로컬에서 만든 tiny random-init `Qwen3_5ForConditionalGeneration`(linear_attention 3 + full_attention 1, vocab 248,320 유지)으로 **실제 `GRPOTrainer`를 학습**시키며 내부 상태를 probe ③ CUDA 전용 경로(TRL Triton logprob kernel)는 소스 기반 추론과, 같은 할당 패턴을 CPU에서 재현한 에뮬레이션의 live byte 측정으로 보완 |
| 실험 환경 | 공용 `/tmp/vf-research/.venv` (변경 없음) + overlay venv `/tmp/vf-research/venv-trl-grpo` (`.pth`로 공용 site-packages를 참조하고 Pillow 12.3.0, torchvision 0.29.1 `--no-deps`만 추가) |
| 실험 코드 | `/tmp/vf-research/scratch/trl-grpo/` (저장소 밖). 실험 목록은 부록 A |
| 증거 태그 | **VERIFIED** = 소스에서 확인했거나 실측함 / **INFERRED** = 소스 기반 추론(CUDA 전용 등, 이 머신에서 실행 불가) / **UNKNOWN** = 확인 불가 |
| 경로 표기 | 인용 경로는 site-packages 기준이다. `trl/…` = trl==1.14.1, `transformers/…` = transformers==5.18.0, `accelerate/…` = accelerate==1.15.0, `peft/…` = peft==0.21.2, `torch/…` = torch==2.14.1 |
| 검증 | 2026-10-04 `verify-trl-grpo`가 적대적으로 재검증했다(소스 재독 + 독립 실험 V1–V6). 본문의 **[검증 수정]**/**[검증 보강]** 표시가 바뀐 곳이다. 판정과 근거는 문서 끝 "검증 로그"에 있다 |

기호는 plan.md §8.3을 따른다. `G` = `num_generations`, `U` = generation 1회의 unique prompt 수(전체 process 합), `C` = device 하나에서 동시에 생성하는 sequence 수, `B_update` = `per_device_train_batch_size`, `K` = `gradient_accumulation_steps`, `spg` = `steps_per_generation`, `gbs` = `generation_batch_size`, `W` = process 수, `N` = 데이터셋 row 수, `P` = generation batch 안에서 패딩된 prompt 길이, `L` = generation batch 안에서 패딩된 completion 길이, `V` = vocab size(예시 248,320).

## 0. 핵심 결론

1. TRL 1.14.1 GRPO에는 **`max_prompt_length` 필드가 없다.** 넘기면 `TypeError`가 나고, 기본 설정의 프롬프트 토큰화 경로에는 truncation이 없다. VERIFIED (E5, §9, 검증 V1·V2). **[검증 수정]** "어떤 설정으로도 잘리지 않는다"는 과장이다. `chat_template_kwargs`가 `apply_chat_template(**chat_template_kwargs)`로 그대로 전달되므로 `chat_template_kwargs={"truncation": True, "max_length": N}`을 주면 프롬프트가 잘린다(검증 V4-trunc: 98-token 프롬프트가 생성 입력에서 16 token이 됨). VERIFIED
2. `RepeatSampler`는 epoch마다 **`N mod U`개 프롬프트를 버린다** (`U = gbs / G`). 예시 설정(G=4, gbs=4)은 U=1이라 4,656개를 모두 쓴다. VERIFIED (E2, E3c-H)
3. Rollout은 device의 generation batch 전체를 **`generate` 한 번**으로 처리한다. `C = B_update × spg = gbs / W`이고 예시는 C=4다. 같은 프롬프트의 G개 복사본은 각자 KV cache를 갖는다(prefix sharing 없음). VERIFIED (E3c)
4. 기본 정렬 설정(`spg = K`, `num_iterations = 1`, `beta = 0`, vLLM 끔)에서는 old/ref log-prob 패스가 **없다.** 조건은 `K % (spg × num_iterations) != 0` 또는 vLLM IS 보정(old)과 `beta != 0`(ref)이다. VERIFIED (E3c-A/B/D/E)
5. 학습 forward의 logits는 accelerate native AMP wrapper 때문에 **fp32 `(B_update, L+1, V)`**로 돌아온다(기본값 `bf16=True`, 단일 GPU·DDP). **[검증 수정]** "항상"은 아니다. accelerate는 bf16 native AMP를 DeepSpeed·Megatron에서 켜지 않고(`accelerate/accelerator.py:586-593`), mixed precision을 끄면 wrapper 자체가 없다. 이 경우 logits는 모델 dtype이다. CUDA fused kernel 경로에서 logits 관련 피크는 약 **8 B × B_update × (L+1) × V**다(CPU 할당 에뮬레이션 실측 8.2–9.2, L이 작을수록 다른 activation 비중 때문에 큼). old/ref no-grad 패스는 chunk가 2개 이상이면 약 **10 B × B_update × (L+1) × V**다. VERIFIED(에뮬레이션) / INFERRED(CUDA) (E4)
6. TRL은 문자열 모델을 **float32로 로드**한다. PEFT/QLoRA의 rollout은 accelerate autocast **밖에서** 실행되므로 이 기본값에서는 KV cache와 conv state도 fp32가 된다. 내보내는 설정에는 `model_init_kwargs={"dtype": "bfloat16"}`을 명시해야 한다. VERIFIED (E3c-C/G/Q/R, E7)
7. reward source가 없으면 `GRPOTrainer` 생성 자체가 `ValueError`다. memory-only 분석은 Trainer 객체 없이 해야 한다. VERIFIED (E5)
8. `gradient_checkpointing_kwargs`의 `every_n_layers`·`offload`는 GRPO에서 제대로 동작하지 않는다. TRL이 rollout 뒤 원본 dict를 `torch.utils.checkpoint.checkpoint` kwargs로 넘겨 checkpointing을 다시 켜기 때문이다. **[검증 수정]** 결과는 `use_reentrant`에 따라 다르다. 미지정(torch가 True로 간주)이거나 True면 첫 학습 forward에서 `ValueError: Unexpected keyword arguments: every_n_layers`(또는 `offload`)가 난다(E3c-J, 검증 V4-gcA/gcC). `use_reentrant=False`면 오류 없이 학습되지만 모든 layer가 checkpoint되고(`every_n_layers`→1) offload는 꺼진다(검증 V4-gcB/gcD). 어느 경우든 두 옵션은 효과가 없다. VERIFIED(CPU)
9. 예시 모델의 기본 processing class(`AutoProcessor` → `Qwen3VLProcessor`)는 Pillow와 torchvision을 요구한다. pinned 환경에는 둘 다 없어서 `processing_class`를 생략하면 생성이 실패한다. tokenizer를 직접 넘기면 4,656개 프롬프트 모두 같은 token id가 나온다. VERIFIED (E3a, E3b)

---

## 1. GRPOConfig 기본값 (Q1)

`GRPOConfig(output_dir=...)`를 기본값으로 만들고 `__post_init__` 이후 값을 출력했다(E1). `GRPOConfig`는 `_BaseConfig(TrainingArguments)`를 상속하고, TRL 공통으로 바꾼 기본값(`gradient_checkpointing=True`, `bf16=None→True`, `logging_steps=10`)을 갖는다 — `trl/trainer/base_config.py:53-107 (_BaseConfig)`. VERIFIED

| 필드 | 기본값 (`__post_init__` 이후) | 의미·주의 | 근거 |
|---|---|---|---|
| `num_generations` | 8 | G. 2 미만이면 오류 | `trl/trainer/grpo_config.py:479-485`, E1 |
| `num_generations_eval` | None → `num_generations` | 평가용 G | `grpo_config.py:486-492`, `trl/trainer/grpo_trainer.py:758` |
| `per_device_train_batch_size` | 8 (TrainingArguments) | B_update. GRPO에서 바꾸지 않음 | E1 |
| `gradient_accumulation_steps` | 1 (TrainingArguments) | K | E1 |
| `generation_batch_size` | None → `B_update × W × spg` (기본 8) | **전체 process 합** 단위 | `grpo_config.py:516-522, 1083-1103`, E1 |
| `steps_per_generation` | None → `K` (기본 1) | rollout 1회를 몇 micro-step에 나눠 쓰는지 | `grpo_config.py:523-526, 1085-1087` |
| `max_prompt_length` | **필드 없음** (`TypeError: GRPOConfig.__init__() got an unexpected keyword argument 'max_prompt_length'`) | 전용 truncation 기능이 없다. 단 `chat_template_kwargs`에 `truncation`/`max_length`를 넣으면 잘린다(§9) | E5, 검증 V1·V4-trunc |
| truncation side | 필드 없음. `processing_class`를 넘기지 않아 TRL이 직접 로드할 때만 `truncation_side="left", padding_side="left"`를 지정하고, truncation 호출은 없다 | 실효 없음 | `grpo_trainer.py:372-379, 1769-1846` |
| `max_completion_length` | 512 | `max_new_tokens`로 전달. `None`이면 무제한이 아니다(§9) | `grpo_config.py:493-496`, `grpo_trainer.py:1105-1106` |
| `num_iterations` | 1 | μ. 같은 rollout을 재사용하는 횟수 | `grpo_config.py:684-687` |
| `beta` | 0.0 | 0이면 reference를 만들지 않음 | `grpo_config.py:676-683`, `grpo_trainer.py:967-971` |
| `loss_type` | `"dapo"` | grpo, dr_grpo, dapo, bnpo, cispo, sapo, luspo, vespo | `grpo_config.py:796-830` |
| `scale_rewards` | `"group"` (`True`→`"group"`, `False`→`"none"`) | | `grpo_config.py:784-795, 1065` |
| `epsilon` / `epsilon_high` / `delta` | 0.2 / None → `epsilon` / None | | `grpo_config.py:688-708`, `grpo_trainer.py:934-935` |
| `importance_sampling_level` | `"token"` | | `grpo_config.py:755-764` |
| `temperature` / `top_p` / `top_k` / `min_p` / `repetition_penalty` | 1.0 / 1.0 / 0 / None / 1.0 | 모델 `generation_config.json`의 0.6 / 0.95 / 20을 **덮어쓴다**. transformers 5.18은 넘긴 config에서 `None`인 값만 모델 값으로 채우기 때문이다(`transformers/generation/utils.py:2085-2086`, `generation/configuration_utils.py:1364`) | `grpo_config.py:527-575`, `grpo_trainer.py:1105-1122`, E3c(`gen_cfg`), 검증 V4(MiMo `generation_config.json`을 넣은 tiny 모델에서 `_sample` 안의 실효값 1.0 / 0 / 1.0, eos 248046, logits processor 0개) |
| `use_vllm` / `vllm_mode` | False / `"colocate"` | | `grpo_config.py:582-598` |
| `vllm_gpu_memory_utilization` / `vllm_tensor_parallel_size` / `vllm_max_model_length` / `vllm_enable_sleep_mode` / `vllm_model_impl` | 0.3 / 1 / None / False / `"vllm"` | colocate 전용 | `grpo_config.py:599-673` |
| `use_transformers_paged` | False (deprecated, v2.0 제거 예정) | True면 `use_transformers_continuous_batching=True`로 바뀜 | `grpo_config.py:1030-1033, 1046-1053` |
| `use_transformers_continuous_batching` / `transformers_continuous_batching_config` | False / None | 켜면 TRL이 `max_memory_percent=0.5`를 기본으로 넣음 | `grpo_config.py:1017-1027`, `grpo_trainer.py:766-780` |
| `cache_implementation` | None | Transformers 생성에서 `DynamicCache` 사용 | `grpo_config.py:576-579`, E3c |
| `generation_kwargs` | None | `GenerationConfig` 인자를 덮어씀(충돌 시 우선) | `grpo_config.py:552-560`, `grpo_trainer.py:1118-1120` |
| `chat_template_kwargs` | None → `{}` | `apply_chat_template`에 전달. **[검증 보강]** `truncation`/`max_length` key는 tokenizer 인자로 들어가 프롬프트를 자른다(검증 V4-trunc) | `grpo_config.py:561-567`, `grpo_trainer.py:759` |
| `mask_truncated_completions` | False | 잘린 completion도 loss에 포함 | `grpo_config.py:831-838` |
| `shuffle_dataset` | True | `RepeatSampler(shuffle=True, seed=args.seed)` | `grpo_config.py:506-509`, `grpo_trainer.py:1260-1267` |
| `pad_to_multiple_of` | None | prompt·completion 패딩 배수(§6.4) | `grpo_config.py:510-513` |
| `gradient_checkpointing` | **True** (TRL override) | | `trl/trainer/base_config.py:61-66` |
| `gradient_checkpointing_kwargs` | None → transformers 기본 `{"use_reentrant": False}` | `every_n_layers`/`offload`는 GRPO에서 오류(`use_reentrant` 미지정/True)이거나 조용히 무시(`use_reentrant=False`)된다(§4.3) | `transformers/modeling_utils.py:3138-3139` |
| `model_init_kwargs` | None → **`dtype=float32`**, 단일 GPU면 `device_map="auto"`(MULTI_GPU·DeepSpeed면 TRL이 None으로 바꿈, `grpo_trainer.py:343-344`) | 문자열 모델에만 적용 | `trl/trainer/utils.py:1292-1305 (create_model_from_path)`, E3c-A, E7 |
| `disable_dropout` | False | | `grpo_config.py:452-458` |
| `cast_lm_head_to_fp32` | False | True면 lm_head 가중치 fp32 | `grpo_config.py:459-467` |
| `sync_ref_model` / `ref_model_mixup_alpha` / `ref_model_sync_steps` | False / 0.6 / 512 | §8 제약 | `grpo_config.py:839-860` |
| `vllm_importance_sampling_correction` / `_mode` / `_clip_max` / `_clip_min` | True / `"sequence_mask"` / 3.0 / None | **vLLM 사용 시에만** 적용 | `grpo_config.py:920-957`, `grpo_trainer.py:2695-2696, 2712` |
| `off_policy_mask_threshold` / `use_bias_correction_kl` | None / True | | `grpo_config.py:958-977` |
| `top_entropy_quantile` / `entropy_coef` / `use_adaptive_entropy` | 1.0 / 0.0 / False | entropy는 metric용으로 항상 계산(§6) | `grpo_config.py:861-911` |
| `log_completions` / `num_completions_to_print` / `log_unique_prompts` / `log_completions_hub_repo` / `log_multimodal` | False / None / False / None / True | host RAM의 deque(maxlen=gbs)에 텍스트 보관 | `grpo_config.py:980-1014`, `grpo_trainer.py:1050-1058` |
| `remove_unused_columns` | False (TRL override) | dataset의 다른 열이 reward 함수 kwargs로 전달됨 | `grpo_config.py:472-478` |
| `reward_weights` / `multi_objective_aggregation` | None → 모두 1.0 / `"sum_then_normalize"` | | `grpo_config.py:765-783`, `grpo_trainer.py:540-548` |
| `ds3_gather_for_generation` | True | ZeRO-3에서 생성 시 전체 파라미터 gather | `grpo_config.py:497-505` |
| `bf16` | None → True (fp16 미설정 시) | accelerate native AMP 활성화(§4.4, §6) | `trl/trainer/base_config.py:67-74, 104-105`, E1 |
| `learning_rate` / `logging_steps` / `num_train_epochs` / `max_steps` / `seed` | 1e-6 / 10 / 3.0 / -1 / 42 | | E1 |
| `optim` / `max_grad_norm` / `weight_decay` | `"adamw_torch_fused"` / 1.0 / 0.0 | transformers 5.18 기본 | E1 |
| `dataloader_drop_last` / `torch_empty_cache_steps` / `use_liger_kernel` / `auto_find_batch_size` | False / None / False / False(True 금지) | | E1 |

`num_generations` docstring은 "effective batch size(`W × B_update × K`)가 G로 나누어떨어져야 한다"고 적었지만, 실제 검사는 `generation_batch_size % G`다. `generation_batch_size`나 `steps_per_generation`을 직접 주면 K는 G와 무관하다(E1: G=4, spg=8, K=4 허용). VERIFIED — `grpo_config.py:64-66` vs `1118-1122`

## 2. batch·generation 검증 규칙 (Q2)

### 2.1 `GRPOConfig.__post_init__`

`trl/trainer/grpo_config.py:1083-1103`. `num_processes = self.world_size`. VERIFIED

```python
if self.generation_batch_size is None and self.steps_per_generation is None:
    self.steps_per_generation = self.gradient_accumulation_steps
    self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
elif self.generation_batch_size is not None and self.steps_per_generation is None:
    if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
        raise ValueError(...)  # "generation_batch_size (X) must be divisible by the global batch size (Y)."
    self.steps_per_generation = self.generation_batch_size // (self.per_device_train_batch_size * num_processes)
elif self.generation_batch_size is None and self.steps_per_generation is not None:
    self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
else:
    raise ValueError("'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time")
```

| # | 조건 | 결과·메시지 | 근거 |
|---|---|---|---|
| C1 | `use_transformers_paged=True` | `FutureWarning`, `use_transformers_continuous_batching=True`로 바꿈 | 1046-1053 |
| C2 | `parallelism_config`의 cp 또는 sp 사용 | `ValueError` ("GRPOTrainer does not support sequence-dim parallelism …") | 1055-1063 |
| C3 | `log_completions_hub_repo` 설정 + `log_completions=False` | `ValueError` | 1067-1071 |
| C4 | `auto_find_batch_size=True` | `ValueError` ("auto_find_batch_size is not supported by GRPO …") | 1076-1081 |
| C5 | gbs/spg 해석 (위 코드) | 둘 다 지정하면 `ValueError`(**상호 배타**). gbs만 주면 `gbs % (B_update × W) == 0` 필요 | 1083-1103 |
| C6 | `do_eval`이고 `eval_strategy != "no"` | `(per_device_eval_batch_size × W) % (num_generations_eval or G) != 0`이면 `ValueError` ("The global eval batch size (a * b) must be divisible by the number of generations used for evaluation (G).") | 1105-1114 |
| C7 | `gbs % G != 0` | `ValueError` ("generation_batch_size (X) must be divisible by num_generations (G).") | 1118-1122 |
| C8 | `G < 2` | `ValueError` ("GRPO requires at least 2 generations per prompt …") | 1124-1128 |
| C9 | `vllm_importance_sampling_cap` 설정 | deprecated, `clip_max`로 복사 | 1130-1137 |
| C10 | `clip_min >= clip_max` | `ValueError` | 1139-1147 |
| C11 | IS 보정 + `*_truncate` + clip 둘 다 None | `ValueError` | 1149-1158 |

정리한 공식 (VERIFIED, E1):

```text
spg = K                         (gbs, spg 둘 다 미지정)
spg = gbs / (B_update × W)      (gbs 지정, 정수여야 함)
gbs = B_update × W × spg        (spg 지정 또는 둘 다 미지정)
요구: gbs % G == 0, G >= 2
U = gbs / G                     (rollout 1회의 unique prompt 수, 전체 process 합)
C = B_update × spg = gbs / W    (device 하나의 rollout 행 수)
```

E1 결과(W=1):

| 입력 | 결과 |
|---|---|
| G=4, gbs=4, B=1, K=4 (예시) | OK, spg=4 |
| G=4, B=1, K=4 (gbs/spg 미지정) | OK, gbs=4, spg=4 |
| G=4, B=1, K=1 | `generation_batch_size (1) must be divisible by num_generations (4).` |
| G=4, spg=8, B=1, K=4 | OK, gbs=8 (K와 spg가 달라도 허용) |
| G=4, gbs=4, spg=4 | `'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time` |
| G=4, gbs=6, B=4 | `generation_batch_size (6) must be divisible by the global batch size (4).` |
| G=3, B=1, K=4 | `generation_batch_size (4) must be divisible by num_generations (3).` |
| G=1 | `GRPO requires at least 2 generations per prompt …` |
| G=4, gbs=16, B=2, K=4 | OK, spg=8 |

### 2.2 `GRPOTrainer.__init__` 검증 (`trl/trainer/grpo_trainer.py`)

| 조건 | 결과 | 근거 |
|---|---|---|
| `quantization_config`를 trainer 인자와 `model_init_kwargs`에 동시 지정 | `ValueError` | 335-341 |
| processing class가 tokenizer도 processor도 아님 | `TypeError` | 386-394 |
| `use_transformers_continuous_batching` + `ProcessorMixin` | `ValueError` ("does not support multimodal models") — **예시 모델의 기본 processor가 여기에 해당** | 381-384 |
| `peft_config`가 `PeftConfig`가 아님 / `PeftModel` + `peft_config` | `TypeError` / `ValueError` | 419-435 |
| `reward_weights` 길이 ≠ reward 함수 수 | `ValueError` | 540-546 |
| `reward_processing_classes` 길이 불일치 | `ValueError` | 555-559 |
| reward source 없음 | `ValueError` ("No reward source provided. …") | 705-711, E5 |
| `train_dataset=None`이고 environment 없음 | `ValueError` | 851-870 |
| IterableDataset + `dispatch_batches=True` | `ValueError`. iterable train set은 `dataloader_num_workers=0`으로 강제 | 876-903 |
| `loss_type="vespo"` + vLLM + IS mode가 token_* 아님 | `ValueError` | 925-930 |
| `use_liger_kernel` + lm_head LoRA / prompt-learning PEFT / liger 미설치 | `ValueError` / `ValueError` / `ImportError` | 808-832, 1032-1037 |
| `cast_lm_head_to_fp32` + lm_head adapter | `ValueError` | 999-1004 |
| `router_aux_loss_coef` 지정 + 비-MoE 모델 | `ValueError` | 794-804 |
| `sync_ref_model=True` + `beta=0` / + PEFT | `ValueError` / `NotImplementedError` | 1141-1157, E5 |

런타임 검증: generation 결과가 비면 `RuntimeError` (2335-2340), reward 함수가 반환한 개수가 틀리면 `ValueError` (1716-1721), `rollout_func`의 필수 key가 빠지면 `ValueError` (2269-2273). VERIFIED(소스)

상호 배타 관계: gbs ↔ spg(C5)만 오류로 막는다. `use_vllm`과 `use_transformers_continuous_batching`을 둘 다 켜면 오류 없이 vLLM이 우선한다(`grpo_trainer.py:1852-1869`의 if/elif). VERIFIED(소스)

## 3. Sampler coverage (Q3)

### 3.1 사용되는 sampler와 tail drop

map-style dataset이면 `RepeatSampler`를 쓴다. `trl/trainer/grpo_trainer.py:1231-1267 (_get_train_sampler)`, `trl/trainer/utils.py:819-912 (RepeatSampler)`. VERIFIED

```python
RepeatSampler(data_source=dataset, mini_repeat_count=self.num_generations,          # G
              batch_size=self.args.generation_batch_size // self.num_generations,   # U
              repeat_count=self.num_iterations * self.args.steps_per_generation,
              shuffle=self.shuffle_dataset, seed=self.args.seed)
# RepeatSampler.__iter__ (utils.py:890-909)
indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()   # shuffle=True
indexes = [indexes[i : i + self.batch_size] for i in range(0, len(indexes), self.batch_size)]
indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]          # <-- 꼬리 chunk 버림
for chunk in indexes:
    for _ in range(self.repeat_count):
        for index in chunk:
            for _ in range(self.mini_repeat_count):
                yield index
```

- 마지막 불완전 chunk를 버리므로 **epoch마다 `N mod U`개 프롬프트가 빠진다.** `__len__ = (N // U) × U × G × repeat_count` (utils.py:911-912). VERIFIED (E2, E3c-H)
- generator는 한 번만 seed되고 `__iter__`마다 새 `randperm`을 뽑는다. 그래서 `N mod U != 0`이면 epoch마다 **다른** 무작위 프롬프트가 빠진다(E2: 10개, U=3에서 1 epoch에는 7번, 2 epoch에는 2번이 빠짐). VERIFIED. **[검증 보강]** 정확히는 epoch마다 새로 뽑을 뿐이라 같은 row가 우연히 다시 빠질 수 있다(검증 V3: 4 epoch 누락 row 7, 2, 2, 6). `RepeatSampler`에는 `set_epoch`가 없어 Trainer의 `set_epoch` 호출(`transformers/trainer.py:1816-1823`)도 영향을 주지 않는다. VERIFIED
- DataLoader batch 크기는 `B_update × spg`다(`grpo_trainer.py:1223-1229`). micro-step마다 batch 하나를 소비하고, 생성은 `_step % (spg × num_iterations) == 0`일 때만 하며 나머지 batch(반복 복사본)는 버린다(`grpo_trainer.py:1621-1631`). VERIFIED
- 다중 process: accelerate `BatchSamplerShard`가 연속된 batch를 rank에 나눈다. process마다 같은 seed의 sampler를 만들어 같은 순서를 공유한다. `gbs`가 `B_update × W × spg`로 정의되므로 rank당 batch 수가 W로 나누어떨어지고 `even_batches` 보충(중복)은 생기지 않는다. VERIFIED (E2 시뮬레이션, W=2·3에서 누락 0)
- `IterableDataset`: `repeat_iterable_dataset`이 같은 순서를 stream으로 재현하고, 마지막 불완전 batch를 똑같이 버린다(utils.py:915-978의 `repeat_batch` 주석). 셔플은 buffered shuffle이다. VERIFIED(소스)
- 평가 sampler는 `RepeatSampler(eval_dataset, mini_repeat_count=G_eval, seed)`이고 batch_size=1이라 누락이 없다(`grpo_trainer.py:1269-1275`). VERIFIED(소스)

### 3.2 예시 설정 계산 (G=4, gbs=4, B_update=1, K=4, W=1)

- spg = gbs / (B_update × W) = 4, U = gbs / G = **1**, C = 4.
- `4656 mod 1 = 0` → **누락 0, epoch당 unique prompt 4,656개.** completion 생성 수는 epoch당 4,656 × 4 = 18,624개.
- process당 micro-step = `(N // U) × spg × num_iterations` = 4,656 × 4 × 1 = 18,624, optimizer step = 18,624 / 4 = **4,656 per epoch**. 기본 `num_train_epochs=3`이면 13,968 step(`transformers/trainer.py:2466-2475`).
- 근거: E2(실제 `RepeatSampler` + `BatchSampler` + accelerate `BatchSamplerShard`로 4,656 row 전체 시뮬레이션: unique 4,656, dropped 0, micro-step 18,624), E3c-A/I(실제 `GRPOTrainer` 학습에서 6행·7행 모두 정확히 한 번씩 생성). VERIFIED

E2 결과 (N=4,656):

| G | B_update | W | spg | iter | gbs | U | unique/epoch | dropped |
|---|---|---|---|---|---|---|---|---|
| 4 | 1 | 1 | 4 | 1 | 4 | 1 | 4,656 | 0 |
| 8 | 8 | 1 | 1 | 1 | 8 (TRL 기본) | 1 | 4,656 | 0 |
| 4 | 1 | 1 | 8 | 1 | 8 | 2 | 4,656 | 0 |
| 4 | 2 | 1 | 8 | 1 | 16 | 4 | 4,656 | 0 |
| 4 | 1 | 1 | 20 | 1 | 20 | 5 | 4,655 | 1 |
| 4 | 1 | 1 | 28 | 1 | 28 | 7 | 4,655 | 1 |
| 4 | 1 | 1 | 128 | 1 | 128 | 32 | 4,640 | 16 |
| 4 | 1 | 2 | 4 | 1 | 8 | 2 | 4,656 | 0 |
| 4 | 1 | 3 | 4 | 1 | 12 | 3 | 4,656 | 0 |
| 4 | 1 | 1 | 4 | 2 | 4 | 1 | 4,656 | 0 |

E3c-H(실제 Trainer, 7 rows, gbs=8, G=4 → U=2): 3번 생성해 6행만 쓰고 1행(이번 seed에서는 row 1)이 빠졌다. VERIFIED

### 3.3 전체 coverage가 되는 설정

```text
full_coverage_per_epoch  ⇔  N mod U == 0,  U = gbs / G
그리고 학습 길이가 최소 1 epoch: max_steps 미지정(epoch 기반, num_train_epochs >= 1)
                                또는 max_steps >= ceil((N // U) × spg × num_iterations / K)
```

N = 4,656 = 2⁴ × 3 × 97이므로 U ∈ {1, 2, 3, 4, 6, 8, 12, 16, 24, 48, 97, 194, 291, 388, 582, 776, 1164, 1552, 2328, 4656}이면 전체 coverage다. 그 밖의 U는 TRL에 꼬리를 살리는 옵션이 없다. 중복을 채워 넣는 방법은 학습 분포를 바꾸므로 plan.md §7.4에 따라 별도 표시해야 한다. VERIFIED(계산 + E2)

## 4. vLLM 없는 생성 경로 (Q4)

### 4.1 프롬프트 구성과 토큰화

- GRPO는 dataset을 미리 토큰화하지 않는다(`data_collator=identity`, `grpo_trainer.py:950-965`). `prompt` 열을 생성 시점에 `_tokenize_prompts`로 처리한다. VERIFIED
- conversational prompt면 `processing_class.apply_chat_template(conversation=prompts, tools=self.tools or None, chat_template=self.chat_template, add_generation_prompt=True, tokenize=True, return_dict=True, **chat_template_kwargs)`를 호출한다(`grpo_trainer.py:1822-1839`). VLM processor면 먼저 `prepare_multimodal_messages`로 문자열 content를 `[{"type": "text", …}]` block으로 바꾼다(1773-1774). VERIFIED
- tokenizer의 `apply_chat_template`은 `padding=False, truncation=False, max_length=None`이 기본이고, 토큰화할 때 `add_special_tokens=False`를 쓴다 — `transformers/tokenization_utils_base.py:2990-3001, 3123-3131`. VERIFIED
- 비대화형(plain string) prompt는 `processing_class(text=prompts)["input_ids"]`라 tokenizer 기본 `add_special_tokens`를 따른다(1842-1845). Qwen tokenizer는 BOS가 없어서 결과에 BOS가 없다(E3b: `"hello world"` → `[14556, 1814]`). VERIFIED
- 패딩: 생성 입력은 `pad(..., padding_side="left")`로 **왼쪽** 패딩된다(1896-1900). truncation은 없다. VERIFIED
- 예시 실측(E3a/E3b): `system` 열 4,656개가 전부 빈 문자열이다. system+user 매핑이면 최대 프롬프트 272 tokens(row 2355), user만 쓰면 268 tokens다. processor 경로와 tokenizer 경로의 token id는 4,656개 모두 같았다. processor 경로는 `mm_token_type_ids`(프롬프트와 같은 길이, int64)를 추가로 만들어 `generate`에 넘긴다. 텍스트 전용 학습 forward에는 넘기지 않는다(2584-2585). VERIFIED

### 4.2 동시 생성 수 C와 generate 호출

```python
# grpo_trainer.py:1928-1944 (regular path)
with (profiling_context(self, "transformers.generate"),
      unwrap_model_for_generation(self.model_wrapped, self.accelerator,
          gather_deepspeed3_params=self.args.ds3_gather_for_generation,
          generation_kwargs=self.generation_kwargs) as unwrapped_model,
      torch.no_grad(),
      self._dist.summon_full_params(self.model_wrapped, recurse=False)):
    prompt_completion_ids = unwrapped_model.generate(**generate_inputs, generation_config=self.generation_config)
prompt_length = generate_inputs["input_ids"].size(1)
completion_ids = prompt_completion_ids[:, prompt_length:]
```

- **C = device의 generation batch 행 수 = `B_update × spg`** (평가 시 `per_device_eval_batch_size`). sampler가 각 프롬프트를 G번 반복하므로 같은 프롬프트의 G개 행이 각자 prefill되고 KV cache를 따로 갖는다(`num_return_sequences` 미사용, prefix sharing 없음). VERIFIED (E3c-A: 4행 `[48,48,48,48]`, cache batch 4; E3c-N: spg=8 → 8행)
- C를 C보다 작은 묶음으로 나누는 옵션은 Transformers 경로에 없다. plan.md §5.4의 `max_live_sequences`는 이 경로에서 C로 고정이다. VERIFIED(소스)
- `GenerationConfig` (1104-1122): `max_new_tokens=max_completion_length, do_sample=True, pad_token_id=tokenizer.pad, bos_token_id, eos_token_id=tokenizer.eos_token_id(단일 값), temperature, top_p, top_k, min_p, repetition_penalty, cache_implementation`과 `generation_kwargs`, `disable_compile=True`. 예시 tokenizer는 eos=`<|im_end|>`(248046), pad=`<|endoftext|>`(248044)라 모델 generation_config의 eos 목록 `[248046, 248044]` 대신 248046만 쓴다. VERIFIED (E3c `gen_cfg`)
- 생성 뒤 첫 EOS(포함)까지만 남기고 나머지를 마스킹해 Python list로 옮긴다(1946-1954). 이후 다시 GPU tensor로 패딩한다(§5.4).
- prefill: transformers가 `logits_to_keep=1`을 넣어 prefill logits는 `(C, 1, V)`다(`transformers/generation/utils.py:2920-2924`). decode마다 `outputs.logits[:, -1].to(copy=True, dtype=float32)`로 `(C, V)` fp32 복사본을 만든다(`generation/utils.py:3216`). 기본 sampling 설정(top_k=0, top_p=1.0, temperature=1.0)이면 추가 warper 텐서가 거의 없다. VERIFIED (E3c: prefill `logits_to_keep=1`, logits `(4,1,248320)`)
- forward 횟수: `max_new_tokens=8`이면 prefill 1회 + decode 7회 = 8회이고 최종 KV 길이는 `P + L − 1`이다(E3c-A: P=48, L=8 → K/V 길이 55). 끝난 sequence도 batch에서 빠지지 않고 pad를 생성하며 계속 돈다(`transformers/generation/utils.py:3160, 3198, 3251-3257`). 따라서 cache 길이는 batch에서 가장 늦게 끝나는 sequence가 정한다. VERIFIED (검증 V4: P=98, `max_new_tokens=6` → DynamicLayer K 길이 103 = P + L − 1). **[검증 보강]** 정지 판정을 한 step 늦추고 마지막 forward를 되돌리는 `DeferredStopCheck`는 MPS 전용이다(`transformers/generation/utils.py:432-441 (DeferredStopCheck.is_supported)`). 그래서 CUDA·CPU에는 추가 forward가 없고 `P + L − 1`이 그대로 맞다. VERIFIED(소스)

### 4.3 gradient checkpointing·train mode·unwrap

- 생성 동안 gradient checkpointing을 끄고(`trl/models/utils.py:133-135`), 끝나면 **인자 없이** `gradient_checkpointing_enable()`로 다시 켠다(150-151). transformers 5.18에서 인자 없음 = `{"use_reentrant": False}`, `every_n_layers=1`, `offload=False`다. VERIFIED (E3c: prefill 시 `gc=False`, 직후 `gc_after=True`)
- 이어서 `_generate_and_score_completions`가 old/ref 계산이 없어도 **항상** `with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs)` 블록에 들어간다(2686). 블록을 나올 때 `model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)`로 **원본 dict**를 그대로 넘긴다(`trl/models/utils.py:388-405`). Trainer는 처음에 `every_n_layers`/`offload`를 dict에서 빼서 쓰지만(`transformers/trainer.py:1490-1501`), 이 재활성화는 빼지 않으므로 그 key가 `torch.utils.checkpoint.checkpoint`로 들어간다. E3c-J: `gradient_checkpointing_kwargs={"every_n_layers": 2}` → 첫 학습 forward에서 `ValueError: Unexpected keyword arguments: every_n_layers`. `use_reentrant=True`는 정상 유지된다(E3c-I). VERIFIED.
- **[검증 수정]** 이 `ValueError`는 torch `_checkpoint_impl`이 reentrant 경로에서만 extra kwargs를 거부해서 생긴다. `use_reentrant`가 None이면 경고 뒤 True로 바꾸고, `kwargs and use_reentrant`면 raise한다(`torch/utils/checkpoint.py:635-649`). decoder layer는 자기 인자를 `partial(super().__call__, **kwargs)`로 묶으므로(`transformers/modeling_layers.py:81-110 (GradientCheckpointingLayer.__call__)`) `checkpoint`의 kwargs에는 GC dict의 extra key만 남는다. 실측(검증 V4, tiny Qwen3.5, 실제 `GRPOTrainer.train()`):

| `gradient_checkpointing_kwargs` | 결과 | 첫 학습 forward 시점의 layer 상태 |
|---|---|---|
| `{"every_n_layers": 2}` | `ValueError: Unexpected keyword arguments: every_n_layers` | 4/4 layer checkpoint, partial kwargs `{every_n_layers: 2}` |
| `{"offload": True}` | `ValueError: Unexpected keyword arguments: offload` (원 문서의 INFERRED를 확인) | 4/4 layer checkpoint |
| `{"use_reentrant": False, "every_n_layers": 2}` | 오류 없이 2 step 학습 | 4/4 layer checkpoint(**every_n_layers 무시**). extra key는 layer 호출 kwargs로 흘러가 버려진다 |
| `{"use_reentrant": False, "offload": True}` | 오류 없이 2 step 학습 | 4/4 layer checkpoint, checkpoint 함수가 offload wrapper가 아닌 plain `checkpoint`(**offload 꺼짐**) |

  rollout의 `_unwrap_model_for_generation`이 먼저 인자 없이 다시 켜고(`every_n_layers=1`, `offload=False`), 이어서 `disable_gradient_checkpointing`이 원본 dict를 `gradient_checkpointing_kwargs` 자리에만 넘긴다. 그래서 첫 학습 forward 전에 이미 두 옵션이 사라진다. 메모리 추정에서 두 옵션의 절감 효과를 반영하면 안 된다. VERIFIED(CPU)
- 모델은 생성 중에도 `train()` 상태다(E3c: prefill 시 `training=True`). dropout 확률이 0이 아니면 rollout에도 dropout이 걸린다(LoRA dropout 포함). VERIFIED
- `unwrap_model_for_generation`: DDP는 `accelerator.unwrap_model`만 한다. DeepSpeed ZeRO-3 + `ds3_gather_for_generation=True`는 **모든 파라미터를 `GatheredParameters`로 모아** 생성한다. FSDP v1은 `summon_full_params(recurse=False)`를 쓰고 FSDP v2는 no-op이다(`trl/models/utils.py:106-152`, `trl/distributed.py:74-91`). VERIFIED(소스) / 다중 GPU 메모리 영향은 INFERRED

### 4.4 생성 시 dtype: PEFT 여부에 따라 autocast가 다르다

- accelerate는 native AMP일 때 `prepare_model`에서 **`model.forward`를 autocast로 감싸고 출력을 fp32로 바꾸는 wrapper로 교체**한다(`accelerate/accelerator.py:1824-1835`, `accelerate/utils/operations.py:889-948 ConvertOutputsToFp32`). **[검증 보강]** native AMP는 `bf16=True`(또는 `fp16=True`)이고 DeepSpeed·Megatron이 아닐 때만 켜진다(`accelerate/accelerator.py:564-593`). 아래 dtype 결론은 모두 이 전제(1차 범위인 단일 GPU 기본 설정)에서만 성립한다. Trainer 자체는 autocast를 걸지 않는다(`transformers/trainer.py:2178-2183`). ref·reward 모델도 `prepare_model(evaluation_mode=True)`로 같은 wrapper를 받는다(wrapper 적용이 evaluation_mode 분기보다 앞, `accelerator.py:1824-1882`). VERIFIED (E3c-B: ref `_original_forward` 존재, E5: reward 모델도)
- 비-PEFT 정책: `generate`가 `self(...)`를 부르면 감싼 forward가 실행되어 rollout도 bf16 autocast 아래에서 돈다. VERIFIED (E3c-A: conv state bf16)
- PEFT 정책: `PeftModel.generate` → 내부 transformers 모델의 `generate` → **감싸지 않은 forward**라 autocast가 없다(`peft/peft_model.py:1009-1012, 2229-2250`). VERIFIED (E3c-C: fp32 로드 + LoRA에서 conv state fp32, E3c-G: bf16 로드면 prefill logits bf16)
- 결과 cache dtype (E3c, E7, 모두 CPU 실측):

| 경우 | 로드 dtype | full-attn K/V | linear-attn conv state | recurrent state | prefill logits |
|---|---|---|---|---|---|
| A: full FT, `model_init_kwargs` 없음 | fp32 | **fp32** | bf16 | fp32 | fp32 |
| F: full FT, `dtype=bfloat16` | bf16 | bf16 | bf16 | fp32 | fp32 |
| C: LoRA, dtype 없음 | fp32 | fp32 | fp32 | fp32 | fp32 |
| G: LoRA, `dtype=bfloat16` | bf16 | bf16 | bf16 | fp32 | bf16 |
| Q: QLoRA(nf4, compute bf16), dtype 없음 | 비양자화 모듈 fp32 | **fp32** | **fp32** | fp32 | fp32 |
| R: QLoRA, `dtype=bfloat16` | 비양자화 모듈 bf16 | bf16 | bf16 | fp32 | bf16 |

A에서 K/V가 fp32인 이유: rotary embedding이 `cos.to(dtype=x.dtype)`로 hidden state(fp32 embedding 출력) dtype을 따르므로, autocast로 bf16이 된 k와 곱할 때 type promotion으로 K가 fp32가 된다(`transformers/models/qwen3_5/modeling_qwen3_5.py:201, 703-707`; elementwise 곱은 autocast가 cast하지 않는다). V는 RoPE를 거치지 않지만 `DynamicLayer.lazy_initialization`이 K/V 빈 cache를 **K의 dtype**으로 만들고 `torch.cat`이 promotion하므로 fp32가 된다(`transformers/cache_utils.py:123-127, 144-147`). Q에서는 `Linear4bit`가 출력을 입력 dtype(fp32)으로 되돌린다. 관측은 VERIFIED(CPU), 원인 설명과 CUDA 동작은 INFERRED

### 4.5 Qwen3.5가 쓰는 cache class

- `cache_implementation=None` → `DynamicCache(config=text_config)`(`transformers/generation/utils.py:2298-2300`). `layer_types`에 따라 `full_attention` → `DynamicLayer`, `linear_attention` → `LinearAttentionLayer`(`transformers/cache_utils.py:1250-1258, 1807-1820`). VERIFIED (E3c: `LinearAttentionLayer`×3 + `DynamicLayer`)
- `DynamicLayer`: K/V `(C, n_kv, T, head_dim)`이고 decode마다 `torch.cat`으로 늘어난다(`cache_utils.py:144-147`). 한 층의 K(또는 V)를 cat하는 순간 old+new 두 벌이 잠깐 공존한다. VERIFIED(소스)
- `LinearAttentionLayer`: conv state `(C, conv_dim, kernel)`은 생성 forward의 dtype, recurrent state `(C, n_v_heads, d_k, d_v)`는 torch fallback(`torch_chunk_gated_delta_rule`이 fp32로 계산)에서 fp32다(`transformers/models/qwen3_5/modeling_qwen3_5.py:300-433 (torch_chunk_gated_delta_rule; fp32 cast 338, state 406), 573-653`; `cache_utils.py:1029-1117`). VERIFIED (E3c: conv `(4,128,4)`, rec `(4,4,16,16) float32`)
- pinned 환경에는 `fla`, `causal_conv1d`, `kernels` 패키지가 없어서 CUDA에서도 torch fallback이 쓰인다(`transformers/integrations/hub_kernels.py:984-1014`의 우선순위: hub kernel(요청 시) → 원 패키지 → torch). VERIFIED(패키지 부재 확인) / CUDA 동작 INFERRED
- 다른 선택지: `"static"` → `StaticCache`(full-attn을 `max_length`까지 미리 할당), `"offloaded"` → layer offload `DynamicCache`, `"quantized"` → full_attention 외 layer type이 있으면 `ValueError`라 **Qwen3.5에서 사용 불가**(`cache_utils.py:1956-1966`). VERIFIED(소스)
- 예시 모델(n_full=8, n_kv=4, head_dim=256, n_lin=24, conv_dim = 2×16×128 + 32×128 = 8,192, kernel 4, n_v=32, d_k=d_v=128) 공식:

```text
KV_full   = 2 × 8 × 4 × 256 × b_kv × C × T_kv          = 16,384 × b_kv × C × T_kv  bytes,  T_kv = P + L − 1
LinState  = 24 × C × (8,192 × 4 × b_conv + 32 × 128 × 128 × 4)
          = C × 49.5 MiB (b_conv=2) | C × 51.0 MiB (b_conv=4)
```

### 4.6 다른 생성 backend (1차 범위 밖)

- continuous batching(`use_transformers_continuous_batching`, deprecated `use_transformers_paged`): `generate_batch`를 쓰고, `max_memory_percent=0.5` 기본이라 **생성 시점의 free memory 50%를 KV pool로 예약**한다(`grpo_trainer.py:774-778`; `transformers/generation/continuous_batching/cache.py:622-629`). 생성 전에 `unwrapped_model.to(torch.bfloat16)`를 부르므로(1879-1882) fp32 정책 가중치가 bf16으로 바뀐다. processor(VLM)와 함께 쓰면 오류다. shape로 정해지지 않는 예약형 메모리라 1차 지원 대상에서 뺀다. VERIFIED(소스)
- vLLM colocate: `model.name_or_path`로 **별도 vLLM 엔진이 가중치를 다시 로드**한다(정책에 `Linear4bit`가 있으면 `quantization="bitsandbytes"`). `gpu_memory_utilization=0.3`, `max_num_seqs = B_update × vllm_tensor_parallel_size × spg`, `max_num_batched_tokens=4096`이고 IS 보정 때문에 old log-prob 패스가 항상 생긴다(`grpo_trainer.py:1068-1103, 2695-2696`; `trl/generation/vllm_generation.py:337-368`). VERIFIED(소스) / 메모리량 UNKNOWN

## 5. old/ref log-prob 계산 (Q5)

### 5.1 언제 계산하나

```python
# grpo_trainer.py:2686-2709 (요약)
with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
    generate_every = self.args.steps_per_generation * self.num_iterations
    if self.args.gradient_accumulation_steps % generate_every != 0 or (
        self.use_vllm and self.vllm_importance_sampling_correction):
        old_per_token_logps, _, _ = self._get_per_token_logps_and_entropies(
            self.model, prompt_completion_ids, attention_mask, logits_to_keep, batch_size=batch_size, ...)
    else:
        old_per_token_logps = None
    ...
    if self.beta != 0.0:   # ref_model 있으면 ref_model로, PEFT면 use_adapter(model, "ref" 또는 None)로
```

| 패스 | 조건 | 사용 모델 | 근거 |
|---|---|---|---|
| old | `K % (spg × num_iterations) != 0` **또는** (`use_vllm` and IS 보정) | 현재 정책 | 2694-2709 |
| ref | `beta != 0` | 비-PEFT: 별도 `ref_model`. PEFT: 정책을 `use_adapter(model, "ref" if "ref" in peft_config else None)`로 (None이면 `disable_adapter()`) | 2761-2791, `trl/trainer/utils.py:1365-1402` |

old를 계산하지 않으면 `_compute_loss`가 `per_token_logps.detach()`를 old로 쓴다(3126-3127). E3c 확인: A(정렬, beta=0) old·ref 없음, B(beta=0.04) ref 6회(`ref_model`), C(LoRA, beta=0.04) ref 6회(정책), D(`num_iterations=2`) old 6회, E(spg=8, K=4) old 3회. VERIFIED

### 5.2 함수, chunking, logits_to_keep

- `_get_per_token_logps_and_entropies` → `use_liger_kernel=False`(기본)면 `_full_logits_logps`(1369-1386, 1481-1593). VERIFIED
- 행 chunking: `batch_size = per_device_train_batch_size`(평가는 `per_device_eval_batch_size`)(2557). device의 generation batch C행을 B_update행씩 forward한다. 학습 loss에서는 `batch_size=None`이라 micro-batch 전체를 한 번에 처리한다. VERIFIED (E3c-B: ref 패스 `batch_size=1`, 입력 `(4,56)` → forward 4회 `(1,9,V)`)
- `logits_to_keep = completion_ids.size(1) = L`(2556)이고 모델에는 `L + 1`을 넘긴 뒤 마지막 위치를 버린다(1551-1569). Qwen3.5는 `logits_to_keep`을 지원하므로 lm_head는 `(rows, L+1, H)`에만 적용된다(`modeling_qwen3_5.py:1878-1879`). VERIFIED (E3c: `lkeep_kw=9`, logits `(1,9,248320)`)
- 각 forward에 `use_cache=False`를 넘긴다(1556). VERIFIED(소스)

### 5.3 temperature, dtype, 커널

- 반환 logits는 accelerate native AMP wrapper 때문에 **fp32**다(DeepSpeed·Megatron이거나 mixed precision을 끄면 모델 dtype, §4.4 [검증 보강]). lm_head 출력(autocast bf16)의 fp32 복사본이 만들어진다(§4.4). VERIFIED (E3c-C/G/Q/R: 내부 모듈 출력 bf16, wrapper 출력 fp32)
- `selective_log_softmax(logits, completion_ids, temperature, row_mask)`: CUDA/XPU이고 dtype과 stride 조건을 만족하면 TRL Triton kernel을 쓴다(`trl/trainer/utils.py:500-532, 535-598`). kernel 안에서 `logits / TEMPERATURE`를 계산하므로 **logits 사본을 만들지 않는다**("Scale inside the kernel", `grpo_trainer.py:1571`; `trl/kernels/logprob_entropy.py:64`). Triton이 없으면(fallback) `temperature != 1.0`일 때만 `logits / temperature` 사본(+4 B/elem)이 생긴다(utils.py:573-574). VERIFIED(소스) / CUDA 실행 INFERRED
- 출력: `(rows, L)` fp32.

### 5.4 측정한 메모리와 버퍼 수명

- no-grad 패스의 live byte(E4b, kernel 할당 에뮬레이션, bf16 가중치, rows=4, chunk=1, L=1024): **peak = 10.13 B × (L+1) × V**. chunk *i*의 fp32 logits(4 B)를 Python 변수 `outputs`/`logits`가 잡고 있는 동안 chunk *i+1*의 bf16 logits(2 B)와 fp32 변환본(4 B)이 생기기 때문이다. chunk가 1개면 약 6 B다. VERIFIED(CPU) / CUDA INFERRED
- 결과물(`old_per_token_logps`, `ref_per_token_logps`, vLLM이면 `sampling_per_token_logps`, `importance_sampling_ratio`)과 prompt/completion id·mask, `advantages`는 모두 **accelerator device(GPU)**의 tensor다(`.to(device)`, 2495-2530). `_prepare_inputs`가 셔플(`shuffle_sequence_dict`, 행 인덱싱 복사)하고 spg개 view로 나눠 `self._buffered_inputs`에 둔다(1621-1631; utils.py:1017-1081). **다음 생성이 `_buffered_inputs`를 바꿀 때까지**, 즉 `spg × num_iterations` micro-step 동안 산다. 다음 생성 동안에는 이전 버퍼가 아직 살아 있다. 크기는 `C × (P × 16 + L × (16 + 4 × n_logp)) + C × 4` bytes 정도로 작다. VERIFIED(소스, E3c `buffered=4/8`)

## 6. Loss 계산 메모리 (Q6)

### 6.1 경로

`_compute_loss`(3081-3351): `input_ids = cat(prompt_ids, completion_ids)` `(B, P+L)` → `_get_per_token_logps_and_entropies(..., compute_entropy=True, batch_size=None)` → `selective_log_softmax_and_entropy(logits, completion_ids, entropy_requires_grad=self._entropy_bonus_enabled, temperature, row_mask)`. entropy는 `entropy_coef=0`이어도 metric용으로 항상 계산된다(3091-3108, 3331). VERIFIED

### 6.2 CUDA fused kernel의 할당

```python
# trl/kernels/logprob_entropy.py (발췌)
logprobs = torch.empty(logits.shape[:-1], device=logits.device, dtype=torch.float32)   # :165 (+entropy, log_z, expected_logit)
ctx.save_for_backward(logits, index, log_z, expected_logit, row_mask)                   # :192  logits 자체를 저장
grad_logits = torch.empty_like(logits, memory_format=torch.contiguous_format)           # :210  backward에서 logits와 같은 shape·dtype
```

조건(`trl/trainer/utils.py:507-532`): device cuda/xpu, logits fp16/bf16/fp32, `stride(-1)==1`, index·row_mask가 `(B, L)`이고 int32/int64(bool 허용). GRPO의 sliced fp32 logits view와 int64 mask가 모두 만족한다. Linux CUDA torch는 triton을 함께 설치하므로 이 경로가 기본이다. VERIFIED(소스) / 실행 INFERRED

### 6.3 측정 (E4, CPU, live-storage tracker)

TRL 소스와 같은 할당(출력 fp32 `(B,L)` 4개, logits 저장, backward의 `empty_like(logits)`)을 하는 autograd Function으로 `_fused_logprob_entropy`를 바꿔 실제 `GRPOTrainer.compute_loss` + `backward`를 실행했다. 값은 `B_update × (L+1) × V` 원소당 byte다. 작은 모델이라 나머지 activation은 원소당 0.3–0.7 B 정도만 더해진다.

| 경로 | 가중치 | B, L | forward peak | forward 끝(=backward로 넘기는 양) | backward peak − grads |
|---|---|---|---|---|---|
| fused 에뮬레이션 | bf16 | 1, 256 | 6.73 | 4.73 | 9.21 |
| fused 에뮬레이션 | bf16 | 1, 1024 | 6.33 | 4.33 | 8.45 |
| fused 에뮬레이션 | fp32 | 1, 1024 | 6.34 | 4.34 | 8.21 |
| fused 에뮬레이션 + entropy bonus | bf16 | 1, 1024 | 6.33 | 4.33 | 8.51 |
| fused 에뮬레이션 | bf16 | 2, 512 | 6.34 | 4.34 | 8.45 |
| **fallback(실제 CPU 경로)** | bf16 | 1, 256 | 10.71 | 4.73 | 15.68 |
| **fallback(실제 CPU 경로)** | bf16 | 1, 1024 | 6.33 | 4.33 | 16.07 |

해석 (VERIFIED(CPU 측정) / CUDA INFERRED):
- forward: lm_head bf16 출력(2 B) + accelerate fp32 변환본(4 B) = **6 B** 순간 peak. 변환 뒤 bf16 원본은 해제되고 fp32 logits(**4 B**)가 backward까지 남는다.
- backward: 저장된 fp32 logits(4 B) + `grad_logits` fp32(4 B) = **8 B**. 이어지는 두 번의 slice backward(`[:, -L:]`, `[:, :-1]`)는 zero 텐서를 새로 만들고 bf16 cast backward는 2 B를 만들지만, 순서대로 실행되므로 peak는 8 B 근처다.
- B와 L에 선형이다(B=2, L=512와 B=1, L=1024가 같은 값).
- fallback(Triton 없음): `logsumexp`/gather backward 때문에 backward가 약 16 B다. no-grad entropy 계산(`entropy_from_logits`, 128행 chunk)의 고정 임시값 때문에 L이 작을 때 forward 비율이 커진다(`utils.py:601-640`).

### 6.4 shape (B = B_update)

| tensor | shape | dtype | 수명 |
|---|---|---|---|
| `input_ids`, `attention_mask` | `(B, P+L)` | int64 | micro-step |
| backbone activation | `(B, P+L, H)` 기준, 구조·checkpointing 의존 | 계산 dtype | forward→backward (구조 조사 범위) |
| lm_head 출력 | `(B, L+1, V)` | bf16(autocast) | forward 순간 |
| 반환 logits | `(B, L+1, V)` | **fp32** | forward → kernel backward |
| `grad_logits` 등 | `(B, L, V)` → `(B, L+1, V)` | fp32 → bf16 | backward 순간 |
| per-token tensor(`per_token_logps`, `entropies`, `log_z`, `expected_logit`, `old`, `coef_1/2`, `per_token_loss*`, `per_token_kl`) | `(B, L)` | fp32 | micro-step, 무시할 크기 |

- P와 L은 **device generation batch 전체**의 최댓값이다. 생성 결과를 batch 전체 최대 길이로 패딩한 뒤 spg개로 나누기 때문이다(2492-2514, 1629). E3c-N: 98-token과 42-token 프롬프트가 섞인 batch에서 micro-batch 폭이 모두 98이었다. VERIFIED
- `pad_to_multiple_of=m`이면 P와 L이 둘 다 m의 배수로 올라간다(E3c-M: P 98→128, L 8→64, logits `(B, 65, V)`). VERIFIED
- TRL chunked 경로(`use_liger_kernel=True`)는 `[N, V]` logits를 만들지 않지만(`trl/trainer/utils.py:1460-1664`, 2,048 token × 8,192 vocab tile) `liger_kernel` 패키지를 요구한다. pinned 환경에는 없어서 `ImportError`다(`grpo_trainer.py:1032-1037`). VERIFIED(소스)

## 7. Reward 함수 (Q7)

- reward가 문자열이면 `AutoModelForSequenceClassification.from_pretrained(id, num_labels=1, **model_init_kwargs)`로 로드한다. `model_init_kwargs`는 **정책과 같은 dict**이고 `trust_remote_code`가 추가된다(515-537). trainer 인자 `quantization_config`는 reward 모델에 적용되지 않는다. dtype이 없으면 transformers 5.18 기본 `"auto"`(config dtype)다(`transformers/modeling_utils.py:4106-4107`). 정책(TRL 기본 fp32)과 dtype이 다를 수 있다. VERIFIED (E5: dtype 미지정 → bf16 reward + fp32 정책, `dtype=float32` → fp32 reward)
- tokenizer: `AutoTokenizer.from_pretrained(reward model id)`. pad가 없으면 eos를 쓴다(561-578). VERIFIED
- 배치: `prepare_model(evaluation_mode=True, device_placement=True)`로 **accelerator device(GPU)에 학습 내내 상주**한다(1159-1167). DeepSpeed면 `prepare_deepspeed`다. forward wrapper(autocast + fp32 출력)를 받고 eval mode다. 파라미터의 `requires_grad`는 True로 남지만 `torch.inference_mode()`에서만 호출되고 optimizer에 없어서 grad 메모리는 없다. VERIFIED (E5)
- 계산(1690-1705): 대화형이면 reward tokenizer의 chat template로 `prompt + completion` 텍스트를 만들고 `padding=True, padding_side="right", add_special_tokens=False`, truncation 없이 토큰화해서 **device의 C행 전체를 한 번에** forward한다. 메모리는 reward 가중치(상주) + `C × T_reward` no-grad activation(순간)이다. VERIFIED(소스) / 수치 INFERRED
- callable reward는 host에서 실행되고 `(C, n_funcs)` 결과만 GPU로 간다(1708-1722). VERIFIED
- reward가 없으면 `ValueError("No reward source provided. Pass `reward_funcs`, or an `environment_factory` whose environment defines a `get_reward` method.")`(705-711). **memory-only 분석은 Trainer 객체를 만들 수 없다.** 실행용 config에는 reward가 필요하다(plan.md §8.4와 일치). VERIFIED (E5)

## 8. Reference 모델 (Q8)

| 경우 | 동작 | 메모리 | 근거 |
|---|---|---|---|
| `beta == 0` (기본) | `ref_model = None`, ref 패스 없음 | 없음 | 967-971, E3c-A |
| `beta != 0`, 비-PEFT | `create_model_from_path(config._name_or_path, **model_init_kwargs)`로 **두 번째 전체 모델**을 로드(trainer 인자 `quantization_config` 포함). `prepare_model(evaluation_mode=True)`로 GPU 상주, forward wrapper 적용 | 정책과 같은 dtype의 전체 가중치(TRL 기본 fp32!). `requires_grad=True`로 남지만 no_grad로만 호출됨 | 976-985, 1133-1139, E3c-B |
| `beta != 0`, 새 LoRA(`peft_config` 전달) | `ref_model = None`. ref 패스에서 `disable_adapter()` | 추가 가중치 없음, ref 패스 activation만 | 972-975, 2774-2789, E3c-C |
| `beta != 0`, adapter가 이미 붙은 `PeftModel` | `"ref"` adapter를 추가하고 `default`를 복사해 ref로 사용 | adapter 크기만큼 추가 | 452-480 |

- `sync_ref_model=True`는 `beta != 0`이고 비-PEFT일 때만 허용된다(1141-1157). `SyncRefModelCallback`이 `ref_model_sync_steps`마다 in-place `mul_().add_()`로 섞는다(`trl/trainer/callbacks.py:125-149`). 추가 tensor는 없다. VERIFIED(소스, E5 오류 메시지)
- `disable_dropout=True`면 정책과 ref의 `nn.Dropout.p`를 0으로 만든다(987-991). VERIFIED(소스)

## 9. 프롬프트 무절단과 completion 상한 (Q9)

- **프롬프트**: TRL 1.14.1 GRPO에는 프롬프트 길이 제한이 없다. `max_prompt_length`는 존재하지 않고(`TypeError`, E5) `_tokenize_prompts`(1769-1846)에는 truncation 인자가 없으며 tokenizer 기본도 `truncation=False`다. 따라서 **기본 설정에서는 프롬프트가 잘리지 않는다.** E3b에서 최대 272-token 프롬프트가 그대로 남았다(검증 V2: 4,656행 재계산, 최대 272 token(row 2355), user만 매핑하면 268, 빈 system은 모든 행에서 +4). VERIFIED
- **[검증 수정]** 그러나 무절단이 설정과 무관하게 보장되지는 않는다. `_tokenize_prompts`는 `**self.chat_template_kwargs`를 `apply_chat_template`에 그대로 넘기고(`grpo_trainer.py:1822-1831`), 이 함수는 `truncation`·`max_length`를 이름 있는 인자로 받는다(`transformers/tokenization_utils_base.py:2990-3005`). `chat_template_kwargs={"truncation": True, "max_length": 16}`이면 tokenizer 단독 호출에서 268 → 16 token이 됐고(검증 V2), 실제 `GRPOTrainer` 학습에서도 생성 입력 폭이 98 → 16이 됐다(검증 V4-trunc). 이때 사용자가 넘긴 tokenizer는 `truncation_side="right"`가 기본이라 생성 프롬프트 접미사까지 잘린다. 엄격 무절단 계약에는 "`chat_template_kwargs`에 `truncation`/`max_length` key가 없을 것"이라는 조건이 필요하다. VERIFIED
- 다만 Transformers 경로는 `P + max_completion_length`가 모델 context를 넘는지 검사하지 않는다. tool loop만 `max_position_embeddings`를 본다(2115-2123). 그래서 context 검사는 우리 쪽(plan.md §7.5)에서 해야 한다. vLLM 경로는 `vllm_max_model_length`가 `최대 프롬프트 + max_completion_length` 이상이어야 한다(`grpo_config.py:659-665` 도움말). VERIFIED(소스)
- **completion**: `max_completion_length`가 `max_new_tokens`다. 상한에 닿은 completion은 EOS 없이 끝나고 `completions/clipped_ratio`로 기록되며, 기본(`mask_truncated_completions=False`)이면 loss에 **포함**된다. `True`면 해당 행의 `completion_mask`를 0으로 만든다. tensor shape는 그대로라 메모리는 같다(2353-2357, 2539-2547). VERIFIED
- `max_completion_length=None`은 무제한이 아니다. E3c-L: `max_new_tokens=None`인데 20 token만 생성됐다(transformers 기본 길이). `loss_type="dr_grpo"`에서는 `None × int`라 실패한다(3225). 우리 config는 항상 유한 정수를 써야 한다. VERIFIED
- **[검증 보강]** 20 token의 코드 경로(미확정 사항 7 해결): TRL이 `max_new_tokens=None`을 넘기고 모델 `generation_config`에도 `max_length`가 없으면 `has_default_max_length=True`다(`transformers/generation/utils.py:2758-2762`). `_prepare_generation_config`가 전역 기본 `max_length=20`(`generation/configuration_utils.py:615`)을 None 자리에 채우고(`generation/utils.py:2085-2086`), `_prepare_generated_length`의 `elif has_default_max_length:  # by default let's always generate 20 new tokens` 분기가 `max_length = 20 + 프롬프트 길이`로 바꾼다(`generation/utils.py:2023-2027`, `max_position_embeddings`로 상한). 검증 V4-none: 입력 98 token → 출력 118 token(+20). VERIFIED. 모델 `generation_config.json`에 `max_length`가 있으면(예시 모델은 없음) 그 절대 길이가 상한이 된다. INFERRED(소스)
- `pad_to_multiple_of`는 데이터를 자르지 않지만 P와 L을 늘려 메모리를 키운다(§6.4). 기본 None을 유지한다. VERIFIED

## 10. optimizer step당 phase 타임라인 (Q10)

기본 정렬 설정(`spg = K`, `num_iterations = 1`, `beta = 0`, vLLM·CB 없음, 단일 GPU)이다. micro-step 0이 rollout을 포함한다. Trainer는 window 시작 시 K개 batch(원시 dict, host RAM)를 먼저 가져온다(`transformers/trainer.py:1827-1838`). optimizer step 뒤 `model.zero_grad()`가 `set_to_none=True`로 grad를 해제한다(`trainer.py:1908`, `torch/nn/modules/module.py:2957`). VERIFIED

| # | phase (plan §9.2) | 실행 내용 | 이 phase에서 살아 있는 것 (상주분 외) | 근거 |
|---|---|---|---|---|
| 0 | 상주 | — | 정책 가중치(+LoRA), (beta≠0 비-PEFT) ref 모델, reward 모델, optimizer state(첫 step 이후), `_buffered_inputs`(작음) | §5.4, §7, §8 |
| 1 | `ROLLOUT_PREFILL_AND_DECODE` (prefill) | no_grad, GC 끔, train mode. 입력 `(C, P)` 왼쪽 패딩 | prefill activation(no-grad, layer별 순간값), 생성 중 cache, prefill logits `(C,1,V)` | §4.2–4.5 |
| 2 | 같은 phase (decode) | `L − 1`회 decode | KV `(C, n_kv, T, d)` × n_full이 `T = P + L − 1`까지 증가(cat 순간 +1 layer분), linear state `C × 49.5–51 MiB`, step별 `(C, V)` fp32 logits 사본 | §4.2, §4.5 |
| 3 | (logprob, 조건부) | old: `K % generate_every != 0` 또는 vLLM. ref: `beta != 0` | chunk(B_update행) no-grad forward activation + logits 약 10 B × B_update × (L+1) × V | §5 |
| 4 | `REWARD` | 텍스트 decode(host), reward 함수 | reward 모델 forward `C × T_reward`(no-grad) 또는 host 계산. old/ref logp `(C, L)` fp32는 살아 있음(작음) | §7 |
| 5 | `POLICY_FORWARD_BACKWARD` × K | slice별 `compute_loss` + backward | GC 경계 activation, logits fp32 4 B → backward 8 B × B_update × (L+1) × V, grad(첫 backward부터 window 끝까지 누적) | §6 |
| 6 | `OPTIMIZER_STEP` | grad clip(`max_grad_norm=1.0`), AdamW fused step, `zero_grad` | grad + optimizer state(첫 step에 생성) + optimizer 임시값 | `transformers/trainer.py:1890-1908` |

**[검증 수정]** 원래 표는 `REWARD`(3)를 조건부 logprob(4)보다 앞에 두었다. 실제 `_generate_and_score_completions`의 순서는 generate(`grpo_trainer.py:2488`) → old/ref logprob(`2686-2791`) → 텍스트 decode(`2793`) → reward(`2809`)다. 위 표는 이 순서로 고쳤다. VERIFIED(소스)

- 정렬 설정에서는 rollout 시점에 **grad가 없다**(None). optimizer state는 두 번째 rollout부터 있다(E3c-T: rollout마다 grad 0, 두 번째 rollout에 state 168개). `generate_every % K != 0`이면 window 중간에 rollout이 일어나 **grad가 살아 있다**(E3c-S: spg=2, K=4 → micro-step 2, 6 rollout 때 grad 56개). VERIFIED
- 비-PEFT full FT에서 텍스트 전용 데이터면 vision tower 파라미터는 grad를 받지 않고 AdamW state도 생기지 않았다(E3c-S/T: 77개 중 56개만 grad, state 56×3). VERIFIED(CPU)
- `num_iterations > 1`이면 같은 버퍼로 `spg × num_iterations` micro-step을 학습하고, 그동안 optimizer step이 여러 번 일어난다(E3c-D: 6행 → global_step 12). VERIFIED
- 평가(기본 꺼짐)는 `prediction_step`에서 생성 + `compute_loss`를 no_grad로 한다(3355-3361). backward가 없어 logits peak는 약 6 B다. INFERRED

---

## 구현 시사점 (Implementation implications)

### R1. config 해석기 (`trainers/grpo` adapter)

```text
입력: G, B_update, K, W, gbs?, spg?, num_iterations, beta, use_vllm, vllm_is_correction
spg, gbs  ← §2.1 공식 (gbs와 spg를 둘 다 받으면 GRPO_CONFIG_CONFLICT)
검증      ← gbs % (B_update·W) == 0 (gbs 지정 시), gbs % G == 0, G >= 2, auto_find_batch_size == False
U = gbs / G;  C = B_update · spg
generate_every = spg · num_iterations
old_logps   = (K % generate_every != 0) or (use_vllm and vllm_is_correction)
ref_logps   = beta != 0
ref_mode    = none | separate_model (비-PEFT) | disable_adapter (새 LoRA) | ref_adapter_copy (기존 adapter)
grads_alive_during_rollout = (generate_every % K != 0)
micro_steps_per_epoch = (N // U) · spg · num_iterations      # process당
optimizer_steps_per_epoch = ceil(micro_steps_per_epoch / K)
dropped_per_epoch = N mod U                                    # 0이 아니면 plan §7.4에 따라 전체 coverage 실패
```

예시(G=4, gbs=4, B=1, K=4, W=1): spg=4, U=1, C=4, old/ref 없음, rollout 시 grad 없음, dropped 0, epoch당 4,656 optimizer step.

### R2. shape 규칙

- `P_worst` = 데이터셋 최대 프롬프트 길이(학습과 같은 `_tokenize_prompts` 결과). 셔플 때문에 어느 프롬프트든 어느 batch에 들어갈 수 있다. `L_worst = max_completion_length`(시나리오 budget). `pad_to_multiple_of`가 있으면 둘 다 올림한다.
- micro-batch 폭은 **generation batch 전체의 최댓값**이다. B_update가 작아도 `P_worst + L_worst`로 계산한다.

### R3. rollout cache 공식 (Transformers 공유 정책 경로)

```text
T_kv     = P + L − 1
KV_full  = 2 · n_full · n_kv · head_dim · b_kv · C · T_kv  (+ cat 순간 1 layer의 K 또는 V 한 벌)
LinState = n_lin · C · (conv_dim · conv_kernel · b_conv + n_v · d_k · d_v · 4)
b_kv, b_conv:  PEFT/QLoRA → 로드 dtype의 byte(TRL 기본 fp32=4, dtype=bfloat16이면 2)
               full FT   → 로드 dtype이 bf16이면 2, fp32면 b_kv=4·b_conv=2 (autocast + RoPE promotion)
decode logits: C · V · 4 (+ prefill (C,1,V)); top_k/top_p를 켜면 (C,V) 임시값 추가
```

**[검증 보강]** b_kv·b_conv 규칙은 accelerate native AMP(`bf16=True`, 단일 GPU·DDP) 전제에서 검증했다. 검증 V4에서 A(full FT, fp32 로드: K/V fp32, conv bf16), F(full FT, bf16: K/V·conv bf16), C(LoRA, fp32: 전부 fp32, prefill logits fp32), G(LoRA, bf16: K/V·conv bf16, prefill logits bf16)를 다시 재현했다. recurrent state는 모든 경우 fp32였다. DeepSpeed 등 native AMP가 꺼진 경로는 이 규칙을 쓰지 않는다. 표의 수치(161.875 / 323.75 MiB, `C × 49.5 / 51.0 MiB` 등)는 다시 계산해 일치를 확인했다.

공식 적용 예시(예시 모델, C=4, P=272, GPU 실측 아님):

| L (budget) | T_kv | KV bf16 | KV fp32 | linear state (C=4) |
|---|---|---|---|---|
| 1,024 | 1,295 | 161.9 MiB | 323.8 MiB | 198.0 MiB (bf16 conv) / 204.0 MiB (fp32 conv) |
| 2,048 | 2,319 | 289.9 MiB | 579.8 MiB | 같음 |
| 4,096 | 4,367 | 545.9 MiB | 1,091.8 MiB | 같음 |
| 8,192 | 8,463 | 1,057.9 MiB | 2,115.8 MiB | 같음 |

### R4. logits·loss 공식 (CUDA fused, bf16 mixed precision)

```text
E = B_update · (L + 1) · V
policy forward 순간      : 6 · E bytes   (bf16 lm_head 출력 + fp32 변환본)
policy forward→backward  : 4 · E bytes   (fp32 logits 보관)
policy backward peak     : 8 · E bytes   (저장 logits + grad_logits)
old/ref no-grad 패스     : 10 · E bytes  (chunk 2개 이상; 1개면 6 · E)
Triton 없음(fallback)    : backward 16 · E bytes로 profile을 바꾼다
```

전제는 accelerate native AMP(fp32 출력 wrapper)와 Linux CUDA의 Triton 경로다. **[검증 보강]** 독립 tracker로 다시 측정했다(검증 V6: 실제 `GRPOTrainer._full_logits_logps` + accelerate `prepare_model` wrapper + kernel 할당 에뮬레이션, vocab 크기 파라미터는 freeze). forward peak 6.29–6.68, forward 뒤 보관 4.29–4.68, backward peak 8.28–8.67, no-grad 4행·chunk 1은 10.00–10.13 B/elem이었다. 8E는 logits 관련 항만이다(E4에서 0.2–1.2 B/elem의 나머지는 작은 모델의 다른 activation). 공식 적용 예시(B_update=1, V=248,320, GPU 실측 아님): L=1,024 → 8E = 1.896 GiB, L=2,048 → 3.791 GiB, L=4,096 → 7.580 GiB, L=8,192 → 15.158 GiB. 긴 completion budget에서 가장 큰 항목이다. evidence는 `measured(CPU emulation)`, CUDA는 `analytic`으로 표시하고 M5 GPU 보정 대상에 넣는다.

### R5. 모델 로딩 기본값과 내보내기 config

- 문자열 모델은 `model_init_kwargs.dtype`이 없으면 fp32다. 예시 QLoRA에서는 embedding·lm_head·norm·conv1d·A_log·vision 비양자화 모듈이 모두 fp32이고 rollout KV/conv도 fp32다(E7, E3c-Q). **기본 product preset은 `model_init_kwargs={"dtype": "bfloat16"}`을 명시**하고 requested/resolved에 dtype을 기록한다.
- QLoRA면 TRL이 trainable 파라미터(LoRA)를 bf16으로 바꾼다(506-513). 비양자화 LoRA는 PEFT 기본으로 fp32 adapter다(E3c-G). `target_modules="all-linear"`는 vision tower Linear에도 adapter를 붙인다(E7: Linear4bit 37개 모두 대상).
- 예시 모델은 `processing_class=AutoTokenizer.from_pretrained(<id>, revision=<rev>, padding_side="left")`를 명시해서 내보낸다(pinned 환경에 torchvision/Pillow 없음). 그러면 `_is_vlm=False`가 되고 token id는 같다(E3b).
- `gradient_checkpointing_kwargs`는 `None` 또는 `{"use_reentrant": bool}`만 허용한다. `every_n_layers`/`offload`는 GRPO profile에서 `UNSUPPORTED_OPTION`으로 막는다(§4.3). **[검증 수정]** 막는 이유는 "항상 오류"가 아니다. `use_reentrant` 미지정/True면 오류이고, `use_reentrant=False`면 조용히 무시되어 `every_n_layers=1`, offload 없음으로 학습된다. 사용자 config를 그대로 해석해야 하는 경우에도 두 옵션의 메모리 절감은 0으로 계산한다.
- `cache_implementation`은 None(DynamicCache)만 1차 지원한다. `"quantized"`는 Qwen3.5에서 오류다. `use_vllm`, `use_transformers_continuous_batching`은 1차에서 unsupported로 표시한다(CB는 free memory 50% 예약, vLLM은 별도 엔진).
- `max_completion_length`는 항상 유한 정수 budget으로 넣고, `None`은 거부한다.

예시 preset의 내보내기 초안(검증 미완료 표시 필요, reward 미정이면 실행 대상 아님):

```python
GRPOConfig(
    num_generations=4, generation_batch_size=4,          # steps_per_generation은 넣지 않는다(상호 배타)
    per_device_train_batch_size=1, gradient_accumulation_steps=4,
    max_completion_length=<budget>, beta=0.0, num_iterations=1,
    gradient_checkpointing=True, gradient_checkpointing_kwargs=None,
    bf16=True, model_init_kwargs={"dtype": "bfloat16"},
    use_vllm=False, use_transformers_continuous_batching=False, cache_implementation=None,
    pad_to_multiple_of=None, shuffle_dataset=True,
)
# GRPOTrainer(model=<id>, reward_funcs=<필수>, processing_class=<tokenizer, left padding>,
#             quantization_config=BitsAndBytesConfig(nf4, double quant, compute bf16), peft_config=LoraConfig(...))
```

### R6. 데이터 변환·coverage 검사

- GRPO 전처리 parity는 `GRPOTrainer._tokenize_prompts`와 같아야 한다(`apply_chat_template(add_generation_prompt=True, tokenize=True)`, special token 추가 없음, truncation 없음). 예시의 빈 `system` 메시지는 4 token을 더한다(272 vs 268). mapping 결정은 화면에 표시한다.
- **[검증 보강]** 엄격 무절단 검사: `chat_template_kwargs`에 `truncation` 또는 `max_length` key가 있으면 GRPO 무절단 계약 위반으로 거부한다(§9 [검증 수정], 검증 V4-trunc). 우리 전처리 parity 구현도 `chat_template_kwargs`를 TRL과 똑같이 `apply_chat_template`에 넘겨야 같은 token id가 나온다.
- coverage 검사기는 `N mod U`, `max_steps`, `num_train_epochs`로 판정하고, 누락이 있으면 `dropped_per_epoch`와 "epoch마다 새로 뽑히는 무작위 프롬프트(같은 row가 다시 빠질 수도 있음)"라는 사실을 함께 보고한다.

### R7. reward·reference ledger

- `reward.kind=local_model`: reward 가중치를 **전 phase 상주**로 넣고(dtype = `model_init_kwargs.dtype` 또는 config의 `"auto"`), REWARD phase에 `C × T_reward` no-grad activation을 더한다. `reward.kind=unspecified`면 Trainer를 만들 수 없으므로 결과는 conditional이다(plan.md §8.4).
- `beta != 0`인 비-PEFT는 정책과 같은 dtype·양자화의 두 번째 전체 모델을 상주로 넣는다. PEFT는 가중치 추가 없이 ref no-grad 패스(R4의 10E)만 넣는다.

### R8. phase 스케줄 (TrainerAdapter 출력)

§10 표를 그대로 phase 목록으로 쓴다(검증 후 순서: rollout → 조건부 old/ref logprob → reward → policy forward/backward → optimizer step). `peak = max(phase별 합)`이고 상주분(가중치, optimizer state, ref/reward 모델)은 모든 phase에 더한다. `grads_alive_during_rollout`이 참이면 rollout phase에 grad를 더한다.

## 미확정 사항 (Open questions)

1. **CUDA 실측 부재**: fused kernel, accelerate fp32 변환, slice backward의 피크(8E/10E)는 CPU에서 할당 패턴을 에뮬레이션한 값이다. CUDA caching allocator의 블록 반올림, `DynamicLayer` `torch.cat` 증가로 인한 fragmentation, reserved와 allocated의 차이는 측정하지 않았다. M5 GPU 보정에서 phase별 `max_memory_allocated`/`max_memory_reserved`로 확인해야 한다. UNKNOWN
2. **hybrid layer의 prefill·학습 activation**: pinned 환경에는 FLA·causal-conv1d가 없어 CUDA에서도 torch fallback(`torch_chunk_gated_delta_rule`, fp32 중간값)을 쓴다. 이 경로의 prefill과 학습 activation 크기, 그리고 FLA를 설치했을 때 recurrent state dtype은 아키텍처 조사 범위다. UNKNOWN
3. **full-attention prefill의 SDPA backend**: 왼쪽 패딩 mask가 있는 prefill에서 memory-efficient backend가 선택되는지(`(C, H, P, P)` 행렬을 만드는지)는 GPU와 attention 구현에 따라 다르다. INFERRED만 가능
4. **decode 중 sampling 임시값**: 모델 권장값(top_k=20, top_p=0.95)을 `generation_kwargs`로 켜면 `(C, V)` 정렬·인덱스 임시값이 생긴다. C가 작을 때는 무시할 수 있지만 정확한 배수는 측정하지 않았다. INFERRED
5. **vLLM colocate / continuous batching**: 별도 엔진 가중치, KV pool, CUDA graph, `gpu_memory_utilization`/`max_memory_percent` 예약이 학습 phase와 어떻게 겹치는지(sleep mode 포함)는 조사하지 않았다. 1차 범위 밖. UNKNOWN
6. **다중 GPU**: ZeRO-3의 `ds3_gather_for_generation=True`에서 생성 시 전체 파라미터 gather, FSDP1 `summon_full_params`의 순간 메모리. 소스에서 경로만 확인했다. INFERRED
7. ~~**`max_completion_length=None`이 20 token이 되는 정확한 원인**~~ → **검증에서 해결.** `transformers/generation/utils.py:2758-2762, 2085-2086, 2023-2027`과 `generation/configuration_utils.py:615`가 원인이다(§9 [검증 보강]). VERIFIED
8. ~~**`offload` 키 실패**~~ → **검증에서 해결.** `{"offload": True}`는 `ValueError`이고, `use_reentrant=False`를 함께 주면 오류 없이 offload만 꺼진다(§4.3 [검증 수정] 표, 검증 V4-gcC/gcD). VERIFIED(CPU)
9. **reward model의 `C × T_reward` activation**: reward tokenizer의 chat template과 모델 구조에 의존한다. local reward model 지원 시 별도 architecture adapter가 필요하다. UNKNOWN

---

## 부록 A. 실험 목록

공용 venv로 실행한 E1·E2를 제외하면 `/tmp/vf-research/venv-trl-grpo/bin/python`(공용 site-packages + Pillow + torchvision)으로 실행했다. 모두 `HF_HOME=/tmp/vf-research/hf`, `HF_HUB_OFFLINE=1`, `use_cpu=True`, `report_to="none"`이다. 스크립트 위치는 `/tmp/vf-research/scratch/trl-grpo/`다.

| ID | 스크립트 (인자) | 내용 | 핵심 출력 |
|---|---|---|---|
| E1 | `e1_config.py` | `GRPOConfig` 기본값 dump, 검증 규칙 16가지 | §1 표, §2.1 표. `has max_prompt_length attr: False` |
| E2 | `e2_sampler.py` | 실제 `RepeatSampler` + `BatchSampler` + accelerate `BatchSamplerShard`로 4,656 row coverage 시뮬레이션 (rank마다 같은 seed의 sampler) | §3.2 표. 예시: unique 4,656, dropped 0, micro-step 18,624 |
| E3a | `e3a_tokenize.py` | 공용 venv: `AutoProcessor` 실패(Pillow/torchvision 없음). overlay venv: `Qwen3VLProcessor`, `_tokenize_prompts` 결과 | 공용 venv `ValueError: Could not load any image processor class …`, 이어서 `Qwen3VLVideoProcessor requires the Torchvision library`. 최대 272 token |
| E3b | `e3b_tok_vs_proc.py` | processor 경로 vs tokenizer 경로 4,656행 비교, 빈 system 수, plain text BOS | mismatch 0, system 빈 값 4,656/4,656, user만 매핑 시 최대 268 |
| — | `make_tiny.py` | tiny random `Qwen3_5ForConditionalGeneration`(hidden 64, layer 4 = linear 3 + full 1, head_dim 128, kv 2, linear k/v heads 2/4, dim 16, vocab 248,320, vision depth 1) + 실제 processor/tokenizer 저장 | 32,223,752 params |
| E3c | `e3c_trainer_probe.py <A..T>` | 실제 `GRPOTrainer.train()`을 hook으로 probe: 로드 dtype, generate 호출 행 수, cache class·shape·dtype, prefill `logits_to_keep`, GC 상태, logprob 호출(모델, grad, chunk), 버퍼 key·shape, rollout 시 grad/optimizer state | A 기본, B beta=0.04, C LoRA+beta, D num_iterations=2, E spg=8, F bf16, G LoRA+bf16, H gbs=8(7행), I use_reentrant=True, J every_n_layers=2(오류), K GC 끔+beta, L max_completion_length=None, M pad_to_multiple_of=64, N spg=8 생성 행 수, Q/R QLoRA(dtype 없음/bf16), S spg=2·K=4, T 정렬 |
| E4 | `e4_logits_mem.py <fused_emul\|fallback> <bf16\|fp32> <L> [1]`, `BUPD=` | `livetrack.py`(TorchDispatchMode + storage weakref live-byte tracker, 기지 패턴 12 MiB/4 MiB로 검증)로 `compute_loss` + `backward` live byte 측정. `fused_emul`은 `trl/kernels/logprob_entropy.py`와 같은 할당을 하는 autograd Function | §6.3 표 |
| E4b | `e4b_nograd.py bf16 1024`, `ROWS=4` | old/ref 패스(`batch_size=1` chunk) live byte | peak 10.13 B/elem, 출력 `(4, 1024) float32` |
| E5 | `e5_reward_ref.py` | `max_prompt_length` 거부, reward 없음, `sync_ref_model` 제약, reward 모델 id 로딩(tiny `LlamaForSequenceClassification`, bf16 저장) | §7, §8 메시지와 dtype |
| E7 | `e7_qlora.py <none\|bfloat16>` | 실제 `GRPOTrainer` QLoRA(nf4, double quant, compute bf16) 로드와 1 step 학습(bitsandbytes CPU backend) | dtype 없음: 비양자화 40개 fp32, Linear4bit 37개(vision 포함), LoRA bf16. 1 step 성공 |

---

## 검증 로그 (Verification log)

검증자는 `verify-trl-grpo`이고 날짜는 2026-10-04다. 원 문서에서 구현에 직접 들어가는 주장(기본값, 배치 공식, coverage, 무절단, cache·logits byte 공식, dtype, phase 순서, 호환성 결론)을 골라 설치된 소스를 다시 읽었다. 원 실험 스크립트(E1–E7)는 쓰지 않고 독립 실험 V1–V6으로 재현했다.

- VERIFIED: 소스를 다시 읽었거나 독립 실험으로 재현했다.
- CORRECTED: 틀렸거나 과장되어 본문을 고쳤다(**[검증 수정]** 표시).
- UNVERIFIABLE: 이 머신(macOS arm64, CPU)에서 확인할 수 없다.

| # | 원 문서 주장 | 판정 | 근거 (한 줄) |
|---|---|---|---|
| 1 | `GRPOConfig`에 `max_prompt_length`가 없어 `TypeError`가 난다 | VERIFIED | V1: `TypeError: GRPOConfig.__init__() got an unexpected keyword argument 'max_prompt_length'`. 이름에 `prompt`가 들어간 필드는 `log_unique_prompts` 하나다 |
| 2 | 프롬프트는 어떤 설정으로도 잘리지 않는다 | CORRECTED | `grpo_trainer.py:1822-1831`이 `**chat_template_kwargs`를 `apply_chat_template`(`tokenization_utils_base.py:2990-3005`의 `truncation`/`max_length` 인자)에 넘긴다. V2에서 268 → 16 token, V4-trunc의 실제 학습에서 생성 입력 폭 98 → 16. 기본값에서 무절단인 것은 맞다 |
| 3 | gbs/spg 해석, `gbs % (B·W)`, `gbs % G`, `G >= 2`, `auto_find_batch_size` 금지 | VERIFIED | `grpo_config.py:1076-1128`을 다시 읽었다. V1에서 원 E1 표 9가지를 같은 메시지로 재현했다. 추가로 G=4, gbs=4, B=1, K=3 → spg=4가 허용됐다(K는 G와 무관) |
| 4 | 기본값: G=8, B=8, K=1 → gbs=8·spg=1, `max_completion_length=512`, `beta=0`, `dapo`, `scale_rewards="group"`, ε=0.2, 1.0/1.0/0, vLLM 끔, `cache_implementation=None`, GC 켬, bf16 켬, `adamw_torch_fused`, 3 epoch, seed 42, lr 1e-6 | VERIFIED | V1 `GRPOConfig` dump가 모두 일치했다. `trl/trainer/base_config.py:61-74, 104-105` |
| 5 | TRL의 1.0/1.0/0이 모델 `generation_config.json`의 0.6/0.95/20을 덮어쓴다 | VERIFIED | transformers 5.18 `_prepare_generation_config`는 넘긴 config의 None 값만 모델 값으로 채운다(`generation/utils.py:2085-2086`, `generation/configuration_utils.py:1364`). TRL의 override는 transformers 5 이상에서 no-op이다(`trl/models/utils.py:174-180`). V4: MiMo `generation_config.json`을 넣은 tiny 모델에서 `_sample` 안의 실효값은 1.0/0/1.0, eos 248046 하나, logits processor 0개였다 |
| 6 | `RepeatSampler`가 epoch마다 `N mod U`개를 버리고, epoch마다 다른 무작위 프롬프트가 빠진다 | VERIFIED (표현 보강) | `trl/trainer/utils.py:890-912`를 다시 읽었다. generator는 `__init__`에서만 seed되고 `set_epoch`가 없다. V3: N=4,656, U=5/7/32에서 누락 1/1/16. N=10, U=3, 4 epoch의 누락 row는 7, 2, 2, 6이었다. 매 epoch 새로 뽑지만 같은 row가 다시 빠질 수 있다 |
| 7 | 예시(G=4, gbs=4, B=1, K=4, W=1): spg=4, U=1, C=4, 누락 0, micro-step 18,624, optimizer step 4,656/epoch, 3 epoch 13,968 | VERIFIED | V3(실제 `RepeatSampler` + `BatchSampler`): unique 4,656, micro-step 18,624. `transformers/trainer.py:2466-2475`가 `ceil(len_dataloader / K)`이므로 4,656이고, `max_steps = ceil(3 × 4,656) = 13,968`이다 |
| 8 | rollout은 C = B·spg행을 left-pad해 `generate` 1회로 처리한다. train mode, GC 끔, prefill `logits_to_keep=1`, 최종 KV 길이 P + L − 1 | VERIFIED | `grpo_trainer.py:1895-1944`, `generation/utils.py:2920-2924`를 다시 읽었다. V4: `training=True`, GC 꺼짐, prefill logits `(2,1,248320)`, P=98·L=6 → K 길이 103. 한 step을 더 도는 `DeferredStopCheck`는 MPS 전용이다(`generation/utils.py:432-441`) |
| 9 | Qwen3.5 cache = DynamicCache(full → DynamicLayer, linear → LinearAttentionLayer), conv `(C, 8192, 4)`, recurrent `(C, 32, 128, 128)` fp32, `"quantized"`는 오류, R3 수치 | VERIFIED | `cache_utils.py:1249-1258, 1961-1966`, `modeling_qwen3_5.py:520(conv_dim), 406(fp32 state)`를 다시 읽었다. MiMo config로 conv_dim 8,192를 확인했다. V4 tiny: conv `(2,128,4)`, rec `(2,4,16,16)` fp32. R3 표의 수치(161.875 MiB 등)와 `C × 49.5/51.0 MiB`를 다시 계산했다 |
| 10 | TRL은 문자열 모델을 fp32로 로드하고, PEFT `generate`는 autocast wrapper를 우회한다. dtype 표 A/F/C/G/Q/R | VERIFIED (Q/R은 소스만) | `trl/trainer/utils.py:1292-1296`(dtype 기본 `"float32"`), `peft/peft_model.py:1009-1012, 2229-2250`, accelerate `unwrap_model(keep_fp32_wrapper=True)` 기본값(`accelerate/accelerator.py:3254`, 그래서 비-PEFT 정책의 `generate`는 wrapper를 유지한다). V4에서 A·F·C·G 행이 모두 일치했다. QLoRA Q/R은 다시 실행하지 않았다. `bitsandbytes` `Linear4bit.forward`가 출력을 `inp_dtype`으로 되돌리는 코드와는 맞는다 |
| 11 | old logp는 `K % (spg·μ) != 0` 또는 vLLM IS일 때, ref는 `beta != 0`일 때 계산한다. chunk = B_update, `logits_to_keep = L`, 모델에는 L + 1 | VERIFIED (소스) | `grpo_trainer.py:2686-2791, 2556-2557, 1551-1569`를 다시 읽었다. 호출 횟수는 재실행하지 않았다 |
| 12 | 학습 logits는 항상 fp32다 | CORRECTED | accelerate의 bf16 native AMP 분기는 DeepSpeed·Megatron을 제외하고(`accelerate/accelerator.py:586-593`), mixed precision을 끄면 wrapper가 없다. 기본 단일 GPU 설정에서는 fp32가 맞다 |
| 13 | fused kernel의 logits 항: forward 6E, 보관 4E, backward 8E, no-grad 2개 이상 chunk 10E | VERIFIED (CPU emulation) / INFERRED (CUDA) | `trl/kernels/logprob_entropy.py:165-168, 192, 210`, `trl/trainer/utils.py:507-532`를 다시 읽었다. V6 독립 tracker 결과는 6.29–6.68 / 4.29–4.68 / 8.28–8.67 / 10.00–10.13 B/elem이다. torch 2.14.1 metadata가 Linux에 `triton~=3.8.0`을 요구하므로 Linux CUDA에서는 fused 경로가 기본이다 |
| 14 | micro-batch 폭 = generation batch 전체의 최대 P·L, `pad_to_multiple_of`는 둘 다 올린다 | VERIFIED (소스) | `grpo_trainer.py:2492-2514`(prompt는 left, completion은 right, 둘 다 `pad_to_multiple_of` 적용), `1621-1631`(split만 하고 다시 자르지 않음) |
| 15 | 정렬 설정이면 rollout 때 grad가 없고 아니면 살아 있다. optimizer state는 2번째 rollout부터 있다. vision 파라미터는 grad·state가 없다 | VERIFIED | V5-S(spg=2, K=4): micro-step 2·6의 rollout 때 grad 56개, 4에서는 grad 0개 + state 56개. V5-T(spg=K=2): 매번 grad 0개, 2번째부터 state 56개. vision 파라미터 21개는 `requires_grad=True`지만 state가 없었다. `trainer.py:1908`, `torch/nn/modules/module.py:2957` |
| 16 | reward 모델은 `AutoModelForSequenceClassification(num_labels=1, **model_init_kwargs)`로 로드한다. dtype 미지정이면 `"auto"`, GPU 상주, C행을 한 번에 `inference_mode`로 처리, reward source가 없으면 init에서 ValueError | VERIFIED (소스) | `grpo_trainer.py:515-537, 705-711, 1159-1167, 1690-1705`, `transformers/modeling_utils.py:4106-4107`. 원 E5는 재실행하지 않았다 |
| 17 | `every_n_layers`는 첫 학습 forward에서 ValueError가 나고, `offload`도 같을 것이다(INFERRED) | CORRECTED | 오류는 torch reentrant 경로에서만 난다(`torch/utils/checkpoint.py:635-649`). V4: `{"every_n_layers": 2}`, `{"offload": True}`는 ValueError였다(offload도 이제 VERIFIED). `use_reentrant=False`를 함께 주면 오류 없이 학습되며 `every_n_layers`는 1이 되고 offload는 꺼진다 |
| 18 | `max_completion_length=None`이면 약 20 token이 생성되고 `dr_grpo`는 실패한다 | VERIFIED (원인 확정) | V4-none: 98 → 118 token. 코드 경로는 `generation/utils.py:2758-2762, 2085-2086, 2023-2027`과 `configuration_utils.py:615`다. `dr_grpo` 실패 위치는 `grpo_trainer.py:3225` |
| 19 | phase 순서: REWARD 다음에 조건부 logprob | CORRECTED | 실제 순서는 generate(2488) → old/ref logprob(2686-2791) → decode(2793) → reward(2809)다. §10 표와 R8을 고쳤다 |
| 20 | 기본 processing class(`AutoProcessor`)는 Pillow·torchvision이 필요하고 pinned venv에는 둘 다 없다 | VERIFIED | `importlib.metadata`로 공용 venv를 확인했다. pillow, torchvision, triton, fla, causal-conv1d, kernels, liger-kernel, vllm이 모두 없다. 공용 venv에서 `AutoProcessor.from_pretrained` → `ValueError: Could not load any image processor class ...` |

### 검증 실험 (V1–V6)

모두 공용 venv를 수정 없이 사용했다. 명령은 `HF_HOME=/tmp/vf-research/hf HF_HUB_OFFLINE=1 /tmp/vf-research/.venv/bin/python <script>`이고, 스크립트는 저장소 밖 `/tmp/vf-research/scratch/verify-trl-grpo/`에 있다. tokenizer를 직접 넘겨서 overlay venv(Pillow·torchvision)가 필요 없었다.

| ID | 스크립트 (인자) | 내용 | 핵심 출력 |
|---|---|---|---|
| V1 | `v1_config.py` | `GRPOConfig` 기본값 dump, 검증 규칙 12가지 | §1 기본값 일치, `max_prompt_length` `TypeError` |
| V2 | `v2_prompts.py` | MiMo tokenizer로 4,656행 GRPO 프롬프트 길이 재계산, `chat_template_kwargs` truncation | system+user 최대 272(row 2355), user만 268, 차이는 모든 행에서 4. `{"truncation": True, "max_length": 16}` → 16 |
| V3 | `v3_sampler.py` | 실제 `RepeatSampler` + `BatchSampler` + accelerate `BatchSamplerShard` coverage | 예시는 unique 4,656, micro-step 18,624, step 4,656/epoch. 다중 epoch에서 누락 row를 새로 뽑음 |
| V4 | `make_tiny_v.py`, `v4_trainer_probe.py <base\|fullF\|loraC\|loraG\|gcA\|gcB\|gcC\|gcD\|trunc\|none>` | 독자 tiny `Qwen3_5ForConditionalGeneration`(hidden 64, linear 3 + full 1, head_dim 64, kv 1, linear k/v heads 2/4·dim 16, vocab 248,320, vision depth 1, MiMo `generation_config.json` 복사, 32,125,320 params)로 실제 `GRPOTrainer.train()`을 실행하고 `_sample`·`_prefill`·layer pre-hook으로 probe | 실효 샘플링 값, cache class·shape·dtype, GC 상태, GC kwargs 4가지 결과, truncation, `None` budget |
| V5 | `v5_grad_probe.py <S\|T>` | rollout 시점의 grad 수와 optimizer state 수, vision 파라미터 state | S: 0/56/0(+state 56)/56, T: 0/0/0(state는 2번째부터 56), vision state 0 |
| V6 | `v6_logits_mem.py <L> <bf16\|fp32> [rows]` | 독자 `TorchDispatchMode` + `StorageWeakRef` live-storage tracker. 실제 `GRPOTrainer._full_logits_logps`, accelerate `prepare_model`(bf16 native AMP, CPU), kernel 할당 에뮬레이션 | L=256/1024: forward 6.68/6.29, 보관 4.68/4.29, backward 8.67/8.28 B/elem. no-grad 4행·chunk 1: 10.13(bf16)/10.00(fp32) |

### 남은 위험

- CUDA 실측이 없다. 6E/8E/10E와 KV·linear state 공식은 CPU 재현과 소스에 근거한다. CUDA caching allocator의 반올림과 fragmentation은 M5에서 확인해야 한다.
- QLoRA dtype 행(Q/R), reward 모델 dtype, old/ref 호출 횟수는 다시 실행하지 않았다(소스 재독만 했다).
- `use_reentrant=False`일 때 남는 `every_n_layers`/`offload` key는 decoder layer kwargs로 흘러간다. CPU SDPA 경로에서는 무시됐지만 CUDA attention backend(flash-attn 등)에서도 무시되는지는 실행하지 않았다. INFERRED
