# 예시 모델·데이터셋 실측 조사: tokenizer, chat template, 전체 길이 golden 값

> 예시 입력(plan.md §21)의 모델 tokenizer와 chat template, 그리고 데이터셋 전체 4,656 row를 **실제 TRL 1.14.1 전처리 함수**로 토큰화해 얻은 기준값(golden)을 정리한다.
> 이후 회귀 테스트 fixture는 문서 끝 부분의 `golden-example-stats` JSON 블록을 그대로 복사해 쓴다.

| 항목 | 값 |
|---|---|
| 주제 | 예시 모델 tokenizer·chat template 분석, 예시 데이터셋 사실 확인, objective별 전체 토큰 길이 golden 값 |
| 작성 | `research-example-data` (Milestone M0) |
| 기준일 | 2026-10-04 |
| 고정 버전 (학습 환경) | torch==2.14.1, transformers==5.18.0, trl==1.14.1, peft==0.21.2, bitsandbytes==0.50.2, accelerate==1.15.0, datasets==5.0.1, huggingface_hub==1.33.0, tokenizers==0.23.2, safetensors==0.8.0 (부수: jinja2==3.1.6, numpy==2.5.3, pyarrow==25.0.1) |
| 모델 | `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` @ `2367e865d009c13ac81713a2878291d33ab28177` (가중치 `*.safetensors`는 받지 않음) |
| 데이터셋 | `CyberNative/Code_Vulnerability_Security_DPO` @ `81aeacf06cf43b16d7278a3a01f019a496a53c51`, split `train` |
| 방법 | 설치된 패키지 소스 정독 + CPU 실험. 길이 값은 손으로 만든 근사 식이 아니라 `SFTTrainer._prepare_dataset`, `DPOTrainer._prepare_dataset`, `GRPOTrainer._tokenize_prompts`를 직접 호출해서 얻었다. 별도로 tiny random-init 모델로 실제 Trainer 객체를 만들어 같은 값을 확인했다 |
| 실행 환경 | macOS 27.0.1 arm64, Apple M1 Max (10 core), 32 GiB RAM, CPU only, Python 3.12.13 |
| 보조 환경 | `AutoProcessor` 경로 확인용 private venv `/tmp/vf-research/venv-example-data` (공유 venv + Pillow 12.3.0 + torchvision 0.29.1 `--no-deps`) |
| 저장소 밖 산출물 | `/tmp/vf-research/scratch/example-data/` (스크립트 `q*.py`, `golden_lengths.csv`, `golden_result.json`, `golden_example_stats.json`) |
| 근거 표기 | `<pkg>==<ver> <site-packages 기준 경로>:<줄 범위> (<심볼>)` 또는 실험 ID(E1–E12). 태그는 VERIFIED(소스 확인 또는 실측), INFERRED(소스 기반 추론, 미실행), UNKNOWN |
| 검증 | 2026-10-04 `verify-example-model-dataset`가 핵심 주장 21개를 소스 재확인과 독립 실험(V1–V15)으로 다시 검증했다. 수정한 곳과 근거는 문서 끝 "검증 로그" 절에 있다 |

### 실험 목록

모든 실험은 `HF_HOME=/tmp/vf-research/hf HF_DATASETS_CACHE=/tmp/vf-research/scratch/example-data/ds-cache /tmp/vf-research/.venv/bin/python <script>` 형식으로 실행했다. E2(성공 경로)와 E8만 private venv의 python을 사용했다.

| ID | 스크립트 | 내용 |
|---|---|---|
| E1 | `q1_tokenizer.py` | tokenizer class, vocab, special token, 기본 special token 추가 여부 |
| E2 | `q1_processor.py` | `AutoProcessor` 결과. 공유 venv에서는 Pillow/torchvision이 없어 실패하고, private venv에서는 `Qwen3VLProcessor`로 로드된다 |
| E3 | `q1_regex_diff.py` | transformers tokenizer와 raw `tokenizer.json`(`tokenizers.Tokenizer.from_file`)의 결과 차이 |
| E4 | `q1_hashes.py` | 파일 SHA256과 Hub tree의 git blob id/LFS sha256 대조 |
| E5 | `q2_template.py` 및 추가 실행 | template 렌더링 17가지 경우, assistant mask, TRL template 호환성 검사 |
| E6 | `q4_dataset.py`, `q4_stats.py`, `q4_hub.py`, `q4_parquet.py` | 데이터셋 형식, 로딩, 컬럼 통계, 중복, parquet 변환본 |
| E7 | `q5_golden.py` | **golden 통계**: 실제 TRL 함수로 전체 row 처리, content 손실 검사, 불변식 검사, CSV 생성 |
| E8 | `q5_vlm_parity.py` | VLM processor 경로와 tokenizer 경로의 token ID를 전체 row에서 비교 |
| E9 | `q5_trainer_parity.py` | 실제 `SFTTrainer`/`DPOTrainer`/`GRPOTrainer` 객체(tiny Qwen3, CPU)로 48 row 대조, collator shape, 기본 `max_length` 영향 |
| E10 | `q6_timing.py` | 처리량. `TOKENIZERS_PARALLELISM=true/false` 각각 3회 측정 후 중앙값 |
| E11 | `q8_config.py` | `AutoConfig` 기준 context 관련 값 |
| E12 | 즉석 실행 | `is_conversational`의 hash seed 의존성, SFT `enable_thinking=False`, `json.load` 실패, 원본을 DPO에 그대로 넣었을 때의 `extract_prompt` |

---

## 1. Tokenizer

| 사실 | 값 | 근거 | 태그 |
|---|---|---|---|
| 로드되는 class | `Qwen3_5Tokenizer` (`TokenizersBackend` 하위, `is_fast=True`). Hub의 `tokenizer_config.json`은 `"tokenizer_class": "Qwen2Tokenizer"`를 적고 있지만, `qwen3_5`가 "Hub tokenizer class가 틀린 model type" 목록에 있어 등록 class로 교체된다 | E1. transformers==5.18.0 `models/auto/tokenization_auto.py:295` (mapping `qwen3_5 → Qwen3_5Tokenizer`), `:378-429` (`MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS`, `qwen3_5`는 419행), `:843-870` (교체 로직) | VERIFIED |
| 실제 토큰화 파이프라인 | class가 `__init__`을 정의하므로 `tokenizer.json`에서는 vocab, merges, post_processor(그리고 padding/truncation 설정. 이 파일에서는 둘 다 `null`)를 가져온다. added token 33개도 `tokenizer.json`의 `added_tokens`에서 읽는다. normalizer(`NFC`), pre-tokenizer(`Split(PRETOKENIZE_REGEX)` + `ByteLevel`), decoder는 class 코드가 다시 만든다 | transformers==5.18.0 `tokenization_utils_tokenizers.py:103-197` (`convert_to_native_format`, 조건은 113-119행, padding/truncation은 149-158행), `tokenization_utils_base.py:1904-1914` (`added_tokens` 읽기), `models/qwen3_5/tokenization_qwen3_5.py:25,54-80` | VERIFIED (검증 시 "vocab, merges, post_processor만"을 보완) |
| regex 차이 | class regex는 `[\p{L}\p{M}]+`, `tokenizer.json`의 regex는 `\p{L}+`(`\p{M}` 없음)이다. 따라서 raw `tokenizers.Tokenizer.from_file("tokenizer.json")`은 결합 문자(mark)가 있는 문자열에서 결과가 다르다. 예: `"नमस्ते दुनिया"`는 6 대 10 token, `"สวัสดีครับ"`는 3 대 7 token. 이 데이터셋에서는 13,968개 시퀀스 모두 동일했다(검증 시 content 문자열 13,968개와 렌더링된 대화 문자열 13,968개 모두 차이 0). 참고로 Hub `tokenizer_config.json`에는 `pretokenize_regex` 키가 있고 값이 class regex(`[\p{L}\p{M}]+`)와 같다. transformers 5.18.0에는 이 키를 읽는 코드가 없다(grep 0건). 즉 class 동작은 Hub 설정에 적힌 의도와 일치하고, `tokenizer.json`만 다르다 | E3, E7 (`raw_tokenizers_json_id_diff_sequences = 0`), 검증 V1·V2 | VERIFIED |
| `len(tokenizer)` 대 config | `len(tok)=248,077` (BPE vocab 248,044 + added token 33개, ID 0–248,076 연속), `tok.vocab_size=248,044` (added token 제외). `config.text_config.vocab_size=248,320`이므로 embedding/`lm_head`에 쓰이지 않는 행이 243개 있다 (248,320 = 64×3,880) | E1, `config.json` | VERIFIED |
| BOS / EOS / PAD / UNK | `bos_token=None`, `eos_token="<\|im_end\|>"`(248046), `pad_token="<\|endoftext\|>"`(248044), `unk_token=None` | E1, `tokenizer_config.json` | VERIFIED |
| EOS 불일치 | `config.text_config.eos_token_id=248044`(`<\|endoftext\|>`)이고 tokenizer EOS는 248046(`<\|im_end\|>`)이다. `generation_config.json`의 `eos_token_id=[248046, 248044]` | `config.json`, `generation_config.json`, E11 | VERIFIED |
| added token | 33개. special 21개 (`<\|endoftext\|>`, `<\|im_start\|>`=248045, `<\|im_end\|>`, vision/audio/tts 계열)와 non-special 12개 (`<tool_call>`=248058, `</tool_call>`=248059, FIM 4개, `<\|repo_name\|>`, `<\|file_sep\|>`, `<tool_response>`, `</tool_response>`, `<think>`=248068, `</think>`=248069). `<think>`/`</think>`는 special이 아니므로 `skip_special_tokens=True`로 decode해도 남는다 | E1 | VERIFIED |
| `model_max_length` | 262,144. `tokenizer_config.json`에 명시된 값이고 `max_position_embeddings`와 같다. sentinel인 `VERY_LARGE_INTEGER=int(1e30)`가 아니다. 이 값을 넘는 길이를 truncation 없이 토큰화하면 warning만 남기고 자르지는 않는다 | `tokenizer_config.json`. transformers==5.18.0 `tokenization_utils_base.py:130` (`VERY_LARGE_INTEGER`), `:2950-2968` (`_eventual_warn_about_too_long_sequence`) | VERIFIED |
| `tokenizer(text)`의 special token 추가 | 추가하지 않는다. `tok("hello world")`와 `add_special_tokens=False`의 결과가 모두 `[14556, 1814]`이다. `add_bos_token=False`, `add_eos_token=False`이고 post_processor는 `ByteLevel` 하나뿐이다(TemplateProcessing 없음). `apply_chat_template(tokenize=True)`는 내부에서 `add_special_tokens=False`로 호출한다 | E1, `tokenizer.json`, transformers==5.18.0 `tokenization_utils_base.py:3124-3132` (`apply_chat_template`) | VERIFIED |
| content 안의 special token 문자열 | `split_special_tokens=False`이므로 content에 들어 있는 `"<\|im_end\|>"` 같은 문자열이 그 special token ID로 바뀐다 | E5 추가 실행 (`'a <\|im_end\|> b'` → `[…, 'a', 'Ġ', '<\|im_end\|>', 'Ġb', …]`) | VERIFIED |
| 숫자 | regex의 `\p{N}` 대안 때문에 숫자 1자리가 token 1개다 (`"1000000"` → 7 token) | E3 | VERIFIED |
| `tokenizer.json` 구성 | BPE, vocab 248,044, merges 247,587, `byte_fallback=false`, normalizer `NFC`, `truncation=null`, `padding=null` | `tokenizer.json` 직접 파싱 | VERIFIED |

### 1.1 파일 해시 (revision `2367e865…`)

| 파일 | bytes | SHA256 | Hub 대조 |
|---|---|---|---|
| `tokenizer.json` | 19,989,325 | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` | LFS `lfs_sha256`와 일치 |
| `tokenizer_config.json` | 1,165 | `792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b` | git blob `1d134cd2…`와 일치 |
| `chat_template.jinja` | 3,916 | `59a64ebb4df6d1489d09a91267cf3ceb106162d4a893c4f84833cfb8c897ff63` | git blob `c6a37693…`와 일치 |
| `config.json` | 2,784 | `407c46388b8fa2ae9bf69fe27d40af236d373e86a6b48f5284c86df5cd183633` | git blob `007c2eb8…`와 일치 |
| `generation_config.json` | 126 | `aaed5b26f1cae55c1ceb58fc483c3cd65ee8d386b61daec3d7a9df82432d9205` | git blob `06320848…`와 일치 |
| `vocab.json` / `merges.txt` | 6,722,759 / 3,353,259 | `ce99b4cb…` / `a9d356d7…` | git blob과 일치 |

E4로 확인했다(VERIFIED). `tokenizer_config.json`에는 `chat_template` 키가 없고 template은 `chat_template.jinja`에서 읽힌다. 로드된 `tok.chat_template` 문자열은 파일 내용과 같으며, 마지막 `\n`까지 포함해 SHA256이 `59a64ebb…`로 동일하다.

## 2. `chat_template.jinja` 분석

이 template은 일반 Qwen3/Qwen3.5 template과 구조가 다르다. TRL에 포함된 `chat_templates/qwen3.jinja`, `qwen3_5_think.jinja`, `qwen3_5_nothink.jinja`는 `<|im_end|>\n`을 5–6곳에서 출력하고(`qwen3.jinja` 5곳, `qwen3_5_think`/`qwen3_5_nothink` 각 6곳. 검증 시 "4–5곳"을 정정) `{% generation %}`이 없다. 이 template은 `<|im_end|>\n`이 0곳이고 generation marker가 있다(trl==1.14.1 `chat_templates/` 비교, VERIFIED). 다른 Qwen template으로 대체하면 길이와 mask가 달라진다(plan.md §6.4). 핵심 부분은 다음과 같다 (`chat_template.jinja:60-70`).

```jinja
{%- macro render_assistant_message(message) -%}
    {%- generation -%}
    {%- set content = render_content(message.content) -%}
    {%- set reasoning = message.reasoning_content if message.reasoning_content is string else '' -%}
    {{- '<|im_start|>assistant\n<think>' ~ reasoning ~ '</think>' ~ content -}}
    {%- if message.tool_calls is defined and message.tool_calls is iterable and message.tool_calls | length > 0 -%}
        {{- render_tool_calls(message.tool_calls) -}}
    {%- endif -%}
    {{- '<|im_end|>' -}}
    {%- endgeneration -%}
{%- endmacro -%}
```

`chat_template.jinja:76-81`(assistant가 아닌 메시지)와 `:92-97`(generation prompt):

```jinja
{%- for message in messages -%}
    {%- if message.role == 'assistant' -%}
        {{- render_assistant_message(message) -}}
    {%- else -%}
        {%- set body = render_content(message.content) -%}
        {{- '<|im_start|>' ~ message.role ~ '\n' ~ body -}}
```

```jinja
{%- if add_generation_prompt -%}
    {{- '<|im_start|>assistant\n' -}}
    {%- if enable_thinking is false -%}
        {{- '<think></think>' -}}
    {%- endif -%}
{%- endif -%}
```

| 질문 | 결과 | 근거 | 태그 |
|---|---|---|---|
| 지원 role | assistant를 제외한 모든 role 문자열을 `<\|im_start\|>{role}\n{content}<\|im_end\|>`로 렌더링한다 (`tool`, `developer`도 확인). TRL의 VLM 경로는 `prepare_multimodal_messages`에서 role을 system/user/assistant/tool로 제한하고 그 외는 `ValueError`를 낸다 | E5 case 12. template `:76-90`. trl==1.14.1 `data_utils.py:84-97` | VERIFIED |
| 빈 system 문자열 | **생략하지 않고 렌더링한다.** `{"role":"system","content":""}`는 `<\|im_start\|>system\n<\|im_end\|>`가 되며 4 token(`<\|im_start\|>`, `system`, `Ċ`, `<\|im_end\|>`)을 더한다. 공백만 있는 system도 그대로 렌더링한다 | E5 case 5 및 추가 실행 | VERIFIED |
| thinking 처리 | 모든 assistant turn은 `<\|im_start\|>assistant\n<think>{reasoning_content 또는 ''}</think>{content}[tool_calls]<\|im_end\|>`로 렌더링된다. `enable_thinking`은 generation prompt에만 영향을 준다. `enable_thinking=False`이면 generation prompt 뒤에 `<think></think>`가 붙고, `True`이거나 지정하지 않으면 아무것도 붙지 않는다. assistant turn 렌더링은 `enable_thinking` 값과 관계없이 같다 | E5 case 2-4, 7-8, 15 | VERIFIED |
| 이전 assistant turn의 reasoning 제거 | 하지 않는다. 앞선 turn의 `reasoning_content`도 그대로 남는다 (`<think>R1</think>`). content 안의 `<think>…</think>`도 해석하거나 제거하지 않는다 | E5 case 9, 10 | VERIFIED |
| tools | `tools` 인자를 주면 모든 메시지보다 **앞에** 별도의 system turn(`You are provided with the following tools:\n\n<tools>\n{json}…\n</tools>`)을 만든다. 사용자 system 메시지가 있으면 system turn이 2개가 된다. 메시지별 `tools`는 해당 메시지 body 뒤에 붙는다. tool call 형식은 `<tool_call><function=f><parameter=x>1</parameter></function></tool_call>` | E5 case 13-14 및 추가 실행. template `:29-35,72-74,82-87` | VERIFIED |
| `{% generation %}` marker | 있다. assistant turn 전체를 감싸므로 role header `<\|im_start\|>assistant\n`과 `<think></think>`, `<\|im_end\|>`까지 `assistant_masks=1`이 된다. TRL 검사 결과는 `has_generation_markers=True`, `supports_tool_calling=True`, `is_chat_template_prefix_preserving=True`, `is_chat_template_stop_token_trained=True`이고, `get_training_chat_template(tok)`는 `None`(패치 불필요)을 반환한다 | E5. trl==1.14.1 `chat_template_utils.py:37-42`, `:825-884`, `:886-953`, `:1105-1107` | VERIFIED |
| `<\|im_end\|>` 뒤 줄바꿈 | **없다.** 메시지는 `…<\|im_end\|><\|im_start\|>…`처럼 바로 이어지고, 렌더링 결과는 `<\|im_end\|>`(또는 generation prompt)로 끝난다. 파일 마지막의 `\n`은 출력되지 않는다(`-%}` trim, Jinja 기본 `keep_trailing_newline=False`) | E5 case 1, 7, 9. transformers==5.18.0 `utils/chat_template_utils.py:495-500` (`trim_blocks=True, lstrip_blocks=True`) | VERIFIED |
| `add_generation_prompt=True` | `<\|im_start\|>assistant\n` (3 token). `enable_thinking=False`이면 `<\|im_start\|>assistant\n<think></think>` (5 token) | E5 case 2-4 | VERIFIED |
| `content=None` | 빈 content로 렌더링한다 (`render_content`는 문자열도 iterable도 아니면 아무것도 출력하지 않는다) | E5 case 16-17 | VERIFIED |
| assistant mask 계산 | generation block이 macro 안에 있어도 transformers의 `AssistantTracker`는 최상위 render stream 기준 문자 위치를 기록하고, `char_to_token`으로 token mask를 만든다. 결과 mask는 위에 적은 범위와 같다 | transformers==5.18.0 `utils/chat_template_utils.py:407-422,437-477`, `tokenization_utils_base.py:3133-3150`. E5 | VERIFIED |

## 3. Template content 손실 검사

**방법 (E7):** row마다 [user(question)] + [assistant(chosen)] 또는 [assistant(rejected)]을 렌더링하고, 다음 기대 문자열과 **정확히 일치**하는지 비교했다.

```text
prompt   = "<|im_start|>user\n{question}<|im_end|><|im_start|>assistant\n"
chosen   = prompt + "<think></think>{chosen}<|im_end|>"
rejected = prompt + "<think></think>{rejected}<|im_end|>"
keep     = "<|im_start|>system\n{system}<|im_end|>" + prompt
```

그 외에 content가 원문 그대로 포함되는지, `decode(encode(text)) == text`인지, NFC 정규화로 원문이 바뀌는지, 33개 added token 문자열이나 Jinja 구문(`{{`, `{%`)이 들어 있는지도 검사했다.

| 검사 | 결과 (4,656 row) | 태그 |
|---|---|---|
| 구조 문자열 정확 일치 (prompt / prompt+chosen / prompt+rejected / keep-system) | 불일치 0 / 0 / 0 / 0 | VERIFIED |
| content가 원문 그대로 포함되지 않은 row (system, question, chosen, rejected) | 0, 0, 0, 0 | VERIFIED |
| decode 왕복 실패 (13,968 시퀀스) | 0 | VERIFIED |
| NFC가 아닌 content (question/chosen/rejected) | 0 / 0 / 0. NFC normalizer가 원문을 바꾸는 경우가 없다 | VERIFIED |
| added token 문자열 포함 (`<think>`, `<\|im_end\|>`, `<tool_call>` 등 33개) | 0 | VERIFIED |
| `{{` 포함 | chosen 3 row, rejected 2 row. Jinja는 데이터 값을 평가하지 않으므로 그대로 렌더링된다 (정확 일치 검사 통과, `{{ 1+1 }}` 실험도 그대로 출력) | VERIFIED |
| Unicode mark(category M) 포함 | chosen 1 row (row 2014, U+FE0F VARIATION SELECTOR-16). raw `tokenizer.json`과 ID가 같다 | VERIFIED |

**결론:** 이 template은 학습 대상 content를 지우거나 바꾸는 row가 하나도 없다(0/4,656). template이 **추가**하는 token은 assistant turn마다 붙는 `<think></think>` 2개와, keep 정책에서 생기는 빈 system 블록 4개다. 이것은 손실이 아니라 주입(injection)으로 따로 보고해야 한다.

## 4. 데이터셋 사실

| 항목 | 값 | 근거 | 태그 |
|---|---|---|---|
| 파일 | `secure_programming_dpo.json` 하나 (그 외 `README.md`, `.gitattributes`). 6,867,898 bytes, SHA256 `ad93a85feadcaee3f9cc2ff34899adcede280bb47a3ac82f680c5924754ce82c`. LFS가 아닌 일반 git blob이며 blob sha1 `b2a8b2cae63af35f96dfc75440f1b50fac6a7346`이 tree와 일치한다 | E6, tree 메타데이터 | VERIFIED |
| 형식 | 확장자는 `.json`이지만 내용은 **JSON Lines**다. 4,656줄이고 줄마다 객체 1개, 마지막에 `\n`, BOM과 CR 없음. `json.load()`는 `JSONDecodeError: Extra data`로 실패하고 `pyarrow.json.read_json`은 4,656 row를 읽는다 | E6, E12 | VERIFIED |
| `datasets.load_dataset` (non-streaming) | builder `json`, config `default`, split은 `train`만 있음, 4,656 row, 약 2.0–2.3 s (Hub 조회 포함), Arrow `dataset_size` 6,306,335 | E6 | VERIFIED |
| streaming | `IterableDataset`, features가 미리 알려져 있음, 4,656 row. streaming으로 읽은 모든 row가 원본 JSONL과 순서·값 모두 같다(4,656/4,656), non-streaming `Dataset`도 같다. 전체 순회 약 2.4 s | E6 및 추가 실행 | VERIFIED |
| 컬럼 dtype | `lang`, `vulnerability`, `system`, `question`, `chosen`, `rejected` 모두 `Value('string')`. 모든 row의 key 집합이 같다 | E6 | VERIFIED |
| null / 빈 문자열 | null은 모든 컬럼에서 0. **`system`은 4,656 row 모두 `""`**(빈 문자열, distinct 1개). 나머지 컬럼은 빈 문자열 0, 앞뒤 공백 0 | E6 | VERIFIED |
| `lang` 분포 | c++/python/java/javascript 각 424, c#/php/ruby/swift/go/kotlin 각 423, fortran 422 (11종) | E6 | VERIFIED |
| 중복 | 완전히 같은 row 2쌍: [2213, 2312], [2709, 4326]. 중복 question은 228 그룹(728 row, 초과분 500 row, 고유 question 4,156개, 최대 그룹 40 row: row 97부터 시작하는 kotlin 질문). 중복 chosen 97 그룹(초과 195), 중복 rejected 250 그룹(초과 637) | E6 | VERIFIED |
| chosen == rejected | 26 row: 227, 284, 706, 788, 854, 909, 1099, 1327, 1363, 1450, 1659, 1909, 2176, 2300, 2431, 2851, 3104, 3376, 3547, 3564, 3740, 3743, 3863, 4204, 4435, 4642. DPO 기준으로 margin이 0인 쌍이다 | E6 | VERIFIED |
| 모든 응답 끝 | chosen과 rejected 모두 4,656 row가 코드 펜스(백틱 3개)로 끝난다(VERIFIED). 이 사실로 보아 생성 도중 잘린 응답은 없어 보인다(INFERRED, 내용 수준 검사는 하지 않음) | E7 보조 분석, 검증 V6 | VERIFIED / INFERRED |
| Parquet 변환본 | `refs/convert/parquet` → commit `1cc07a693ee6d376fd33f1fa80596537a3e43c74` (2024-02-29 15:24:19 UTC). 파일 `default/train/0000.parquet`, 2,564,627 bytes, SHA256 `eb9b1f8b1541d99e23c8a7c1bb562b3a9de57496c31b7fa57d768f60cdeaf179`, 4,656 row, row group 5개. **row 순서와 값이 JSONL과 완전히 같다** | E6 (`q4_hub.py`, `q4_parquet.py`) | VERIFIED |
| Parquet 변환본의 원본 revision | commit 메시지에 기록이 없다. main의 마지막 commit `81aeacf…`(15:24:07 UTC) 12초 뒤에 만들어졌으므로 이 revision에서 변환했다고 본다. 내용이 같다는 점은 실측으로 확인했다 | E6 | INFERRED |
| main 이력 | commit 8개. 고정 revision `81aeacf…`가 2026-10-04 기준 main head다 (마지막 수정 2024-02-29) | E6 | VERIFIED |

### 4.1 문자 길이 통계 (Python `len`, code point 기준, p50/p99는 numpy linear)

| 컬럼 | min | max (row) | mean | p50 | p99 | 합계 | non-ASCII row |
|---|---|---|---|---|---|---|---|
| `lang` | 2 | 10 | 4.7 | 4 | 10 | 22,012 | 0 |
| `vulnerability` | 51 | 237 (2860) | 117.2 | 113 | 203 | 545,770 | 4 |
| `system` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `question` | 50 | 1,007 (3718) | 286.7 | 249 | 691 | 1,334,780 | 2 |
| `chosen` | 55 | 8,925 (2077) | 534.1 | 461 | 1,606 | 2,486,796 | 233 |
| `rejected` | 63 | 8,003 (706) | 387.3 | 305 | 1,236 | 1,803,137 | 62 |

**주의:** 문자 수 순위와 token 수 순위는 다르다. 문자 수가 가장 긴 chosen(row 2077, 8,925자)은 1,861 token이고, token 수가 가장 긴 chosen(row 2355)은 2,062자에 2,001 token이다. 상위 outlier는 대부분 `100000…000`처럼 숫자가 길게 이어지는 문자열이고, 이 tokenizer는 숫자 1자리를 1 token으로 만든다. 전체 chosen 대화 텍스트의 평균은 4.14 chars/token(4,166,120자 / 1,007,173 token)이다.

## 5. GOLDEN token 길이 통계 (전체 4,656 row)

### 5.0 방법과 정책

- **매핑:** `system → system message`, `question → user`, `chosen → assistant`, `rejected → assistant`. `lang`과 `vulnerability`는 사용하지 않는다(plan.md §7.2). 매핑 뒤에는 **원본 컬럼을 모두 제거**했다(`remove_columns=raw.column_names`). 이유는 §구현 시사점 R3 참고.
- **빈 system 정책 (명시):**
  - 1차 정책 **`omit`**: system 값이 `None`이거나 정확히 `""`이면 system 메시지를 만들지 않는다. 이 데이터셋은 모든 row가 해당하므로 prompt는 `[user]`다.
  - 비교 정책 **`keep`**: 항상 `{"role":"system","content":system}`을 넣는다. 이 template에서는 prompt를 포함하는 모든 시퀀스가 정확히 **+4 token**이 된다(전체 row에서 확인).
- `chat_template_kwargs={}`(즉 `enable_thinking`을 지정하지 않음), `tools=None`, `max_length=None`, `packing=False`.
- **실제 TRL 함수 호출 (E7):** 필요한 속성만 가진 stub을 `self`로 넘겨 unbound method를 호출했다.
  - SFT: `SFTTrainer._prepare_dataset(SimpleNamespace(_tokenizer=tok, chat_template=None, completion_only_loss=True), ds, tok, SFTConfig(max_length=None, …), packing=False, formatting_func=None, dataset_name="train")`. stub 값은 실제 `__init__`이 정하는 값과 같다(trl==1.14.1 `trainer/sft_trainer.py:1213-1217`, `:1265-1268`).
  - DPO: `DPOTrainer._prepare_dataset(SimpleNamespace(_tokenizer=tok), ds, tok, DPOConfig(max_length=None, …), "train")`.
  - GRPO: `GRPOTrainer._tokenize_prompts(SimpleNamespace(processing_class=tok, _is_vlm=False, environment_factories=None, tools=[], chat_template=None, chat_template_kwargs={}), prompts[i:i+64])`.
- **교차 검증:** E9에서 tiny 모델로 실제 Trainer 객체를 만들어 48 row(상위 outlier, 중복, 동일 쌍, 무작위 30개 포함)의 길이가 모두 같음을 확인했다. E8에서는 VLM processor 경로가 전체 row에서 같은 ID를 내는 것을 확인했다.
- **통계 정의:** count, min, max(최대값을 갖는 모든 row index), mean, p50/p90/p95/p99는 `numpy.percentile(a, q)`(method `linear`, numpy 기본값). 괄호 안의 NR은 nearest-rank(`method="inverted_cdf"`). total은 token 합계. row index는 데이터셋 순서 기준 0부터 센다.

TRL SFT의 prompt-completion 토큰화 핵심 (trl==1.14.1 `trainer/sft_trainer.py:1550-1587`, 일부 생략):

```python
if "prompt" in example:  # prompt-completion case
    if is_conversational(example):
        prompt_ids = _tokenize(processing_class, example["prompt"],
                               add_generation_prompt=True, **apply_chat_template_kwargs)["input_ids"]
        prompt_completion_processed = _tokenize(processing_class, example["prompt"] + example["completion"],
                               return_assistant_tokens_mask=assistant_only_loss, **apply_chat_template_kwargs)
        prompt_completion_ids = prompt_completion_processed["input_ids"]
    ...
    if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
        logger.warning("Mismatch between tokenized prompt and the start of tokenized prompt+completion. ...")
    completion_mask = [0] * len(prompt_ids) + [1] * (len(prompt_completion_ids) - len(prompt_ids))
```

DPO도 같은 방식으로 `prompt_ids`를 만든 뒤 `chosen_ids = prompt_chosen_ids[len(prompt_ids):]`, `rejected_ids = prompt_rejected_ids[len(prompt_ids):]`를 저장한다(trl==1.14.1 `trainer/dpo_trainer.py:1046-1092`). GRPO는 `apply_chat_template(conversation=prompts, add_generation_prompt=True, tokenize=True, return_dict=True, tools=None, chat_template=None, **chat_template_kwargs)`를 batch로 호출한다(trl==1.14.1 `trainer/grpo_trainer.py:1769-1846`).

### 5.1 주 결과 (정책 `omit`, VERIFIED, E7)

| branch | count | min | max (row) | mean | p50 | p90 | p95 | p99 (NR) | total |
|---|---|---|---|---|---|---|---|---|---|
| a) GRPO prompt (`add_generation_prompt=True`) | 4,656 | 18 | **268** (2355, 3169) | 70.098 | 61 | 123 | 137 | 165.45 (166) | 326,378 |
| b) SFT 전체 시퀀스 | 4,656 | 51 | **2,272** (2355) | 216.317 | 207 | 316 | 355 | 453.35 (455) | 1,007,173 |
| b) SFT completion = loss token | 4,656 | 17 | **2,004** (2355) | 146.219 | 134 | 230 | 263.25 (264) | 353.8 (356) | 680,795 |
| c) DPO prompt | GRPO prompt와 row별로 같음 | | 268 (2355, 3169) | | | | | | 326,378 |
| c) DPO prompt+chosen | SFT 전체와 row별로 같음 | | 2,272 (2355) | | | | | | 1,007,173 |
| c) DPO prompt+rejected | 4,656 | 49 | **2,272** (3169) | 180.834 | 166 | 267.5 (268) | 300 | 395.7 (399) | 841,963 |
| c) DPO pair max = max(chosen, rejected) | 4,656 | 53 | **2,272** (2355, 3169) | 221.071 | 208 | 319 | 360 | 495.45 (496) | 1,029,308 |
| c) DPO chosen completion | SFT completion과 같음 | | 2,004 (2355) | | | | | | 680,795 |
| c) DPO rejected completion | 4,656 | 21 | 2,007 (2707) | 110.736 | 93 | 182 | 204 | 288.45 (289) | 515,585 |

### 5.2 변형 (모두 전체 row에서 상수 offset 확인, VERIFIED)

| 변형 | 차이 | max | p50 | p99 | total |
|---|---|---|---|---|---|
| GRPO prompt `keep` | +4 | 272 (2355, 3169) | 65 | 169.45 | 345,002 |
| GRPO prompt `omit` + `chat_template_kwargs={"enable_thinking": False}` | +2 | 270 | 63 | 167.45 | 335,690 |
| SFT 전체 / DPO prompt+chosen `keep` | +4 | 2,276 (2355) | 211 | 457.35 | 1,025,797 |
| DPO prompt+rejected `keep` | +4 | 2,276 (3169) | 170 | 399.7 | 860,587 |
| DPO pair max `keep` | +4 | 2,276 (2355, 3169) | 212 | 499.45 | 1,047,932 |
| SFT/DPO completion `keep` | 0 | 2,004 | 134 | 353.8 | 680,795 |
| SFT에 example별 `chat_template_kwargs={"enable_thinking": False}` | 전체 길이 0, completion −2 (`<think></think>`가 prompt 쪽으로 이동) | — | — | — | — |

SFTConfig와 DPOConfig에는 `chat_template_kwargs` 필드가 없다. 이 두 Trainer에서는 데이터셋의 example별 `chat_template_kwargs` 컬럼으로만 넣을 수 있다(trl==1.14.1 `trainer/sft_trainer.py:1543-1548`, `trainer/dpo_trainer.py:1046-1049`). GRPO만 config 필드가 있다(`trainer/grpo_config.py:561`).

### 5.3 branch별 상위 10 row (길이 내림차순, 같은 길이면 index 오름차순)

| branch | row(길이) |
|---|---|
| GRPO prompt `omit` | 2355(268), 3169(268), 2905(266), 3718(226), 3136(215), 2871(212), 3618(212), 4229(207), 4015(203), 1727(201) |
| SFT 전체 = DPO prompt+chosen | 2355(2272), 2905(2265), 2200(2091), 591(2073), 4143(2061), 4400(2056), 1261(2041), 2967(2038), 2077(1905), 945(1491) |
| SFT completion (loss) | 2355(2004), 2905(1999), 4400(1996), 1261(1987), 2967(1987), 4143(1986), 2200(1984), 591(1982), 2077(1864), 945(1443) |
| DPO prompt+rejected | 3169(2272), 2355(2270), 2905(2267), 2200(2092), 2707(2090), 591(2088), 4143(2060), 1261(2053), 3708(2046), 4170(2042) |
| DPO pair max | 2355(2272), 3169(2272), 2905(2267), 2200(2092), 2707(2090), 591(2088), 4143(2061), 4400(2056), 1261(2053), 3708(2046) |
| DPO rejected completion | 2707(2007), 3708(2005), 3169(2004), 4005(2003), 2355(2002), 4170(2002), 2905(2001), 1261(1999), 591(1997), 2967(1987) |

언어별 SFT 전체 길이 max/mean: go 2272/207.6, swift 2091/261.8, javascript 2061/221.1, c# 2038/259.0, kotlin 1905/167.0, fortran 1491/133.9, ruby 1138/195.6, java 1068/220.7, c++ 743/278.0, php 600/229.1, python 555/205.5.

### 5.4 전체 row에서 성립하는 불변식 (4,656/4,656, VERIFIED, E7)

`n_q`, `n_c`, `n_r`은 각각 `tok(question|chosen|rejected, add_special_tokens=False)`의 길이다.

| 불변식 | 의미 |
|---|---|
| `grpo_prompt_omit = n_q + 7` | `<\|im_start\|>`,`user`,`\n` + question + `<\|im_end\|>` + `<\|im_start\|>`,`assistant`,`\n` |
| `sft_completion = n_c + 3` | `<think>`,`</think>` + chosen + `<\|im_end\|>` |
| `sft_total_omit = n_q + n_c + 10`, `dpo_rejected_total_omit = n_q + n_r + 10` | template 오버헤드 10 |
| `keep = omit + 4` (prompt를 포함하는 모든 시퀀스) | 빈 system 블록 |
| `nothink GRPO prompt = omit + 2` | `<think></think>` |
| `dpo_prompt = grpo_prompt` (token ID까지 동일), `dpo_chosen_total = sft_total` | 같은 렌더링 경로 |
| SFT `labels`의 completion 구간이 연속 suffix이고, loss 위치 수(`labels[1:] != -100`) = completion 길이 | prompt 첫 token은 항상 mask된다 |
| TRL "Mismatch between tokenized prompt…" warning | SFT 0건, DPO 0건 (모든 row에서 prompt가 prompt+completion의 prefix) |

이 분해식은 **이 template, 이 데이터에서만** 성립하는 회귀 검사용 값이다. content 앞뒤 공백이 0이고 경계마다 added token이 있어서 BPE 병합이 경계를 넘지 않기 때문이다. 일반 계산식으로 쓰면 안 된다(plan.md §7.3).

### 5.5 Parity 결과

| 검사 | 결과 | 근거 | 태그 |
|---|---|---|---|
| VLM processor 경로 (`AutoProcessor` → `Qwen3VLProcessor`, `_is_vlm=True`, `prepare_multimodal_messages`)와 tokenizer 경로 비교 | SFT `input_ids`/`labels`, DPO 3개 ID, GRPO prompt ID 모두 4,656/4,656 row 동일 (`omit`, `keep` 둘 다). SFT `input_ids` 다이제스트도 같다. VLM 경로 GRPO는 `mm_token_type_ids` 필드를 추가로 반환한다. 검증 V8에서 `processing_class=Qwen3VLProcessor`로 실제 `SFTTrainer`/`DPOTrainer`/`GRPOTrainer`를 만들어 같은 결과(모든 필드 4,656/4,656, 두 정책)를 재현했다. 이 processor는 torchvision이 있어야 로드된다(R2 참고) | E8, 검증 V8·V12 | VERIFIED |
| 실제 `SFTTrainer` (tiny Qwen3, `processing_class=tok`) | 48/48 row에서 전체 길이와 completion 일치. `completion_only_loss=True`, `chat_template` override 없음(None), collator `DataCollatorForLanguageModeling`, `pad_token_id=248044` | E9 | VERIFIED |
| SFT loss token 변형 | prompt-completion + `assistant_only_loss=True` → completion과 같음. `messages`(LM) 형식 + `assistant_only_loss=True` → **completion + 3** (`<\|im_start\|>assistant\n` header 포함). `messages` 형식 기본값 → 전체 길이 − 1 (`completion_only_loss=False`, shift 때문에 1 감소). 전체 길이는 형식과 관계없이 같다. 검증 V9에서 네 경우 모두 전체 4,656 row로 재현했다 | E9 (48/48), 검증 V9 (4,656/4,656). trl==1.14.1 `trainer/sft_trainer.py:187` (`labels[..., 1:]`, `_chunked_cross_entropy_loss`. `SFTConfig.loss_type` 기본값이 `None → "chunked_nll"`이므로 이것이 기본 경로다, `trainer/sft_config.py:332-334`), `:1641-1654` (`build_labels`). `loss_type="nll"`이면 transformers==5.18.0 `loss/loss_utils.py:61-64` (`ForCausalLMLoss`)가 같은 shift를 한다 | VERIFIED |
| 실제 `DPOTrainer` | 48/48 row에서 prompt, chosen total, rejected total 일치. 컬럼 `prompt_ids/chosen_ids/rejected_ids` | E9 | VERIFIED |
| 실제 `GRPOTrainer._tokenize_prompts` | 48/48 row에서 일치. `chat_template` override 없음, `chat_template_kwargs={}`, `max_prompt_length` 필드 없음 | E9 | VERIFIED |
| collator shape (golden) | SFT rows [2355, 2905] → `input_ids [2, 2272]`. DPO rows [2355, 3169] → `input_ids/attention_mask/completion_mask [4, 2272]`(chosen 2 + rejected 2, 4개 시퀀스의 최대 길이로 padding). DPO row 227(동일 쌍) → `[2, 239]` | E9, 검증 V7(실제 Trainer의 `data_collator`로 재현). trl==1.14.1 `trainer/dpo_trainer.py:145-206` (`DataCollatorForPreference.torch_call`) | VERIFIED |

### 5.6 TRL 기본 길이 제한에 걸리는 row (무절단 계약 위반 규모)

| 기준 L | SFT 전체 > L | DPO pair max > L | 비고 |
|---|---|---|---|
| 512 (GRPO `max_completion_length` 기본값과 같은 크기) | 31 | 43 | |
| 1024 (`SFTConfig.max_length`, `DPOConfig.max_length` 기본값) | **17** | **28** | SFT: 438, 479, 591, 706, 820, 945, 997, 1261, 2077, 2200, 2355, 2637, 2905, 2967, 4003, 4143, 4400. DPO에서 rejected 때문에만 넘는 row 11개: 292, 1976, 2640, 2680, 2707, 3169, 3202, 3708, 4005, 4170, 4597 |
| 2048 | 6 | 9 | |
| 4096 | 0 | 0 | |

E9에서 기본 `SFTConfig()`(max_length=1024)는 48개 중 6 row를 `keep_start`로 잘랐고 row를 삭제하지는 않았다. 기본 `DPOConfig()`는 collator의 `max_length=1024`로 batch 단계에서 자른다. DPO pair 중 rejected가 chosen보다 긴 경우는 411건, 길이가 같은 경우는 184건(동일 쌍 26건 포함)이다.

검증 V9에서 기본 설정으로 전체 4,656 row를 다시 처리했다. `SFTConfig()`는 `_prepare_dataset` 단계에서 17 row를 1,024 token으로 잘라 모두 11,127 token을 버렸고 row 수는 4,656으로 유지됐다. `DPOConfig()`는 dataset 단계에서는 자르지 않았고(길이가 golden과 같음), 실제 Trainer의 collator(`max_length=1024`)가 28 pair를 잘랐다. (VERIFIED)

**주의 (검증 시 추가).** 정수 `max_length`에서는 잘림뿐 아니라 **길이 기준 row 삭제**도 일어난다. 이 데이터는 prompt 최대가 268 token이라 삭제가 0건이었을 뿐이다.
- SFT: truncation 뒤 loss 대상 token이 하나도 남지 않은 example을 삭제한다(trl==1.14.1 `trainer/sft_trainer.py:1676-1683`, `dataset.filter(lambda example: any(label != -100 ...))`).
- DPO: `truncation_mode="keep_start"`이면 `len(prompt_ids) >= max_length`인 example을 삭제한다(trl==1.14.1 `trainer/dpo_trainer.py:1100-1106`).
- 두 filter 모두 `max_length=None`이면 실행되지 않는다(SFT 1659행, DPO 1103행 조건). (VERIFIED, 소스)

### 5.7 Per-row CSV (저장소에 커밋하지 않음)

- 경로: `/tmp/vf-research/scratch/example-data/golden_lengths.csv`. E7 `q5_golden.py`가 생성한다.
- 형식: UTF-8, 쉼표 구분, `\n` 줄끝(Python `csv.writer(lineterminator="\n")`), header 1줄 + 데이터 4,656줄, 283,440 bytes.
- **SHA256: `d9a2e7b74f315e4e8e2a6604e825d078a53671e4337c3428dfc60dd1f6c59c7d`**
- 컬럼과 첫 데이터 줄:

```text
row,lang,n_question,n_chosen,n_rejected,grpo_prompt_omit,grpo_prompt_keep,sft_total_omit,sft_total_keep,sft_completion,dpo_prompt_omit,dpo_chosen_total_omit,dpo_rejected_total_omit,dpo_prompt_keep,dpo_chosen_total_keep,dpo_rejected_total_keep
0,c++,91,96,98,98,102,197,201,99,98,197,199,102,201,203
```

- Token ID 다이제스트: `sha256(json.dumps(list_of_list_of_int).encode())`. Python 기본 separator(`", "`, `": "`)를 쓰고 row 순서는 데이터셋 순서다.
  - GRPO prompt IDs (`omit`): `fe2aae558681a513b806792a974c6b4a6561d7f8a9afd21d7477b2f9083208d1`
  - SFT `input_ids` (`omit`): `bc12f48d22f4b0b8d2c17afdf51c9dfa426cbac3859c5e0ec5c91a0e38daa417`
  - SFT `input_ids` (`keep`): `de059a3f16be60878803630107523041868a37ab484046704cedc346406b1ee8`

### 5.8 `golden-example-stats`

아래 JSON은 `golden_lengths.csv`에서 다시 계산해 만든 것이다(`q9_build_json.py`). 생성 과정에서 `keep = omit + 4`, `dpo_prompt = grpo_prompt`, `dpo_chosen_total = sft_total`을 assert로 확인했다. 테스트는 count, min, max, max_rows, total을 정확히 비교하고, 분위수는 같은 numpy 정의를 쓸 때만 정확히 비교한다.

<!-- golden-example-stats:begin -->
```json
{
 "schema": "vramforge/golden-example-stats@1",
 "generated_on": "2026-10-04",
 "env": {
  "transformers": "5.18.0",
  "trl": "1.14.1",
  "tokenizers": "0.23.2",
  "datasets": "5.0.1",
  "huggingface_hub": "1.33.0",
  "jinja2": "3.1.6",
  "numpy": "2.5.3"
 },
 "model": {
  "id": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
  "revision": "2367e865d009c13ac81713a2878291d33ab28177",
  "sha256": {
   "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
   "tokenizer_config.json": "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
   "chat_template.jinja": "59a64ebb4df6d1489d09a91267cf3ceb106162d4a893c4f84833cfb8c897ff63",
   "config.json": "407c46388b8fa2ae9bf69fe27d40af236d373e86a6b48f5284c86df5cd183633"
  },
  "tokenizer_class": "Qwen3_5Tokenizer",
  "len_tokenizer": 248077,
  "config_vocab_size": 248320,
  "bos_token_id": null,
  "eos_token": "<|im_end|>",
  "eos_token_id": 248046,
  "pad_token": "<|endoftext|>",
  "pad_token_id": 248044,
  "config_text_eos_token_id": 248044,
  "generation_config_eos_token_id": [248046, 248044],
  "model_max_length": 262144,
  "max_position_embeddings": 262144,
  "has_generation_markers": true
 },
 "dataset": {
  "id": "CyberNative/Code_Vulnerability_Security_DPO",
  "revision": "81aeacf06cf43b16d7278a3a01f019a496a53c51",
  "file": "secure_programming_dpo.json",
  "file_format": "jsonl",
  "file_sha256": "ad93a85feadcaee3f9cc2ff34899adcede280bb47a3ac82f680c5924754ce82c",
  "file_bytes": 6867898,
  "split": "train",
  "rows": 4656,
  "system_empty_rows": 4656,
  "parquet_convert_revision": "1cc07a693ee6d376fd33f1fa80596537a3e43c74",
  "parquet_sha256": "eb9b1f8b1541d99e23c8a7c1bb562b3a9de57496c31b7fa57d768f60cdeaf179"
 },
 "mapping": {"system": "system", "prompt": "question", "chosen": "chosen", "rejected": "rejected"},
 "policy": {
  "primary_empty_system": "omit",
  "chat_template_kwargs": {},
  "max_length": null,
  "packing": false,
  "quantiles": "numpy.percentile(method='linear'); nearest_rank = method='inverted_cdf'"
 },
 "constants_tokens": {
  "prompt_overhead_omit": 7,
  "empty_system_block": 4,
  "enable_thinking_false_gen_prompt": 2,
  "completion_overhead": 3,
  "sft_total_minus_content": 10
 },
 "grpo_prompt": {
  "omit": {"count": 4656, "min": 18, "max": 268, "max_rows": [2355, 3169], "p50": 61.0, "p99": 165.45, "total": 326378, "mean": 70.098, "p90": 123.0, "p95": 137.0, "nearest_rank": {"p50": 61, "p90": 123, "p95": 137, "p99": 166}, "top10_rows": [2355, 3169, 2905, 3718, 3136, 2871, 3618, 4229, 4015, 1727]},
  "keep": {"count": 4656, "min": 22, "max": 272, "max_rows": [2355, 3169], "p50": 65.0, "p99": 169.45, "total": 345002}
 },
 "sft": {
  "total_omit": {"count": 4656, "min": 51, "max": 2272, "max_rows": [2355], "p50": 207.0, "p99": 453.35, "total": 1007173, "mean": 216.317, "p90": 316.0, "p95": 355.0, "nearest_rank": {"p50": 207, "p90": 316, "p95": 355, "p99": 455}, "top10_rows": [2355, 2905, 2200, 591, 4143, 4400, 1261, 2967, 2077, 945]},
  "total_keep": {"count": 4656, "min": 55, "max": 2276, "max_rows": [2355], "p50": 211.0, "p99": 457.35, "total": 1025797},
  "completion_loss_tokens": {"count": 4656, "min": 17, "max": 2004, "max_rows": [2355], "p50": 134.0, "p99": 353.8, "total": 680795, "mean": 146.219, "p90": 230.0, "p95": 263.25, "nearest_rank": {"p50": 134, "p90": 230, "p95": 264, "p99": 356}, "top10_rows": [2355, 2905, 4400, 1261, 2967, 4143, 2200, 591, 2077, 945]}
 },
 "dpo": {
  "prompt_omit": {"count": 4656, "min": 18, "max": 268, "max_rows": [2355, 3169], "p50": 61.0, "p99": 165.45, "total": 326378},
  "chosen_total_omit": {"count": 4656, "min": 51, "max": 2272, "max_rows": [2355], "p50": 207.0, "p99": 453.35, "total": 1007173, "mean": 216.317, "p90": 316.0, "p95": 355.0, "nearest_rank": {"p50": 207, "p90": 316, "p95": 355, "p99": 455}, "top10_rows": [2355, 2905, 2200, 591, 4143, 4400, 1261, 2967, 2077, 945]},
  "rejected_total_omit": {"count": 4656, "min": 49, "max": 2272, "max_rows": [3169], "p50": 166.0, "p99": 395.7, "total": 841963, "mean": 180.834, "p90": 267.5, "p95": 300.0, "nearest_rank": {"p50": 166, "p90": 268, "p95": 300, "p99": 399}, "top10_rows": [3169, 2355, 2905, 2200, 2707, 591, 4143, 1261, 3708, 4170]},
  "pair_max_omit": {"count": 4656, "min": 53, "max": 2272, "max_rows": [2355, 3169], "p50": 208.0, "p99": 495.45, "total": 1029308, "mean": 221.071, "p90": 319.0, "p95": 360.0, "nearest_rank": {"p50": 208, "p90": 319, "p95": 360, "p99": 496}, "top10_rows": [2355, 3169, 2905, 2200, 2707, 591, 4143, 4400, 1261, 3708]},
  "chosen_total_keep": {"count": 4656, "min": 55, "max": 2276, "max_rows": [2355], "p50": 211.0, "p99": 457.35, "total": 1025797},
  "rejected_total_keep": {"count": 4656, "min": 53, "max": 2276, "max_rows": [3169], "p50": 170.0, "p99": 399.7, "total": 860587},
  "pair_max_keep": {"count": 4656, "min": 57, "max": 2276, "max_rows": [2355, 3169], "p50": 212.0, "p99": 499.45, "total": 1047932}
 },
 "rows_over_length": {
  "512": {"sft_total": 31, "dpo_pair_max": 43},
  "1024": {"sft_total": 17, "dpo_pair_max": 28},
  "2048": {"sft_total": 6, "dpo_pair_max": 9},
  "4096": {"sft_total": 0, "dpo_pair_max": 0}
 },
 "collator_shapes": {"sft_rows_2355_2905": [2, 2272], "dpo_rows_2355_3169": [4, 2272], "dpo_row_227": [2, 239]},
 "per_row_csv": {
  "columns": ["row", "lang", "n_question", "n_chosen", "n_rejected", "grpo_prompt_omit", "grpo_prompt_keep", "sft_total_omit", "sft_total_keep", "sft_completion", "dpo_prompt_omit", "dpo_chosen_total_omit", "dpo_rejected_total_omit", "dpo_prompt_keep", "dpo_chosen_total_keep", "dpo_rejected_total_keep"],
  "rows": 4656,
  "sha256": "d9a2e7b74f315e4e8e2a6604e825d078a53671e4337c3428dfc60dd1f6c59c7d"
 },
 "token_id_digests_sha256": {
  "grpo_prompt_ids_omit": "fe2aae558681a513b806792a974c6b4a6561d7f8a9afd21d7477b2f9083208d1",
  "sft_input_ids_omit": "bc12f48d22f4b0b8d2c17afdf51c9dfa426cbac3859c5e0ec5c91a0e38daa417",
  "sft_input_ids_keep": "de059a3f16be60878803630107523041868a37ab484046704cedc346406b1ee8"
 }
}
```
<!-- golden-example-stats:end -->

## 6. 처리량 (worker timeout과 진행 이벤트 주기 산정용)

Apple M1 Max, 단일 프로세스, `dataset_num_proc=None`, datasets 캐시 비활성화(`datasets.disable_caching()`), 정책 `omit`. 3회 측정의 중앙값이다(E10). 대상은 4,656 row다.

| 단계 | `TOKENIZERS_PARALLELISM=true` | `=false` |
|---|---|---|
| `import torch, transformers, trl` | 3.43 s | 3.89 s |
| `AutoTokenizer.from_pretrained` (로컬 캐시) | 1.23 s | 1.28 s |
| `load_dataset` (Hub 조회 + Arrow 생성) | 1.97 s | 1.93 s |
| TRL SFT `_prepare_dataset` | 4.59 s (1,015 row/s, 약 220k token/s) | 3.86 s (1,208 row/s, 약 261k token/s) |
| TRL DPO `_prepare_dataset` | 4.83 s (963 row/s) | 4.53 s (1,028 row/s) |
| TRL GRPO `_tokenize_prompts` (batch 64) | 0.19 s | 0.46 s |
| TRL GRPO `_tokenize_prompts` (batch 1) | 0.64 s | 0.50 s |
| Jinja 렌더링만 (chosen 대화 4,656개) | 0.19 s | 0.12 s |
| tokenizer 호출만, row마다 | 1.28 s | 1.09 s |
| tokenizer 호출만, 전체 batch | 0.24 s | 1.03 s |

- VLM processor 경로(E8, 1회 측정)는 SFT 5.6–5.9 s, DPO 7.0–7.3 s, GRPO 0.38 s로 tokenizer 경로보다 느리다. 검증 V8(Trainer 생성 시간 포함, 같은 프로세스에서 두 경로 비교)에서는 SFT 1.20–1.33배, DPO 1.26–1.43배, GRPO 1.22–1.29배였다. 따라서 배율은 "1.4–1.8배"가 아니라 **약 1.2–1.8배**(측정 방법과 부하에 따라 다름)로 본다.
- 검증 V10 재측정(같은 M1 Max, 다른 에이전트가 동시에 실행 중이라 load average 9–14, 3회 중앙값): SFT `_prepare_dataset` 4.45–4.89 s(952–1,047 row/s), DPO 5.57–5.60 s(831–836 row/s), Jinja 렌더링 0.10–0.11 s, import 1.5–1.9 s(파일 캐시가 따뜻한 상태), tokenizer 1.24–1.44 s, `load_dataset` 2.0–2.1 s. 위 표와 ±20% 안에서 일치한다. 처리량 값은 부하에 민감하다.
- 비용 대부분은 Jinja나 BPE가 아니라 TRL의 `datasets.map(batched=False)` row별 오버헤드다. SFT 약 4 s 중 렌더링과 토큰화는 약 1.3 s다.
- streaming으로 전체를 한 번 순회하는 데 약 2.4 s가 걸렸다. golden 스크립트(E7) 전체의 최대 RSS는 1.13 GB였다. 여러 변형의 token ID 리스트를 모두 메모리에 두었으므로 상한 참고값이다.
- Docker linux/amd64 호스트의 처리량은 측정하지 않았다(UNKNOWN).

## 7. 렌더링·토큰화 실패 row

**0건 (VERIFIED, E7/E8/E9).** 근거는 다음과 같다.

1. 실제 TRL `map`이 SFT, DPO, GRPO와 `omit`, `keep` 조합 모두에서 예외 없이 끝났고 출력이 4,656 row였다. `max_length=None`이면 filter와 truncate 단계가 실행되지 않는다.
2. 독립적인 row별 참조 루프(try/except, row당 렌더링 4회와 토큰화 9회)에서 예외가 0건이었다.
3. TRL prefix mismatch warning이 0건이었다.
4. VLM 경로에서도 실패가 없었다.

데이터 품질 경고 후보(실패는 아님): 완전히 같은 row 2쌍, chosen==rejected 26쌍, 중복 question 500 row. no-truncation 계약 때문에 이 row들을 제거하지 않고 경고로만 남겨야 한다.

## 8. Context 상한 관련 config

| 항목 | 값 | 근거 | 태그 |
|---|---|---|---|
| `text_config.max_position_embeddings` | 262,144 (class 기본값은 32,768이고 checkpoint가 덮어쓴다) | `config.json`, E11. transformers==5.18.0 `models/qwen3_5/configuration_qwen3_5.py:89` | VERIFIED |
| `rope_parameters` | `rope_type="default"`(YaRN 같은 scaling 없음), `rope_theta=1e7`, `partial_rotary_factor=0.25` → head_dim 256 중 64차원만 회전, `mrope_section=[11,11,10]`(합 32 = 64/2), `mrope_interleaved=true`. 이 값들은 checkpoint `config.json`의 `text_config.rope_parameters`에서 온다. class 기본값은 `rope_theta=10000.0`이므로 반드시 checkpoint config를 읽어야 한다 | E11, 검증 V15. `configuration_qwen3_5.py:108-111`은 `mrope_*` key의 rope 검증 제외(108행)와 `partial_rotary_factor` 기본값 0.25(111행)를 정의한다 | VERIFIED |
| sliding window | 없음. `Qwen3_5TextConfig`에는 `sliding_window` 필드 자체가 없고 `modeling_qwen3_5.py`에도 sliding 관련 코드가 없다(grep 0건). E11의 `None`은 `getattr(cfg, "sliding_window", None)`의 기본값일 뿐이다 | E11, 검증 V15 (`hasattr(text_config, "sliding_window") == False`) | VERIFIED (검증 시 근거 정정) |
| tokenizer `model_max_length` | 262,144 (§1) | E1 | VERIFIED |
| 기타 | `mtp_num_hidden_layers=1`, layer 구성 linear 24 / full 8, vision `num_position_embeddings=2304`. 단 `mtp_num_hidden_layers`는 `Qwen3_5TextConfig`에 정의된 필드가 아니라 `config.json`에서 넘어온 추가 속성이다. MTP 관련 사실은 O7 참고 | E11, 검증 V15·V11 | VERIFIED |
| 관측 최대와 비교 | SFT/DPO 최대 2,272(`keep` 2,276)는 상한의 0.87%다. GRPO prompt 최대 268에 budget 1,024/2,048/4,096/8,192를 더하면 1,292/2,316/4,364/8,460이다. 예시 입력에서는 `CONTEXT_EXCEEDED`가 발생하지 않는다 | E7 | VERIFIED |
| backend가 실제로 지원하는 최대 길이 | 이 조사 범위 밖이다 (attention kernel, linear attention kernel의 길이 제약은 확인하지 않음) | — | UNKNOWN |

---

## 구현 시사점 (Implementation implications)

**R1. tokenizer 로딩과 fingerprint.**
- `AutoTokenizer.from_pretrained(model_id, revision=<commit sha>)`(transformers==5.18.0)로 로드하면 `Qwen3_5Tokenizer`가 된다. 실제 Trainer와 같은 결과를 내려면 이 경로로 토큰화한다.
- 속도를 위해 `tokenizers.Tokenizer.from_file("tokenizer.json")`을 직접 쓰면 안 된다. pre-tokenizer regex가 달라 결합 문자가 있는 언어에서 결과가 달라진다(§1, E3).
- `tokenizer_fingerprint`에는 파일 해시만이 아니라 tokenizer class 이름과 transformers 버전을 넣는다. 파이프라인 일부가 라이브러리 코드에서 오기 때문이다. 권장 구성:
  `{tokenizer_class, transformers_version, sha256(tokenizer.json), sha256(tokenizer_config.json), sha256(chat_template.jinja)}`
- `template_fingerprint = sha256(tokenizer.chat_template.encode("utf-8"))`로 둔다. 예시 모델에서는 `59a64ebb…`이고 파일 해시와 같다.

**R2. TRL 기본 processing class.** TRL은 `processing_class=None`이면 세 Trainer 모두 `AutoProcessor`를 사용한다. 학습 환경에 Pillow와 torchvision이 있으면 이 모델은 `Qwen3VLProcessor`(VLM 경로)가 된다. 텍스트 전용 데이터에서는 token ID가 같으므로(E8) 분석기는 tokenizer 경로를 써도 된다. 다만 결과에 `processing_path: tokenizer|processor`와 "VLM 경로와 동일성 검증됨(text-only)"을 기록한다(trl==1.14.1 `trainer/sft_trainer.py:1004`, `trainer/dpo_trainer.py:591`, `trainer/grpo_trainer.py:373`).
- (검증 시 추가, VERIFIED V12) 이 모델의 `AutoProcessor`는 **torchvision이 반드시 있어야** 로드된다. Pillow와 torchvision이 모두 없으면 `ValueError: Could not load any image processor class`, Pillow만 있으면 `ImportError: Qwen3VLVideoProcessor requires the Torchvision library`가 난다. TRL은 위 세 위치에서 이 예외를 잡지 않으므로 `processing_class=None`인 Trainer 생성이 **실패한다**. tokenizer로 fallback하지 않는다.
- 따라서 내보내는 실행 설정은 `processing_class=AutoTokenizer.from_pretrained(...)`를 명시하거나(텍스트 전용 데이터, E8/V8로 결과 동일), torchvision 의존성을 명시해야 한다.

**R3. 컬럼 매핑 규칙.**
- objective별 TRL 형식 컬럼을 만들고 **원본 컬럼을 모두 제거**한다.
  - GRPO: `{"prompt"}`
  - SFT: `{"prompt", "completion"}` (prompt-completion 형식이 기본)
  - DPO: `{"prompt", "chosen", "rejected"}`
- 제거해야 하는 이유는 두 가지다.
  - (a) `is_conversational`이 지원 key 집합에서 `set.pop()`으로 아무 key나 하나 고른다(trl==1.14.1 `data_utils.py:185-191`). 문자열 `chosen`/`rejected`가 list 형식 `prompt`/`completion`과 함께 남아 있으면 `PYTHONHASHSEED`에 따라 형식 판정이 바뀐다(E12). 어떤 seed가 False가 되는지는 key 삽입 순서에도 달려 있다. 검증 V13(seed 0–15)에서 key 순서 `prompt, completion, chosen, rejected`는 seed 0, 3, 9, 10, 14가 False였고, 원본 컬럼을 남긴 `map` 결과와 같은 순서(`lang, …, chosen, rejected, prompt, completion`)는 seed 0, 9, 10, 12, 14가 False(seed 3은 True)였다. E12의 "seed 0과 3에서 False, 나머지는 True"는 특정 key 순서와 범위에서만 맞다.
  - (a′) 판정이 False가 되면 SFT `add_eos`가 list에 `.endswith`를 호출해 `AttributeError: 'list' object has no attribute 'endswith'`로 Trainer 생성이 실패한다(trl==1.14.1 `trainer/sft_trainer.py:1519-1536`). 검증 V13에서 원본 컬럼을 남긴 8 row로 seed 0, 9는 실패, seed 1은 성공했다. 같은 코드와 데이터가 실행마다 다르게 동작한다.
  - (b) 원본을 매핑 없이 DPO에 넣으면 `extract_prompt`가 chosen/rejected 코드의 공통 prefix를 prompt로 만든다. 실측에서 question이 prompt에 들어간 row는 0개였고, 추출된 prompt 길이 중앙값은 66자였다(E12, trl==1.14.1 `data_utils.py:557-641`).

**R4. 빈 system 정책.**
- 기본값은 `system_policy="omit_if_empty"`를 권장한다. 값이 `None`이거나 정확히 `""`일 때만 생략하고, 공백만 있는 문자열은 렌더링한다.
- 이 정책을 preprocessing fingerprint와 결과에 넣고 UI 데이터 탭에 "system 컬럼 4,656/4,656 row가 비어 system 메시지를 생략"처럼 표시한다.
- `keep`을 고르면 이 template에서는 시퀀스마다 +4 token이다. 제품 기본값을 확정하는 일은 미확정 사항 O1로 남긴다.

**R5. objective별 길이 정의.** 모두 실제 TRL 1.14.1과 같다.
- GRPO:
  `prompt_len = len(apply_chat_template(prompt, add_generation_prompt=True, tools=None, **chat_template_kwargs))`
  `scenario_context = prompt_len + completion_budget`
  `prompt_len`에는 truncation이 없다(GRPOConfig에 `max_prompt_length` 필드가 없음, E9).
- SFT (prompt-completion):
  `total = len(apply_chat_template(prompt + completion))`
  `completion = total - len(apply_chat_template(prompt, add_generation_prompt=True))`
  `loss_tokens = completion` (`completion_only_loss` 기본값 None → True)
  `messages` 형식에서는 `assistant_only_loss=True`이면 `loss_tokens = completion + 3`, 기본값이면 `total - 1`. activation 계산은 언제나 `total`을 기준으로 한다(plan.md §8.1).
- DPO:
  `prompt`, `chosen_total = prompt + len(chosen_ids)`, `rejected_total = prompt + len(rejected_ids)`
  collator shape = `[2·B, max(모든 chosen_total, rejected_total)]` (+`pad_to_multiple_of` 올림). 단일 pair 기준 최악값은 `pair_max`다.
- prefix 불일치(TRL은 warning만 남김)는 row별로 세어 `processing_status`에 반영한다. 예시에서는 0건이다.

**R6. 내보내는 실행 설정에 숨은 길이 제한을 남기지 않는다.**
- 기본값이 `SFTConfig.max_length=1024`, `DPOConfig.max_length=1024`이다(trl==1.14.1 `trainer/sft_config.py:203`, `trainer/dpo_config.py:194`). 예시에서는 그대로 두면 SFT 17 row, DPO 28 pair가 잘린다(§5.6).
- `max_length=None`, `packing=False`로 명시한다. `GRPOConfig.max_completion_length`(기본 512)와 `num_generations`(기본 8)도 선택한 budget과 preset 값으로 명시한다(`trainer/grpo_config.py:479,493`).
- (검증 시 추가) 정수 `max_length`를 쓰면 잘림 외에 **row 삭제**도 생길 수 있다. SFT는 truncation 뒤 loss token이 0개인 example을, DPO는 `len(prompt_ids) >= max_length`인 example을 지운다(§5.6 주의). 정수 상한이 꼭 필요하면 전체 스캔 최대 길이 이상으로 두고, 처리 후 row 수와 길이가 golden과 같은지 확인한다.

**R7. content 손실과 주입 검사.**
- 다음 세 가지를 row별로 검사한다.
  - (a) 렌더링 결과에 각 content가 원문 그대로 들어 있는지
  - (b) content에 added token 문자열(33개, 특히 special 21개)이 있는지. 있으면 토큰화 때 special token으로 바뀌므로 경고한다
  - (c) NFC 정규화로 원문이 바뀌는지
- template이 추가하는 token은 손실이 아니라 주입으로 보고한다. 이 모델은 assistant turn마다 `<think></think>` 2개다.

**R8. Thinking 일관성 주의.**
- 이 template에서 SFT/DPO 응답은 항상 `<think></think>`(빈 reasoning)로 시작한다.
- GRPO 기본 prompt는 `<|im_start|>assistant\n`에서 끝나므로 모델이 thinking token을 생성할 수 있다. GRPO completion budget에는 thinking token이 포함된다.
- `chat_template_kwargs={"enable_thinking": False}`를 쓰면 GRPO prompt가 +2 token이다. SFT에서 example별로 같은 kwarg를 주면 전체 길이는 그대로이고 loss token만 −2다(E12).

**R9. Context 검사.**
- `context_limit = config.text_config.max_position_embeddings`(262,144)를 쓴다. tokenizer `model_max_length`는 따로 저장한다.
- `model_max_length ≥ 1e20`(transformers의 `LARGE_INTEGER`/`VERY_LARGE_INTEGER` 계열)이면 sentinel로 보고 상한으로 쓰지 않는다. 이 모델의 262,144는 실제 선언값이다.

**R10. vocab 차원.** logits와 `lm_head` 메모리 계산에는 `config.text_config.vocab_size=248,320`을 쓴다. `len(tokenizer)=248,077`을 쓰면 안 된다. padding token ID는 248044(`<|endoftext|>`)다. 검증 V11에서 safetensors header만 읽어 확인했다: `lm_head.weight`와 `model.language_model.embed_tokens.weight`가 모두 BF16 `[248320, 4096]`이고, 전체 tensor 760개, BF16 parameter 9,409,813,744개다(huggingface_hub==1.33.0 `get_safetensors_metadata`, VERIFIED).

**R11. 숫자가 많은 content.** 숫자 1자리가 1 token이므로 문자 수 기반 추정은 outlier를 놓친다. 예시의 최장 row는 숫자열이고 1.03 chars/token이다. 진행률이나 예상 시간 추정에만 문자 수를 쓰고, 길이 결과는 반드시 실제 토큰화로 낸다.

**R12. worker timeout과 진행 이벤트.**
- 측정값: TRL과 같은 row별 경로는 objective당 약 1,000 row/s(M1 Max. 검증 V10에서 부하가 있을 때 DPO 약 830 row/s), GRPO prompt는 1만 row/s 이상, cold start는 약 5–7 s(import + tokenizer + dataset. 파일 캐시 상태에 따라 다름).
- 진행 이벤트는 `max(1 s, 256 row)`마다 보내기를 권장한다.
- timeout은 `30 s + 10 ms × row × objective 수`를 권장한다. 측정값보다 약 10배 여유를 둔 값이다. 이 권장값은 제품 결정이므로 INFERRED로 둔다. 한 번 측정해 보정한다(O3).

**R13. 데이터셋 읽기.**
- 확장자가 `.json`이라고 JSON array로 가정하지 않는다. 형식을 탐지하거나 `datasets`의 `json` builder를 사용한다. 이 파일은 JSONL이다.
- 빠른 경로로 Hub `refs/convert/parquet`(commit `1cc07a69…`)를 써도 이 데이터셋에서는 원본과 같다. 다만 변환본의 원본 revision이 메타데이터에 없으므로 기본 source는 고정한 원본 파일로 하고, parquet는 내용 대조 후 캐시로만 쓴다(O2).

**R14. 회귀 테스트(plan.md §19.2).** `golden-example-stats`를 fixture로 복사해 다음을 검사한다.
- (a) count, max, max_rows, total 정확 일치
- (b) §5.4 불변식
- (c) collator shape `[2,2272]`, `[4,2272]`, `[2,239]`
- (d) "DPO outlier": rejected 때문에만 길어지는 row(예: 3169, 2707)가 DPO 피크에 반영되는지
- (e) 동일 쌍(26)과 중복 row가 삭제되지 않는지
- (f) 기본 `max_length=1024`이면 17/28건이 잘린다는 사실로 "숨은 기본값 제거" 테스트를 만든다

전체 데이터셋 다운로드가 CI에서 부담되면 상위 outlier row만 고정한 작은 fixture를 쓰고 원본 revision과 hash를 함께 기록한다(git-conventions §10).

## 미확정 사항 (Open questions)

- **O1.** 제품 기본 빈 system 정책(`omit_if_empty` 대 `keep`). 이 문서는 `omit`을 1차 golden으로 삼았다. 두 정책의 차이는 이 template에서 정확히 +4 token이다. 오케스트레이터 또는 사양에서 결정해야 한다.
- **O2.** Hub parquet 변환본의 원본 revision은 commit 메타데이터에 없다(timestamp로 추론). 제품이 변환본을 1차 source로 쓸지는 정하지 않았다.
- **O3.** Docker(linux/amd64, linux/arm64) 안에서의 처리량은 측정하지 않았다(UNKNOWN). R12의 timeout 공식은 실제 compose 환경에서 다시 보정해야 한다.
- **O4.** 실제 CUDA 학습 환경에 Pillow와 torchvision이 설치되어 있는지(즉 TRL이 VLM processor 경로를 탈지)는 환경마다 다르다. 텍스트 전용 데이터에서 결과가 같다는 것만 확인했다. 이미지가 포함된 데이터셋은 범위 밖이다. (검증 시 보완) torchvision이 없으면 `processing_class=None`인 TRL Trainer는 생성 단계에서 실패한다(R2). 따라서 남은 질문은 "어느 경로를 타는가"보다 "내보내는 설정이 `processing_class`를 명시할 것인가"다.
- **O5.** transformers의 `Qwen3_5Tokenizer.PRETOKENIZE_REGEX`가 이후 버전에서 바뀌면 같은 `tokenizer.json`으로도 token ID가 달라질 수 있다. 버전 고정과 fingerprint(R1)로 대응하지만 다른 transformers 버전과의 호환 범위는 확인하지 않았다.
- **O6.** GRPO의 실제 생성 길이(thinking 포함)는 데이터로 정할 수 없다. budget 시나리오만 가능하다(plan.md §8.3). 이 모델은 `generation_config`가 `do_sample=true, temperature=0.6, top_p=0.95, top_k=20`이며, 생성 종료 token이 248046과 248044 두 개다(rollout 쪽 조사에서 반영 필요).
- **O7.** `mtp_num_hidden_layers=1`(multi-token prediction 층)을 Trainer가 로드하거나 학습하는지는 이 조사 범위 밖이다(모델 inventory/메모리 조사 담당). (검증 시 일부 해소, VERIFIED) checkpoint header에는 이름에 `mtp`가 들어간 tensor가 0개다(760개 중. 구성은 `lm_head`, `embed_tokens`, text layer 0–31, `norm`, `visual.*`, V11). transformers==5.18.0 `models/qwen3_5/modeling_qwen3_5.py`에는 MTP 모듈이 없고, `Qwen3_5PreTrainedModel`(916행)이 `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`(924행)를, `Qwen3_5ForCausalLM`(1673행)이 `[r"^mtp.*", r"^model.visual.*"]`(1679행)를 둔다. 따라서 transformers 경로에서는 MTP 층이 만들어지지도, 로드되지도, 학습되지도 않는다. 다른 backend(예: vLLM rollout)의 MTP 처리는 여전히 미확인이다.
- **O8.** 중복 row와 chosen==rejected 쌍을 UI에서 어떤 수준의 경고로 보여줄지 정하지 않았다(삭제는 무절단 계약상 금지).

## 검증 로그 (Verification log)

| 항목 | 값 |
|---|---|
| 검증자 | `verify-example-model-dataset` (Milestone M0), 2026-10-04 |
| 방법 | 원 저자의 스크립트를 재사용하지 않고 새로 작성해 실행했다. golden 길이는 unbound method stub 대신 tiny random-init Qwen3(vocab 248,320)로 **실제 `SFTTrainer`/`DPOTrainer`/`GRPOTrainer` 객체**를 만들어 전체 4,656 row로 다시 계산했다. 인용한 줄 번호는 설치된 소스에서 다시 읽었다 |
| 실행 | `HF_HOME=/tmp/vf-research/hf /tmp/vf-research/.venv/bin/python <script>`. processor 경로(V8, V12)는 private overlay venv `/tmp/vf-research/venv-verify-example-model-dataset`(공유 venv + Pillow 12.3.0 + torchvision 0.29.1, `--no-deps`)와 `/tmp/vf-research/venv-verify-example-model-dataset-pilonly`(Pillow만)를 썼다 |
| 스크립트 | `/tmp/vf-research/scratch/verify-example-model-dataset/` (저장소 밖) |
| 결론 | golden 수치(통계, top-10, 초과 row 수, CSV, digest)는 **모두 정확**했다. 수정은 근거 표현, 일반화 범위, 빠진 주의사항에 한정된다 |

| ID | 스크립트 | 내용 |
|---|---|---|
| V1 | `v1_tokenizer.py` | tokenizer class, 길이, special token, backend 구성, raw `tokenizer.json` 비교, `tokenizer_config.json` 키 |
| V2 | `v2_prefix_rawtok.py` | TRL 로그와 별개로 prompt prefix를 직접 검사(`omit`/`keep`), raw `tokenizer.json`과 content 비교 |
| V3 | (스크립트 없음, `sed`/`grep`) | 문서가 인용한 줄 번호를 설치된 소스에서 다시 확인 |
| V4 | `v4_hashes.py` | 파일 SHA256, git blob sha1, LFS sha256, 데이터셋 tree/commit/refs |
| V5 | `v5_template.py` | 렌더링 경우, assistant mask, TRL template 검사 4종, TRL 내장 Qwen template 비교 |
| V6 | `v6_dataset.py`, `v6_parquet.py` | 데이터셋 사실, 중복, 문자 통계, 불변식, CSV `n_*` 대조, parquet 대조 |
| V7 | `v7_golden.py`, `v7_compare.py` | 실제 Trainer 객체로 golden 재계산, JSON 블록·CSV·digest 대조, collator shape |
| V8 | `v8_vlm.py` | `Qwen3VLProcessor`로 실제 Trainer 3종을 만들어 tokenizer 경로와 전체 비교 |
| V9 | `v9_variants.py` | SFT loss 변형 4종, 기본 `SFTConfig`/`DPOConfig`의 잘림 수(전체 row) |
| V10 | `v10_timing.py` | 처리량 재측정 |
| V11 | `v11_header.py` | safetensors header만 읽기(가중치 미다운로드) |
| V12 | `v12_processor_deps.py` | `AutoProcessor` 의존성(공유 venv, Pillow만, Pillow+torchvision) |
| V13 | `v13_isconv.py`, `v13_isconv2.py`, `v13_sft_leftover.py`, `v13_extract_prompt.py` | hash seed 의존성, 원본 컬럼이 남았을 때의 영향, `extract_prompt` |
| V14 | `v14_misc.py` | `add_bos/eos_token`, `skip_special_tokens` decode, `model_max_length` warning, decode 왕복, U+FE0F, streaming |
| V15 | `v15_config.py` | context 관련 config, class 기본값, `generation_config.json` |

| # | 주장 (위치) | 판정 | 근거 (한 줄) |
|---|---|---|---|
| 1 | `AutoTokenizer` → `Qwen3_5Tokenizer`, `len(tok)` 248,077, `vocab_size` 248,044, config vocab 248,320, BOS None, EOS 248046, PAD 248044, config text EOS 248044, `model_max_length` 262,144 (§1) | VERIFIED | V1 출력이 모두 일치. 인용 중 집합 리터럴 범위만 `:378-427`→`:378-429`로 고침(V3) |
| 2 | normalizer/pre-tokenizer를 class 코드가 다시 만들고 raw `tokenizer.json`과 regex가 다름(힌디어 6 대 10, 태국어 3 대 7), 데이터셋 13,968 시퀀스 동일 (§1) | CORRECTED | V1·V2로 재현. 다만 added token도 `tokenizer.json`에서 읽으므로(`tokenization_utils_base.py:1904-1914`) "vocab, merges, post_processor만"을 고쳤다 |
| 3 | `tok(text)`는 special token을 붙이지 않음, `apply_chat_template`은 `add_special_tokens=False`(3124-3132), content 안 `<\|im_end\|>`는 248046 (§1) | VERIFIED | V1·V14: `[14556, 1814]`, `add_bos/eos_token=False`, `'a <\|im_end\|> b'`→`[64, 220, 248046, 292]` |
| 4 | 파일 SHA256과 Hub blob/LFS 대조 (§1.1) | VERIFIED | V4: 7개 파일 SHA256 일치, git blob 6개 일치, `tokenizer.json` LFS sha256 일치, `tok.chat_template` SHA256 = `59a64ebb…` |
| 5 | template 규칙: `<\|im_end\|>` 뒤 줄바꿈 없음, 빈 system 4 token, assistant turn 형식, generation prompt 3/5 token, 이전 reasoning 유지 (§2) | VERIFIED | V5 렌더링이 모두 일치. 곁가지인 TRL 내장 template의 `<\|im_end\|>\n` 위치 수만 "4–5곳"→"5–6곳"으로 고침 |
| 6 | `{% generation %}`이 header를 포함한 turn 전체를 감쌈, TRL 검사 4종 True, `get_training_chat_template`→`None` (§2) | VERIFIED | V5: mask가 `<\|im_start\|>`부터 `<\|im_end\|>`까지 1. `chat_template_utils.py:37-42,825,886-952,1105-1107` 재확인 |
| 7 | 데이터셋이 JSONL, 6,867,898 bytes, SHA256 `ad93a85f…`, 4,656 row, null 0, system 전부 `""`, 완전 중복 2쌍, chosen==rejected 26, 중복 question 초과 500 (§4) | VERIFIED | V4·V6: 모두 일치. lang 분포, 중복 chosen/rejected 그룹, 문자 통계(p99는 반올림 값), streaming 동일성(V14)도 일치 |
| 8 | Parquet 변환본 `1cc07a69…`가 JSONL과 row 단위로 같음 (§4) | VERIFIED | V6: 2,564,627 bytes, SHA256 `eb9b1f8b…`, row group 5개, 4,656/4,656 동일 |
| 9 | 변환본의 원본 revision이 `81aeacf…` (§4) | UNVERIFIABLE | convert commit 7개의 메시지가 "Update parquet files"/"initial commit"뿐이고 원본 sha가 없다. 12초 차 timestamp로만 추론되므로 INFERRED 태그를 유지 |
| 10 | golden 통계 전부 (§5.1–5.3, JSON 블록) | VERIFIED | V7: 실제 Trainer 객체로 다시 계산. JSON 블록 모든 필드 차이 0, top-10 표 6개 일치, CSV 길이 컬럼 11개 4,656 row 차이 0 |
| 11 | `keep = omit + 4`, nothink GRPO +2, 불변식(`n_q+7`, `n_c+3`, `n_q+n_c+10`), mismatch warning 0, 실패 0 (§5.2, §5.4, §7) | VERIFIED | V6·V7: offset 집합 {4}, {2}, completion {0}. 불변식 4종 4,656/4,656. prefix 위반도 직접 검사해 0(V2) |
| 12 | VLM processor 경로가 tokenizer 경로와 같고 1.4–1.8배 느림 (§5.5, §6) | CORRECTED | V8: 모든 ID/label이 4,656/4,656 같음(두 정책). 속도 배율은 1.20–1.43배로 측정돼 "약 1.2–1.8배"로 고침 |
| 13 | 기본 `max_length=1024`(`sft_config.py:203`, `dpo_config.py:194`), SFT 17 row / DPO 28 pair 잘림, GRPO에 `max_prompt_length` 없음, `max_completion_length` 512, `num_generations` 8, collator shape (§5.5–5.6, R6) | CORRECTED | 값은 config 인스턴스화와 V9(전체 row, SFT 11,127 token 손실, 삭제 0)로 확인. 빠져 있던 길이 기준 row 삭제 filter(`sft_trainer.py:1676-1683`, `dpo_trainer.py:1100-1106`)를 추가 |
| 14 | loss token 변형: prompt-completion=completion, `messages`+`assistant_only_loss`=completion+3, `messages` 기본=total−1, example별 `enable_thinking=False`는 completion −2 (§5.5, R5, R8) | VERIFIED | V9: 네 경우 모두 4,656/4,656 (원문은 48 row). 187행이 기본 `loss_type="chunked_nll"` 경로라는 점을 근거에 덧붙임 |
| 15 | `is_conversational`의 `set.pop()` 때문에 hash seed에 의존하고, seed 0과 3에서 False (R3) | CORRECTED | V13: 의존성은 사실. False가 되는 seed는 key 순서마다 다르다(실제 `map` 순서에서는 0, 9, 10, 12, 14). False이면 SFT가 `AttributeError`로 실패한다는 점을 추가 |
| 16 | 원본을 그대로 DPO에 넣으면 `extract_prompt`가 공통 prefix를 prompt로 씀: question 포함 0 row, 중앙값 66자 (R3) | VERIFIED | V13: 0 row, 중앙값 66자 (`data_utils.py:632-641`) |
| 17 | 처리량과 cold start (§6, R12) | VERIFIED | V10: 표와 ±20% 안에서 일치(부하 있는 상태). cold start는 "약 7 s"에서 캐시 상태를 반영한 "약 5–7 s"로 범위를 넓힘 |
| 18 | context config: `max_position_embeddings` 262,144, rope `default`, theta 1e7, partial 0.25, mrope `[11,11,10]`, sliding window 없음 (§8) | CORRECTED | V15: 값은 일치. 다만 `sliding_window` 필드는 아예 없다(`hasattr` False). rope 값은 `config.json`에서 오며 class 기본 theta는 1e4라는 점을 명시 |
| 19 | per-row CSV SHA256 `d9a2e7b7…`, token ID digest 3종, JSON 블록 = fixture (§5.7–5.8) | VERIFIED | V7: CSV SHA256·283,440 bytes·4,657줄 일치. 내 Trainer 결과로 계산한 digest 3종이 일치하고, 블록이 fixture 파일 2개와 같다 |
| 20 | R10: logits/`lm_head`는 vocab 248,320 기준 | VERIFIED | V11: header 기준 `lm_head`와 `embed_tokens`가 모두 BF16 `[248320, 4096]` |
| 21 | R2: `processing_class=None`이면 `AutoProcessor`를 쓰고, Pillow+torchvision이 있으면 `Qwen3VLProcessor` (`sft_trainer.py:1004`, `dpo_trainer.py:591`, `grpo_trainer.py:373`) | CORRECTED | V12: 의존성이 없으면 `ValueError`/`ImportError`로 Trainer 생성이 실패하고 tokenizer로 fallback하지 않는다는 점을 R2·O4에 추가 |
| — | O7 (MTP) | 일부 해소 | V11: checkpoint의 `mtp` tensor 0개. transformers는 `^mtp.*` key를 무시한다(`modeling_qwen3_5.py:924,1679`) |

확인하지 않은 항목: Docker 처리량(O3, 원문도 UNKNOWN), E7 스크립트의 최대 RSS 1.13 GB(구현 영향 없음), E9의 48 row 표본 구성(전체 row 재현으로 대체).
