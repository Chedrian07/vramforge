# Fine-Tuning VRAM Calculator — 최종 구현 계획

> **제품 정의:** 모델·데이터셋·파인튜닝 방식을 입력하면, 실제 토크나이저와 학습 전처리로 전체 데이터 길이를 분석하고 **데이터를 자르지 않는 조건의 GPU별 피크 VRAM**을 산정하는 웹 계산기.
>
> **UI 방향:** ApX VRAM Calculator를 참고한 단일 화면 계산기. 기본 입력은 간단하게, 결과는 크게, 상세 설정과 계산 근거는 펼쳐서 확인한다.

| 문서 항목 | 값 |
|---|---|
| 파일 | `plan.md` |
| 작성일 | 2026-10-03 |
| 문서 버전 | 1.0 |
| 상태 | 최종 구현 명세. 애플리케이션 구현·전체 데이터 스캔·GPU 실측 결과가 아님 |
| 기본 배포 | Self-hosted 웹앱. 공개 서비스로 확장 가능한 구조 |
| 1차 계산 대상 | NVIDIA/CUDA 학습 환경. 분석 서버 자체에는 GPU 불필요 |
| 필수 학습 방식 | SFT, DPO, GRPO |
| 핵심 보장 범위 | 선택한 유한 데이터 snapshot의 전체 길이 분석과 데이터 보존 검사 |
| 산정 결과의 성격 | 정적 추정 / 보정된 추정 / 수행 구간 실측을 명확히 구분 |

---

## 0. 확정할 제품 결정

1. **첫 화면은 계산기다.** 프로젝트 관리, 학습 모니터링, 모델 배포 대시보드로 확장하지 않는다.
2. **기본 입력은 Model, Dataset, Method, 4-bit 여부다.** 나머지는 공개된 자동 프리셋으로 채운다.
3. **데이터셋 전체 분석이 기본이다.** 샘플 분석은 선택 기능이며 전체 최대 길이를 확인한 것처럼 표시하지 않는다.
4. **SFT/DPO의 학습 길이는 데이터에서 도출한다.** 모델의 최대 context나 임의의 2K/4K 값을 기본 학습 길이로 쓰지 않는다.
5. **GRPO는 prompt 분석과 completion budget을 분리한다.** 생성 길이를 지정하지 않으면 복수 시나리오를 제공한다.
6. **SFT/DPO/GRPO와 Full/LoRA/QLoRA를 다른 축으로 관리한다.** 4-bit 체크는 지원되는 QLoRA 프리셋을 선택한다.
7. **모델 구조와 실제 Trainer 구현에 맞춰 계산한다.** 모든 모델을 `파라미터 수 × 상수`로 처리하지 않는다.
8. **하드웨어를 선택하지 않아도 필요한 용량을 계산한다.** 다만 GPU 적합 판정은 하드웨어·백엔드 조건이 있어야 제공한다.
9. **단위는 내부 byte, 기본 화면 GiB다.** GB 변환은 보조 표기로만 제공한다.
10. **기본 분석은 CPU-only다.** 전체 가중치 다운로드, 모델 실행, 학습, 유료 GPU 사용을 자동으로 시작하지 않는다.
11. **GPU 검증은 별도 opt-in 작업이다.** 수행한 구간의 관측치이지 전체 학습의 OOM 부재 보증이 아니다.
12. **추정할 수 없는 항목은 `null + 사유`다.** 0으로 채우거나 임의의 절감률로 정상 결과를 만들지 않는다.

이 문서의 UI·기술 스택·프리셋·여유율은 제품 설계 결정이다. 특정 구성의 학습 품질, 가장 높은 보급률, 실측 정확도가 이미 검증되었다는 주장이 아니다.

## 1. 참고 서비스에서 가져올 것과 바꿀 것

### 1.1 확인한 참고 범위

참고 페이지에는 모델 선택, 양자화, 하드웨어와 GPU 수, batch/sequence 입력, 접이식 고급 설정, offload 옵션, VRAM 결과 영역이 있다. Inference와 Fine-tuning 선택도 제공한다. 이 문서는 해당 **계산기형 정보 구조**를 참고한다. 세부 동작 전체나 내부 수식, 모든 탭을 실측 검증한 것은 아니다. [R01]

| 참고 요소 | 본 제품에서의 적용 |
|---|---|
| 간단한 모델 선택 | HF ID/URL 또는 로컬 모델 경로 붙여넣기 |
| 모드 전환 | SFT / DPO / GRPO segmented control |
| 양자화 설정 | 기본은 `Load in 4-bit`; 세부 dtype은 Advanced |
| 하드웨어 선택 | `용량만 계산` 또는 실제 GPU/사용 가능 VRAM 지정 |
| Sequence length 입력 | 기본 화면에서는 전체 데이터 분석 결과를 읽기 전용으로 표시 |
| 고급 설정 펼치기 | LoRA, Batch, Runtime, 방법별 옵션을 accordion으로 구성 |
| 큰 VRAM 결과 | 예상 피크·계획용 권장 용량·적합 여부를 고정 요약 카드로 표시 |
| 조건을 바꾸며 비교 | 토큰화 결과를 재사용하는 시나리오 재계산 |

### 1.2 의도적으로 제외할 것

추론 TPS, TTFT, GPU 임대 가격, 전력 비용, 채팅 도우미, 자동 모델 추천은 첫 버전에 넣지 않는다. 본 제품은 **파인튜닝에 필요한 메모리 산정**에 집중한다.

ApX의 로고, 문구, 소스코드, 내부 계산 계수, 디자인 자산을 복제하지 않는다. 아래 화면 치수와 색상은 본 제품의 독립적인 제안이며, 참고 페이지를 픽셀 단위로 재현한 사양이 아니다.

## 2. 출시 범위와 지원 수준

### 2.1 첫 공개 버전의 필수 범위

| 영역 | 필수 기능 |
|---|---|
| 입력 | HF 모델/데이터셋 ID·URL, self-hosted 로컬 경로, 데이터 파일 업로드 |
| 모델 분석 | config, tokenizer/template, safetensors tensor inventory, 로딩 범위 분석 |
| 데이터 | Text, prompt-completion, messages, preference pair 형식 |
| 분석 | 전체 유한 snapshot 스캔, 무절단 검사, branch별 길이, 최악 batch 산정 |
| 학습 | SFT / DPO / GRPO 조건부 메모리 모델 |
| 전략 | Full / LoRA / QLoRA 중 검증된 조합만 활성화 |
| 구조 | 일반 Dense decoder와 Qwen3.5 hybrid 텍스트 학습 경로 |
| 장치 | 단일 NVIDIA GPU의 정적 추정. 하드웨어 미지정 시 가정 표시 |
| 결과 | GPU 피크, 학습 RAM, 데이터 통계, 근거, 경고, 시나리오 비교 |
| 내보내기 | 분석 JSON, 계획용 YAML, 보고서 Markdown |
| 운영 | 작업 진행 상태, 취소, 오류 복구, 캐시, 접근 제어 |

Qwen3.5의 **구조 분석 지원**과 특정 GPU에서의 **학습 검증 완료**는 같은 의미가 아니다. 로더·커널·학습 방식별 지원 상태를 따로 표시한다.

### 2.2 확장 범위

GPU profiling, DDP/FSDP/ZeRO/TP/PP, 별도 vLLM 서버, MoE 상세 routing, 실제 이미지·영상 입력, MLX/MPS, 로컬 companion은 후속 기능으로 분리한다. 기본 데이터 모델에는 확장 필드를 준비하되 미구현 옵션을 작동하는 것처럼 노출하지 않는다.

다중 GPU 입력을 받았는데 해당 topology adapter가 없으면 `지원하지 않는 분산 구성`을 반환한다. 총 용량을 GPU 수로 나눈 값을 대신 제공하지 않는다.

### 2.3 완료의 정의

예시 모델과 데이터셋을 입력하면 모델·토크나이저·컬럼을 실제로 확인하고 전체 데이터 길이를 분석해야 한다. 메모리 모델이 지원되는 실행 조합이면 근거 있는 피크 시나리오를 제공한다. 지원되지 않는 조합이면 확인한 정보와 미지원 원인을 반환해야 한다.

**UI에 고정 숫자를 표시하는 데모만으로는 완료가 아니다.** 반대로 GPU 없이 실행한 결과에 `실측 완료`가 없어도 정상이다.

## 3. 화면 명세

### 3.1 데스크톱 레이아웃

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ Fine-Tuning VRAM Calculator                     [테마] [도움말]            │
│ 데이터셋을 자르지 않고 학습할 때 필요한 GPU 메모리                          │
├───────────────────────────────────────────┬──────────────────────────────┤
│ Model                                     │ 예상 메모리                  │
│ [HF ID / URL / 로컬 경로                ] │                              │
│ 구조·tokenizer 상태                       │     — GiB / GPU              │
│                                           │     전체 데이터 분석 필요    │
│ Dataset                                   │                              │
│ [HF ID / URL / 파일 선택                ] │ 피크 추정     —              │
│ [config] [train split] [컬럼 매핑 확인]    │ 권장 용량     —              │
│                                           │ 학습 RAM      —              │
│ Method [ SFT | DPO | GRPO ]                │                              │
│ [✓] Load in 4-bit    전략: QLoRA          │ [데이터 보존 상태]           │
│                                           │ [추정 / 보정 / 실측]         │
│ [✓] 전체 데이터 분석 · Truncation 없음    │ [학습 설정 유효성]           │
│                                           │                              │
│ Hardware [용량만 계산 ▼]                  │ 선택 GPU의 사용량 막대       │
│ ▶ Advanced                               │ 또는 용량 선택 안내          │
│                                           │                              │
│ [전체 데이터 분석 및 계산]               │ [결과 내보내기]              │
├───────────────────────────────────────────┴──────────────────────────────┤
│ 진행 상태: 구조 확인 → 데이터 토큰화 → 배치 분석 → 메모리 산정             │
├──────────────────────────────────────────────────────────────────────────┤
│ [메모리 구성] [데이터 길이] [단계별 피크] [비교] [적용 설정·근거]           │
│ 선택한 탭의 상세 내용                                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

우측 요약은 스크롤 중에도 보이도록 sticky로 두되, 카드가 화면보다 길면 내부 스크롤을 강제하기보다 sticky를 해제한다. 오류와 미확정 가정은 숨겨진 상세 탭에만 두지 않고 요약에도 표시한다.

### 3.2 디자인 시스템

| 항목 | 사양 |
|---|---|
| 기본 테마 | 밝은 배경 + 흰 카드. 다크 모드 제공 |
| 콘텐츠 폭 | 최대 1,320px, 데스크톱 좌우 여백 24px 이상 |
| 본문 구조 | 1,024px 이상 2열, 미만 1열 |
| 열 비율 | 입력 약 58%, 결과 약 42% |
| 타이포그래피 | 한글 system sans-serif, 숫자는 tabular-nums, 모델 ID는 monospace |
| 크기 | 본문 14–16px, section 18–20px, 핵심 수치 40–48px |
| 카드 | radius 12px, 1px border, 강한 그림자·장식 최소화 |
| 강조색 | 제안값 `#4F46E5`; 정상·주의·오류는 색상과 텍스트를 함께 사용 |
| 간격 | 4/8/12/16/24/32px 토큰 |
| 숫자 | 화면 1자리 소수, tooltip 원본 bytes; 계산 중 조기 반올림 금지 |
| 접근성 | 키보드 조작, 연결된 label, visible focus, reduced motion, 차트 대체 표 |

모바일에서는 입력 → 진행 상태 → 핵심 결과 → 상세 순서로 배치한다. 긴 모델 ID와 tensor 이름은 줄바꿈/복사 버튼으로 처리하고 페이지 전체 가로 스크롤을 만들지 않는다.

### 3.3 입력 영역 동작

Model/Dataset 입력 후 blur 또는 명시적 확인 버튼으로 메타데이터를 조회한다. 타이핑마다 전체 데이터 스캔을 시작하지 않는다. URL 검증은 로컬에서 즉시 수행하고 외부 조회는 취소·중복 방지가 가능한 요청으로 처리한다.

Advanced는 다음 다섯 그룹으로 나눈다.

- **Adapter:** rank, alpha, dropout, target modules, 추가 trainable modules.
- **Batch & precision:** microbatch, accumulation, dtype, optimizer.
- **Runtime:** checkpointing, attention, loss, packing, offload.
- **Method-specific:** DPO reference 또는 GRPO rollout/reward.
- **Dataset & reproducibility:** split, mapping, template, revision, 평가 범위.

### 3.4 결과 카드 표시 규칙

분석 전에는 `0 GB`, `100% 적합` 대신 `— / 분석 필요`를 표시한다. 데이터 전체 스캔 중 수치를 보여주면 `현재까지 확인한 데이터 기준`을 붙인다. 중간 결과를 최종 권장 용량처럼 강조하지 않는다.

상단에는 **예상 피크 범위**, **계획용 권장 용량**, **GPU 적합 상태**를 구분한다. 하드웨어를 입력하지 않았으면 사용률 게이지를 숨긴다. 100% 초과는 그래프가 화면을 넘지 않도록 하되 `예상 126%`처럼 초과 사실을 텍스트로 보존한다.

공개 공유 링크는 첫 버전 기본 기능이 아니다. 결과 내보내기는 사용자가 요청할 때 생성하고, 경로·데이터 원문·토큰을 기본 제외한다.

## 4. 사용자 흐름과 재계산

### 4.1 기본 흐름

```text
입력 → 출처·접근 확인 → 스키마 및 프리셋 해석 → 전체 데이터 토큰화
     → 무절단·context 검증 → batch 계획 → 메모리 산정 → 결과 표시
```

컬럼 매핑이 명백하면 자동 제안 후 분석을 진행하고 적용한 mapping을 노출한다. 여러 split/config 또는 복수 해석이 실제 결과를 바꾸면 해당 항목만 인라인으로 선택하게 한다. 임의 추측으로 분석하지 않는다.

GRPO에서 reward가 없더라도 **조건부 메모리 계산**은 가능하다. 실행 가능한 학습 설정으로 내보내는 것은 별도 검증을 통과해야 한다.

### 4.2 변경에 따른 비용 분리

| 변경 | 필요한 재작업 |
|---|---|
| r, alpha, optimizer, GPU 용량 | 모델 상태/메모리 재계산. 토큰화 재사용 |
| microbatch, accumulation | batch 계획 + 메모리 재계산 |
| GRPO completion budget | 생성·업데이트 shape + 메모리 재계산 |
| attention/loss kernel | 호환성 및 activation 모델 재선택; mask 정보가 부족하면 재분석 |
| SFT ↔ DPO ↔ GRPO | 해당 방식 전처리 재실행. 원본 데이터 다운로드 캐시는 재사용 |
| tokenizer/template/mapping/thinking | 토큰화 캐시 무효화 |
| 데이터 revision/config/split | source manifest와 데이터 분석 무효화 |
| 모델 revision | inventory 재검사; tokenizer와 loading scope 변경 여부에 따라 캐시 결정 |

가벼운 설정 변경은 약 300ms debounce 후 서버 계산을 요청한다. 이전 설정의 응답이 늦게 도착하면 request fingerprint로 폐기한다. 재계산 중 기존 값에는 `이전 설정` 표시를 붙여 stale 결과를 현재 값처럼 보이지 않게 한다.

전체 스캔을 다시 실행해야 하는 변경은 버튼에 `데이터 재분석 필요`를 표시한다. 사용자의 작은 조작이 대규모 다운로드를 반복하게 만들지 않는다.

## 5. 입력 데이터 계약

### 5.1 기본 입력

| 필드 | 기본값/규칙 |
|---|---|
| `model.reference` | HF ID/URL 또는 self-hosted에 등록된 모델 경로 |
| `dataset.reference` | HF ID/URL, 업로드 ID 또는 등록된 데이터 경로 |
| `objective` | SFT / DPO / GRPO. 예시 불러오기에는 사용자가 제시한 GRPO를 사용 |
| `load_in_4bit` | 체크 시 QLoRA 제안, 미체크 시 LoRA 제안. Full은 별도 선택 |
| `scan_mode` | `full` |
| `data_policy` | `strict_no_truncation` |
| `backend_profile` | 릴리스에 포함된 검증 profile 또는 가정이 명시된 정적 profile |
| `hardware.mode` | `capacity_only` |
| `evaluation` | 기본 꺼짐. 결과에 평가·저장 피크 포함 범위를 표시 |

`load_in_4bit`와 strategy는 UI에서 연동하고 API에서는 모순을 오류로 처리한다. 일반 bitsandbytes의 4/8-bit 학습 경로는 추가 파라미터 학습에 해당하므로 `Full + 4-bit`를 일반 지원 조합으로 만들지 않는다. [R05]

### 5.2 공통 Advanced

| 그룹 | 필드 | 초기 제안/처리 |
|---|---|---|
| LoRA | `r`, `alpha`, `dropout` | 16, 32, 0.0. 제품 초기값이며 품질 최적값 아님 |
| 대상 | `target_modules`, `exclude_modules` | 구조별 검증 preset. 해석한 모듈 목록을 표시 |
| 추가 학습 | `modules_to_save`, `bias`, `rank_pattern` | 기본 없음. 추가 파라미터 수 별도 집계 |
| Quantization | format, double quant | 지원 경로에서 NF4 + double quantization |
| dtype | storage/compute/adapter/gradient/state | 각각 별도 값으로 resolve |
| Batch | microbatch / accumulation | SFT·DPO 시작값 1 / 8. pair/sample 단위를 표시 |
| GRPO batch | update microbatch / accumulation | 예시 1 / 4. generation batch 제약과 함께 검사 |
| Checkpointing | enabled / 방식 / 구간 | 지원 경로에서 enabled |
| Attention | auto / explicit backend | 선택 구조·GPU·dtype의 실제 경로 기준 |
| Loss | auto / supported kernel | 호환 profile에서만 메모리 절감 적용 |
| Optimizer | AdamW / 지원 8-bit AdamW | 정적 기준은 명시적 AdamW profile. 검증된 8-bit 경로는 별도 선택 |
| Packing | off / verified preserving | 기본 off |
| Offload | parameter / optimizer / activation | 기본 off, 지원 여부와 host RAM 증가 표시 |
| Compile | enabled | 기본 off. 별도 메모리 profile 필요 |
| 분산 | topology/device layout | 미지원 조합 선택 시 수치 대신 사유 반환 |

LoRA의 일반 Linear 파라미터 수는 rank와 모듈 차원에 의해 정해진다. alpha는 scaling 설정이며 alpha만 바뀐다고 저장 파라미터 수를 변경하지 않는다. `modules_to_save`와 layer별 rank는 별도 집계한다. [R06]

### 5.3 DPO Advanced

`reference_strategy`, reference model/adapter identity, `beta`, loss type, precompute 여부와 batch, reference sync, dropout 실효값, collator/pair 처리 방식을 기록한다.

Reference 선택지는 `frozen_base_switch`, `standalone_model`, `precomputed_log_probs`로 정규화한다. 기존 adapter가 붙어 있거나 일부 base 모듈이 학습 가능하다면 단순 adapter off가 올바른 reference인지 검사해야 한다. Precompute와 reference sync 등 양립하지 않는 조합은 Trainer adapter에서 차단한다. [R09]

### 5.4 GRPO Advanced

| 필드 | 처리 |
|---|---|
| `num_generations` | 예시 기본 4. batch 제약 검증 |
| `completion_budget` | 미지정이면 1,024 / 2,048 / 4,096 / 8,192 시나리오 |
| `generation_batch_size` | 예시 4. 해당 profile에서의 단위를 명시 |
| `max_live_sequences` | 실행 backend가 실제 지원하는 제한만 선택 가능 |
| `steps_per_generation` | generation batch와의 배타 조건을 버전별 검증 |
| `num_iterations` | rollout 재사용 횟수와 버퍼 수명 반영 |
| `beta` | 명시적 preset 값. 메모리 절감을 위해 자동으로 0으로 변경하지 않음 |
| `reward.kind` | unspecified / cpu_rule / remote / local_model |
| `reward.resource_budget` | 로컬 reward 모델·CPU 예산·원격 실행 구분 |
| `rollout.backend` | 1차는 지원되는 Transformers 공유-policy 경로 |
| rollout dtype/quantization | 학습 policy와 별도 기록 |
| reference mode | beta와 실제 Trainer 동작으로 결정 |

TRL GRPO는 prompt별 generation 수와 effective/generation batch의 제약을 갖는다. `num_generations`를 update microbatch와 동일한 값으로 취급하지 않는다. 정확한 제약과 기본값은 선택한 dependency profile에서 검증한다. [R10]

## 6. 모델 자동 분석

### 6.1 모델 주소와 로컬 경로

HF ID, 모델 메인 URL, `tree/{revision}` 형태를 정규화한다. HF dataset viewer URL은 Dataset resolver에서 처리하며 URL의 `row=0`을 전체 데이터 범위 제한으로 오해하지 않는다.

로컬 경로는 **API/worker가 실행되는 서버 기준**이다. 공개 웹서비스에서 사용자 PC의 경로 문자열만으로 파일에 접근할 수 없으며 브라우저 파일 접근에는 사용자 선택·권한이 필요하다. 공개 배포는 업로드 또는 후속 local companion으로 해결한다. [R19]

모델·데이터 폴더는 self-hosted 배포에서 read-only mount한다. UI에 허용된 root와 실제 서버 경로를 표시하고 Windows 경로와 WSL/container 경로를 혼용하지 않는다.

### 6.2 Model Inspector의 순서

```text
입력 정규화 및 권한 확인
  → model revision을 immutable commit으로 resolve
  → config / tokenizer files / template / processor metadata 조회
  → safetensors index와 각 shard의 header 조회
  → tensor shape·dtype·소속 모듈 inventory 작성
  → 등록된 architecture adapter로 실제 로딩·양자화·학습 범위 해석
  → tokenizer 및 모듈 수준 결과를 fingerprint와 함께 저장
```

HF Hub의 safetensors 메타데이터 API는 shard별 shape/dtype 정보를 확인하는 기반으로 사용한다. 전체 weight를 읽는 대신 제한된 header 요청을 우선한다. `index.metadata.total_size`만으로 모듈별 파라미터 수나 실제 VRAM을 계산하지 않는다. [R12]

등록된 클래스의 meta-device 구성을 보완 수단으로 사용한다. 메모리 절약형 meta 초기화는 실제 가중치 실행 없이 구조를 확인할 수 있지만 실제 GPU activation peak 측정은 아니다. [R13]

### 6.3 inventory에서 반드시 구별할 항목

| 항목 | 구별 이유 |
|---|---|
| logical parameter / serialized tensor / unique resident storage | tied tensor와 중복 로딩의 과소·과대 산정 방지 |
| quantized / non-quantized modules | embedding, norm, head 등이 전부 4-bit라는 가정 방지 |
| frozen / trainable parameters | gradient·optimizer 대상 분리 |
| text / vision / auxiliary modules | 데이터 modality와 실제 가중치 상주량은 별개 |
| config dtype / effective loading dtype | 학습 준비 과정의 실제 dtype 반영 |
| full / sliding / linear attention layers | attention·KV·recurrent state 경로 분리 |
| explicit head_dim / inferred head_dim | hidden_size/head 수만으로 고정 계산하지 않음 |
| adapter / full modules_to_save | rank 공식만으로 설명되지 않는 상태 집계 |

양자화된 배포 파일(GGUF, AWQ 등)을 발견했다고 bitsandbytes QLoRA 입력으로 자동 인정하지 않는다. 포맷과 Trainer 지원을 별도 검증한다. 지원되지 않으면 사용 가능한 메타데이터까지만 반환한다.

### 6.4 예시 모델에 대한 적용

2026-10-03에 확인한 예시 config는 다음과 같다. 이는 기준 fixture로 저장하되 실행 시에는 실제 revision을 다시 고정한다. [R02]

| 항목 | 값 |
|---|---|
| 모델 ID | `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` |
| architecture | `Qwen3_5ForConditionalGeneration` |
| 텍스트 레이어 | 32 |
| attention 구성 | linear 24 / full 8 |
| hidden size / vocab | 4,096 / 248,320 |
| 선언된 position 상한 | 262,144 |
| embedding/head 공유 | `false` |
| vision config | 존재 |

따라서 일반 Dense 식만으로 계산하지 않고 hybrid adapter를 적용한다. 비전 모듈을 실제 loader가 로딩하면 frozen이어도 상주 weight에 포함한다. 텍스트 전용 loader는 해당 checkpoint와의 호환성을 검증한 경우에만 선택한다.

모델 카드에는 자체 tokenizer와 MiMo chat template이 포함된다고 명시되어 있다. 다른 Qwen 모델의 template으로 대체하지 않는다. [R03]

### 6.5 실패 처리

Tokenizer가 없거나 remote code가 필요하거나 header를 읽을 수 없으면 원인을 반환한다. 큰 모델을 일반 CPU 초기화하여 RAM을 소진하는 fallback, 무제한 weight 다운로드, 사용자에게 알리지 않는 다른 tokenizer 사용은 금지한다.

## 7. 데이터셋 분석과 무절단 계약

### 7.1 Dataset Inspector

지원 입력은 HF dataset과 표준 JSON/JSONL/Parquet/Arrow/CSV이다. 스키마 탐지용 소량 preview와 실제 전체 스캔은 분리한다. 데이터셋에 포함된 코드, 환경, reward 스크립트는 길이 분석을 위해 실행하지 않는다.

학습용 split과 선택된 eval split만 분석 범위로 삼는다. `train` 자동 선택 여부, config, 파일 목록, row 수, snapshot identity를 결과에 표시한다. 사용자가 선택하지 않은 test split까지 학습 데이터로 결합하지 않는다.

### 7.2 컬럼 매핑

예시 데이터셋의 공개 컬럼은 `system`, `question`, `chosen`, `rejected`, `lang`, `vulnerability`를 포함한다. 다음 mapping을 제안한다. [R04]

| 원본 컬럼 | 역할 |
|---|---|
| `system` | system message |
| `question` | user prompt |
| `chosen` | preferred response |
| `rejected` | dispreferred response |
| `lang`, `vulnerability` | metadata. 자동 prompt 삽입 금지 |

방식별 변환을 화면에 명시한다. DPO는 preference pair, SFT는 chosen branch, GRPO는 prompt를 사용한다. SFT에서 rejected를 학습하지 않는 것은 **선택한 데이터 변환**이며, 응답 중간을 잘라내는 truncation과 구분한다.

무절단의 범위는 선택한 mapping에 의해 정의된 학습 레코드이다. `모든 원본 컬럼을 모두 loss에 사용한다`는 뜻이 아니다.

### 7.3 실제 Trainer와 같은 토큰화

```text
원본 row
 → objective별 구조화
 → system / messages / tools / thinking 옵션 적용
 → 모델의 실제 chat template 적용
 → Trainer와 동일한 BOS/EOS·경계·label-mask 처리
 → truncation 없이 token IDs 생성
 → 길이·mask·branch·오류 통계 저장
```

역할 구분과 제어 토큰은 최종 시퀀스의 일부이므로 원문 글자 수로 대체하지 않는다. Template을 문자열로 만든 뒤 토큰화하는 경로에서는 특수 토큰이 중복 추가되지 않도록 처리한다. [R07]

Prompt와 completion을 따로 토큰화해 길이를 무조건 더하지 않는다. DPO의 prompt 추출·response 경계, SFT의 label shift와 assistant-only loss, GRPO의 generation prompt 처리까지 **backend adapter의 전처리와 동등하게** 구성한다.

단순히 `truncation=False` 한 줄을 설정하는 것으로 계약을 충족했다고 판단하지 않는다. 실제 Trainer 입력 token IDs와 collator shape를 회귀 테스트로 대조한다.

분석용 tokenizer에는 길이 절단을 적용하지 않는다. 실행 설정을 내보낼 때는 선택한 API가 지원하는 경우 `max_length=None`처럼 길이 제한 해제를 명시한다. 정수 상한이 반드시 필요한 구현은 전체 스캔으로 확인한 필요 길이 이상을 사용하고, 실제 전처리가 어떤 row도 자르지 않는지 검증한다. 라이브러리의 숨은 기본 길이 제한을 그대로 남기지 않는다.

### 7.4 무절단 통과 조건

| 조건 | 검사 |
|---|---|
| 선택한 데이터 전체 읽기 | snapshot의 모든 shard 끝까지 읽고 성공/실패/미처리 row 집계 |
| 길이에 따른 삭제 없음 | overlength row drop, tokenizer truncation, collator clipping 금지 |
| 샘플 문맥 보존 | 긴 샘플 자동 분할, unrelated sample 연결 금지 |
| Template 손실 검사 | 학습 대상 content를 template이 제거하면 별도 경고 또는 실패 |
| 마지막 batch 포함 | `drop_last` 등으로 일부 row가 빠지지 않는지 검사 |
| Packing 안전성 | 켜는 경우 overflow·경계·attention isolation 검증 |
| Context 검증 | 모델 선언값과 실제 backend 지원 범위를 구분하여 검사 |
| 평가 범위 일치 | eval을 포함한다면 eval 데이터도 같은 검사를 통과 |

전체 scan 완료는 **데이터 길이 coverage**를 의미한다. Trainer sampler가 모든 row를 사용한다는 것은 별도의 실행 계획 검사다. GRPO의 group/batch 배수 조건으로 일부 row가 빠지는 구현은 통과시키지 않는다. Tail을 지원하는 sampler가 없으면 전체 coverage를 지원하지 않는 조합으로 보고한다. 중복 샘플을 추가하는 대안도 학습 분포 변경으로 표시한다.

현재 TRL 문서에는 기본 길이 제한과 packing 전략에 따른 overflow 처리 차이가 설명되어 있다. `bfd_split`처럼 토큰 수를 보존하더라도 원래의 긴 문맥을 분할하는 방식은 본 제품의 엄격 모드 기본값이 아니다. [R11]

### 7.5 Context 상한 처리

`config.max_position_embeddings`, tokenizer의 `model_max_length`, backend에서 검증한 최대 길이는 별도로 저장한다. Tokenizer의 비정상적으로 큰 sentinel 값을 실제 지원 context로 해석하지 않는다.

초과 시 `CONTEXT_EXCEEDED`를 반환하며 자동 truncation, RoPE 변경, row 제외를 하지 않는다. GPU 용량을 늘리는 것만으로 모델의 context 제약이 해결된다고 안내하지 않는다.

### 7.6 스캔과 저장 구조

원문 전체를 RAM에 올리지 않고 유한 snapshot을 순차 처리한다. Datasets streaming은 읽기 경로로 활용하되 스트리밍이라는 이유만으로 전체 coverage가 자동 확정되는 것은 아니다. [R14]

기본적으로 row별 길이 레코드는 Parquet/Arrow artifact에 기록하고, 원문·전체 token IDs는 저장하지 않는다. GPU 검증 시 필요한 row는 고정 snapshot에서 다시 읽는다. 연결이 끊긴 source를 다시 읽을 수 없으면 검증할 수 없다고 표시한다.

저장할 최소 정보:

```text
row_id, shard_id, split, objective
prompt_tokens, chosen_total_tokens, rejected_total_tokens
sft_total_tokens, loss_token_count, completion_token_count
template_fingerprint, tokenizer_fingerprint, content_digest
processing_status, context_status, error_code
```

Token IDs 전체를 저장하지 않는 모드에서도 검증용 hash와 필요한 mask 요약은 남긴다. 새 loss backend가 요약보다 더 자세한 mask를 요구하면 해당 분석을 재실행한다.

### 7.7 통계와 전체 스캔 완료 조건

총 row, 성공/실패/미처리 row, 총 tokens, P50/P90/P95/P99, 평균, 최대 길이, 최대 길이 row 위치, context 초과 수, 누락·중복 상태를 제공한다.

Count와 maximum은 정확 집계한다. Quantile을 sketch로 계산하면 `근사 분위수`로 표시한다. 전체 스캔의 VRAM 기준에는 P99 대신 최악 batch shape를 사용한다.

원본 전체 row 수를 사전에 알 수 없으면 진행률 퍼센트를 꾸며내지 않고 `처리 row 수 / 읽은 shard 수`를 표시한다. 모든 shard의 정상 EOF와 manifest 일치 후에만 complete로 전환한다. quota·parse failure·취소·무한 stream은 partial이다.

## 8. 학습 방식별 길이와 배치

### 8.1 SFT

각 row의 길이는 실제 학습 대화 전체로 계산한다. 일반 padding batch의 token slots는 다음과 같다.

```text
T_sft = B_actual × round_up(max(lengths_in_batch), pad_multiple)
```

`assistant_only_loss`처럼 loss를 계산하지 않는 prompt token이 있더라도 그 token의 forward와 학습에 필요한 activation을 0으로 간주하지 않는다. Loss backend에 따라 head projection 대상 위치가 달라질 수 있으므로 label mask는 별도 보존한다. SFT의 loss/masking 동작은 선택한 Trainer 구현에 맞춘다. [R08]

### 8.2 DPO

한 row에 대해 `prompt + chosen`, `prompt + rejected` 길이를 각각 유지한다. 두 response를 하나의 이어진 응답으로 합치지 않는다.

일반적인 concatenated branch batch의 출발점은 다음이다.

```text
T_dpo = 2 × B_pairs_actual
          × round_up(max(all_chosen_and_rejected_lengths), pad_multiple)
```

이는 모든 DPO backend의 보편적 고정식이 아니다. 실제 collator가 만드는 두 branch의 shape와 tensor 수명을 기준으로 수정한다. Branch를 순차 forward하더라도 backward 전에 두 graph가 남는지 확인해야 하므로 `SFT와 동일 메모리`로 단정하지 않는다.

### 8.3 GRPO

```text
prompt_length_i = 실제 generation prompt의 토큰 수
scenario_context_i = prompt_length_i + completion_budget
```

데이터셋의 chosen/rejected 길이는 미래 generation 길이를 정하지 않는다. 생성 응답에는 선택한 유한 budget을 적용한 시나리오를 제공한다. 완전히 무제한 생성의 피크를 하나의 유한 숫자로 보증하지 않는다.

다음 단위를 구분한다.

| 기호 | 의미 |
|---|---|
| U | generation 작업의 unique prompt 수 |
| G | prompt당 생성 수 |
| C | 동시에 live인 생성 sequence 수 |
| B_update | backward를 수행하는 update microbatch |
| K | gradient accumulation 수 |

실제 실행 adapter가 이를 generation batch와 device layout에 매핑한다. 기본 예시 `G=4, generation_batch=4, B_update=1, K=4`도 등록된 backend 검증을 통과해야 한다. UI에만 존재하는 임의의 `C=1` 최적화를 지원되는 실행 경로처럼 계산하지 않는다.

### 8.4 보상 미정의 GRPO

Preference dataset이 있다는 사실만으로 임의의 새 응답을 평가할 reward가 정의되는 것은 아니다. GRPO 실행에는 reward source가 필요하며 memory-only 분석과 실행 준비 상태를 분리한다. [R10]

Reward가 미정이면 policy/rollout 부분의 조건부 추정과 `reward footprint 미포함`을 표시한다. 이 결과를 **전체 학습 시스템의 확정 권장 용량**으로 표시하지 않는다.

Remote reward는 학습 GPU에 reward model을 올리지 않는 시나리오로 표현하되 원격 서버 자원이 0이라는 뜻으로 쓰지 않는다. Local reward model은 독립 모델 inventory와 resident schedule을 계산해야 한다.

### 8.5 최악 batch 계획

기본 microbatch=1에서는 최대 길이 row가 중요한 기준이다. Batch>1에서는 row별 joint length를 유지한 채 실제 sampler/seed의 batch 계획과 구조적 보수 시나리오를 함께 제공한다.

길이에 대해 메모리가 단조적인 등록 profile에서는 가장 긴 shape로 상한 시나리오를 구성한다. MoE routing, dynamic workspace처럼 길이만으로 최대를 설명할 수 없는 요소는 미확정으로 남긴다. 여러 epoch의 우연한 shuffle 결과가 모든 가능한 batch를 포괄한다고 주장하지 않는다.

## 9. 메모리 계산 엔진

### 9.1 GPU별 시간축 모델

각 실행 시점에 살아 있는 allocation을 합산한 뒤 최대값을 취한다.

```text
M(device, t) = W_base + W_adapter + W_other_models
             + gradients + optimizer_states + master_weights
             + saved_activations + logits_and_loss_buffers
             + generation_cache_or_recurrent_state
             + workspace + communication_buffers
             + allocator_slack + non_framework_allocations

M_peak(device) = max_t M(device, t)
```

일반 학습 forward의 saved activations와 생성 단계의 KV cache를 혼동하지 않는다. Gradient checkpointing을 사용하는 학습 경로에서는 일반적인 inference cache를 별도로 계속 더하지 않고 실제 `use_cache` 동작을 반영한다.

### 9.2 실행 단계

```text
MODEL_LOAD_AND_QUANTIZE
REFERENCE_PRECOMPUTE
ROLLOUT_PREFILL_AND_DECODE
REWARD
POLICY_FORWARD_BACKWARD
OPTIMIZER_STEP
WEIGHT_SYNC
EVALUATION
CHECKPOINT_SAVE_OR_CONSOLIDATE
```

서로 겹치지 않는 단계의 개별 최대값을 전부 합산하지 않는다. 그러나 optimizer state, 별도 reference, rollout 모델처럼 단계 사이에도 남는 메모리는 계속 포함한다. 비동기 단계는 실제 overlap을 반영한다.

기본 결과는 학습에 필요한 load→update 경로의 전체 피크다. Eval·save·분산 consolidation을 제외했다면 명확히 표시하고 `전체 수명주기 검증 완료`라고 표현하지 않는다.

### 9.3 가중치와 양자화

```text
W_dense = Σ(unique_resident_elements × effective_dtype_bytes)

W_quantized = Σ(quantized_payload + scales_and_other_metadata + alignment)
              + W_nonquantized_modules
```

4-bit payload는 대략 `ceil(elements/2)`에서 시작하지만 실제 backend별 block metadata, 제외 모듈, 로딩 과정의 transient가 추가된다. NF4 로딩은 모델 전체 실행 상태를 4-bit로 만드는 옵션이 아니다. [R05]

Base weight는 actual loading scope에 따라 계산하고, quantization 전 원본 shard와 변환 결과가 일시적으로 동시에 존재하는지 load-phase adapter에서 처리한다. 정적 분석 단계에서 실제 변환을 수행할 필요는 없지만 변환 경로의 가정은 기록해야 한다.

### 9.4 LoRA·gradient·optimizer

일반적인 2차원 Linear adapter의 파라미터 수:

```text
P_lora = Σ_j rank_j × (in_features_j + out_features_j)
```

Embedding LoRA, fused QKV, expert tensor, DoRA, bias, `modules_to_save`는 별도 규칙을 갖는다. 이름 문자열의 개수를 세는 방식 대신 실제 module dimensions로 계산하고 fixture에서 생성한 adapter count와 교차 검증한다.

```text
M_trainable = trainable_weights
            + trainable_gradients
            + actual_optimizer_states
            + required_master_copies
```

예를 들어 FP32 Adam moment 두 개는 대상 parameter당 합계 8 bytes지만, 전체 학습 저장량이 항상 일정 bytes/parameter인 것은 아니다. 실제 gradient dtype, master copy 존재, 작은 tensor의 state 처리, sharding/offload를 따로 모델링한다.

Accumulation 횟수를 activation 메모리에 그대로 곱하지 않는다. 일반적으로 반복 backward 사이에 graph는 해제되지만 gradient와 optimizer state는 유지된다. Graph 누적이나 rollout buffer 재사용이 있는 실행은 Trainer schedule로 별도 처리한다.

### 9.5 Activation과 logits: 구현 규격

Activation은 `레이어 수 × 길이 × 임의 계수`만으로 끝내지 않는다. Architecture adapter가 주요 allocation의 shape와 수명을 기술하는 **shape ledger**를 반환하게 한다.

```text
AllocationSpec
  name
  device
  shape_expression
  dtype
  storage_alias_group
  created_at / last_used_at
  saved_for_backward
  recompute_group
  evidence: analytic | calibrated | measured | unknown
```

최소 분리 대상은 embedding output, Q/K/V 또는 fused projection, attention output, normalization, FFN 중간값, residual/checkpoint boundary, LM head, loss buffers다. Padding shape, 실제 모듈 구조, trainable boundary와 checkpoint granularity를 반영한다.

Dense attention의 eager/메모리 효율 구현을 서로 다른 profile로 둔다. Hybrid의 recurrent inference state를 training activation과 동일하게 계산하지 않는다. 구현을 설명할 수 없는 custom kernel은 임의 상수 대신 unknown으로 처리한다.

일반 LM head의 단일 logits tensor 크기는 다음과 같다.

```text
M_logits_tensor = processed_positions × vocab_size × logits_dtype_bytes
```

이는 전체 loss/backward 피크와 같지 않다. Chunked/fused loss는 어떤 tensor를 만들지 않는지 profile에 명시해야 한다. 현재 SFT 문서에도 chunked 경로와 일반 경로가 구별되어 있으므로, 버전과 호환성을 확인하지 않고 모든 학습에 같은 절감률을 적용하지 않는다. [R08]

### 9.6 DPO reference의 수명

Reference의 별도 가중치, shared-base adapter 전환, reference log-prob precompute를 다른 schedule로 구현한다. Precompute는 학습 단계의 reference 부담을 바꾸지만 사전 계산 단계의 피크와 cache 저장 비용을 없애지는 않는다.

Reference cache key에는 reference checkpoint/adapter hash, model/tokenizer/template, source snapshot, mapping, objective preprocessing을 포함한다. Reference가 변하면 캐시를 무효화한다. IterableDataset과 precompute의 호환성도 선택한 Trainer에서 검사한다. [R09]

### 9.7 GRPO rollout과 cache

일반 full-attention 부분의 KV payload 출발점:

```text
M_KV = 2 × Σ_attention_layers(kv_heads × head_dim × kv_dtype_bytes)
           × resident_cached_token_positions
```

C개 sequence가 모두 최대 길이에 도달하는 시나리오를 기본 보수 입력으로 삼되, block rounding, prefix sharing, sliding window, quantization metadata, TP 배치를 실제 backend 규칙으로 처리한다. Prefix sharing 절감은 해당 경로가 확인된 경우에만 적용한다.

Hybrid recurrent state와 linear-attention workspace는 독립 계산한다. 또한 rollout old/reference log-probs, masks, advantages와 생성 결과의 CPU/GPU 저장 위치를 ledger에 포함한다. GRPO라는 이유로 PPO의 별도 value model을 자동 추가하지 않고 실제 구성된 모델만 센다.

별도 vLLM 경로에서는 학습 policy의 4-bit 형식이 rollout 가중치에도 그대로 공유된다고 가정하지 않는다. Cache pool 예약량, CUDA graph/workspace, device allocation을 계산해야 한다. `gpu_memory_utilization`은 동작 설정이며 그 값 자체가 실제 token payload 크기는 아니다. [R18]

### 9.8 다중 GPU 확장 계약

DDP는 복제 상태와 통신 buffer, FSDP/ZeRO는 어떤 상태를 shard하는지와 gather 순간, TP/PP는 모듈·activation 배치를 각각 모델링한다. FSDP와 DeepSpeed의 sharding 동작은 구별해야 한다. [R17]

`16 GiB + 16 GiB = 단일 32 GiB`로 판단하지 않는다. 결과는 rank별 maximum과 phase다. 통신 속도, PCIe/NVLink 병목, 학습 시간은 VRAM 적합과 별개의 지표로 남긴다.

## 10. RAM·디스크·메모리 여유

### 10.1 메모리 영역의 분리

| 출력 | 포함 대상 |
|---|---|
| 분석 서버 RAM | tokenizer, row buffer, 통계, worker 병렬 처리 |
| 학습 노드 RAM | 모델 로딩 staging, dataloader, CPU offload, reference cache |
| GPU VRAM | 각 장치의 실제 학습/생성/변환 상태 |
| 로컬 디스크 | 다운로드 cache, row-length artifact, 선택적 token cache, checkpoint |

RAM도 단계별 최대값으로 추정한다. Dataset 파일 전체 크기를 그대로 필요 RAM으로 취급하지 않는다. Worker 수, prefetch, fork/spawn 복제, offload buffer, pinned memory를 입력 가정으로 남긴다.

분석 서버가 Mac이어도 대상 학습 환경이 CUDA이면 결과는 CUDA 환경 가정이다. Apple unified memory 128GB를 CUDA VRAM 128GiB로 자동 변환하지 않는다. MLX/MPS 추정은 별도 engine에서 구현한다.

### 10.2 예상 범위와 권장 용량

확인된 resident allocation은 `known_floor_bytes`, 지원 profile의 통상·보수 시나리오는 `scenario_low_bytes`, `scenario_high_bytes`로 제공한다. 이 범위는 검증 없이 통계적 신뢰구간이라고 부르지 않는다.

`scenario_high`를 산정할 근거가 없으면 `null`이다. Floor에 임의의 배율을 곱해 권장 용량을 생성하지 않는다.

초기 제품의 계획용 margin 정책은 다음으로 제안한다.

```text
planning_margin = max(2 GiB, scenario_high × 0.15)
recommended_application_capacity = scenario_high + planning_margin
required_total_device_capacity = recommended_application_capacity
                                + external_reserved_bytes
```

이는 사용자가 조정할 수 있는 **운영 여유 정책**이지 오차 보증이 아니다. Profile의 allocator slack과 운영 margin을 별도 항목으로 표시한다. 외부 점유를 이미 free-memory 입력에 반영했다면 다시 더하지 않는다.

### 10.3 적합 판정

| 조건 | 화면 표시 |
|---|---|
| 확정된 상주 floor만으로도 가용량 초과 | 확정된 구성만으로 용량 초과 |
| 전체 분석·호환성 검사가 통과하고 high+margin이 가용량 이내 | 선택한 가정에서 예상 적합 |
| high는 들어가지만 margin 확보 불가 | 여유 부족 |
| high가 가용량 초과하지만 floor는 이하 | 예상 용량 초과, 실측/설정 검토 필요 |
| 중요한 footprint 또는 backend 지원이 미확정 | 판정 보류 |
| 하드웨어 미선택 | 용량만 표시, 적합 판정 없음 |

OOM 없이 실행된 GPU 검증 결과가 있어도 학습 전체의 무조건 성공으로 표현하지 않는다. 검증된 길이·단계·장치·버전을 함께 제시한다.

## 11. 프리셋과 버전별 호환성

### 11.1 자동 모드의 의미

`Auto`는 모든 메모리 절감 옵션을 켜는 기능이 아니다. 등록된 profile 중 **데이터 보존 → 학습 목적 보존 → 실행 호환 → 메모리 효율** 순서로 선택하는 resolver다.

품질에 영향을 주는 rank, beta, generation 수, response budget, 데이터 변환을 조용히 변경하지 않는다. Microbatch를 줄이면서 accumulation을 바꾸는 경우에도 effective batch와 Trainer의 group 제약이 유지되는지 검사한다.

자동 최적화는 원본 설정을 수정하는 대신 **비교 시나리오**를 생성한다. 사용자가 선택한 뒤에만 활성 설정으로 적용한다.

### 11.2 Compatibility registry

```text
profile_id / profile_version
architecture_adapter / trainer_adapter
dependency_lock_digest / container_image_digest
supported_objectives / supported_strategies
supported_hardware_capabilities
loading_scope / effective_dtypes / quantization_rules
attention_path_by_layer_type / loss_path
checkpointing / optimizer / reference / rollout_schedule
preprocessing_contract / no_truncation_contract
workspace_assumptions / calibration_coverage
unsupported_combinations / fallback_rules
```

Profile은 Python, PyTorch, Transformers, PEFT, TRL, bitsandbytes, attention/loss kernel, CUDA, driver 조건을 기록한다. 계획서에서 `latest` 조합을 확정하지 않는다. 구현 단계에서 호환 smoke test를 통과한 버전을 lock하고 CI에 넣는다.

원격 문서의 기본값은 설명 자료다. 실제 프로그램은 설치한 클래스의 config signature, 검증된 source revision, golden preprocessing fixture를 기준으로 동작한다.

### 11.3 실제 적용 옵션을 기준으로 계산

현재 TRL 문서는 DPO padding-free의 일반 padding fallback과 SFT chunked loss의 PEFT/VLM 등 호환 제한을 명시한다. 따라서 QLoRA에 모든 최신 memory 옵션이 동시에 적용된다고 가정하면 안 된다. [R11]

결과는 다음 세 값을 함께 저장한다.

```text
requested:  사용자가 요청한 설정
resolved:   registry가 계산에 반영한 설정
observed:   GPU 검증에서 실제 확인한 실행 경로, 미검증이면 null
```

실측에서 다른 kernel이 선택되면 기존 결과와 다르다는 경고를 내고 적용 profile을 재선택한다. Unsupported option을 무시한 채 이전 절감량을 유지하지 않는다.

### 11.4 지원 등급

| 등급 | 허용 결과 |
|---|---|
| Metadata only | 구조·weights·trainable inventory. 전체 VRAM 적합 판정 금지 |
| Analytic | 명시적 allocation·workspace 가정을 가진 정적 시나리오 |
| Calibrated | 등록 hardware/software 영역의 보정된 정적 결과 |
| Measured | 해당 job에서 관측한 단계별 peak |
| Unsupported | 지원하지 않는 조합과 해결에 필요한 조건 |

첫 버전은 최소한 등록 Dense의 SFT/DPO/GRPO 조합에서 analytic 결과를 실제 계산해야 한다. Hybrid adapter도 구조별 allocation ledger를 구현하되 미검증 kernel 구간을 표시한다. 어떤 모델이든 이름만으로 generic multiplier에 넣어 모든 입력을 성공 처리하지 않는다.

Analytic profile의 workspace·allocator 가정은 수식, 근거, 설정값을 버전 관리한다. 보수 시나리오라고 해서 수학적으로 증명된 upper bound라는 뜻은 아니며, 근거가 없는 경우 upper 값을 만들지 않는다.

## 12. 결과 데이터와 상세 화면

### 12.1 상태는 독립된 축으로 표현

| 축 | 예시 값 |
|---|---|
| scan coverage | not_started / partial / complete / failed |
| data preservation | pending / verified / violated / unknown |
| training readiness | ready / conditional / unsupported |
| estimate evidence | metadata_only / analytic / calibrated / measured |
| hardware fit | not_evaluated / expected_fit / low_margin / exceeds / unknown |

전체 스캔은 완료했지만 GRPO reward가 미정일 수 있다. 데이터가 모두 보존되어도 모델 context를 초과할 수 있다. 이 상황을 단일 초록색 `성공` 배지로 뭉치지 않는다.

### 12.2 상세 탭

| 탭 | 내용 |
|---|---|
| 메모리 구성 | 피크 순간의 weights, adapters, gradients, optimizer, activations, cache, temporary 구성 |
| 데이터 길이 | histogram, branch별 분포, 최대 길이, coverage, 오류/초과 목록 |
| 단계별 피크 | load, precompute, rollout, update, optimizer, eval, save 비교 |
| 시나리오 비교 | rank, batch, response budget, offload 등 변경별 delta와 조건 |
| 적용 설정·근거 | requested/resolved/observed, 정확한 버전, formulas, 불확실성, source identities |

메모리 누적 막대의 합은 **선택한 동일 시점**의 합이어야 한다. 각 component가 서로 다른 시점에서 기록한 개별 maximum을 쌓아서 총 피크처럼 그리지 않는다.

데이터 탭의 긴 row는 기본적으로 ID와 길이만 표시한다. 원문 미리보기는 권한 검사 후 사용자가 요청할 때만 제공한다.

### 12.3 필수 결과 필드

```text
analysis_id, schema_version, created_at, analysis_fingerprint
source_manifests, tokenizer_manifest, model_inventory_summary
dataset_scan, preservation_audit, context_validation
requested_config, resolved_config, compatibility_report
memory_by_device_and_phase, peak_timepoint_breakdown
host_ram_estimate, analysis_ram_estimate, disk_estimate
planning_margin_policy, hardware_fit
assumptions, unknown_components, warnings
profile_id, dependency_lock_digest, measurement_scope
```

GiB로 표현한 화면 수치는 표시용이며 원본 result는 정수 bytes를 사용한다. 알려지지 않은 값, 해당 없음, 측정하지 않음은 구별한다.

### 12.4 내보내기

| 파일 | 용도 |
|---|---|
| `analysis.json` | 원본 결과, 가정, schema version, 근거 |
| `resolved-plan.yaml` | 앱의 재현 가능한 계획 설정. TRL 인자와 1:1 동일하다고 주장하지 않음 |
| `report.md` | 사람이 읽는 결과, 데이터 통계, 지원 상태, 권장 용량 |
| `trainer-config.yaml` | 정확한 backend adapter가 ready 상태에서 생성한 실행용 설정만 제공 |

GRPO reward가 미정이거나 unsupported backend이면 계획 YAML은 생성하되 실행용 Trainer config에는 준비 미완료 상태를 표시하고 실행 대상으로 제공하지 않는다.

원문 데이터, HF token, 절대 로컬 경로, 사설 URL은 export에서 기본 제외한다. 필요한 경우 사용자가 범위를 선택해야 한다.

## 13. 시스템 아키텍처와 기술 스택

### 13.1 확정 스택

| 계층 | 선택 |
|---|---|
| Web | Next.js + React + TypeScript |
| UI | Tailwind CSS + 접근성 기반 headless components |
| Forms | React Hook Form + Zod |
| 서버 상태 | TanStack Query + SSE client |
| Charts | Recharts. 모든 chart에 표 형태 대체 제공 |
| API | FastAPI + Pydantic |
| CPU 분석 | Python + huggingface_hub + datasets + transformers/tokenizers + safetensors |
| 작업 큐 | Redis + RQ |
| 메타데이터 DB | PostgreSQL + SQLAlchemy + Alembic |
| 분석 artifact | 로컬 파일 볼륨. 확장 시 S3-compatible adapter |
| 테스트 | pytest, Vitest, Playwright |
| 패키지 관리 | Python uv lock, frontend pnpm lock |
| 배포 | Docker Compose. GPU worker는 별도 선택 profile |

스택 선택은 구현 계획이며, 특정 패키지 버전 간 호환성을 이 문서에서 실증했다는 의미가 아니다. Backend profile별 버전 잠금과 smoke test를 필수 작업으로 둔다.

### 13.2 구성도

```text
Browser
  │ same-origin HTTPS / REST / SSE
  ▼
Web + reverse proxy
  │
  ▼
FastAPI
  ├─ 인증·접근 제어 / source inspection API
  ├─ 분석 요청·상태·취소 / 시나리오 재계산
  ├─ 결과 내보내기
  ├─ PostgreSQL: jobs, ownership, manifests, results
  └─ Redis / RQ
       ├─ CPU worker: inspection / tokenization / batch / estimate
       └─ GPU worker: 명시적으로 요청된 검증만 실행

Workers ↔ 사용자별 cache / artifacts / read-only source mounts
```

Frontend에서 Python 메모리 식을 복제하지 않는다. 모든 최종 계산은 버전 관리되는 단일 Python core가 수행하며 브라우저는 결과와 근거를 렌더링한다.

### 13.3 작업 분리

긴 tokenization 작업을 API 프로세스의 일반 request thread에서 실행하지 않는다. Queue worker에서 수행하고 API는 job ID와 상태를 제공한다.

토큰화 worker가 대형 가중치 로딩을 수행하지 않게 의존성과 실행 권한을 분리한다. CPU 분석 서비스는 GPU 없이 실행 가능해야 한다. GPU worker는 별도 이미지와 장치 allowlist를 사용한다.

공개 웹앱과 학습 노드가 같은 머신일 필요는 없다. GPU worker 연결이 없으면 UI에 `GPU 검증 미연결`을 표시하되 정적 분석을 막지 않는다.

## 14. 모듈 인터페이스와 저장소 구조

### 14.1 핵심 인터페이스

| 모듈 | 입력 → 출력 |
|---|---|
| SourceResolver | source request → immutable manifest + authorized handles |
| ModelInspector | model manifest → tensor inventory + architecture facts |
| DatasetInspector | dataset manifest → configs/splits/schema/mapping candidates |
| PreprocessingAdapter | row + objective + tokenizer → tokenized record + preservation audit |
| BatchPlanner | row lengths/masks + trainer config → batch shapes + worst-case scenarios |
| ArchitectureAdapter | model inventory + shapes → allocation specs |
| TrainerAdapter | objective + resolved config → phase schedule + persistent buffers |
| MemoryEngine | allocations + schedule + device map → per-device peak scenarios |
| CompatibilityResolver | requested config + environment → resolved config + blockers |
| CalibrationRegistry | profile key + shape domain → supported calibration or miss |
| Exporter | authorized result → JSON/YAML/Markdown |

`ArchitectureAdapter`는 모델 구조, `TrainerAdapter`는 실행 절차를 담당한다. 두 책임을 하나의 거대한 `if model_name contains ...` 함수에 합치지 않는다.

### 14.2 계산 파이프라인 의사코드

```python
# 구현 흐름 명세이며 실행 가능한 앱 소스가 아니다.
def analyze(request, job_context):
    sources = source_resolver.resolve_and_authorize(request.sources)
    model = model_inspector.inspect_metadata(sources.model)
    resolved = compatibility.resolve(request, model)
    preprocessing = preprocessing_registry.for_objective(resolved)

    scan = scanner.full_scan(
        source=sources.dataset,
        preprocessing=preprocessing,
        strict_preservation=True,
        checkpoint=job_context.checkpoint,
    )
    shapes = batch_planner.plan(scan.length_records, resolved)
    allocations = architecture_registry.for_model(model).build_ledger(shapes, resolved)
    schedule = trainer_registry.for_profile(resolved).build_schedule(shapes)
    estimates = memory_engine.evaluate(allocations, schedule, resolved.devices)

    return result_builder.combine(
        scan=scan,
        estimates=estimates,
        assumptions=resolved.assumptions,
        blockers=resolved.blockers,
        evidence=resolved.evidence,
    )
```

미지원이 source/shape까지 영향을 주면 그 단계에서 멈추고 partial result를 반환한다. 예를 들어 tokenizer가 없는데 임의 길이를 만들어 나머지 단계를 실행하지 않는다.

### 14.3 디렉터리 구조

```text
vram-calculator/
├── plan.md
├── README.md
├── compose.yaml
├── .env.example
├── apps/
│   └── web/
│       ├── app/
│       ├── components/calculator/
│       ├── components/results/
│       ├── lib/api/
│       └── tests/
├── services/
│   ├── api/
│   ├── worker_cpu/
│   └── worker_gpu/
├── packages/
│   └── estimator/
│       ├── schemas/
│       ├── sources/
│       ├── inspection/
│       ├── preprocessing/
│       ├── batching/
│       ├── architectures/
│       ├── trainers/
│       ├── memory/
│       ├── compatibility/
│       ├── calibration/
│       └── exports/
├── profiles/
│   ├── analytic/
│   ├── environments/
│   └── calibrated/
├── tests/
│   ├── fixtures/
│   ├── unit/
│   ├── preprocessing_parity/
│   ├── integration/
│   ├── security/
│   └── gpu_regression/
└── docs/
    ├── methodology.md
    ├── support-matrix.md
    ├── privacy-and-retention.md
    └── deployment.md
```

## 15. API와 요청 예제

### 15.1 API 목록

| Method | Endpoint | 역할 |
|---|---|---|
| POST | `/api/v1/sources/inspect` | 모델/데이터셋 메타데이터 확인 |
| POST | `/api/v1/uploads` | 허용 데이터 파일 업로드 |
| GET | `/api/v1/backend-profiles` | 지원 조합과 환경 조건 |
| GET | `/api/v1/local-roots` | 권한이 있는 self-hosted root 목록 |
| POST | `/api/v1/analyses` | 분석 job 생성 |
| GET | `/api/v1/analyses/{id}` | 상태·부분/최종 결과 |
| GET | `/api/v1/analyses/{id}/events` | 진행 상태 SSE |
| POST | `/api/v1/analyses/{id}/cancel` | 취소 요청 |
| POST | `/api/v1/analyses/{id}/scenarios` | 캐시를 이용한 조건 비교 |
| GET | `/api/v1/analyses/{id}/export` | 권한 검사 후 보고서 생성 |
| DELETE | `/api/v1/analyses/{id}` | 결과·artifact 삭제 요청 |
| POST | `/api/v1/analyses/{id}/profile` | 후속 GPU 검증. 명시적 실행 요청 필요 |

생성 요청은 202와 job ID를 반환한다. 멱등 키, 사용자별 동시 작업 제한, 취소 상태, fingerprint를 지원한다. Schema validation과 backend capability validation은 모두 서버에서 다시 수행한다.

### 15.2 사용자가 제시한 입력의 내부 표현

다음은 **앱 내부 API 요청 예시**다. 그대로 TRL config에 전달하는 형식이 아니다. 숫자는 사용자 요구를 재현하기 위한 제품 preset이며 분석 결과가 아니다.

```json
{
  "schema_version": "1.0",
  "model": {
    "source_type": "huggingface",
    "reference": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
    "revision": null,
    "loading_scope": "auto_verified"
  },
  "dataset": {
    "source_type": "huggingface",
    "reference": "CyberNative/Code_Vulnerability_Security_DPO",
    "revision": null,
    "config": null,
    "split": "train",
    "scan_mode": "full",
    "mapping": {
      "system": "system",
      "prompt": "question",
      "chosen": "chosen",
      "rejected": "rejected"
    }
  },
  "training": {
    "objective": "grpo",
    "strategy": "qlora",
    "quantization": {
      "enabled": true,
      "format": "nf4",
      "double_quant": true,
      "compute_dtype": "auto"
    },
    "lora": {
      "r": 16,
      "alpha": 32,
      "dropout": 0.0,
      "target_modules": "auto_verified",
      "modules_to_save": []
    },
    "microbatch_per_device": 1,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": true,
    "packing": false,
    "data_policy": "strict_no_truncation",
    "backend_profile": "auto"
  },
  "grpo": {
    "num_generations": 4,
    "generation_batch_size": 4,
    "completion_budget": null,
    "completion_budget_candidates": [1024, 2048, 4096, 8192],
    "beta": 0.0,
    "reward": { "kind": "unspecified" },
    "rollout_backend": "transformers_shared_policy"
  },
  "hardware": { "mode": "capacity_only" },
  "scope": { "include_evaluation": false, "include_checkpoint_save": false },
  "profiling": { "enabled": false }
}
```

이 예시의 `beta=0.0`은 KL reference 항을 사용하지 않는 시나리오를 명시적으로 선택한 초기값이다. 메모리를 맞추기 위해 기존 사용자 값을 0으로 바꾸는 기능이 아니다.

`revision=null`은 분석을 시작할 때 실제 commit으로 resolve하라는 의미다. 최종 result에는 null 대신 확정 revision을 저장한다. Reward가 미정이므로 결과는 conditional이며 reward 메모리를 제외한 계산이라는 사실을 명시한다.

### 15.3 SSE 이벤트 계약

```text
id: <monotonic_event_id>
event: progress | partial_result | warning | completed | failed | cancelled
data: {analysis_id, stage, processed_rows, total_rows, shard_progress,
       fingerprint, message_code, timestamp}
```

원문 row나 인증 token을 SSE에 넣지 않는다. 브라우저 새로고침·재연결은 `Last-Event-ID`와 API 상태 조회로 복구한다. 진행률 event는 제한된 빈도로 합치되 최종 상태는 누락하지 않는다.

### 15.4 오류 모델

`code`, `stage`, `severity`, `retryable`, `user_message`, `technical_detail_ref`, `affected_component`를 반환한다. Stack trace와 credential은 사용자 메시지에서 제외한다.

```text
SOURCE_ACCESS_DENIED              SOURCE_REVISION_CHANGED
LOCAL_PATH_NOT_ALLOWED            MODEL_METADATA_UNAVAILABLE
TOKENIZER_REQUIRED                TEMPLATE_REQUIRED
REMOTE_CODE_REQUIRED              COLUMN_MAPPING_REQUIRED
SCAN_PARTIAL                      SCAN_FAILED_ROWS
DATA_PRESERVATION_VIOLATION        CONTEXT_EXCEEDED
GRPO_REWARD_UNSPECIFIED            GRPO_BUDGET_UNSPECIFIED
TRAINER_BATCH_CONSTRAINT          UNSUPPORTED_ARCHITECTURE
UNSUPPORTED_BACKEND_COMBINATION   REQUESTED_OPTION_NOT_EFFECTIVE
UNKNOWN_MEMORY_COMPONENT          CALIBRATION_OUT_OF_DOMAIN
PROFILE_OOM                       PROFILE_SCOPE_INCOMPLETE
```

## 16. 작업 상태·캐시·재현성

### 16.1 작업 상태 머신

```text
QUEUED
 → RESOLVING
 → INSPECTING
 → NEEDS_INPUT 또는 TOKENIZING
 → VALIDATING_DATA
 → PLANNING_BATCHES
 → ESTIMATING
 → COMPLETED

어느 단계에서든:
 → CANCEL_REQUESTED → CANCELLED
 → FAILED / PARTIAL
```

`COMPLETED`는 작업 실행이 끝났다는 뜻이다. 결과 자체의 data preservation, readiness, evidence 상태는 별도로 판정한다. Conditional 결과도 job은 정상 종료할 수 있다.

Queue는 실행 전달을 담당하고 PostgreSQL은 job 상태의 기준 저장소가 된다. Redis 재시작으로 정상 분석 결과가 사라지지 않게 한다. 재전달된 task는 job lease와 idempotency key로 중복 실행을 방지한다.

### 16.2 재개와 source 일관성

Shard/record checkpoint와 성공 artifact를 원자적으로 기록한다. Worker가 종료되면 새 worker는 만료된 lease를 회수하고 이어서 처리한다. Streaming source가 seek를 지원하지 않으면 checkpoint까지 다시 읽어야 할 수 있으며, 이를 무조건적인 즉시 재개로 표현하지 않는다.

HF snapshot은 commit과 파일 identity를 고정한다. 로컬 입력은 파일 manifest와 content digest로 변경을 검사하며, 분석 중 변경되면 완료 결과를 publish하지 않는다. 크기와 mtime만으로 불변성을 보장했다고 주장하지 않는다.

### 16.3 계층별 캐시 키

```text
source_key = source_type + immutable_revision_or_content_manifest

preprocess_key = source_key + selected_config_and_splits
               + mapping + objective + tokenizer_files_hash
               + chat_template_hash + template_kwargs
               + preprocessing_adapter_version + dependency_lock_digest

batch_key = preprocess_key + collator_config + sampler_config
          + microbatch + padding + packing + distribution_config

estimate_key = batch_key + model_inventory_hash + loading_scope
             + trainable_config + effective_dtypes + optimizer
             + trainer_schedule + runtime_profile + device_layout
             + calibration_version + planning_margin_policy
```

GRPO budget처럼 tokenization에 영향을 주지 않는 값은 estimate/batch 계층만 갱신한다. 텍스트를 바꾸는 tokenizer/template 설정은 반드시 preprocessing 계층을 갱신한다.

Private source의 cache와 결과는 owner/tenant별로 격리한다. 공개 모델의 검증된 메타데이터만 명시적 정책 아래 공유할 수 있다. 비공개 데이터 내용이나 존재 여부가 cache hit 응답으로 다른 사용자에게 노출되지 않게 한다.

### 16.4 보존 정책

기본 운영 정책은 raw source를 서비스 로그에 남기지 않고 분석 산출물을 작업 소유자에게만 제공하는 것이다. 업로드 원본, length artifact, 결과, GPU 검증 자료별 retention을 설정하고 UI에 표시한다.

삭제 요청은 DB record만 지우지 않고 연결된 source upload·artifact·cache 참조를 정리한다. 다른 사용자가 소유한 shared public metadata cache까지 함께 삭제하지 않는다.

## 17. 선택적 GPU 검증

### 17.1 실행 전 계약

사용자가 `GPU 검증 실행`을 누르면 사용할 장치, dependency profile, 최대 작업 범위, 모델 다운로드 및 저장 범위, 데이터 전달 범위를 먼저 표시한다. 정적 분석 버튼에서 자동 실행하지 않는다.

사용자의 기존 GPU 작업을 종료하거나 드라이버를 변경하거나 메모리를 강제로 비우지 않는다. 검증에 필요한 여유가 없으면 시작하지 않고 이유를 반환한다.

### 17.2 측정 시퀀스

1. GPU별 total/free, 외부 점유, driver/CUDA와 패키지 버전을 기록한다.
2. 모델 로딩·양자화 transient부터 계측한다.
3. 정적 분석과 동일한 Trainer, preprocessing, collator, optimizer, kernel을 구성한다.
4. Warm-up 후 실제 forward, backward, optimizer update를 수행한다.
5. Optimizer state가 생성된 다음에도 추가 update를 수행하여 첫 step 이전의 낮은 수치만 기록하지 않는다.
6. 대표 길이, 긴 실제 row, 최악 batch shape를 검증한다.
7. GRPO는 rollout → reward → update → sync → 다음 rollout까지 검사한다.
8. 선택된 eval/save/consolidation을 별도 phase로 검사한다.
9. OOM이면 실패 장치·phase·shape를 저장하고 정상 결과로 바꾸지 않는다.

생성 결과가 일찍 EOS로 끝나면 해당 실측은 실제 생성 길이에 대한 결과다. 요청한 completion budget 최악값을 검증했다고 표시하지 않는다. 최대 길이 stress test가 필요하면 별도의 합성 shape 검증으로 분리하고 실제 학습과 다른 조건을 기록한다.

### 17.3 측정 지표

`max_memory_allocated`와 `max_memory_reserved`를 구분한다. Reserved는 allocator가 확보한 공간을 포함하므로 두 값을 더하지 않는다. Phase 간 동기화와 peak reset 경계도 명시한다. [R15]

NVML 등으로 장치·프로세스 사용량을 함께 관찰한다. PyTorch 메모리 추적은 NCCL 등 직접 CUDA 할당을 전부 포착하지 못할 수 있으므로 관측 범위를 기록한다. [R16]

장치 전체 사용량에는 다른 프로세스가 포함될 수 있다. 외부 점유가 변하는 검증은 그 사실을 표시하고 calibration 학습에서 제외하거나 구분한다. 서로 다른 지표의 peak를 한 그래프에 같은 의미로 섞지 않는다.

### 17.4 보정 profile

실측값과 analytic ledger의 차이는 같은 phase·같은 metric으로 비교한다. Profile의 key에는 architecture, loading scope, objective, adapter, dtype, kernel, GPU, dependency lock, 길이 범위, batch 구성을 넣는다.

일부 짧은 길이에서 맞았다고 128K 이상의 context까지 같은 정확도를 주장하지 않는다. Calibration 영역을 벗어나면 `CALIBRATION_OUT_OF_DOMAIN`으로 표시하고 analytic 수준으로 낮추거나 결과를 보류한다.

GPU validation에서 임시로 optimizer/LoRA checkpoint가 생성되더라도 원본 모델 디렉터리를 수정하지 않는다. 격리된 작업 디렉터리에 저장하고 retention 정책에 따라 정리한다.

## 18. 보안·개인정보·자원 제한

| 위협/문제 | 필수 설계 |
|---|---|
| 외부 repository code 실행 | 기본 `trust_remote_code=False`; 필요한 경우 unsupported. 분석용 자동 승인 금지 |
| 악성 tokenizer/template | 검증된 라이브러리, 제한된 subprocess, template sandbox, 시간·메모리 제한 |
| 임의 URL/내부망 접근 | provider별 resolver, redirect 재검사, 사설 주소·metadata endpoint 제한 |
| 경로 탈출 | canonical path·등록 root 검사, symlink와 mount 경계 검사 |
| 위험 직렬화 | safetensors·표준 데이터 우선, 임의 pickle 실행 금지 |
| 압축 폭탄/과대 입력 | 파일·압축 해제·row·시간·메모리 quota; 초과 시 partial/error |
| 인증정보 유출 | 최소 권한 HF token, 서버 secret 참조, 로그·URL·export·browser storage 제외 |
| 다른 사용자 job 조회 | analysis/artifact/credential 전체 owner 검증 |
| 업로드·결과 XSS | 원문은 text로 렌더링, Markdown HTML 정제 |
| 장시간 작업 악용 | 사용자별 동시 작업 제한, 취소·timeout, worker 격리 |
| GPU 검증의 임의 reward 실행 | 첫 버전 계산기는 reward 메타데이터만 사용; 실제 코드는 별도 승인된 검증 환경에서만 |

Private 모델 접근 시 서비스의 전역 HF 계정으로 사용자 접근권한을 대신하지 않는다. 원격 redirect나 로그에 signed URL·token이 남지 않게 redaction을 적용한다.

Quota에 걸린 데이터는 자르거나 건너뛰고 `전체 스캔 완료`라고 표시하지 않는다. **서비스 자원 보호와 데이터 무절단은 동시에 지키되, 처리하지 못한 경우 미완료로 보고한다.**

Self-hosted의 기본 접근은 localhost로 제한한다. 공개 배포에는 HTTPS, 인증, 작업 소유권 검사, cookie 사용 시 CSRF 대책을 요구한다. Admin용 local root 등록 기능을 일반 사용자에게 노출하지 않는다.

## 19. 테스트와 수락 기준

### 19.1 CPU-only 단위 테스트

아래는 실제 모델 VRAM 값이 아니라 수식과 shape 로직 검증을 위한 합성 fixture다.

| 항목 | 입력 | 기대값 |
|---|---|---|
| GiB 변환 | `1,073,741,824 bytes` | 1 GiB |
| Linear LoRA | in=4096, out=4096, r=16 | 131,072 parameters |
| Alpha 변화 | 동일 모듈/r, alpha 16→64 | adapter parameter 수 동일 |
| SFT padding | B=2, 길이 100/300, pad_multiple=1 | 600 token slots |
| DPO padding | 2 pairs, branch 길이 100/150/700/200 | 2,800 token slots |
| KV payload | 2 full-attention layers, 4 KV heads, head_dim=128, 2-byte dtype, cached positions=1024 | 4,194,304 bytes |
| Phase 최대값 | 동시에 존재하지 않는 10GiB/12GiB phase | 22GiB가 아니라 12GiB |
| Unknown | activation profile 없음 | activation=0이 아니라 null, 적합 판정 보류 |

### 19.2 전처리·무절단 회귀 테스트

| 테스트 | 수락 기준 |
|---|---|
| Trainer parity | 등록 profile의 실제 Trainer와 token IDs, masks, collator shape 일치 |
| Special tokens | BOS/EOS 중복 없음, prompt 경계 일치 |
| 한글·코드·긴 문자열 | 입력 형식별 정확한 tokenizer 경로 사용 |
| DPO outlier | rejected만 긴 row도 피크에 반영 |
| Prompt masking | loss 미적용 prompt가 activation에서 사라지지 않음 |
| Template 변경 | 기존 preprocessing cache를 재사용하지 않음 |
| Oversized sample | 자동 truncation·분할·삭제 없이 초과 보고 |
| 손상 row | 실패 건수와 row 위치 보존, complete 배지 금지 |
| Tail batch | 마지막 row 포함 여부 검사 |
| Split 범위 | 선택하지 않은 split이 분석·학습에 섞이지 않음 |
| 데이터 변환 | DPO→SFT chosen 선택과 GRPO prompt-only 변환을 명시 |
| Partial stream | quota/취소/중단을 전체 최대 길이로 포장하지 않음 |

### 19.3 메모리 모델 테스트

Quantization 제외 모듈, tied weights, frozen module, modules_to_save, reference precompute, shared-policy rollout, optimizer state 생성, load-phase transient, gradient accumulation, unsupported kernel fallback을 각각 fixture로 만든다.

Synthetic model은 random/meta 구조로 작게 생성해 실제 module parameter count와 비교한다. 큰 모델 weight를 CI마다 다운로드하지 않는다. 공개 config fixture는 source revision과 hash를 남기고 업데이트 변경점을 검토한다.

Hybrid fixture는 full-attention 8개만 KV 식에 넣고 linear layer의 별도 state/activation 경로를 호출하는지 검사한다. 모든 32개 레이어에 일반 KV 식을 적용하면 실패해야 한다.

### 19.4 UI/API 통합 테스트

| 시나리오 | 수락 기준 |
|---|---|
| 초기 진입 | 가짜 0GB/적합 상태 없음 |
| 모델/데이터 입력 | metadata 상태와 mapping 표시 |
| 전체 분석 | 진행 상태 → coverage → 결과 흐름 정상 |
| 재계산 | r 변경 시 원본 데이터 재다운로드·재토큰화 없음 |
| 오래된 요청 응답 | 다른 fingerprint의 결과가 현재 화면을 덮지 않음 |
| GRPO 미정 입력 | budget 시나리오와 reward 미지정 경고 |
| GPU 미선택 | 불필요한 적합 게이지 없음 |
| 모바일 | 주요 조작과 결과 확인에 페이지 가로 스크롤 불필요 |
| 키보드 | 입력·Advanced·탭·export 모두 접근 가능 |
| 작업 재연결 | 새로고침 후 job 진행 상태 복구 |
| 결과 export | byte 수치, fingerprints, 설정, 경고 일치 |
| Unauthorized access | 다른 owner의 analysis/artifact/credential 접근 차단 |
| CPU-only 배포 | GPU 없는 환경에서 전체 데이터 분석 완료 |

### 19.5 GPU 정확도 검증과 발표 규칙

등록된 calibration profile별로 모델 구조, objective, rank, batch, 길이, dtype, kernel이 다른 검증 구간을 구성한다. Calibration에 사용하지 않은 holdout 조합을 별도로 평가한다.

다음 지표를 profile과 함께 보고한다.

```text
signed_error_bytes = estimated_peak - measured_peak
relative_absolute_error = abs(estimated_peak - measured_peak) / measured_peak
false_fit_count = 예상 적합으로 분류했으나 검증 중 OOM한 경우
measurement_coverage = 검증한 단계 / 요청한 단계
```

등록 지원 구간의 holdout에서 false-fit이 발생하면 여유 정책·profile을 수정하거나 그 구간을 지원 대상에서 제외한다. 단순 평균 오차만으로 underestimation을 숨기지 않는다. 실측이 없는데 `정확도 95%` 같은 문구를 넣지 않는다.

## 20. 구현 단계와 산출물

시간 추정 대신 **완료 조건이 있는 milestone**으로 진행한다. 각 milestone은 앞 단계의 API/fixture를 재사용한다.

| 단계 | 구현 내용 | 완료 조건 |
|---|---|---|
| M0: 계약·골격 | monorepo, schema, profiles, compose, mock server, CI | CPU-only 서비스 기동, API types 생성, 상태 모델 확정 |
| M1: 계산기 UI | 기본 입력, Advanced, sticky result, 상세 탭, 모바일 | 모든 상태에 대한 UI 테스트. Mock 수치는 개발 모드에만 존재 |
| M2: 실제 source·토큰화 | HF/local resolver, inventory, mapping, full scan, coverage audit | 실제 tokenizer 사용, 전체 scan, no-truncation 회귀 통과 |
| M3: 계산 엔진 | Dense/hybrid ledger, SFT/DPO/GRPO schedule, RAM, margin, warnings | 합성 수식·module count·Trainer shape parity 통과 |
| M4: 통합 출시 | 재계산 캐시, job 복구, export, 보안, 운영 문서 | 예시 입력 end-to-end 실제 분석, 허위 지원·가짜 수치 없음 |
| M5: GPU 검증 | isolated worker, phase peak, observed config, calibration | 실제 측정 범위와 오차 보고, OOM phase 기록 |
| M6: 확장 | 분산/vLLM/MoE/멀티모달/local companion/MLX | 각 확장마다 독립 profile·fixture·지원표 추가 |

첫 사용 가능한 버전은 M0–M4다. M5 없이도 정적 계산기는 사용할 수 있지만 해당 결과를 실측이라고 표시하지 않는다.

### 20.1 구현자가 우선 작성할 문서

`methodology.md`에는 실제 사용하는 수식·phase·unknown 처리와 margin을 기술한다. `support-matrix.md`에는 objective × architecture × strategy × backend 조합의 지원 등급을 기록한다. 두 문서는 코드 registry와 자동 대조하는 테스트를 둔다.

### 20.2 출시 체크리스트

- [ ] 입력 예제의 모델/데이터셋 ID와 HF URL을 모두 처리한다.
- [ ] 실제 모델 tokenizer와 template을 내려받고 전체 dataset을 분석한다.
- [ ] 전체 coverage와 데이터 보존을 별도로 검사한다.
- [ ] SFT/DPO 길이는 데이터에서 산출하고 GRPO budget은 별도다.
- [ ] r/alpha/모듈 범위/dtype/optimizer를 서로 다른 영향으로 계산한다.
- [ ] GPU 없이 분석·재계산·export가 동작한다.
- [ ] Unsupported architecture/kernel과 unknown memory를 정직하게 표시한다.
- [ ] 단계별 최대값과 단일 피크 시점 breakdown이 일치한다.
- [ ] 결과에는 resolved config, source revisions, profile version이 포함된다.
- [ ] 취소·실패·부분 분석·다른 사용자 접근에 대한 테스트가 통과한다.
- [ ] 개발 mock 결과가 production build에 표시되지 않는다.
- [ ] README에 기동, local path 등록, HF 인증, 데이터 보존, 지원 범위가 있다.

## 21. 예시 입력의 최종 동작

```text
Model: XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
Method: GRPO
Load in 4-bit: ON
Dataset: CyberNative/Code_Vulnerability_Security_DPO
Advanced: 변경 없음
```

**모델 단계:** config와 tensor inventory를 확인하고 Qwen3.5 hybrid 구조로 분류한다. 자체 tokenizer/template을 사용하며 actual loading scope와 적용 가능한 QLoRA profile을 표시한다.

**데이터 단계:** system/question/chosen/rejected mapping을 제안한다. GRPO에서는 prompt-only 변환이 적용된다는 사실을 표시하고 선택한 train snapshot의 모든 prompt를 토큰화한다. 원본 preference 응답이 reward로 자동 변환되었다고 주장하지 않는다.

**조건 해석:** completion budget이 없으므로 1K/2K/4K/8K 출력 예산별 시나리오를 만든다. Reward가 미지정이면 조건부 결과로 남긴다. Reference beta, generation batch, update microbatch 등은 선택한 preset과 일치하는지 검증한다.

**메모리 단계:** 지원 profile에 한해 policy/adapter/optimizer/activation/rollout/cache를 phase별로 계산한다. Reward footprint나 kernel workspace가 미확정이면 전체 시스템의 확정 권장 용량 대신 해당 범위를 명시한다.

**화면 단계:** 분석한 실제 최대 prompt 길이, 전체 coverage, 생성 예산별 결과, 적용 profile, 조건부 경고를 보여준다. Hardware 미선택 상태에서는 가용 VRAM 대비 적합 판정을 하지 않는다.

**DPO로 변경:** 같은 source snapshot을 재사용하고 preference preprocessing을 수행한다. Prompt+chosen과 prompt+rejected 전체 길이를 계산하며 reference 전략에 따른 피크를 비교한다. 선택한 DPO 조건이 모두 해석되면 전체 분석 결과에 기반한 계획용 권장 용량을 제공한다.

이 계획서에는 예시 모델/데이터셋의 실제 최대 token 수나 VRAM 실측치를 넣지 않는다. 해당 수치는 구현된 분석기가 snapshot을 전체 처리한 후 생성해야 한다.

## 22. 주요 위험과 대응

| 위험 | 대응 |
|---|---|
| 구조 미지원인데 그럴듯한 숫자 출력 | inventory-only/unsupported 등급, 필요 구성요소 null |
| tokenizer는 같지만 Trainer 전처리 불일치 | objective별 golden token IDs·mask·collator parity |
| 긴 outlier 누락 | 전체 EOF 검사, 최대값과 실패 row 기록 |
| 옵션 fallback으로 실제 메모리 증가 | requested/resolved/observed 분리 및 profile 재선택 |
| GRPO 생성·보상 길이의 불확실성 | budget별 조건부 시나리오, reward footprint 별도 |
| 모델 로딩 단계에서 OOM | load/quantize transient를 training peak와 함께 계산 |
| 다중 GPU 합산 오류 | rank별 device map과 미지원 topology 차단 |
| metadata만으로 지나친 정밀도 주장 | 근거 등급과 가정 공개, 실측 calibration 분리 |
| 공개 서비스가 사용자 로컬 경로를 읽는다고 오해 | self-hosted 경로와 업로드/companion을 별도 UI로 안내 |
| 웹 계산기가 학습 플랫폼으로 비대해짐 | 단일 계산기 M0–M4를 고정 범위로 유지 |

## 23. 참고 자료와 검증 범위

다음은 2026-10-03에 확인한 서비스 페이지와 공식 문서다. UI와 제품 사양은 본 문서의 설계 제안이며, 외부 사실은 본문에서 해당 번호로 연결했다. 변경 가능한 문서의 내용을 runtime 동작으로 가정하지 말고 구현 시 dependency lock과 대조한다.

**[R01] ApX VRAM Calculator — 계산기형 UI 참고**  
`https://apxml.com/tools/vram-calculator`

**[R02] XiaomiMiMo 예시 모델 config — hybrid 구조와 모델 설정**  
`https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B/raw/main/config.json`

**[R03] XiaomiMiMo 예시 모델 카드 — tokenizer와 chat template 안내**  
`https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`

**[R04] CyberNative 예시 데이터셋 — 컬럼과 데이터 구조**  
`https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO`

**[R05] Transformers bitsandbytes — 4/8-bit 로딩과 학습 범위**  
`https://huggingface.co/docs/transformers/en/quantization/bitsandbytes`

**[R06] PEFT LoRA — rank, scaling, target modules, 추가 학습 모듈**  
`https://huggingface.co/docs/peft/v0.21.0/package_reference/lora`

**[R07] Transformers chat templates — 실제 입력 시퀀스 구성**  
`https://huggingface.co/docs/transformers/en/chat_templating`

**[R08] TRL SFTTrainer — 데이터 형식, masking, loss 경로**  
`https://huggingface.co/docs/trl/en/sft_trainer`

**[R09] TRL DPOTrainer — reference, precompute, 지원 제약**  
`https://huggingface.co/docs/trl/en/dpo_trainer`

**[R10] TRL GRPOTrainer — rollout, generation batch, reward 설정**  
`https://huggingface.co/docs/trl/en/grpo_trainer`

**[R11] TRL Reducing Memory Usage — 최적화와 호환 제약**  
`https://huggingface.co/docs/trl/en/reducing_memory_usage`

**[R12] Hugging Face Hub metadata API — safetensors header 분석**  
`https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#get_safetensors_metadata`

**[R13] Accelerate model memory estimator — meta-device 기반 구조 확인**  
`https://huggingface.co/docs/accelerate/en/usage_guides/model_size_estimator`

**[R14] Datasets streaming — 데이터 순차 분석 기반**  
`https://huggingface.co/docs/datasets/en/stream`

**[R15] PyTorch CUDA semantics — allocated와 reserved 구분**  
`https://docs.pytorch.org/docs/2.14/notes/cuda.html#memory-management`

**[R16] PyTorch CUDA memory profiling — 외부 CUDA 할당 관측 한계**  
`https://docs.pytorch.org/docs/2.14/torch_cuda_memory.html`

**[R17] Accelerate FSDP/DeepSpeed — sharding 구성의 차이**  
`https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed`

**[R18] vLLM engine arguments — cache pool 및 메모리 설정**  
`https://docs.vllm.ai/en/latest/configuration/engine_args/`

**[R19] MDN File System API — 브라우저의 사용자 파일 접근**  
`https://developer.mozilla.org/en-US/docs/Web/API/File_System_API`

---

## 최종 구현 원칙

**입력은 ApX처럼 간단하게, 길이는 실제 데이터로, 계산은 실제 실행 경로로, 결과는 가정과 근거까지 보여준다.**

화면에 수치를 보여주기 위해 데이터를 자르거나 미지원 옵션을 지원한다고 가정하지 않는다. 전체 분석과 추정·실측을 구분하면서, 사용자가 실제 파인튜닝 자원을 판단할 수 있는 웹 계산기를 구현한다.
