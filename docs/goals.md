# 프로젝트 목표와 완료 기준 (Goals & Definition of Done)

> 이 문서는 VRAMForge 개발을 **언제 끝났다고 말할 수 있는지**를 정의한다.
> 사양의 원문은 [`plan.md`](../plan.md), 작업 규칙은 [`git-conventions.md`](git-conventions.md)다.
> 각 항목의 상태는 실제 검증 근거(명령·테스트 이름·커밋)가 있을 때만 `완료`로 바꾼다.

| 항목 | 값 |
|---|---|
| 기준일 | 2026-10-04 |
| 납품 범위 | plan.md §20의 **M0–M4** (첫 사용 가능 버전) |
| 부분 범위 | M5는 인터페이스·상태 표시만 (`GPU 검증 미연결`). 실측 기능은 구현했다고 주장하지 않는다 |
| 범위 밖 | M6 (분산, vLLM, MoE 상세, 멀티모달 입력, MLX/MPS, local companion) |
| 개발 환경 | macOS arm64 + Docker (CPU only). CUDA GPU 없음 |

## 1. 최상위 목표 (North Star)

**G0. 어떤 Docker 호스트에서든 `docker compose up -d --build` 한 번으로 계산기 웹앱이 배포되고, 예시 입력으로 실제 전체 데이터 분석과 VRAM 산정을 끝까지 수행할 수 있다.**

- 대상 호스트: Linux amd64/arm64, macOS Docker Desktop, Windows Docker Desktop(WSL2).
- GPU, `.env` 파일, 사전 설치 도구 없이 동작한다 (Docker Engine + Compose v2만 필요).
- 기본 접속 주소: `http://localhost:8080` (기본 바인드는 localhost, plan.md §18).
- 예시 입력 (plan.md §21)

  ```text
  Model:   XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
  Method:  GRPO, Load in 4-bit: ON
  Dataset: CyberNative/Code_Vulnerability_Security_DPO
  ```

## 2. 마일스톤별 목표와 완료 조건

### M0 — 계약·골격

| ID | 목표 | 완료 조건 (검증 방법) | 상태 |
|---|---|---|---|
| M0-1 | monorepo 골격 | `apps/web`, `services/{api,worker_cpu,worker_gpu}`, `packages/estimator`, `profiles/`, `tests/`, `docs/` 존재. uv workspace + pnpm lock 커밋 | 대기 |
| M0-2 | 계약(schema) 확정 | 요청·결과·오류·SSE·ledger Pydantic 모델과 JSON Schema/OpenAPI 생성. TS 타입이 OpenAPI에서 생성됨 | 대기 |
| M0-3 | 상태 모델 확정 | 작업 상태 머신(§16.1)과 결과 상태 5축(§12.1)이 코드 enum으로 존재하고 테스트됨 | 대기 |
| M0-4 | 컨테이너 기동 | `docker compose up -d --build` 후 web/api/worker/postgres/redis/proxy 모두 healthy | 대기 |
| M0-5 | CI | GitHub Actions에서 Python lint/test, web lint/typecheck/test, compose 빌드 검사 | 대기 |

### M1 — 계산기 UI

| ID | 목표 | 완료 조건 | 상태 |
|---|---|---|---|
| M1-1 | 입력 영역 | Model/Dataset/Method/4-bit/Hardware + Advanced 5그룹 (plan §3.3, §5) | 대기 |
| M1-2 | 결과 요약 카드 | sticky 요약, 분석 전 `— / 분석 필요`, 피크·권장·적합 구분, 상태 배지 5축 (§3.4, §12.1) | 대기 |
| M1-3 | 진행 상태와 상세 탭 | 단계별 진행 + 5개 탭 (메모리 구성, 데이터 길이, 단계별 피크, 비교, 적용 설정·근거) | 대기 |
| M1-4 | 디자인·접근성 | 라이트/다크, 1,024px 기준 2열/1열, 키보드 조작, 차트 대체 표, 가로 스크롤 없음 | 대기 |
| M1-5 | UI 테스트 | 모든 상태(초기·진행·부분·완료·실패·입력 필요·조건부)에 대한 Vitest 테스트. production 빌드에 mock 수치 없음 | 대기 |

### M2 — 실제 source·토큰화

| ID | 목표 | 완료 조건 | 상태 |
|---|---|---|---|
| M2-1 | Source resolver | HF ID/URL/`tree/{rev}`/dataset viewer URL 정규화, revision → commit 고정, 로컬 root·업로드 처리 | 대기 |
| M2-2 | Model inspector | safetensors header 기반 tensor inventory, tokenizer/template 확인, 로딩 범위 해석, remote code 거부 | 대기 |
| M2-3 | Dataset inspector | config/split/컬럼 탐지, mapping 제안, 모호하면 `NEEDS_INPUT` | 대기 |
| M2-4 | 전체 스캔 | 실제 tokenizer + chat template으로 전체 row 토큰화, row-length Parquet artifact, 정확한 count/max | 대기 |
| M2-5 | 무절단 계약 | §7.4 조건 검사, context 초과 보고, partial/failed 구분, §19.2 회귀 테스트 통과 | 대기 |
| M2-6 | Trainer parity | 고정한 TRL 버전의 SFT/DPO/GRPO 전처리와 token IDs·mask 일치 (tiny fixture) | 대기 |

### M3 — 계산 엔진

| ID | 목표 | 완료 조건 | 상태 |
|---|---|---|---|
| M3-1 | 가중치·양자화 | dense/NF4(+double quant) 저장량, 비양자화 모듈 분리, load transient | 대기 |
| M3-2 | 학습 상태 | LoRA 파라미터(실제 모듈 차원), gradient, optimizer state, master copy 분리 | 대기 |
| M3-3 | Activation ledger | Dense + Qwen3.5 hybrid AllocationSpec ledger, checkpointing, logits/loss 버퍼 | 대기 |
| M3-4 | Trainer schedule | SFT/DPO/GRPO phase schedule, 피크 = phase별 동시 상주량의 최대값 | 대기 |
| M3-5 | 범위·여유·적합 | floor/low/high, margin 정책, 적합 판정 6가지, unknown → `null + 사유` | 대기 |
| M3-6 | 문서·테스트 | §19.1, §19.3 테스트 통과, `methodology.md`·`support-matrix.md`와 registry 자동 대조 테스트 | 대기 |

### M4 — 통합 출시

| ID | 목표 | 완료 조건 | 상태 |
|---|---|---|---|
| M4-1 | 작업 파이프라인 | 상태 머신, SSE 진행, 취소, lease/멱등 키, 재연결 복구 | 대기 |
| M4-2 | 재계산 | r/alpha/optimizer/batch/budget/hardware 변경 시 토큰화 재사용, 재분석 필요 변경은 버튼으로 안내 | 대기 |
| M4-3 | 내보내기 | `analysis.json`, `resolved-plan.yaml`, `report.md`, ready일 때만 `trainer-config.yaml` | 대기 |
| M4-4 | 보안 | owner 검증, SSRF·경로 탈출·업로드 제한, remote code 금지, 비밀정보 redaction, XSS 방지 | 대기 |
| M4-5 | 운영 문서 | README(기동, local path, HF 인증, 데이터 보존, 지원 범위), `deployment.md`, `privacy-and-retention.md` | 대기 |
| M4-6 | E2E | compose 스택 위에서 예시 입력 GRPO → 결과, DPO 전환, r 변경 재계산, export 일치 (Playwright) | 대기 |

### M5 — GPU 검증 (부분 범위)

| ID | 목표 | 완료 조건 | 상태 |
|---|---|---|---|
| M5-0 | 정직한 미연결 표시 | `/profile` API가 `GPU_WORKER_UNAVAILABLE`을 반환하고 UI에 `GPU 검증 미연결` 표시. compose `gpu` profile은 문서화된 자리만 제공 | 대기 |

## 3. 품질 게이트

| ID | 게이트 | 기준 |
|---|---|---|
| Q1 | 테스트 | `pytest`(unit·integration·security), `vitest`, Playwright E2E 전부 통과 |
| Q2 | 정직성 | 가짜 수치·미지원 기능의 작동 표시·`0`으로 채운 unknown이 없음 (리뷰로 확인) |
| Q3 | 사양 준수 | plan.md 섹션별 적대적 리뷰에서 확인된 결함 0건 (또는 문서화된 한계) |
| Q4 | 보안 | 보안 리뷰의 High/Critical 0건 |
| Q5 | 이식성 | linux/arm64 로컬 기동 + linux/amd64 이미지 빌드 확인 |
| Q6 | Git 규칙 | 모든 커밋이 `git-conventions.md` 형식, 다중 영역 혼합 커밋 없음 |

## 4. 출시 체크리스트 (plan.md §20.2)

| 항목 | 근거 | 상태 |
|---|---|---|
| 입력 예제의 모델/데이터셋 ID와 HF URL을 모두 처리한다 | | 대기 |
| 실제 모델 tokenizer와 template을 내려받고 전체 dataset을 분석한다 | | 대기 |
| 전체 coverage와 데이터 보존을 별도로 검사한다 | | 대기 |
| SFT/DPO 길이는 데이터에서 산출하고 GRPO budget은 별도다 | | 대기 |
| r/alpha/모듈 범위/dtype/optimizer를 서로 다른 영향으로 계산한다 | | 대기 |
| GPU 없이 분석·재계산·export가 동작한다 | | 대기 |
| Unsupported architecture/kernel과 unknown memory를 정직하게 표시한다 | | 대기 |
| 단계별 최대값과 단일 피크 시점 breakdown이 일치한다 | | 대기 |
| 결과에는 resolved config, source revisions, profile version이 포함된다 | | 대기 |
| 취소·실패·부분 분석·다른 사용자 접근에 대한 테스트가 통과한다 | | 대기 |
| 개발 mock 결과가 production build에 표시되지 않는다 | | 대기 |
| README에 기동, local path 등록, HF 인증, 데이터 보존, 지원 범위가 있다 | | 대기 |

## 5. 진행 방식

1. **문서화** — plan.md, Git 컨벤션, 이 목표 문서 (완료 후 push).
2. **조사(Research)** — 고정할 라이브러리 버전(TRL·Transformers·PEFT·bitsandbytes)의 실제 소스에서 전처리·기본값·메모리 경로를 확인해 `docs/research/`에 근거와 함께 기록.
3. **계약·골격(M0)** — schema, 디렉터리, compose를 먼저 고정해 병렬 작업의 경계를 만든다.
4. **병렬 구현(M1–M3)** — 디렉터리 소유권으로 나눈 서브 에이전트들이 커밋 규칙에 따라 구현.
5. **통합(M4)** — compose 스택 위에서 예시 입력 E2E.
6. **적대적 검증** — 사양 준수·정확성·보안·UI 리뷰 → 수정 반복.
7. **마무리** — 이 문서의 상태와 근거를 갱신하고 push.

마일스톤이 끝날 때마다 이 문서의 상태 열을 갱신하고, 검증이 끝난 커밋을 `main`에 push한다.
