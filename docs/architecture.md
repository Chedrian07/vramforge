# 아키텍처와 모듈 계약

> 구현자(사람·서브 에이전트)가 **서로 독립적으로 작업해도 맞물리도록** 경계와 계약을 고정하는 문서다.
> 사양은 [`plan.md`](../plan.md), 목표는 [`goals.md`](goals.md), Git 규칙은 [`git-conventions.md`](git-conventions.md).
> 라이브러리 동작의 근거는 [`docs/research/`](research/)에 있다. 이 문서와 코드가 다르면 **코드의 schema가 기준**이고, 이 문서를 고친다.

## 1. 런타임 구성

```text
Browser ──HTTP──▶ proxy (Caddy :80, host ${VRAMFORGE_BIND:-127.0.0.1}:${VRAMFORGE_PORT:-8080})
                    ├─ /api/*  ─▶ api    (FastAPI, uvicorn :8000)  ── SSE는 버퍼링 없이 flush
                    └─ /*      ─▶ web    (Next.js standalone :3000)
api ──▶ postgres (작업 상태·결과·이벤트의 기준 저장소)
api ──▶ redis    (RQ 큐; 결과 저장소 아님)
worker (RQ, CPU only) ──▶ postgres, redis, /data (artifact·업로드·HF cache), /sources/local (read-only)
migrate (one-shot) ── alembic upgrade head 후 종료
```

| 서비스 | 이미지 | 비고 |
|---|---|---|
| `proxy` | `caddy` (alpine) | same-origin 단일 진입점, `/api` SSE flush |
| `web` | `infra/docker/web.Dockerfile` | Next.js `output: 'standalone'` |
| `api` | `infra/docker/python.Dockerfile` | `vramforge-api` |
| `worker` | 같은 Python 이미지, 다른 command | `vramforge-worker` |
| `migrate` | 같은 Python 이미지 | `alembic upgrade head` |
| `postgres`, `redis` | 공식 alpine 이미지 | named volume |

- `.env` 없이 `docker compose up -d --build`만으로 기동한다. 모든 값은 `${VAR:-default}` 기본값을 갖는다.
- GPU worker는 기본 구성에 없다(M5 범위 밖). API는 `GPU_WORKER_UNAVAILABLE`, UI는 `GPU 검증 미연결`을 표시한다.
- 컨테이너 내부 경로: `/data/artifacts`, `/data/uploads`, `/data/hf`(HF_HOME), `/sources/local`(host `./local-sources`, ro).

## 2. 저장소 구조와 Python 패키지

```text
pyproject.toml                 # uv workspace root (dev 의존성: pytest, ruff, mypy ...)
packages/estimator/            # vramforge-estimator  : 계산 코어 (프레임워크 무관, I/O 최소)
  src/vramforge_estimator/
    schemas/                   # ★ 공용 계약 (Pydantic). 모든 모듈·API·웹 타입의 원천
    sources/                   # SourceResolver: 참조 정규화, revision 고정, 로컬 root, SSRF 방어
    inspection/                # ModelInspector, TokenizerInspector, DatasetInspector
    preprocessing/             # objective별 PreprocessingAdapter (TRL 1.14.1 전처리 재현)
    scan/                      # 전체 스캔 실행기, row-length artifact, 통계, 보존 감사
    batching/                  # BatchPlanner
    architectures/             # ArchitectureAdapter (dense_decoder, qwen3_5_hybrid)
    trainers/                  # TrainerAdapter (sft, dpo, grpo) → phase schedule
    memory/                    # MemoryEngine, 가중치·양자화·LoRA·optimizer 수식, margin, fit
    compatibility/             # profile registry 로더, CompatibilityResolver
    calibration/               # CalibrationRegistry (현재는 항상 miss)
    exports/                   # JSON / YAML / Markdown / trainer-config
    pipeline.py                # analyze(), recompute() — 단계 orchestration
services/api/                  # vramforge-api   : FastAPI, DB(SQLAlchemy/Alembic), 큐, SSE, 보안
services/worker_cpu/           # vramforge-worker: RQ worker 진입점, 작업 실행기
services/worker_gpu/           # 자리만 존재 (README) — M5
apps/web/                      # Next.js 앱
profiles/                      # analytic / environments / calibrated / hardware YAML
tests/                         # unit / preprocessing_parity / integration / security / fixtures
infra/                         # docker/*.Dockerfile, proxy/Caddyfile
```

의존 방향은 한 방향이다: `web → (HTTP) → api → estimator`, `worker → api(db·settings) + estimator`. **estimator는 api/worker를 import하지 않는다.**

| 패키지 | 필수 의존성 | 비고 |
|---|---|---|
| `vramforge-estimator` | pydantic, pyyaml | extra `analysis`: huggingface_hub(<2), transformers(tokenizer 전용, torch 없음), tokenizers, datasets, pyarrow, jinja2 |
| `vramforge-api` | fastapi, uvicorn, sqlalchemy, psycopg, alembic, redis, rq, sse-starlette, pydantic-settings, python-multipart | estimator[analysis] |
| `vramforge-worker` | rq | api, estimator[analysis] |
| dev group `parity` | trl==1.14.1, torch(CPU), peft, accelerate | Docker 이미지에 넣지 않는다. parity 테스트 전용 |

## 3. 분석 파이프라인

```text
POST /api/v1/analyses ─▶ DB insert (QUEUED) ─▶ RQ enqueue(job_id = analysis_id)
worker: lease 획득 ─▶ pipeline.analyze(request, ctx)
  RESOLVING        sources.resolve(model), sources.resolve(dataset)        → SourceManifest ×2
  INSPECTING       inspection.inspect_model / inspect_tokenizer / inspect_dataset
                   compatibility.resolve(request, inventory)              → ResolvedConfig + blockers
                   (mapping/config/split 모호 → NEEDS_INPUT 으로 종료)
  TOKENIZING       scan.full_scan(dataset, preprocessing adapter, ctx)    → DatasetScanResult + artifact
  VALIDATING_DATA  preservation audit, context validation
  PLANNING_BATCHES batching.plan(length artifact, resolved)               → BatchPlan
  ESTIMATING       trainers.build_schedule + architectures ledger
                   memory.evaluate(...) per scenario                      → MemoryEstimate
  COMPLETED        AnalysisResult 저장 (+ 이벤트 completed)
```

- `ctx` (`JobContext`)는 진행 보고, 취소 확인, artifact 디렉터리, 캐시, 자격 증명, 한도를 제공한다. 파이프라인은 DB·Redis를 직접 알지 못한다.
- 어떤 단계가 실패해도 그 단계까지 확인한 정보를 담은 **부분 결과**를 반환한다. 앞 단계 결과가 없으면 뒤 단계는 실행하지 않는다 (예: tokenizer 없음 → 길이를 지어내지 않음).
- `recompute(base_result, overrides, artifacts)`는 저장된 inventory와 row-length artifact로 batch 계획과 메모리만 다시 계산한다. 전처리 계층을 바꾸는 override는 `requires_reanalysis=true`로 거절한다 (plan §4.2, §16.3).

### 3.1 재분석이 필요한 변경 / 재계산으로 충분한 변경

| 재분석 필요 (preprocess_key 변경) | 재계산 (estimate/batch 계층) |
|---|---|
| objective, model/dataset reference·revision, config, split, mapping, empty-system 정책, template 옵션(thinking 등), packing | strategy, quantization, LoRA(r/alpha/dropout/targets/modules_to_save), optimizer, precision, microbatch, accumulation, checkpointing, attention/loss 경로, DPO reference·beta, GRPO budget·generation 수·generation batch·beta·reward, hardware, margin, scope |

## 4. 메모리 모델의 핵심 계약

- 모든 수치는 **정수 byte**. 화면만 GiB(1 GiB = 1,073,741,824 bytes).
- `AllocationSpec`은 이름·분류·shape 식·dtype·`bytes_low`/`bytes_high`·근거(evidence)·**살아 있는 timepoint 목록**을 가진다. 크기를 모르면 `None` + `note`(사유). 0으로 채우지 않는다.
- `TrainerAdapter`가 phase와 phase 안의 timepoint(예: `POLICY_FORWARD_BACKWARD:loss`)를 정의하고, 각 allocation이 어느 timepoint에 살아 있는지 정한다.
- `ArchitectureAdapter`는 모델 구조만 안다: 상주 가중치, 학습 가능 파라미터, 주어진 shape의 학습 forward/backward ledger, no-grad forward 작업 집합, generation cache.
- `MemoryEngine`은 timepoint마다 살아 있는 allocation을 합산하고 **최대 timepoint**를 피크로 고른다. 피크 breakdown은 그 timepoint의 allocation 목록이다. 서로 다른 timepoint의 최대값을 더하지 않는다 (plan §9.1–9.2, §12.2).
- `known_floor_bytes` = 확정된 상주분(가중치·adapter·gradient·optimizer state 등 크기가 확정된 resident 항목)의 최대 timepoint 합.
- `scenario_low/high_bytes` = 모든 allocation의 low/high 합의 최대 timepoint 값. 피크 경로에 크기 미상 항목이 있으면 둘 다 `None`.
- **제외(excluded)**와 **미상(unknown)**을 구분한다. reward 미지정·평가 미포함처럼 범위에서 뺀 항목은 `excluded_components`에 사유와 함께 기록하고 결과는 conditional이 된다. 범위 안인데 산정 불가한 항목은 `unknown_components`다.
- 계획용 여유: `planning_margin = max(2 GiB, high × 0.15)` (요청의 `margin_policy`로 조정 가능), `recommended = high + margin`, `required_total = recommended + external_reserved` (plan §10.2).

## 5. 상태 모델

작업 상태(`JobStatus`, plan §16.1)와 결과 상태 5축(`StatusAxes`, plan §12.1)은 독립이다. `COMPLETED`는 작업이 끝났다는 뜻일 뿐, 결과의 보존·준비·근거 상태는 축별로 따로 본다.

```text
QUEUED → RESOLVING → INSPECTING → (NEEDS_INPUT | TOKENIZING) → VALIDATING_DATA
       → PLANNING_BATCHES → ESTIMATING → COMPLETED
어느 단계에서든 → CANCEL_REQUESTED → CANCELLED,  → FAILED,  → PARTIAL
```

## 6. API 표면

| Method | Path | 응답 |
|---|---|---|
| GET | `/api/v1/health` | 서비스·DB·Redis·worker 상태 |
| POST | `/api/v1/sources/inspect` | `InspectResponse` (동기, 시간 제한) |
| POST | `/api/v1/uploads` | `UploadResponse` |
| GET | `/api/v1/backend-profiles` | 지원 조합·환경·하드웨어 preset |
| GET | `/api/v1/local-roots` | 허용된 로컬 root 목록 |
| POST | `/api/v1/analyses` | 202 `AnalysisCreated` (멱등 키 `Idempotency-Key`) |
| GET | `/api/v1/analyses/{id}` | `AnalysisStatus` (진행·부분/최종 결과) |
| GET | `/api/v1/analyses/{id}/events` | SSE (`Last-Event-ID` 재개) |
| POST | `/api/v1/analyses/{id}/cancel` | `AnalysisStatus` |
| POST | `/api/v1/analyses/{id}/scenarios` | `ScenarioResponse` |
| GET | `/api/v1/analyses/{id}/export?format=json|yaml|md|trainer-config` | 파일 |
| DELETE | `/api/v1/analyses/{id}` | 204 |
| POST | `/api/v1/analyses/{id}/profile` | 503 `GPU_WORKER_UNAVAILABLE` |

- 오류 응답은 항상 `ErrorResponse { error: Issue }` 형식이다. stack trace·token·절대경로를 넣지 않는다.
- 소유권: API가 발급하는 `vf_owner` httpOnly cookie(SameSite=Strict)로 익명 소유자를 식별한다. 상태 변경 요청은 `X-VramForge-Request: 1` 헤더를 요구한다(CSRF). `VRAMFORGE_ACCESS_TOKEN`을 설정하면 모든 API가 토큰을 요구한다.
- 웹의 TypeScript 타입은 FastAPI OpenAPI에서 생성한다(`apps/web/lib/api/schema.d.ts`). 프런트엔드는 메모리 수식을 복제하지 않는다.

## 7. 저장소(DB)와 artifact

| 테이블 | 내용 |
|---|---|
| `owners` | 익명 소유자 |
| `analyses` | 요청, fingerprint, 멱등 키, 상태, 진행, 결과(JSON), 오류, lease, 시도 횟수, 취소 플래그 |
| `analysis_events` | 단조 증가 id의 SSE 이벤트 (재연결 재생용) |
| `uploads` | 업로드 메타데이터와 만료 |
| `scan_cache` | `preprocess_key` → artifact 경로·스캔 요약 (owner 범위) |

artifact: `/data/artifacts/<owner>/<analysis_id>/` 아래 `lengths/*.parquet`(row 길이 레코드, plan §7.6), `model_inventory.json`, `checkpoint.json`. 원문과 전체 token id는 저장하지 않는다.

## 8. 작업 소유권 (병렬 구현 단계)

| 소유 경로 | 담당 |
|---|---|
| `packages/estimator/src/vramforge_estimator/schemas/`, `pipeline.py`의 시그니처, 루트 설정 | 오케스트레이터 |
| `sources/`, `inspection/` + `tests/unit/{sources,inspection}` | ingest 에이전트 |
| `preprocessing/`, `scan/`, `batching/` + `tests/unit/{preprocessing,scan,batching}`, `tests/preprocessing_parity/` | scan 에이전트 |
| `architectures/` + `tests/unit/architectures` | architecture 에이전트 |
| `trainers/`, `memory/`, `compatibility/`, `calibration/`, `profiles/` + 관련 테스트, `docs/methodology.md`, `docs/support-matrix.md` | memory 에이전트 |
| `services/`, `exports/`, `pipeline.py` 본문, `tests/{integration,security}`, `tests/unit/{api,worker,exports,pipeline}` | api 에이전트 |
| `apps/web/` | web 에이전트 |
| `infra/`, `compose.yaml`, `.github/`, `docs/deployment.md` | infra 에이전트 |

공용 schema 변경이 필요하면 소유 에이전트는 직접 고치지 않고 `CHANGE REQUEST`로 보고한다 (git-conventions §6.1).
