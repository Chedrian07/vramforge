# VRAMForge — Fine-Tuning VRAM Calculator

모델·데이터셋·파인튜닝 방식(SFT / DPO / GRPO)을 입력하면 **실제 토크나이저와 TRL 학습 전처리로 데이터셋 전체 길이를 분석**하고, **데이터를 자르지 않는 조건에서 GPU별 피크 VRAM**을 산정하는 self-hosted 웹 계산기입니다.

- 데이터는 자르지 않습니다. 전체 row를 실제 chat template으로 토큰화하고, 무절단 조건 8가지를 검사합니다.
- 메모리는 모델 구조와 trainer 실행 경로로 계산합니다. 가중치(4-bit 포함), LoRA, gradient, optimizer, activation, logits/loss, KV·recurrent cache를 **같은 시점에 살아 있는 것끼리만** 더해 피크를 구합니다.
- 모르는 값은 0으로 채우지 않습니다. `null`과 사유를 반환하고, 정적 추정 / 보정 / 실측 근거 등급을 구분해 표시합니다.
- 분석 서버는 CPU만 씁니다. 모델 가중치를 내려받거나 실행하지 않습니다(config·tokenizer·safetensors header만 사용).

사양: [`plan.md`](plan.md) · 목표와 완료 기준: [`docs/goals.md`](docs/goals.md)

## 빠른 시작

요구 사항: Docker Engine 23+(BuildKit 기본) 와 Docker Compose v2 (Docker Desktop, OrbStack, Linux 모두 가능). GPU와 `.env` 파일은 필요 없습니다.

```bash
git clone https://github.com/Chedrian07/vramforge.git
cd vramforge
docker compose up -d --build        # 처음에는 이미지 빌드로 몇 분 걸립니다
# → http://localhost:8080
```

화면에서 **예시 입력 불러오기**를 누르면 아래 예시가 채워집니다. **전체 데이터 분석 및 계산**을 눌러 실행합니다.

```text
Model:   XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
Method:  GRPO, Load in 4-bit: ON
Dataset: CyberNative/Code_Vulnerability_Security_DPO
```

기동 상태 확인과 종료:

```bash
sh infra/scripts/smoke-test.sh      # proxy·API·DB·Redis·worker·web 점검
docker compose down                 # 컨테이너 종료 (분석 결과·캐시 볼륨 유지)
docker compose down -v              # 이 프로젝트의 볼륨까지 삭제
```

기본 바인드는 `127.0.0.1:8080`입니다. 다른 컴퓨터에서 접속하게 하려면 HTTPS와 접근 토큰을 함께 설정해야 합니다. 자세한 내용은 [`docs/deployment.md`](docs/deployment.md) §3을 보세요.

## 할 수 있는 것과 범위

| 항목 | 내용 |
|---|---|
| 학습 방식 | SFT, DPO, GRPO (TRL 1.14.1 동작 기준) |
| 전략 | Full, LoRA, QLoRA (bitsandbytes NF4 + double quant) |
| 구조 | Dense decoder (Llama·Qwen2/3·Mistral 계열), Qwen3.5 hybrid (Gated DeltaNet + full attention) |
| 데이터 형식 | text, prompt-completion, messages, preference (명시적·암묵적 prompt), prompt-only |
| 입력 | HF 모델·데이터셋 ID/URL(`tree/<rev>`, dataset viewer URL 포함), 데이터 파일 업로드, 서버 로컬 경로 |
| 결과 | 피크 범위(low–high)와 계획용 권장 용량, 학습 RAM, 단계별 피크, 데이터 길이 통계, 재계산 비교, 근거·가정 |
| 내보내기 | `analysis.json`, `resolved-plan.yaml`, `report.md`, `trainer-config.yaml`(실행 준비가 된 경우만) |

지원 등급의 전체 표는 [`docs/support-matrix.md`](docs/support-matrix.md)에 있습니다. 이번 릴리스에 **포함하지 않는** 것:

- GPU 실측 검증(M5). 모든 수치는 `analytic` 등급의 정적 추정입니다. UI에는 `GPU 검증 미연결`로 표시됩니다.
- 다중 GPU(DDP·FSDP·ZeRO·TP·PP), vLLM rollout, MoE 상세 routing, 이미지·영상 입력, MLX/MPS.
- 엄격 무절단 모드에서의 packing, offload, `torch.compile`. 요청하면 미지원 사유를 반환합니다.

## 결과를 읽는 법

결과는 하나의 "성공" 배지로 뭉치지 않고 다섯 축으로 따로 표시합니다 (plan.md §12.1).

| 축 | 의미 |
|---|---|
| 스캔 범위 | 선택한 split의 모든 row를 읽고 토큰화했는가 (`complete` / `partial` / `failed`) |
| 데이터 보존 | 절단·삭제·분할·template 손실이 없는가 (`verified` / `violated` / `unknown`) |
| 학습 준비 | 그대로 실행 가능한 설정인가 (`ready` / `conditional` / `unsupported`). 예: GRPO reward 미지정은 `conditional` |
| 근거 등급 | `metadata_only` / `analytic` / `calibrated` / `measured` |
| GPU 적합 | 하드웨어를 고른 경우에만 판정. 중요한 footprint가 미확정이면 `판정 보류` |

계획용 권장 용량은 `scenario_high + max(2 GiB, scenario_high × 15%)`입니다. 이 값은 오차 보증이 아니라 운영 여유 정책이며, 요청에서 조정할 수 있습니다. 계산 방법과 가정은 [`docs/methodology.md`](docs/methodology.md)와 [`docs/methodology-architectures.md`](docs/methodology-architectures.md)에 있습니다.

## Hugging Face 인증

공개 모델·데이터셋은 토큰 없이 분석합니다. 비공개·gated 저장소를 분석하려면 서버에 토큰을 주고, 서버 토큰을 모든 사용자에게 쓰도록 **명시적으로** 허용해야 합니다 (plan.md §18).

```bash
HF_TOKEN=hf_xxx VRAMFORGE_SHARE_SERVER_HF_TOKEN=true docker compose up -d
```

서버 토큰은 접속한 모든 사용자의 요청에 쓰이므로 단일 사용자 배포에서만 허용하세요. 토큰은 결과·로그·내보내기에 기록되지 않습니다. 자세한 내용은 [`docs/deployment.md`](docs/deployment.md) §5를 보세요.

## 로컬 모델·데이터셋

`./local-sources` 디렉터리가 컨테이너에 읽기 전용(`/sources/local`)으로 마운트됩니다. 파일을 그 안에 두고 `local:local/<상대 경로>`로 참조합니다. 다른 경로를 쓰려면 `VRAMFORGE_LOCAL_SOURCES_DIR`을 설정합니다. 경로 탈출과 root 밖을 가리키는 symlink는 거부합니다. 자세한 내용은 [`local-sources/README.md`](local-sources/README.md)와 [`docs/deployment.md`](docs/deployment.md) §6을 보세요.

## 데이터 보존과 개인정보

- 데이터 원문과 token id는 저장하지 않습니다. row별 길이 기록(Parquet)과 결과만 저장합니다.
- 결과와 업로드는 소유자(브라우저 cookie)에게만 보이며, 기본 7일 뒤 삭제됩니다(`VRAMFORGE_RETENTION_DAYS`).
- 내보내기에는 원문, 토큰, 절대 경로, 사설 URL이 들어가지 않습니다.

자세한 내용은 [`docs/privacy-and-retention.md`](docs/privacy-and-retention.md)에 있습니다.

## 구성

```text
Browser → proxy (Caddy :8080) ─┬─ /api/* → api (FastAPI) → postgres · redis
                               └─ /*     → web (Next.js)
                                  worker (RQ, CPU) : 전체 스캔·배치 계획·메모리 산정
```

| 경로 | 내용 |
|---|---|
| `packages/estimator` | 계산 코어: 계약(schema), source·inspection, 전처리·스캔·배치, 구조·trainer adapter, 메모리 엔진 |
| `services/api`, `services/worker_cpu` | HTTP API, 작업 큐 worker, DB, SSE, 내보내기 |
| `apps/web` | 계산기 UI |
| `profiles/` | backend profile, 학습 환경, GPU preset |
| `infra/`, `compose.yaml` | Dockerfile, Caddyfile, 스모크 테스트 |
| `docs/research/` | 고정한 라이브러리 버전의 실제 동작을 소스와 실험으로 확인한 조사 기록 |

모듈 경계와 계약은 [`docs/architecture.md`](docs/architecture.md)에 있습니다.

## 개발

```bash
uv sync --all-groups                         # Python (dev + parity 그룹)
uv run --no-sync ruff check . && uv run --no-sync pytest -q
VRAMFORGE_NETWORK_TESTS=1 uv run --no-sync pytest -q -m network   # 실제 HF Hub 사용

cd apps/web && pnpm install
pnpm lint && pnpm typecheck && pnpm test && pnpm build
E2E_BASE_URL=http://localhost:8080 pnpm test:e2e                    # compose 스택 대상
```

- `parity` 테스트는 실제 TRL 1.14.1 trainer와 토큰 길이·mask·collator shape를 대조합니다.
- CI(`.github/workflows/ci.yml`)는 Python, web, 그리고 linux/amd64·linux/arm64 runner에서 compose 기동과 GPU 없는 전체 분석을 검사합니다.
- 커밋 규칙은 [`docs/git-conventions.md`](docs/git-conventions.md), 기여 안내는 [`CONTRIBUTING.md`](CONTRIBUTING.md)에 있습니다.

## 한계

- 수치는 CPU에서 검증한 식과 라이브러리 소스에 근거한 정적 추정입니다. CUDA 전용 커널 경로, allocator 단편화, CUDA context 크기 같은 항목은 가정 범위(low–high)로 표시하며, GPU로 보정하지 않았습니다.
- 예시 데이터셋의 길이 기준값(행 4,656개, GRPO 최대 프롬프트 268, DPO 최대 2,272 토큰)은 고정 revision `81aeacf`에서 측정한 값입니다.
- GRPO 생성 길이는 데이터에서 알 수 없으므로 completion budget 시나리오로만 제공합니다.
