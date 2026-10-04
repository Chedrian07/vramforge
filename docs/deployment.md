# 배포 가이드 (Docker Compose)

> VRAMForge를 self-hosted로 띄우고 운영하는 방법입니다. 사양 근거는 [`plan.md`](../plan.md) §13·§18, 런타임 구성은 [`architecture.md`](architecture.md) §1, 완료 기준은 [`goals.md`](goals.md) G0·M0-4·Q5입니다.
> 이미지·버전·healthcheck 선택의 근거는 [`research/stack-compat.md`](research/stack-compat.md) §7·§10에 있습니다.

| 항목 | 값 |
|---|---|
| 기동 명령 | `docker compose up -d --build` |
| 접속 주소 | `http://localhost:8080` (기본은 이 컴퓨터에서만 접속 가능) |
| 필요한 것 | Docker Engine + Docker Compose v2. GPU, `.env`, Python·Node 설치는 필요 없음 |
| 대상 호스트 | Linux amd64/arm64, macOS Docker Desktop·OrbStack, Windows Docker Desktop(WSL2). 실제로 검증한 범위는 [§15](#15-검증-기록) |
| 포함하지 않는 것 | GPU 실측 검증(M5). [§13](#13-gpu-검증m5은-이번-릴리스에-없습니다) 참고 |

## 1. 빠른 시작

```bash
git clone https://github.com/Chedrian07/vramforge.git
cd vramforge
docker compose up -d --build
```

브라우저에서 `http://localhost:8080`을 엽니다. 처음에는 이미지를 받고 빌드하느라 몇 분 걸릴 수 있고, 다음부터는 캐시를 씁니다.

모든 서비스가 healthy가 될 때까지 기다리려면 `--wait`를 붙입니다. 기동 확인 스크립트는 프록시를 거쳐 API와 웹을 확인합니다.

```bash
docker compose up -d --build --wait
docker compose ps                      # 모든 서비스 healthy, migrate는 Exited (0)
sh infra/scripts/smoke-test.sh         # 프록시 → /api/v1/health(db·redis·worker 모두 ok), 웹 루트, 보안 헤더
```

분석 한 건을 끝까지 돌려 보려면 저장소의 작은 테스트용 모델·데이터셋(`tests/fixtures`)으로 오프라인 분석을 실행합니다. GPU와 Hugging Face 접속이 필요 없고, CI도 같은 검사를 합니다. 확인용 폴더를 잠시 `/sources/local`에 연결하므로, 끝나면 마지막 명령으로 원래 mount로 되돌립니다. 분석 기록은 스크립트가 지웁니다.

```bash
python3 infra/scripts/analysis_smoke.py prepare /tmp/vramforge-smoke     # 비어 있는 새 폴더
VRAMFORGE_LOCAL_SOURCES_DIR=/tmp/vramforge-smoke docker compose up -d --wait
python3 infra/scripts/analysis_smoke.py run   # DPO+LoRA 전체 스캔 → COMPLETED, coverage complete
docker compose up -d --wait                   # 원래 local-sources mount로 복귀
```

| 동작 | 명령 | 데이터 |
|---|---|---|
| 중지 | `docker compose stop` | 유지 |
| 내리기 | `docker compose down` | **유지** (볼륨은 남음) |
| 완전 초기화 | `docker compose down -v` | **삭제** (이 프로젝트의 볼륨 3개를 지움, 되돌릴 수 없음) |
| 로그 | `docker compose logs -f api worker` | — |

## 2. 요구 사항

| 항목 | 내용 |
|---|---|
| Docker | Docker Engine과 Compose v2 플러그인(`docker compose`). Docker Desktop, OrbStack, Linux의 Docker Engine 모두 해당 |
| 빌드 기능 | BuildKit(`RUN --mount`, Dockerfile별 ignore 파일). Docker Engine 23 이상과 Docker Desktop은 BuildKit이 기본입니다 |
| CPU 아키텍처 | linux/amd64, linux/arm64. 호스트 아키텍처용 이미지를 그 자리에서 빌드합니다 |
| 네트워크 | 빌드할 때 이미지 레지스트리, PyPI, npm에 접속합니다. 분석할 때 worker와 api가 Hugging Face Hub에 접속합니다(로컬 source만 쓰면 필요 없음) |
| GPU | 필요 없습니다. 분석과 산정은 모두 CPU에서 합니다 |
| `.env` | 필요 없습니다. 모든 설정에 기본값이 있습니다([§4](#4-환경-변수)) |

compose 파일이 쓰는 기능은 top-level `name`, `depends_on`의 `service_healthy`·`service_completed_successfully`, `up --wait`, YAML anchor, 값 없는 환경 변수 전달입니다. 검증한 환경은 [§15](#15-검증-기록)에 있습니다. 그보다 오래된 Docker·Compose 버전에서는 시험하지 않았습니다.

## 3. 포트와 바인딩

호스트에 포트를 여는 서비스는 `proxy` 하나뿐입니다. 기본값은 `127.0.0.1:8080`이라서 **이 컴퓨터에서만** 접속할 수 있습니다(plan.md §18). PostgreSQL, Redis, API, 웹 서버는 호스트에 publish하지 않습니다.

```text
브라우저 ──▶ proxy (Caddy, ${VRAMFORGE_BIND:-127.0.0.1}:${VRAMFORGE_PORT:-8080} → :80)
               ├─ /api/*  ─▶ api (FastAPI :8000, SSE 진행 상황)
               └─ 그 밖   ─▶ web (Next.js :3000)
api·worker·migrate ──▶ postgres, redis  (외부와 연결되지 않는 내부 네트워크)
```

포트만 바꿀 때는 `.env`에 `VRAMFORGE_PORT=9000`을 넣고 `docker compose up -d`를 다시 실행합니다.

### 3.1 다른 컴퓨터에서 접속하게 할 때

`VRAMFORGE_BIND=0.0.0.0`은 같은 네트워크의 누구나 분석을 실행하게 만듭니다(CPU·디스크·Hugging Face 트래픽 사용). 외부에 열 때는 아래를 **모두** 지킵니다.

1. **HTTPS를 앞에 둡니다.** proxy는 평문 HTTP만 제공합니다. 권장 방식은 `VRAMFORGE_BIND=127.0.0.1`을 그대로 두고, 같은 호스트의 TLS reverse proxy(호스트의 Caddy·nginx·Traefik, Cloudflare Tunnel, Tailscale Serve 등)가 `127.0.0.1:8080`으로 전달하게 하는 것입니다. 이 경우 `0.0.0.0` 바인드가 필요 없습니다.
2. **접근 토큰을 설정합니다.** `.env`에 `VRAMFORGE_ACCESS_TOKEN=<충분히 긴 임의 문자열>`을 넣으면 모든 API 요청에 토큰이 필요합니다. 예: `openssl rand -hex 32`.
3. **Secure cookie를 켭니다.** HTTPS 뒤에서는 `VRAMFORGE_COOKIE_SECURE=true`로 바꿉니다.
4. **방화벽을 확인합니다.** Linux에서 Docker가 publish한 포트는 iptables 규칙을 직접 추가하므로 `ufw` 같은 호스트 방화벽 규칙을 거치지 않을 수 있습니다. 바인드 주소로 노출 범위를 제한하세요.

앞단 reverse proxy는 경로를 나누지 말고 모든 요청을 `127.0.0.1:8080`으로 전달합니다. `/api`와 웹이 같은 origin이어야 cookie와 SSE가 동작합니다. 또 SSE(`/api/v1/analyses/{id}/events`)를 버퍼링하지 않게 하고(nginx: `proxy_buffering off;`, 충분히 긴 `proxy_read_timeout`), 업로드 크기 제한(nginx: `client_max_body_size`)을 [§4](#4-환경-변수)의 업로드 상한 이상으로 맞춥니다.

## 4. 환경 변수

`.env`는 선택입니다. 바꾸고 싶은 값만 [`.env.example`](../.env.example)에서 복사해 `.env`에 둡니다(`cp .env.example .env`). `.env.example`과 `compose.yaml`의 기본값이 같은지는 CI가 `infra/scripts/check_env_example.py`로 검사합니다. 값을 바꾼 뒤에는 `docker compose up -d`(이미지 관련 값이면 `--build`)로 다시 적용합니다.

### 4.1 compose 설정

| 변수 | 기본값 | 설명 |
|---|---|---|
| `VRAMFORGE_BIND` | `127.0.0.1` | proxy를 publish할 호스트 주소. 외부 공개 전 [§3.1](#31-다른-컴퓨터에서-접속하게-할-때) 확인 |
| `VRAMFORGE_PORT` | `8080` | proxy를 publish할 호스트 포트 |
| `VRAMFORGE_COOKIE_SECURE` | `false` | owner cookie에 `Secure`를 붙임. HTTPS 뒤에서는 `true` |
| `VRAMFORGE_LOCAL_SOURCES_DIR` | `./local-sources` | `/sources/local`에 읽기 전용으로 연결할 호스트 폴더. 상대 경로는 `compose.yaml`이 있는 폴더 기준 |
| `VRAMFORGE_LOCAL_ROOTS` | `local=/sources/local` | 허용할 local root(`이름=/컨테이너/절대경로`, 쉼표 구분) |
| `VRAMFORGE_PROXY_MAX_BODY_SIZE` | `2049MiB` | proxy의 요청 본문 상한. API 업로드 상한(2 GiB) + multipart 여유 1 MiB |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `vramforge` / `vramforge` / `vramforge` | 내부 DB 계정. 비밀번호는 **볼륨을 처음 만들 때만** 적용됩니다 |
| `PYTHON_IMAGE` | `python:3.12.15-slim-trixie` | api·worker·migrate 기반 이미지 |
| `UV_IMAGE` | `ghcr.io/astral-sh/uv:0.12.23` | Python 의존성 설치에 쓰는 uv |
| `NODE_IMAGE` | `node:24.21.0-trixie-slim` | 웹 빌드·실행 이미지 (Node 24 LTS) |
| `CADDY_IMAGE` | `caddy:2.11.6-alpine` | proxy 기반 이미지 |
| `POSTGRES_IMAGE` | `postgres:18.6-alpine` | DB 이미지 |
| `REDIS_IMAGE` | `redis:8.10.2-alpine` | 작업 큐 이미지 |

`POSTGRES_PASSWORD`는 DB 연결 URL에 그대로 들어가므로 영문·숫자 위주로 정합니다(`@ : / %` 등은 피함). DB는 내부 네트워크에만 있고 호스트에 publish하지 않습니다.

### 4.2 선택 항목 (설정할 때만 전달)

아래 변수는 compose에 값 없이 선언되어 있습니다. 셸이나 `.env`에 설정했을 때만 컨테이너로 전달되고, 설정하지 않으면 컨테이너에 아예 없습니다. 그래서 빈 토큰이 전달되지 않고, API 기본값이 그대로 쓰입니다. 비밀값은 쓰는 서비스에만 전달됩니다. `HF_TOKEN`은 api·worker, `VRAMFORGE_ACCESS_TOKEN`은 api에만 가고, `migrate`는 DB 접속 정보(`VRAMFORGE_DATABASE_URL`)와 `VRAMFORGE_LOG_LEVEL`만 받습니다.

| 변수 | 설정하지 않았을 때 | 설명 |
|---|---|---|
| `HF_TOKEN` | 익명 접근 | Hugging Face 토큰. [§5](#5-hugging-face-토큰) |
| `VRAMFORGE_ACCESS_TOKEN` | 토큰 없음 | 설정하면 모든 API 요청에 토큰이 필요합니다 |
| `VRAMFORGE_MAX_UPLOAD_BYTES` | `2147483648` (2 GiB) | 업로드 파일 상한. 올리면 `VRAMFORGE_PROXY_MAX_BODY_SIZE`도 함께 올립니다 |
| `VRAMFORGE_MAX_CONCURRENT_JOBS_PER_OWNER` | `2` | 사용자별 동시 분석 수 |
| `VRAMFORGE_JOB_TIMEOUT_S` | `21600` (6시간) | 분석 작업 하나의 시간 상한 |
| `VRAMFORGE_RETENTION_DAYS` | `7` | 업로드·분석 결과 보존 기간(일) |
| `VRAMFORGE_LOG_LEVEL` | `INFO` | api·worker 로그 수준 |

기본값은 API 설정(`services/api/src/vramforge_api/settings.py`)의 값입니다. compose 안에서 고정된 값(`VRAMFORGE_DATABASE_URL`, `VRAMFORGE_REDIS_URL`, `VRAMFORGE_DATA_DIR=/data`, `HF_HOME=/data/hf`, `VRAMFORGE_PROFILES_DIR=/app/profiles`)은 바꿀 필요가 없습니다.

## 5. Hugging Face 토큰

- 공개 모델·데이터셋은 토큰 없이 분석합니다.
- gated 또는 private 저장소를 분석하려면 **읽기 전용(fine-grained read) 토큰**을 만들어 `.env`에 `HF_TOKEN=hf_...`로 넣고 `docker compose up -d`를 실행합니다. gated 모델은 같은 계정으로 Hugging Face 웹사이트에서 이용 조건에 먼저 동의해야 합니다.
- 토큰은 서버 쪽 api·worker 컨테이너에만 전달됩니다. 브라우저, 분석 결과, 내보내기 파일, 로그에는 넣지 않습니다(plan.md §18).
- **셸에 `HF_TOKEN`이 export되어 있으면 compose가 그 값을 컨테이너로 전달합니다.** 개인 토큰을 쓰고 싶지 않다면 `unset HF_TOKEN` 뒤에 기동합니다.
- 토큰을 바꾸거나 지우면 `docker compose up -d`로 api·worker를 다시 만듭니다. 이미 받은 파일은 `vfdata` 볼륨의 HF cache(`/data/hf`)에 남습니다.

## 6. 로컬 모델·데이터셋

호스트의 `./local-sources` 폴더가 api·worker 컨테이너의 `/sources/local`에 **읽기 전용**으로 연결됩니다. 웹 화면에서는 `local:local/<상대 경로>`로 참조합니다. 사용법은 [`local-sources/README.md`](../local-sources/README.md)에 있습니다.

```text
local-sources/models/my-model      →  local:local/models/my-model
local-sources/datasets/my-dataset  →  local:local/datasets/my-dataset
```

- **다른 폴더를 쓰려면** `.env`에 `VRAMFORGE_LOCAL_SOURCES_DIR=/data/models`처럼 지정합니다. 참조 형식은 그대로 `local:local/...`입니다.
- **root를 더 추가하려면** `compose.override.yaml`을 만들어 mount를 추가하고 `VRAMFORGE_LOCAL_ROOTS`에 이름을 더합니다. compose는 이 파일을 자동으로 함께 읽습니다.

  ```yaml
  # compose.override.yaml
  services:
    api:
      volumes: ["/srv/datasets:/sources/datasets:ro"]
    worker:
      volumes: ["/srv/datasets:/sources/datasets:ro"]
  ```

  ```bash
  # .env
  VRAMFORGE_LOCAL_ROOTS=local=/sources/local,datasets=/sources/datasets
  ```

- **경로는 서버 기준입니다.** 브라우저를 연 PC가 아니라 Docker 호스트의 폴더입니다. 원격 Docker context(`DOCKER_HOST=ssh://...`)를 쓰면 원격 호스트의 경로가 연결됩니다.
- **권한.** 컨테이너는 uid `10001`로 실행됩니다. Linux에서는 파일을 다른 사용자도 읽을 수 있어야 합니다(`chmod -R a+rX <폴더>`).
- **SELinux(Fedora·RHEL 등).** bind mount를 읽지 못하면 `compose.override.yaml`의 mount에 `:z`를 붙이거나(`/srv/datasets:/sources/datasets:ro,z`) `chcon -Rt container_file_t <폴더>`로 레이블을 붙입니다.
- **Docker Desktop.** 폴더가 Docker Desktop의 File sharing 대상 경로 안에 있어야 합니다(macOS `/Users`는 기본 포함).

## 7. 데이터, 볼륨, 백업, 초기화

| 볼륨 | 컨테이너 경로 | 내용 |
|---|---|---|
| `vramforge_pgdata` | postgres `/var/lib/postgresql` | 분석 요청·상태·결과·이벤트 (기준 저장소) |
| `vramforge_redisdata` | redis `/data` | 작업 큐 (AOF) |
| `vramforge_vfdata` | api·worker `/data` | `artifacts/`(row 길이 산출물), `uploads/`(업로드 원본), `hf/`(Hugging Face cache) |

- `docker compose down`은 컨테이너와 네트워크만 지웁니다. 다시 `up` 하면 이전 분석이 그대로 있습니다.
- `docker compose down -v`는 **이 프로젝트의 볼륨 3개를 함께 지웁니다.** 모든 분석 결과, 업로드, HF cache가 사라지며 되돌릴 수 없습니다.
- 업로드와 분석 결과의 보존 기간은 API 설정 `VRAMFORGE_RETENTION_DAYS`(기본 7일)입니다. 만료된 업로드는 분석에 쓸 수 없고, worker의 정리 작업이 기간이 지난 항목을 지웁니다(HF cache는 지우지 않음). 볼륨 자체는 `down -v`를 실행하기 전까지 남습니다.

백업 (서비스가 실행 중일 때):

```bash
# DB (custom format)
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > vramforge-db.dump

# /data (HF cache는 다시 받을 수 있으므로 제외)
docker run --rm -v vramforge_vfdata:/data:ro -v "$PWD:/backup" alpine \
  tar czf /backup/vramforge-data.tgz -C /data --exclude=./hf .
```

복원:

```bash
docker compose up -d --wait
docker compose exec -T postgres sh -c 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists' < vramforge-db.dump
docker run --rm -v vramforge_vfdata:/data -v "$PWD:/backup" alpine \
  sh -c 'tar xzf /backup/vramforge-data.tgz -C /data && chown -R 10001:10001 /data'
```

## 8. 업그레이드

```bash
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > vramforge-db-before-upgrade.dump
git pull
docker compose up -d --build --wait
```

- 일회성 `migrate` 서비스가 api·worker보다 먼저 `alembic upgrade head`를 실행합니다. 실패하면 api와 worker가 시작되지 않습니다(`docker compose logs migrate`).
- PostgreSQL **메이저 버전**(예: 18 → 19)은 데이터 디렉터리 형식이 다르므로 `POSTGRES_IMAGE`만 바꾸면 안 됩니다. dump → 새 볼륨 → restore 순서로 옮깁니다.
- 이전 빌드 이미지는 `docker image ls 'vramforge-*'`로 확인합니다. 이 프로젝트가 빌드한 이미지만 지우려면 `docker compose down --rmi local`을 씁니다.

## 9. 이미지 mirror (Docker Hub pull 한도)

`toomanyrequests: You have reached your unauthenticated pull rate limit` 오류가 나면 Docker Hub 익명 pull 한도에 걸린 것입니다(공유 IP·CI에서 자주 발생). 둘 중 하나로 해결합니다.

1. `docker login`으로 Docker Hub에 로그인합니다.
2. 같은 이미지를 mirror에서 받도록 `.env`에 `*_IMAGE`를 지정합니다. `mirror.gcr.io`의 아래 태그는 Docker Hub와 digest가 같습니다(2026-10-04 확인). `UV_IMAGE`는 ghcr.io라서 바꿀 필요가 없습니다.

   ```bash
   PYTHON_IMAGE=mirror.gcr.io/library/python:3.12.15-slim-trixie
   NODE_IMAGE=mirror.gcr.io/library/node:24.21.0-trixie-slim
   CADDY_IMAGE=mirror.gcr.io/library/caddy:2.11.6-alpine
   POSTGRES_IMAGE=mirror.gcr.io/library/postgres:18.6-alpine
   REDIS_IMAGE=mirror.gcr.io/library/redis:8.10.2-alpine
   ```

사내 레지스트리가 있다면 `*_IMAGE`를 그 주소로 바꾸거나 Docker daemon의 `registry-mirrors` 설정을 씁니다. CI(`.github/workflows/ci.yml`)도 같은 mirror 변수를 사용합니다.

## 10. 멀티 아키텍처

- `docker compose up --build`는 호스트 아키텍처(linux/amd64 또는 linux/arm64)용 이미지를 빌드합니다. Python 의존성은 모두 두 아키텍처용 binary wheel이 있어 컴파일러가 필요 없고, Node·Caddy·PostgreSQL·Redis 이미지도 두 아키텍처를 제공합니다.
- Apple Silicon Mac은 arm64 이미지를 그대로 씁니다.
- 다른 아키텍처용 이미지를 확인하려면 buildx로 빌드합니다(arm64 호스트에서는 에뮬레이션).

  ```bash
  docker buildx build --platform linux/amd64 -f infra/docker/python.Dockerfile .
  docker buildx build --platform linux/amd64 -f infra/docker/web.Dockerfile apps/web
  ```

- 토큰화 결과는 아키텍처와 무관하게 같습니다(같은 데이터에서 amd64·arm64의 전체 token id sha256 일치, [`research/stack-compat.md`](research/stack-compat.md) §9).
- 32-bit ARM, ppc64le, s390x는 시험하지 않았습니다.

## 11. 리소스 예상

| 항목 | 값 | 비고 |
|---|---|---|
| 이미지 크기 (압축 해제, arm64) | Python 729 MB(api·worker·migrate가 같은 layer 공유), web 290 MB, proxy 64 MB, postgres 298 MB, redis 119 MB. 합계 약 1.5 GB | [§15](#15-검증-기록) |
| 대기 메모리 | 합계 약 280 MiB (api 86, worker 78, proxy 40, web 39, postgres 30, redis 7 MiB) | 분석이 없을 때, `docker stats` |
| 분석 메모리 | 데이터셋과 tokenizer 크기에 비례 | 예시 데이터셋(4,656 row) template 토큰화 smoke test의 peak RSS 약 614 MiB(research §5.4) |
| 디스크 | HF cache(tokenizer·config·데이터셋 파일), 업로드 원본, 산출물만큼 증가 | 모델 가중치는 받지 않음 |
| 권장 | Docker에 CPU 2개 이상, 메모리 4 GiB 이상 | 이미지 빌드(Next.js) 포함 기준의 권장값이며 측정값이 아님. 큰 데이터셋은 더 필요 |

분석은 worker 컨테이너에서 한 번에 하나씩 실행되고 나머지는 큐에서 기다립니다. worker 메모리를 제한하려면 `compose.override.yaml`에 `services: { worker: { mem_limit: 8g } }`처럼 지정합니다. 한도를 넘어 중단된 분석은 완료로 표시되지 않고 미완료 또는 실패로 남으며, 데이터를 잘라서 끝내지 않습니다(plan.md §18).

## 12. 문제 해결

| 증상 | 원인과 해결 |
|---|---|
| `Bind for 127.0.0.1:8080 failed: port is already allocated` | 다른 프로그램이 8080을 사용 중입니다. `.env`에 `VRAMFORGE_PORT=9000` |
| `up --wait`가 실패하거나 서비스가 unhealthy | `docker compose ps -a`로 상태를 보고 `docker compose logs <서비스>`로 원인을 확인합니다 |
| `migrate`가 0이 아닌 코드로 종료 | DB 접속 실패가 대부분입니다. 볼륨을 만든 뒤 `POSTGRES_PASSWORD`를 바꾸면 기존 DB 비밀번호와 달라집니다. 원래 값으로 되돌리거나, 데이터를 버려도 되면 `docker compose down -v` |
| `toomanyrequests` (이미지 pull) | [§9](#9-이미지-mirror-docker-hub-pull-한도) |
| 빌드 중 PyPI·npm·레지스트리 접속 실패 | 프록시 환경이면 Docker client 설정(`~/.docker/config.json`의 `proxies`)에 HTTP(S) 프록시를 넣으면 빌드와 컨테이너에 함께 적용됩니다. TLS 검사용 사내 CA는 시험하지 않았습니다 |
| Hugging Face 401/403 | gated·private 저장소입니다. [§5](#5-hugging-face-토큰)의 토큰과 이용 조건 동의를 확인합니다 |
| 로컬 경로를 읽지 못함 | 경로 형식(`local:local/...`), 파일 권한, SELinux를 확인합니다([§6](#6-로컬-모델데이터셋)) |
| `exec format error` | 다른 아키텍처용으로 빌드된 이미지입니다. `docker compose build --no-cache` |
| Windows에서 느림 | 저장소를 WSL2 파일시스템(예: `~/vramforge`) 안에 두고 그 안에서 `docker compose`를 실행합니다. Windows 호스트는 직접 시험하지 않았습니다([§15](#15-검증-기록)) |
| 로그가 계속 커짐 | Docker daemon의 로그 회전(`daemon.json`의 `"log-opts": {"max-size": "10m", "max-file": "3"}`)을 설정합니다 |

## 13. GPU 검증(M5)은 이번 릴리스에 없습니다

- 이번 릴리스는 CPU 정적 분석만 제공합니다. 분석, 재계산, 내보내기는 GPU 없이 모두 동작합니다.
- GPU 실측 검증(plan.md §17)은 구현되지 않았습니다. API의 `POST /api/v1/analyses/{id}/profile`은 `503 GPU_WORKER_UNAVAILABLE`을 반환하고 UI는 `GPU 검증 미연결`을 표시합니다.
- `compose.yaml`에는 나중에 쓸 `gpu` profile 자리가 주석으로만 있습니다. 기본 스택은 GPU 장치를 요구하지 않습니다.

## 14. 공개 배포 점검표

- [ ] HTTPS 종단 뒤에 두었다 (권장: `VRAMFORGE_BIND=127.0.0.1` + 호스트 TLS proxy)
- [ ] `VRAMFORGE_ACCESS_TOKEN`을 충분히 긴 임의 값으로 설정했다
- [ ] `VRAMFORGE_COOKIE_SECURE=true`
- [ ] 처음 기동하기 전에 `POSTGRES_PASSWORD`를 기본값이 아닌 값으로 정했다
- [ ] `HF_TOKEN`은 읽기 전용 최소 권한이다
- [ ] 방화벽과 바인드 주소로 노출 범위를 확인했다
- [ ] DB 백업 절차를 정했다([§7](#7-데이터-볼륨-백업-초기화))

## 15. 검증 기록

2026-10-04, `impl-infra`. 호스트는 macOS arm64 + OrbStack(Docker Engine 29.4.0, Compose v5.1.2, buildx 0.33.0, BuildKit 0.29.0)입니다.

| 검증 | 결과 |
|---|---|
| `.env` 없이 `docker compose config -q`, `env -i PATH=$PATH HOME=$HOME docker compose config -q` | 통과 |
| `docker compose build --no-cache` (base image와 uv·pnpm download cache는 로컬에 있음) | 28초 |
| 처음 빌드 (같은 호스트, 서비스별) | Python 이미지 20초, web 이미지 20초 |
| `docker compose up -d --wait` (프로젝트 `vramforge`, `127.0.0.1:8080`) | 18초 만에 상시 6개 healthy, `migrate` Exited (0) |
| `infra/scripts/smoke-test.sh` | `/healthz` 200, `/api/v1/health` 200(db·redis·worker 모두 ok), `/` 200 text/html, 보안 헤더 있음 |
| LAN 주소로 8080 접속 | 거부됨 (localhost에만 바인드) |
| 두 번째 `up -d --wait` | 3초, `migrate`가 다시 실행되어 변경 없이 종료(idempotent) |
| 접근 토큰 | `VRAMFORGE_ACCESS_TOKEN` 설정 시 `/api/v1/health`는 200, 다른 API는 토큰 없이 401·Bearer로 200. 해제하면 컨테이너에서 변수가 사라짐 |
| SSE | 실제 분석 이벤트가 Caddy를 거쳐 즉시 도착. `Accept-Encoding: gzip, zstd`를 보내도 압축하지 않음 |
| 브라우저 | Chromium으로 프록시 경유 페이지 로드. console 오류·CSP 위반 0건, 정적 asset 모두 200 |
| 네트워크 분리 | web에서 postgres·redis 이름 해석 불가, api는 접근 가능, worker는 Hugging Face 접속 가능, postgres는 외부 접속 불가 |
| 업로드 상한 | 상한을 넘는 본문은 Content-Length·chunked 모두 프록시에서 413 |
| 백업·복원 | [§7](#7-데이터-볼륨-백업-초기화)의 `pg_dump`·`pg_restore`, `tar` 명령이 그대로 동작 |
| linux/amd64 이미지 (arm64 호스트, 에뮬레이션) | `docker buildx build --platform linux/amd64` Python 이미지 25초, web 이미지 37초. 컨테이너가 x86_64로 실행되고 웹은 HTTP 200 |
| 종료 | `docker compose down -v`로 이 프로젝트의 컨테이너·네트워크·볼륨만 삭제 |

시험하지 않은 것: Docker Desktop(macOS·Windows) 자체, Windows(WSL2) 호스트, 네이티브 Linux 호스트에서의 기동, 이보다 오래된 Docker·Compose 버전. 네이티브 Linux amd64·arm64 기동은 CI의 `docker` job(`.github/workflows/ci.yml`)이 GitHub runner에서 확인합니다.
