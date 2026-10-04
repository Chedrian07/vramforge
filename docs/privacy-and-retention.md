# 개인정보·데이터 보존 정책

> 이 문서는 self-hosted VRAMForge 인스턴스가 **무엇을, 어디에, 얼마 동안** 저장하고 언제 지우는지 설명한다.
> 근거 사양은 [`plan.md`](../plan.md) §12.4, §16.3–16.4, §18이고, 구현은 `services/api`, `services/worker_cpu`, `packages/estimator/src/vramforge_estimator/exports/`에 있다.

| 항목 | 기본값 | 설정 |
|---|---|---|
| 보존 기간 | 7일 | `VRAMFORGE_RETENTION_DAYS` |
| 업로드 최대 크기 | 2 GiB | `VRAMFORGE_MAX_UPLOAD_BYTES` |
| 데이터 디렉터리 | `/data` (`artifacts/`, `uploads/`, `hf/`) | `VRAMFORGE_DATA_DIR` |
| 접근 토큰 | 없음 (localhost 바인드 전제) | `VRAMFORGE_ACCESS_TOKEN` |
| HTTPS 전용 쿠키 | 꺼짐 | `VRAMFORGE_COOKIE_SECURE=true` (HTTPS 배포에서 켠다) |

## 1. 저장하는 것

| 위치 | 내용 | 원문 포함 여부 |
|---|---|---|
| PostgreSQL `owners` | 익명 소유자 id. 브라우저 쿠키 값이 아니라 그 **sha256**만 저장한다 | 없음 |
| PostgreSQL `analyses` | 요청 설정, fingerprint, 작업 상태·진행, 결과 JSON, 오류 | 없음 (모델·데이터셋 참조와 설정값만) |
| PostgreSQL `analysis_events` | SSE 재연결용 진행 이벤트 | 없음 (단계, row 수, 최대 길이 같은 수치) |
| PostgreSQL `uploads` | 업로드 파일 이름(정리된 basename), 크기, sha256, 만료 시각 | 없음 |
| PostgreSQL `scan_cache` | 같은 소유자의 전처리 결과 재사용 정보(`preprocess_key` → 길이 artifact 경로, 스캔 요약) | 없음 |
| `/data/uploads/<owner>/<id>/` | **사용자가 올린 데이터 파일 원본** | 있음 |
| `/data/artifacts/<owner>/<analysis>/lengths/` | row별 길이 기록(Parquet): row id, 토큰 수, 상태, 내용 digest | 없음 (원문·token id 미저장) |
| `/data/artifacts/<owner>/<analysis>/model_inventory.json` | 모델 tensor 이름·shape·dtype 목록 | 없음 |
| `/data/artifacts/<owner>/<analysis>/checkpoint.json`, `artifacts.json` | 스캔 재개 위치, artifact 목록 | 없음 |
| `/data/hf/` | Hugging Face cache (config, tokenizer, 데이터셋 파일). **소유자 구분 없는 공용 cache** | 공개·접근 가능한 원본 파일 |

결과, SSE 이벤트, 로그, 내보내기에는 데이터셋 원문 row, 전체 token id, HF token, 호스트 절대경로를 넣지 않는다.

## 2. 소유권과 접근 제어

- 첫 API 호출 때 256-bit 난수 `vf_owner` 쿠키(httpOnly, SameSite=Strict, `VRAMFORGE_COOKIE_SECURE`이면 Secure)를 발급한다. 이 쿠키가 브라우저의 익명 소유자 신원이다. 쿠키를 지우면 이전 분석에 다시 접근할 수 없다.
- 분석, 이벤트, 내보내기, 업로드, 스캔 cache 조회는 모두 소유자로 거른다. 다른 소유자의 id는 존재하지 않는 id와 **같은 404 응답**을 받는다.
- 스캔 cache는 같은 소유자 안에서만 재사용한다. 다른 사용자의 비공개 데이터 존재 여부가 cache hit로 드러나지 않는다.
- 상태를 바꾸는 요청(POST/PUT/PATCH/DELETE)은 `X-VramForge-Request: 1` 헤더가 있어야 한다(CSRF 방어).
- `VRAMFORGE_ACCESS_TOKEN`을 설정하면 `/api/v1/health`와 `/api/v1/session`을 뺀 모든 API가 토큰을 요구한다. 브라우저는 `POST /api/v1/session`으로 토큰을 확인받아 httpOnly `vf_access` 쿠키(토큰에서 유도한 값)를 받고, API 클라이언트는 `Authorization: Bearer <token>`을 쓴다.

## 3. Hugging Face token

- `VRAMFORGE_HF_TOKEN`(또는 `HF_TOKEN`)은 서버 설정으로만 읽는다. 결과·이벤트·로그·내보내기·브라우저 저장소에 들어가지 않는다.
- 이 token은 **인스턴스의 모든 사용자 요청에 쓰인다.** 여러 사람이 쓰는 배포에서 비공개 repo에 접근할 수 있는 token을 설정하면, 그 인스턴스에 접근할 수 있는 모든 사용자가 해당 repo를 분석할 수 있다. 공용 배포에서는 공개 모델만 쓰거나 접근 범위를 공용으로 허용할 수 있는 token만 설정한다 (plan §18: 서비스 전역 계정으로 사용자 권한을 대신하지 않는다).

## 4. 보존 기간과 자동 삭제

worker가 시작할 때와 대기 중 약 1분마다(`VRAMFORGE_MAINTENANCE_INTERVAL_S`) 정리 작업을 실행한다.

| 대상 | 삭제 시점 |
|---|---|
| 끝난 분석(완료·부분·실패·취소·입력 필요) | 종료 시각으로부터 보존 기간이 지나면 DB row, 이벤트, artifact, cache 참조, 그 분석만 쓰던 업로드를 함께 삭제 |
| 업로드 | 업로드 후 보존 기간이 지나면 삭제. 단 실행 중인 분석이 쓰고 있으면 끝날 때까지 유지 |
| 스캔 cache 항목 | 보존 기간 동안 쓰이지 않았거나 artifact가 없어지면 삭제 |
| 고아 디렉터리 | DB row가 없는 서비스 형식(`<owner>/<id>`)의 디렉터리를 하루 뒤 삭제 |

- 실행 중인 분석은 보존 기간 계산에서 제외한다.
- 정리 작업은 서비스가 만든 경로(`/data/artifacts/<64자 hex>/<32자 hex>`, `/data/uploads/...`)만 지운다. 공용 HF cache(`/data/hf`)와 읽기 전용 로컬 root(`/sources/local`)는 지우지 않는다.
- worker가 멈춰 있으면 정리도 멈춘다. 데이터베이스·볼륨 백업은 이 정책의 범위 밖이다.

## 5. 사용자 삭제 요청

`DELETE /api/v1/analyses/{id}`는 즉시 다음을 지운다.

1. 분석 row와 이벤트 (실행 중이면 먼저 작업을 중지한다)
2. `/data/artifacts/<owner>/<id>/` 전체
3. 이 분석을 가리키는 스캔 cache 항목
4. 이 분석만 참조하던 업로드 파일과 row (같은 소유자의 다른 분석이 쓰는 업로드는 남긴다)

공용 HF cache와 다른 소유자의 데이터는 건드리지 않는다.

## 6. 로그

- 로그 출력은 포맷 단계에서 `hf_...` token, `Authorization`/Bearer 값, `vf_owner`/`vf_access` 쿠키, URL 자격증명과 서명 파라미터를 가린다(traceback 포함).
- 서비스 코드는 데이터셋 row 원문을 로그에 쓰지 않는다. 예기치 않은 오류의 세부 내용(traceback)은 서버 로그에만 남고 사용자 응답에는 내부 정보 없는 한국어 메시지만 간다.
- 단, 외부 라이브러리(datasets, pyarrow, tokenizer 등)가 예외 메시지에 입력 일부를 담으면 그 traceback이 서버 로그에 남을 수 있다. 서버 로그 접근은 운영자로 제한한다.

## 7. 내보내기

- `analysis.json`, `resolved-plan.yaml`, `report.md`는 저장된 결과에서 사용자가 요청할 때 만들고, 문자열에 섞인 token·절대경로·사설 URL·URL 자격증명을 가린다.
- `report.md`는 신뢰할 수 없는 문자열을 escape하므로 원시 HTML이나 Markdown 링크·이미지 문법을 포함하지 않는다. 가려지지 않은 공개 `http(s)` URL 문자열은 GFM 렌더러가 자동 링크로 표시할 수 있다.
- `trainer-config.yaml`은 학습 준비 상태가 `ready`이고 파이프라인이 중단 없이 메모리 산정까지 끝난 결과에서만 만든다. 원문 데이터는 포함하지 않는다.

## 8. GPU 검증

GPU 검증 worker는 이 릴리스에 연결되어 있지 않다(`POST /api/v1/analyses/{id}/profile` → `GPU_WORKER_UNAVAILABLE`). 데이터가 GPU 노드로 전송되는 경로가 없다.
