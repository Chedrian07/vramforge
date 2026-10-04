# GPU 검증 worker (자리만 있음, M5)

이 디렉터리는 선택적 GPU 검증 worker(plan.md §17)의 자리다. **이 릴리스에는 GPU worker가 없다.**

| 항목 | 현재 동작 |
|---|---|
| API | `POST /api/v1/analyses/{id}/profile` → `503` + `GPU_WORKER_UNAVAILABLE` (다른 소유자의 id는 `404`) |
| `GET /api/v1/health` | `components.gpu_worker = "not_connected"` |
| `GET /api/v1/backend-profiles` | `gpu_worker_connected = false` |
| UI | `GPU 검증 미연결` 표시. 정적 분석은 그대로 동작 |
| compose | 기본 구성에 서비스 없음 |

결과의 `estimate_evidence`는 `measured`가 될 수 없고 `measurement_scope.measured = false`다. 정적 추정을 실측처럼 표시하지 않는다.

## 나중에 구현할 때의 계약 (plan.md §17)

- **opt-in만**: 정적 분석 버튼에서 자동 실행하지 않는다. 사용자가 장치, dependency profile, 작업 범위, 모델 다운로드·저장 범위, 데이터 전달 범위를 확인한 뒤에만 시작한다.
- **격리**: 별도 이미지와 장치 allowlist를 쓰고 CPU worker와 다른 RQ queue에서 작업을 받는다. 사용자 GPU 작업을 종료하거나 드라이버를 바꾸지 않는다.
- **측정**: `max_memory_allocated`와 `max_memory_reserved`를 구분해 phase별로 기록하고, OOM이면 실패 장치·phase·shape를 저장한다. 관측치는 수행한 구간의 결과일 뿐 학습 전체의 OOM 부재 보증이 아니다.
- **보정**: 같은 phase·같은 지표끼리만 analytic ledger와 비교하고, 보정 영역을 벗어나면 `CALIBRATION_OUT_OF_DOMAIN`으로 표시한다.
- **보존**: 임시 checkpoint는 격리된 작업 디렉터리에 두고 [`docs/privacy-and-retention.md`](../../docs/privacy-and-retention.md)의 보존 정책을 따른다.
