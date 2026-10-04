# VRAMForge web (`apps/web`)

계산기 UI. Next.js 16 App Router + React 19 + TypeScript 6 + Tailwind CSS 4 + Radix + React Hook Form/Zod + TanStack Query + Recharts. 버전은 [`docs/research/stack-compat.md`](../../docs/research/stack-compat.md) §10.1에 맞춰 고정했다(TypeScript 6.0.3, ESLint 9.39.5, `react-is` 19.3.0 직접 의존성, Vitest 5 + vite 8).

자체 `package.json`과 `pnpm-lock.yaml`을 가진 **단독 pnpm 프로젝트**다. Node 24 LTS(`.node-version`), pnpm 10.34.6(`packageManager`)을 쓴다.

## 명령

```bash
pnpm install
pnpm dev                       # http://localhost:3000 (API_PROXY_TARGET=http://localhost:8000 이면 /api를 dev에서만 프록시)
NEXT_PUBLIC_VF_DEV_MOCKS=1 pnpm dev   # 백엔드 없이 fixture로 화면 확인 (화면 상단에 mock 배너)
pnpm lint                      # eslint (eslint-config-next core-web-vitals + typescript)
pnpm typecheck                 # next typegen && tsc --noEmit (fresh checkout에서도 통과)
pnpm test                      # vitest run (jsdom + Testing Library)
pnpm build                     # next build + scripts/check-no-mocks.mjs (fixture 표식이 .next에 있으면 실패)
pnpm test:e2e                  # Playwright, 기본 대상 http://localhost:8080 (E2E_BASE_URL로 변경)
pnpm gen:api                   # docs/api/openapi.json -> lib/api/schema.d.ts (생성물은 커밋)
```

E2E는 compose 스택(proxy :8080)을 대상으로 한다. 실제 분석은 tokenizer·데이터를 내려받아 전체 row를 스캔하므로 `E2E_ANALYSIS_TIMEOUT_MS`(기본 15분)로 대기 시간을 조정한다. 로컬에 번들 Chromium이 없으면 `E2E_CHANNEL=chrome`으로 설치된 Chrome을 쓸 수 있다. Docker에서는 `mcr.microsoft.com/playwright:v1.63.0-noble`을 쓴다.

## 빌드 산출물 (컨테이너)

`next.config.ts`는 `output: "standalone"`이다. 단독 프로젝트 레이아웃이므로 산출물은 다음처럼 복사한다(research §1.5, §10.7).

```text
.next/standalone/   -> /app            (server.js가 루트에 있다)
.next/static/       -> /app/.next/static
public/             -> /app/public
CMD ["node", "server.js"]   # ENV HOSTNAME=0.0.0.0 PORT=3000 필수
```

`/api` 라우팅은 reverse proxy가 맡는다. `rewrites`는 빌드 시 고정되므로 `next dev` 전용이다. 빌드는 Linux 컨테이너 안에서 한다(standalone에 호스트용 sharp 바이너리가 들어간다).

## API 계약

- REST 타입은 `lib/api/schema.d.ts`(OpenAPI 생성물)에서 온다. 요청 본문은 `lib/form/request-schema.ts`의 strict Zod mirror로 검증하며, 생성 타입과 key가 어긋나면 typecheck가 실패한다.
- 같은 출처 `/api/v1`만 호출한다. 소유자는 `vf_owner` cookie로 식별하고, POST/DELETE에는 `X-VramForge-Request: 1`을 보낸다. 분석 생성은 클릭마다 새 `Idempotency-Key`를 쓴다.
- 소유자 cookie 경합 방지: API는 cookie 없이 온 요청마다 새 `vf_owner`를 발급하므로, HTTP client는 첫 요청 전에 `GET /api/v1/session`을 한 번 순차로 보내고 나머지 요청은 그 응답을 기다린다. API에 닿지 못한 경우(네트워크, 프록시 5xx)에만 다음 요청에서 다시 시도한다. 응답이 토큰을 요구하면 접근 토큰 창을 연다.
- SSE(`/analyses/{id}/events`) 타입은 생성 스키마의 `AnalysisEvent`·`EventType`이고, 받은 payload는 `lib/api/events.ts`의 Zod mirror로 검증한다(key·출력 타입이 생성 타입과 다르면 typecheck 실패). 종료는 이벤트 type이 아니라 status로 판단한다(`failed` 이벤트가 PARTIAL을 담을 수 있다). 종료 이벤트 뒤에는 `GET /analyses/{id}`로 저장된 최종 상태를 읽어 화면을 정하고, 그 응답이 아직 실행 중이면 마지막 이벤트 다음부터 다시 따라간다.
- `?analysis=<id>`로 새로고침하면 재연결하고, `AnalysisStatus.request`로 입력 폼을 복원한다(결과가 생기기 전에도). 폼이 이미 편집됐으면 덮어쓰지 않는다. 이전 실행의 늦은 응답은 새 실행을 대신하지 않는다.
- 401이면 접근 토큰 입력 창을 띄우고 `POST /api/v1/session` `{ "token": "…" }`으로 보낸다. 서버는 httpOnly cookie를 발급하고, 토큰은 브라우저 저장소에 남기지 않는다.
- 메타데이터 조회는 응답이 없을 때·429·5xx에만 한 번 재시도한다. 4xx 응답은 바로 표시한다.
- 컬럼 매핑: 매핑을 지정하지 않으면 `mapping: null`(서버 자동 감지, 빈 system 생략)을 보낸다. 직접 지정한 매핑은 서버가 그대로 쓰므로 형식과 필수 역할을 갖춰야 하고, 선택한 형식의 역할만 보내고 요약에도 그 역할만 보인다. 조회 결과의 명백한 제안은 자동 적용하되(`mappingAutoApplied`), 자동 적용한 매핑은 이후 조회(학습 방식·config 변경 시 다시 조회)를 따라 바뀌거나 자동 감지로 돌아간다. 사용자가 고르거나 고친 매핑은 컬럼이 남아 있는 한 유지한다. 데이터셋을 바꾸면 config·split·revision·매핑을, config를 바꾸면 split을 자동 선택으로 되돌린다.
- 내보내기: 화면에 재계산 시나리오가 표시되면 `POST /analyses/{id}/scenarios/export` `{request, format}`으로 그 시나리오의 요청을 보내 파일을 받고, 아니면 저장된 분석을 `GET /export`로 받는다. 메뉴는 어떤 결과를 내보내는지 항상 밝히고, 결과 보관 기한(`AnalysisStatus.expires_at`)을 내보내기 버튼 아래와 메뉴에 보여 준다.
- 메모리 수식은 복제하지 않는다. 화면은 서버 결과(정수 bytes)를 표시만 하고, 모르는 값은 `산정 불가`와 사유로 보여 준다. context 상한을 모르면 초과 row 수도 `산정 불가 (context 상한 미상)`, 하한이면 `N개 이상`, 실패 row가 있는 스캔은 `전체 완료` 대신 `읽음 N · 실패 M`으로 표시한다.

## Mock 정책

Fixture는 `tests/fixtures/`에만 있고 모든 id에 `vf-fixture` 표식이 있다. dev mock 모듈(`lib/dev-mocks/index.ts`)은 `NEXT_PUBLIC_VF_DEV_MOCKS=1`일 때만 `@vf/dev-mocks` alias로 연결되며, 그 밖의 빌드는 빈 stub(`lib/dev-mocks/disabled.ts`)을 쓴다. `pnpm build`는 `.next` 결과물에서 표식을 찾으면 실패한다.

## 구조

```text
app/                    layout(테마 초기화), page, providers
components/calculator/  입력(모델·데이터·매핑·방법·하드웨어·Advanced), 실행 버튼, 진행 상태, 페이지 조립
components/results/     요약 카드, 상태 배지, 사용량 막대, 내보내기·삭제, 상세 탭(차트 + 표)
components/layout/      헤더, 테마, 도움말, 접근 토큰 창
components/ui/          접근성 기본 요소
lib/api/                클라이언트, SSE, 생성 타입
lib/form/               폼 값, 요청 변환·검증, 변경 분류, fingerprint
lib/hooks/              실행 수명주기, 재계산(300 ms debounce), 메타데이터 조회
lib/result/             요약 선택자, 상태 tone, 메모리 그룹
tests/unit/             Vitest
tests/e2e/              Playwright (실제 스택용)
```
