# 웹·API·인프라 스택 호환성과 Docker 이식성 조사

> plan.md §13.1의 확정 스택(Next.js·React·TypeScript, Tailwind·headless UI, RHF·Zod, TanStack Query·SSE, Recharts, FastAPI·Pydantic, Redis·RQ, PostgreSQL·SQLAlchemy·Alembic, pytest·Vitest·Playwright, uv·pnpm, Docker Compose)을 **실제 버전으로 설치·빌드·기동해** 호환 여부를 확인했다.
> 목표는 [`goals.md`](../goals.md)의 G0, 즉 GPU와 `.env` 없이 `docker compose up -d --build` 한 번으로 linux/amd64·linux/arm64·Docker Desktop 어디서나 기동되는 구성이다.
> 모든 실험은 저장소 밖 `/tmp/vf-stack/`에서 했고, 만든 컨테이너·볼륨·이미지·build cache는 끝난 뒤 지웠다.

| 항목 | 값 |
|---|---|
| 주제 | 웹/API/인프라 스택 버전 호환성, Next.js standalone, SSE 프록시, RQ 작업 제어, Docker 멀티 아키텍처 이식성 |
| 작성 | `research-stack` (Milestone M0) |
| 기준일 | 2026-10-04 |
| 대상 버전 (web) | next 16.3.8, react/react-dom 19.3.0, typescript 6.0.3 (7.0.2 검토 후 제외), tailwindcss 4.3.3, radix-ui 1.6.7, @tanstack/react-query 5.104.1, react-hook-form 7.89.0, zod 4.6.5, @hookform/resolvers 5.9.1, recharts 3.10.1, vitest 5.0.3, @playwright/test 1.63.0, openapi-typescript 7.13.0 |
| 대상 버전 (Python) | CPython 3.12.15 (3.13.16 병행 확인), fastapi 0.142.2, starlette 1.7.0, pydantic 2.13.5, pydantic-settings 2.15.0, sse-starlette 3.5.0, uvicorn 0.54.0, sqlalchemy 2.1.3, psycopg 3.3.6, alembic 1.20.0, rq 2.12.0, redis(-py) 8.1.0, transformers 5.18.0, tokenizers 0.23.2, huggingface_hub 1.33.0, datasets 5.0.1, pyarrow 25.0.1, safetensors 0.8.0 |
| 대상 이미지 | `node:24.21.0-trixie-slim`, `python:3.12.15-slim-trixie`, `ghcr.io/astral-sh/uv:0.12.23`, `postgres:18.6-alpine`, `redis:8.10.2-alpine`, `caddy:2.11.6-alpine`, `mcr.microsoft.com/playwright:v1.63.0-noble` |
| 실행 환경 | macOS arm64 + OrbStack (Docker Engine 29.4.0, Compose v5.1.2, buildx 0.33.0, BuildKit 0.29.0, 10 CPU, 17.6 GiB). 로컬 도구 node v25.8.0, pnpm 10.15.0, uv 0.11.19 |
| 방법 | (1) throwaway 프로젝트로 설치·빌드·lint·test 실행 (2) 설치된 패키지 소스를 줄 번호로 인용 (3) npm/PyPI/컨테이너 레지스트리 메타데이터 조회 (4) compose proof stack을 실제 기동해 curl·Playwright로 검증 (5) buildx 에뮬레이션으로 linux/amd64 이미지 빌드·실행 |
| 저장소 밖 산출물 | `/tmp/vf-stack/` (`web*`, `py`, `proof`, `e2e`), `/tmp/vf-research/scratch/stack/` (`tok_no_torch.py`, `fastapi_sse_204.py`) |
| 근거 표기 | npm 패키지는 `<pkg>@<ver> <패키지 기준 경로>:<줄>`, Python은 `<pkg>==<ver> <site-packages 기준 경로>:<줄> (<심볼>)`, 그 밖은 실험 ID(E1–E21). 태그는 VERIFIED(소스 확인 또는 실측), INFERRED(근거 기반 추론, 미실행), UNKNOWN |
| 주의 | Docker Hub 익명 pull 한도에 걸려서(E11) proof의 Docker Hub 이미지는 **같은 digest의** `mirror.gcr.io/library/*`로 받았다. compose의 `*_IMAGE` 변수 override만 사용했고 파일은 바꾸지 않았다. Docker Desktop 자체가 아니라 OrbStack에서 실행했다 |

## 핵심 결론

1. **TypeScript는 6.0.3으로 고정한다.** TS 7.0.2로도 `next build`와 `tsc`는 통과하지만 ESLint(typescript-eslint)와 openapi-typescript가 TS JS API가 없어서 실패한다. (Q1)
2. **ESLint는 9.39.5로 고정한다.** 10.x는 eslint-config-next 16.3.8 안의 eslint-plugin-react가 로드 단계에서 죽는다. (Q1)
3. **recharts를 쓰면 `react-is@19.3.0`을 직접 의존성으로 넣는다.** 넣지 않으면 pnpm이 peer를 react-is 16/17로 맞추고, fragment 안의 `<Cell>`이 조용히 무시된다. (Q3)
4. **Next.js standalone 규칙 4가지.** `HOSTNAME=0.0.0.0`을 지정하고, `.next/static`과 `public`을 직접 복사하고, rewrites는 빌드 시 고정되므로 API 라우팅은 프록시에 맡기고, fresh checkout에서는 `next typegen`을 먼저 실행해야 `tsc`가 통과한다. (Q1)
5. **Radix(`radix-ui` 1.6.7)는 필요한 primitive 전부가 React 19.3에서 동작한다.** jsdom에서는 Select용 polyfill이 필요하다. (Q2, Q4)
6. **Python 스택은 lock 하나로 3.12와 3.13에서 모두 해석된다.** torch 없이 tokenizer, chat template, datasets가 동작하고, binary 의존성은 전부 amd64/arm64 manylinux wheel이 있다. (Q5, Q9)
7. **SSE는 FastAPI 0.142 자체 기능으로 충분하다.** Caddy 2.11.6은 `text/event-stream`을 설정 없이 즉시 flush한다(gzip `encode`를 켜도 마찬가지). Chromium `EventSource`는 `Last-Event-ID` 자동 재연결과 204 중단 동작을 그대로 한다. (Q3, Q8)
8. **RQ는 기본값에 함정이 있다.** job timeout 기본값이 180초이고, 직렬화 기본값이 pickle이며, SimpleWorker는 정지할 수 없다. work-horse가 SIGKILL되면 `on_failure`도 실행되지 않는다. (Q6)
9. **proof stack 결과.** cold build 49초, 기동 30초 만에 상시 서비스 6개(proxy, web, api, worker, redis, postgres)가 healthy가 됐고 일회성 migrate는 exit 0으로 끝났다. Playwright 공식 이미지로 실제 브라우저 e2e 3건이 통과했다. (Q8)
10. **linux/amd64 이미지는 arm64 호스트에서 에뮬레이션으로 빌드하고 실행할 수 있다.** worker 빌드 27초, web 빌드 49초였고, 실제 tokenizer 결과도 arm64와 같다. (Q9)

## 실험 목록

| ID | 내용 (명령 요약) | 결과 |
|---|---|---|
| E1 | `CI=1 npx -y create-next-app@16.3.8 web --ts --tailwind --eslint --app --no-src-dir --import-alias '@/*' --use-pnpm --disable-git --no-react-compiler --no-agents-md --yes` | 11초. 설치된 버전은 next 16.3.8, react 19.2.8, typescript 5.9.3, eslint 9.39.5, @types/node 20.x, tailwind 4.3.3 |
| E2 | E1 사본에 `typescript@7.0.2`, react 19.3.0을 설치하고 `pnpm build`, 의도적인 타입 오류, `pnpm lint`, `tsc --noEmit` 실행 | build·tsc 성공, 타입 오류 검출, **lint 실패** |
| E3 | E1 사본에 TS 6.0.3, React 19.3.0, `output:'standalone'`, 시스템 폰트 적용 후 build/lint/typecheck | 모두 통과 |
| E4 | `openapi-typescript@7.13.0`을 FastAPI OpenAPI 3.1에 적용 (TS 6.0.3 / 7.0.2) | TS 6 성공, TS 7 TypeError |
| E5 | `eslint@10.12.0` + eslint-config-next 16.3.8 | 규칙 로드 중 crash |
| E6 | Radix 8종, TanStack Query, RHF+Zod, Recharts probe 컴포넌트로 build, lint, Vitest 7건 실행 (jsdom / happy-dom) | 모두 통과 (jsdom은 polyfill 필요) |
| E7 | react-is 16/17/19와 React 19 Fragment, recharts `<Cell>` in fragment 테스트 | react-is 19 아닐 때 실패 |
| E8 | torch 없는 venv(py3.12/3.13)에서 `AutoTokenizer` + `apply_chat_template` + `load_dataset` (`tok_no_torch.py`) | 성공, torch import 없음 |
| E9 | 3-member uv workspace `uv lock`, wheel 매트릭스 분석, uv 0.12.23의 `uv lock --check` | 80 패키지, 모든 binary wheel 존재 |
| E10 | 저장소 현재 `pyproject.toml`·`uv.lock` 사본으로 `uv sync --frozen --no-dev --package … --dry-run` | api 74개, worker 75개 설치, torch 없음 |
| E11 | proof compose(7 서비스) `build`, `up -d --wait`, curl, `docker stats`, `down -v --rmi local` | 상시 6개 healthy, migrate exit 0, 시간·크기 측정 |
| E12 | SSE 도착 시각 측정 스크립트로 Caddy 경유 FastAPI native / sse-starlette / gzip 스트림 측정 | 1초 간격 그대로 도착 |
| E13 | 분석 job 생성 → SSE 진행 이벤트 → `Last-Event-ID` 재개 (curl) | 재개 시 이후 이벤트만 재생 |
| E14 | `mcr.microsoft.com/playwright:v1.63.0-noble` 컨테이너를 compose 네트워크에 붙여 e2e 3건 실행 | 3 passed (6.4초) |
| E15 | RQ 실험: queued cancel, 협조적 cancel, `send_stop_job_command` + `on_stopped`, `Retry`, `unique`, JSONSerializer | 문서화한 의미 그대로 동작 |
| E16 | worker healthcheck와 idle heartbeat (`--worker-ttl` 기본 / 60) | 기본값에서는 2분 뒤 unhealthy |
| E17 | buildx `--platform linux/amd64` worker/web 빌드와 실행, 컨테이너 안 tokenizer 전체 스캔, py3.13 이미지 | 성공, 결과 동일 |
| E18 | Next 함정 확인: HOSTNAME bind, 빌드 시 rewrites 고정, fresh checkout typecheck, named volume 소유권, CRLF Caddyfile, gpu profile, nginx 비교, `# syntax` 지시어 | 각 절 참조 |
| E19 | 레지스트리 메타데이터: `npm view`, PyPI JSON, `docker buildx imagetools inspect`, Node release schedule | 각 절 참조 |
| E20 | 단일 프로젝트(`apps/web` 자체 lockfile) 레이아웃 Docker 빌드와 실행 | 38초, HTTP 200 |
| E21 | 이 문서 §10.7의 Dockerfile 블록을 그대로 추출해 proof 트리(`infra/docker/*`)에서 api/worker/web 빌드와 실행 | 75초. api import, alembic 1.20.0, `/data` 소유권 `app`, worker torch 없음, web HTTP 200 |

---

## 1. Next.js 16 스캐폴드, TypeScript 7, standalone (Q1)

### 1.1 create-next-app 16.3.8 기본값

- **[VERIFIED E1]** 스캐폴드가 고정하는 버전은 `react`/`react-dom` **19.2.8**(정확한 버전), `typescript ^5`(5.9.3 설치), `eslint ^9`(9.39.5, "no longer supported" deprecated 경고), `@types/node ^20`, `tailwindcss ^4`, `@tailwindcss/postcss ^4`, `eslint-config-next 16.3.8`이다. `pnpm-workspace.yaml`에 `ignoredBuiltDependencies: [sharp, unrs-resolver]`가 생성된다(**[INFERRED]** pnpm 10이 의존성 lifecycle script 실행에 명시 승인을 요구하기 때문으로 보인다. 이 설정 그대로 Docker 빌드와 실행이 정상인 것은 E11에서 확인했다).
- **[VERIFIED E1]** 기본 `app/layout.tsx`가 `next/font/google`(Geist)을 쓴다. **[INFERRED]** 이 폰트는 빌드 시 Google Fonts에서 받아 오므로 오프라인·사설망 빌드에서 실패할 수 있다. plan §3.2는 한글 system sans-serif를 요구하므로 제거한다. 제거한 뒤에도 빌드가 정상인 것은 확인했다(E3).
- **[VERIFIED E3]** React 19.3.0, TS 6.0.3, `@types/node` 24.19.1로 올려도 `next build`, `eslint`, `tsc --noEmit`이 모두 통과한다. `next build`는 Turbopack이 기본이다(출력 `▲ Next.js 16.3.8 (Turbopack)`).
- **[VERIFIED]** App Router 런타임은 프로젝트의 `react`가 아니라 Next에 vendoring된 React를 쓴다. `next@16.3.8 dist/compiled/react/cjs/react.production.js`의 버전 문자열은 `19.3.0-canary-cbb046ab-20260731`이다. 프로젝트의 `react@19.3.0`은 Vitest/RTL 테스트와 서드파티 peer 해석에 쓰인다.

### 1.2 Next 16.3.8의 type-check 경로

- **[VERIFIED]** `next@16.3.8 dist/server/config-shared.js:257`에서 `experimental.useTypeScriptCli`의 기본값이 `true`다. 그래서 `next build`는 TypeScript JS API를 로드하지 않고 **프로젝트 로컬 `tsc` CLI를 실행**한다. 번들 문서 `dist/docs/01-app/03-api-reference/05-config/02-typescript.md`의 "Using TypeScript 7" 절도 같은 내용이다(TS 7 지원, 대신 Next 전용 code frame과 오류 재작성은 적용되지 않음).
- **[VERIFIED]** `next@16.3.8 dist/lib/verify-typescript-setup.js:82-87`에서 TS가 없을 때 Next가 자동 설치하는 버전은 `typescript@^6.0.0`이다. 즉 Next 16.3이 기준으로 삼는 버전은 TS 6이다.
- **[VERIFIED]** `dist/lib/verify-typescript-setup.js:129`와 `dist/lib/typescript/runTypeScriptCli.js:85-86`: `useTypeScriptCli:false`인데 JS API가 없는 TS(=7)가 설치돼 있으면 `TypeScript 7.0.2 does not provide the compiler API required by Next.js ... or install TypeScript 6 instead.`(E1467)로 실패한다.

### 1.3 TypeScript 7.0.2 실험

- **[VERIFIED E2]** `typescript@7.0.2`의 `package.json` exports에서 `"."`는 `./lib/version.cjs`(`version`, `versionMajorMinor`만 있음)를 가리키고, 나머지는 `./unstable/*`뿐이다. `lib/typescript.js`가 없다. 실제 compiler는 플랫폼별 native 바이너리 패키지(`@typescript/typescript-linux-arm64` 등 optionalDependencies)다.
- **[VERIFIED E2]** `next build`는 성공한다. `Finished TypeScript in 359ms`로 TS 5.9.3의 1166ms보다 빠르다. 의도적으로 넣은 오류(`const n: number = "x"`)는 `error TS2322 … Failed to type check.`, exit 1로 잡힌다. 같은 코드를 `npx -p typescript@7.0.2 tsc --noEmit`으로 검사해도 통과한다.
- **[VERIFIED E2]** `pnpm lint`는 `typescript-eslint does not support TS 7.0.`을 내고 exit 2로 끝난다. 원인은 `typescript-eslint@8.71.0 dist/index.js:41-52`의 `if (versionMajor >= 7) throw`이고, peer 범위도 `typescript >=4.8.4 <6.1.0`이다(E19 `npm view`). 오류 메시지는 TS≥7.1 지원 추적 이슈(typescript-eslint#10940)를 안내한다.
- **[VERIFIED E4]** `openapi-typescript@7.13.0`을 TS 7.0.2와 함께 쓰면 `TypeError: Cannot read properties of undefined (reading 'createKeywordTypeNode')`가 난다(`dist/lib/ts.mjs:11`에서 `ts.factory` 사용).
- **결론 [VERIFIED E3/E4]** git-conventions §7의 web 게이트(lint + typecheck + test)와 API 타입 생성까지 모두 통과하는 최신 TS는 **6.0.3**(2026-04-16 릴리스)이다. openapi-typescript의 peer는 `typescript ^5.x`라서 경고만 나고 생성 결과는 정상이다.

### 1.4 ESLint 10

- **[VERIFIED E5]** `eslint@10.12.0`에서 `pnpm lint`를 실행하면 `TypeError: Error while loading rule 'react/display-name': contextOrFilename.getFilename is not a function`이 난다. 원인은 eslint-config-next 16.3.8이 끌어오는 eslint-plugin-react 7.37.5다.
- **[VERIFIED E19]** eslint-plugin-react 7.37.5의 peer는 `^9.7`까지, eslint-plugin-import 2.32.0과 eslint-plugin-jsx-a11y 6.10.2는 `^9`까지다. 따라서 **ESLint 9.39.5로 고정**한다(deprecated 경고만 나고 동작은 정상).

### 1.5 `output: 'standalone'`과 Docker 실행

- **[VERIFIED E3]** 단일 프로젝트(lockfile이 `apps/web` 안에 있는 경우)는 `.next/standalone/` 아래에 `server.js`, `package.json`, `.next/`(서버 산출물), `node_modules`(pnpm `.pnpm` 구조)가 만들어진다. probe 앱 기준 standalone 디렉터리 전체가 약 41 MB였다. `.next/static`(632 KB)과 `public`은 **포함되지 않으므로** 직접 복사해야 한다.
- **[VERIFIED E11]** 루트 pnpm workspace(`pnpm-workspace.yaml`이 루트에 있고 `apps/*`가 멤버인 경우)는 `apps/web/.next/standalone/apps/web/server.js`처럼 경로가 **한 단계 중첩**되고 `node_modules`는 standalone 루트에 놓인다. **[INFERRED]** Next가 `outputFileTracingRoot`를 workspace 루트로 추론하기 때문으로 보인다. 두 레이아웃 모두 Docker로 빌드해 HTTP 200을 확인했다(E11, E20).
- **[VERIFIED]** 생성된 `.next/standalone/server.js:8-9`(next 16.3.8)는 `PORT`(기본 3000)와 `HOSTNAME`을 읽고, `HOSTNAME`이 없을 때만 `'0.0.0.0'`을 쓴다. `KEEP_ALIVE_TIMEOUT`도 읽는다.
- **[VERIFIED E18] HOSTNAME 함정.** `--hostname webtest -e HOSTNAME=webtest`로 실행하면 서버가 `http://webtest:3000`에만 bind한다. 그러면 컨테이너 안의 `127.0.0.1:3000` healthcheck는 `ECONNREFUSED`, `webtest:3000`은 200이 된다. 이미지에 `ENV HOSTNAME=0.0.0.0`을 넣으면 `docker run`에서도 그 값이 유지된다(확인함).
- **[VERIFIED E3/E18]** `next.config`는 빌드할 때 `server.js` 안에 JSON으로 박힌다(`"_originalRewrites":{"beforeFiles":[],…}`). 런타임에 `API_PROXY_TARGET`을 줘도 rewrites는 비어 있어서 web의 `/api/v1/health`가 404를 반환한다. 따라서 **`/api` 라우팅은 reverse proxy(Caddy)에서 하고, `next.config`의 rewrites는 `next dev` 전용으로만 쓴다.**
- **[VERIFIED E3]** standalone `node_modules`에는 빌드한 호스트용 sharp 바이너리(macOS에서 빌드하면 `@img/sharp-darwin-arm64`)가 들어간다. 그러므로 **빌드는 반드시 Linux 컨테이너 안에서** 한다(호스트 빌드 결과를 COPY하지 않는다).
- **[VERIFIED E18] fresh checkout typecheck.** `.next/`와 `next-env.d.ts`는 gitignore 대상이다. 그래서 Node 24 컨테이너에서 새로 설치한 뒤 `tsc --noEmit`을 돌리면 `app/layout.tsx(9,50): error TS2304: Cannot find name 'LayoutProps'`로 실패한다. `next typegen`(`.next/types/*` 생성) 뒤에는 통과한다. 그러므로 `typecheck` 스크립트는 `next typegen && tsc --noEmit`이어야 한다.

### 1.6 Tailwind CSS 4.3.3

- **[VERIFIED E3]** `postcss.config.mjs`에 `{"@tailwindcss/postcss": {}}`, `globals.css`에 `@import "tailwindcss";`만 있으면 된다. `tailwind.config.js`는 필요 없다.
- **[VERIFIED E18]** 다크 모드를 class로 토글하려면(plan §3.2: 밝은 기본 + 다크 모드 제공) `globals.css`에 `@custom-variant dark (&:where(.dark, .dark *));`를 넣는다. 빌드된 CSS에 `.dark\:bg-zinc-950:where(.dark,.dark *)`가 생성되는 것을 확인했다. 이 지시어가 없으면 `dark:`는 `prefers-color-scheme`를 따른다(기본 스캐폴드 CSS도 이 방식). arbitrary value(`max-w-[1320px]`, `text-[#4F46E5]`)도 생성된다.

## 2. 접근성 headless 컴포넌트 (Q2)

- **[VERIFIED E6/E19]** umbrella 패키지 `radix-ui@1.6.7`(2026-07-24 배포)은 `import { Accordion, Collapsible, Dialog, Select, Switch, Tabs, ToggleGroup, Tooltip } from "radix-ui"` 형태로 쓴다. peer는 `react ^16.8 || … || ^19.0 || ^19.0.0-rc`이고, React 19.3.0에서 peer 경고 없이 설치된다. 개별 패키지(`@radix-ui/react-tabs@1.1.21`, `react-dialog@1.1.23`, `react-select@2.3.7` 등)의 peer도 같다.
- **[VERIFIED E6]** 위 8종을 쓰는 client component로 `next build`, `eslint`, `tsc`가 통과한다. Vitest+RTL에서 Tabs 전환(`role=tab`/`tabpanel`), Switch `aria-checked` 토글, ToggleGroup(single → `role=radio`), Dialog 열기와 Escape 닫기, Select 열기와 옵션 선택을 테스트했고 모두 통과했다.
- **[VERIFIED E6] jsdom 30.1.2 + Radix Select.** polyfill 없이 실행하면 `TypeError: target.hasPointerCapture is not a function`가 난다. setup 파일에 `hasPointerCapture`, `setPointerCapture`, `releasePointerCapture`, `scrollIntoView` polyfill(§10.4)을 넣으면 통과한다. happy-dom 20.14.5에서는 polyfill 없이 7/7이 통과한다.
- **[VERIFIED E6]** Tooltip은 앱 provider와 테스트 wrapper 모두에 `Tooltip.Provider`를 둔 상태로 확인했다.
- 대안 **[VERIFIED E19는 peer만, 동작은 INFERRED]** `@base-ui/react@1.8.0`(peer react `^17 || ^18 || ^19`), `react-aria-components@1.21.1`(peer `^16.8 … ^19.0.0-rc.1`), `@headlessui/react@2.2.10`(peer `^18 || ^19`)이 있다. Radix에 막히는 문제가 없으므로 대안을 쓸 이유는 없다.

## 3. 라이브러리 호환성 (Q3)

### 3.1 TanStack Query 5.104.1
- **[VERIFIED E6/E14]** peer는 `react ^18 || ^19`다. `QueryClientProvider`와 `useQuery`로 `/api/v1/health`를 가져오는 것을 Vitest(fetch stub)와 실제 브라우저(프록시 경유)에서 모두 확인했다. `QueryClient`는 `useState(() => new QueryClient())`로 client component 안에서 만든다(§10.3 `providers.tsx`).

### 3.2 React Hook Form 7.89.0 + Zod 4.6.5 + @hookform/resolvers 5.9.1
- **[VERIFIED E19]** `@hookform/resolvers@5.9.1`의 peer는 `zod ^3.25.0 || ^4.0.0`, `react-hook-form ^7.55.0`이다. react-hook-form 7.89.0의 peer는 `react ^16.8 … ^19`다.
- **[VERIFIED]** `@hookform/resolvers@5.9.1 zod/dist/zod.d.ts:47-52`는 Zod3Type/Zod4Type 오버로드를 제공한다. Zod 4 스키마를 넘기면 `Resolver<z4.input<T>, Context, z4.output<T>>`이 된다. 런타임은 `"_zod" in schema`로 v4를 구분한다(`zod/dist/zod.mjs`).
- **[VERIFIED E6]** `z.coerce.number()`가 들어간 스키마에서는 `useForm<z.input<typeof s>, unknown, z.output<typeof s>>({ resolver: zodResolver(s) })`처럼 제네릭 3개를 지정해야 타입이 맞는다. 제출 값은 `{rank: 16}`(number)로 변환됐고 오류 메시지 렌더링도 통과했다. `import { z } from "zod"`가 v4 classic API다.

### 3.3 Recharts 3.10.1
- **[VERIFIED E19]** peer는 `react`, `react-dom`, `react-is`가 모두 `^16.8 … ^19`다.
- **[VERIFIED E7] react-is 함정.** pnpm의 auto-install-peers는 `react-is`를 의존성 그래프에 이미 있는 오래된 버전으로 맞춘다. 실제로 eslint-plugin-react → prop-types 경로의 16.13.1, 다른 fresh install에서는 17.0.2로 맞춰졌다. React 19 element의 `$$typeof`는 `Symbol(react.transitional.element)`인데, react-is 16.13.1의 `isFragment`는 이것을 false로 판정한다. 19.3.0은 true로 판정한다(node 실험).
- **[VERIFIED]** `recharts@3.10.1 es6/util/ReactUtils.js:3,41`이 `isFragment`로 children을 펼치고, `Bar.js:527`, `Pie.js:69/568`, `Scatter.js:563` 등이 `findAllByType(children, Cell)`을 쓴다. 따라서 react-is 17.0.2일 때 fragment 안의 `<Cell fill>`이 무시되어 fill이 `[null, null]`이 되고, react-is 19.3.0에서는 `["#ff0000", "#00ff00"]`이 된다(E7 테스트). **해결: `react-is@19.3.0`을 dependencies에 명시한다.**
- **[VERIFIED E6]** jsdom 테스트에서는 `width`/`height`를 고정한 차트로 검증했다(`.recharts-bar-rectangle` 수 확인). **[INFERRED]** `ResponsiveContainer`는 jsdom에서 크기가 0이라 아무것도 그리지 않을 수 있으므로 테스트에서는 고정 크기를 쓴다.

### 3.4 openapi-typescript 7.13.0
- **[VERIFIED]** FastAPI 0.142.2는 OpenAPI `3.1.0`을 생성한다(`fastapi==0.142.2 fastapi/applications.py:942`).
- **[VERIFIED E4]** `openapi-typescript openapi.json -o lib/api/schema.d.ts`는 29 ms에 끝난다. 생성된 `paths`/`components`로 `openapi-fetch@0.17.0` 클라이언트(`api.GET("/api/v1/analyses/{aid}", { params: { path: { aid } } })`)를 만들면 `tsc`와 `eslint`가 통과한다.
- **[VERIFIED E4]** FastAPI는 SSE 응답의 `text/event-stream` content에 OpenAPI 3.2 키워드 `itemSchema`를 넣는다(`fastapi/openapi/utils.py:436-455`). openapi-typescript는 이 키워드를 무시해서 `"text/event-stream": unknown`이 된다. **SSE 이벤트 payload의 TS 타입은 별도로 만들어야 한다**(§구현 시사점 W8).

### 3.5 네이티브 EventSource
- **[VERIFIED E14, Chromium / Playwright 1.63.0]** 서버가 이벤트 3개를 보낸 뒤 연결을 닫으면 브라우저가 자동으로 재연결하고 `Last-Event-ID` 헤더에 마지막으로 받은 id를 보낸다. 서버가 받은 값은 `[null×3, "2"×3, "5"×3]`이었고 `onopen`은 3회 호출됐다. 스트림의 `retry: 500`도 적용된 것으로 보인다(전체 3.5초. 브라우저 기본 재연결 대기였다면 더 길었을 것이므로 시간으로 판단).
- **[VERIFIED E14]** 서버가 204를 응답하면 EventSource가 재연결을 멈추고 `readyState === CLOSED(2)`가 된다.
- **[VERIFIED E6]** jsdom과 happy-dom에는 `EventSource`가 없다(`"EventSource" in globalThis === false`). 테스트에서는 `vi.stubGlobal("EventSource", Fake)`로 대체한다(§10.4 테스트 예시).
- **[INFERRED, HTML spec]** EventSource는 임의 헤더(`Authorization`, `X-VramForge-Request`)를 보낼 수 없다. same-origin 요청에는 cookie가 붙는다(credentials mode `same-origin`). 그래서 SSE 인증은 owner cookie로 하고, `VRAMFORGE_ACCESS_TOKEN` 모드에서도 cookie 또는 별도 메커니즘이 필요하다. 클라이언트는 종료 이벤트를 받으면 `es.close()`를 호출해야 자동 재연결 루프가 생기지 않는다.

## 4. 테스트 러너 (Q4)

### 4.1 Vitest 5.0.3 + RTL + jsdom
- **[VERIFIED E19]** vitest 5.0.3은 vite를 dependency가 아니라 **peer**(`^6.4.0 || ^7.0.0 || ^8.0.0`)로 갖는다. 따라서 `vite@8.3.2`를 직접 설치한다. `@vitejs/plugin-react@6.1.1`의 peer는 `vite ^8`이다.
- **[VERIFIED]** Vite 8은 tsconfig `paths`를 자체 지원한다: `resolve.tsconfigPaths: true`(`vite@8.3.2 dist/node/index.d.ts:2720-2726`, 기본 false). `vite-tsconfig-paths` 플러그인은 필요 없다.
- **[VERIFIED E6]** `@testing-library/jest-dom@7.0.1`은 `@testing-library/jest-dom/vitest` entry를 제공한다(peer `vitest >= 0.32`). `test.globals`를 끄면 RTL 자동 cleanup이 등록되지 않으므로 `afterEach(cleanup)`을 직접 둔다. 이 설정(§10.4)으로 테스트 7건이 1.7초에 통과했다.
- **[VERIFIED E19] engines.** vitest는 `^22.12.0 || ^24.0.0 || >=26.0.0`, jsdom 30.1.2는 `^22.22.2 || ^24.15.0 || >=26.0.0`, jest-dom 7은 `>=22`다. 로컬 Node 25(EOL)는 범위 밖이지만 실행은 통과했다. **[VERIFIED E18]** Node 24.21.0 Linux 컨테이너에서 `pnpm install --frozen-lockfile`, `lint`, `test`가 통과했다(17초). `typecheck`는 §1.5대로 `next typegen`이 먼저 필요하다.

### 4.2 Playwright 1.63.0 Docker 이미지
- **[VERIFIED E19]** `mcr.microsoft.com/playwright:v1.63.0-noble`은 `v1.63.0`과 같은 index digest(`sha256:eff16c30e6f3…`)이고 linux/amd64와 linux/arm64를 제공한다. 압축 크기는 amd64 956 MB, arm64 937 MB다(로컬 풀어 놓은 크기 2.46 GB). Ubuntu 24.04 기반이고 `PLAYWRIGHT_BROWSERS_PATH=/ms-playwright`, 이미지 안 Node는 v24.20.0이다. `-jammy`(22.04)와 `-resolute` 변형도 있다.
- **[VERIFIED E14]** compose 네트워크에 붙여 실행한 명령은 다음과 같다. 3건 모두 통과했다(테스트 6.4초, `npm install` 포함 12초).

  ```bash
  docker run --rm --network <project>_default --ipc=host \
    -v "$PWD/e2e:/e2e" -w /e2e -e BASE_URL=http://proxy -e CI=1 \
    mcr.microsoft.com/playwright:v1.63.0-noble \
    sh -c 'npm install --no-audit --no-fund && npx playwright test'
  ```
- **[INFERRED]** `@playwright/test` 버전은 이미지 태그와 정확히 같아야 한다(브라우저 빌드가 버전에 묶여 있음). Chromium에는 `--ipc=host`(compose에서는 `ipc: host`)를 권장한다.

## 5. Python workspace와 의존성 해석 (Q5)

### 5.1 uv workspace
- **[VERIFIED E9]** 멤버 3개(`packages/estimator`, `services/api`, `services/worker_cpu`)를 둔 virtual root(`[project]` 없이 `[tool.uv.workspace]`만 있는 형태)에서 `uv lock`이 0.5초 만에 **80개 패키지**로 해석됐다. lock 헤더는 `version = 1`, `revision = 3`, `requires-python = ">=3.12, <3.14"`다.
- **[VERIFIED E10]** 저장소에 이미 있는 형태(root에 `[project]`와 `[tool.uv] package = false`, extra `analysis`, `parity` group에 torch)로도 문제없다. `uv sync --frozen --no-dev --package vramforge-api --dry-run`은 74개, `--package vramforge-worker`는 75개를 설치하고 **torch, trl, peft, accelerate는 포함되지 않는다**. `parity` group은 기본 group이 아니므로 Docker 빌드에 들어가지 않는다.
- **[VERIFIED E9]** uv 0.12.23(현재 최신, `ghcr.io/astral-sh/uv:0.12.23`)으로 0.11.19가 만든 lock에 `uv lock --check`를 돌리면 통과하고 파일도 바뀌지 않는다.

### 5.2 버전 제약 사슬 **[VERIFIED E19, PyPI `requires_dist`]**

| 패키지 | 핵심 제약 | 결과 |
|---|---|---|
| transformers 5.18.0 | `huggingface-hub>=1.31.0,<3.0`, `tokenizers>=0.23.1,<0.24.0`, `safetensors>=0.8.0`, torch는 의존성 아님 | — |
| datasets 5.0.1 | `huggingface-hub>=0.25.0,<2.0`, `pyarrow>=21.0.0`, `fsspec[http]<=2026.6.0`, `dill<0.4.2`, `multiprocess<0.70.20` | fsspec 2026.6.0으로 고정됨 (최신 2026.9.0 제외) |
| tokenizers 0.23.2 | `huggingface-hub<2.0` | hub 1.33.0 (PyPI 최신 2.1.1 제외) |
| fastapi 0.142.2 | `starlette>=0.46.0`, `pydantic>=2.9.0`, **`opentelemetry-api>=1.44.0`**, `annotated-doc` | starlette 1.7.0, opentelemetry-api 1.45.0이 따라옴 |
| sse-starlette 3.5.0 | `starlette>=0.49.1`, `anyio>=4.7.0` | — |
| rq 2.12.0 | `redis!=6,>=3.5`, `croniter`, `click` | redis 8.1.0 |
| sqlalchemy 2.1.3 / numpy 2.5.3 / pandas 3.0.6 | Python `>=3.11` / `>=3.12` / `>=3.11` | Python 하한은 3.12 |

### 5.3 wheel 가용성 **[VERIFIED E9, `uv.lock` wheel 목록]**
- 모든 binary 패키지가 manylinux **x86_64와 aarch64**, **cp312와 cp313** wheel을 갖는다. 대상은 pyarrow 25.0.1, tokenizers 0.23.2(abi3), psycopg-binary 3.3.6, pydantic-core 2.46.5, numpy 2.5.3, pandas 3.0.6, safetensors 0.8.0(abi3), hf-xet 1.6.0(abi3), regex, xxhash, uvloop, httptools, watchfiles, websockets, aiohttp, yarl, multidict, propcache, frozenlist, markupsafe, pyyaml, charset-normalizer, sqlalchemy 등이다. **컨테이너 빌드에 컴파일러가 필요 없다.**
- pyarrow와 numpy wheel 태그는 `manylinux_2_28`이므로 glibc 2.28 이상이 필요하다. Debian trixie(`python:3.12-slim` = trixie)라 문제없다. musllinux aarch64 wheel도 대부분 있지만(dev 의존성 ruff만 없음) alpine 이미지는 검증하지 않았다.
- psycopg-binary는 libpq를 내장한다. macOS wheel에서 `psycopg.pq.version() == 180006`(libpq 18.0.6), `__impl__ == "binary"`였고, Linux 컨테이너(amd64/arm64)에서도 `__impl__ == "binary"`였다(E17). 이미지에 시스템 libpq가 필요 없다.

### 5.4 torch 없는 tokenizer 사용 **[VERIFIED E8, `/tmp/vf-research/scratch/stack/tok_no_torch.py`]**
실행 명령: `HF_HOME=/tmp/vf-research/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 /tmp/vf-stack/py/.venv/bin/python tok_no_torch.py`. venv는 `uv sync --package vramforge-estimator`로 만든 것이라 torch가 없다.

- `importlib.util.find_spec("torch") is None` 상태에서 `import transformers`에 2.06초, `is_torch_available()`은 False다. import 시점에 torch를 요구하지 않는다. 다만 advisory 경고 `PyTorch was not found. Models won't be available…`가 출력된다(`transformers==5.18.0 transformers/__init__.py:873-876`, `logger.warning_advice`). `TRANSFORMERS_NO_ADVISORY_WARNINGS=1`이면 숨겨진다(`transformers/utils/logging.py:317-325`, 확인함).
- offline에서 `AutoTokenizer.from_pretrained(repo, revision=…)`는 `Qwen3_5Tokenizer`, `is_fast=True`, `len(tok)=248077`, chat_template 있음, 로드 0.54초다.
- **`apply_chat_template`의 기본값은 `tokenize=True, return_dict=True`**(`transformers/tokenization_utils_base.py:2990-3007`)다. 따라서 반환값은 list가 아니라 `BatchEncoding`(`input_ids`, `attention_mask`)이므로 `out["input_ids"]`로 꺼내야 한다.
- `return_tensors="pt"`는 `ImportError: Unable to convert output to PyTorch tensors format, PyTorch is not installed.`를 내고, `"np"`는 ndarray를 반환한다. 실행이 끝난 뒤 `sys.modules`에 torch, tensorflow, jax 모듈이 없었다.
- 처리량(smoke test이며 TRL parity 길이가 아님): chosen 분기 4,656개 template 렌더링 0.22초, batch 토큰화 0.31초. peak RSS는 데이터셋 로드를 포함해 약 614 MiB(`/usr/bin/time -l` 643,317,760 B)였다.
- Python 3.13.13(macOS)과 Docker `python:3.13-slim`(3.13.16)에서도 같은 결과다(E8, E17).

## 6. RQ 2.12 핵심 API (Q6)

| 항목 | 동작 | 근거 |
|---|---|---|
| 기본 job timeout | **180초** (`Queue.DEFAULT_TIMEOUT`). 긴 토큰화 작업에는 `job_timeout`을 반드시 명시한다(정수 초 또는 `"2h"`, `"30m"`) | `rq==2.12.0 rq/queue.py:106`, `rq/utils.py:432-451 (parse_timeout)` VERIFIED |
| enqueue 인자 | `job_id`, `job_timeout`, `result_ttl`(기본 500초), `ttl`, `failure_ttl`(기본 1년), `depends_on`, `at_front`, `meta`, `retry`, `repeat`, `on_success`, `on_failure`, `on_stopped`, `unique`, `pipeline` | `rq/queue.py:963-1000 (Queue.parse_args)`, `rq/defaults.py` VERIFIED |
| 함수 참조 | `"vramforge_worker_cpu.tasks.analyze"` 같은 문자열 경로를 쓸 수 있다. API 이미지가 worker 코드를 import하지 않아도 된다 | E11, E15 VERIFIED |
| 멱등 enqueue | `unique=True`이면 `job_id`가 필요하고, 같은 id를 다시 넣으면 `DuplicateJobError`가 난다 | `rq/queue.py:755-764`, `rq/exceptions.py:17`, E15 VERIFIED |
| 상태 | `JobStatus`: created, queued, finished, failed, started, deferred, scheduled, stopped, canceled. `Job.fetch(id, connection=, serializer=)`, `job.get_status()`, `job.return_value()`, `job.latest_result()`로 조회 | `rq/job.py:59-70`, `:865`, `:934` VERIFIED |
| 진행 보고 | task 안에서 `get_current_job()`을 얻고 `job.meta[...] = …`, `job.save_meta()`(job hash의 `meta` 필드에 hset) → API에서 `job.get_meta()`. E13에서 `progress: {processed_rows: 3, total_rows: 8}` 확인 | `rq/job.py:139`, `:1172-1175` VERIFIED |
| queued job 취소 | `job.cancel()` → CANCELED, 큐에서 제거, CanceledJobRegistry에 추가. **실행 중인 horse를 멈추지는 않는다** | `rq/job.py:1177-1240`, E15 VERIFIED |
| 실행 중 job 정지 | `send_stop_job_command(conn, job_id, serializer=None)`: 실행 중이 아니면 `InvalidJobOperation`을 낸다. `rq:pubsub:<worker>` 채널에 `stop-job`을 publish하면 worker가 `_stopped_job_id`를 설정하고 `kill_horse()`(`os.killpg(…, SIGKILL)`)를 호출한다. 결과는 **STOPPED**이고 retry하지 않으며 FailedJobRegistry로 간다 | `rq/command.py:14,70-82,128-141`, `rq/worker/worker_classes.py:30-37`, `rq/worker/base.py:100,715-719`, E15 VERIFIED |
| `on_stopped` | 강제 정지 뒤 **부모 worker 프로세스**가 `callback(job, connection)`을 실행한다. E15에서 DB를 CANCELLED로 바꾸고 이벤트를 추가했다 | `rq/worker/worker_classes.py:131-135`, E15 VERIFIED |
| SimpleWorker | `BaseWorker.kill_horse`가 no-op이라 stop 명령이 효과가 없다. **forking `rq.Worker`(기본값)를 쓴다** | `rq/worker/base.py:1649-1651` VERIFIED |
| horse 비정상 종료 | OOM 등으로 stop 명령 없이 SIGKILL되면 `Work-horse terminated unexpectedly`, `handle_work_horse_killed`, FAILED 순으로 처리된다. `on_success`/`on_failure`는 horse 안의 `perform_job`에서만 실행되므로 **이 경우 on_failure는 실행되지 않는다**. `Worker(work_horse_killed_handler=…)` 훅이 있다 | `rq/worker/worker_classes.py:137-146`, `rq/worker/base.py:150,1325-1331,1574,1587` VERIFIED |
| retry | `Retry(max, interval=int\|list)`. interval이 0보다 크면 SCHEDULED가 되어 scheduler가 다시 넣으므로 **`--with-scheduler`가 필요하다**. E15에서 `queued(2) → scheduled(1) → scheduled(0) → failed(0)` 확인 | `rq/job.py:1710-1733,1847-1878` VERIFIED |
| 직렬화 | 기본은 **pickle**(`DefaultSerializer`)이고, `JSONSerializer`가 있다(CLI 별칭 `json`). JSON 왕복(payload는 zlib 압축 JSON)은 E15에서 확인했다 | `rq/serializers.py:19-21,27,38` VERIFIED |
| worker CLI | `rq worker [QUEUES] -u/--url (env RQ_REDIS_URL) -s/--with-scheduler --worker-ttl -S/--serializer -w/--worker-class -j/--job-class -b/--burst --max-jobs --max-idle-time -n/--name` | `rq/cli/helpers.py:393`, `rq worker --help` VERIFIED |
| worker 이름 | 기본값은 `uuid4().hex`다. 같은 이름의 활성 worker가 있으면 `register_birth`가 `ValueError`를 낸다. `--name $HOSTNAME`처럼 고정하면 컨테이너 재시작 때 충돌할 수 있다 | `rq/worker/base.py:205,905-911` VERIFIED |
| idle heartbeat | `dequeue_timeout = max(1, worker_ttl − 15)`라서 기본 `worker_ttl` 420초면 idle worker는 **최대 405초마다** heartbeat한다. 작업 중에는 30초 간격으로 모니터링한다 | `rq/worker/base.py:442-447`, E16 VERIFIED |
| 종료 신호 | 첫 SIGTERM/SIGINT는 warm shutdown(현재 job 완료 후 종료), 두 번째는 cold shutdown이다 | `rq/worker/base.py:500-530` VERIFIED |
| redis-py 8 | redis-py 8.1.0 + Redis 서버 8.10.2에서 위 실험이 모두 통과했고 deprecation 경고는 없었다 | E11, E15 VERIFIED |

E15 실험 요약 **[VERIFIED]**:
- queued B에 `job.cancel()`: rq `canceled`. 이 경로에서는 API가 DB 상태를 직접 갱신해야 한다(proof에서는 하지 않아 QUEUED로 남음).
- 실행 중 A에 cancel flag 설정: worker가 다음 행에서 확인하고 `cancelled` 이벤트를 남긴 뒤 정상 return했다. rq는 `finished`, DB는 CANCELLED.
- 실행 중 C에 `send_stop_job_command`: rq `stopped`, `on_stopped`가 DB를 CANCELLED로 바꾸고 `reason: hard_stop` 이벤트를 남겼다. worker 로그는 `killed horse pid 14 … stopped by user, moving job to FailedJobRegistry`.
- `Retry(max=2, interval=[1,2])`: 2회 재시도 후 FAILED, `latest_result().exc_string`에 `FileNotFoundError`가 남았다.

## 7. Docker 이미지, healthcheck, Compose 기능 (Q7)

### 7.1 Node LTS **[VERIFIED E19: nodejs/Release `schedule.json`, nodejs.org `dist/index.json`, 2026-10-04 조회]**
| 메이저 | 상태 (2026-10-04) | 최신 | 일정 |
|---|---|---|---|
| 24 (Krypton) | **Active LTS** | 24.21.0 (2026-09-07) | Maintenance 2026-10-20부터, EOL 2028-04-30 |
| 26 | Current | 26.10.0 (2026-09-21) | **Active LTS 2026-10-28부터**, EOL 2029-04-30 |
| 22 (Jod) | Maintenance LTS | 22.23.3 | EOL 2027-04-30 |
| 25 | **EOL (2026-06-01)** | 25.9.0 | 로컬 개발 도구 v25.8.0이 여기에 해당 |

- **[VERIFIED E18]** `node:24.21.0-trixie-slim`에는 corepack 0.36.0, npm 11.19.0, yarn이 들어 있고 Debian 13(trixie), `node` 사용자(uid 1000)가 있다. **`node:26.10.0-trixie-slim`에는 corepack과 yarn이 없다**(`which corepack`이 실패). Node 26으로 옮길 때는 `npm i -g pnpm@<ver>`가 필요하다.

### 7.2 이미지 표 **[VERIFIED E19: `docker buildx imagetools inspect`; 압축 크기는 manifest layer 합계]**

| 용도 | 권장 태그 (버전 env) | 플랫폼 | 압축 크기 amd64/arm64 | index digest (2026-10-04) | healthcheck (E11에서 healthy 확인) |
|---|---|---|---|---|---|
| web 빌드·런타임 | `node:24.21.0-trixie-slim` (`NODE_VERSION=24.21.0`) | amd64, arm64/v8, ppc64le, s390x | 82 / 83 MB | `sha256:8ec5d7557396cfe32d21c3f9c13072355ceab22b584578ca4bb28af31120cffe` | `node -e "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.ok?0:1),()=>process.exit(1))"` |
| Python 런타임 | `python:3.12.15-slim-trixie` (`PYTHON_VERSION=3.12.15`) | amd64, arm64/v8 외 5종 | 46 / 47 MB | `sha256:29113dcae7aad06daa8e95260fa09f27d62be33b9687ea3774f771d601a02256` | api: `python -c "import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/api/v1/health',timeout=2).status==200 else 1)"` / worker: §10.8 모듈 |
| uv 바이너리 | `ghcr.io/astral-sh/uv:0.12.23` (distroless: `/uv`, `/uvx`) | amd64, arm64 | 24 / 22 MB | `sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21` | (빌드 전용) |
| DB | `postgres:18.6-alpine` (`PG_VERSION=18.6`) | amd64, arm64/v8 외 6종 | 120 / 118 MB | `sha256:77f585114c32fbca283dc835b0596f4e52b51b4c6662d7810b2f4084f60a1873` | `pg_isready -h 127.0.0.1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"` |
| 큐 | `redis:8.10.2-alpine` (`REDIS_VERSION=8.10.2`) | amd64, arm64/v8 외 6종 | 39 / 39 MB | `sha256:3811787313eba226a2ef38658c6ccb91cd5e110edc89c37767de373120a0e5a0` | `redis-cli ping` |
| 프록시 | `caddy:2.11.6-alpine` (`CADDY_VERSION=v2.11.6`) | amd64, arm64/v8 외 4종 | 25 / 24 MB | `sha256:c776e0c6413b544d0459665e54ec7b8b2a15000c0cbee8b254da0067b1d184ff` | `wget -q -O /dev/null http://127.0.0.1/healthz` (busybox) |
| e2e | `mcr.microsoft.com/playwright:v1.63.0-noble` | amd64, arm64 | 956 / 937 MB | `sha256:eff16c30e6f3f4af0a03fa4b706120d5e9b0891c344a27d64559aff5900a4a27` | (일회성) |
| 대안 프록시 | `nginx:1.30-alpine` (`NGINX_VERSION=1.30.5`) | amd64, arm64/v8 외 | 26 / 26 MB | — | — |

- digest는 Docker Hub 이미지의 경우 `mirror.gcr.io`가 내려준 값이다. node, python, postgres는 Docker Hub의 digest와 일치하는 것을 확인했다. redis와 caddy는 rate limit 때문에 Hub와 직접 대조하지 못했다(UNKNOWN). 비교용 버전: `postgres:17-alpine`은 17.11, `node:24-alpine`은 압축 62 MB다. `python:3.12-slim` 같은 부동(floating) 태그는 trixie를 가리킨다.

### 7.3 이미지별 주의점
- **[VERIFIED E19/E11] PostgreSQL 18.** `postgres:18-alpine`은 `PGDATA=/var/lib/postgresql/18/docker`, `VOLUME /var/lib/postgresql`이다(17은 `/var/lib/postgresql/data`). 따라서 **named volume은 `/var/lib/postgresql`에 마운트한다.**
- **[VERIFIED]** `postgres:18.6-alpine /usr/local/bin/docker-entrypoint.sh:295-297`(`docker_temp_server_start`): initdb 중에 띄우는 임시 서버는 `listen_addresses=''`, 즉 unix socket만 연다. TCP로 묻는 `pg_isready -h 127.0.0.1`은 최종 서버가 뜬 뒤에만 성공한다.
- **[VERIFIED E11]** alpine postgres 로그에 `no usable system locales were found`가 나온다. ICU/locale collation이 필요하지 않으면 무시해도 된다.
- **[VERIFIED E19/E11]** `redis:8-alpine` 이미지 설정에는 VOLUME이 없다. `--appendonly yes`와 `/data` 마운트를 명시한다. 로그에 `Redis does not require authentication…` 경고가 나오므로 포트를 publish하지 않는다.
- **[VERIFIED E19/E11]** `caddy:2-alpine`은 `XDG_DATA_HOME=/data`, `XDG_CONFIG_HOME=/config`이고 80/443/2019를 노출한다. 로컬 HTTP 전용이면 `admin off`와 `auto_https off`를 쓴다. 이때 `HTTP/2 skipped because it requires TLS` 경고는 정상이다. CRLF 줄 끝의 Caddyfile도 `caddy adapt`가 정상 처리한다(Windows checkout, E18).

### 7.4 Compose 기능 **[VERIFIED E11, Compose v5.1.2]**
- `env -i PATH=$PATH HOME=$HOME docker compose config -q`는 `.env` 없이 통과하고, 모든 `${VAR:-default}`가 기본값으로 해석된다.
- `depends_on`의 `condition: service_healthy`와 `service_completed_successfully`(일회성 `migrate`가 exit 0)를 썼다. `docker compose up -d --wait`는 migrate가 Exited(0)여도 **exit 0**으로 끝난다.
- `profiles: ["gpu"]`인 `worker_gpu`는 기본 `config --services`와 `up`에서 제외된다. NVIDIA가 없는 호스트에서 `--profile gpu up`을 하면 `could not select device driver "nvidia" with capabilities: [[gpu]]`가 나고, 기본 스택에는 영향이 없다.
- YAML anchor와 merge key(`x-*`, `<<:`)가 동작한다. CMD-SHELL healthcheck에서는 `$${VAR}`로 escape해야 컨테이너 env가 쓰인다.

### 7.5 Docker Hub 의존성
- **[VERIFIED E11]** 이 호스트의 daemon pull이 `toomanyrequests: You have reached your unauthenticated pull rate limit`에 걸렸다. 레지스트리 응답 헤더는 `ratelimit-limit: 100;w=3600`이었다. 일반 사용자는 이미지 5–6개만 받으므로 보통 문제가 되지 않지만, 공유 IP나 CI에서는 실패할 수 있다.
- **[VERIFIED E11]** 그래서 compose가 모든 base image를 `${NODE_IMAGE:-…}` 같은 변수(Dockerfile에서는 `ARG …_IMAGE`)로 받게 했다. proof는 이 변수만 `mirror.gcr.io/library/...`로 바꿔서 기동했다.
- **[VERIFIED E18]** `# syntax=docker/dockerfile:1` 줄이 있으면 BuildKit이 빌드 때마다 `docker.io/docker/dockerfile:1`을 resolve한다. 이 줄을 빼도 BuildKit 0.29 내장 frontend가 `RUN --mount=type=cache/bind`를 처리했다. **[INFERRED]** 내장 frontend가 이 기능을 지원하려면 Docker Engine 23 / BuildKit 0.11 이상이 필요하다.

## 8. Proof stack (Q8)

구성(`/tmp/vf-stack/proof`)은 `proxy`(Caddy :80 → host `127.0.0.1:18080`), `web`(Next standalone), `api`(FastAPI/uvicorn), `migrate`(alembic 일회성), `worker_cpu`(rq worker), `redis`, `postgres`의 7개 서비스와 `worker_gpu`(profile `gpu`, 기동 안 함)다. §10의 스니펫은 이 proof 파일을 저장소 경로와 이름에 맞춰 옮긴 것이다.

| 측정 | 값 | 근거 |
|---|---|---|
| cold build (BuildKit cache 없음, node base만 로컬에 있음, 4개 이미지 병렬) | **49초** | E11 VERIFIED |
| `up -d --wait` (postgres/redis/caddy pull 포함) → 상시 6개 healthy, migrate Exited(0) | **30초** | E11 VERIFIED |
| api만 증분 rebuild / api+worker rebuild | 12초 / 14초 | E11, E15 VERIFIED |
| `down -v` 후 `up -d --build --wait` (warm cache) | 33초, exit 0 | E11 VERIFIED |
| 이미지 크기 (`docker images`, arm64) | web **291 MB**, api **727 MB**, worker **699 MB** (단일 프로젝트 web 310 MB) | E11, E20 VERIFIED |
| worker venv 554 MB 구성 | pyarrow 142, transformers 111, pandas 73, numpy 39+27(libs), sqlalchemy 28.5, psycopg_binary.libs 15.4, tokenizers 11.0, hf_xet 10.9 MB | E11 VERIFIED |
| idle 메모리 | api 70, worker 56, web 41, postgres 30, redis 16, caddy 14 MiB | E11 VERIFIED (`docker stats`) |

curl 결과 **[VERIFIED E11]**: `GET /`는 200 text/html(title 확인), `/api/v1/health`는 `{"status":"ok"}`, `/api/v1/ready`(DB와 Redis 확인)는 `{"status":"ready"}`, `/_next/static/*.js`는 200이었다. API 응답 헤더에는 `Server: uvicorn`, `Via: 1.1 Caddy`가 붙는다.

SSE 도착 시각(Caddy 경유, `interval=1s`) **[VERIFIED E12]**:

| 엔드포인트 | 응답 헤더 | 이벤트 도착 시각 |
|---|---|---|
| FastAPI native (`fastapi.sse.EventSourceResponse`) | `text/event-stream; charset=utf-8`, `Cache-Control: no-cache`, `X-Accel-Buffering: no`, chunked | +0.04 / 1.05 / 2.05 / 3.05초 |
| sse-starlette 3.5.0 | `Cache-Control: no-store`, `X-Accel-Buffering: no` | +0.01 / 1.02 / 2.01 / 3.02초 |
| FastAPI native + `Accept-Encoding: gzip` (Caddy `encode zstd gzip` 활성) | `Content-Encoding: gzip` | +0.01 / 1.01 / 2.01 / 3.01초 |

- **[VERIFIED]** Caddy는 응답 Content-Type이 `text/event-stream`이면 `flushInterval`을 -1(즉시)로 둔다(`caddy v2.11.6 modules/caddyhttp/reverseproxy/streaming.go:271-292 (Handler.flushInterval)`). `encode`도 SSE면 헤더를 즉시 쓰고 Flush 때 encoder까지 flush한다(`modules/caddyhttp/encode/encode.go:291-298, 320-345`). 그래서 **`flush_interval` 설정이 필요 없다.**
- **[VERIFIED]** FastAPI 0.142.2는 generator가 15초 동안 조용하면 `: ping` keepalive를 끼워 넣고(`fastapi/sse.py:237,241`, `fastapi/routing.py:575-628`), `Cache-Control: no-cache`와 `X-Accel-Buffering: no`를 설정한다(`routing.py:664-666`). sse-starlette의 기본값은 ping 15초, `Cache-Control: no-store`, `X-Accel-Buffering: no`, 줄 구분자 `\r\n`이다(`sse_starlette/sse.py:264-318`).
- **[VERIFIED `fastapi_sse_204.py`]** FastAPI native SSE에서 `data`는 항상 JSON으로 직렬화된다(문자열도 따옴표가 붙음). 미리 만든 문자열은 `raw_data`로 보낸다. generator 엔드포인트에서 **첫 yield 전에 `HTTPException(204)`를 던지면 204가 되지 않고 `ExceptionGroup`이 난다.** 재연결을 멈추는 204가 필요하면 generator가 아닌 엔드포인트에서 `Response(status_code=204)`나 sse-starlette `EventSourceResponse`를 반환한다(E14에서 sse-starlette로 확인).
- **[VERIFIED E18]** nginx 1.30 비교: 기본 설정과 `proxy_ignore_headers X-Accel-Buffering` 두 경우 모두 이 테스트에서는 이벤트가 1초 간격으로 도착했다. **[INFERRED]** nginx를 쓴다면 그래도 `proxy_buffering off`와 `proxy_read_timeout`을 명시하는 것이 안전하다.

분석 job과 재개 **[VERIFIED E13]**: `POST /api/v1/analyses`가 202를 주고 DB에 QUEUED를 쓴 뒤 RQ에 넣는다. worker는 `job.meta`를 갱신하고 Redis Stream에 XADD한다. `GET …/events`는 XREAD BLOCK으로 스트림을 읽어 SSE로 내보내고, `completed`에서 끝난다. `Last-Event-ID: <3번째 id>`로 다시 연결하면 4–6번째 이벤트만 재생된다. 이벤트 id는 Redis Stream id(`1791117879768-0` 형식)다.

정리 **[VERIFIED]**: `docker compose --profile gpu down -v --rmi local`로 컨테이너, 볼륨, 네트워크, 빌드 이미지를 지웠다. pull한 base image와 수동 태그 이미지는 `docker rmi`로, 이번 세션의 build cache record는 `docker buildx prune --filter id=…`로 지웠다. 마지막에 `docker system df`가 시작 전과 같은 상태(이미지 2개, build cache 13개 169.6 MB, 볼륨 29개, 컨테이너 0개)인 것을 확인했다.

## 9. arm64 호스트에서 linux/amd64 빌드 (Q9)

- **[VERIFIED E17]** OrbStack builder는 `linux/amd64 (+2), linux/arm64, …` 플랫폼을 지원한다(Rosetta 사용).
  - `docker buildx build --platform linux/amd64 --target worker-cpu … --load`: **27초**. 이 중 의존성 `uv sync`가 11.0초이고, 모든 wheel이 받기만 하면 되는 binary라 컴파일이 없다.
  - web(`next build` 포함) amd64: **49초**(pnpm install 13.0초, next build 26.3초). 실행하면 `node -p process.arch`가 `x64 v24.21.0`이고 HTTP 200을 반환한다.
- **[VERIFIED E17]** 컨테이너 안에서 실제 tokenizer로 전체 스캔(4,656 row, chosen과 rejected 9,312 시퀀스, system+user+assistant template)을 돌렸다. 두 아키텍처 모두 `max_len 2276`, `total_tokens 1,886,384`로 같았고 torch 미설치, `psycopg impl: binary`였다. 시간은 amd64(에뮬레이션) 11.2초, arm64(네이티브) 6.9초, `python:3.13-slim` arm64도 같은 결과였다. 이 값은 smoke test이며 TRL 전처리 길이가 아니다. golden 값은 [`example-model-dataset.md`](example-model-dataset.md)를 쓴다.
- **[INFERRED]** 최종 사용자는 `docker compose up --build`가 호스트 아키텍처용으로 빌드하므로 cross-build가 필요 없다. cross-build는 CI 검증(goals Q5)이나 멀티 아키텍처 이미지 배포에만 쓴다. QEMU가 없는 Linux CI에서는 `docker/setup-qemu-action`(binfmt 등록)이 필요하고, Rosetta 없는 QEMU 에뮬레이션은 위 시간보다 느릴 것이다(측정 안 함).

## 10. 권장 고정 버전과 검증한 설정 (Q10)

### 10.1 고정 버전

| 범주 | 패키지 / 이미지 | 버전 | 비고 |
|---|---|---|---|
| runtime | Node.js | **24.21.0** (LTS) | 26은 2026-10-28 LTS 전환 뒤 별도 검증한다(corepack 없음) |
| 패키지 관리 | pnpm | **10.34.6** (`packageManager`) | pnpm 10.15.0이 자동으로 이 버전으로 전환한다(manage-package-manager-versions, E11 확인). pnpm 11.28.2는 `minimumReleaseAge` 정책으로 frozen install을 거부했고, 12.9.1은 "lockfile not up to date"(packageManagerDependencies)로 실패했다(E3 사본의 lockfile로 `npx pnpm@<ver> install --frozen-lockfile --ignore-scripts` 실행, 10.34.6은 성공). lockfileVersion `9.0` |
| web | next / eslint-config-next | 16.3.8 / 16.3.8 | |
| web | react / react-dom / **react-is** | 19.3.0 / 19.3.0 / **19.3.0** | react-is는 recharts용 (§3.3) |
| web | typescript | **6.0.3** | 7.x 금지 (§1.3) |
| web | @types/react / @types/react-dom / @types/node | 19.3.0 / 19.3.0 / 24.19.1 | |
| web | eslint | **9.39.5** | 10.x 금지 (§1.4) |
| web | tailwindcss / @tailwindcss/postcss | 4.3.3 / 4.3.3 | |
| web | radix-ui | 1.6.7 | |
| web | @tanstack/react-query | 5.104.1 | |
| web | react-hook-form / zod / @hookform/resolvers | 7.89.0 / 4.6.5 / 5.9.1 | |
| web | recharts | 3.10.1 | |
| web (dev) | openapi-typescript (+선택 openapi-fetch) | 7.13.0 (0.17.0) | peer 경고(`typescript ^5.x`)만 남 |
| test | vitest / vite / @vitejs/plugin-react | 5.0.3 / 8.3.2 / 6.1.1 | |
| test | jsdom / @testing-library/react / dom / jest-dom / user-event | 30.1.2 / 16.3.3 / 10.4.2 / 7.0.1 / 14.6.7 | happy-dom 20.14.5도 가능 |
| e2e | @playwright/test + 이미지 | 1.63.0 + `mcr.microsoft.com/playwright:v1.63.0-noble` | 버전을 일치시킨다 |
| Python | CPython | **3.12.15** (`python:3.12.15-slim-trixie`) | 3.13.16 호환 확인. 학습 환경 venv와 저장소 `.python-version`이 3.12 |
| Python | fastapi / starlette / pydantic / pydantic-settings | 0.142.2 / 1.7.0 / 2.13.5 / 2.15.0 | |
| Python | uvicorn[standard] / sse-starlette | 0.54.0 / 3.5.0 | SSE는 FastAPI native만으로도 충분 |
| Python | sqlalchemy / psycopg[binary] / alembic | 2.1.3 / 3.3.6 / 1.20.0 | |
| Python | rq / redis | 2.12.0 / 8.1.0 | |
| Python | transformers / tokenizers / huggingface_hub / datasets / pyarrow / safetensors | 5.18.0 / 0.23.2 / 1.33.0 / 5.0.1 / 25.0.1 / 0.8.0 | 학습 환경과 같게 둔다 |
| 도구 | uv (로컬 / 이미지) | ≥0.11.19 / `ghcr.io/astral-sh/uv:0.12.23` | lock 호환 확인 |
| 인프라 | postgres / redis / caddy | `18.6-alpine` / `8.10.2-alpine` / `2.11.6-alpine` | |

### 10.2 web 설정 파일 (VERIFIED E3, E6, E11)

`apps/web/package.json` (scripts와 버전)

```json
{
  "name": "@vramforge/web",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "eslint",
    "typecheck": "next typegen && tsc --noEmit",
    "test": "vitest run",
    "gen:api": "openapi-typescript ../../docs/api/openapi.json -o lib/api/schema.d.ts"
  },
  "dependencies": {
    "@hookform/resolvers": "5.9.1", "@tanstack/react-query": "5.104.1", "next": "16.3.8",
    "radix-ui": "1.6.7", "react": "19.3.0", "react-dom": "19.3.0", "react-hook-form": "7.89.0",
    "react-is": "19.3.0", "recharts": "3.10.1", "zod": "4.6.5"
  },
  "devDependencies": {
    "@tailwindcss/postcss": "4.3.3", "@testing-library/dom": "10.4.2", "@testing-library/jest-dom": "7.0.1",
    "@testing-library/react": "16.3.3", "@testing-library/user-event": "14.6.7", "@types/node": "24.19.1",
    "@types/react": "19.3.0", "@types/react-dom": "19.3.0", "@vitejs/plugin-react": "6.1.1",
    "eslint": "9.39.5", "eslint-config-next": "16.3.8", "jsdom": "30.1.2", "openapi-typescript": "7.13.0",
    "tailwindcss": "4.3.3", "typescript": "6.0.3", "vite": "8.3.2", "vitest": "5.0.3"
  }
}
```

`pnpm-workspace.yaml` (lockfile이 있는 위치에 둔다)

```yaml
packages:            # 루트 workspace일 때만. apps/web 단독 프로젝트면 이 키를 뺀다
  - apps/*
ignoredBuiltDependencies:   # create-next-app 16.3.8이 생성한 값. 이 상태로 빌드·실행 정상(E11)
  - sharp
  - unrs-resolver
```

`next.config.ts`

```ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  poweredByHeader: false,
  // rewrites는 빌드 시점에 server.js에 고정된다. 프로덕션의 /api 라우팅은 Caddy가 맡고,
  // 이 rewrite는 프록시 없이 `next dev`를 쓸 때만 의미가 있다.
  async rewrites() {
    const target = process.env.API_PROXY_TARGET;
    return target ? [{ source: "/api/:path*", destination: `${target}/api/:path*` }] : [];
  },
};
export default nextConfig;
```

`postcss.config.mjs`와 `app/globals.css` 앞부분

```js
export default { plugins: { "@tailwindcss/postcss": {} } };
```

```css
@import "tailwindcss";
@custom-variant dark (&:where(.dark, .dark *));   /* class 기반 다크 모드 */
@theme inline {
  --font-sans: system-ui, -apple-system, "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR", sans-serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, monospace;
}
```

`eslint.config.mjs`는 create-next-app 16.3.8이 만든 것을 그대로 쓴다(`eslint-config-next/core-web-vitals` + `/typescript` + `globalIgnores`). 이 설정의 `react-hooks/refs` 규칙(React Compiler 계열)이 render 중 `ref.current` 접근을 실제 오류로 잡는 것도 확인했다(E6).

### 10.3 client provider (VERIFIED E6/E14)

```tsx
"use client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Tooltip } from "radix-ui";
import { useState, type ReactNode } from "react";

export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { staleTime: 5_000, retry: 1 } } }));
  return (
    <QueryClientProvider client={client}>
      <Tooltip.Provider delayDuration={300}>{children}</Tooltip.Provider>
    </QueryClientProvider>
  );
}
```

### 10.4 Vitest 설정 (VERIFIED E6: jsdom 7/7 통과)

`vitest.config.mts`

```ts
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

export default defineConfig({
  plugins: [react()],
  resolve: { tsconfigPaths: true },          // Vite 8 내장. vite-tsconfig-paths 불필요
  test: {
    environment: "jsdom",
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    css: false,
  },
});
```

`tests/setup.ts`

```ts
import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => cleanup()); // globals=false라 RTL 자동 cleanup이 등록되지 않음

// jsdom에 없는 API: Radix Select/Popper가 사용
if (!Element.prototype.hasPointerCapture) {
  Element.prototype.hasPointerCapture = () => false;
  Element.prototype.setPointerCapture = () => {};
  Element.prototype.releasePointerCapture = () => {};
}
if (!Element.prototype.scrollIntoView) Element.prototype.scrollIntoView = () => {};
```

EventSource 대체 패턴(jsdom/happy-dom에 없음, E6 통과)

```ts
class FakeES extends EventTarget {
  static CLOSED = 2; readyState = 0;
  onopen: (() => void) | null = null; onerror: (() => void) | null = null;
  constructor(public url: string) { super(); instances.push(this); }
  close() { this.readyState = 2; }
  emit(type: string, data: unknown, id: string) {
    this.dispatchEvent(new MessageEvent(type, { data: JSON.stringify(data), lastEventId: id }));
  }
}
vi.stubGlobal("EventSource", FakeES);
```

### 10.5 Playwright 설정 (VERIFIED E14)

```ts
import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  use: { baseURL: process.env.BASE_URL ?? "http://127.0.0.1:8080", trace: "off" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
});
```

compose에 둘 경우의 예시(**INFERRED**: 실험에서는 같은 설정을 `docker run`으로 실행했다)

```yaml
  e2e:
    profiles: ["e2e"]
    image: mcr.microsoft.com/playwright:v1.63.0-noble
    ipc: host
    working_dir: /e2e
    environment: { BASE_URL: http://proxy, CI: "1" }
    volumes: ["./tests/e2e:/e2e"]
    command: ["sh", "-c", "npm ci && npx playwright test"]
    depends_on: { proxy: { condition: service_healthy } }
```

### 10.6 uv workspace root (VERIFIED E9; 저장소의 기존 root 형식은 E10)

```toml
# virtual root: [project]가 없다. 저장소처럼 [project] + [tool.uv] package = false 형식도 같은 방식으로 동작한다.
[tool.uv.workspace]
members = ["packages/estimator", "services/api", "services/worker_cpu"]

[tool.uv]
required-version = ">=0.11.19"

[dependency-groups]
dev = ["pytest==9.1.1", "ruff==0.16.10", "httpx==0.28.1"]
```

멤버에서 workspace 의존성을 쓰는 방법(api 예시): `dependencies = ["vramforge-estimator", …]`와 `[tool.uv.sources] vramforge-estimator = { workspace = true }`.

### 10.7 Dockerfile (VERIFIED E11, E17, E20, E21; 경로만 저장소 구조 `infra/docker/*`에 맞춤)

`infra/docker/python.Dockerfile` (build context는 저장소 루트이고, target은 `api`/`worker`)

```dockerfile
ARG PYTHON_IMAGE=python:3.12.15-slim-trixie
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.12.23

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS build
COPY --from=uv /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
ARG PACKAGE
# 1) 서드파티 의존성만 설치: uv.lock 또는 pyproject가 바뀔 때까지 cache를 재사용한다
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=packages/estimator/pyproject.toml,target=packages/estimator/pyproject.toml \
    --mount=type=bind,source=services/api/pyproject.toml,target=services/api/pyproject.toml \
    --mount=type=bind,source=services/worker_cpu/pyproject.toml,target=services/worker_cpu/pyproject.toml \
    uv sync --frozen --no-dev --no-install-workspace --package "${PACKAGE}"
# 2) workspace 멤버를 non-editable로 설치: 런타임에 /src가 필요 없다
COPY pyproject.toml uv.lock README.md ./
COPY packages packages
COPY services services
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package "${PACKAGE}"

FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --system --uid 10001 --create-home --home-dir /home/app app \
 && mkdir -p /data/artifacts /data/uploads /data/hf && chown -R app:app /data
COPY --from=build /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 HF_HOME=/data/hf
WORKDIR /app
USER app

FROM runtime AS api
COPY --chown=app:app services/api/alembic.ini /app/alembic.ini
COPY --chown=app:app services/api/migrations /app/migrations
EXPOSE 8000
CMD ["uvicorn", "vramforge_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]

FROM runtime AS worker
CMD ["rq", "worker", "--worker-ttl", "60", "--with-scheduler", "--serializer", "json", "analysis"]
```

- proof의 상시 worker는 기본 pickle 직렬화(`rq worker --worker-ttl 60 --with-scheduler analysis`)로 돌았다. `--serializer json`은 같은 이미지에서 burst worker(`rq worker --burst -S json jsonq`)로 따로 확인했다(E15). 이 옵션을 쓰면 API 쪽 Queue, `Job.fetch`, `send_stop_job_command`도 모두 `JSONSerializer`로 맞춰야 한다.
- **[VERIFIED E21]** 위 두 Dockerfile 블록을 이 문서에서 그대로 추출해 proof 트리에서 빌드했다(api, worker, web 합계 75초). api 이미지는 `vramforge_api.main` import와 `alembic 1.20.0`이 동작했고 `/data`, `/data/hf`가 `app` 소유, `HF_HOME=/data/hf`였다. worker 이미지는 `rq 2.12.0`이고 torch가 없었으며 CMD가 위와 같았다. web 이미지는 HTTP 200을 반환했다. 단 실행 조건이 저장소와 다르다. proof 트리의 패키지 이름은 `vramforge-worker-cpu`였고 build backend는 `uv_build`였다.
- 저장소 멤버는 `hatchling`을 쓴다. 마지막 `uv sync --no-editable`이 hatchling을 내려받아 wheel을 만드는 경로는 **INFERRED**다. 루트 `README.md` bind는 파일이 있으면 동작한다(E21). 저장소 root의 `readme = "README.md"` 때문에 이 bind가 꼭 필요한지는 **INFERRED**다. 모듈 경로(`vramforge_api.main:app`, `services/api/migrations`)는 실제 구현에 맞춘다.
- `/data`를 이미지에서 미리 만들어 소유권을 `app`에 줘야 한다. 처음 붙는 named volume은 이미지의 디렉터리 소유권을 물려받는다. 디렉터리가 없으면 root 소유가 되어 uid 10001이 `Permission denied`를 받는다(E18 VERIFIED).

`infra/docker/web.Dockerfile` (루트 pnpm workspace일 때. 단독 프로젝트 변형은 아래 주석 참고)

```dockerfile
ARG NODE_IMAGE=node:24.21.0-trixie-slim

FROM ${NODE_IMAGE} AS base
ENV PNPM_HOME=/pnpm PATH=/pnpm:$PATH COREPACK_ENABLE_DOWNLOAD_PROMPT=0 NEXT_TELEMETRY_DISABLED=1
RUN corepack enable
WORKDIR /repo

FROM base AS deps
COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/web/package.json apps/web/
RUN --mount=type=cache,id=pnpm-store,target=/pnpm/store \
    pnpm install --frozen-lockfile --filter @vramforge/web...

FROM deps AS build
COPY apps/web apps/web
RUN pnpm --filter @vramforge/web build

FROM ${NODE_IMAGE} AS runner
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
WORKDIR /app
COPY --from=build --chown=node:node /repo/apps/web/.next/standalone ./
COPY --from=build --chown=node:node /repo/apps/web/.next/static ./apps/web/.next/static
COPY --from=build --chown=node:node /repo/apps/web/public ./apps/web/public
USER node
EXPOSE 3000
CMD ["node", "apps/web/server.js"]
# 단독 프로젝트(apps/web/pnpm-lock.yaml, context=apps/web)일 때(E20 VERIFIED):
#   COPY .next/standalone ./ ; COPY .next/static ./.next/static ; COPY public ./public ; CMD ["node","server.js"]
```

### 10.8 Caddyfile, worker healthcheck, compose (VERIFIED E11–E16)

`infra/proxy/Caddyfile`

```caddyfile
{
	admin off
	auto_https off
}

:80 {
	encode zstd gzip
	handle /healthz {
		respond "ok" 200
	}
	handle /api/* {
		reverse_proxy api:8000      # text/event-stream은 Caddy가 자동으로 즉시 flush
	}
	handle {
		reverse_proxy web:3000
	}
}
```

worker healthcheck 모듈(같은 컨테이너 hostname의 worker가 등록돼 있고 heartbeat가 120초 안에 있으면 0). 기본 `worker_ttl`에서는 idle heartbeat가 최대 405초 간격이라 unhealthy가 된다. 그래서 `--worker-ttl 60`과 함께 쓴다(E16).

```python
import os, socket, sys
from datetime import datetime, timezone
from redis import Redis
from rq import Worker

def main() -> int:
    conn = Redis.from_url(os.environ.get("REDIS_URL", "redis://redis:6379/0"), socket_timeout=3)
    host, now = socket.gethostname(), datetime.now(timezone.utc)
    for w in Worker.all(connection=conn):
        hb = w.last_heartbeat
        if w.hostname == host and hb and (now - hb.replace(tzinfo=timezone.utc)).total_seconds() < 120:
            return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())
```

`compose.yaml` (proof를 저장소 서비스 이름과 경로에 맞춘 것. 구조, 기본값, healthcheck, depends_on은 E11에서 확인했다. proof와 다른 부분은 **INFERRED**다: 서비스·모듈 이름(`worker`, `vramforge_worker.healthcheck`), `infra/` 경로, [`architecture.md`](../architecture.md) §1의 `app-data:/data`와 `./local-sources:/sources/local:ro` 마운트. GPU profile 서비스는 §7.4처럼 별도로 둔다)

```yaml
x-python-build-args: &python-build-args
  PYTHON_IMAGE: ${PYTHON_IMAGE:-python:3.12.15-slim-trixie}
  UV_IMAGE: ${UV_IMAGE:-ghcr.io/astral-sh/uv:0.12.23}
x-app-env: &app-env
  DATABASE_URL: postgresql+psycopg://${POSTGRES_USER:-vramforge}:${POSTGRES_PASSWORD:-vramforge}@postgres:5432/${POSTGRES_DB:-vramforge}
  REDIS_URL: redis://redis:6379/0
  RQ_REDIS_URL: redis://redis:6379/0
x-hc: &hc { interval: 5s, timeout: 3s, retries: 30, start_period: 10s }

services:
  proxy:
    image: ${CADDY_IMAGE:-caddy:2.11.6-alpine}
    ports: ["${VRAMFORGE_BIND:-127.0.0.1}:${VRAMFORGE_PORT:-8080}:80"]
    volumes: ["./infra/proxy/Caddyfile:/etc/caddy/Caddyfile:ro"]
    depends_on: { web: { condition: service_healthy }, api: { condition: service_healthy } }
    healthcheck: { <<: *hc, test: ["CMD", "wget", "-q", "-O", "/dev/null", "http://127.0.0.1/healthz"] }
    restart: unless-stopped
  web:
    build: { context: ., dockerfile: infra/docker/web.Dockerfile, args: { NODE_IMAGE: "${NODE_IMAGE:-node:24.21.0-trixie-slim}" } }
    healthcheck: { <<: *hc, test: ["CMD", "node", "-e", "fetch('http://127.0.0.1:3000/').then(r=>process.exit(r.ok?0:1),()=>process.exit(1))"] }
    restart: unless-stopped
  migrate:
    build: &api-build
      context: .
      dockerfile: infra/docker/python.Dockerfile
      target: api
      args: { <<: *python-build-args, PACKAGE: vramforge-api }
    environment: *app-env
    command: ["alembic", "-c", "/app/alembic.ini", "upgrade", "head"]
    depends_on: { postgres: { condition: service_healthy } }
    restart: "no"
  api:
    build: *api-build
    environment: *app-env
    volumes: ["app-data:/data", "./local-sources:/sources/local:ro"]
    depends_on: { migrate: { condition: service_completed_successfully }, redis: { condition: service_healthy } }
    healthcheck: { <<: *hc, test: ["CMD", "python", "-c", "import urllib.request as u,sys; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/api/v1/health',timeout=2).status==200 else 1)"] }
    restart: unless-stopped
  worker:
    build:
      context: .
      dockerfile: infra/docker/python.Dockerfile
      target: worker
      args: { <<: *python-build-args, PACKAGE: vramforge-worker }
    environment: *app-env
    volumes: ["app-data:/data", "./local-sources:/sources/local:ro"]
    depends_on: { migrate: { condition: service_completed_successfully }, redis: { condition: service_healthy } }
    healthcheck: { <<: *hc, test: ["CMD", "python", "-m", "vramforge_worker.healthcheck"] }
    stop_grace_period: 30s
    restart: unless-stopped
  redis:
    image: ${REDIS_IMAGE:-redis:8.10.2-alpine}
    command: ["redis-server", "--appendonly", "yes"]
    volumes: ["redis-data:/data"]
    healthcheck: { <<: *hc, test: ["CMD", "redis-cli", "ping"] }
    restart: unless-stopped
  postgres:
    image: ${POSTGRES_IMAGE:-postgres:18.6-alpine}
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-vramforge}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-vramforge}
      POSTGRES_DB: ${POSTGRES_DB:-vramforge}
    volumes: ["pg-data:/var/lib/postgresql"]      # PG 18 레이아웃
    healthcheck: { <<: *hc, test: ["CMD-SHELL", "pg_isready -h 127.0.0.1 -U \"$${POSTGRES_USER}\" -d \"$${POSTGRES_DB}\""] }
    restart: unless-stopped

volumes: { pg-data: {}, redis-data: {}, app-data: {} }
```

`.dockerignore` (루트)

```text
**/node_modules
**/.next
**/.venv*
**/__pycache__
**/*.pyc
**/.pytest_cache
**/.ruff_cache
**/coverage
**/test-results
**/playwright-report
**/*.tsbuildinfo
.git
.env
.env.*
local-sources
data
artifacts
uploads
```

### 10.9 SSE 엔드포인트 패턴 (VERIFIED E12–E14)

```python
from collections.abc import AsyncIterator
from typing import Annotated
from fastapi import Header
from fastapi.sse import EventSourceResponse, ServerSentEvent

@app.get("/api/v1/analyses/{aid}/events", response_class=EventSourceResponse)
async def analysis_events(
    aid: str, last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None
) -> AsyncIterator[ServerSentEvent]:
    r = aredis.Redis.from_url(settings.redis_url)        # redis.asyncio
    key, cursor = f"vf:analysis:{aid}:events", last_event_id or "0-0"
    try:
        while True:
            for _stream, entries in await r.xread({key: cursor}, block=10_000, count=100) or []:
                for entry_id, fields in entries:
                    cursor = entry_id.decode()
                    ev = fields[b"event"].decode()
                    yield ServerSentEvent(id=cursor, event=ev, raw_data=fields[b"data"].decode())
                    if ev in {"completed", "failed", "cancelled"}:
                        return
    finally:
        await r.aclose()
```

[`architecture.md`](../architecture.md) §7처럼 이벤트의 기준 저장소를 PostgreSQL `analysis_events`로 두는 경우에도 구조는 같다. 커서를 단조 증가 DB id로 바꾸고 XREAD BLOCK 대신 짧은 polling이나 Redis 알림을 쓰면 된다(**INFERRED**).

RQ enqueue 패턴(E11/E15)

```python
queue.enqueue(
    "vramforge_worker.tasks.analyze", analysis_id,          # 인자는 JSON 직렬화 가능한 id만
    job_id=analysis_id, unique=True, job_timeout="6h",       # 기본 180초를 반드시 덮어쓴다
    result_ttl=86_400, failure_ttl=7 * 86_400,
    on_stopped=Callback("vramforge_worker.tasks.on_stopped"),
)
```

---

## 구현 시사점 (Implementation implications)

### Web (`apps/web`)
- **W1.** 버전은 §10.1로 고정한다. 특히 `typescript 6.0.3`, `eslint 9.39.5`, `react-is 19.3.0`(직접 의존성)이다. TS 7로 올리는 시점은 typescript-eslint와 openapi-typescript가 TS 7 API를 지원한 뒤로 미룬다(typescript-eslint#10940 추적).
- **W2.** `typecheck` 스크립트는 `next typegen && tsc --noEmit`이다. 이것이 없으면 CI(fresh checkout)에서 `LayoutProps` 같은 route 타입이 정의되지 않아 실패한다(E18).
- **W3.** `next.config.ts`에는 `output: "standalone"`을 둔다. 프로덕션의 `/api` 라우팅은 Caddy가 하고 Next rewrites에 기대지 않는다(빌드 시 고정). 런타임 이미지에는 `HOSTNAME=0.0.0.0`과 `PORT=3000`을 두고, `.next/static`과 `public`을 복사하고, 빌드는 컨테이너 안에서 한다.
- **W4.** `next/font/google`을 쓰지 않는다(빌드 시 외부 네트워크가 필요함). 시스템 폰트 스택과 `tabular-nums`를 쓴다(plan §3.2).
- **W5.** 다크 모드는 `@custom-variant dark (&:where(.dark, .dark *));` + `<html class="dark">` 토글로 구현한다.
- **W6.** Vitest는 §10.4 설정과 polyfill을 쓴다. EventSource는 `vi.stubGlobal`로 대체한다. Recharts 테스트는 고정 크기 차트로 한다.
- **W7.** REST 타입은 `openapi-typescript 7.13.0`으로 `docs/api/openapi.json`에서 생성한다(`lib/api/schema.d.ts`). 클라이언트는 `openapi-fetch`(선택)나 TanStack Query의 queryFn에서 `paths` 타입을 쓴다.
- **W8.** **SSE 이벤트 payload 타입은 OpenAPI 생성물에 들어가지 않는다**(`"text/event-stream": unknown`). 다음 중 하나를 고른다. (a) SSE payload Pydantic 모델을 일반 응답 모델로도 노출해 `components.schemas`에 넣는다. (b) Pydantic `model_json_schema()` 결과에서 TS 타입을 따로 만든다. (c) TS 쪽에 Zod 스키마를 두고 수신 시 검증한다. 어느 쪽이든 Python 모델이 기준이다.
- **W9.** SSE client는 네이티브 `EventSource(url)`를 쓴다. 이름 있는 이벤트(`progress`, `partial_result`, `warning`, `completed`, `failed`, `cancelled`)는 `addEventListener`로 받고, 종료 이벤트에서는 `es.close()`를 호출한다. 새로고침 복구는 브라우저의 `Last-Event-ID` 자동 전송과 `GET /api/v1/analyses/{id}` 조회를 함께 쓴다(plan §15.3). EventSource는 헤더를 붙일 수 없으므로 인증은 same-origin cookie로 한다.

### Python (`packages/`, `services/`)
- **P1.** Docker에서는 `uv sync --frozen --no-dev --package <member>`로 멤버 하나만 설치한다. `parity` group(torch, trl)이 이미지에 들어가지 않는 것을 확인했다(E10). 이미지에는 `UV_PYTHON_DOWNLOADS=0`을 두고 시스템 Python을 쓴다.
- **P2.** tokenizer 경로는 torch 없이 동작한다. `apply_chat_template(...)`의 반환값은 **`BatchEncoding`**이므로 `["input_ids"]`로 꺼낸다. `return_tensors="pt"`는 쓰지 않는다. `TRANSFORMERS_NO_ADVISORY_WARNINGS=1`을 둔다. torch-free import guard 테스트(커밋 086c5c5)와 같은 방향이다.
- **P3.** starlette 1.7.0의 `TestClient`는 `httpx2`를 먼저 찾는다. 없으면 `httpx`로 동작하지만 `StarletteDeprecationWarning`을 낸다(`starlette==1.7.0 starlette/testclient.py:33-51`, VERIFIED). pytest에서 경고를 오류로 바꾼다면 `httpx2`(PyPI 2.13.1, 미설치·미검증)를 dev group에 넣거나 그 경고를 필터링한다.
- **P4.** compose healthcheck에는 의존성 없는 liveness 엔드포인트(항상 200)를 쓴다. DB·Redis·worker 상태를 담는 상세 `health`/`ready`를 healthcheck에 쓰면 worker 하나가 지연될 때 proxy까지 기동이 멈춘다.
- **P5.** Alembic은 일회성 `migrate` 서비스(`service_completed_successfully`)에서 실행한다. non-editable 설치에서는 `alembic.ini`와 `migrations/`를 이미지에 COPY한다(검증한 방식). `script_location = vramforge_api:migrations` 패키지 리소스 방식도 가능하지만(`alembic==1.20.0 alembic/util/pyfiles.py:52-74`), 소스 주석에 "zero tests"라고 적혀 있어서 **INFERRED**로 둔다.

### RQ·작업 제어
- **R1.** 모든 enqueue에 `job_timeout`을 명시한다(기본 180초). `job_id = analysis_id`, `unique=True`로 중복 enqueue를 막는다. `result_ttl`/`failure_ttl`도 명시하고, 결과의 기준 저장소는 PostgreSQL이다.
- **R2.** **pickle 대신 `JSONSerializer`**를 쓴다(plan §18 "임의 pickle 실행 금지"). Queue, `Job.fetch`, `send_stop_job_command(serializer=...)`, worker의 `--serializer json`을 모두 맞춘다. task 인자는 id 같은 JSON 값만 넘긴다.
- **R3.** 취소는 세 단계로 한다. (1) DB/Redis 취소 플래그를 세우고 worker가 shard 경계에서 확인해 checkpoint를 남기고 `CANCELLED`로 끝낸다(협조적, 기본). (2) queued 상태면 `job.cancel()`을 호출하고 DB도 직접 `CANCELLED`로 바꾼다. (3) 유예 시간 뒤에도 실행 중이면 `send_stop_job_command`로 강제 정지하고 `on_stopped` callback에서 DB를 갱신한다. worker는 forking `rq.Worker`를 쓴다(SimpleWorker 금지).
- **R4.** horse가 OOM 등으로 죽으면 `on_failure`가 실행되지 않는다. lease 만료 회수(plan §16.1–16.2)나 커스텀 worker의 `work_horse_killed_handler`로 DB 상태를 맞춘다. RQ의 FAILED 상태를 DB 상태의 근거로 쓰지 않는다.
- **R5.** worker는 `--worker-ttl 60 --with-scheduler`로 띄운다. retry interval에 scheduler가 필요하고, healthcheck 신선도에 짧은 heartbeat가 필요하기 때문이다. `--name`은 고정하지 않는다. `stop_grace_period`는 shard 하나를 마무리할 시간 이상으로 둔다. SIGTERM은 warm shutdown이고 그 뒤 SIGKILL이 온다.
- **R6.** 진행률은 두 곳에 쓴다. UI 폴링용 최신값은 `job.meta`+`save_meta`나 DB에, `Last-Event-ID` 재생용 이벤트는 단조 id를 가진 영속 저장소(Postgres `analysis_events` 또는 Redis Stream)에 쓴다. 재생이 안 되는 pub/sub만으로는 plan §15.3의 재연결 복구를 만족하지 못한다(INFERRED).

### SSE·프록시
- **S1.** Caddy에는 flush 설정이 필요 없다(VERIFIED). `encode zstd gzip`을 켜도 SSE는 이벤트마다 전달된다. nginx로 바꾸면 `proxy_buffering off`, `proxy_http_version 1.1`, 긴 `proxy_read_timeout`을 둔다(INFERRED).
- **S2.** FastAPI native SSE의 15초 ping이 idle 연결을 유지한다. 재연결을 멈추게 할 204가 필요한 경로(이미 종료된 분석)는 generator가 아닌 엔드포인트로 분기한다. native generator 안에서 HTTPException을 던지면 204가 되지 않는다.
- **S3.** SSE `data`에는 원문 row나 token을 넣지 않는다(plan §15.3). 같은 이유로 FastAPI native에서는 `data=`(자동 JSON)나 `raw_data=`(이미 직렬화한 JSON)를 일관되게 쓴다.

### Docker·Compose
- **D1.** 모든 값에 `${VAR:-default}` 기본값을 두고 `.env`가 없어도 기동되게 한다. 이미지도 `*_IMAGE` 변수로 받아 mirror로 바꿀 수 있게 한다. `env -i … docker compose config -q`를 CI 게이트에 넣는다.
- **D2.** base image 태그는 §7.2의 패치 버전까지 고정한다. 필요하면 digest도 고정한다. Dockerfile 첫 줄 `# syntax=`는 빼서 Docker Hub 의존을 하나 줄인다.
- **D3.** PostgreSQL 18 볼륨은 `/var/lib/postgresql`에 마운트하고 healthcheck는 `pg_isready -h 127.0.0.1`이다. Redis는 포트를 publish하지 않고 `--appendonly yes`와 `/data` 볼륨을 쓴다.
- **D4.** 앱 컨테이너는 non-root(uid 10001 / `node`)로 실행하고, 볼륨 마운트 지점(`/data/*`)은 이미지에서 미리 만들어 소유권을 준다.
- **D5.** 기동 확인은 `docker compose up -d --build --wait`로 한다(migrate가 exit 0이어도 exit 0). 기본 바인드는 `127.0.0.1:8080`이다(plan §18).
- **D6.** amd64 이미지 빌드 검증(goals Q5)은 `docker buildx build --platform linux/amd64 --target worker …`로 한다. arm64 호스트에서 worker 27초, web 49초였다. CI에서는 QEMU 설정이 필요하다.
- **D7.** **[INFERRED]** Windows checkout 대비로 `.gitattributes`에 `*.sh`, `Dockerfile*`, `Caddyfile` 등의 `eol=lf`를 둔다. 현재 설계에는 컨테이너 안에서 실행하는 셸 스크립트가 없으므로 그대로 유지하는 것이 가장 안전하다.

## 미확정 사항 (Open questions)

1. **Docker Desktop(macOS/Windows)과 네이티브 Linux amd64 호스트에서는 실행하지 않았다.** 검증한 것은 OrbStack(arm64)과 그 위의 amd64 에뮬레이션뿐이다. Windows(WSL2)의 bind mount, 줄 끝, 성능은 INFERRED다.
2. **Node 26 전환**(2026-10-28 LTS)은 미검증이다. corepack이 없으므로 web Dockerfile을 바꿔야 하고, Next 16.3.8, vitest, jsdom engines는 `>=26`을 허용한다.
3. **TypeScript 7 도입 시점**은 typescript-eslint의 TS≥7.1 지원과 openapi-typescript의 새 API 대응에 달려 있다. `@typescript/typescript6` 별칭(6.0.2, bin `tsc6`)을 쓰는 side-by-side 구성은 시험하지 않았다.
4. **이벤트 저장소 선택.** architecture.md는 Postgres `analysis_events`이고, proof는 Redis Stream으로 검증했다. Postgres 방식에서 push 지연(polling 주기, LISTEN/NOTIFY)과 부하는 측정하지 않았다.
5. **`VRAMFORGE_ACCESS_TOKEN` 모드와 EventSource.** 헤더를 붙일 수 없으므로 token을 cookie로 바꾸는 흐름이 필요하다. cookie 전송은 spec 기반 INFERRED이고 실제로 확인하지 않았다.
6. **redis·caddy digest의 Docker Hub 직접 대조**는 rate limit 때문에 하지 못했다(mirror 값만 기록). CI에서 Docker Hub 인증(`docker login`)이나 mirror를 쓸지는 결정하지 않았다.
7. **API 이미지 크기(727 MB).** api가 `vramforge-estimator[analysis]`에 의존해서 pyarrow, transformers, pandas가 들어간다. API에서 source inspection을 동기로 하려면 필요하지만, worker로 넘긴다면 extra 없이 슬림하게 만들 수 있다. 결정은 오케스트레이터 몫이다.
8. **pnpm 11/12 전환**은 보류했다. 11은 `minimumReleaseAge` 기본 정책 때문에 frozen install이 막혔고, 12는 lockfile 재생성이 필요했다. 공급망 정책을 일부러 도입할지 결정해야 한다.
9. **Redis 8 서버 이미지의 라이선스 조건**은 이번 조사에서 확인하지 않았다(UNKNOWN). Redis 프로토콜 호환 대안(Valkey 등)으로 RQ가 동작하는지도 시험하지 않았다(INFERRED: RQ는 Redis 명령만 사용).
10. **RQ `job_timeout=-1`(무제한) 같은 특수값**은 확인하지 않았다. 긴 스캔에는 명시적인 큰 값과 협조적 취소를 권장한다.
11. **hatchling 멤버의 `--no-editable` Docker 빌드**와 alembic 패키지 리소스 `script_location`은 저장소 구조로 실제 빌드하지 않았다(INFERRED).
12. **Docker Engine 최소 버전.** `--wait`, `service_completed_successfully`, 내장 BuildKit frontend의 `RUN --mount` 등의 하한을 오래된 엔진(예: 24.x)에서 시험하지 않았다. README에 최소 버전을 적을 때 근거가 필요하다.
