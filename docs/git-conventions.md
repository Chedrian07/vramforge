# Git 컨벤션 (멀티 에이전트 개발용)

> 이 문서는 VRAMForge 저장소에서 사람과 서브 에이전트가 **동시에** 작업할 때 지켜야 하는 Git 규칙이다.
> 모든 에이전트는 작업 시작 전에 이 문서를 읽고, 커밋 직전에 §11 체크리스트를 확인한다.

| 항목 | 값 |
|---|---|
| 적용 대상 | 오케스트레이터(메인 세션), 모든 서브 에이전트, 사람 기여자 |
| 기본 브랜치 | `main` (trunk-based) |
| 원격 | `origin` = `github.com/Chedrian07/vramforge` (**public 저장소**) |
| 상위 원칙 | 로컬 전용 `GIT_RULES.md` (gitignore 대상, 커밋 금지) |

## 1. 상위 원칙 (GIT_RULES.md 요약)

1. **핵심 코드를 수정할 때마다 커밋한다.** 작업을 몰아서 한 번에 커밋하지 않는다.
2. **핵심 태스크를 완수하면 `main`에 push할 수 있다.** push는 검증을 마친 오케스트레이터가 수행한다(§8).
3. **여러 파일을 한 번에 커밋하는 일을 막는다.** 커밋은 작은 논리 단위로 쪼갠다(§3).

`GIT_RULES.md`는 의도적으로 `.gitignore`에 포함된 로컬 전용 파일이다. 추적 대상으로 바꾸거나 내용을 다른 파일에 복사해 커밋하지 않는다.

## 2. 브랜치 전략

- 기본은 **`main` 단일 트렁크**다. 장기 브랜치를 만들지 않는다.
- 병렬 작업은 기본적으로 **같은 작업 트리에서 디렉터리 소유권(§6.1)으로 분리**한다.
- 오케스트레이터가 격리를 지정한 경우에만 git worktree + `agent/<label>` 브랜치를 사용한다(§9).
- 위험한 실험은 `exp/<topic>` 브랜치에서 하고, 결과는 오케스트레이터가 병합 여부를 결정한다.

## 3. 커밋 단위

**1 커밋 = 1 논리 변경.** 리뷰어가 커밋 하나만 보고 "무엇을 왜 바꿨는지" 이해할 수 있어야 한다.

| 규칙 | 기준 |
|---|---|
| 권장 크기 | 파일 1–4개 (구현 파일 + 그 테스트) |
| 상한 | 파일 6개. 초과 시 커밋을 쪼갠다 |
| 예외 | 도구가 생성한 scaffold, lockfile, 자동 생성 코드(API 타입 등), DB 마이그레이션 자동 생성분 → **별도 커밋**으로 분리하고 본문에 생성 명령을 적는다 |
| 커밋 시점 | 모듈 하나(또는 의미 있는 함수 묶음)가 구현되고 관련 테스트가 통과하면 즉시 커밋 |
| 누적 금지 | 커밋되지 않은 변경 파일이 6개를 넘기 전에 커밋한다 |
| 혼합 금지 | 기능 추가 + 무관한 리팩터링 + 포맷팅을 한 커밋에 섞지 않는다 |
| 문서 | 코드와 함께 바뀌어야 하는 문서는 같은 커밋 가능. 독립 문서는 별도 커밋 |

`main`은 항상 빌드·테스트 가능한 상태를 목표로 한다. 의도적으로 깨진 상태를 커밋하지 않는다. 미완성 기능은 호출되지 않는 경로에 두거나 명시적인 `NotImplementedError`/미지원 응답으로 막는다(가짜 수치 금지, plan.md §0-12).

## 4. 커밋 메시지

[Conventional Commits 1.0](https://www.conventionalcommits.org/) 형식을 따른다.

```text
<type>(<scope>): <subject>

<body — 무엇을, 왜. 한국어 또는 영어. 72자 내외 줄바꿈>

Milestone: M3
Agent: impl-memory
```

### 4.1 type

| type | 용도 |
|---|---|
| `feat` | 사용자/API에 보이는 기능 추가 |
| `fix` | 버그 수정 |
| `refactor` | 동작 변화 없는 구조 변경 |
| `perf` | 성능 개선 |
| `test` | 테스트 추가·수정만 |
| `docs` | 문서만 |
| `build` | 빌드 시스템, 의존성, Dockerfile |
| `ci` | CI 설정 |
| `chore` | 그 외 유지보수(scaffold, 설정 파일 등) |
| `style` | 포맷팅만 (동작·구조 변화 없음) |
| `revert` | 이전 커밋 되돌리기 |

### 4.2 scope

모듈 디렉터리와 1:1로 맞춘다. 목록에 없는 scope가 필요하면 오케스트레이터에게 보고한다.

| 영역 | scope |
|---|---|
| Python 코어 (`packages/estimator`) | `estimator`, `schemas`, `sources`, `inspection`, `preprocessing`, `scan`, `batching`, `architectures`, `trainers`, `memory`, `compatibility`, `calibration`, `exports`, `pipeline` |
| 서비스 | `api`, `worker`, `db`, `queue`, `security` |
| 프런트엔드 (`apps/web`) | `web`, `ui` |
| 인프라 | `docker`, `compose`, `proxy`, `ci` |
| 데이터·설정 | `profiles`, `fixtures` |
| 문서 | `docs`, `research` |
| 공통 | `deps`, `repo` |

### 4.3 subject·body·trailer

- subject는 **영어 명령형**, 소문자로 시작, 마침표 없음, 72자 이하. 예: `feat(memory): add phase-aware peak evaluation`
- body는 선택이지만 **"왜"가 코드만으로 드러나지 않으면 반드시 작성**한다. 한국어 사용 가능.
- trailer
  - `Agent: <label>` — 서브 에이전트 커밋에 **필수**. 오케스트레이터는 생략 가능.
  - `Milestone: M0`–`M6` — 기능 커밋에 권장 (plan.md §20).
  - `Refs: plan.md §9.4` — 사양 근거가 있는 경우 권장.
- 깨진 변경을 되돌릴 때는 `git revert <sha>`를 사용하고 메시지에 이유를 적는다.

### 4.4 예시

```text
좋음  feat(memory): evaluate peak as max over phases instead of sum
좋음  test(batching): cover DPO padding slots for 2 pairs (plan §19.1)
좋음  build(deps): add pyarrow to estimator analysis extra
나쁨  update files                       ← type/scope/의미 없음
나쁨  feat: implement estimator, api, ui ← 여러 영역을 한 커밋에
나쁨  fix(memory): fix                   ← 무엇을 고쳤는지 불명
```

## 5. 스테이징과 커밋 명령

**항상 경로를 명시한다.** 다른 에이전트가 같은 작업 트리에서 작업 중일 수 있다.

```bash
# 1) 내 파일만 스테이징
git add -- packages/estimator/src/vramforge_estimator/memory/engine.py \
           packages/estimator/tests/unit/test_engine.py

# 2) 스테이징 결과 확인 (파일 수, 의도하지 않은 파일 여부)
git diff --cached --stat -- packages/estimator/src/vramforge_estimator/memory/engine.py \
                            packages/estimator/tests/unit/test_engine.py

# 3) 경로를 지정해 커밋 (다른 사람이 스테이징한 파일이 섞이지 않도록 pathspec 필수)
git commit -m "feat(memory): add phase-aware peak evaluation" \
           -m "Agent: impl-memory" \
           -- packages/estimator/src/vramforge_estimator/memory/engine.py \
              packages/estimator/tests/unit/test_engine.py
```

`git commit -- <paths>`는 지정한 경로만 커밋하므로 다른 에이전트가 스테이징해 둔 파일을 끌고 오지 않는다. 새 파일은 반드시 먼저 `git add`한 뒤 같은 경로로 커밋한다.

### 5.1 금지 명령 (공유 작업 트리)

| 금지 | 이유 |
|---|---|
| `git add -A`, `git add .`, `git add -u`, `git commit -a` | 다른 에이전트의 미완성 파일을 쓸어 담는다 |
| `git stash`, `git reset`(모든 형태), `git clean`, `git checkout -- .`, `git restore .` | 다른 에이전트의 작업을 지운다 |
| `git rebase`, `git merge`, `git pull`, `git cherry-pick` | 작업 트리 전체를 바꾼다 (오케스트레이터 전용) |
| `git push`, `git tag` | 오케스트레이터 전용 (§8) |
| `git commit --amend`, `git commit --no-verify` | 남의 커밋을 바꾸거나 검증을 우회한다 |
| `git config` 변경, hooks 수정 | 저장소 전역 설정 변경 |
| `.git/index.lock` 삭제 | 다른 git 프로세스가 실행 중일 수 있다 (§6.3) |

자기 소유 파일의 변경을 되돌릴 때만 `git restore -- <내 파일 경로>`를 허용한다.

## 6. 멀티 에이전트 동시 작업 프로토콜

### 6.1 디렉터리 소유권

- 오케스트레이터는 작업을 배정할 때 **에이전트별 소유 경로**를 명시한다.
- 에이전트는 **소유 경로 안의 파일만** 생성·수정·커밋한다.
- 소유 경로 밖의 변경이 필요하면 수정하지 말고 최종 보고에 `CHANGE REQUEST: <파일> — <필요한 변경과 이유>`로 남긴다.
- 명시적으로 배정되지 않은 공용 파일은 오케스트레이터 소유다.

| 공용 파일 (기본 오케스트레이터 소유) |
|---|
| 루트 `pyproject.toml`, `uv.lock`, `compose.yaml`, `.env.example`, `.gitignore`, `README.md`, `plan.md` |
| `apps/web/package.json`, `pnpm-lock.yaml`, `.github/` |
| `docs/git-conventions.md`, `docs/goals.md` |

### 6.2 의존성 변경

1. 가능하면 이미 선언된 의존성만 사용한다.
2. 꼭 추가해야 하면 **자기 패키지의 manifest만** 수정한다 (`packages/estimator/pyproject.toml` 등).
3. lockfile 갱신은 상호 배제로 수행한다.

```bash
until mkdir .git/vramforge-deps.lock 2>/dev/null; do sleep 2; done
uv add --package vramforge-estimator "pyarrow>=25"   # 또는 pnpm add ...
git add -- packages/estimator/pyproject.toml uv.lock
git commit -m "build(deps): add pyarrow to estimator" -m "Agent: <label>" \
    -- packages/estimator/pyproject.toml uv.lock
rmdir .git/vramforge-deps.lock
```

4. 의존성 커밋은 기능 커밋과 분리한다.

### 6.3 `index.lock` 충돌

여러 에이전트가 동시에 커밋하면 `Unable to create '.git/index.lock': File exists`가 날 수 있다. **잠시 기다렸다가 재시도**한다.

```bash
for i in $(seq 1 10); do
  git commit -m "..." -- <paths> && break
  sleep $(( (RANDOM % 3) + 1 ))
done
```

10회 실패하면 lock 파일을 지우지 말고 오케스트레이터에게 보고한다.

### 6.4 테스트와 공유 상태

- 다른 에이전트의 미완성 코드 때문에 전체 테스트가 실패할 수 있다. **자기 소유 영역의 테스트**로 커밋 여부를 판단하고, 영역 밖 실패는 보고만 한다.
- 공용 캐시(`.venv`, `node_modules`, HF cache)는 공유하되 삭제·재생성하지 않는다.
- 장시간 서버(dev server, docker compose)를 띄웠다면 작업 종료 전에 반드시 내린다. 포트는 배정받은 범위를 사용한다.

### 6.5 최종 보고 형식 (서브 에이전트)

```text
COMMITS: <sha> <subject> (작업 순서대로 전부)
UNCOMMITTED: 남은 변경 파일 (없으면 none)
TESTS: 실행한 명령과 결과 요약
CHANGE REQUEST: 소유 범위 밖에 필요한 변경 (없으면 none)
RISKS: 미검증 가정, 알려진 한계
```

## 7. 커밋 전 검증 게이트

| 영역 | 최소 검증 |
|---|---|
| Python (`packages/`, `services/`) | `uv run ruff check <변경 경로>` + 관련 `uv run pytest <테스트 경로> -q` |
| Web (`apps/web`) | `pnpm -C apps/web lint` + `pnpm -C apps/web typecheck` + 관련 `pnpm -C apps/web test` |
| Docker/compose | `docker compose config -q` (빌드가 바뀌면 해당 서비스 `docker compose build <svc>`) |
| 문서 | 링크·경로가 실제 파일과 일치하는지 확인 |

검증을 생략했다면 커밋 body에 `Not verified: <이유>`를 남긴다.

## 8. Push 정책

- **push는 오케스트레이터만** 수행한다. 서브 에이전트는 push하지 않는다.
- push 시점: 핵심 태스크(마일스톤 또는 그 하위 목표, `docs/goals.md`)가 완료되고 해당 영역 테스트가 통과했을 때.
- 명령: `git push origin main`. **force push 금지**, 이미 push한 커밋의 history 수정 금지.
- push 전 확인
  1. `git log origin/main..main --oneline` 으로 나갈 커밋 목록 검토
  2. 비밀정보 검사 (§10)
  3. 진행 중인 에이전트가 없거나, 있더라도 커밋된 내용만 나간다는 점 확인

## 9. Worktree 격리 모드 (선택)

오케스트레이터가 `isolation: worktree`로 에이전트를 실행하면:

- 에이전트는 자신의 worktree에서 `agent/<label>` 브랜치로 작업하고 위 커밋 규칙을 동일하게 지킨다.
- 에이전트는 병합·push하지 않는다.
- 오케스트레이터가 리뷰 후 `git merge --ff-only agent/<label>`(불가하면 `--no-ff`)로 `main`에 합치고 브랜치를 정리한다.

## 10. 보안과 데이터 (public 저장소)

이 저장소는 **공개**다. 한 번 push한 내용은 삭제해도 캐시·포크에 남을 수 있다.

절대 커밋하지 않는다.

- `.env`, HF token(`hf_...`), GitHub token, API key, 인증서·개인키
- 모델 가중치(`*.safetensors`, `*.gguf`, `*.bin` 등), 데이터셋 원문, 분석 artifact, 업로드 파일
- 개인 절대경로(`/Users/<name>/...`), 사설 URL, 실제 사용자 데이터

커밋 전 자기 점검:

```bash
git diff --cached | grep -nE 'hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY' \
  && echo "STOP: secret-like string staged"
```

테스트 fixture는 작게 유지한다(파일당 200KB 이하 권장, 1MB 초과 금지). 외부 config를 fixture로 쓸 때는 출처 repo·revision·hash를 함께 기록한다.

## 11. 커밋 직전 체크리스트

- [ ] 소유 경로 안의 파일만 바뀌었다
- [ ] 파일 수 ≤ 6 이고 하나의 논리 변경이다 (예외는 본문에 사유)
- [ ] `git add -- <경로>` / `git commit ... -- <경로>`로 경로를 명시했다
- [ ] 메시지가 `type(scope): subject` 형식이고, 서브 에이전트라면 `Agent:` trailer가 있다
- [ ] 해당 영역의 lint/test를 실행했다 (또는 `Not verified:` 사유 기재)
- [ ] 비밀정보·대용량·개인 경로가 없다
- [ ] 금지 명령(§5.1)을 쓰지 않았다
