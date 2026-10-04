# Contributing

VRAMForge는 사람과 여러 서브 에이전트가 같은 저장소에서 동시에 작업한다. 기여 전에 다음 문서를 읽는다.

1. [`plan.md`](plan.md) — 제품 사양 (무엇을 만드는가)
2. [`docs/goals.md`](docs/goals.md) — 마일스톤 목표와 완료 기준 (언제 끝나는가)
3. [`docs/git-conventions.md`](docs/git-conventions.md) — 커밋·브랜치·push 규칙 (어떻게 기록하는가)

핵심 규칙 요약

- 커밋은 `type(scope): subject` 형식이고, 하나의 논리 변경만 담는다 (파일 6개 이하 권장).
- 스테이징과 커밋에는 항상 경로를 명시한다. `git add -A`, `git add .`, `git commit -a`는 금지다.
- 서브 에이전트는 배정받은 디렉터리만 수정하고 push하지 않는다.
- 공개 저장소이므로 토큰, `.env`, 모델 가중치, 데이터셋 원문을 커밋하지 않는다.
