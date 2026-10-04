# VRAMForge — Fine-Tuning VRAM Calculator

모델·데이터셋·파인튜닝 방식(SFT / DPO / GRPO)을 입력하면 실제 토크나이저와 학습 전처리로 **전체 데이터 길이**를 분석하고, **데이터를 자르지 않는 조건의 GPU별 피크 VRAM**을 산정하는 self-hosted 웹 계산기입니다.

> 개발 진행 중입니다. 범위와 완료 기준은 [`docs/goals.md`](docs/goals.md), 전체 사양은 [`plan.md`](plan.md)를 참고하세요.

## 빠른 시작 (목표 동작)

```bash
docker compose up -d --build
# → http://localhost:8080
```

## 문서

- [`plan.md`](plan.md) — 최종 구현 명세
- [`docs/goals.md`](docs/goals.md) — 목표와 완료 기준
- [`docs/git-conventions.md`](docs/git-conventions.md) — Git 규칙 (멀티 에이전트)
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — 기여 안내
