"""Integration fixtures: real API + real worker task + real pipeline orchestration with fake
estimator modules, on SQLite and fakeredis (jobs execute synchronously at enqueue time)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import fakeredis
import pytest
from fastapi.testclient import TestClient
from integration_testkit import CSRF, load_pipeline_fakes, make_settings

from vramforge_api.app import create_app
from vramforge_api.db import dispose_engines
from vramforge_api.settings import Settings, reset_settings_cache
from vramforge_estimator.schemas import (
    GrpoResolved,
    RewardKind,
    RolloutBackend,
    TrainingReadiness,
)


@pytest.fixture
def settings_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[Callable[..., Settings]]:
    def make(**overrides: Any) -> Settings:
        return make_settings(tmp_path, monkeypatch, **overrides)

    yield make
    reset_settings_cache()
    dispose_engines()


@pytest.fixture
def settings(settings_factory: Callable[..., Settings]) -> Settings:
    return settings_factory()


@pytest.fixture
def fake_redis() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


@pytest.fixture
def client_factory(fake_redis: fakeredis.FakeRedis) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def make(settings: Settings, *, queue_is_async: bool = False, **kwargs: Any) -> TestClient:
        app = create_app(settings, redis=fake_redis, queue_is_async=queue_is_async)
        client = TestClient(app, headers=CSRF, **kwargs)
        client.__enter__()
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def modules(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Fake estimator modules whose compatibility result is a ready GRPO/SFT/DPO plan."""
    fakes = load_pipeline_fakes()
    from vramforge_estimator import compatibility

    monkeypatch.setattr(compatibility, "validate_request", lambda request: [])

    def resolve(request, inventory, tokenizer):
        objective = request.training.objective
        extra: dict[str, Any] = {}
        if objective.value == "grpo":
            budget = request.grpo.completion_budget
            extra["grpo"] = GrpoResolved(
                num_generations=request.grpo.num_generations,
                generation_batch_size=request.grpo.generation_batch_size or 4,
                steps_per_generation=4,
                num_iterations=1,
                completion_budgets=[budget] if budget else [1024, 2048, 4096, 8192],
                budget_explicit=budget is not None,
                beta=request.grpo.beta,
                reference_needed=False,
                reward_kind=RewardKind(request.grpo.reward.kind),
                rollout_backend=RolloutBackend.TRANSFORMERS_SHARED_POLICY,
                live_sequences=4,
                update_microbatch=1,
                accumulation=4,
            )
        lora = fakes.resolved_config(objective).lora.model_copy(
            update={"r": request.training.lora.r}
        )
        resolved = fakes.resolved_config(objective, lora=lora, **extra)
        ready = objective.value != "grpo" or (
            request.grpo.completion_budget is not None and request.grpo.reward.kind != "unspecified"
        )
        report = fakes.compat_report(
            readiness=TrainingReadiness.READY if ready else TrainingReadiness.CONDITIONAL
        )
        return resolved, report

    return fakes.FakeModules(resolve=resolve).install(monkeypatch)
