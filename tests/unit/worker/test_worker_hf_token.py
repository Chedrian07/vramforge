"""Analyses use the server's Hugging Face token only when the operator shares it, and access
failures say which token mode applies (plan §18)."""

from __future__ import annotations

from typing import Any

import pytest
from worker_testkit import create_analysis, get, load_pipeline_fakes

from vramforge_api import store
from vramforge_api.db import get_engine, session_factory, session_scope
from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import ErrorCode, Stage
from vramforge_worker import tasks

fakes = load_pipeline_fakes()
TOKEN = "hf_" + "W" * 32


@pytest.fixture(autouse=True)
def _no_shell_token(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("HF_TOKEN", "VRAMFORGE_HF_TOKEN", "VRAMFORGE_SHARE_SERVER_HF_TOKEN"):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.parametrize("share", [False, True])
def test_access_denied_analysis_explains_the_token_policy(
    settings_factory: Any, monkeypatch: pytest.MonkeyPatch, share: bool
) -> None:
    settings = settings_factory(hf_token=TOKEN, share_server_hf_token=share)
    sessions = session_factory(get_engine(settings.database_url))
    seen: list[str | None] = []

    def denied(kind: str) -> Any:
        def resolve(ref: Any, access: Any) -> Any:
            seen.append(access.hf_token)
            raise EstimatorError(
                make_issue(
                    ErrorCode.SOURCE_ACCESS_DENIED,
                    f"{kind} 저장소에 접근할 권한이 없습니다.",
                    stage=Stage.RESOLVING,
                    component=kind,
                    reason="forbidden",
                )
            )

        return resolve

    fakes.FakeModules(resolve_model=denied("모델"), resolve_dataset=denied("데이터셋")).install(
        monkeypatch
    )
    aid = create_analysis(sessions)
    assert tasks.run_analysis(aid, 1) == "FAILED"
    assert seen == ([TOKEN, TOKEN] if share else [None, None])
    mode = "shared" if share else "configured_not_shared"

    row = get(sessions, aid)
    assert row.error["details"]["server_hf_token"] == mode
    assert {e["details"]["server_hf_token"] for e in row.result["errors"]} == {mode}
    if not share:
        assert "VRAMFORGE_SHARE_SERVER_HF_TOKEN=true" in row.error["user_message"]
    with session_scope(sessions) as db:
        events = store.events_after(db, aid, 0)
    streamed = [e.issue for e in events if e.issue is not None]
    assert streamed and all(i.details.get("server_hf_token") == mode for i in streamed)
    assert TOKEN not in str(row.result) and TOKEN not in str(row.error)
