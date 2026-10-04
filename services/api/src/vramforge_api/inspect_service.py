"""Metadata-only source inspection for `POST /sources/inspect` (plan.md §3.3, §6.2, §7.1).

Model and dataset are inspected concurrently in worker threads with an overall time limit
(`VRAMFORGE_INSPECT_TIMEOUT_S`). Each finished step is published immediately, so a timeout or a
failing step still returns everything verified so far plus the issue (never invented values).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TypeVar

import anyio

from vramforge_estimator import compatibility, inspection, pipeline, sources
from vramforge_estimator.errors import EstimatorError, make_issue
from vramforge_estimator.schemas import (
    DatasetInspection,
    DatasetSourceRef,
    ErrorCode,
    InspectRequest,
    InspectResponse,
    Issue,
    ModelInspection,
    ModelInventory,
    ModelSourceRef,
    Objective,
    Severity,
    Stage,
)
from vramforge_estimator.sources import SourceAccess

log = logging.getLogger(__name__)

T = TypeVar("T")


class _Step:
    """Runs one library call and converts failures into issues (never raises)."""

    def __init__(self, issues: list[Issue], stage: Stage) -> None:
        self.issues = issues
        self.stage = stage

    def __call__(self, component: str, fn: Callable[[], T]) -> T | None:
        try:
            return fn()
        except EstimatorError as exc:
            self.issues.append(exc.issue)
        except NotImplementedError:
            self.issues.append(
                make_issue(
                    ErrorCode.INTERNAL_ERROR,
                    "이 확인 단계는 현재 빌드에서 아직 구현되지 않았습니다.",
                    stage=self.stage,
                    component=component,
                    reason="not_implemented",
                )
            )
        except Exception:
            log.exception("inspection step %s failed", component)
            self.issues.append(
                make_issue(
                    ErrorCode.INTERNAL_ERROR,
                    "메타데이터 확인 중 내부 오류가 발생했습니다.",
                    stage=self.stage,
                    component=component,
                    retryable=True,
                )
            )
        return None


class _Holder:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model: ModelInspection | None = None
        self.dataset: DatasetInspection | None = None

    def set_model(self, value: ModelInspection) -> None:
        with self.lock:
            self.model = value.model_copy(deep=True)

    def set_dataset(self, value: DatasetInspection) -> None:
        with self.lock:
            self.dataset = value.model_copy(deep=True)


def inspect_model_part(ref: ModelSourceRef, access: SourceAccess, holder: _Holder) -> None:
    issues: list[Issue] = []
    out = ModelInspection(issues=issues)
    step = _Step(issues, Stage.RESOLVING)
    holder.set_model(out)
    source = step("sources.resolve_model", lambda: sources.resolve_model(ref, access))
    if source is None:
        holder.set_model(out)
        return
    out.manifest = source.manifest
    holder.set_model(out)
    step.stage = Stage.INSPECTING
    inventory: ModelInventory | None = step(
        "inspection.inspect_model", lambda: inspection.inspect_model(source, access)
    )
    if inventory is not None:
        out.summary = pipeline.inventory_summary(inventory)
    holder.set_model(out)
    handle = step("inspection.load_tokenizer", lambda: inspection.load_tokenizer(source, access))
    if handle is not None:
        out.tokenizer = handle.manifest
    holder.set_model(out)
    if inventory is not None:
        support = step(
            "compatibility.support_for", lambda: compatibility.support_for(inventory.facts)
        )
        if support is not None:
            adapter_id, entries = support
            out.architecture_adapter = adapter_id
            out.support = list(entries)
            if adapter_id is None:
                issues.append(
                    make_issue(
                        ErrorCode.UNSUPPORTED_ARCHITECTURE,
                        "등록된 구조 adapter가 없어 메타데이터만 제공합니다. "
                        "메모리 산정은 지원하지 않습니다.",
                        severity=Severity.WARNING,
                        stage=Stage.INSPECTING,
                    )
                )
    holder.set_model(out)


def inspect_dataset_part(
    ref: DatasetSourceRef,
    access: SourceAccess,
    objective: Objective | None,
    holder: _Holder,
) -> None:
    issues: list[Issue] = []
    out = DatasetInspection(issues=issues)
    step = _Step(issues, Stage.RESOLVING)
    holder.set_dataset(out)
    source = step("sources.resolve_dataset", lambda: sources.resolve_dataset(ref, access))
    if source is None:
        holder.set_dataset(out)
        return
    out.manifest = source.manifest
    holder.set_dataset(out)
    step.stage = Stage.INSPECTING
    found = step(
        "inspection.inspect_dataset",
        lambda: inspection.inspect_dataset(source, ref, access, objective),
    )
    if found is not None:
        merged = found.model_copy(update={"issues": [*issues, *found.issues]})
        if merged.manifest is None:
            merged.manifest = source.manifest
        holder.set_dataset(merged)
        return
    holder.set_dataset(out)


def _timeout_issue(component: str, timeout_s: float) -> Issue:
    return make_issue(
        ErrorCode.JOB_TIMEOUT,
        f"메타데이터 확인이 제한 시간({timeout_s:g}초) 안에 끝나지 않았습니다. "
        "확인된 정보까지만 표시합니다.",
        stage=Stage.INSPECTING,
        retryable=True,
        component=component,
    )


async def inspect_sources(
    body: InspectRequest,
    access: SourceAccess,
    *,
    timeout_s: float,
    model_issue: Issue | None = None,
    dataset_issue: Issue | None = None,
    limiter: anyio.CapacityLimiter | None = None,
) -> InspectResponse:
    """Run the requested parts concurrently; `*_issue` short-circuits a part (e.g. an upload
    reference that does not belong to the caller)."""
    holder = _Holder()
    done = {"model": False, "dataset": False}

    async def run_model() -> None:
        assert body.model is not None
        await anyio.to_thread.run_sync(
            inspect_model_part,
            body.model,
            access,
            holder,
            limiter=limiter,
            abandon_on_cancel=True,
        )
        done["model"] = True

    async def run_dataset() -> None:
        assert body.dataset is not None
        await anyio.to_thread.run_sync(
            inspect_dataset_part,
            body.dataset,
            access,
            body.objective,
            holder,
            limiter=limiter,
            abandon_on_cancel=True,
        )
        done["dataset"] = True

    with anyio.move_on_after(timeout_s):
        async with anyio.create_task_group() as tg:
            if body.model is not None and model_issue is None:
                tg.start_soon(run_model)
            if body.dataset is not None and dataset_issue is None:
                tg.start_soon(run_dataset)

    response = InspectResponse()
    with holder.lock:
        model, dataset = holder.model, holder.dataset
    if body.model is not None:
        if model_issue is not None:
            response.model = ModelInspection(issues=[model_issue])
        else:
            response.model = model or ModelInspection()
            if not done["model"]:
                response.model.issues.append(_timeout_issue("model", timeout_s))
    if body.dataset is not None:
        if dataset_issue is not None:
            response.dataset = DatasetInspection(issues=[dataset_issue])
        else:
            response.dataset = dataset or DatasetInspection()
            if not done["dataset"]:
                response.dataset.issues.append(_timeout_issue("dataset", timeout_s))
    return response


__all__ = ["inspect_dataset_part", "inspect_model_part", "inspect_sources"]
