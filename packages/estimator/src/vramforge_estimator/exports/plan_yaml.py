"""`resolved-plan.yaml`: this app's reproducible analysis plan (plan.md §12.4).

It pins the source revisions, the dataset selection and mapping transform, the requested and
resolved configuration, the batch plan and the scenario numbers. It is NOT a list of TRL
arguments; `trainer-config.yaml` is the (gated) execution config.
"""

from __future__ import annotations

from typing import Any

from vramforge_estimator.schemas import AnalysisResult, SourceManifest

HEADER = (
    "# VRAMForge resolved plan (schema 1.0)\n"
    "# 이 파일은 VRAMForge 분석 계획의 재현용 기록입니다. TRL 인자와 1:1로 대응하지 않습니다.\n"
    "# This is VRAMForge's reproducible analysis plan, not a 1:1 list of TRL arguments.\n"
    "# 실행용 설정은 trainer-config.yaml(학습 준비 상태가 ready일 때만 제공)을 사용하세요.\n"
)


def _source(manifest: SourceManifest | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {
        "reference": manifest.reference,
        "source_type": manifest.source_type.value,
        "repo_id": manifest.repo_id,
        "requested_revision": manifest.requested_revision,
        "resolved_revision": manifest.resolved_revision,
        "fingerprint": manifest.fingerprint,
    }


def build_plan(result: AnalysisResult) -> dict[str, Any]:
    scan = result.dataset_scan
    tok = result.tokenizer_manifest
    plan = result.batch_plan
    memory = result.memory
    data: dict[str, Any] = {
        "kind": "vramforge.resolved-plan",
        "schema_version": result.schema_version,
        "estimator_version": result.estimator_version,
        "analysis_id": result.analysis_id,
        "analysis_fingerprint": result.analysis_fingerprint,
        "created_at": result.created_at.isoformat(),
        "status": result.status.model_dump(mode="json"),
        "sources": {
            "model": _source(result.source_manifests.model),
            "dataset": _source(result.source_manifests.dataset),
        },
        "tokenizer": None
        if tok is None
        else {
            "class": tok.tokenizer_class,
            "fingerprint": tok.fingerprint,
            "chat_template_sha256": tok.chat_template_sha256,
            "chat_template_source": tok.chat_template_source,
        },
        "dataset_plan": None
        if scan is None
        else {
            "objective": scan.objective.value,
            "config": scan.config,
            "split": scan.split,
            "split_auto_selected": scan.split_auto_selected,
            "mapping": scan.mapping_applied.model_dump(mode="json")
            if scan.mapping_applied
            else None,
            "transformation": scan.transformation_note,
            "coverage": scan.coverage.value,
            "rows_seen": scan.rows_seen,
            "rows_ok": scan.rows_ok,
            "rows_failed": scan.rows_failed,
            "preprocess_key": scan.preprocess_key,
            "preprocessing_adapter": scan.preprocessing_adapter,
            "preprocessing_adapter_version": scan.preprocessing_adapter_version,
            "truncation": "none (strict_no_truncation)",
        },
        "requested": result.requested_config.model_dump(mode="json"),
        "resolved": result.resolved_config.model_dump(mode="json")
        if result.resolved_config
        else None,
        "batch_plan": None
        if plan is None
        else {
            "unit": plan.unit,
            "microbatch": plan.microbatch,
            "accumulation": plan.accumulation,
            "effective_batch": plan.effective_batch,
            "pad_to_multiple_of": plan.pad_to_multiple_of,
            "sampler": plan.sampler.model_dump(mode="json"),
            "worst_case": plan.worst_case.model_dump(mode="json", exclude={"source_row_ids"}),
            "batch_key": plan.batch_key,
        },
        "memory": None
        if memory is None
        else {
            "evidence_level": memory.evidence_level.value,
            "primary_scenario_id": memory.primary_scenario_id,
            "scenarios": [
                {
                    "id": s.scenario_id,
                    "label": s.label,
                    "params": s.params,
                    "devices": [
                        {
                            "device": d.device,
                            "known_floor_bytes": d.known_floor_bytes,
                            "scenario_low_bytes": d.scenario_low_bytes,
                            "scenario_high_bytes": d.scenario_high_bytes,
                            "peak_phase": d.peak_phase.value if d.peak_phase else None,
                            "peak_timepoint": d.peak_timepoint,
                        }
                        for d in s.devices
                    ],
                    "recommendation": s.recommendation.model_dump(mode="json")
                    if s.recommendation
                    else None,
                    "hardware_fit": s.hardware_fit.model_dump(mode="json"),
                }
                for s in memory.scenarios
            ],
        },
        "planning_margin_policy": result.planning_margin_policy.model_dump(mode="json"),
        "profile_id": result.profile_id,
        "dependency_lock_digest": result.dependency_lock_digest,
        "assumptions": [a.model_dump(mode="json") for a in result.assumptions],
        "unknown_components": [u.model_dump(mode="json") for u in result.unknown_components],
        "excluded_components": [x.model_dump(mode="json") for x in result.excluded_components],
        "issues": [
            {"code": i.code.value, "severity": i.severity.value, "message": i.user_message}
            for i in (*result.errors, *result.warnings)
        ],
    }
    return data


__all__ = ["HEADER", "build_plan"]
