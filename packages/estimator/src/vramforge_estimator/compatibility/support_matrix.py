"""Render docs/support-matrix.md from the profile registry (plan.md §20.1).

Regenerate with `COMMAND` below. tests/unit/compatibility/test_support_matrix.py renders the
document and compares it with the committed file, so the matrix cannot drift from the registry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vramforge_estimator.schemas import Objective, Strategy
from vramforge_estimator.units import MiB, to_gib

from .profiles import AnalyticProfile, EnvironmentProfile, ProfileRegistry, load_registry

COMMAND = (
    "uv run --no-sync python -m vramforge_estimator.compatibility.support_matrix "
    "--write docs/support-matrix.md"
)


def _bytes(value: float) -> str:
    if value >= 1024 * MiB:
        return f"{to_gib(int(value)):.2f} GiB ({int(value):,} B)"
    return f"{value / MiB:.0f} MiB ({int(value):,} B)"


def _esc(text: str) -> str:
    """Escape pipes inside Markdown table cells."""
    return text.replace("|", "\\|")


def _cell(profile: AnalyticProfile, objective: Objective, strategy: Strategy) -> str:
    rule = profile.support_rule(objective, strategy)
    if rule.grade is None:
        return "unsupported"
    return f"{rule.grade.value} / {rule.readiness.value}"


def _environment(env: EnvironmentProfile) -> list[str]:
    lines = [
        f"### `{env.id}` (version {env.version})",
        "",
        env.description,
        "",
        "| 패키지 | 버전 |",
        "|---|---|",
        *[f"| {name} | {ver} |" for name, ver in sorted(env.packages.items())],
        "",
        f"- Python: {env.python}, platform: {env.platform}, CUDA: "
        f"{env.cuda or '미고정 (GPU 보정 M5에서 고정)'}, NVIDIA driver: "
        f"{env.driver or '미고정 (GPU 보정 M5에서 고정)'}",
        f"- `dependency_lock_digest`: `{env.dependency_lock_digest}`",
        "- 학습 컨테이너 image digest: "
        + (f"`{env.container_image_digest}`" if env.container_image_digest else "없음 (미배포)"),
        f"- 설치된 선택 kernel: {', '.join(env.kernels.installed) or '없음'}",
        f"- 설치되지 않은 패키지: {', '.join(env.kernels.absent)}",
        f"- linear attention 경로: `{env.kernels.linear_attention}`, "
        f"log-prob kernel: `{env.kernels.logprob_kernel}`",
    ]
    lines += [f"- {note}" for note in env.kernels.notes]
    return [*lines, ""]


def _matrix(registry: ProfileRegistry) -> list[str]:
    head = "| objective | architecture adapter | profile | environment | full | lora | qlora |"
    lines = [head, "|---|---|---|---|---|---|---|"]
    profiles = sorted(registry.analytic.values(), key=lambda p: p.architecture_adapter)
    for objective in Objective:
        for p in profiles:
            cells = " | ".join(_cell(p, objective, s) for s in Strategy)
            lines.append(
                f"| {objective.value} | `{p.architecture_adapter}` | {p.id}@{p.version} | "
                f"{p.environment} | {cells} |"
            )
    return [*lines, ""]


def _profile(p: AnalyticProfile) -> list[str]:
    load = p.loading
    q = p.quantization
    lines = [
        f"### {p.id} ({p.version}) — `{p.architecture_adapter}`",
        "",
        p.description,
        "",
        f"- 환경: `{p.environment}`",
        f"- model types (참고용, 판정은 구조 기반): {', '.join(p.model_types)}",
        f"- architectures (참고용): {', '.join(p.architectures)}",
        "- trainer adapters: "
        + ", ".join(f"{o.value} → `{t}`" for o, t in sorted(p.trainers.items())),
        f"- 로딩: `{load.entrypoint}`, 클래스 = `{load.architecture_source}`, 기본 범위 "
        f"`{load.default_scope}` ({load.default_scope_reason})",
    ]
    lines += [f"  - `{r.architecture}` → `{r.scope}`: {r.reason}" for r in load.scope_rules]
    lines += [
        f"  - text_only 검증 여부: {'예' if load.text_only.verified else '아니오'} — "
        f"{load.text_only.reason}",
        f"- load dtype 기본값: `{load.default_load_dtype}`",
    ]
    lines += [f"  - `{d}`: {note}" for d, note in load.load_dtypes.items()]
    factors = ", ".join(f"{k} {v:g}" for k, v in load.device_map_budget_factor.items())
    lines += [
        f"- device_map `{load.device_map}` 예산 계수: {factors}. {load.device_map_note}",
        f"- 4-bit preset: {q.library} {'/'.join(q.formats)} (기본 {q.default_format}), double "
        f"quant {'on' if q.double_quant else 'off'}, compute `{q.compute_dtype}`, storage "
        f"`{q.quant_storage}`, blocksize {q.blocksize}/{q.nested_blocksize}, 제외 모듈 "
        f"{', '.join(q.skip_modules)}. {q.note}",
        "- attention 경로: "
        + ", ".join(f"`{k}` → `{v}`" for k, v in p.attention.by_layer_type.items())
        + " (지원: "
        + "; ".join(f"{k}: {', '.join(v)}" for k, v in p.attention.supported.items())
        + ")",
    ]
    lines += [
        f"- loss 경로 {o.value}: 기본 `{r.default}`, 지원 {', '.join(r.supported)}. {r.note}"
        for o, r in sorted(p.loss.items())
    ]
    c = p.checkpointing
    lines += [
        f"- gradient checkpointing: 기본 {'on' if c.default_enabled else 'off'}, "
        f"`{c.granularity}`, use_reentrant={c.use_reentrant}. {c.note}",
        "- DPO reference 자동 선택: "
        + ", ".join(f"{k} → `{v.value}`" for k, v in p.dpo_reference_auto.items()),
        f"- GRPO rollout backend: {', '.join(b.value for b in p.rollout_backends)}",
        "- batch preset (microbatch / accumulation): "
        + ", ".join(
            f"{o.value} {b.microbatch}/{b.accumulation}" for o, b in sorted(p.presets.items())
        ),
        f"- calibration: {', '.join(p.calibration_coverage) or '없음 (analytic만)'}",
        f"- 지원 하드웨어: compute capability ≥ {p.hardware.min_compute_capability}. "
        f"{p.hardware.reason}",
        "",
        "| optimizer | state | 비고 |",
        "|---|---|---|",
    ]
    for o in p.optimizers:
        kind = "8-bit" if o.eight_bit else "param dtype"
        flags = [f for f, on in (("fused", o.fused), ("paged", o.paged)) if on]
        state = f"{o.states_per_param} × {kind}" + (f" ({', '.join(flags)})" if flags else "")
        lines.append(f"| `{o.name.value}` | {state} | {_esc(o.note)} |")
    lines += ["", "| objective | strategy | 등급 | readiness | 비고 |", "|---|---|---|---|---|"]
    for s in p.support:
        grade = s.grade.value if s.grade else "unsupported"
        lines.append(
            f"| {s.objective.value} | {s.strategy.value} | {grade} | {s.readiness.value} | "
            f"{_esc(s.note)} |"
        )
    lines += ["", "| 지원하지 않는 옵션 | 이유 |", "|---|---|"]
    lines += [f"| `{_esc(u.option)}` | {_esc(u.reason)} |" for u in p.unsupported_options]
    lines += ["", "| fallback 규칙 | 계산에 쓰는 경로 |", "|---|---|"]
    lines += [f"| `{_esc(r.option)}` | {_esc(r.behavior)} |" for r in p.fallback_rules]
    ws = p.workspace
    lines += [
        "",
        "| workspace 가정 (ASSUMPTION) | low | high | 근거 |",
        "|---|---|---|---|",
        f"| CUDA context | {_bytes(ws.cuda_context_bytes.low)} | "
        f"{_bytes(ws.cuda_context_bytes.high)} | {ws.cuda_context_bytes.source} |",
        f"| library workspace | {_bytes(ws.library_workspace_bytes.low)} | "
        f"{_bytes(ws.library_workspace_bytes.high)} | {ws.library_workspace_bytes.source} |",
        f"| allocator slack (할당량 대비) | {ws.allocator_slack_fraction.low:.0%} | "
        f"{ws.allocator_slack_fraction.high:.0%} | {ws.allocator_slack_fraction.source} |",
        "",
    ]
    return lines


def render_support_matrix(registry: ProfileRegistry | None = None) -> str:
    reg = registry or load_registry()
    lines = [
        "<!-- Generated from the profile registry. Do not edit by hand. Regenerate with:",
        f"     {COMMAND}",
        "     tests/unit/compatibility/test_support_matrix.py fails when this file is stale. -->",
        "",
        "# 지원 매트릭스 (Support matrix)",
        "",
        "objective × architecture × strategy × backend 조합의 지원 등급입니다 "
        "(plan.md §11.4, §20.1). 값은 `profiles/`의 registry에서 생성하며, 계산 방법은 "
        "[methodology.md](methodology.md)에 있습니다.",
        "",
        "- `analytic`: 명시적 allocation·workspace 가정을 가진 정적 추정 (GPU 실측 아님).",
        "- `calibrated`, `measured`: 아직 등록된 profile이 없습니다 (GPU 검증 M5, "
        "`profiles/calibrated/README.md`).",
        "- `unsupported`: 지원하지 않는 조합. 숫자 대신 원인을 반환합니다.",
        "- readiness `ready`는 backend 조합 기준입니다. 요청 단위 조건(예: GRPO reward 미지정)은 "
        "결과를 `conditional`로 바꿉니다.",
        "- 등록되지 않은 구조(adapter 없음)는 metadata만 반환하며 메모리 수치를 만들지 않습니다.",
        "",
        "## 조합별 등급",
        "",
        "셀 값: `등급 / readiness`.",
        "",
        *_matrix(reg),
        "## 학습 환경",
        "",
    ]
    for env in sorted(reg.environments.values(), key=lambda e: e.id):
        lines += _environment(env)
    lines += ["## Profile 상세", ""]
    for prof in sorted(reg.analytic.values(), key=lambda p: p.id):
        lines += _profile(prof)
    lines += [
        "## GPU preset",
        "",
        reg.hardware.note,
        "",
        "| id | 이름 | 공칭 용량 |",
        "|---|---|---|",
    ]
    lines += [f"| `{g.id}` | {g.name} | {g.nominal_gib} GiB |" for g in reg.hardware.gpus]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", type=Path, help="write the document to this path")
    parser.add_argument("--check", type=Path, help="exit 1 if this file is stale")
    args = parser.parse_args(argv)
    text = render_support_matrix()
    if args.check:
        current = args.check.read_text(encoding="utf-8") if args.check.exists() else ""
        return 0 if current == text else 1
    if args.write:
        args.write.write_text(text, encoding="utf-8")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
