"""Every `formula_ref` the adapters emit points at an anchor of the methodology document."""

from __future__ import annotations

import re
from pathlib import Path
from types import ModuleType

from arch_helpers import make_cfg

from vramforge_estimator.architectures import GenerationTimepoints, StepTimepoints, get_adapter
from vramforge_estimator.schemas import ModelInventory, Objective, SequenceShape, Strategy

DOC = Path(__file__).resolve().parents[3] / "docs" / "methodology-architectures.md"
ANCHOR = re.compile(r'<a id="([a-z0-9-]+)"></a>')


def _refs(mimo: ModelInventory, ib: ModuleType) -> set[str]:
    hybrid = get_adapter("qwen3_5_hybrid")
    dense = get_adapter("dense_decoder")
    names = [m.name for m in hybrid.lora_target_modules(mimo, "auto_verified", [])]
    tps = StepTimepoints("P:f", "P:l", "P:b")
    gtp = GenerationTimepoints("G:p", "G:d")
    allocs = []
    for gc in (True, False):
        cfg = make_cfg(objective=Objective.DPO, targets=names, gc=gc, dora=True, load="float32")
        allocs += hybrid.resident_weights(mimo, cfg, ["t"])
        allocs += hybrid.load_transient(mimo, cfg, ["t"])
        allocs += hybrid.train_step_ledger(mimo, cfg, SequenceShape(batch=2, seq_len=256), tps, "p")
        allocs += hybrid.no_grad_forward_ledger(
            mimo, cfg, SequenceShape(batch=2, seq_len=64), ["r"], "r"
        )
        allocs += hybrid.generation_ledger(mimo, cfg, 2, 32, 8, gtp, "g")
    tiny = ib.tiny_q35_inventory()
    full = make_cfg(
        strategy=Strategy.FULL, gc=False, use_cache=True, paths={"full_attention": "eager"}
    )
    allocs += hybrid.train_step_ledger(tiny, full, SequenceShape(batch=1, seq_len=40), tps, "p")
    allocs += hybrid.train_step_ledger(
        tiny,
        make_cfg(strategy=Strategy.LORA, targets=["model.layers.0.mlp.up_proj"], gc=False),
        SequenceShape(batch=1, seq_len=40),
        tps,
        "q",
    )
    dinv = ib.tiny_dense_inventory("mistral", sliding_window=16)
    allocs += dense.generation_ledger(dinv, make_cfg(strategy=Strategy.LORA), 2, 20, 4, gtp, "d")
    allocs += dense.resident_weights(dinv, make_cfg(strategy=Strategy.LORA, load="float32"), ["t"])
    allocs += dense.load_transient(dinv, make_cfg(strategy=Strategy.LORA, load="float32"), ["t"])
    return {a.formula_ref for a in allocs}


def test_formula_refs_resolve_to_doc_anchors(mimo: ModelInventory, ib: ModuleType) -> None:
    anchors = set(ANCHOR.findall(DOC.read_text(encoding="utf-8")))
    refs = _refs(mimo, ib)
    assert all(r and r.startswith("methodology-architectures.md#") for r in refs)
    missing = {r.split("#", 1)[1] for r in refs} - anchors
    assert not missing, f"undocumented formula anchors: {sorted(missing)}"
    assert len(refs) >= 15
