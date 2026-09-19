from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .config import ProjectConfig
from .deformation.decompose import decompose_insar
from .deformation.regimes import classify_deformation
from .extensometer.analysis import analyze_extensometer
from .groundwater.field import build_groundwater_field
from .hydromechanics.seasonal import compute_joint_harmonics, estimate_lag
from .hydromechanics.storativity import estimate_regularized_ske
from .hydrostratigraphy.analysis import analyze_hydrostratigraphy
from .io.groundwater import prepare_groundwater
from .io.insar import prepare_insar
from .storage.budget import compute_storage_budget
from .synthesis import synthesize

STAGES = OrderedDict([
    ("prepare-insar", prepare_insar),
    ("prepare-groundwater", prepare_groundwater),
    ("build-groundwater-field", build_groundwater_field),
    ("decompose-insar", decompose_insar),
    ("classify-deformation", classify_deformation),
    ("joint-harmonics", compute_joint_harmonics),
    ("estimate-lag", estimate_lag),
    ("estimate-ske", estimate_regularized_ske),
    ("storage-budget", compute_storage_budget),
    ("hydrostratigraphy", analyze_hydrostratigraphy),
    ("extensometer", analyze_extensometer),
    ("synthesize", synthesize),
])


def run_stage(cfg: ProjectConfig, stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise KeyError(f"Unknown stage {stage!r}. Choices: {list(STAGES)}")
    return STAGES[stage](cfg)


def run_pipeline(cfg: ProjectConfig, start: str | None = None, stop: str | None = None, only: str | None = None):
    if only:
        return {only: run_stage(cfg, only)}
    names = list(STAGES)
    i0 = names.index(start) if start else 0
    i1 = names.index(stop) if stop else len(names)-1
    if i1 < i0:
        raise ValueError("stop stage occurs before start stage")
    results = {}
    for name in names[i0:i1+1]:
        results[name] = run_stage(cfg, name)
    return results
