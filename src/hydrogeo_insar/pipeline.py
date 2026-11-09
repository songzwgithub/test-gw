from __future__ import annotations

from collections import OrderedDict
from typing import Any

from .config import ProjectConfig
from .deformation.decompose import decompose_insar
from .deformation.regimes import classify_deformation
from .extensometer.analysis import analyze_extensometer
from .groundwater.field import build_groundwater_field
from .hydromechanics.lag import estimate_lag
from .hydromechanics.pixelwise_storativity import estimate_pixelwise_ske
from .hydromechanics.seasonal import compute_joint_harmonics
from .hydromechanics.storativity import estimate_regularized_ske
from .hydrostratigraphy.analysis import analyze_hydrostratigraphy
from .io.groundwater import prepare_groundwater
from .io.insar import prepare_insar
from .storage.annual_maps import export_annual_storage_maps
from .storage.budget import compute_storage_budget
from .synthesis import synthesize


CORE_STAGES = OrderedDict([
    ("prepare-insar", prepare_insar),
    ("prepare-groundwater", prepare_groundwater),
    ("build-groundwater-field", build_groundwater_field),
    ("decompose-insar", decompose_insar),
    ("joint-harmonics", compute_joint_harmonics),
    ("estimate-lag", estimate_lag),
    ("estimate-ske", estimate_pixelwise_ske),
    ("storage-budget", compute_storage_budget),
    ("annual-storage-maps", export_annual_storage_maps),
])

OPTIONAL_STAGES = OrderedDict([
    ("hydrostratigraphy", analyze_hydrostratigraphy),
    ("extensometer", analyze_extensometer),
    ("classify-deformation", classify_deformation),
    ("estimate-ske-regularized", estimate_regularized_ske),
    ("synthesize-legacy", synthesize),
])

STAGES = OrderedDict([
    *CORE_STAGES.items(),
    *OPTIONAL_STAGES.items(),
])


def run_stage(cfg: ProjectConfig, stage: str) -> dict[str, Any]:
    if stage not in STAGES:
        raise KeyError(f"Unknown stage {stage!r}. Choices: {list(STAGES)}")
    return STAGES[stage](cfg)


def run_pipeline(
    cfg: ProjectConfig,
    start: str | None = None,
    stop: str | None = None,
    only: str | None = None,
):
    """Run only the publication core unless an optional stage is explicit."""
    if only:
        if only not in CORE_STAGES:
            raise ValueError(
                "--only accepts core stages; use `hydrogeo-insar stage` "
                "for optional/legacy stages"
            )
        return {only: run_stage(cfg, only)}

    names = list(CORE_STAGES)
    i0 = names.index(start) if start else 0
    i1 = names.index(stop) if stop else len(names) - 1
    if i1 < i0:
        raise ValueError("stop stage occurs before start stage")

    results = {}
    for name in names[i0:i1 + 1]:
        results[name] = run_stage(cfg, name)
    return results
