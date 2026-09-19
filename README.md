# hydrogeo-insar v0.2

V0.2 is the first scientifically reworked version of the generic InSAR-groundwater workflow discussed for the Hengshui paper. It keeps the proven v0.1/test-gw input contracts and rebuilds the scientific core.

## Input contract

### InSAR

Only corrected **cumulative vertical deformation** GeoTIFFs are accepted:

```text
geo_YYYYMMDD_YYYYMMDD.tif
```

All files must share the same first date. The second date is the observation date. The software does **not** re-reference the InSAR time series. It only converts units/sign into the canonical convention:

```text
positive = uplift
negative = subsidence
unit     = mm
```

### Groundwater

Groundwater reading intentionally follows v0.1/test-gw: CSV/Excel, wide/long layouts, explicit aquifer labels. Well depth is never used to infer aquifer groups.

## Scientific changes from v0.1

1. groundwater model selection by spatial block cross-validation;
2. no temporal extrapolation outside observed groundwater support;
3. one shared temporal-function engine for InSAR and groundwater;
4. seasonal InSAR and groundwater harmonics are fitted on identical common epochs;
5. groundwater trend degree can be selected globally by AICc (linear vs quadratic);
6. Meng-2026-style deformation clustering uses `a, b, t_vertex, terminal_slope`;
7. annual lag is output both as a pixel phase-lag field and a regional effective lag;
8. `Ske` is a bounded spatially regularized inversion, not a raw pixel amplitude ratio;
9. storage analysis uses its own fixed domain and preserves signed irreversible GWS change;
10. true annual storage increments are separated from cumulative storage states;
11. hydrostratigraphic totals require all configured layers to be valid;
12. extensometer input explicitly distinguishes interval compaction and cumulative-marker displacement.

## Core equations

Annual response:

```text
d_A = Ske * R(tau) h_A
```

Jiang-style storage partition:

```text
V_total = V_recoverable + V_irreversible
V_recoverable = integral Ske * Delta h_confined dA
V_irreversible = V_total - V_recoverable
```

`V_irreversible` is signed. A negative value means irreversible storage depletion. `irreversible_storage_loss_magnitude_m3` is only a derived magnitude and is not used in the identity.

## Install and run

```bash
pip install -e .
hydrogeo-insar run configs/example_project.yaml
```

Stages:

```text
prepare-insar
prepare-groundwater
build-groundwater-field
decompose-insar
classify-deformation
joint-harmonics
estimate-lag
estimate-ske
storage-budget
hydrostratigraphy
extensometer
synthesize
```

Run from one stage:

```bash
hydrogeo-insar run configs/example_project.yaml --from joint-harmonics
```

## Current physical scope

V0.2 implements the confined-system main model used for the Hengshui workflow. The optional unconfined geostatic-loading correction discussed in Li et al. (2025) is intentionally not enabled yet because v0.2 keeps the existing input/data path unchanged. It can be added later without changing the current file readers.
