# hydrogeo-insar v0.3.1

V0.3 keeps the verified `test-gw` input formats and replaces the scientific core with a continuous-field hydrogeodetic workflow. The code is designed for corrected cumulative vertical InSAR GeoTIFF time series and regional groundwater monitoring networks.

## Input contract

### InSAR

```text
geo_YYYYMMDD_YYYYMMDD.tif
```

All files are cumulative deformation relative to one common first date. The software does not re-reference the time series. Internally:

```text
positive = uplift
negative = subsidence
unit     = mm
```

### Groundwater

Groundwater readers remain compatible with the verified v0.2/test-gw CSV/Excel wide/long formats. Aquifer classes are explicit; well depth is not used to infer aquifer group.

## Scientific workflow

```text
corrected cumulative InSAR
        +
confined groundwater observations
        |
        +--> low-rank temporal model + Gaussian RBF spatial field
        |
        +--> quadratic + annual deformation model --> Meng-style regimes
        |
        +--> common-epoch annual harmonics --> regional lag
        |
        +--> continuous bounded Ske field on physical-km basis nodes
        |
        +--> piecewise-linear low-frequency deformation/head
        |
        +--> TGWS / RGWS / IGWS (Jiang-style partition)
```

## Main v0.3 changes

1. Groundwater spatial model selection uses spatial block-CV and reports full-series RMSE, annual-amplitude RMSE, phase MAE and harmonic-vector RMSE.
2. Groundwater model selection first keeps models within 5% of the minimum full-series RMSE, then selects the best annual harmonic reconstruction.
3. Time functions support continuous piecewise-linear hinges in addition to polynomial and periodic terms.
4. Deformation clustering keeps the Meng-style feature set but trains on a spatially balanced sample and predicts the full raster in chunks.
5. The near-zero-curvature `t_vertex` feature bug is fixed.
6. Pixel phase lag remains a diagnostic map; one quality-weighted regional lag is used in the Ske inversion.
7. `Ske` is no longer estimated on pixel-count coarse cells. It is represented as a continuous normalized-RBF basis field with node spacing in kilometres, bounded coefficients and graph smoothing.
8. Ske data support, Ske solution support and groundwater support are separate products. Ske extrapolation is limited by a physical distance from seasonal observations.
9. Annual TGWS/RGWS/IGWS use a continuous piecewise-linear low-frequency model plus annual harmonic, rather than differences from one full-period quadratic trend.
10. Storage output distinguishes signed irreversible change, net irreversible-loss magnitude and gross negative irreversible change.


## v0.3.1 science fixes

- Groundwater temporal interpolation no longer bridges long periods rejected by the active-well support criterion.
- Regional lag and Ske weighting now include the held-out groundwater harmonic-vector CV RMSE as an uncertainty floor.
- Ske fitting reports and uses the lag-corrected seasonal vector cosine so non-coherent seasonal response is not forced into near-zero storativity.
- Storage cumulative and annual outputs now obey the configured `baseline_date` to `end_date` interval exactly.
- Added an independent visualization module for stage-by-stage result checks. Scientific calculations do not depend on plotting.

Plot all available checks:

```bash
hydrogeo-insar plot configs/example_project.yaml --stage all
```

Plot one stage:

```bash
hydrogeo-insar plot configs/example_project.yaml --stage estimate-ske
```

Figures are written to `outputs/figures/checks/` by default.

## Core equations

Seasonal response:

```text
d_A(x) = Ske(x) * R(tau_region) * h_A(x)
```

Continuous Ske parameterization:

```text
Ske(x) = sum_j B_j(x) beta_j
```

where normalized RBF basis functions satisfy `B_j >= 0` and `sum_j B_j = 1`. The inversion minimizes harmonic deformation misfit plus graph smoothing with bounded `beta_j`.

Jiang-style storage partition:

```text
V_total       = integral Delta d_lowfreq dA
V_recoverable = integral Ske * Delta h_lowfreq dA
V_irreversible = V_total - V_recoverable
```

`V_irreversible` remains signed. Negative values indicate irreversible storage depletion.

## Run

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

Run to one stage:

```bash
hydrogeo-insar run configs/example_project.yaml --to estimate-ske
```

Continue from one stage:

```bash
hydrogeo-insar run configs/example_project.yaml --from storage-budget
```

## Main outputs

```text
outputs/groundwater/groundwater_field.h5
outputs/groundwater/groundwater_model_cv.csv
outputs/regimes/deformation_regime_id.tif
outputs/seasonal/phase_lag_days.tif
outputs/seasonal/ske_effective.tif
outputs/seasonal/ske_data_support_mask.tif
outputs/seasonal/ske_support_mask.tif
outputs/seasonal/ske_model_cv.csv
outputs/storage/storage_annual_change.csv
outputs/storage/total_gws_change_equivalent_mm.tif
outputs/storage/recoverable_gws_change_equivalent_mm.tif
outputs/storage/irreversible_gws_change_equivalent_mm.tif
outputs/storage/head_lowfreq_change_m.tif
```

The code intentionally avoids release hashes, fixed expected scientific values and large audit/gate systems. Only checks needed to preserve data semantics and mathematical validity are retained.
