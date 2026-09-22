# hydrogeo-insar v0.4.0

`hydrogeo-insar` is a reproducible hydrogeodetic workflow for combining corrected cumulative InSAR time series with groundwater observations. The v0.4 publication workflow makes **pixelwise** `Ske` and pixelwise storage partition the primary scientific products.

## Publication workflow

```text
corrected vertical InSAR
        +
groundwater observations
        |
        +--> regional groundwater field
        |
        +--> common-epoch annual harmonics
        |
        +--> quality-weighted regional lag
        |
        +--> pixelwise Ske
        |
        +--> piecewise-linear low-frequency deformation/head
        |
        +--> pixelwise total / recoverable / irreversible change
        |
        +--> annual pixelwise maps + regional volume integration
```

The default workflow deliberately excludes deformation clustering and the older regularized-RBF `Ske` estimator. Those are retained only as optional sensitivity/legacy stages.

## Core equations

For each pixel, with the groundwater annual harmonic rotated by the regional lag:

```text
Ske = (d · h_tau) / (h_tau · h_tau)
```

The low-frequency storage partition is:

```text
Delta b_total        = observed low-frequency vertical deformation
Delta b_recoverable  = Ske * Delta h_lowfreq
Delta b_irreversible = Delta b_total - Delta b_recoverable
```

Regional volumes are integrated **after** pixelwise calculation:

```text
V = sum(Delta b * pixel_area)
```

Sign convention: positive displacement is uplift; negative is subsidence. Negative irreversible change denotes residual compaction/storage depletion within the adopted Jiang-style partition.

## Install and run

```bash
pip install -e .
hydrogeo-insar run configs/example_project.yaml
```

Continue from the hydrogeodetic core:

```bash
hydrogeo-insar run configs/example_project.yaml --from joint-harmonics
```

List stages:

```bash
hydrogeo-insar list-stages
```

## Core stages

```text
prepare-insar
prepare-groundwater
build-groundwater-field
decompose-insar
joint-harmonics
estimate-lag
estimate-ske
storage-budget
annual-storage-maps
```

Optional/legacy stages:

```text
hydrostratigraphy
extensometer
classify-deformation
estimate-ske-regularized
synthesize-legacy
```

## Main outputs

```text
outputs/seasonal/
  lag_summary.json
  phase_lag_days.tif
  seasonal_vector_cosine.tif
  ske_pixelwise_raw_ratio.tif
  ske_pixelwise.tif
  ske_pixelwise_support_mask.tif
  ske_pixelwise_high_confidence.tif
  ske_pixelwise_high_confidence_mask.tif
  ske_pixelwise_summary.json

outputs/storage/
  storage_domain_mask.tif
  total_gws_change_equivalent_mm.tif
  recoverable_gws_change_equivalent_mm.tif
  irreversible_gws_change_equivalent_mm.tif
  negative_irreversible_change_magnitude_mm.tif
  head_lowfreq_change_m.tif
  storage_cumulative_observed.csv
  storage_annual_change.csv
  storage_ske_cosine_sensitivity.csv
  annual_maps_summary.csv
  annual_maps/
    YYYY_total_change_mm.tif
    YYYY_recoverable_change_mm.tif
    YYYY_irreversible_change_mm.tif
    YYYY_head_lowfreq_change_m.tif
    YYYY_recovery_with_continued_compaction.tif
```

## Performance

Large least-squares stages group pixels by identical temporal-validity masks. A pseudoinverse is solved once for each unique mask rather than independently for every pixel. `storage-budget` writes whole-interval and annual pixelwise maps in one fitting pass; `annual-storage-maps` independently reintegrates the saved rasters for consistency checking and does not refit the temporal model. Long-running stages print progress and ETA.

## Interpretation

`Ske` is an effective skeletal storage coefficient inferred from the annual groundwater/deformation response. The storage partition is an equivalent hydromechanical decomposition; it is not direct pumping volume. Residual deformation should not be interpreted as proven permanent storage loss without independent hydrostratigraphic or extensometer support.

Generated `outputs*` directories and local raw data are not part of the source release.
