# v0.4.1 validation

The publication workflow is validated with the reproducible end-to-end
synthetic case in:

```text
tests/integration/test_full_pipeline.py
```

The synthetic truth contains:

- common-reference cumulative vertical InSAR GeoTIFFs;
- daily confined-head observations at 12 wells;
- spatially varying annual groundwater forcing;
- deformation-response lag of 50 days;
- spatially constant pixelwise effective elastic skeletal storage coefficient
  `Ske = 0.002`;
- known piecewise-linear irreversible deformation.

The integration test runs through:

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

It checks recovery of the synthetic lag and `Ske`, analytical irreversible
storage change, the storage identity, finite annual estimates, annual
GeoTIFF/CSV volume consistency, and result-check plotting.

In v0.4.1 `storage-budget` writes annual maps during the same temporal-fitting
pass used for the regional storage budget. `annual-storage-maps` is an
independent raster reintegration check and does not refit the temporal model.

Run:

```bash
PYTHONPATH=src pytest -q
```

A release is acceptable only when the complete test suite passes.

## Interpretation checks

The storage partition follows the configured Jiang-style decomposition:

```text
Delta b_total
= Delta b_recoverable + Delta b_irreversible

Delta b_recoverable
= Ske * Delta h_lowfreq
```

`Ske` is the lag-aligned annual-harmonic scale factor and is reported
pixelwise. `storage_ske_cosine_sensitivity.csv` reports storage estimates for
multiple seasonal-vector-cosine support thresholds so the primary product can
be compared with stricter seasonal-coupling definitions.

Residual/irreversible deformation is an equivalent hydromechanical residual;
it should not be interpreted as independently proven permanent storage loss
without hydrostratigraphic or extensometer evidence.
