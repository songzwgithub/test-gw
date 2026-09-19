# v0.3 validation

Validation uses the reproducible end-to-end synthetic case in:

```text
tests/integration/test_full_pipeline.py
```

Synthetic truth includes:

- common-reference cumulative vertical InSAR GeoTIFFs;
- daily confined-head observations at 12 wells;
- spatially varying annual groundwater forcing;
- deformation response lag = **50 days**;
- spatially constant effective elastic skeletal storativity = **0.002**;
- two different long-term deformation regimes;
- known piecewise-linear irreversible deformation, allowing analytical IGWS truth.

Latest local run recovered:

```text
regional lag                         50.0 days
median Ske                           0.00199583
groundwater spatial-CV RMSE          0.08391 m
groundwater annual amplitude RMSE    0.08716 m
groundwater phase MAE                0.11277 days
Ske CV deformation RMSE              8.00e-05 m
```

For the final synthetic storage interval:

```text
estimated irreversible GWS change  -1.57937e7 m3
analytical truth                    -1.57992e7 m3
relative error                       0.035 %
```

Automated tests:

```text
5 passed
```

Run with:

```bash
PYTHONPATH=src pytest -q
```
