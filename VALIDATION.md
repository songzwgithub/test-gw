# v0.2 validation

Validation was executed on a reproducible synthetic end-to-end case included in:

```text
tests/integration/test_full_pipeline.py
```

Synthetic truth:

- common-reference cumulative vertical InSAR GeoTIFF series;
- daily confined-head observations at 12 wells;
- annual groundwater forcing;
- deformation response lag = **50 days**;
- effective elastic skeletal storativity = **0.002**;
- two spatially different long-term deformation regimes.

Recovered by the complete pipeline through storage-budget:

- regional lag = **50.0 days**;
- median regularized Ske = **0.00199963**;
- groundwater spatial-CV RMSE = **0.0839 m** for the synthetic setup;
- storage identity satisfied numerically: `V_total = V_recoverable + V_irreversible`.

Automated tests:

```text
4 passed
```

Run with:

```bash
PYTHONPATH=src python -m pytest -q
```
