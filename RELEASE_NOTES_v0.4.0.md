# v0.4.0 release refactor

This overlay reorganizes the tested Hengshui development scripts into package modules and leaves generated outputs untouched.

The following root-level transitional scripts are superseded after applying the overlay:

- `joint_harmonics_fast.py`
- `estimate_ske_pixelwise.py`
- `storage_budget_pixelwise_fast.py`
- `export_annual_pixelwise_storage_maps_fast.py`
- `run_pixelwise_ske_storage.sh`
- `run_export_annual_pixelwise_storage_maps.sh`
- `run_remaining_core_pipeline.sh`

Validation utilities are moved to `scripts/validation/`.

Before a public release:
1. run `pytest -q`;
2. run `hydrogeo-insar list-stages`;
3. smoke-test `joint-harmonics` through `annual-storage-maps`;
4. confirm `outputs*` is ignored by Git;
5. add an explicit repository license;
6. tag the tested commit as `v0.4.0`.
