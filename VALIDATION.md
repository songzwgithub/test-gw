# v0.1 validation

The first version was smoke-tested with a synthetic 20×20 raster, 31 InSAR epochs, and 12 confined wells.

Synthetic truth used:

- groundwater annual forcing with a 50-day deformation lag;
- effective skeletal storativity `Ske = 0.002`;
- spatially varying long-term subsidence.

Recovered by the full pipeline:

- lag = `50.0 days`;
- median `Ske = 0.00201145`;
- all stages from InSAR ingestion through storage budget and synthesis completed successfully.

Unit tests: `3 passed`.
