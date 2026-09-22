# v0.4.0

- changed the primary `estimate-ske` result from spatially regularized RBF `Ske` to pixelwise lag-aligned harmonic-vector least squares;
- retained a single quality-weighted regional lag for `Ske` inversion while keeping pixel phase lag diagnostic only;
- added primary and high-confidence pixelwise `Ske` products;
- changed `storage-budget` to consume pixelwise `Ske`;
- added annual pixelwise total, recoverable, irreversible and low-frequency head-change maps;
- added annual recovery-with-continued-compaction masks and area fractions;
- integrated the tested grouped-validity-mask least-squares optimization;
- added progress/ETA reporting to long-running hydrogeodetic stages;
- removed deformation clustering, regularized `Ske`, and legacy synthesis from the default run;
- retained those methods as explicit optional/legacy stages;
- cleaned transitional and validation scripts out of the repository root.

# v0.3.1

- prevented long unsupported groundwater gaps from being silently bridged;
- made storage reporting obey configured baseline/end dates;
- added groundwater harmonic CV error to lag/Ske uncertainty diagnostics;
- added stage-by-stage visualization.

# v0.3

- introduced groundwater-field reconstruction;
- introduced common-epoch seasonal harmonics and regional lag;
- introduced continuous piecewise-linear low-frequency storage partition;
- used spatially regularized `Ske` as the original v0.3 primary method.
