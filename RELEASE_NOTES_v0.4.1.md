# hydrogeo-insar v0.4.1

v0.4.1 consolidates the publication workflow introduced in v0.4.0.

Key changes:

- pixelwise `Ske` is the primary hydrogeodetic product;
- hydrostratigraphic analysis uses pixelwise `Ske` by default;
- the streaming joint-harmonic implementation is consolidated in `seasonal.py`;
- the global analysis interval is applied consistently to joint harmonics and storage;
- `storage-budget` writes whole-interval and annual pixelwise storage maps in one fitting pass;
- `annual-storage-maps` independently reintegrates saved rasters and does not refit the temporal model;
- storage sensitivity is reported for multiple seasonal-vector-cosine thresholds;
- the end-to-end synthetic test extends through annual-map volume verification.

Generated scientific outputs are intentionally excluded from the source repository.
