# v0.2 changes

- kept v0.1 groundwater and GeoTIFF input formats;
- enforced one common InSAR reference date and canonical uplift-positive sign;
- prohibited groundwater temporal extrapolation;
- added spatial block-CV for low-rank + RBF groundwater model selection;
- unified temporal design/fitting code;
- fitted seasonal deformation and groundwater on the same dates;
- implemented Meng-2026 feature set for deformation regimes;
- implemented phase-lag raster + regional lag scan;
- replaced raw `Ske` ratio as the final product with bounded coarse-grid Laplacian inversion;
- separated `Ske` observation support from storage analysis support;
- retained signed irreversible groundwater-storage change;
- added model-based annual storage increments;
- fixed partial-layer totals in hydrostratigraphy;
- fixed extensometer ordering and cumulative-marker conversion.
