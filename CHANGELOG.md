# v0.3.1 changes

- prevent long unsupported groundwater time gaps from being silently bridged by temporal interpolation;
- make storage cumulative and annual reporting obey the configured baseline/end interval;
- add groundwater spatial-CV harmonic error as an uncertainty floor in regional lag and Ske weighting;
- add lag-corrected seasonal vector-coherence diagnostics to the Ske inversion;
- add stage-by-stage visualization via `hydrogeo-insar plot`;
- tighten the synthetic storage regression tolerance and add a long-gap temporal support test.

# v0.3 changes

- preserved the verified v0.2/test-gw data readers;
- added minimum active-well temporal support for groundwater field construction;
- expanded groundwater spatial block-CV to annual amplitude, phase, harmonic-vector and trend diagnostics;
- changed groundwater model selection to full-series-RMSE shortlist + harmonic-vector criterion;
- added continuous piecewise-linear time functions;
- fixed the near-zero-curvature deformation vertex feature;
- changed K-means training to spatially balanced sampling with chunked full-raster prediction;
- separated lag estimation into its own module and added fit-quality weighting;
- replaced pixel-stride/coarse-cell Ske with a continuous physical-km normalized-RBF basis inversion;
- separated seasonal-data support from Ske solution support;
- limited Ske solution support by physical distance from seasonal observations;
- selected Ske node spacing and regularization by spatial block-CV;
- replaced quadratic-derived annual storage changes with piecewise-linear + annual low-frequency changes;
- retained signed IGWS and added separate net/gross loss diagnostics;
- synthesis now uses low-frequency groundwater recovery rather than the quadratic model's linear coefficient.
