from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import rasterio

from ..common import affine_to_list, dates_to_days, ensure_dir, write_json
from ..config import ProjectConfig

DEFAULT_REGEX = r"geo_(\d{8})_(\d{8})\.tif$"
UNIT_TO_MM = {"mm": 1.0, "cm": 10.0, "m": 1000.0}


def discover_insar_files(directory: Path, pattern: str = "geo_*.tif", filename_regex: str = DEFAULT_REGEX):
    rx = re.compile(filename_regex)
    rows = []
    for path in sorted(directory.glob(pattern)):
        m = rx.search(path.name)
        if not m:
            continue
        ref = np.datetime64(f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}")
        obs = np.datetime64(f"{m.group(2)[:4]}-{m.group(2)[4:6]}-{m.group(2)[6:8]}")
        rows.append((obs, ref, path))
    rows.sort(key=lambda x: x[0])
    if not rows:
        raise FileNotFoundError(f"No InSAR GeoTIFF matched {pattern!r} with regex {filename_regex!r} in {directory}")
    return rows


def _canonical_sign(positive: str) -> float:
    value = str(positive).strip().lower()
    if value in {"uplift", "up", "positive_up", "positive=uplift"}:
        return 1.0
    if value in {"subsidence", "down", "positive_down", "positive=subsidence"}:
        return -1.0
    raise ValueError("insar.positive must state whether positive values mean 'uplift' or 'subsidence'")


def prepare_insar(cfg: ProjectConfig) -> dict[str, Any]:
    """Read corrected cumulative deformation GeoTIFFs into a canonical HDF5 stack.

    V0.2 input contract:
      * files are cumulative deformation relative to one common source reference date;
      * values are already atmospherically/geodetically corrected upstream;
      * storage calculations require vertical_displacement;
      * internal sign convention is always positive=uplift, negative=subsidence.
    """
    sec = cfg.section("insar")
    input_dir = cfg.resolve(sec["path"])
    pattern = sec.get("pattern", "geo_*.tif")
    regex = sec.get("filename_regex", DEFAULT_REGEX)
    unit = str(sec.get("unit", "mm")).lower()
    if unit not in UNIT_TO_MM:
        raise ValueError(f"Unsupported InSAR unit: {unit}")
    scale = UNIT_TO_MM[unit]
    sign = _canonical_sign(sec.get("positive", "uplift"))
    files = discover_insar_files(input_dir, pattern, regex)

    refs = np.asarray([r[1] for r in files], dtype="datetime64[D]")
    unique_refs = np.unique(refs)
    if len(unique_refs) != 1:
        raise ValueError(
            "InSAR input must be a cumulative time series with one common reference date. "
            f"Found {len(unique_refs)} reference dates in file names."
        )
    obs = np.asarray([r[0] for r in files], dtype="datetime64[D]")
    if len(np.unique(obs)) != len(obs):
        raise ValueError("Duplicate InSAR observation dates were found")

    out_dir = ensure_dir(cfg.outputs / "canonical")
    out_path = out_dir / "insar_stack.h5"
    block = int(sec.get("block_size", 256))

    with rasterio.open(files[0][2]) as src0:
        height, width = src0.height, src0.width
        transform, crs = src0.transform, str(src0.crs)
        if crs in {"None", ""}:
            raise ValueError("InSAR GeoTIFFs require a valid CRS")

    with h5py.File(out_path, "w") as h5:
        ds = h5.create_dataset(
            "displacement_mm",
            shape=(len(files), height, width),
            dtype="float32",
            chunks=(1, min(block, height), min(block, width)),
            compression="gzip",
            compression_opts=4,
            fillvalue=np.nan,
        )
        h5.create_dataset("date_days", data=dates_to_days(obs))
        h5.create_dataset("source_reference_date_days", data=dates_to_days(refs))
        h5.attrs["height"] = height
        h5.attrs["width"] = width
        h5.attrs["crs"] = crs
        h5.attrs["transform"] = affine_to_list(transform)
        h5.attrs["unit"] = "mm"
        h5.attrs["quantity"] = sec.get("quantity", "vertical_displacement")
        h5.attrs["positive"] = "uplift"
        h5.attrs["source_positive"] = sec.get("positive", "uplift")
        h5.attrs["common_reference_date"] = str(unique_refs[0])
        h5.attrs["input_semantics"] = "cumulative_deformation_relative_to_common_reference_date"

        for i, (_, _, path) in enumerate(files):
            with rasterio.open(path) as src:
                if src.height != height or src.width != width or src.transform != transform or str(src.crs) != crs:
                    raise ValueError(f"Raster geometry differs from first InSAR file: {path}")
                arr = src.read(1).astype("float32")
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
                ds[i] = arr * scale * sign

    manifest = {
        "status": "ok",
        "input_directory": str(input_dir),
        "output": str(out_path),
        "n_epochs": len(files),
        "first_date": str(obs[0]),
        "last_date": str(obs[-1]),
        "reference_date": str(unique_refs[0]),
        "shape": [height, width],
        "crs": crs,
        "unit": "mm",
        "quantity": sec.get("quantity", "vertical_displacement"),
        "positive": "uplift",
    }
    write_json(out_dir / "insar_manifest.json", manifest)
    return manifest
