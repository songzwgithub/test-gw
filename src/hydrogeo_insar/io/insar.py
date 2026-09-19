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


def prepare_insar(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("insar")
    input_dir = cfg.resolve(sec["path"])
    pattern = sec.get("pattern", "geo_*.tif")
    regex = sec.get("filename_regex", DEFAULT_REGEX)
    unit = str(sec.get("unit", "mm")).lower()
    if unit not in UNIT_TO_MM:
        raise ValueError(f"Unsupported InSAR unit: {unit}")
    scale = UNIT_TO_MM[unit]
    files = discover_insar_files(input_dir, pattern, regex)

    out_dir = ensure_dir(cfg.outputs / "canonical")
    out_path = out_dir / "insar_stack.h5"
    block = int(sec.get("block_size", 256))

    with rasterio.open(files[0][2]) as src0:
        height, width = src0.height, src0.width
        transform, crs = src0.transform, str(src0.crs)

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
        h5.create_dataset("date_days", data=dates_to_days([r[0] for r in files]))
        h5.create_dataset("source_reference_date_days", data=dates_to_days([r[1] for r in files]))
        h5.attrs["height"] = height
        h5.attrs["width"] = width
        h5.attrs["crs"] = crs
        h5.attrs["transform"] = affine_to_list(transform)
        h5.attrs["unit"] = "mm"
        h5.attrs["quantity"] = sec.get("quantity", "vertical_displacement")
        h5.attrs["positive"] = sec.get("positive", "uplift")

        for i, (_, _, path) in enumerate(files):
            with rasterio.open(path) as src:
                if src.height != height or src.width != width or src.transform != transform or str(src.crs) != crs:
                    raise ValueError(f"Raster geometry differs from first InSAR file: {path}")
                arr = src.read(1).astype("float32")
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
                ds[i] = arr * scale

    manifest = {
        "status": "ok",
        "input_directory": str(input_dir),
        "output": str(out_path),
        "n_epochs": len(files),
        "first_date": str(files[0][0]),
        "last_date": str(files[-1][0]),
        "shape": [height, width],
        "crs": crs,
        "unit": "mm",
        "positive": sec.get("positive", "uplift"),
    }
    write_json(out_dir / "insar_manifest.json", manifest)
    return manifest
