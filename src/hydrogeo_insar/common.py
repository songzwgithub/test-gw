from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS, Geod
from rasterio.warp import Resampling, reproject


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def dates_to_days(dates: Iterable[np.datetime64]) -> np.ndarray:
    arr = np.asarray(list(dates), dtype="datetime64[D]")
    return (arr - np.datetime64("1970-01-01", "D")).astype(np.int32)


def days_to_dates(days: np.ndarray) -> np.ndarray:
    return np.datetime64("1970-01-01", "D") + np.asarray(days, dtype="timedelta64[D]")


def affine_to_list(transform: Affine) -> list[float]:
    return [transform.a, transform.b, transform.c, transform.d, transform.e, transform.f]


def affine_from_list(values: list[float] | tuple[float, ...]) -> Affine:
    return Affine(*[float(v) for v in values])


def h5_grid_metadata(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as h5:
        return {
            "height": int(h5.attrs["height"]),
            "width": int(h5.attrs["width"]),
            "crs": str(h5.attrs["crs"]),
            "transform": affine_from_list(h5.attrs["transform"]),
        }


def geotiff_profile(height: int, width: int, crs: str, transform: Affine, dtype: str = "float32", nodata=np.nan) -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "deflate",
        "tiled": True,
        "BIGTIFF": "IF_SAFER",
    }


def write_tif(path: str | Path, array: np.ndarray, crs: str, transform: Affine, nodata=np.nan, dtype="float32") -> None:
    path = Path(path)
    ensure_dir(path.parent)
    arr = np.asarray(array)
    profile = geotiff_profile(arr.shape[0], arr.shape[1], crs, transform, dtype=dtype, nodata=nodata)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype, copy=False), 1)


def aligned_raster(path: str | Path, dst_height: int, dst_width: int, dst_crs: str, dst_transform: Affine, resampling=Resampling.bilinear) -> np.ndarray:
    with rasterio.open(path) as src:
        src_arr = src.read(1).astype("float32")
        if src.nodata is not None:
            src_arr[src_arr == src.nodata] = np.nan
        out = np.full((dst_height, dst_width), np.nan, dtype="float32")
        reproject(
            source=src_arr,
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return out


def pixel_area_rows(height: int, width: int, crs: str, transform: Affine) -> np.ndarray:
    """Return pixel area (m²) for each raster row, assuming north-up rasters."""
    crs_obj = CRS.from_user_input(crs)
    if not crs_obj.is_geographic:
        return np.full(height, abs(transform.a * transform.e - transform.b * transform.d), dtype=float)

    geod = Geod(ellps="WGS84")
    areas = np.empty(height, dtype=float)
    x0 = transform.c
    x1 = transform.c + transform.a
    for r in range(height):
        y0 = transform.f + transform.e * r
        y1 = transform.f + transform.e * (r + 1)
        lons = [x0, x1, x1, x0]
        lats = [y0, y0, y1, y1]
        area, _ = geod.polygon_area_perimeter(lons, lats)
        areas[r] = abs(area)
    return areas


def block_slices(height: int, width: int, block_size: int):
    for r0 in range(0, height, block_size):
        r1 = min(height, r0 + block_size)
        for c0 in range(0, width, block_size):
            c1 = min(width, c0 + block_size)
            yield r0, r1, c0, c1
