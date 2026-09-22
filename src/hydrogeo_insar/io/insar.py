from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path
from typing import Any

import fiona
import h5py
import numpy as np
import rasterio
from rasterio.features import bounds as geometry_bounds
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from rasterio.warp import transform_geom

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


def _load_domain(cfg: ProjectConfig, raster_crs: str, raster_transform, raster_height: int, raster_width: int):
    sec = cfg.section("domain")
    boundary_value = sec.get("boundary_path")
    if not boundary_value:
        mask = np.ones((raster_height, raster_width), dtype=bool)
        return Window(0, 0, raster_width, raster_height), raster_transform, mask, None

    boundary = cfg.resolve(boundary_value)
    if not boundary.exists():
        raise FileNotFoundError(f"Domain boundary does not exist: {boundary}")

    geoms = []
    with fiona.open(boundary) as src:
        src_crs = src.crs_wkt or src.crs
        if not src_crs:
            raise ValueError(f"Domain boundary has no CRS: {boundary}")
        for feat in src:
            geom = feat.get("geometry")
            if geom:
                geoms.append(transform_geom(src_crs, raster_crs, geom, precision=-1))

    if not geoms:
        raise ValueError(f"Domain boundary contains no geometry: {boundary}")

    bs = [geometry_bounds(g) for g in geoms]
    left = min(b[0] for b in bs)
    bottom = min(b[1] for b in bs)
    right = max(b[2] for b in bs)
    top = max(b[3] for b in bs)

    raw = from_bounds(left, bottom, right, top, transform=raster_transform)
    c0 = max(0, int(np.floor(raw.col_off)))
    r0 = max(0, int(np.floor(raw.row_off)))
    c1 = min(raster_width, int(np.ceil(raw.col_off + raw.width)))
    r1 = min(raster_height, int(np.ceil(raw.row_off + raw.height)))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("Domain boundary does not intersect the InSAR raster")

    window = Window(c0, r0, c1 - c0, r1 - r0)
    transform = rasterio.windows.transform(window, raster_transform)
    mask = geometry_mask(
        geoms,
        out_shape=(int(window.height), int(window.width)),
        transform=transform,
        invert=True,
        all_touched=bool(sec.get("all_touched", False)),
    )
    return window, transform, mask, str(boundary)


def _read_one_fast(path: Path, source_height: int, source_width: int, source_transform, source_crs: str,
                   window: Window, domain_mask: np.ndarray, factor: np.float32, gdal_num_threads: str) -> np.ndarray:
    with rasterio.Env(GDAL_NUM_THREADS=str(gdal_num_threads)):
        with rasterio.open(path, sharing=False) as src:
            if src.height != source_height or src.width != source_width or src.transform != source_transform or str(src.crs) != source_crs:
                raise ValueError(f"Raster geometry differs from first InSAR file: {path}")
            arr = src.read(1, window=window, out_dtype="float32")
            if src.nodata is not None and np.isfinite(src.nodata):
                arr[arr == src.nodata] = np.nan
    if factor != np.float32(1.0):
        arr *= factor
    arr[~domain_mask] = np.nan
    return arr


def _write_domain_mask(path: Path, mask: np.ndarray, crs: str, transform):
    profile = {
        "driver": "GTiff",
        "height": mask.shape[0],
        "width": mask.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "nodata": 0,
        "BIGTIFF": "IF_SAFER",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype("uint8"), 1)


def prepare_insar(cfg: ProjectConfig) -> dict[str, Any]:
    sec = cfg.section("insar")
    input_dir = cfg.resolve(sec["path"])
    pattern = sec.get("pattern", "geo_*.tif")
    regex = sec.get("filename_regex", DEFAULT_REGEX)
    unit = str(sec.get("unit", "mm")).lower()
    if unit not in UNIT_TO_MM:
        raise ValueError(f"Unsupported InSAR unit: {unit}")

    factor = np.float32(UNIT_TO_MM[unit] * _canonical_sign(sec.get("positive", "uplift")))
    files = discover_insar_files(input_dir, pattern, regex)

    refs = np.asarray([r[1] for r in files], dtype="datetime64[D]")
    unique_refs = np.unique(refs)
    if len(unique_refs) != 1:
        raise ValueError(f"InSAR input must have one common reference date; found {len(unique_refs)}")
    obs = np.asarray([r[0] for r in files], dtype="datetime64[D]")
    if len(np.unique(obs)) != len(obs):
        raise ValueError("Duplicate InSAR observation dates were found")

    out_dir = ensure_dir(cfg.outputs / "canonical")
    out_path = out_dir / "insar_stack.h5"
    domain_mask_path = out_dir / "domain_mask.tif"
    block = int(sec.get("block_size", 256))

    with rasterio.open(files[0][2]) as src0:
        source_height = src0.height
        source_width = src0.width
        source_transform = src0.transform
        source_crs = str(src0.crs)
        if source_crs in {"None", ""}:
            raise ValueError("InSAR GeoTIFFs require a valid CRS")

    window, transform, domain_mask, domain_path = _load_domain(
        cfg, source_crs, source_transform, source_height, source_width
    )
    height, width = domain_mask.shape
    _write_domain_mask(domain_mask_path, domain_mask, source_crs, transform)

    compression = str(sec.get("hdf5_compression", "none")).strip().lower()
    if compression in {"", "none", "null", "off"}:
        compression_kwargs = {}
    elif compression == "lzf":
        compression_kwargs = {"compression": "lzf"}
    elif compression == "gzip":
        compression_kwargs = {"compression": "gzip", "compression_opts": int(sec.get("hdf5_gzip_level", 1))}
    else:
        raise ValueError("insar.hdf5_compression must be one of: none, lzf, gzip")

    read_workers = max(1, int(sec.get("read_workers", 8)))
    prefetch = max(read_workers, int(sec.get("prefetch", 16)))
    gdal_num_threads = str(sec.get("gdal_num_threads", "2"))
    cache_mb = max(1, int(sec.get("hdf5_cache_mb", 128)))

    print(
        f"prepare-insar domain: source={source_height}x{source_width}, crop={height}x{width}, "
        f"inside={int(domain_mask.sum()):,}/{domain_mask.size:,} "
        f"({100.0*domain_mask.mean():.1f}%), boundary={domain_path}",
        flush=True,
    )

    with h5py.File(
        out_path, "w", libver="latest",
        rdcc_nbytes=cache_mb * 1024 * 1024, rdcc_nslots=1000003,
    ) as h5:
        ds = h5.create_dataset(
            "displacement_mm",
            shape=(len(files), height, width),
            dtype="float32",
            chunks=(1, min(block, height), min(block, width)),
            fillvalue=np.nan,
            **compression_kwargs,
        )
        h5.create_dataset("date_days", data=dates_to_days(obs))
        h5.create_dataset("source_reference_date_days", data=dates_to_days(refs))
        h5.attrs["height"] = height
        h5.attrs["width"] = width
        h5.attrs["crs"] = source_crs
        h5.attrs["transform"] = affine_to_list(transform)
        h5.attrs["unit"] = "mm"
        h5.attrs["quantity"] = sec.get("quantity", "vertical_displacement")
        h5.attrs["positive"] = "uplift"
        h5.attrs["source_positive"] = sec.get("positive", "uplift")
        h5.attrs["common_reference_date"] = str(unique_refs[0])
        h5.attrs["input_semantics"] = "cumulative_deformation_relative_to_common_reference_date"
        h5.attrs["domain_boundary"] = domain_path or ""
        h5.attrs["domain_mask"] = str(domain_mask_path)

        def read_idx(i: int):
            return _read_one_fast(
                files[i][2], source_height, source_width, source_transform, source_crs,
                window, domain_mask, factor, gdal_num_threads
            )

        if read_workers == 1:
            for i in range(len(files)):
                ds[i] = read_idx(i)
                if (i + 1) % 20 == 0 or i + 1 == len(files):
                    print(f"prepare-insar: {i+1}/{len(files)}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=read_workers) as pool:
                futures = {}
                next_submit = 0
                while next_submit < len(files) and len(futures) < prefetch:
                    futures[next_submit] = pool.submit(read_idx, next_submit)
                    next_submit += 1
                for i in range(len(files)):
                    arr = futures.pop(i).result()
                    ds[i] = arr
                    del arr
                    while next_submit < len(files) and len(futures) < prefetch:
                        futures[next_submit] = pool.submit(read_idx, next_submit)
                        next_submit += 1
                    if (i + 1) % 20 == 0 or i + 1 == len(files):
                        print(f"prepare-insar: {i+1}/{len(files)}", flush=True)
        h5.flush()

    manifest = {
        "status": "ok",
        "input_directory": str(input_dir),
        "output": str(out_path),
        "n_epochs": len(files),
        "first_date": str(obs[0]),
        "last_date": str(obs[-1]),
        "reference_date": str(unique_refs[0]),
        "source_shape": [source_height, source_width],
        "shape": [height, width],
        "crs": source_crs,
        "unit": "mm",
        "quantity": sec.get("quantity", "vertical_displacement"),
        "positive": "uplift",
        "domain_boundary": domain_path,
        "domain_mask": str(domain_mask_path),
        "domain_pixels": int(domain_mask.sum()),
        "hdf5_compression": compression,
    }
    write_json(out_dir / "insar_manifest.json", manifest)
    return manifest
