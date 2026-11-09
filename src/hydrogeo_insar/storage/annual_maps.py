from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import rasterio

from ..common import (
    block_slices,
    h5_grid_metadata,
    pixel_area_rows,
)
from ..config import ProjectConfig


def _integrate_map(
    path,
    area_rows: np.ndarray,
    block_size: int,
):
    total = 0.0
    n = 0

    with rasterio.open(path) as src:
        for r0, r1, c0, c1 in block_slices(
            src.height,
            src.width,
            block_size,
        ):
            window = rasterio.windows.Window(
                c0,
                r0,
                c1 - c0,
                r1 - r0,
            )
            arr = src.read(
                1,
                window=window,
            ).astype(float)

            good = np.isfinite(arr)
            if not good.any():
                continue

            area = np.broadcast_to(
                area_rows[r0:r1, None],
                arr.shape,
            )

            total += float(
                np.sum(
                    arr[good]
                    / 1000.0
                    * area[good]
                )
            )
            n += int(good.sum())

    return total, n


def export_annual_storage_maps(
    cfg: ProjectConfig,
) -> dict[str, Any]:
    """Verify annual maps written by ``storage-budget``.

    v0.4.1 writes annual maps during the storage fitting pass. This stage no
    longer refits the temporal model; it independently reintegrates the saved
    rasters and checks them against ``storage_annual_change.csv``.
    """
    sec = cfg.section("storage")
    block_size = int(
        sec.get(
            "annual_map_verify_block_size",
            512,
        )
    )

    storage_dir = cfg.outputs / "storage"
    annual_dir = storage_dir / "annual_maps"
    annual_csv = (
        storage_dir / "storage_annual_change.csv"
    )
    summary_csv = (
        storage_dir / "annual_maps_summary.csv"
    )

    if not annual_csv.exists():
        raise FileNotFoundError(
            "storage_annual_change.csv is missing; "
            "run storage-budget first"
        )

    if not summary_csv.exists():
        raise FileNotFoundError(
            "annual_maps_summary.csv is missing; "
            "run storage-budget with "
            "storage.export_annual_maps=true"
        )

    table = pd.read_csv(annual_csv)

    grid = h5_grid_metadata(
        cfg.outputs
        / "canonical"
        / "insar_stack.h5"
    )
    area_rows = pixel_area_rows(
        grid["height"],
        grid["width"],
        grid["crs"],
        grid["transform"],
    )

    rows = []

    for record in table.to_dict(
        orient="records"
    ):
        year = int(record["year"])

        paths = {
            "total":
                annual_dir
                / f"{year}_total_change_mm.tif",
            "recoverable":
                annual_dir
                / f"{year}_recoverable_change_mm.tif",
            "irreversible":
                annual_dir
                / f"{year}_irreversible_change_mm.tif",
        }

        for path in paths.values():
            if not path.exists():
                raise FileNotFoundError(
                    f"Annual storage map missing: {path}"
                )

        vt, nt = _integrate_map(
            paths["total"],
            area_rows,
            block_size,
        )
        vr, nr = _integrate_map(
            paths["recoverable"],
            area_rows,
            block_size,
        )
        vi, ni = _integrate_map(
            paths["irreversible"],
            area_rows,
            block_size,
        )

        rows.append(
            {
                "year": year,
                "total_gws_change_m3_maps": vt,
                "recoverable_gws_change_m3_maps": vr,
                "irreversible_gws_change_m3_maps": vi,
                "total_gws_change_m3_previous":
                    float(
                        record[
                            "total_gws_change_m3"
                        ]
                    ),
                "recoverable_gws_change_m3_previous":
                    float(
                        record[
                            "recoverable_gws_change_m3"
                        ]
                    ),
                "irreversible_gws_change_m3_previous":
                    float(
                        record[
                            "irreversible_gws_change_m3"
                        ]
                    ),
                "map_pixels_total": nt,
                "map_pixels_recoverable": nr,
                "map_pixels_irreversible": ni,
            }
        )

    check = pd.DataFrame(rows)

    for name in (
        "total_gws_change_m3",
        "recoverable_gws_change_m3",
        "irreversible_gws_change_m3",
    ):
        check[
            f"{name}_difference"
        ] = (
            check[
                f"{name}_maps"
            ]
            - check[
                f"{name}_previous"
            ]
        )

    output = (
        storage_dir
        / "annual_maps_volume_check.csv"
    )
    check.to_csv(output, index=False)

    diff_cols = [
        c
        for c in check.columns
        if c.endswith("_difference")
    ]
    max_abs_difference = float(
        np.nanmax(
            np.abs(
                check[
                    diff_cols
                ].to_numpy(float)
            )
        )
    )

    return {
        "status": "ok",
        "implementation":
            "map_reintegration_verification_no_refit",
        "output_directory": str(annual_dir),
        "summary": str(summary_csv),
        "volume_check": str(output),
        "years": [
            int(v)
            for v in table["year"]
        ],
        "max_abs_volume_difference_m3":
            max_abs_difference,
    }
