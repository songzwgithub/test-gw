from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from ..common import days_to_dates, ensure_dir, read_tif, write_json
from ..config import ProjectConfig


def _figure_dir(cfg: ProjectConfig) -> Path:
    sec = cfg.section("visualization")
    value = sec.get("output_directory")
    return ensure_dir(cfg.resolve(value) if value else cfg.outputs / "figures" / "checks")


def _dpi(cfg: ProjectConfig) -> int:
    return int(cfg.section("visualization").get("dpi", 180))


def _decimate(arr: np.ndarray, cfg: ProjectConfig) -> np.ndarray:
    arr = np.asarray(arr)
    nmax = int(cfg.section("visualization").get("max_display_pixels", 1400))
    step = max(1, int(np.ceil(max(arr.shape[-2:]) / max(1, nmax))))
    return arr[..., ::step, ::step]


def _robust_limits(arr: np.ndarray, symmetric: bool = False):
    x = np.asarray(arr, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None, None
    lo, hi = np.nanpercentile(x, [2, 98])
    if symmetric:
        vmax = max(abs(lo), abs(hi), 1e-12)
        return -vmax, vmax
    if lo == hi:
        pad = max(abs(lo) * 0.05, 1e-6)
        return lo - pad, hi + pad
    return lo, hi


def _imshow(ax, arr, title: str, cmap: str = "viridis", symmetric: bool = False, categorical: bool = False):
    arr = np.asarray(arr, dtype=float)
    if categorical:
        im = ax.imshow(arr, cmap="tab20", interpolation="nearest")
    else:
        vmin, vmax = _robust_limits(arr, symmetric=symmetric)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)


def _save(fig, path: Path, cfg: ProjectConfig):
    fig.tight_layout()
    fig.savefig(path, dpi=_dpi(cfg), bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_prepare_insar(cfg: ProjectConfig) -> list[str]:
    path = cfg.outputs / "canonical" / "insar_stack.h5"
    if not path.exists(): return []
    with h5py.File(path, "r") as h5:
        n = h5["displacement_mm"].shape[0]
        idx = sorted(set([0, n // 2, n - 1]))
        dates = days_to_dates(h5["date_days"][:])
        arrs = [_decimate(h5["displacement_mm"][i].astype(float), cfg) for i in idx]
    lim = _robust_limits(np.concatenate([a[np.isfinite(a)] for a in arrs if np.isfinite(a).any()]), symmetric=True)
    fig, axes = plt.subplots(1, len(idx), figsize=(4.3 * len(idx), 4))
    axes = np.atleast_1d(axes)
    for ax, i, arr in zip(axes, idx, arrs):
        im = ax.imshow(arr, cmap="RdBu", vmin=lim[0], vmax=lim[1], interpolation="nearest")
        ax.set_title(f"InSAR {dates[i]}\nmm (+ uplift)", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    return [_save(fig, _figure_dir(cfg) / "01_prepare_insar.png", cfg)]


def _plot_prepare_groundwater(cfg: ProjectConfig) -> list[str]:
    path = cfg.outputs / "canonical" / "groundwater.csv"
    if not path.exists(): return []
    df = pd.read_csv(path, parse_dates=["date"])
    valid = df[np.isfinite(df["head_m"])].copy()

    wells = (
        df.groupby("station_id", as_index=False)
        .agg(
            lon=("lon", "first"),
            lat=("lat", "first"),
            n=("head_m", "count"),
        )
    )

    monthly = (
        valid.assign(month=valid["date"].dt.to_period("M").dt.to_timestamp())
        .groupby("month")["station_id"]
        .nunique()
    )
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    sc=axes[0].scatter(wells["lon"], wells["lat"], c=wells["n"], s=28)
    axes[0].set_title(f"Groundwater wells (n={len(wells)})"); axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
    plt.colorbar(sc, ax=axes[0], label="Observations")
    axes[1].plot(monthly.index, monthly.values)
    axes[1].set_title("Active wells by month"); axes[1].set_ylabel("Well count"); axes[1].tick_params(axis="x", rotation=30)
    return [_save(fig, _figure_dir(cfg) / "02_prepare_groundwater.png", cfg)]


def _plot_groundwater_field(cfg: ProjectConfig) -> list[str]:
    path = cfg.outputs / "groundwater" / "groundwater_field.h5"
    if not path.exists(): return []
    with h5py.File(path,"r") as h5:
        n=h5["head_anomaly_m"].shape[0]; idx=sorted(set([0,n//2,n-1])); dates=days_to_dates(h5["date_days"][:])
        arrs=[_decimate(h5["head_anomaly_m"][i].astype(float),cfg) for i in idx]
    fig,axes=plt.subplots(1,len(idx),figsize=(4.3*len(idx),4)); axes=np.atleast_1d(axes)
    for ax,i,a in zip(axes,idx,arrs): _imshow(ax,a,f"Head anomaly {dates[i]}\nm","viridis",False)
    outputs=[_save(fig,_figure_dir(cfg)/"03a_groundwater_field.png",cfg)]
    cvp=cfg.outputs/"groundwater"/"groundwater_model_cv.csv"
    if cvp.exists():
        t=pd.read_csv(cvp); fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(len(t)); ax.plot(x,t["cv_rmse_m"],marker="o",label="Full RMSE")
        if "cv_harmonic_vector_rmse_m" in t: ax.plot(x,t["cv_harmonic_vector_rmse_m"],marker="o",label="Harmonic vector RMSE")
        ax.set_xlabel("Candidate model"); ax.set_ylabel("Error (m)"); ax.set_title("Groundwater spatial CV"); ax.legend()
        outputs.append(_save(fig,_figure_dir(cfg)/"03b_groundwater_cv.png",cfg))
    return outputs


def _plot_deformation(cfg: ProjectConfig) -> list[str]:
    base = cfg.outputs / "deformation"

    main_paths = [
        base / "end_rate_mm_yr.tif",
        base / "annual_amplitude_mm.tif",
        base / "fit_rmse_mm.tif",
    ]
    if not all(p.exists() for p in main_paths):
        return []

    arr = [_decimate(read_tif(p), cfg) for p in main_paths]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    _imshow(axes[0], arr[0], "End rate (mm/yr)", "RdBu", True)
    _imshow(axes[1], arr[1], "Annual amplitude (mm)")
    _imshow(axes[2], arr[2], "Fit RMSE (mm)")
    outputs = [
        _save(
            fig,
            _figure_dir(cfg) / "04_deformation_decomposition.png",
            cfg,
        )
    ]

    delta_path = base / "delta_bic_linear_minus_quadratic.tif"
    pref_path = base / "preferred_model.tif"
    evidence_path = base / "bic_evidence_class.tif"

    if delta_path.exists() and pref_path.exists() and evidence_path.exists():
        delta = _decimate(read_tif(delta_path), cfg)
        pref = _decimate(read_tif(pref_path), cfg)
        evidence = _decimate(read_tif(evidence_path), cfg)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        _imshow(
            axes[0],
            delta,
            "ΔBIC = BIC linear - BIC quadratic",
            "RdBu",
            True,
        )
        _imshow(
            axes[1],
            pref,
            "Preferred model\n1=linear, 2=quadratic",
            categorical=True,
        )
        _imshow(
            axes[2],
            evidence,
            "BIC evidence\n1=linear, 2=weak, 3=quadratic",
            categorical=True,
        )
        outputs.append(
            _save(
                fig,
                _figure_dir(cfg) / "04b_deformation_model_selection.png",
                cfg,
            )
        )

    return outputs

def _plot_regimes(cfg: ProjectConfig) -> list[str]:
    mp=cfg.outputs/"regimes"/"deformation_regime_id.tif"
    if not mp.exists(): return []
    m=_decimate(read_tif(mp),cfg); kp=cfg.outputs/"regimes"/"k_selection.csv"
    fig,axes=plt.subplots(1,2,figsize=(10,4)); _imshow(axes[0],m,"Deformation regimes",categorical=True)
    if kp.exists():
        t=pd.read_csv(kp); axes[1].plot(t["k"],t["silhouette"],marker="o",label="Silhouette")
        axes[1].set_xlabel("K"); axes[1].set_title("K selection"); axes[1].legend()
    else: axes[1].axis("off")
    return [_save(fig,_figure_dir(cfg)/"05_deformation_regimes.png",cfg)]


def _plot_joint_harmonics(cfg: ProjectConfig) -> list[str]:
    b=cfg.outputs/"seasonal"; ps=[b/"deformation_annual_amplitude_mm.tif",b/"head_annual_amplitude_m.tif",b/"deformation_fit_rmse_mm.tif",b/"head_fit_rmse_m.tif"]
    if not all(p.exists() for p in ps): return []
    a=[_decimate(read_tif(p),cfg) for p in ps]; fig,axes=plt.subplots(2,2,figsize=(9,7))
    titles=["Deformation annual amplitude (mm)","Head annual amplitude (m)","Deformation fit RMSE (mm)","Head fit RMSE (m)"]
    for ax,x,t in zip(axes.ravel(),a,titles): _imshow(ax,x,t)
    return [_save(fig,_figure_dir(cfg)/"06_joint_harmonics.png",cfg)]


def _plot_lag(cfg: ProjectConfig) -> list[str]:
    b=cfg.outputs/"seasonal"; mp=b/"phase_lag_days.tif"; cp=b/"lag_scan.csv"
    if not mp.exists(): return []
    fig,axes=plt.subplots(1,2,figsize=(10,4)); _imshow(axes[0],_decimate(read_tif(mp),cfg),"Pixel phase lag (days)")
    if cp.exists():
        t=pd.read_csv(cp); axes[1].plot(t.iloc[:,0],t.iloc[:,1]); j=int(np.nanargmax(t.iloc[:,1].to_numpy(float))); axes[1].axvline(t.iloc[j,0],ls="--")
        axes[1].set_xlabel("Lag (days)"); axes[1].set_ylabel("Weighted cosine"); axes[1].set_title("Regional lag scan")
    else: axes[1].axis("off")
    return [_save(fig,_figure_dir(cfg)/"07_lag.png",cfg)]


def _plot_ske(cfg: ProjectConfig) -> list[str]:
    b = cfg.outputs / "seasonal"
    primary = b / "ske_pixelwise.tif"
    cosine = b / "seasonal_vector_cosine.tif"
    support = b / "ske_pixelwise_support_mask.tif"
    high_support = b / "ske_pixelwise_high_confidence_mask.tif"
    if not primary.exists():
        return []

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    _imshow(
        axes[0, 0],
        _decimate(read_tif(primary), cfg),
        "Pixelwise Ske",
    )

    if cosine.exists():
        _imshow(
            axes[0, 1],
            _decimate(read_tif(cosine), cfg),
            "Seasonal vector cosine",
            "RdBu",
            False,
        )
    else:
        axes[0, 1].axis("off")

    if support.exists():
        _imshow(
            axes[1, 0],
            _decimate(read_tif(support), cfg),
            "Pixelwise Ske support",
            categorical=True,
        )
    else:
        axes[1, 0].axis("off")

    if high_support.exists():
        _imshow(
            axes[1, 1],
            _decimate(read_tif(high_support), cfg),
            "High-confidence Ske support",
            categorical=True,
        )
    else:
        axes[1, 1].axis("off")

    return [
        _save(
            fig,
            _figure_dir(cfg) / "08_ske_pixelwise.png",
            cfg,
        )
    ]

def _plot_storage(cfg: ProjectConfig) -> list[str]:
    b=cfg.outputs/"storage"; ps=[b/"irreversible_gws_change_equivalent_mm.tif",b/"recoverable_gws_change_equivalent_mm.tif",b/"head_lowfreq_change_m.tif"]
    if not ps[0].exists(): return []
    fig,axes=plt.subplots(1,3,figsize=(12,4)); _imshow(axes[0],_decimate(read_tif(ps[0]),cfg),"Irreversible change (mm)","RdBu",True); _imshow(axes[1],_decimate(read_tif(ps[1]),cfg),"Recoverable change (mm)","RdBu",True); _imshow(axes[2],_decimate(read_tif(ps[2]),cfg),"Low-frequency head change (m)","RdBu",True)
    outputs=[_save(fig,_figure_dir(cfg)/"09a_storage_maps.png",cfg)]
    cp=b/"storage_annual_change.csv"
    if cp.exists():
        t=pd.read_csv(cp); fig,ax=plt.subplots(figsize=(8,4))
        for col,label in [("total_gws_change_m3","TGWS"),("recoverable_gws_change_m3","RGWS"),("irreversible_gws_change_m3","IGWS")]: ax.plot(t["year"],t[col]/1e8,marker="o",label=label)
        ax.axhline(0,lw=0.8); ax.set_xlabel("Year"); ax.set_ylabel("Change ($10^8$ m³)"); ax.set_title("Annual storage partition"); ax.legend()
        outputs.append(_save(fig,_figure_dir(cfg)/"09b_storage_annual.png",cfg))
    return outputs


def _plot_hydrostratigraphy(cfg: ProjectConfig) -> list[str]:
    b=cfg.outputs/"hydrostratigraphy"; clay=b/"total_clay_thickness_m.tif"; sand=b/"total_sand_thickness_m.tif"
    if not clay.exists() or not sand.exists(): return []
    fig,axes=plt.subplots(1,2,figsize=(9,4)); _imshow(axes[0],_decimate(read_tif(clay),cfg),"Total clay thickness (m)"); _imshow(axes[1],_decimate(read_tif(sand),cfg),"Total sand thickness (m)")
    return [_save(fig,_figure_dir(cfg)/"10_hydrostratigraphy.png",cfg)]


def _plot_extensometer(cfg: ProjectConfig) -> list[str]:
    b=cfg.outputs/"extensometer"; cp=b/"depth_interval_contributions.csv"
    if not cp.exists(): return []
    t=pd.read_csv(cp); fig,axes=plt.subplots(1,2,figsize=(10,4)); labels=[f"{a:g}-{z:g}" for a,z in zip(t["depth_top_m"],t["depth_bottom_m"])]
    axes[0].bar(labels,t["change_mm"]); axes[0].set_title("Depth-interval compaction"); axes[0].set_ylabel("Change (mm)"); axes[0].tick_params(axis="x",rotation=45)
    ts=b/"collocated_insar_groundwater.csv"
    if ts.exists():
        d=pd.read_csv(ts,parse_dates=["date"]); ax2=axes[1].twinx(); axes[1].plot(d["date"],d["insar_displacement_mm"],label="InSAR"); ax2.plot(d["date"],d["groundwater_head_anomaly_m"],ls="--",label="Head")
        axes[1].set_title("Collocated time series"); axes[1].set_ylabel("InSAR (mm)"); ax2.set_ylabel("Head anomaly (m)")
    else: axes[1].axis("off")
    return [_save(fig,_figure_dir(cfg)/"11_extensometer.png",cfg)]


def _plot_synthesis(cfg: ProjectConfig) -> list[str]:
    p=cfg.outputs/"synthesis"/"aquifer_system_response_by_regime.csv"
    if not p.exists(): return []
    t=pd.read_csv(p); x=np.arange(len(t)); fig,axes=plt.subplots(1,3,figsize=(12,4)); labels=t["regime"].astype(str)
    for ax,col,title in zip(axes,["end_rate_median_mm_yr","ske_median","irreversible_gws_change_median_mm"],["End rate (mm/yr)","Median Ske","Irreversible change (mm)"]):
        ax.bar(x,t[col]); ax.set_xticks(x); ax.set_xticklabels(labels,rotation=35,ha="right"); ax.set_title(title)
    return [_save(fig,_figure_dir(cfg)/"12_synthesis.png",cfg)]


PLOT_STAGES: dict[str, Callable[[ProjectConfig], list[str]]] = {
    "prepare-insar": _plot_prepare_insar,
    "prepare-groundwater": _plot_prepare_groundwater,
    "build-groundwater-field": _plot_groundwater_field,
    "decompose-insar": _plot_deformation,
    "classify-deformation": _plot_regimes,
    "joint-harmonics": _plot_joint_harmonics,
    "estimate-lag": _plot_lag,
    "estimate-ske": _plot_ske,
    "storage-budget": _plot_storage,
    "hydrostratigraphy": _plot_hydrostratigraphy,
    "extensometer": _plot_extensometer,
    "synthesize": _plot_synthesis,
}


def plot_stage(cfg: ProjectConfig, stage: str) -> dict[str, Any]:
    if stage not in PLOT_STAGES:
        raise KeyError(f"Unknown plot stage {stage!r}")
    files = PLOT_STAGES[stage](cfg)
    return {"status": "ok" if files else "skipped", "stage": stage, "files": files}


def plot_all(cfg: ProjectConfig) -> dict[str, Any]:
    results = {stage: plot_stage(cfg, stage) for stage in PLOT_STAGES}
    payload = {"status": "ok", "output_directory": str(_figure_dir(cfg)), "stages": results}
    write_json(_figure_dir(cfg) / "plot_summary.json", payload)
    return payload
