"""
Matplotlib figures: time-varying MV weights and kernel diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _yyyymm_to_datetimes(dates_yyyymm: np.ndarray) -> np.ndarray:
    """YYYYMM integers → numpy datetime64 (first day of month), aligned with series rows."""
    s = np.asarray(dates_yyyymm, dtype=np.int64).ravel()
    y = (s // 100).astype(np.int32)
    m = np.clip(s % 100, 1, 12).astype(np.int32)
    ymd = y.astype(np.int64) * 10000 + m.astype(np.int64) * 100 + 1
    return pd.to_datetime(ymd.astype(str), format="%Y%m%d").to_numpy()


def _style_time_axis_years(ax) -> None:
    """Readable calendar axis (years / concise dates)."""
    locator = mdates.AutoDateLocator(minticks=5, maxticks=12)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_mv_weights_over_time(
    dates_yyyymm: np.ndarray,
    w_mv: np.ndarray,
    columns: list[str],
    out_path: Path,
    title: str = "Time-varying mean–variance weights (selected portfolios)",
    max_legend_items: int = 15,
) -> None:
    """Stacked area is unreadable with many series; plot top movers + rest pooled."""
    T, p = w_mv.shape
    mean_abs = np.nanmean(np.abs(w_mv), axis=0)
    order = np.argsort(-mean_abs)
    n_show = min(max_legend_items, p)
    show_idx = order[:n_show]
    x = _yyyymm_to_datetimes(dates_yyyymm[:T])

    fig, ax = plt.subplots(figsize=(11, 5))
    for j in show_idx:
        ax.plot(x, w_mv[:, j], lw=1.0, alpha=0.85, label=str(columns[j])[:40])
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Weight")
    _style_time_axis_years(ax)
    fig.autofmt_xdate(bottom=0.15)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=7)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_state_and_premium(
    dates_yyyymm: np.ndarray,
    states: np.ndarray,
    rp: np.ndarray,
    out_path: Path,
    feat1_name: str,
    feat2_name: str,
) -> None:
    T = len(dates_yyyymm)
    x = _yyyymm_to_datetimes(dates_yyyymm[:T])
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(x, states[:, 0], label="LME vw")
    axes[0].set_ylabel("State")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].plot(x, states[:, 1], color="C1", label=feat1_name)
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(x, states[:, 2], color="C2", label=feat2_name)
    axes[2].legend(loc="upper right", fontsize=8)
    axes[3].plot(x, rp, color="C3", label="w′μ (cond.)")
    axes[3].set_ylabel("Premium proxy")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Year")
    fig.suptitle("States vs conditional MV premium proxy")
    _style_time_axis_years(axes[-1])
    fig.autofmt_xdate(bottom=0.12)
    plt.tight_layout(rect=(0, 0.03, 1, 0.97))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_effective_analogues(
    dates_yyyymm: np.ndarray,
    ess: np.ndarray,
    out_path: Path,
    title: str | None = None,
) -> None:
    T = len(ess)
    x = _yyyymm_to_datetimes(dates_yyyymm[:T])
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(x, ess, lw=1)
    ax.set_title(title or "Effective number of kernel analogues (1/Σw²)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Eff. analogues")
    _style_time_axis_years(ax)
    fig.autofmt_xdate(bottom=0.2)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_depth_mass(depth_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(depth_df["depth"].astype(str), depth_df["sum_mean_abs_w"])
    ax.set_xlabel("Tree depth (parsed)")
    ax.set_ylabel("Sum of mean |weight|")
    ax.set_title("Where splits load: weight mass by depth")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
