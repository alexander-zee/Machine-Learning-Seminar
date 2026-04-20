from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _ensure_path(path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_oos_sharpe_comparison(
    labels: Sequence[str],
    sharpes: Sequence[float],
    out_path: Path | str,
    *,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    out = _ensure_path(out_path)
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    xs = np.arange(len(labels))
    ax.bar(xs, np.asarray(sharpes, dtype=float), color="#4C72B0")
    ax.set_xticks(xs)
    ax.set_xticklabels(list(labels))
    ax.set_ylabel("Sharpe (monthly)")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_kernel_analogue_heatmap(
    H: np.ndarray,
    dates_yyyymm: Sequence[int] | np.ndarray,
    out_path: Path | str,
    *,
    t_start_row: int = 0,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    out = _ensure_path(out_path)
    Hm = np.asarray(H, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    im = ax.imshow(Hm, aspect="auto", interpolation="nearest")
    ax.set_title(title or "Kernel analogue heatmap")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_kernel_weight_slices(
    kernels_gaussian: Iterable,
    kernels_gaussian_time: Iterable,
    dates_yyyymm: Sequence[int] | np.ndarray,
    months_pick: Sequence[int],
    out_path: Path | str,
) -> None:
    import matplotlib.pyplot as plt

    out = _ensure_path(out_path)
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.plot(np.arange(10), np.linspace(0, 1, 10), label="slice")
    ax.set_title("Kernel weight slices (placeholder)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_mv_weights_over_time(
    dates_yyyymm: Sequence[int] | np.ndarray,
    W: np.ndarray,
    cols: Sequence[str],
    out_path: Path | str,
    *,
    title: str | None = None,
    max_legend_items: int = 10,
    index_range: tuple[int, int] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    out = _ensure_path(out_path)
    Wm = np.asarray(W, dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    t0, t1 = (0, Wm.shape[0]) if index_range is None else index_range
    t0 = int(max(0, t0))
    t1 = int(min(Wm.shape[0], t1))
    tt = np.arange(t0, t1)
    show_cols = list(cols)[: int(max_legend_items)]
    for j, name in enumerate(show_cols):
        if j < Wm.shape[1]:
            ax.plot(tt, Wm[t0:t1, j], linewidth=1.0, label=str(name))
    ax.set_title(title or "MV weights over time")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_effective_analogues(
    dates_yyyymm: Sequence[int] | np.ndarray,
    ess: np.ndarray,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    out = _ensure_path(out_path)
    y = np.asarray(ess, dtype=float).reshape(-1)
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.plot(np.arange(len(y)), y, color="#55A868")
    ax.set_title(title or "Effective analogues")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_l1_static_vs_tv(
    dates_yyyymm: Sequence[int] | np.ndarray,
    W_tv: np.ndarray,
    w_static: np.ndarray,
    t_start: int,
    out_path: Path | str,
    *,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    out = _ensure_path(out_path)
    W = np.asarray(W_tv, dtype=float)
    w = np.asarray(w_static, dtype=float).reshape(-1)
    if W.ndim != 2:
        raise ValueError("W_tv must be 2D")
    dist = np.sum(np.abs(W - w.reshape(1, -1)), axis=1)
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.plot(np.arange(len(dist)), dist, color="#C44E52")
    ax.axvline(int(t_start), color="k", linewidth=0.8, alpha=0.35)
    ax.set_title(title or "L1 distance static vs TV")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
