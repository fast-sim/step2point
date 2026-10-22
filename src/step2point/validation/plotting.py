from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PRESENTATION_TITLE_FONTSIZE = 28
PRESENTATION_AXIS_LABEL_FONTSIZE = 24
PRESENTATION_TICK_FONTSIZE = 19
PRESENTATION_LEGEND_FONTSIZE = 18
PRESENTATION_RATIO_LABEL_FONTSIZE = 24
PRESENTATION_HIST_LINEWIDTH = 3.2
PRESENTATION_LINEWIDTH = 3.0
PRESENTATION_GUIDE_LINEWIDTH = 1.8
PRESENTATION_MARKER_SIZE = 64


def _prepare_outpath(outpath: str | Path) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    return outpath


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    ratio = np.full(numerator.shape, np.nan, dtype=np.float64)
    mask = denominator > 0.0
    ratio[mask] = numerator[mask] / denominator[mask]
    return ratio


def _safe_ratio_sigma(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    sigma = np.full(numerator.shape, np.nan, dtype=np.float64)
    mask = (denominator > 0.0) & (numerator > 0.0)
    ratio = numerator[mask] / denominator[mask]
    sigma[mask] = ratio * np.sqrt(1.0 / numerator[mask] + 1.0 / denominator[mask])
    return sigma


def _infer_bin_edges(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 1:
        return np.array([x[0] - 0.5, x[0] + 0.5], dtype=np.float64)
    edges = np.empty(x.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    return edges


def _ratio_ylim(*ratio_series: np.ndarray) -> tuple[float, float]:
    finite = []
    for ratio in ratio_series:
        arr = np.asarray(ratio, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite.append(arr)
    if not finite:
        return 0.8, 1.2
    values = np.concatenate(finite)
    lower = min(float(np.min(values)), 1.0)
    upper = max(float(np.max(values)), 1.0)
    span = max(upper - lower, 0.02)
    pad = max(0.08 * span, 0.02)
    return lower - pad, upper + pad


def _overlay_axes(with_ratio: bool):
    if not with_ratio:
        fig, ax = plt.subplots(figsize=(10, 7))
        return fig, ax, None
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(11.4, 8.8),
        sharex=True,
        gridspec_kw={"height_ratios": [4.7, 0.9], "hspace": 0.02},
    )
    return fig, ax, ax_ratio


def plot_hist(
    values,
    outpath: str | Path,
    title: str,
    xlabel: str,
    logy: bool = False,
    bins=30,
    *,
    xlim: tuple[float, float] | None = None,
):
    outpath = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.hist(np.asarray(values), bins=bins, color="#1f6b3a", alpha=0.9)
    ax.set_title(title, fontsize=PRESENTATION_TITLE_FONTSIZE, pad=14)
    ax.set_xlabel(xlabel, fontsize=PRESENTATION_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PRESENTATION_TICK_FONTSIZE)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if logy:
        ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_hist_series(
    series,
    outpath: str | Path,
    title: str,
    xlabel: str,
    logy: bool = False,
    bins=30,
    *,
    xlim: tuple[float, float] | None = None,
):
    outpath = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#0057D9", "#C2185B", "#1B9E77", "#D95F02", "#6A3D9A", "#B8860B"]
    for idx, (label, values) in enumerate(series):
        values = np.asarray(values)
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=PRESENTATION_HIST_LINEWIDTH,
            color=colors[idx % len(colors)],
            label=label,
        )
    ax.set_title(title, fontsize=PRESENTATION_TITLE_FONTSIZE, pad=14)
    ax.set_xlabel(xlabel, fontsize=PRESENTATION_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PRESENTATION_TICK_FONTSIZE)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=PRESENTATION_LEGEND_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_overlay_hist(
    pre_values,
    post_values,
    outpath: str | Path,
    title: str,
    xlabel: str,
    logy: bool = False,
    bins=40,
    *,
    pre_label: str = "pre",
    post_label: str = "post",
    with_ratio: bool = False,
    ratio_ylim: tuple[float, float] | None = None,
    ratio_band: bool = True,
):
    outpath = _prepare_outpath(outpath)
    fig, ax, ax_ratio = _overlay_axes(with_ratio)
    pre_values = np.asarray(pre_values)
    post_values = np.asarray(post_values)
    counts_pre, edges = np.histogram(pre_values, bins=bins)
    counts_post, _ = np.histogram(post_values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ratio = _safe_ratio(counts_post, counts_pre)
    ratio_sigma = _safe_ratio_sigma(counts_post, counts_pre)

    ax.hist(pre_values, bins=edges, color="#FF7A00", alpha=0.35, label=pre_label, zorder=1)
    ax.hist(
        post_values,
        bins=edges,
        histtype="step",
        color="#0057D9",
        linewidth=PRESENTATION_HIST_LINEWIDTH,
        label=post_label,
        zorder=3,
    )
    ax.set_title(title, fontsize=PRESENTATION_TITLE_FONTSIZE, pad=14)
    ax.tick_params(axis="both", labelsize=PRESENTATION_TICK_FONTSIZE)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=PRESENTATION_LEGEND_FONTSIZE)
    if ax_ratio is not None:
        ax_ratio.step(
            centers,
            ratio,
            where="mid",
            color="#0057D9",
            linewidth=PRESENTATION_GUIDE_LINEWIDTH,
        )
        lower = ratio - ratio_sigma
        upper = ratio + ratio_sigma
        if ratio_band:
            ax_ratio.fill_between(centers, lower, upper, step="mid", color="#0057D9", alpha=0.18)
        ax_ratio.axhline(1.0, color="#666666", linewidth=1.2, linestyle="--")
        ax_ratio.set_ylabel("ratio", fontsize=PRESENTATION_RATIO_LABEL_FONTSIZE)
        ax_ratio.set_xlabel(xlabel, fontsize=PRESENTATION_AXIS_LABEL_FONTSIZE)
        ax_ratio.tick_params(axis="both", labelsize=PRESENTATION_TICK_FONTSIZE)
        if ratio_ylim is not None:
            ax_ratio.set_ylim(*ratio_ylim)
        else:
            ax_ratio.set_ylim(*_ratio_ylim(ratio, lower, upper) if ratio_band else _ratio_ylim(ratio))
        ax_ratio.grid(True, axis="y", alpha=0.25)
    else:
        ax.set_xlabel(xlabel, fontsize=PRESENTATION_AXIS_LABEL_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_overlay_hist_multi(
    pre_values,
    post_series,
    outpath: str | Path,
    title: str,
    xlabel: str,
    logy: bool = False,
    bins=40,
    *,
    pre_label: str = "pre",
    with_ratio: bool = False,
    ratio_ylim: tuple[float, float] | None = None,
    ratio_band: bool = True,
):
    outpath = _prepare_outpath(outpath)
    fig, ax, ax_ratio = _overlay_axes(with_ratio)
    pre_values = np.asarray(pre_values)
    counts_pre, edges = np.histogram(pre_values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ratio_series = []
    ax.hist(pre_values, bins=edges, color="#FF7A00", alpha=0.35, label=pre_label, zorder=1)
    colors = ["#0057D9", "#C2185B", "#1B9E77", "#D95F02", "#6A3D9A", "#B8860B"]
    for idx, (post_label, post_values) in enumerate(post_series):
        post_values = np.asarray(post_values)
        ax.hist(
            post_values,
            bins=edges,
            histtype="step",
            color=colors[idx % len(colors)],
            linewidth=PRESENTATION_HIST_LINEWIDTH,
            label=post_label,
            zorder=3,
        )
        if ax_ratio is not None:
            counts_post, _ = np.histogram(post_values, bins=edges)
            ratio = _safe_ratio(counts_post, counts_pre)
            ratio_sigma = _safe_ratio_sigma(counts_post, counts_pre)
            ratio_series.append(ratio)
            if ratio_band:
                ratio_series.extend([ratio - ratio_sigma, ratio + ratio_sigma])
            ax_ratio.step(
                centers,
                ratio,
                where="mid",
                color=colors[idx % len(colors)],
                linewidth=PRESENTATION_GUIDE_LINEWIDTH,
            )
            if ratio_band:
                ax_ratio.fill_between(
                    centers,
                    ratio - ratio_sigma,
                    ratio + ratio_sigma,
                    step="mid",
                    color=colors[idx % len(colors)],
                    alpha=0.12,
                )
    ax.set_title(title, fontsize=PRESENTATION_TITLE_FONTSIZE, pad=14)
    ax.tick_params(axis="both", labelsize=PRESENTATION_TICK_FONTSIZE)
    if logy:
        ax.set_yscale("log")
    ax.legend(fontsize=PRESENTATION_LEGEND_FONTSIZE)
    if ax_ratio is not None:
        ax_ratio.axhline(1.0, color="#666666", linewidth=1.2, linestyle="--")
        ax_ratio.set_ylabel("ratio", fontsize=PRESENTATION_RATIO_LABEL_FONTSIZE)
        ax_ratio.set_xlabel(xlabel, fontsize=PRESENTATION_AXIS_LABEL_FONTSIZE)
        ax_ratio.tick_params(axis="both", labelsize=PRESENTATION_TICK_FONTSIZE)
        if ratio_ylim is not None:
            ax_ratio.set_ylim(*ratio_ylim)
        else:
            ax_ratio.set_ylim(*_ratio_ylim(*ratio_series))
        ax_ratio.grid(True, axis="y", alpha=0.25)
    else:
        ax.set_xlabel(xlabel, fontsize=PRESENTATION_AXIS_LABEL_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_overlay_line(
    x,
    y_pre,
    y_post,
    outpath: str | Path,
    title: str,
    xlabel: str,
    ylabel: str = "Mean energy",
    *,
    pre_label: str = "pre",
    post_label: str = "post",
    xlim: tuple[float, float] | None = None,
    with_ratio: bool = False,
    ratio_ylim: tuple[float, float] | None = None,
):
    outpath = _prepare_outpath(outpath)
    fig, ax, ax_ratio = _overlay_axes(with_ratio)
    x = np.asarray(x)
    y_pre = np.asarray(y_pre)
    y_post = np.asarray(y_post)
    edges = _infer_bin_edges(x)
    ratio = _safe_ratio(y_post, y_pre)

    # Use markers to emphasize the histogram-bin nature of these profiles.
    pre_color = "#FF7A00"
    post_color = "#0057D9"
    ax.plot(x, y_pre, color=pre_color, linewidth=PRESENTATION_LINEWIDTH, alpha=0.22, linestyle="--", zorder=1)
    ax.scatter(
        x,
        y_pre,
        color=pre_color,
        s=PRESENTATION_MARKER_SIZE,
        marker="s",
        linewidths=0.0,
        label=pre_label,
        zorder=2,
    )

    ax.plot(x, y_post, color=post_color, linewidth=PRESENTATION_LINEWIDTH, alpha=0.22, linestyle="--", zorder=3)
    ax.scatter(
        x,
        y_post,
        color=post_color,
        s=PRESENTATION_MARKER_SIZE,
        marker="o",
        linewidths=0.0,
        label=post_label,
        zorder=4,
    )

    ax.set_title(title, fontsize=PRESENTATION_TITLE_FONTSIZE, pad=14)
    ax.set_ylabel(ylabel, fontsize=PRESENTATION_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PRESENTATION_TICK_FONTSIZE)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(fontsize=PRESENTATION_LEGEND_FONTSIZE)
    if ax_ratio is not None:
        ax_ratio.stairs(ratio, edges, color=post_color, linewidth=PRESENTATION_GUIDE_LINEWIDTH)
        ax_ratio.axhline(1.0, color="#666666", linewidth=1.2, linestyle="--")
        ax_ratio.set_ylabel("ratio", fontsize=PRESENTATION_RATIO_LABEL_FONTSIZE)
        ax_ratio.set_xlabel(xlabel, fontsize=PRESENTATION_AXIS_LABEL_FONTSIZE)
        ax_ratio.tick_params(axis="both", labelsize=PRESENTATION_TICK_FONTSIZE)
        if xlim is not None:
            ax_ratio.set_xlim(*xlim)
        if ratio_ylim is not None:
            ax_ratio.set_ylim(*ratio_ylim)
        else:
            ax_ratio.set_ylim(*_ratio_ylim(ratio))
        ax_ratio.grid(True, axis="y", alpha=0.25)
    else:
        ax.set_xlabel(xlabel, fontsize=PRESENTATION_AXIS_LABEL_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_overlay_line_multi(
    x,
    y_pre,
    post_series,
    outpath: str | Path,
    title: str,
    xlabel: str,
    ylabel: str = "Mean energy",
    *,
    pre_label: str = "pre",
    xlim: tuple[float, float] | None = None,
    with_ratio: bool = False,
    ratio_ylim: tuple[float, float] | None = None,
):
    outpath = _prepare_outpath(outpath)
    fig, ax, ax_ratio = _overlay_axes(with_ratio)
    x = np.asarray(x)
    y_pre = np.asarray(y_pre)
    edges = _infer_bin_edges(x)

    pre_color = "#FF7A00"
    ax.plot(x, y_pre, color=pre_color, linewidth=PRESENTATION_LINEWIDTH, alpha=0.22, linestyle="--", zorder=1)
    ax.scatter(
        x,
        y_pre,
        color=pre_color,
        s=PRESENTATION_MARKER_SIZE,
        marker="s",
        linewidths=0.0,
        label=pre_label,
        zorder=2,
    )

    colors = ["#0057D9", "#C2185B", "#1B9E77", "#D95F02", "#6A3D9A", "#B8860B"]
    ratio_series = []
    for idx, (post_label, y_post) in enumerate(post_series, start=1):
        color = colors[(idx - 1) % len(colors)]
        y_post = np.asarray(y_post)
        ax.plot(x, y_post, color=color, linewidth=PRESENTATION_LINEWIDTH, alpha=0.22, linestyle="--", zorder=2 * idx + 1)
        ax.scatter(
            x,
            y_post,
            color=color,
            s=PRESENTATION_MARKER_SIZE,
            marker="o",
            linewidths=0.0,
            label=post_label,
            zorder=2 * idx + 2,
        )
        if ax_ratio is not None:
            ratio = _safe_ratio(y_post, y_pre)
            ratio_series.append(ratio)
            ax_ratio.stairs(ratio, edges, color=color, linewidth=PRESENTATION_GUIDE_LINEWIDTH)

    ax.set_title(title, fontsize=PRESENTATION_TITLE_FONTSIZE, pad=14)
    ax.set_ylabel(ylabel, fontsize=PRESENTATION_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=PRESENTATION_TICK_FONTSIZE)
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.legend(fontsize=PRESENTATION_LEGEND_FONTSIZE)
    if ax_ratio is not None:
        ax_ratio.axhline(1.0, color="#666666", linewidth=1.2, linestyle="--")
        ax_ratio.set_ylabel("ratio", fontsize=PRESENTATION_RATIO_LABEL_FONTSIZE)
        ax_ratio.set_xlabel(xlabel, fontsize=PRESENTATION_AXIS_LABEL_FONTSIZE)
        ax_ratio.tick_params(axis="both", labelsize=PRESENTATION_TICK_FONTSIZE)
        if xlim is not None:
            ax_ratio.set_xlim(*xlim)
        if ratio_ylim is not None:
            ax_ratio.set_ylim(*ratio_ylim)
        else:
            ax_ratio.set_ylim(*_ratio_ylim(*ratio_series))
        ax_ratio.grid(True, axis="y", alpha=0.25)
    else:
        ax.set_xlabel(xlabel, fontsize=PRESENTATION_AXIS_LABEL_FONTSIZE)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
