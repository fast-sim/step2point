from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _prepare_outpath(outpath: str | Path) -> Path:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    return outpath


def plot_hist(values, outpath: str | Path, title: str, xlabel: str, logy: bool = False, bins=30):
    outpath = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.asarray(values), bins=bins, color="#1f6b3a", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
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
):
    outpath = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#0057D9", "#C2185B", "#1B9E77", "#D95F02", "#6A3D9A", "#B8860B"]
    for idx, (label, values) in enumerate(series):
        values = np.asarray(values)
        if values.size == 0:
            continue
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=2.0,
            color=colors[idx % len(colors)],
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale("log")
    ax.legend()
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
):
    outpath = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.asarray(pre_values), bins=bins, color="#FF7A00", alpha=0.35, label=pre_label)
    ax.hist(np.asarray(post_values), bins=bins, histtype="step", color="#0057D9", linewidth=2.0, label=post_label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale("log")
    ax.legend()
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
):
    outpath = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.asarray(pre_values), bins=bins, color="#FF7A00", alpha=0.35, label=pre_label)
    colors = ["#0057D9", "#C2185B", "#1B9E77", "#D95F02", "#6A3D9A", "#B8860B"]
    for idx, (post_label, post_values) in enumerate(post_series):
        ax.hist(
            np.asarray(post_values),
            bins=bins,
            histtype="step",
            color=colors[idx % len(colors)],
            linewidth=2.0,
            label=post_label,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale("log")
    ax.legend()
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
):
    outpath = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.asarray(x)
    y_pre = np.asarray(y_pre)
    y_post = np.asarray(y_post)

    # Use markers to emphasize the histogram-bin nature of these profiles.
    pre_color = "#FF7A00"
    post_color = "#0057D9"
    ax.plot(x, y_pre, color=pre_color, linewidth=1.3, alpha=0.22, linestyle="--", zorder=1)
    ax.scatter(x, y_pre, color=pre_color, s=26, marker="s", linewidths=0.0, label=pre_label, zorder=2)

    ax.plot(x, y_post, color=post_color, linewidth=2.2, alpha=0.22, linestyle="--", zorder=3)
    ax.scatter(x, y_post, color=post_color, s=32, marker="o", linewidths=0.0, label=post_label, zorder=4)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
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
):
    outpath = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.asarray(x)
    y_pre = np.asarray(y_pre)

    pre_color = "#FF7A00"
    ax.plot(x, y_pre, color=pre_color, linewidth=1.3, alpha=0.22, linestyle="--", zorder=1)
    ax.scatter(x, y_pre, color=pre_color, s=26, marker="s", linewidths=0.0, label=pre_label, zorder=2)

    colors = ["#0057D9", "#C2185B", "#1B9E77", "#D95F02", "#6A3D9A", "#B8860B"]
    for idx, (post_label, y_post) in enumerate(post_series, start=1):
        color = colors[(idx - 1) % len(colors)]
        y_post = np.asarray(y_post)
        ax.plot(x, y_post, color=color, linewidth=2.0, alpha=0.22, linestyle="--", zorder=2 * idx + 1)
        ax.scatter(
            x,
            y_post,
            color=color,
            s=32,
            marker="o",
            linewidths=0.0,
            label=post_label,
            zorder=2 * idx + 2,
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
