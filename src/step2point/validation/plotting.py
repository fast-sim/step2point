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
    ax.hist(np.asarray(values), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    if logy:
        ax.set_yscale("log")
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
):
    outpath = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.asarray(pre_values), bins=bins, histtype="step", label="pre")
    ax.hist(np.asarray(post_values), bins=bins, histtype="step", label="post")
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
):
    outpath = _prepare_outpath(outpath)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y_pre, label="pre")
    ax.plot(x, y_post, label="post")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
