from __future__ import annotations

import matplotlib.pyplot as plt


def scatter_xz(shower, outpath):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(shower.x, shower.z, s=4)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
