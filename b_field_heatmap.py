#!/usr/bin/env python3
"""
Load measurement CSV (6 columns, no header), compute B = sqrt(bx^2 + by^2),
and plot a heatmap over (x, y).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize


def load_data(path: Path) -> tuple[np.ndarray, ...]:
    data = pd.read_csv(path, header=None, na_values=["nan"])
    if data.shape[1] < 6:
        raise ValueError(f"Expected at least 6 columns, got {data.shape[1]}")
    data = data.dropna()
    x = data.iloc[:, 0].to_numpy(dtype=float)
    y = data.iloc[:, 1].to_numpy(dtype=float)
    bx = data.iloc[:, 2].to_numpy(dtype=float)
    by = data.iloc[:, 3].to_numpy(dtype=float)
    return x, y, bx, by


def compute_b_magnitude(bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    return np.sqrt(bx * bx + by * by)


def _is_regular_grid(x: np.ndarray, y: np.ndarray, rtol: float = 1e-5) -> bool:
    """True if (x,y) form a full Cartesian product of unique sorted axes."""
    ux = np.unique(x)
    uy = np.unique(y)
    n = len(x)
    if ux.size * uy.size != n:
        return False
    # Every x value should appear the same number of times (len(uy))
    counts = np.bincount(
        np.searchsorted(ux, x),
        minlength=ux.size,
    )
    if not np.all(counts == len(uy)):
        return False
    return True


def plot_heatmap(
    x: np.ndarray,
    y: np.ndarray,
    b: np.ndarray,
    title: str = r"$B = \sqrt{b_x^2 + b_y^2}$",
    cmap: str = "viridis",
    outfile: Path | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    if _is_regular_grid(x, y):
        ux = np.sort(np.unique(x))
        uy = np.sort(np.unique(y))
        grid = b.reshape(uy.size, ux.size)
        extent = (ux.min(), ux.max(), uy.min(), uy.max())
        im = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            norm=Normalize(vmin=np.nanmin(b), vmax=np.nanmax(b)),
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="B")
    else:
        # Unstructured or partial grid: triangulated heatmap
        tri = ax.tripcolor(x, y, b, shading="gouraud", cmap=cmap)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title + " (triangulated)")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(tri, ax=ax, label="B")

    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, dpi=150, bbox_inches="tight")
    backend = matplotlib.get_backend().lower()
    if backend in ("agg", "svg", "pdf", "ps", "template"):
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="B-field magnitude heatmap from CSV.")
    parser.add_argument(
        "csv",
        type=Path,
        nargs="?",
        default=Path("rawdata.csv"),
        help="Input CSV path (default: rawdata.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Save figure to this path (e.g. heatmap.png)",
    )
    parser.add_argument(
        "--cmap",
        default="viridis",
        help="Matplotlib colormap name (default: viridis)",
    )
    args = parser.parse_args()

    x, y, bx, by = load_data(args.csv)
    b = compute_b_magnitude(bx, by)
    plot_heatmap(x, y, b, cmap=args.cmap, outfile=args.output)


if __name__ == "__main__":
    main()
