#!/usr/bin/env python3
"""
Load measurement CSV (6 columns, no header), compute B = sqrt(bx^2 + by^2),
and plot a heatmap over (x, y). Missing entries are kept and treated as 0.

On a regular grid, consecutive all-zero B columns from the left and rows from
the bottom (NaN padding forming an L) are cropped so the plot origin matches
the inner corner of that padding. Unstructured (triangulated) plots are not cropped.
"""

from __future__ import annotations

import argparse
import warnings
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
    x = data.iloc[:, 0].to_numpy(dtype=float)
    y = data.iloc[:, 1].to_numpy(dtype=float)
    bx = data.iloc[:, 2].to_numpy(dtype=float)
    by = data.iloc[:, 3].to_numpy(dtype=float)
    # Missing values -> 0 for coordinates and field components (heatmap / B)
    x = np.nan_to_num(x, nan=0.0)
    y = np.nan_to_num(y, nan=0.0)
    bx = np.nan_to_num(bx, nan=0.0)
    by = np.nan_to_num(by, nan=0.0)
    return x, y, bx, by


def compute_b_magnitude(bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    return np.sqrt(bx * bx + by * by)


def crop_bottom_left_zero_padding(
    grid: np.ndarray,
    *,
    atol: float = 0.0,
) -> tuple[np.ndarray, int, int]:
    """
    Crop an L-shaped strip of (near-)zeros: all-zero columns from the left and
    all-zero rows from the bottom (row 0 = bottom with origin='lower').

    The inner corner of the padding (first x column and first y row that contain
    any non-zero B) becomes the new bottom-left of the heatmap.

    Returns (cropped_grid, col_start, row_start).
    """
    ny, nx = grid.shape
    if atol <= 0.0:
        nonzero = grid != 0.0
    else:
        nonzero = np.abs(grid) > atol

    col_start = 0
    while col_start < nx and not np.any(nonzero[:, col_start]):
        col_start += 1

    row_start = 0
    while row_start < ny and not np.any(nonzero[row_start, :]):
        row_start += 1

    if col_start >= nx or row_start >= ny:
        warnings.warn(
            "B is all zero (within tolerance); skipping L-shaped zero-padding crop.",
            stacklevel=2,
        )
        return grid, 0, 0

    return grid[row_start:, col_start:], col_start, row_start


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
    *,
    crop_zero_padding: bool = True,
    crop_atol: float = 0.0,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    if _is_regular_grid(x, y):
        ux = np.sort(np.unique(x))
        uy = np.sort(np.unique(y))
        grid = b.reshape(uy.size, ux.size)
        col0, row0 = 0, 0
        if crop_zero_padding:
            grid, col0, row0 = crop_bottom_left_zero_padding(grid, atol=crop_atol)
        ux_plot = ux[col0 : col0 + grid.shape[1]]
        uy_plot = uy[row0 : row0 + grid.shape[0]]
        extent = (ux_plot.min(), ux_plot.max(), uy_plot.min(), uy_plot.max())
        vmin, vmax = float(np.nanmin(grid)), float(np.nanmax(grid))
        if vmin == vmax:
            vmin -= 1e-12
            vmax += 1e-12
        im = ax.imshow(
            grid,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            norm=Normalize(vmin=vmin, vmax=vmax),
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
    parser.add_argument(
        "--no-crop-padding",
        action="store_true",
        help="Do not crop L-shaped (bottom+left) strips where B is all zero",
    )
    parser.add_argument(
        "--crop-atol",
        type=float,
        default=0.0,
        help="Treat |B| <= this value as zero when detecting padding (default: 0)",
    )
    args = parser.parse_args()

    x, y, bx, by = load_data(args.csv)
    b = compute_b_magnitude(bx, by)
    plot_heatmap(
        x,
        y,
        b,
        cmap=args.cmap,
        outfile=args.output,
        crop_zero_padding=not args.no_crop_padding,
        crop_atol=args.crop_atol,
    )


if __name__ == "__main__":
    main()
