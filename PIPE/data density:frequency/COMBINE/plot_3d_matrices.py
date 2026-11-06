

from pathlib import Path

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
GRID = HERE / "simulation_results" / "combination_grid"
SOURCE = GRID / "combination_metrics.xlsx"
OUTPUT = GRID / "combination_metrics_3d"
NOBS = (1, 2, 3, 4)
INTERVALS = (1, 2, 3, 4)


def metric_table(metric):
    table = pd.read_excel(SOURCE, sheet_name=metric, index_col=0)
    table = table.reindex(index=INTERVALS, columns=NOBS)
    values = table.to_numpy(dtype=float)
    if values.shape != (4, 4) or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite 4x4 {metric} grid.")
    return values


def draw_panel(ax, values, title, zlabel, cmap_name, formatter):
    x, y = np.meshgrid(np.arange(4), np.arange(4))
    x = x.ravel()
    y = y.ravel()
    z = np.zeros_like(x, dtype=float)
    dx = np.full_like(x, 0.58, dtype=float)
    dy = np.full_like(y, 0.58, dtype=float)
    dz = values.ravel()

    source_palette = matplotlib.colormaps[cmap_name]
    
    palette_min, palette_max = ((0.02, 0.78) if cmap_name == "Blues_r"
                                else (0.30, 0.94))
    palette = LinearSegmentedColormap.from_list(
        f"{cmap_name}_cropped",
        source_palette(np.linspace(palette_min, palette_max, 256)),
    )
    value_range = float(dz.max() - dz.min())
    if value_range == 0:
        value_range = max(abs(float(dz[0])) * 0.1, 1e-9)
    norm = Normalize(
        vmin=max(0.0, float(dz.min()) - value_range * 0.08),
        vmax=float(dz.max()) + value_range * 0.08,
    )
    bar_colors = palette(norm(dz))
    ax.bar3d(
        x, y, z, dx, dy, dz,
        color=bar_colors,
        edgecolor="#ffffff",
        linewidth=0.6,
        shade=False,
        zsort="average",
    )

    zmax = float(dz.max())
    ax.set_zlim(0, zmax * 1.08)

    ax.set_xticks(np.arange(4) + dx[0] / 2, [str(v) for v in NOBS])
    ax.set_yticks(np.arange(4) + dy[0] / 2, [str(v) for v in INTERVALS])
    ax.set_xlabel("Nobs", labelpad=5)
    ax.set_ylabel("Interval (steps)", labelpad=7)
    ax.set_title(title, loc="left", pad=0, fontsize=10)
    ax.view_init(elev=25, azim=-55)
    ax.set_box_aspect((1, 1, 0.72))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.zaxis.set_major_formatter(FuncFormatter(lambda value, _: formatter(value)))
    ax.tick_params(axis="both", which="major", labelsize=8, pad=0)
    ax.tick_params(axis="z", which="major", labelsize=7, pad=1)
    ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
    ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.xaxis.pane.set_edgecolor("#dddddd")
    ax.yaxis.pane.set_edgecolor("#dddddd")
    ax.zaxis.pane.set_edgecolor("#dddddd")
    ax.grid(True, linewidth=0.35, alpha=0.38)

    color_source = mpl.cm.ScalarMappable(norm=norm, cmap=palette)
    color_source.set_array([])
    colorbar = ax.figure.colorbar(
        color_source, ax=ax, fraction=0.035, pad=0.055, shrink=0.68,
    )
    colorbar.set_label(zlabel, fontsize=8, labelpad=4)
    colorbar.ax.tick_params(labelsize=6.5, length=2)

def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    metrics = (
        ("PICP", "95% interval coverage", "PICP", "GnBu", lambda v: f"{v:.0%}"),
        ("MPIW", "Mean interval width", "MPIW", "Blues_r", lambda v: f"{v:.3f}"),
        ("Brier_score", "Posterior probability score", "Brier", "YlOrBr", lambda v: f"{v:.2f}"),
    )
    fig = plt.figure(figsize=(16.0, 6.2))
    for index, (metric, title, label, cmap_name, formatter) in enumerate(metrics, start=1):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        draw_panel(ax, metric_table(metric), title, label, cmap_name, formatter)
    fig.suptitle(
        "Observation-density and temporal-frequency grid (health weight = 3)",
        fontsize=12,
        x=0.02,
        ha="left",
    )
    fig.subplots_adjust(left=0.015, right=0.99, bottom=0.08, top=0.87, wspace=0.08)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / "combination_metrics_3d.svg", bbox_inches="tight")
    fig.savefig(OUTPUT / "combination_metrics_3d.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT / "combination_metrics_3d.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT / "combination_metrics_3d.tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D metric figure to {OUTPUT}")


if __name__ == "__main__":
    main()
