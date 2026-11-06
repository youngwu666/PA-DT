

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
DATA_DENSITY_ROOT = HERE.parent
METRICS_SCRIPT = DATA_DENSITY_ROOT / "calculate_uncertainty_metrics.py"
OUTPUT_ROOT = HERE / "simulation_results" / "combination_grid"
NOBS_VALUES = (1, 2, 3, 4)
INTERVAL_VALUES = (1, 2, 3, 4)
HEALTH_WEIGHT = 3
SEED = 42


def _load_action4():
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(
        "combination_action4", HERE / "action4.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.experiment_kind = "combined"
    module.health_weights = (HEALTH_WEIGHT,)
    module.random_seed = SEED
    return module


def _run_metrics(module, workbook_path):
    metrics_script = workbook_path.parent / METRICS_SCRIPT.name
    metrics_script.write_bytes(METRICS_SCRIPT.read_bytes())
    result = subprocess.run(
        [sys.executable, "-B", str(metrics_script)],
        cwd=workbook_path.parent,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(
            f"Uncertainty metrics failed for {workbook_path}:\\n{result.stderr}"
        )
    metrics_path = workbook_path.parent / "uncertainty_metrics.xlsx"
    metrics = pd.read_excel(metrics_path, sheet_name="Summary")
    return metrics_path, dict(zip(metrics["Metric"], metrics["Value"]))


def _matrix(rows, metric):
    frame = pd.DataFrame(rows)
    table = frame.pivot(
        index="observation_interval",
        columns="Nobs",
        values=metric,
    )
    return table.reindex(index=INTERVAL_VALUES, columns=NOBS_VALUES)


def _plot_matrix(table, metric, title, colorbar_label, cmap, annotation):
    values = table.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{metric} matrix contains missing/non-finite cells")

    low = float(values.min())
    high = float(values.max())
    span = high - low
    pad = max(span * 0.08, 1e-6)
    vmin = max(0.0, low - pad)
    vmax = high + pad if span else high + max(abs(high) * 0.08, 1e-6)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    image = ax.imshow(
        values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks(np.arange(len(NOBS_VALUES)), [str(v) for v in NOBS_VALUES])
    ax.set_yticks(np.arange(len(INTERVAL_VALUES)), [str(v) for v in INTERVAL_VALUES])
    ax.invert_yaxis()
    ax.set_xlabel("Observations per time step (Nobs)")
    ax.set_ylabel("Observation interval (decision steps)")
    ax.set_title(title, loc="left", pad=12)

    midpoint = (vmin + vmax) / 2
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            label = annotation(value)
            color = "white" if value >= midpoint else "#202020"
            ax.text(column, row, label, ha="center", va="center",
                    color=color, fontsize=10, fontweight="medium")

    ax.set_xticks(np.arange(-0.5, len(NOBS_VALUES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(INTERVAL_VALUES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    output_base = OUTPUT_ROOT / f"{metric.lower()}_matrix"
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _save_summary(rows, matrices):
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_ROOT / "combination_metrics.csv"
    workbook_path = OUTPUT_ROOT / "combination_metrics.xlsx"
    pd.DataFrame(rows).sort_values(
        ["observation_interval", "Nobs"]
    ).to_csv(csv_path, index=False)

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(rows).sort_values(
            ["observation_interval", "Nobs"]
        ).to_excel(writer, sheet_name="All_combinations", index=False)
        for metric, table in matrices.items():
            table.to_excel(writer, sheet_name=metric[:31])


def main():
    if not METRICS_SCRIPT.is_file():
        raise FileNotFoundError(METRICS_SCRIPT)

    module = _load_action4()
    torch.set_num_threads(1)
    all_rows = []

    with threadpool_limits(limits=1):
        for interval in INTERVAL_VALUES:
            for n_obs in NOBS_VALUES:
                captured = io.StringIO()
                try:
                    with contextlib.redirect_stdout(captured):
                        graph = module.run_experiment(
                            HEALTH_WEIGHT,
                            n_obs=n_obs,
                            observation_interval=interval,
                            save_results=True,
                        )
                    case_name = f"Nobs_{n_obs}_interval_{interval}"
                    result_folder = (
                        HERE / "simulation_results" / case_name / "health_3"
                    )
                    workbook_path = (
                        result_folder / "digital_twin_simulation_results_4.xlsx"
                    )
                    if not workbook_path.is_file():
                        raise FileNotFoundError(workbook_path)

                    metrics_path, metrics = _run_metrics(module, workbook_path)
                    settings = graph.experiment_settings
                    row = {
                        "Nobs": n_obs,
                        "observation_interval": interval,
                        "observation_steps": int(graph.hist_var["observed"].sum()),
                        "evaluated_state_steps": int(metrics["n_time_steps"]),
                        "observation_records": int(
                            graph.hist_var["n_obs_used"].sum()
                        ),
                        "PICP": float(metrics["PICP"]),
                        "MPIW": float(metrics["MPIW"]),
                        "Brier_score": float(metrics["Brier_score"]),
                        "covered_steps": int(metrics["covered_steps"]),
                        "health_weight": HEALTH_WEIGHT,
                        "seed": SEED,
                        "q_retrained": False,
                        "q_table": Path(settings["q_table"]).name,
                        "result_workbook": str(workbook_path.relative_to(HERE)),
                        "uncertainty_workbook": str(metrics_path.relative_to(HERE)),
                    }
                    all_rows.append(row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                except Exception:
                    print(captured.getvalue()[-6000:], file=sys.stderr)
                    raise

    matrices = {
        "PICP": _matrix(all_rows, "PICP"),
        "MPIW": _matrix(all_rows, "MPIW"),
        "Brier_score": _matrix(all_rows, "Brier_score"),
    }
    _save_summary(all_rows, matrices)
    _plot_matrix(
        matrices["PICP"], "PICP",
        "95% credible interval coverage",
        "PICP (higher is better)", "GnBu",
        lambda value: f"{value:.1%}",
    )
    _plot_matrix(
        matrices["MPIW"], "MPIW",
        "Mean posterior interval width",
        "MPIW (lower is narrower)", "Blues_r",
        lambda value: f"{value:.4f}",
    )
    _plot_matrix(
        matrices["Brier_score"], "Brier_score",
        "Posterior probability calibration",
        "Brier score (lower is better)", "YlOrBr",
        lambda value: f"{value:.3f}",
    )

    print(f"Combination summary: {OUTPUT_ROOT / 'combination_metrics.xlsx'}")
    print(f"Matrix figures: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
