










from __future__ import annotations

from pathlib import Path
import argparse
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


METRICS_NAME = "uncertainty_metrics.xlsx"
REQUIRED_SHEETS = {"Simulation", "Prediction", "Simulation_Observations"}
DEFAULT_WALL_THICKNESS = 11.0


def _configure_style():
    mpl.rcParams.update({
        "font.family": ["Times New Roman", "DejaVu Serif"],
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.85,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": "#c9ced6",
        "grid.linewidth": 0.55,
        "grid.alpha": 0.68,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def _find_workbook(folder: Path) -> Path:
    candidates = []
    for path in folder.glob("*.xlsx"):
        if path.name == METRICS_NAME or path.name.startswith("~$"):
            continue
        try:
            with pd.ExcelFile(path) as book:
                sheets = set(book.sheet_names)
        except Exception:
            continue
        if REQUIRED_SHEETS.issubset(sheets):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "No simulation workbook with Simulation, Prediction and "
            "Simulation_Observations sheets was found in " + str(folder)
        )
    if len(candidates) > 1:
        warnings.warn(
            "Several result workbooks were found; using the newest one: "
            + ", ".join(path.name for path in candidates),
            RuntimeWarning,
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _read_result(workbook_path: Path):
    data = pd.read_excel(workbook_path, sheet_name=list(REQUIRED_SHEETS))
    history = data["Simulation"].copy()
    prediction = data["Prediction"].copy()
    observations = data["Simulation_Observations"].copy()
    for frame, name in ((history, "Simulation"), (prediction, "Prediction")):
        if "time" not in frame:
            raise ValueError(f"{name} sheet must contain a time column")
        if frame.empty:
            raise ValueError(f"{name} sheet contains no result rows")
        frame.sort_values("time", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if "time" not in observations or "dmax" not in observations:
        raise ValueError("Simulation_Observations must contain time and dmax")
    observations.sort_values("time", inplace=True)
    return history, prediction, observations


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"Required column '{column}' is missing")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Column '{column}' contains non-finite values")
    return values


def _wall_thickness(history: pd.DataFrame) -> float:
    if "p_dmax" not in history or "p_state_num" not in history:
        return DEFAULT_WALL_THICKNESS
    dmax = pd.to_numeric(history["p_dmax"], errors="coerce").to_numpy(dtype=float)
    ratio = pd.to_numeric(history["p_state_num"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(dmax) & np.isfinite(ratio) & (dmax > 1e-9) & (ratio < 1.0)
    if not valid.any():
        return DEFAULT_WALL_THICKNESS
    estimate = dmax[valid] / np.maximum(1.0 - ratio[valid], 1e-9)
    estimate = estimate[np.isfinite(estimate) & (estimate > 0.0)]
    if not len(estimate):
        return DEFAULT_WALL_THICKNESS
    return float(np.median(estimate))


def _interval(ax, time, lower, upper, color, alpha=0.24, label=None):
    ax.fill_between(time, lower, upper, color=color, alpha=alpha,
                    linewidth=0.0, label=label)


def _action_values(frame: pd.DataFrame) -> np.ndarray:
    if "current_action_value" in frame:
        return _numeric(frame, "current_action_value")
    names = frame.get("current_action", pd.Series(index=frame.index, dtype=str)).astype(str)
    mapping = {
        "Do_nothing": 0.0,
        "Pigging": 1.0,
        "Adding_inhibitor": 2.0,
        "Replacement": 3.0,
    }
    values = names.map(mapping).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("The workbook contains an unrecognized action name")
    return values


def _step(ax, time, values, color, linestyle="-", marker=False, label=None):
    ax.step(time, values, where="post", color=color, linestyle=linestyle,
            linewidth=1.0, label=label, zorder=3)
    if marker:
        ax.plot(time, values, linestyle="None", marker="o", markersize=3.2,
                color=color, zorder=4)


def _panel_header(ax, title, handles):
    
    ax.set_axis_off()
    ax.text(-0.015, 0.84, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=12)
    if handles:
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.05),
                  borderaxespad=0.0, ncol=len(handles), fontsize=8.5,
                  handlelength=2.0, columnspacing=1.3, handletextpad=0.55)


def _phase_header(label_ax, state_ax, history_end, x_min, x_max):
    
    label_ax.set_axis_off()
    fraction = (history_end - x_min) / (x_max - x_min)
    for left, right, label in (
        (0.0, fraction, "Simulation"),
        (fraction, 1.0, "Prediction"),
    ):
        if right <= left:
            continue
        label_ax.text((left + right) / 2.0, 0.62, label,
                transform=label_ax.transAxes,
                ha="center", va="center", fontsize=9, color="#52545a")
        state_ax.annotate("", xy=(x_min + right * (x_max - x_min), 0.0),
                    xytext=(x_min + left * (x_max - x_min), 0.0),
                    xycoords="data", textcoords="data",
                    arrowprops={"arrowstyle": "<->", "color": "#777a80", "lw": 0.6},
                    annotation_clip=False, zorder=8)


def _format_time_axes(axes, x_min, x_max, history_end):
    ticks = MaxNLocator(nbins=7, integer=True).tick_values(x_min, x_max)
    ticks = [value for value in ticks if x_min <= value <= x_max
             and abs(value - history_end) > 0.04 * (x_max - x_min)
             and abs(value - x_max) > 0.04 * (x_max - x_min)]
    ticks = sorted(set(ticks + [history_end, x_max]))
    labels = []
    for value in ticks:
        if value == history_end:
            labels.append(rf"$t_c={value:g}$")
        elif value == x_max:
            labels.append(rf"$t_p={value:g}$")
        else:
            labels.append(f"{value:g}")
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax.axvline(history_end, color="#8c929b", linewidth=0.65, zorder=1)
        if abs(history_end - x_max) > 1e-8:
            ax.axvline(x_max, color="#8c929b", linewidth=0.55, zorder=1)
    axes[-1].set_xticks(ticks, labels)
    axes[-1].get_xticklabels()[-1].set_ha("right")
    for ax in axes[:-1]:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)


def _save_figure(fig, output_path: Path):
    
    fig.savefig(output_path.with_suffix(".svg"), format="svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_simulation_and_prediction(history, prediction, output_path: Path):
    
    past = _numeric(history, "time")
    future = _numeric(prediction, "time")
    tc = float(past[-1])
    x_min = float(min(past.min(), future.min()))
    x_max = float(max(past.max(), future.max()))

    truth = 100.0 * _numeric(history, "p_state_num")
    estimated = 100.0 * _numeric(history, "d_state_num")
    estimate_lower = 100.0 * _numeric(history, "Lower")
    estimate_upper = 100.0 * _numeric(history, "Upper")
    predicted = 100.0 * _numeric(prediction, "d_state_num")
    predict_lower = 100.0 * _numeric(prediction, "Lower")
    predict_upper = 100.0 * _numeric(prediction, "Upper")

    actions_history = _action_values(history)
    actions_future = _action_values(prediction)
    rewards_history = _numeric(history, "reward")
    rewards_future = _numeric(prediction, "reward")
    reward_lower = _numeric(history, "reward_Lower")
    reward_upper = _numeric(history, "reward_Upper")
    reward_predict_lower = _numeric(prediction, "reward_Lower")
    reward_predict_upper = _numeric(prediction, "reward_Upper")

    blue = "#2458a6"
    blue_band = "#8db4dc"
    truth_color = "#4b4f55"
    pink = "#ed8ab6"
    green = "#2e7d4f"
    green_band = "#9bc2a6"

    
    fig = plt.figure(figsize=(7.2, 6.2))
    grid = fig.add_gridspec(8, 1,
                           height_ratios=[0.65, 2.0, 0.30, 0.65,
                                          0.88, 0.30, 0.65, 1.35],
                           hspace=0.10)
    fig.subplots_adjust(left=0.155, right=0.975, top=0.97, bottom=0.09)
    state_header = fig.add_subplot(grid[0])
    state_ax = fig.add_subplot(grid[1])
    phase_footer = fig.add_subplot(grid[2], sharex=state_ax)
    action_header = fig.add_subplot(grid[3])
    action_ax = fig.add_subplot(grid[4], sharex=state_ax)
    action_footer = fig.add_subplot(grid[5])
    action_footer.set_axis_off()
    reward_header = fig.add_subplot(grid[6])
    reward_ax = fig.add_subplot(grid[7], sharex=state_ax)
    axes = (state_ax, action_ax, reward_ax)

    
    _interval(state_ax, past, estimate_lower, estimate_upper, blue_band, 0.34)
    _interval(state_ax, future, predict_lower, predict_upper, blue_band, 0.28)
    state_ax.plot(past, truth, color=truth_color, linestyle="--", linewidth=0.9,
                  label="ground truth", zorder=4)
    state_ax.plot(past, estimated, color=blue, linewidth=1.25,
                  label="estimated", zorder=5)
    state_ax.plot(future, predicted, color=blue, linewidth=1.25,
                  label="prediction", zorder=5)
    _panel_header(state_header, "a) Corrosion state", [
        Line2D([], [], color=truth_color, linestyle="--", lw=0.9, label="Ground truth"),
        Line2D([], [], color=blue, lw=1.25, label="PA-DT estimated"),
        Patch(facecolor=blue_band, alpha=0.34, label="95% interval"),
    ])
    state_ax.set_ylim(0.0, 105.0)
    state_ax.set_yticks([0, 30, 60, 90, 100])
    state_ax.set_yticklabels(["0%", "30%", "60%", "90%", "100%"])
    state_ax.axhline(20.0, color="#c9ced6", linestyle="--",
                     linewidth=0.55, alpha=0.68, zorder=0.5)
    _phase_header(phase_footer, state_ax, tc, x_min, x_max)

    
    action_handles = [
        Line2D([], [], color=pink, marker="o", markersize=3.2, linestyle="None", lw=0, label="DT decision"),
    ]
    if "best_action_value" in history:
        
        action_ax.step(past, _numeric(history, "best_action_value"), where="post",
                       color=truth_color, linestyle="--", linewidth=0.9, zorder=5)
        action_handles.append(Line2D([], [], color=truth_color, linestyle="--",
                                     lw=0.9, label="Best action"))
    action_ax.plot(past, actions_history, linestyle="None", marker="o",
                   markersize=3.2, color=pink, zorder=4)
    action_ax.plot(future, actions_future, linestyle="None", marker="o",
                   markersize=3.2, color=pink, zorder=4)
    _panel_header(action_header, "b) Control inputs", action_handles)
    action_ax.set_yticks([0, 1, 2, 3])
    action_ax.set_yticklabels(["do nothing", "pigging", "inhibitor", "replacement"], fontsize=8)
    action_ax.set_ylim(-0.45, 3.45)

    
    _interval(reward_ax, past, reward_lower, reward_upper, green_band, 0.32)
    _interval(reward_ax, future, reward_predict_lower, reward_predict_upper, green_band, 0.22)
    reward_ax.plot(past, rewards_history, color=green, linewidth=1.1, label="estimated reward")
    reward_ax.plot(future, rewards_future, color=green, linewidth=1.1,
                   label="predicted reward")
    _panel_header(reward_header, "c) Reward function", [])
    reward_ax.set_ylabel("Reward")
    reward_ax.set_yticks([-150, -100, -50, 0])
    reward_ax.set_xlabel("Decision timestep, $t$")
    _format_time_axes(axes, x_min, x_max, tc)
    _save_figure(fig, output_path)


def plot_sensor_dmax(history, observations, output_path: Path):
    
    time = _numeric(history, "time")
    ratio = _numeric(history, "d_state_num")
    lower_ratio = _numeric(history, "Lower")
    upper_ratio = _numeric(history, "Upper")
    wall = _wall_thickness(history)

    estimate = wall * (1.0 - ratio)
    lower = wall * (1.0 - upper_ratio)
    upper = wall * (1.0 - lower_ratio)
    observed_time = _numeric(observations, "time")
    observed_dmax = _numeric(observations, "dmax")
    truth = (_numeric(history, "p_dmax") if "p_dmax" in history
             else wall * (1.0 - _numeric(history, "p_state_num")))

    purple = "#4e2a91"
    purple_band = "#b8a9d8"
    truth_color = "#34383f"
    fig = plt.figure(figsize=(7.2, 3.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.44, 2.0], hspace=0.08)
    fig.subplots_adjust(left=0.155, right=0.975, top=0.97, bottom=0.18)
    header = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1])
    ax.fill_between(time, lower, upper, color=purple_band, alpha=0.42,
                    linewidth=0.0, label="95% posterior interval")
    ax.plot(time, truth, color=truth_color, linestyle="--", linewidth=0.9,
            label="ground truth", zorder=2)
    ax.plot(time, estimate, color=purple, linewidth=1.25,
            label="estimated", zorder=3)
    
    ax.scatter(observed_time, observed_dmax, s=12, color=purple, edgecolor="white",
               linewidth=0.3, alpha=0.75, zorder=5, label="observed")
    _panel_header(header, "Corrosion depth observations", [
        Line2D([], [], color=truth_color, linestyle="--", lw=0.9, label="Ground truth"),
        Line2D([], [], color=purple, lw=1.25, label="PA-DT estimated"),
        Line2D([], [], color=purple, marker="o", markersize=3, lw=0, label="Observed"),
        Patch(facecolor=purple_band, alpha=0.42, label="95% interval"),
    ])
    ax.set_xlabel("Decision timestep, $t$")
    ax.set_ylabel("$d_{\\max}$ [mm]")
    x_min, x_max = float(time.min()), float(time.max())
    ax.set_xlim(x_min, x_max)
    ax.axvline(x_max, color="#8c929b", linewidth=0.55, zorder=1)
    depth_values = np.concatenate([lower, upper, estimate, truth, observed_dmax])
    ax.set_ylim(bottom=min(0.0, float(depth_values.min()) * 1.08),
                top=max(1.0, float(depth_values.max()) * 1.08))
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    _save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path,
                        help="Result folder; defaults to the directory containing this script")
    args = parser.parse_args()
    folder = (args.folder or Path(__file__).resolve().parent).resolve()
    workbook = _find_workbook(folder)
    history, prediction, observations = _read_result(workbook)
    _configure_style()
    plot_simulation_and_prediction(history, prediction, folder / "simulation_and_prediction.svg")
    plot_sensor_dmax(history, observations, folder / "sensor_dmax.svg")
    print(f"Input workbook: {workbook.name}")
    print(f"Saved: {folder / 'simulation_and_prediction.svg'}")
    print(f"Saved: {folder / 'sensor_dmax.svg'}")


if __name__ == "__main__":
    main()
