














from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


REQUIRED_SHEETS = {"Simulation", "Prediction", "Simulation_Observations"}
METRICS_NAME = "uncertainty_metrics.xlsx"
DEFAULT_WALL_THICKNESS = 11.0
ACTION_LABELS = ["do nothing", "pigging", "inhibitor", "replacement"]
ACTION_NAMES = {
    "Do_nothing": 0.0,
    "Pigging": 1.0,
    "Adding_inhibitor": 2.0,
    "Replacement": 3.0,
}


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


def _find_workbooks(folder: Path) -> list[Path]:
    
    candidates = []
    for path in sorted(folder.glob("*.xlsx")):
        if path.name == METRICS_NAME or path.name.startswith("~$"):
            continue
        try:
            with pd.ExcelFile(path) as book:
                sheets = set(book.sheet_names)
        except Exception as error:
            warnings.warn(f" {path.name}: {error}", RuntimeWarning)
            continue
        if REQUIRED_SHEETS.issubset(sheets):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            " SimulationPrediction"
            "Simulation_Observations " + str(folder)
        )
    return candidates


def _read_result(workbook_path: Path) -> dict:
    data = pd.read_excel(workbook_path, sheet_name=list(REQUIRED_SHEETS))
    history = data["Simulation"].copy()
    prediction = data["Prediction"].copy()
    observations = data["Simulation_Observations"].copy()
    for frame, name in ((history, "Simulation"), (prediction, "Prediction")):
        if "time" not in frame or frame.empty:
            raise ValueError(f"{workbook_path.name}  {name}  time ")
        frame.sort_values("time", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if "time" not in observations or "dmax" not in observations:
        raise ValueError(f"{workbook_path.name}  Simulation_Observations  time  dmax")
    observations.sort_values("time", inplace=True)
    return {
        "path": workbook_path,
        "label": workbook_path.stem,
        "history": history,
        "prediction": prediction,
        "observations": observations,
    }


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f" '{column}'")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f" '{column}' ")
    return values


def _wall_thickness(history: pd.DataFrame) -> float:
    if "p_dmax" not in history or "p_state_num" not in history:
        return DEFAULT_WALL_THICKNESS
    dmax = pd.to_numeric(history["p_dmax"], errors="coerce").to_numpy(dtype=float)
    ratio = pd.to_numeric(history["p_state_num"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(dmax) & np.isfinite(ratio) & (dmax > 1e-9) & (ratio < 1.0)
    if not valid.any():
        return DEFAULT_WALL_THICKNESS
    estimates = dmax[valid] / np.maximum(1.0 - ratio[valid], 1e-9)
    estimates = estimates[np.isfinite(estimates) & (estimates > 0.0)]
    return float(np.median(estimates)) if len(estimates) else DEFAULT_WALL_THICKNESS


def _action_values(frame: pd.DataFrame, column: str = "current_action") -> np.ndarray:
    value_column = "current_action_value" if column == "current_action" else "best_action_value"
    if value_column in frame:
        return _numeric(frame, value_column)
    names = frame.get(column, pd.Series(index=frame.index, dtype=str)).astype(str)
    values = names.map(ACTION_NAMES).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f" '{column}' ")
    return values


def _colors(count: int) -> list:
    
    if count <= 10:
        cmap = mpl.colormaps["tab10"]
        return [cmap(index / 10.0) for index in range(count)]
    cmap = mpl.colormaps["tab20"]
    return [cmap(index % 20 / 20.0) for index in range(count)]


def _save_figure(fig, output_path: Path):
    
    fig.savefig(output_path.with_suffix(".svg"), format="svg",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _header(ax, title: str, handles: list, legend_slots: int = 0):
    ax.set_axis_off()
    ax.text(-0.015, 0.84, title, transform=ax.transAxes,
            ha="left", va="top", fontsize=12)
    if handles:
        handles = list(handles)
        handles.extend(
            Line2D([], [], alpha=0.0, label=" ")
            for _ in range(max(0, legend_slots - len(handles)))
        )
        ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, -0.05),
                  borderaxespad=0.0, ncol=2, fontsize=8.3,
                  handlelength=2.0, columnspacing=1.25, handletextpad=0.55)


def _phase_header(label_ax, state_ax, x_min: float, x_max: float, history_end: float):
    
    label_ax.set_axis_off()
    fraction = (history_end - x_min) / (x_max - x_min)
    for left, right, label in (
        (0.0, fraction, "Simulation"),
        (fraction, 1.0, "Prediction"),
    ):
        if right <= left:
            continue
        label_ax.text((left + right) / 2.0, 0.62, label,
                transform=label_ax.transAxes, ha="center", va="center",
                fontsize=9, color="#52545a", clip_on=False)
        state_ax.annotate("", xy=(x_min + right * (x_max - x_min), 0.0),
                    xytext=(x_min + left * (x_max - x_min), 0.0),
                    xycoords="data", textcoords="data",
                    arrowprops={"arrowstyle": "<->", "color": "#777a80", "lw": 0.6},
                    annotation_clip=False, zorder=8)


def _format_time_axes(axes, x_min: float, x_max: float, history_ends: list[float]):
    ticks = MaxNLocator(nbins=7, integer=True).tick_values(x_min, x_max)
    ticks = [value for value in ticks if x_min <= value <= x_max]
    ticks = sorted(set(ticks + [x_max]))
    boundaries = sorted(set(history_ends + [x_max]))
    for ax in axes:
        for boundary in boundaries:
            linewidth = 0.6 if boundary in history_ends else 0.55
            ax.axvline(boundary, color="#8c929b", linewidth=linewidth, zorder=1)
    labels = []
    reference_end = history_ends[0] if history_ends else None
    for value in ticks:
        if reference_end is not None and abs(value - reference_end) < 1e-8:
            labels.append(rf"$t_c={value:g}$")
        elif abs(value - x_max) < 1e-8:
            labels.append(rf"$t_p={value:g}$")
        else:
            labels.append(f"{value:g}")
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
    axes[-1].set_xticks(ticks, labels)
    axes[-1].get_xticklabels()[-1].set_ha("right")
    for ax in axes[:-1]:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)


def _step(ax, time, values, color, linestyle="-", marker=False):
    ax.step(time, values, where="post", color=color, linestyle=linestyle,
            linewidth=1.05, zorder=3)
    if marker:
        ax.plot(time, values, linestyle="None", marker="o", markersize=2.8,
                color=color, zorder=4)


def _same_series(series: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    
    if len(series) < 2:
        return True
    reference_time, reference_values = series[0]
    for time, values in series[1:]:
        if np.array_equal(time, reference_time):
            if not np.allclose(values, reference_values, rtol=1e-6, atol=1e-9):
                return False
            continue
        left = max(float(reference_time.min()), float(time.min()))
        right = min(float(reference_time.max()), float(time.max()))
        if right <= left:
            return False
        grid = np.unique(np.concatenate([
            reference_time[(reference_time >= left) & (reference_time <= right)],
            time[(time >= left) & (time <= right)],
        ]))
        if not np.allclose(
            np.interp(grid, reference_time, reference_values),
            np.interp(grid, time, values),
            rtol=1e-6,
            atol=1e-9,
        ):
            return False
    return True


def _truth_state(item: dict) -> tuple[np.ndarray, np.ndarray]:
    history = item["history"]
    return _numeric(history, "time"), 100.0 * _numeric(history, "p_state_num")


def _truth_depth(item: dict) -> tuple[np.ndarray, np.ndarray]:
    history = item["history"]
    time = _numeric(history, "time")
    if "p_dmax" in history:
        return time, _numeric(history, "p_dmax")
    wall = _wall_thickness(history)
    return time, wall * (1.0 - _numeric(history, "p_state_num"))


def _best_actions(item: dict) -> tuple[np.ndarray, np.ndarray]:
    return _numeric(item["history"], "time"), _action_values(
        item["history"], "best_action"
    )


def plot_simulation_and_prediction(results: list[dict], output_path: Path):
    colors = _colors(len(results))
    past_times = [_numeric(item["history"], "time") for item in results]
    future_times = [_numeric(item["prediction"], "time") for item in results]
    x_min = min(float(time.min()) for time in past_times + future_times)
    x_max = max(float(time.max()) for time in past_times + future_times)
    history_ends = [float(time[-1]) for time in past_times]

    legend_slots = 2 * int(np.ceil((len(results) + 3) / 2))
    panel_header_height = max(0.50, 0.34 + 0.22 * (legend_slots // 2))
    fig = plt.figure(figsize=(8.5, 6.3 + 0.45 * (panel_header_height - 0.56)
                              + 0.70 * (panel_header_height - 0.50)))
    grid = fig.add_gridspec(8, 1,
                            height_ratios=[panel_header_height, 2.0, 0.30,
                                           panel_header_height, 0.88, 0.30,
                                           panel_header_height, 1.35],
                            hspace=0.10)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.97, bottom=0.09)
    state_header = fig.add_subplot(grid[0])
    state_ax = fig.add_subplot(grid[1])
    phase_footer = fig.add_subplot(grid[2], sharex=state_ax)
    action_header = fig.add_subplot(grid[3])
    action_ax = fig.add_subplot(grid[4], sharex=state_ax)
    action_footer = fig.add_subplot(grid[5])
    action_footer.set_axis_off()
    reward_header = fig.add_subplot(grid[6])
    reward_ax = fig.add_subplot(grid[7], sharex=state_ax)

    
    for item, color, past, future in zip(results, colors, past_times, future_times):
        history = item["history"]
        prediction = item["prediction"]
        state_lower = 100.0 * _numeric(history, "Lower")
        state_upper = 100.0 * _numeric(history, "Upper")
        predict_lower = 100.0 * _numeric(prediction, "Lower")
        predict_upper = 100.0 * _numeric(prediction, "Upper")
        _fill = dict(color=color, alpha=0.12, linewidth=0.0)
        state_ax.fill_between(past, state_lower, state_upper, **_fill)
        state_ax.fill_between(future, predict_lower, predict_upper, **_fill)
        state_ax.plot(past, 100.0 * _numeric(history, "d_state_num"),
                      color=color, linewidth=1.25, zorder=5)
        state_ax.plot(future, 100.0 * _numeric(prediction, "d_state_num"),
                      color=color, linewidth=1.25, zorder=5)

    
    truth_states = [_truth_state(item) for item in results]
    same_truth = _same_series(truth_states)
    if same_truth:
        state_ax.plot(truth_states[0][0], truth_states[0][1],
                      color="#4b4f55", linestyle="--", linewidth=0.9, zorder=6)
    else:
        for (time, values), color in zip(truth_states, colors):
            state_ax.plot(time, values, color=color, linestyle="--",
                          linewidth=0.9, alpha=0.9, zorder=6)
    state_handles = (
        [Line2D([], [], color=color, lw=1.25, label=item["label"])
         for item, color in zip(results, colors)]
        if len(results) > 1 else []
    )
    state_handles.extend([
        Line2D([], [], color="#4b4f55", lw=1.25, label="PA-DT estimated"),
        Line2D([], [], color="#4b4f55", linestyle="--", lw=0.9,
               label="Ground truth"),
        Patch(facecolor="#9ab7d8", alpha=0.35, label="95% interval"),
    ])
    _header(state_header,
            "a) Corrosion state", state_handles, legend_slots)
    state_ax.set_ylim(0.0, 105.0)
    state_ax.set_yticks([0, 30, 60, 90, 100])
    state_ax.set_yticklabels(["0%", "30%", "60%", "90%", "100%"])
    state_ax.axhline(20.0, color="#c9ced6", linestyle="--",
                     linewidth=0.55, alpha=0.68, zorder=0.5)
    if np.allclose(history_ends, history_ends[0]):
        _phase_header(phase_footer, state_ax, x_min, x_max, history_ends[0])

    
    for item, color, past, future in zip(results, colors, past_times, future_times):
        history = item["history"]
        prediction = item["prediction"]
        action_ax.plot(past, _action_values(history), linestyle="None", marker="o",
                       markersize=2.8, color=color, zorder=4)
        action_ax.plot(future, _action_values(prediction), linestyle="None", marker="o",
                       markersize=2.8, color=color, zorder=4)
    first_history = results[0]["history"]
    best_series = []
    if "best_action_value" in first_history or "best_action" in first_history:
        best_series = [_best_actions(item) for item in results]
        if _same_series(best_series):
            action_ax.step(best_series[0][0], best_series[0][1], where="post",
                           color="#4b4f55", linestyle="--", linewidth=0.9, zorder=6)
        else:
            for (time, values), color in zip(best_series, colors):
                action_ax.step(time, values, where="post", color=color,
                               linestyle="--", linewidth=0.9, alpha=0.9,
                               zorder=6)
    action_handles = (
        [Line2D([], [], color=color, marker="o", markersize=2.8,
                linestyle="None", lw=0, label=item["label"])
         for item, color in zip(results, colors)]
        if len(results) > 1 else []
    )
    if best_series:
        action_handles.append(Line2D([], [], color="#4b4f55", linestyle="--",
                                     lw=0.9, label="Best action"))
    _header(action_header, "b) Control inputs", action_handles, legend_slots)
    action_ax.set_ylabel("action")
    action_ax.set_yticks([0, 1, 2, 3])
    action_ax.set_yticklabels(ACTION_LABELS, fontsize=8)
    action_ax.set_ylim(-0.45, 3.45)

    
    for item, color, past, future in zip(results, colors, past_times, future_times):
        history = item["history"]
        prediction = item["prediction"]
        reward_color = color
        reward_ax.fill_between(past, _numeric(history, "reward_Lower"),
                               _numeric(history, "reward_Upper"),
                               color=reward_color, alpha=0.10, linewidth=0.0)
        reward_ax.fill_between(future, _numeric(prediction, "reward_Lower"),
                               _numeric(prediction, "reward_Upper"),
                               color=reward_color, alpha=0.10, linewidth=0.0)
        reward_ax.plot(past, _numeric(history, "reward"), color=reward_color,
                       linewidth=1.05)
        reward_ax.plot(future, _numeric(prediction, "reward"), color=reward_color,
                       linewidth=1.05)
    _header(reward_header, "c) Reward function", [], legend_slots)
    reward_ax.set_ylabel("Reward")
    reward_ax.set_yticks([-150, -100, -50, 0])
    reward_ax.set_xlabel("Decision timestep, $t$")

    _format_time_axes((state_ax, action_ax, reward_ax), x_min, x_max, history_ends)
    _save_figure(fig, output_path)


def plot_sensor_dmax(results: list[dict], output_path: Path):
    colors = _colors(len(results))
    times = [_numeric(item["history"], "time") for item in results]
    x_min = min(float(time.min()) for time in times)
    x_max = max(float(time.max()) for time in times)
    header_height = max(0.90, 0.34 + 0.22 * np.ceil((len(results) + 4) / 4))
    fig = plt.figure(figsize=(8.5, 3.6 + 0.45 * (header_height - 0.56)))
    grid = fig.add_gridspec(2, 1, height_ratios=[header_height, 2.0], hspace=0.08)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.97, bottom=0.18)
    header = fig.add_subplot(grid[0])
    ax = fig.add_subplot(grid[1])

    for item, color, time in zip(results, colors, times):
        history = item["history"]
        observations = item["observations"]
        wall = _wall_thickness(history)
        ratio = _numeric(history, "d_state_num")
        lower_ratio = _numeric(history, "Lower")
        upper_ratio = _numeric(history, "Upper")
        estimate = wall * (1.0 - ratio)
        lower = wall * (1.0 - upper_ratio)
        upper = wall * (1.0 - lower_ratio)
        ax.fill_between(time, lower, upper, color=color, alpha=0.12, linewidth=0.0)
        ax.plot(time, estimate, color=color, linewidth=1.25)
        ax.scatter(_numeric(observations, "time"), _numeric(observations, "dmax"),
                   s=13, color=color, edgecolor="white", linewidth=0.25,
                   alpha=0.75, zorder=5)

    truth_depths = [_truth_depth(item) for item in results]
    same_truth = _same_series(truth_depths)
    if same_truth:
        ax.plot(truth_depths[0][0], truth_depths[0][1], color="#34383f",
                linestyle="--", linewidth=0.9)
    else:
        for (time, values), color in zip(truth_depths, colors):
            ax.plot(time, values, color=color, linestyle="--",
                    linewidth=0.9, alpha=0.9)

    handles = (
        [Line2D([], [], color=color, lw=1.25, label=item["label"])
         for item, color in zip(results, colors)]
        if len(results) > 1 else []
    )
    handles.extend([
        Line2D([], [], color="#34383f", linestyle="--", lw=0.9,
               label="Ground truth"),
        Line2D([], [], color="#4b4f55", lw=1.25, label="PA-DT estimated"),
        Line2D([], [], color="#33373d", marker="o", markersize=3, lw=0,
               label="Observed"),
        Patch(facecolor="#9ab7d8", alpha=0.35, label="95% interval"),
    ])
    _header(header, "d) Corrosion depth observations [mm]", handles)
    ax.set_xlabel("Decision timestep, $t$")
    ax.set_ylabel("$d_{\\max}$ [mm]")
    ax.set_xlim(x_min, x_max)
    ax.axvline(x_max, color="#8c929b", linewidth=0.55, zorder=1)
    values = []
    for item in results:
        history = item["history"]
        wall = _wall_thickness(history)
        values.extend(wall * (1.0 - _numeric(history, "d_state_num")))
        values.extend(wall * (1.0 - _numeric(history, "Lower")))
        values.extend(wall * (1.0 - _numeric(history, "Upper")))
        values.extend(_numeric(item["observations"], "dmax"))
    ax.set_ylim(bottom=min(0.0, float(np.min(values)) * 1.08),
                top=max(1.0, float(np.max(values)) * 1.08))
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    _save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path,
                        help="")
    args = parser.parse_args()
    folder = (args.folder or Path(__file__).resolve().parent).resolve()
    workbooks = _find_workbooks(folder)
    results = [_read_result(path) for path in workbooks]
    _configure_style()
    plot_simulation_and_prediction(
        results, folder / "simulation_and_prediction_duibi.svg"
    )
    plot_sensor_dmax(results, folder / "sensor_dmax_duibi.svg")
    print(f"Found {len(results)} simulation workbooks:")
    for item in results:
        print(f"  - {item['path'].name}")
    print(f"Saved: {folder / 'simulation_and_prediction_duibi.svg'}")
    print(f"Saved: {folder / 'sensor_dmax_duibi.svg'}")


if __name__ == "__main__":
    main()
