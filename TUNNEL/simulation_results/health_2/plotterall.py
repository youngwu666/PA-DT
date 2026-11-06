














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

def terminal_rows(frame):
    
    failed = frame.get('failed', pd.Series(False, index=frame.index))
    terminal = failed.fillna(False).eq(True).to_numpy()
    if 'failure_probability' in frame and 'current_action' in frame:
        terminal = terminal | (
            (pd.to_numeric(frame['failure_probability'], errors='coerce') >= 1 - 1e-9)
            & frame['current_action'].isna()
        ).to_numpy()
    return terminal


def read_plot_results(excel_file):
    
    data = pd.read_excel(excel_file, sheet_name=None)
    settings = data.get('Settings', pd.DataFrame(columns=['setting', 'value']))
    metadata = dict(zip(settings['setting'], settings['value']))
    method = metadata.get('method', 'dbn_qlearning')
    result = {}
    for key, sheet in (('history', 'Simulation'), ('prediction', 'Prediction'),
                       ('observations', 'Simulation_Observations')):
        frame = data[sheet].copy()
        frame.attrs['method'] = method
        frame.attrs['strategy_label'] = metadata.get('strategy_label', 'Traditional >80%')
        frame.attrs['maintenance_threshold'] = metadata.get('threshold', 0.80)
        if 'time' not in frame:
            raise ValueError(f'{sheet}  time ')
        frame.sort_values('time', kind='stable', inplace=True)
        frame.reset_index(drop=True, inplace=True)
        if key != 'observations':
            if key == 'history' and frame.empty:
                raise ValueError('Simulation ')
            if frame['time'].duplicated().any():
                raise ValueError(f'{sheet}  time ')
            probability_sheet = f'{sheet}_State_Prob'
            if probability_sheet in data:
                probability = data[probability_sheet].set_index('time')
                if probability.index.has_duplicates:
                    raise ValueError(f'{probability_sheet}  time ')
                if not frame['time'].isin(probability.index).all():
                    raise ValueError(f'{probability_sheet} ')
                frame.attrs['state_probability'] = probability.reindex(frame['time'])
        result[key] = frame
    return result


def damage_regions(history, observations=None):
    
    if 'p_state' in history:
        values = history['p_state'].astype('string').str.split(';').str[0]
    elif observations is not None and 'true_region' in observations:
        values = observations['true_region']
    else:
        values = history['d_state'].astype('string').str.split(';').str[0]
    values = pd.to_numeric(values, errors='coerce').dropna()
    return sorted({int(value) for value in values if value > 0})


def regional_state(frame, region):
    




    if region is None:
        return frame.copy()
    result = frame.copy()
    if 'p_state' in frame:
        states = frame['p_state'].astype('string').str.split(';')
        locations = pd.to_numeric(states.str[0], errors='coerce').to_numpy(dtype=float)
        damage = pd.to_numeric(states.str[1], errors='coerce').to_numpy(dtype=float)
        result['p_state_num'] = np.where(
            np.isfinite(locations), np.where(locations == region, damage, 0.0), np.nan
        )
    if frame.empty or frame.attrs.get('method') == 'traditional_threshold':
        return result
    if 'state_probability' not in frame.attrs:
        raise ValueError(' Simulation_State_Prob  '
                         'Prediction_State_Prob Lower/Upper ')
    source = frame.attrs['state_probability']
    time = frame['time'] if 'time' in frame else frame.index
    columns = [name for name in source if ';' in str(name)]
    if not columns:
        raise ValueError('“;”')
    state_regions = np.array([int(name.split(';')[0]) for name in columns])
    damage = np.array([float(name.split(';')[1]) for name in columns])
    values = np.where(state_regions == region, damage, 0.0)
    weights = source.reindex(time)[columns].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < -1e-12):
        raise ValueError('')
    weights = np.maximum(weights, 0.0)
    totals = weights.sum(axis=1)
    valid = (totals > 0) & ~terminal_rows(frame)
    if np.any((totals == 0) & ~terminal_rows(frame)):
        raise ValueError('')
    if not np.allclose(totals[valid], 1.0, atol=1e-8, rtol=0):
        raise ValueError('1')
    summary = np.full((len(frame), 3), np.nan)
    if valid.any():
        probabilities = weights[valid] / totals[valid, None]
        order = np.argsort(values, kind='stable')
        cdf = np.cumsum(probabilities[:, order], axis=1)
        cdf[:, -1] = 1.0
        summary[valid, 0] = probabilities @ values
        summary[valid, 1] = values[order][np.argmax(cdf >= 0.025, axis=1)]
        summary[valid, 2] = values[order][np.argmax(cdf >= 0.975, axis=1)]
    result[['d_state_num', 'Lower', 'Upper']] = summary
    return result


def regional_observations(observations, region):
    
    result = observations.copy()
    if region is not None and not result.empty:
        if 'estimated_region' not in result:
            raise ValueError(' estimated_region')
        locations = pd.to_numeric(result['estimated_region'], errors='coerce')
        if locations.isna().any():
            raise ValueError(' estimated_region')
        result['estimated_stiffness_loss'] = np.where(
            locations == region, result['estimated_stiffness_loss'], 0.0
        )
    return result



METRICS_NAME = "uncertainty_metrics.xlsx"
REQUIRED_SHEETS = {"Simulation", "Prediction", "Simulation_Observations"}
ACTION_LABELS = ["do nothing", "major rehab", "restrict operations"]
ACTION_NAMES = {
    "do_nothing": 0.0,
    "major_rehab": 1.0,
    "restrict_operations": 2.0,
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


def _result_folder(folder: Path | None) -> Path:
    
    if folder is not None:
        return folder.expanduser().resolve()
    here = Path(__file__).resolve().parent
    for path in here.glob("*.xlsx"):
        if path.name == METRICS_NAME or path.name.startswith("~$"):
            continue
        try:
            with pd.ExcelFile(path) as book:
                if REQUIRED_SHEETS.issubset(book.sheet_names):
                    return here
        except Exception:
            continue
    results = here.parent / "simulation_results"
    return results if results.is_dir() else here


def _find_workbook(folder: Path) -> Path:
    if folder.is_file():
        return folder
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
    data = read_plot_results(workbook_path)
    history = data["history"]
    prediction = data["prediction"]
    observations = data["observations"]
    for frame, name in ((history, "Simulation"), (prediction, "Prediction")):
        if "time" not in frame:
            raise ValueError(f"{name} sheet must contain a time column")
        if frame.empty and name == "Simulation":
            raise ValueError(f"{name} sheet contains no result rows")
        frame.sort_values("time", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if "time" not in observations or "estimated_stiffness_loss" not in observations:
        raise ValueError(
            "Simulation_Observations must contain time and estimated_stiffness_loss"
        )
    observations.sort_values("time", inplace=True)
    return history, prediction, observations


def _terminal_rows(frame: pd.DataFrame) -> np.ndarray:
    
    return terminal_rows(frame)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"Required column '{column}' is missing")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    allowed_missing = (_terminal_rows(frame) & frame[column].isna().to_numpy()
                       if column != "time" else np.zeros(len(frame), dtype=bool))
    if not (np.isfinite(values) | allowed_missing).all():
        raise ValueError(f"Column '{column}' contains non-finite values")
    return values


def _interval(ax, time, lower, upper, color, alpha=0.24, label=None):
    ax.fill_between(time, lower, upper, color=color, alpha=alpha,
                    linewidth=0.0, label=label)


def _action_values(frame: pd.DataFrame, column: str = "current_action") -> np.ndarray:
    value_column = "current_action_value" if column == "current_action" else "best_action_value"
    names = frame.get(column, pd.Series(index=frame.index, dtype=str))
    if names.notna().any():
        values = names.map(ACTION_NAMES).to_numpy(dtype=float)
    elif value_column in frame:
        values = _numeric(frame, value_column)
    else:
        values = names.map(ACTION_NAMES).to_numpy(dtype=float)
    valid = np.isin(values, np.arange(len(ACTION_LABELS)))
    if not (valid | (np.isnan(values) & _terminal_rows(frame))).all():
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
             and abs(value - history_end) > 0.08 * (x_max - x_min)
             and abs(value - x_max) > 0.08 * (x_max - x_min)]
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



def plot_actual_strategies(results, output_path):
    
    times = [_numeric(item['history'], 'time') for item in results]
    if any(not np.array_equal(time, times[0]) for time in times[1:]):
        raise ValueError('')
    regions = sorted({region for item in results
                      for region in damage_regions(item['history'])}) or [None]
    colors = ['#2458a6', '#c17d38', '#608b76', '#856da2']
    fig = plt.figure(figsize=(8.5, 6.2 + 2.2 * (len(regions) - 1)))
    grid = fig.add_gridspec(3 * len(regions) + 5, 1,
        height_ratios=[0.65, 2.0, 0.30] * len(regions)
        + [0.65, 0.88, 0.30, 0.65, 1.35], hspace=0.10)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.09)
    axes = []
    handles = [Line2D([], [], color=colors[i % len(colors)], lw=1.25,
                     label=item['label']) for i, item in enumerate(results)]

    def header(ax, title):
        ax.set_axis_off()
        ax.text(-0.015, 0.84, title, transform=ax.transAxes,
            ha='left', va='top', fontsize=12)
        ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, -0.05),
            borderaxespad=0, frameon=False, ncol=min(4, len(handles)), fontsize=8.5)

    for index, region in enumerate(regions):
        title_ax = fig.add_subplot(grid[3 * index])
        ax = fig.add_subplot(grid[3 * index + 1], sharex=axes[0] if axes else None)
        axes.append(ax)
        for i, (item, time) in enumerate(zip(results, times)):
            state = regional_state(item['history'], region)
            ax.plot(time, 100 * _numeric(state, 'p_state_num'),
                color=colors[i % len(colors)], linewidth=1.25,
                linestyle='--' if item['history'].attrs.get('method') == 'traditional_threshold' else '-')
        header(title_ax, f'a{index + 1}) Region {region}: actual stiffness loss'
            if region is not None else 'a) Actual stiffness loss')
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 15, 30, 45, 60, 75, 90],
            ['0%', '15%', '30%', '45%', '60%', '75%', '90%'])
        ax.axhline(80, color='#999999', linestyle=':', linewidth=0.7)
        if any(item['history'].attrs.get('maintenance_threshold') == 0.60 for item in results):
            ax.axhline(60, color='#999999', linestyle=':', linewidth=0.7)
    row = 3 * len(regions)
    title_ax = fig.add_subplot(grid[row])
    action_ax = fig.add_subplot(grid[row + 1], sharex=axes[0])
    header(title_ax, 'b) Executed control inputs')
    reward_title = fig.add_subplot(grid[row + 3])
    reward_ax = fig.add_subplot(grid[row + 4], sharex=axes[0])
    header(reward_title, 'c) Realized reward after each action')
    for i, (item, time) in enumerate(zip(results, times)):
        history = item['history']
        action_ax.step(time[:-1], _action_values(history.iloc[:-1]), where='post',
            color=colors[i % len(colors)], linewidth=1.0)
        action_ax.plot(time[:-1], _action_values(history.iloc[:-1]), linestyle='None',
            marker='o', markersize=2.6, color=colors[i % len(colors)])
        reward_ax.plot(time[1:], _numeric(history.iloc[1:], 'realized_reward'),
            color=colors[i % len(colors)], linewidth=1.1)
    action_ax.set_yticks(np.arange(len(ACTION_LABELS)))
    action_ax.set_yticklabels(ACTION_LABELS, fontsize=8)
    action_ax.set_ylim(-0.45, len(ACTION_LABELS) - 0.55)
    reward_ax.set_ylabel('Realized reward')
    reward_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
    reward_ax.set_xlabel('Decision timestep, $t$ (actual simulation only)')
    axes.extend([action_ax, reward_ax])
    for ax in axes:
        ax.set_xlim(float(times[0][0]), float(times[0][-1]))
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color='#c9ced6', linewidth=0.55, alpha=0.68)
    for ax in axes[:-1]:
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
    _save_figure(fig, output_path)


def plot_simulation_and_prediction(history, prediction, output_path: Path, observations=None):
    
    if history.attrs.get('method') == 'traditional_threshold':
        return plot_actual_strategies([dict(history=history,
            label=history.attrs.get('strategy_label', 'Traditional >80%'))], output_path)
    past = _numeric(history, "time")
    future = _numeric(prediction, "time")
    tc = float(past[-1])
    x_min = float(min(past.min(), future.min())) if len(future) else float(past.min())
    x_max = float(max(past.max(), future.max())) if len(future) else tc

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
    purple = "#4e2a91"
    pink = "#ed8ab6"
    green = "#2e7d4f"
    green_band = "#9bc2a6"

    
    regions = damage_regions(history) or [None]
    fig = plt.figure(figsize=(7.2, 6.2 + 2.2 * (len(regions) - 1)))
    grid = fig.add_gridspec(3 * len(regions) + 5, 1,
                           height_ratios=[0.65, 2.0, 0.30] * len(regions)
                           + [0.65, 0.88, 0.30, 0.65, 1.35],
                           hspace=0.10)
    fig.subplots_adjust(left=0.155, right=0.975, top=0.97, bottom=0.09)
    state_axes = []
    for i, region in enumerate(regions):
        state_header = fig.add_subplot(grid[3 * i])
        state_ax = fig.add_subplot(grid[3 * i + 1],
            sharex=state_axes[0] if state_axes else None)
        phase_footer = fig.add_subplot(grid[3 * i + 2])
        phase_footer.set_axis_off()
        state_axes.append(state_ax)
        state_history = regional_state(history, region)
        state_prediction = regional_state(prediction, region)
        observed = (regional_observations(observations, region)
                    if observations is not None else None)

        _interval(state_ax, past, 100 * _numeric(state_history, "Lower"),
            100 * _numeric(state_history, "Upper"), blue_band, 0.34)
        _interval(state_ax, future, 100 * _numeric(state_prediction, "Lower"),
            100 * _numeric(state_prediction, "Upper"), blue_band, 0.28)
        state_ax.plot(past, 100 * _numeric(state_history, "p_state_num"),
            color=truth_color, linestyle="--", linewidth=0.9, zorder=4)
        state_ax.plot(past, 100 * _numeric(state_history, "d_state_num"),
            color=blue, linewidth=1.25, zorder=5)
        state_ax.plot(future, 100 * _numeric(state_prediction, "d_state_num"),
            color=blue, linewidth=1.25, zorder=5)
        if observed is not None:
            state_ax.scatter(_numeric(observed, "time"),
                100 * _numeric(observed, "estimated_stiffness_loss"),
                s=12, color=purple, edgecolor="white", linewidth=0.3,
                alpha=0.75, zorder=6)
        title = ("a) Damage state (stiffness loss)" if region is None else
                 f"a{i + 1}) Region {region}: stiffness loss")
        state_handles = [
            Line2D([], [], color=truth_color, linestyle="--", lw=0.9, label="Ground truth"),
            Line2D([], [], color=blue, lw=1.25, label="PA-DT estimated"),
        ]
        if observed is not None:
            state_handles.append(Line2D([], [], color=purple, marker="o",
                markersize=3, lw=0, label="Observed"))
        state_handles.append(Patch(facecolor=blue_band, alpha=0.34, label="95% interval"))
        _panel_header(state_header, title, state_handles)
        state_ax.set_ylim(0.0, 105.0)
        state_ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
        state_ax.set_yticklabels(["0%", "15%", "30%", "45%", "60%", "75%", "90%"])
        state_ax.axhline(80.0, color="#c9ced6", linestyle="--",
            linewidth=0.55, alpha=0.68, zorder=0.5)
    _phase_header(phase_footer, state_axes[-1], tc, x_min, x_max)
    row = 3 * len(regions)
    action_header = fig.add_subplot(grid[row])
    action_ax = fig.add_subplot(grid[row + 1], sharex=state_axes[0])
    action_footer = fig.add_subplot(grid[row + 2])
    action_footer.set_axis_off()
    reward_header = fig.add_subplot(grid[row + 3])
    reward_ax = fig.add_subplot(grid[row + 4], sharex=state_axes[0])
    axes = (*state_axes, action_ax, reward_ax)

    
    action_handles = [
        Line2D([], [], color=pink, marker="o", markersize=3.2, linestyle="None", lw=0, label="DT decision"),
    ]
    if "best_action_value" in history or "best_action" in history:
        
        action_ax.step(past, _action_values(history, "best_action"), where="post",
                       color=truth_color, linestyle="--", linewidth=0.9, zorder=5)
        action_handles.append(Line2D([], [], color=truth_color, linestyle="--",
                                     lw=0.9, label="Best action"))
    action_ax.plot(past, actions_history, linestyle="None", marker="o",
                   markersize=3.2, color=pink, zorder=4)
    action_ax.plot(future, actions_future, linestyle="None", marker="o",
                   markersize=3.2, color=pink, zorder=4)
    _panel_header(action_header, "b) Control inputs", action_handles)
    action_ax.set_yticks(np.arange(len(ACTION_LABELS)))
    action_ax.set_yticklabels(ACTION_LABELS, fontsize=8)
    action_ax.set_ylim(-0.45, len(ACTION_LABELS) - 0.55)

    
    _interval(reward_ax, past, reward_lower, reward_upper, green_band, 0.32)
    _interval(reward_ax, future, reward_predict_lower, reward_predict_upper, green_band, 0.22)
    reward_ax.plot(past, rewards_history, color=green, linewidth=1.1, label="estimated reward")
    reward_ax.plot(future, rewards_future, color=green, linewidth=1.1,
                   label="predicted reward")
    _panel_header(reward_header, "c) Reward function", [])
    reward_ax.set_ylabel("Reward")
    
    reward_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
    reward_ax.set_xlabel("Decision timestep, $t$")
    _format_time_axes(axes, x_min, x_max, tc)
    _save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path,
                        help=" simulation_results")
    args = parser.parse_args()
    _configure_style()
    location = _result_folder(args.folder)
    folders = ([location] if location.is_file() or any(location.glob('*.xlsx')) else
               sorted(path for path in location.glob('health_*') if path.is_dir()) or [location])
    for folder in folders:
        if folder.is_file():
            workbooks = [folder]
        else:
            workbooks = [folder / name for name in
                ('digital_twin_simulation_results.xlsx', 'traditional_strategy_results.xlsx',
                 'traditional_strategy2_results.xlsx', 'traditional_strategy3_results.xlsx')
                if (folder / name).is_file()] or [_find_workbook(folder)]
        for workbook in workbooks:
            history, prediction, observations = _read_result(workbook)
            output_folder = workbook.parent
            traditional = history.attrs.get('method') == 'traditional_threshold'
            output_name = (workbook.stem.removesuffix('_results') + '.svg'
                           if traditional else 'simulation_and_prediction.svg')
            plot_simulation_and_prediction(history, prediction, output_folder / output_name,
                observations=observations)
            print(f'Input workbook: {workbook}')
            print(f"Damage regions: {damage_regions(history, observations) or 'healthy only'}")
            print(f'Saved: {output_folder / output_name}')


if __name__ == "__main__":
    main()
