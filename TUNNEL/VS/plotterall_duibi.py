


















from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, PercentFormatter
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
        if key == 'prediction' and sheet not in data:
            frame = pd.DataFrame(columns=['time'])
        else:
            frame = data[sheet].copy()
        if key == 'prediction' and frame.empty and 'time' not in frame:
            frame['time'] = pd.Series(dtype=float)
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
    
    if history.empty:
        return []
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



REQUIRED_SHEETS = {"Simulation", "Simulation_Observations"}
METRICS_NAME = "uncertainty_metrics.xlsx"
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
            " Simulation"
            "Simulation_Observations " + str(folder)
        )
    return candidates


def _read_result(workbook_path: Path) -> dict:
    data = read_plot_results(workbook_path)
    history = data["history"]
    prediction = data["prediction"]
    observations = data["observations"]
    for frame, name in ((history, "Simulation"), (prediction, "Prediction")):
        if "time" not in frame or (frame.empty and name == "Simulation"):
            raise ValueError(f"{workbook_path.name}  {name}  time ")
        frame.sort_values("time", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    if "time" not in observations or "estimated_stiffness_loss" not in observations:
        raise ValueError(
            f"{workbook_path.name}  Simulation_Observations "
            " time  estimated_stiffness_loss"
        )
    observations.sort_values("time", inplace=True)
    return {
        "path": workbook_path,
        "label": (history.attrs.get('strategy_label', 'Traditional >80%')
                  if history.attrs.get('method') == 'traditional_threshold'
                  else 'DBN-Qlearning' if workbook_path.stem == 'digital_twin_simulation_results'
                  else workbook_path.stem),
        "history": history,
        "prediction": prediction,
        "observations": observations,
    }


def _terminal_rows(frame: pd.DataFrame) -> np.ndarray:
    
    return terminal_rows(frame)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f" '{column}'")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    allowed_missing = (_terminal_rows(frame) & frame[column].isna().to_numpy()
                       if column != "time" else np.zeros(len(frame), dtype=bool))
    if not (np.isfinite(values) | allowed_missing).all():
        raise ValueError(f" '{column}' ")
    return values


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
                facecolor="white")
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
    boundaries = sorted(set(history_ends + [x_max]))
    ticks = [value for value in ticks if x_min <= value <= x_max
             and all(abs(value - boundary) > 0.08 * (x_max - x_min)
                     for boundary in boundaries)]
    ticks = sorted(set(ticks + boundaries))
    for ax in axes:
        for boundary in boundaries:
            linewidth = 0.6 if boundary in history_ends else 0.55
            ax.axvline(boundary, color="#8c929b", linewidth=linewidth, zorder=1)
    labels = []
    reference_end = history_ends[0] if history_ends else None
    for value in ticks:
        if reference_end is not None and abs(value - reference_end) < 1e-8:
            labels.append(rf"$t_c={value:g}$")
        elif history_ends and abs(value - x_max) < 1e-8:
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


def _truth_state(item: dict, region=None) -> tuple[np.ndarray, np.ndarray]:
    history = regional_state(item["history"], region)
    return _numeric(history, "time"), 100.0 * _numeric(history, "p_state_num")


def _best_actions(item: dict) -> tuple[np.ndarray, np.ndarray]:
    return _numeric(item["history"], "time"), _action_values(
        item["history"], "best_action"
    )


def _regions(results, include_prediction=False):
    
    frames = ('history', 'prediction') if include_prediction else ('history',)
    return sorted({region for item in results
                   for frame_name in frames
                   for region in damage_regions(item[frame_name], item['observations'])})



def plot_actual_strategies(results, output_path):
    
    times = [_numeric(item['history'], 'time') for item in results]
    future_times = [_numeric(item['prediction'], 'time') for item in results]
    all_times = []
    for item in results:
        for frame in (item['history'], item['prediction']):
            if not frame.empty:
                all_times.extend(_numeric(frame, 'time'))
    xlim = (float(min(all_times)), float(max(all_times)))
    regions = _regions(results, include_prediction=True) or [None]
    colors = ['#2458a6', '#c17d38', '#608b76', '#856da2']
    pa_indices = [i for i, item in enumerate(results)
                  if item['history'].attrs.get('method') != 'traditional_threshold']
    pa_color = colors[pa_indices[0] % len(colors)] if pa_indices else '#2458a6'
    action_handle_count = sum(
        1 if item['history'].attrs.get('method') == 'traditional_threshold'
        else 1 + int('best_action_value' in item['history'] or 'best_action' in item['history'])
        for item in results
    )
    action_columns = min(4, action_handle_count)
    action_rows = int(np.ceil(action_handle_count / action_columns)) if action_columns else 1
    action_header_height = 0.65 if action_rows == 1 else 1.1
    fig = plt.figure(figsize=(8.5, 6.2 + 0.45 * len(regions)
                              + 2.2 * (len(regions) - 1)))
    grid = fig.add_gridspec(3 * len(regions) + 5, 1,
        height_ratios=[1.10, 2.0, 0.30] * len(regions)
        + [action_header_height, 0.88, 0.30, 0.65, 1.35], hspace=0.10)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.97, bottom=0.09)
    axes = []
    phase_axes = []
    state_handles = []
    method_handles = []
    for i, item in enumerate(results):
        if item['history'].attrs.get('method') == 'traditional_threshold':
            handle = Line2D([], [], color=colors[i % len(colors)], lw=1.25,
                             label=item['label'])
            state_handles.append(handle)
            method_handles.append(handle)
    if pa_indices:
        state_handles.extend([
            Line2D([], [], color='#4b4f55', linestyle='--', lw=0.9,
                   label='Ground truth'),
            Line2D([], [], color=pa_color, lw=1.25,
                   label='PA-DT estimated'),
            Patch(facecolor='#9ab7d8', alpha=0.35, label='95% interval'),
        ])
        method_handles.append(Line2D([], [], color=pa_color, lw=1.25,
                                     label='PA-DT'))

    def header(ax, title, handles, columns):
        ax.set_axis_off()
        ax.text(-0.015, 0.84, title, transform=ax.transAxes,
            ha='left', va='top', fontsize=12)
        if handles:
            ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0, -0.05),
                borderaxespad=0, frameon=False, ncol=min(columns, len(handles)),
                fontsize=8.5)

    for index, region in enumerate(regions):
        title_ax = fig.add_subplot(grid[3 * index])
        ax = fig.add_subplot(grid[3 * index + 1], sharex=axes[0] if axes else None)
        phase_ax = fig.add_subplot(grid[3 * index + 2])
        phase_ax.set_axis_off()
        axes.append(ax)
        phase_axes.append(phase_ax)
        for i, (item, time) in enumerate(zip(results, times)):
            state = regional_state(item['history'], region)
            if item['history'].attrs.get('method') == 'traditional_threshold':
                ax.plot(time, 100 * _numeric(state, 'p_state_num'),
                    color=colors[i % len(colors)], linewidth=1.25, linestyle='-')
            else:
                ax.fill_between(time, 100 * _numeric(state, 'Lower'),
                    100 * _numeric(state, 'Upper'), color='#9ab7d8',
                    alpha=0.35, linewidth=0.0, zorder=2)
                ax.plot(time, 100 * _numeric(state, 'p_state_num'),
                    color='#4b4f55', linewidth=0.9, linestyle='--', zorder=4)
                ax.plot(time, 100 * _numeric(state, 'd_state_num'),
                    color=pa_color, linewidth=1.25, linestyle='-', zorder=5)
                if not item['prediction'].empty:
                    prediction = regional_state(item['prediction'], region)
                    future = _numeric(prediction, 'time')
                    ax.fill_between(future, 100 * _numeric(prediction, 'Lower'),
                        100 * _numeric(prediction, 'Upper'), color='#9ab7d8',
                        alpha=0.35, linewidth=0.0, zorder=2)
                    ax.plot(future, 100 * _numeric(prediction, 'd_state_num'),
                        color=pa_color, linewidth=1.25, linestyle='-', zorder=5)
        header(title_ax, f'a{index + 1}) Region {region}: damage-state evolution'
            if region is not None else 'a) Damage-state evolution',
            state_handles, 3)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 15, 30, 45, 60, 75, 90],
            ['0%', '15%', '30%', '45%', '60%', '75%', '90%'])
        ax.axhline(80, color='#999999', linestyle=':', linewidth=0.7)
        if any(item['history'].attrs.get('maintenance_threshold') == 0.60 for item in results):
            ax.axhline(60, color='#999999', linestyle=':', linewidth=0.7)
    history_ends = [float(time[-1]) for time, future in zip(times, future_times)
                    if len(future) and float(future.max()) > float(time[-1])]
    if history_ends and np.allclose(history_ends, history_ends[0]):
        history_end = history_ends[0]
        _phase_header(phase_axes[-1], axes[-1], xlim[0], xlim[1], history_end)
    row = 3 * len(regions)
    title_ax = fig.add_subplot(grid[row])
    action_ax = fig.add_subplot(grid[row + 1], sharex=axes[0])
    action_handles = []
    has_inspection = False
    for item, color in zip(results, colors):
        history = item['history']
        traditional = history.attrs.get('method') == 'traditional_threshold'
        label = item['label']
        if traditional and 'inspection_performed' in history:
            inspection = history['inspection_performed'].fillna(False).astype(bool)
            if inspection.iloc[:-1].any():
                has_inspection = True
                label = f'{label} inspection'
        elif not traditional:
            label = 'PA-DT decision'
        action_handles.append(Line2D([], [], color=color, marker='o', markersize=2.8,
            linestyle='None', lw=0, label=label))
        if not traditional and ('best_action_value' in history or 'best_action' in history):
            action_handles.append(Line2D([], [], color='#4b4f55', linestyle='--',
                lw=0.9, label='PA-DT optimal action'))
    header(title_ax, 'b) Control inputs', action_handles, 4)
    reward_title = fig.add_subplot(grid[row + 3])
    reward_ax = fig.add_subplot(grid[row + 4], sharex=axes[0])
    header(reward_title, 'c) Reward', method_handles, 4)
    for i, (item, time) in enumerate(zip(results, times)):
        history = item['history']
        action_time = time[:-1]
        action_value = _action_values(history.iloc[:-1])
        if history.attrs.get('method') == 'traditional_threshold':
            if 'inspection_performed' in history:
                inspection = history['inspection_performed'].fillna(False).astype(bool).to_numpy()[:-1]
                action_value = np.where(
                    (action_value == ACTION_NAMES['do_nothing']) & inspection,
                    -1.0, action_value
                )
            action_ax.scatter(action_time, action_value, s=9,
                color=colors[i % len(colors)], zorder=4)
        else:
            action_ax.scatter(action_time, action_value, s=9,
                color=colors[i % len(colors)], zorder=5)
            if 'best_action_value' in history or 'best_action' in history:
                best_time, best_value = _best_actions(item)
                action_ax.step(best_time[:-1], best_value[:-1], where='post',
                    color='#4b4f55', linestyle='--', linewidth=0.9, zorder=4)
        reward_ax.plot(time[1:], _numeric(history.iloc[1:], 'realized_reward'),
            color=colors[i % len(colors)], linewidth=1.1)
        prediction = item['prediction']
        if not prediction.empty:
            future = _numeric(prediction, 'time')
            action_ax.scatter(future, _action_values(prediction), s=9,
                color=colors[i % len(colors)], zorder=5)
            reward_ax.plot(future, _numeric(prediction, 'reward'),
                color=colors[i % len(colors)], linewidth=1.1)
            if history.attrs.get('method') != 'traditional_threshold':
                reward_ax.fill_between(future, _numeric(prediction, 'reward_Lower'),
                    _numeric(prediction, 'reward_Upper'),
                    color=colors[i % len(colors)], alpha=0.15, linewidth=0.0)
    if has_inspection:
        action_ax.set_yticks([-1, *range(len(ACTION_LABELS))])
        action_ax.set_yticklabels(['inspection', *ACTION_LABELS], fontsize=8)
        action_ax.set_ylim(-1.45, len(ACTION_LABELS) - 0.55)
    else:
        action_ax.set_yticks(np.arange(len(ACTION_LABELS)))
        action_ax.set_yticklabels(ACTION_LABELS, fontsize=8)
        action_ax.set_ylim(-0.45, len(ACTION_LABELS) - 0.55)
    reward_ax.set_ylabel('Reward')
    reward_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
    reward_values = [value for item in results
                     for value in _numeric(item['history'].iloc[1:], 'realized_reward')
                     if np.isfinite(value)]
    for item in results:
        if not item['prediction'].empty:
            columns = (('reward',) if item['history'].attrs.get('method') == 'traditional_threshold'
                       else ('reward', 'reward_Lower', 'reward_Upper'))
            for column in columns:
                reward_values.extend(value for value in _numeric(item['prediction'], column)
                                     if np.isfinite(value))
    if reward_values:
        low, high = float(min(reward_values)), float(max(reward_values))
        padding = 0.05 * (high - low) if high > low else max(1.0, abs(low) * 0.05)
        reward_ax.set_ylim(low - padding, high + padding)
    action_ax.set_xlim(*xlim)
    reward_ax.set_xlim(*xlim)
    if history_ends:
        _format_time_axes((*axes, action_ax, reward_ax), xlim[0], xlim[1], history_ends)
    reward_ax.set_xlabel('Decision timestep, $t$')
    axes.extend([action_ax, reward_ax])
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color='#c9ced6', linewidth=0.55, alpha=0.68)
    for ax in axes[:-1]:
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
    _save_figure(fig, output_path)


def plot_simulation_and_prediction(results: list[dict], output_path: Path):
    if any(item['history'].attrs.get('method') == 'traditional_threshold' for item in results):
        return plot_actual_strategies(results, output_path)
    colors = _colors(len(results))
    past_times = [_numeric(item["history"], "time") for item in results]
    future_times = [_numeric(item["prediction"], "time") for item in results]
    all_times = [time for time in past_times + future_times if len(time)]
    x_min = min(float(time.min()) for time in all_times)
    x_max = max(float(time.max()) for time in all_times)
    history_ends = [float(past[-1]) for past, future in zip(past_times, future_times)
                    if len(future) and float(future.max()) > float(past[-1])]

    legend_slots = 2 * int(np.ceil((len(results) + 3) / 2))
    panel_header_height = max(0.50, 0.34 + 0.22 * (legend_slots // 2))
    regions = _regions(results, include_prediction=True) or [None]
    fig = plt.figure(figsize=(8.5, 6.3 + 0.45 * (panel_header_height - 0.56)
                              + 0.70 * (panel_header_height - 0.50)
                              + (2.2 + 0.5 * panel_header_height) * (len(regions) - 1)))
    grid = fig.add_gridspec(3 * len(regions) + 5, 1,
                            height_ratios=[panel_header_height, 2.0, 0.30] * len(regions)
                            + [panel_header_height, 0.88, 0.30,
                               panel_header_height, 1.35],
                            hspace=0.10)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.97, bottom=0.09)
    state_axes = []
    for i, region in enumerate(regions):
        state_header = fig.add_subplot(grid[3 * i])
        state_ax = fig.add_subplot(grid[3 * i + 1],
            sharex=state_axes[0] if state_axes else None)
        phase_footer = fig.add_subplot(grid[3 * i + 2])
        phase_footer.set_axis_off()
        state_axes.append(state_ax)

        
        for item, color, past, future in zip(results, colors, past_times, future_times):
            history = regional_state(item['history'], region)
            _fill = dict(color=color, alpha=0.12, linewidth=0.0)
            state_ax.fill_between(past, 100 * _numeric(history, 'Lower'),
                100 * _numeric(history, 'Upper'), **_fill)
            state_ax.plot(past, 100 * _numeric(history, 'd_state_num'),
                color=color, linewidth=1.25, zorder=5)
            if not item['prediction'].empty:
                prediction = regional_state(item['prediction'], region)
                state_ax.fill_between(future, 100 * _numeric(prediction, 'Lower'),
                    100 * _numeric(prediction, 'Upper'), **_fill)
                state_ax.plot(future, 100 * _numeric(prediction, 'd_state_num'),
                    color=color, linewidth=1.25, zorder=5)

        truth_states = [_truth_state(item, region) for item in results]
        if _same_series(truth_states):
            state_ax.plot(*truth_states[0], color='#4b4f55',
                linestyle='--', linewidth=0.9, zorder=6)
        else:
            for (time, values), color in zip(truth_states, colors):
                state_ax.plot(time, values, color=color, linestyle='--',
                    linewidth=0.9, alpha=0.9, zorder=6)
        state_handles = (
            [Line2D([], [], color=color, lw=1.25, label=item['label'])
             for item, color in zip(results, colors)] if len(results) > 1 else []
        )
        state_handles.extend([
            Line2D([], [], color='#4b4f55', lw=1.25, label='PA-DT estimated'),
            Line2D([], [], color='#4b4f55', linestyle='--', lw=0.9, label='Ground truth'),
            Patch(facecolor='#9ab7d8', alpha=0.35, label='95% interval'),
        ])
        title = ('a) Damage state (stiffness loss)' if region is None else
                 f'a{i + 1}) Region {region}: stiffness loss')
        _header(state_header, title, state_handles, legend_slots)
        state_ax.set_ylim(0.0, 105.0)
        state_ax.set_yticks([0, 15, 30, 45, 60, 75, 90])
        state_ax.set_yticklabels(['0%', '15%', '30%', '45%', '60%', '75%', '90%'])
        state_ax.axhline(80.0, color='#c9ced6', linestyle='--',
            linewidth=0.55, alpha=0.68, zorder=0.5)
    if history_ends and np.allclose(history_ends, history_ends[0]):
        _phase_header(phase_footer, state_axes[-1], x_min, x_max, history_ends[0])
    row = 3 * len(regions)
    action_header = fig.add_subplot(grid[row])
    action_ax = fig.add_subplot(grid[row + 1], sharex=state_axes[0])
    action_footer = fig.add_subplot(grid[row + 2])
    action_footer.set_axis_off()
    reward_header = fig.add_subplot(grid[row + 3])
    reward_ax = fig.add_subplot(grid[row + 4], sharex=state_axes[0])

    
    for item, color, past, future in zip(results, colors, past_times, future_times):
        history = item["history"]
        prediction = item["prediction"]
        action_ax.plot(past, _action_values(history), linestyle="None", marker="o",
                       markersize=2.8, color=color, zorder=4)
        if not prediction.empty:
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
    action_handles = []
    has_inspection = False
    for item, color in zip(results, colors):
        history = item['history']
        traditional = history.attrs.get('method') == 'traditional_threshold'
        label = item['label']
        if traditional and 'inspection_performed' in history:
            inspection = history['inspection_performed'].fillna(False).astype(bool)
            has_inspection = has_inspection or bool(inspection.iloc[:-1].any())
            if inspection.iloc[:-1].any():
                label = f'{label} (inspection)'
        elif not traditional:
            label = 'PA-DT decision'
        action_handles.append(Line2D([], [], color=color, marker='o', markersize=2.8,
            linestyle='None', lw=0, label=label))
    if best_series:
        action_handles.append(Line2D([], [], color="#4b4f55", linestyle="--",
                                     lw=0.9, label="PA-DT optimal action"))
    _header(action_header, "b) Control inputs", action_handles, legend_slots)
    action_ax.set_ylabel("action")
    if has_inspection:
        action_ax.set_yticks([-1, *range(len(ACTION_LABELS))])
        action_ax.set_yticklabels(["inspection", *ACTION_LABELS], fontsize=8)
        action_ax.set_ylim(-1.45, len(ACTION_LABELS) - 0.55)
    else:
        action_ax.set_yticks(np.arange(len(ACTION_LABELS)))
        action_ax.set_yticklabels(ACTION_LABELS, fontsize=8)
        action_ax.set_ylim(-0.45, len(ACTION_LABELS) - 0.55)

    
    for item, color, past, future in zip(results, colors, past_times, future_times):
        history = item["history"]
        prediction = item["prediction"]
        reward_color = color
        reward_ax.fill_between(past, _numeric(history, "reward_Lower"),
                               _numeric(history, "reward_Upper"),
                               color=reward_color, alpha=0.10, linewidth=0.0)
        reward_ax.plot(past, _numeric(history, "reward"), color=reward_color,
                       linewidth=1.05)
        if not prediction.empty:
            reward_ax.fill_between(future, _numeric(prediction, "reward_Lower"),
                                   _numeric(prediction, "reward_Upper"),
                                   color=reward_color, alpha=0.10, linewidth=0.0)
            reward_ax.plot(future, _numeric(prediction, "reward"), color=reward_color,
                           linewidth=1.05)
    _header(reward_header, "c) Reward function", [], legend_slots)
    reward_ax.set_ylabel("Reward")
    
    reward_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=4))
    reward_ax.set_xlabel("Decision timestep, $t$")

    _format_time_axes((*state_axes, action_ax, reward_ax), x_min, x_max, history_ends)
    _save_figure(fig, output_path)


def plot_sensor_dmax(results: list[dict], output_path: Path):
    
    results = [item for item in results
               if item['history'].attrs.get('method') != 'traditional_threshold']
    if not results:
        return
    colors = _colors(len(results))
    times = [_numeric(item["history"], "time") for item in results]
    x_min = min(float(time.min()) for time in times)
    x_max = max(float(time.max()) for time in times)
    header_height = max(0.90, 0.34 + 0.22 * np.ceil((len(results) + 4) / 4))
    regions = _regions(results) or [None]
    fig = plt.figure(figsize=(8.5, (3.6 + 0.45 * (header_height - 0.56)) * len(regions)))
    grid = fig.add_gridspec(2 * len(regions), 1,
        height_ratios=[header_height, 2.0] * len(regions), hspace=0.16)
    fig.subplots_adjust(left=0.14, right=0.985, top=0.97,
        bottom=0.18 / len(regions))
    axes = []
    for i, region in enumerate(regions):
        header = fig.add_subplot(grid[2 * i])
        ax = fig.add_subplot(grid[2 * i + 1], sharex=axes[0] if axes else None)
        axes.append(ax)
        for item, color, time in zip(results, colors, times):
            history = regional_state(item['history'], region)
            observations = regional_observations(item['observations'], region)
            ax.fill_between(time, 100 * _numeric(history, 'Lower'),
                100 * _numeric(history, 'Upper'), color=color, alpha=0.12, linewidth=0.0)
            ax.plot(time, 100 * _numeric(history, 'd_state_num'), color=color, linewidth=1.25)
            ax.scatter(_numeric(observations, 'time'),
                100 * _numeric(observations, 'estimated_stiffness_loss'), s=13,
                color=color, edgecolor='white', linewidth=0.25, alpha=0.75, zorder=5)

        truth_states = [_truth_state(item, region) for item in results]
        if _same_series(truth_states):
            ax.plot(*truth_states[0], color='#34383f', linestyle='--', linewidth=0.9)
        else:
            for (time, values), color in zip(truth_states, colors):
                ax.plot(time, values, color=color, linestyle='--', linewidth=0.9, alpha=0.9)
        handles = (
            [Line2D([], [], color=color, lw=1.25, label=item['label'])
             for item, color in zip(results, colors)] if len(results) > 1 else []
        )
        handles.extend([
            Line2D([], [], color='#34383f', linestyle='--', lw=0.9, label='Ground truth'),
            Line2D([], [], color='#4b4f55', lw=1.25, label='PA-DT estimated'),
            Line2D([], [], color='#33373d', marker='o', markersize=3, lw=0, label='Observed'),
            Patch(facecolor='#9ab7d8', alpha=0.35, label='95% interval'),
        ])
        title = ('Stiffness loss observations' if region is None else
                 f'Region {region}: stiffness loss observations')
        _header(header, title, handles)
        ax.set_ylabel('Stiffness loss')
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.set_xlim(x_min, x_max)
        ax.axvline(x_max, color='#8c929b', linewidth=0.55, zorder=1)
        values = []
        for item in results:
            state = regional_state(item['history'], region)
            observed = regional_observations(item['observations'], region)
            for column in ('d_state_num', 'Lower', 'Upper', 'p_state_num'):
                values.extend(100 * _numeric(state, column))
            values.extend(100 * _numeric(observed, 'estimated_stiffness_loss'))
        ax.set_ylim(bottom=min(0.0, float(np.nanmin(values)) * 1.08),
                    top=max(1.0, float(np.nanmax(values)) * 1.08))
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.tick_params(axis='x', labelbottom=i == len(regions) - 1)
    axes[-1].set_xlabel('Decision timestep, $t$')
    _save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", type=Path,
                        help=" simulation_results")
    args = parser.parse_args()
    _configure_style()
    location = _result_folder(args.folder)
    folders = ([location] if any(location.glob('*.xlsx')) else
               sorted(path for path in location.glob('health_*') if path.is_dir()) or [location])
    for folder in folders:
        workbooks = _find_workbooks(folder)
        results = [_read_result(path) for path in workbooks]
        plot_simulation_and_prediction(results, folder / 'simulation_and_prediction_duibi.svg')
        plot_sensor_dmax(results, folder / 'sensor_dmax_duibi.svg')
        print(f'Found {len(results)} simulation workbooks in {folder}:')
        for item in results:
            print(f"  - {item['path'].name}")
        print(f"Damage regions: {_regions(results) or 'healthy only'}")
        print(f"Saved: {folder / 'simulation_and_prediction_duibi.svg'}")


if __name__ == "__main__":
    main()
