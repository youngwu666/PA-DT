from pathlib import Path

import matplotlib.pyplot as plt
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
    result = {}
    for key, sheet in (('history', 'Simulation'), ('prediction', 'Prediction'),
                       ('observations', 'Simulation_Observations')):
        frame = data[sheet].copy()
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
    if frame.empty:
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


class Plot:
    

    def __init__(self, excel_file, path, actions, denorm_stats=None):

        
        data = read_plot_results(excel_file)

        self.history = data['history'].set_index('time')
        self.prediction = data['prediction'].set_index('time')
        self.observations = data['observations']
        self.regions = damage_regions(self.history, self.observations)

        
        self.denorm_stats = denorm_stats or {}
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.actions = list(actions)

        
        self.colors = {
            'truth': 'black',
            'estimated': '#1769aa',
            'predicted': '#d62728',
            'estimated_band': '#9ecae1',
            'predicted_band': '#f4a6a6',
            'observation': '#2a9d6f',
            'best_action': '#888888',
        }

        
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 12,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'axes.linewidth': 0.8,
        })

    

    def set_denorm_stats(self, denorm_stats):

        self.denorm_stats = denorm_stats or {}

    def denormalize(self, feature_name, values):

        values = np.asarray(values, dtype=float)

        if feature_name not in self.denorm_stats:
            return values

        vmin, vmax = self.denorm_stats[feature_name]

        return (values + 1.0) / 2.0 * (vmax - vmin) + vmin

    def is_denormalizable(self, feature_name):

        return feature_name in self.denorm_stats

    

    def _finish(self, fig, filename):

        fig.tight_layout()

        fig.savefig(
            self.path / f'{filename}.png',
            dpi=180,
            bbox_inches='tight'
        )

        plt.close(fig)

    def _format_state_axis(self, ax, ylabel='State value (stiffness-loss ratio)'):

        ax.set_ylabel(ylabel)

        ax.grid(True, axis='y', color='#d9d9d9', linewidth=.6)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _format_action_axis(self, ax):

        ax.set_ylabel('Action')
        ax.set_yticks(np.arange(len(self.actions)))
        ax.set_yticklabels([a.replace('_', ' ') for a in self.actions])

        ax.grid(True, axis='y', color='#d9d9d9', linewidth=.6)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _draw_action_values(self,
                           ax,
                           time,
                           values,
                           color,
                           linestyle,
                           label,
                           marker=False):

        ax.step(
            time,
            values,
            where='post',
            color=color,
            linestyle=linestyle,
            linewidth=1.7,
            label=label
        )

        if marker:
            ax.plot(
                time,
                values,
                color=color,
                linestyle='None',
                marker='o',
                markersize=3,
                alpha=.9
            )

    def _format_reward_axis(self, ax):

        ax.set_ylabel('Reward $R_t$')

        ax.grid(True, axis='y', color='#d9d9d9', linewidth=.6)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


    

    def plot_all_together(self, figsize=(18, 10)):

        
        hist, pred = self.history, self.prediction
        past_time = hist.index.to_numpy()
        future_time = pred.index.to_numpy(dtype=float)
        future_end = future_time[-1] if len(future_time) else past_time[-1]
        tc = past_time[-1]

        
        r_est = hist['reward'].to_numpy(dtype=float)
        r_pred = pred['reward'].to_numpy(dtype=float)
        r_est_low = hist['reward_Lower'].to_numpy(dtype=float)
        r_est_high = hist['reward_Upper'].to_numpy(dtype=float)
        r_pred_low = pred['reward_Lower'].to_numpy(dtype=float)
        r_pred_high = pred['reward_Upper'].to_numpy(dtype=float)


        
        regions = self.regions or [None]
        fig, axes = plt.subplots(
            len(regions) + 2,
            1,
            sharex=True,
            squeeze=False,
            figsize=(figsize[0], figsize[1] + 2.5 * (len(regions) - 1)),
            gridspec_kw={'height_ratios': [2] * len(regions) + [1, 1.5]}
        )
        state_axes = axes[:-2, 0]
        action_ax, reward_ax = axes[-2:, 0]

        
        for i, (region, state_ax) in enumerate(zip(regions, state_axes)):
            state_hist = regional_state(hist, region)
            state_pred = regional_state(pred, region)
            state_ax.fill_between(past_time, state_hist['Lower'], state_hist['Upper'],
                color=self.colors['estimated_band'], alpha=.2,
                label='Estimated 95% interval')
            state_ax.fill_between(future_time, state_pred['Lower'], state_pred['Upper'],
                color=self.colors['predicted_band'], alpha=.2,
                label='Predicted 95% interval')
            state_ax.plot(past_time, state_hist['p_state_num'],
                color=self.colors['truth'], linewidth=1.0, label='Ground truth')
            state_ax.plot(past_time, state_hist['d_state_num'],
                color=self.colors['estimated'], linewidth=1.6, marker='o',
                markersize=3, label='Estimated (mean)')
            state_ax.plot(future_time, state_pred['d_state_num'],
                color=self.colors['predicted'], linestyle='--', linewidth=1.7,
                label='Predicted (mean)')
            state_ax.axvline(tc, color='#777777', linewidth=.8, alpha=.6)
            state_ax.text(tc, 1.01, '$t_c$', transform=state_ax.get_xaxis_transform(),
                ha='center', va='bottom')
            ylabel = ('State value (stiffness-loss ratio)' if region is None else
                      f'Region {region}\nStiffness-loss ratio')
            self._format_state_axis(state_ax, ylabel)
            if i == 0:
                state_ax.legend(loc='lower left', bbox_to_anchor=(0, 1.08),
                    frameon=False, ncol=3)

        
        self._draw_action_values(
            action_ax,
            past_time,
            hist['current_action_value'].to_numpy(dtype=float),
            self.colors['estimated'],
            '-',
            'DT action (estimated)',
            marker=True
        )
        self._draw_action_values(
            action_ax,
            future_time,
            pred['current_action_value'].to_numpy(dtype=float),
            self.colors['predicted'],
            '--',
            'DT action (predicted)',
            marker=False
        )
        if 'best_action_value' in hist.columns:
            self._draw_action_values(
                action_ax,
                past_time,
                hist['best_action_value'].to_numpy(dtype=float),
                self.colors['best_action'],
                ':',
                'Best action (ground truth)',
                marker=False
            )

        action_ax.axvline(tc, color='#777777', linewidth=.8, alpha=.6)
        self._format_action_axis(action_ax)
        action_ax.set_xlabel('Time')
        action_ax.set_xlim(past_time[0], future_end)
        action_ax.legend(loc='upper left', frameon=False, ncol=3)


        
        reward_ax.fill_between(
            past_time,
            r_est_low,
            r_est_high,
            color=self.colors['estimated_band'],
            alpha=.2,
            label='Estimated 95% interval'
        )

        reward_ax.fill_between(
            future_time,
            r_pred_low,
            r_pred_high,
            color=self.colors['predicted_band'],
            alpha=.2,
            label='Predicted 95% interval'
        )

        reward_ax.plot(
            past_time,
            r_est,
            color=self.colors['estimated'],
            linewidth=1.6,
            marker='o',
            markersize=3,
            label='Estimated reward (mean)'
        )

        reward_ax.plot(
            future_time,
            r_pred,
            color=self.colors['predicted'],
            linestyle='--',
            linewidth=1.7,
            label='Predicted reward (mean)'
        )

        reward_ax.axvline(tc, color='#777777', linewidth=.8, alpha=.6)
        self._format_reward_axis(reward_ax)
        reward_ax.set_xlabel('Time')
        reward_ax.set_xlim(past_time[0], future_end)
        reward_ax.legend(
            loc='lower left',
            bbox_to_anchor=(0, 1.02),
            frameon=False,
            ncol=3
        )


        self._finish(fig, 'simulation_and_prediction')
        return fig

    

    def plot_sensor_dmax(self, figsize=(12, 4)):

        regions = self.regions or [None]
        time = self.history.index.to_numpy()
        fig, axes = plt.subplots(len(regions), 1, sharex=True, squeeze=False,
            figsize=(figsize[0], figsize[1] * len(regions)))
        for region, ax in zip(regions, axes[:, 0]):
            hist = regional_state(self.history, region)
            obs = self.observations
            observed_loss = obs['true_stiffness_loss'].to_numpy(dtype=float)
            if region is not None:
                observed_loss = np.where(obs['true_region'] == region, observed_loss, 0.0)
            ax.fill_between(time, hist['Lower'], hist['Upper'],
                color=self.colors['estimated_band'], alpha=.2,
                label='Estimated 95% interval')
            ax.scatter(obs['time'], observed_loss, s=16,
                facecolors='none', edgecolors=self.colors['observation'],
                linewidths=.8, alpha=.7, zorder=4, label='Observations')
            ax.plot(time, hist['d_state_num'], color=self.colors['estimated'],
                linewidth=1.6, marker='o', markersize=3, label='Estimated (mean)')
            ax.set_ylabel('Stiffness-loss ratio')
            title = 'Sensor observation: stiffness loss'
            if region is not None:
                title = f'Region {region}: stiffness loss observations'
            ax.set_title(title)
            ax.grid(True, axis='y', color='#d9d9d9', linewidth=.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.legend(loc='best', frameon=False)
        axes[-1, 0].set_xlabel('Time')

        self._finish(fig, 'sensor_dmax')

        return fig

    

    def plot_sensor_environment(self, feature_name, figsize=(12, 4)):

        obs = self.observations

        if obs.empty:
            raise ValueError('Simulation_Observations sheet ')

        if feature_name not in obs.columns:
            raise KeyError(f'{feature_name} : {list(obs.columns)}')

        values = self.denormalize(feature_name, obs[feature_name].to_numpy(dtype=float))
        denormalized = self.is_denormalizable(feature_name)

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(
            obs['time'],
            values,
            s=16,
            facecolors='none',
            edgecolors=self.colors['observation'],
            linewidths=.8,
            alpha=.7,
            zorder=4
        )
        ylabel = feature_name if denormalized else f'{feature_name} (normalized)'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time')
        ax.set_title(f'Sensor observation: {feature_name}')
        ax.grid(True, axis='y', color='#d9d9d9', linewidth=.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self._finish(fig, f'sensor_{feature_name}')
        return fig
