from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Plot:
    

    def __init__(self, excel_file, path, wall_thickness, denorm_stats=None):
        data = pd.read_excel(excel_file, sheet_name=None)
        self.history = data['Simulation'].set_index('time')
        self.prediction = data['Prediction'].set_index('time')
        self.observations = data['Simulation_Observations']
        self.wall_thickness = wall_thickness
        self.denorm_stats = denorm_stats or {}
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        self.actions = ['Do_nothing', 'Pigging', 'Adding_inhibitor', 'Replacement']

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
        fig.savefig(self.path / f'{filename}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)

    def _format_state_axis(self, ax, ylabel='State value (remaining wall ratio)'):
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

    def _draw_action_values(self, ax, time, values, color, linestyle, label,
                            marker=False):
        ax.step(time, values, where='post', color=color, linestyle=linestyle,
                linewidth=1.7, label=label)
        if marker:
            ax.plot(time, values, color=color, linestyle='None', marker='o',
                    markersize=3, alpha=.9)
            
    def _format_reward_axis(self, ax):
        ax.set_ylabel('Reward $R_t$')
        ax.grid(True, axis='y', color='#d9d9d9', linewidth=.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        
        
    

    def plot_all_together(self, figsize=(18, 10)):
        hist, pred = self.history, self.prediction
        past_time = hist.index.to_numpy()
        future_time = pred.index.to_numpy()
        tc = past_time[-1]

        truth = hist['p_state_num'].to_numpy(dtype=float)
        estimated = hist['d_state_num'].to_numpy(dtype=float)
        predicted = pred['d_state_num'].to_numpy(dtype=float)
        
        est_low = hist['Lower'].to_numpy(dtype=float)
        est_high = hist['Upper'].to_numpy(dtype=float)
        pred_low = pred['Lower'].to_numpy(dtype=float)
        pred_high = pred['Upper'].to_numpy(dtype=float)

        
        r_est = hist['reward'].to_numpy(dtype=float)
        r_pred = pred['reward'].to_numpy(dtype=float)
        r_est_low = hist['reward_Lower'].to_numpy(dtype=float)
        r_est_high = hist['reward_Upper'].to_numpy(dtype=float)
        r_pred_low = pred['reward_Lower'].to_numpy(dtype=float)
        r_pred_high = pred['reward_Upper'].to_numpy(dtype=float)


        fig, (state_ax, action_ax, reward_ax) = plt.subplots(
            3, 1, sharex=True, figsize=figsize,
            gridspec_kw={'height_ratios': [2, 1, 1.5]}
        )

        
        state_ax.fill_between(past_time, est_low, est_high,
                              color=self.colors['estimated_band'], alpha=.2,
                              label='Estimated 95% interval')
        state_ax.fill_between(future_time, pred_low, pred_high,
                              color=self.colors['predicted_band'], alpha=.2,
                              label='Predicted 95% interval')
        state_ax.plot(past_time, truth, color=self.colors['truth'], linewidth=1.0,
                      label='Ground truth')
        state_ax.plot(past_time, estimated, color=self.colors['estimated'],
                      linewidth=1.6, marker='o', markersize=3,
                      label='Estimated (mean)')
        state_ax.plot(future_time, predicted, color=self.colors['predicted'],
                      linestyle='--', linewidth=1.7, label='Predicted (mean)')

        state_ax.axvline(tc, color='#777777', linewidth=.8, alpha=.6)
        state_ax.text(tc, 1.01, '$t_c$', transform=state_ax.get_xaxis_transform(),
                      ha='center', va='bottom')
        self._format_state_axis(state_ax)
        state_ax.legend(loc='lower left', bbox_to_anchor=(0, 1.08),
                        frameon=False, ncol=3)

        
        self._draw_action_values(
            action_ax, past_time,
            hist['current_action_value'].to_numpy(dtype=float),
            self.colors['estimated'], '-', 'DT action (estimated)',
            marker=True
        )
        self._draw_action_values(
            action_ax, future_time,
            pred['current_action_value'].to_numpy(dtype=float),
            self.colors['predicted'], '--', 'DT action (predicted)',
            marker=False
        )
        if 'best_action_value' in hist.columns:
            self._draw_action_values(
                action_ax, past_time,
                hist['best_action_value'].to_numpy(dtype=float),
                self.colors['best_action'], ':', 'Best action (ground truth)',
                marker=False
            )

        action_ax.axvline(tc, color='#777777', linewidth=.8, alpha=.6)
        self._format_action_axis(action_ax)
        action_ax.set_xlabel('Time')
        action_ax.set_xlim(past_time[0], future_time[-1])
        action_ax.legend(loc='upper left', frameon=False, ncol=3)
        
        
        
        reward_ax.fill_between(past_time, r_est_low, r_est_high,
                            color=self.colors['estimated_band'], alpha=.2,
                            label='Estimated 95% interval')
        reward_ax.fill_between(future_time, r_pred_low, r_pred_high,
                            color=self.colors['predicted_band'], alpha=.2,
                            label='Predicted 95% interval')
        reward_ax.plot(past_time, r_est, color=self.colors['estimated'],
                    linewidth=1.6, marker='o', markersize=3,
                    label='Estimated reward (mean)')
        reward_ax.plot(future_time, r_pred, color=self.colors['predicted'],
                    linestyle='--', linewidth=1.7,
                    label='Predicted reward (mean)')
        reward_ax.axvline(tc, color='#777777', linewidth=.8, alpha=.6)
        self._format_reward_axis(reward_ax)
        reward_ax.set_xlabel('Time')
        reward_ax.set_xlim(past_time[0], future_time[-1])
        reward_ax.legend(loc='lower left', bbox_to_anchor=(0, 1.02),
                        frameon=False, ncol=3)
        
        
        self._finish(fig, 'simulation_and_prediction')
        return fig

    

    def plot_sensor_dmax(self, figsize=(12, 4)):
        obs = self.observations
        hist = self.history

        obs_time = obs['time'].to_numpy()
        obs_dmax = obs['dmax'].to_numpy(dtype=float)

        time = hist.index.to_numpy()
        d_state = hist['d_state_num'].to_numpy(dtype=float)
        lower_ratio = hist['Lower'].to_numpy(dtype=float)
        upper_ratio = hist['Upper'].to_numpy(dtype=float)

        w = self.wall_thickness
        est_dmax = w * (1.0 - d_state)
        est_dmax_low = w * (1.0 - upper_ratio)
        est_dmax_high = w * (1.0 - lower_ratio)

        fig, ax = plt.subplots(figsize=figsize)
        ax.fill_between(time, est_dmax_low, est_dmax_high,
                        color=self.colors['estimated_band'], alpha=.2,
                        label='Estimated 95% interval')
        ax.scatter(obs_time, obs_dmax,
                   s=16, facecolors='none',
                   edgecolors=self.colors['observation'],
                   linewidths=.8, alpha=.7, zorder=4, label='Observations')
        ax.plot(time, est_dmax, color=self.colors['estimated'], linewidth=1.6,
                marker='o', markersize=3, label='Estimated (mean)')

        ax.set_ylabel('dmax (mm)')
        ax.set_xlabel('Time')
        ax.set_title('Sensor observation: dmax')
        ax.grid(True, axis='y', color='#d9d9d9', linewidth=.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='best', frameon=False)
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
        ax.scatter(obs['time'], values,
                   s=16, facecolors='none',
                   edgecolors=self.colors['observation'],
                   linewidths=.8, alpha=.7, zorder=4)
        ylabel = feature_name if denormalized else f'{feature_name} (normalized)'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time')
        ax.set_title(f'Sensor observation: {feature_name}')
        ax.grid(True, axis='y', color='#d9d9d9', linewidth=.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        self._finish(fig, f'sensor_{feature_name}')
        return fig