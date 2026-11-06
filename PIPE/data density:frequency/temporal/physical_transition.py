

from functools import lru_cache
import numpy as np


class PhysicalTransition:
    def __init__(self, asset_factory, X_regressors, wall_thickness,
                 corrosion_states, actions_list):
        self.asset_factory = asset_factory
        self.X_regressors = X_regressors
        self.wall_thickness = float(wall_thickness)
        self.values = np.asarray(corrosion_states, dtype=float)
        self.actions_list = tuple(str(action) for action in actions_list)
        nodes, weights = np.polynomial.hermite.hermgauss(16)
        self.nodes = np.sqrt(2.0) * nodes
        self.weights = weights / np.sqrt(np.pi)
        self.normal_coordinate = 0.0

    def normal(self, mean, std):
        return mean + std * self.normal_coordinate

    @lru_cache(maxsize=None)
    def matrices(self, age, df_value):
        
        n_states = len(self.values)
        transitions = np.zeros((len(self.actions_list), n_states, n_states))
        asset = self.asset_factory('power_function', self.X_regressors,
                                   self.wall_thickness)
        
        ascending = self.values[::-1]
        for action_index, action in enumerate(self.actions_list):
            stochastic = action in ('Adding_inhibitor', 'Pigging')
            nodes = self.nodes if stochastic else (0.0,)
            weights = self.weights if stochastic else (1.0,)
            for state_index, value in enumerate(self.values):
                for node, weight in zip(nodes, weights):
                    asset.degradation_steps = int(age)
                    asset.dmax = self.wall_thickness * (1.0 - value)
                    asset.pre_dmax = asset.dmax
                    asset.df = float(df_value)
                    self.normal_coordinate = float(node)
                    asset.update_physical_state(action, rng=self)

                    if action == 'Replacement':
                        transitions[action_index, 0, state_index] += weight
                        continue

                    health = float(asset.ratio)
                    upper = int(np.searchsorted(ascending, health))
                    if upper == 0:
                        transitions[action_index, -1, state_index] += weight
                    elif upper == n_states:
                        transitions[action_index, 0, state_index] += weight
                    else:
                        low = ascending[upper - 1]
                        high = ascending[upper]
                        fraction = (health - low) / (high - low)
                        transitions[action_index, n_states - 1 - upper, state_index] += weight * fraction
                        transitions[action_index, n_states - upper, state_index] += weight * (1.0 - fraction)

        transitions /= transitions.sum(axis=1, keepdims=True)
        return transitions
