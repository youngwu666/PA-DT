




from __future__ import annotations

from copy import copy
from hashlib import sha256
import inspect
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr


DAMAGE_EDGES = np.array([0.30, 0.35, 0.45, 0.55, 0.65, 0.75, 0.80])


def state_index(states, state, *, observation=False):
    
    region, damage = int(state[0]), float(state[1])
    if region == 0:
        return 0
    if not 1 <= region <= 6 or not np.isfinite(damage):
        raise ValueError('invalid tunnel state')
    if not observation and damage > 0.8:
        return None
    level = int(np.searchsorted(DAMAGE_EDGES[1:-1], damage, side='right'))
    return 1 + (region - 1) * 6 + level


def _major_rehab_probabilities(damage, edges, mean, std):
    





    damage = float(damage)
    edges = np.asarray(edges, dtype=float)
    probabilities = np.zeros(len(edges))
    threshold = damage - 0.30
    probabilities[0] = ndtr(
        (mean - threshold) / std
    )

    for target, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        amount_lower = max(0.0, damage - upper)
        amount_upper = damage - lower
        if amount_upper > amount_lower:
            probabilities[target + 1] += (
                ndtr((amount_upper - mean) / std)
                - ndtr((amount_lower - mean) / std)
            )

    
    source_level = int(np.searchsorted(edges[1:-1], damage, side='right'))
    probabilities[source_level + 1] += ndtr(-mean / std)

    probabilities = np.maximum(probabilities, 0.0)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError('invalid stochastic rehabilitation probabilities')
    return probabilities / total


def update_posterior(prior, estimated_index, action_index, model, likelihood):
    
    prediction = model.propagate(np.asarray(prior, dtype=float), action_index)
    posterior = prediction * likelihood[model.grid.coarse, int(estimated_index)]
    total = float(posterior.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError('observation has no probability under the survival model')
    return posterior / total


class DamageGrid:
    
    def __init__(self, step=.01):
        self.step = float(step)
        self.n_levels = round(.5 / self.step)
        self.edges = np.round(np.linspace(.30, .80, self.n_levels + 1), 12)
        if not np.isclose(self.n_levels * self.step, .5) or any(
                not np.any(np.isclose(self.edges, edge, atol=1e-12, rtol=0)) for edge in DAMAGE_EDGES):
            raise ValueError('internal damage step must divide .05 and .50 exactly')
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        self.n_states = 1 + 6 * self.n_levels
        self.values = np.r_[0., np.tile(self.centers, 6)]
        self.states = np.array(['0;0.0'] + [f'{region};{value:.6f}'
                               for region in range(1, 7) for value in self.centers])
        coarse_level = np.searchsorted(DAMAGE_EDGES[1:-1], self.centers, side='right')
        self.coarse = np.r_[0, np.concatenate([1 + region * 6 + coarse_level for region in range(6)])]
        self.coarse_sizes = np.bincount(self.coarse, minlength=37)

    def index(self, state):
        region, damage = int(state[0]), float(state[1])
        if region == 0:
            return 0
        if not 1 <= region <= 6 or not np.isfinite(damage) or damage < .30:
            raise ValueError('invalid physical state for the internal grid')
        if damage > .80:
            return None
        level = np.searchsorted(self.edges[1:-1], damage, side='right')
        return int(1 + (region - 1) * self.n_levels + level)

    def aggregate(self, belief):
        return np.bincount(self.coarse, weights=belief, minlength=37)

    def lift(self, coarse_belief, reference=None):
        
        conditional = 1. / self.coarse_sizes[self.coarse]
        if reference is not None:
            totals = self.aggregate(reference)[self.coarse]
            conditional = np.divide(reference, totals, out=conditional, where=totals > 0)
        return np.asarray(coarse_belief)[self.coarse] * conditional


class _PhysicalLaw:
    
    def __init__(self, gate=0):
        self.gate = gate
        self.calls = 0
        self.cardinality = None
        self.normal_parameters = None

    def randrange(self, stop):
        self.calls += 1
        if self.calls == 1:
            self.cardinality = stop
            return self.gate
        return 0

    def uniform(self, low, high):
        return (low + high) / 2

    def normal(self, mean, std):
        self.normal_parameters = mean, std
        return mean


class PhysicalTransitionModel:
    





    def __init__(self, context, grid):
        self.context, self.grid = context, grid
        n, size = grid.n_levels, grid.n_states
        n_actions = len(context.ACTIONS)
        self._matrices = np.zeros((n_actions, size, size), dtype=float)
        self.failure = np.zeros((n_actions, size), dtype=float)
        self.kernel = np.zeros((n_actions, n, n), dtype=float)
        self.initiation = np.zeros(n_actions)
        self.initial_density = np.maximum(0, np.minimum(grid.edges[1:], .35) -
                                          np.maximum(grid.edges[:-1], .30)) / .05
        nodes, weights = np.polynomial.legendre.leggauss(32)
        weights /= 2
        for action_id, action in enumerate(context.ACTIONS):
            if action == 'major_rehab':
                self._matrices[action_id, 0, 0] = 1
                for region in range(6):
                    for level, damage in enumerate(grid.centers):
                        source = 1 + region * n + level
                        probabilities = _major_rehab_probabilities(
                            damage, grid.edges, context.physical_repair_mean,
                            context.physical_repair_std)
                        self._matrices[action_id, 0, source] = probabilities[0]
                        for target, probability in enumerate(probabilities[1:]):
                            self._matrices[
                                action_id, 1 + region * n + target, source
                            ] = probability
                continue
            probe = _PhysicalLaw()
            self._asset([0, .30]).update_physical_state(str(action), probe, probe)
            for gate in range(probe.cardinality):
                event = _PhysicalLaw(gate)
                asset = self._asset([0, .30])
                asset.update_physical_state(str(action), event, event)
                self.initiation[action_id] += int(asset.state[0] != 0) / probe.cardinality
            probe = _PhysicalLaw()
            self._asset([1, .40]).update_physical_state(str(action), probe, probe)
            mean, std = probe.normal_parameters
            for level, (lower, upper) in enumerate(zip(grid.edges[:-1], grid.edges[1:])):
                damage = (lower + upper) / 2 + (upper - lower) / 2 * nodes
                distances = grid.edges[:, None] - damage
                cdf = np.where(distances >= 0, ndtr((distances - mean) / std), 0.)
                self.kernel[action_id, :, level] = np.maximum(0, np.diff(cdf, axis=0) @ weights)
            self._matrices[action_id, 0, 0] = 1 - self.initiation[action_id]
            for region in range(6):
                block = slice(1 + region * n, 1 + (region + 1) * n)
                self._matrices[action_id, block, block] = self.kernel[action_id]
                self._matrices[action_id, block, 0] = self.initiation[action_id] * self.initial_density / 6
            self.failure[action_id] = np.maximum(0, 1 - self._matrices[action_id].sum(axis=0))
        if not np.allclose(self._matrices.sum(axis=1) + self.failure, 1, atol=1e-12):
            raise ValueError('internal transition and terminal mass do not sum to one')

    def _asset(self, state):
        asset = self.context.PhysicalAsset(str(self.context.ROOT / 'data'), 'stochastic')
        asset.n_classes = 7
        asset.degradation_steps = 0
        asset.state = np.asarray(state, dtype=float)
        return asset

    def matrices(self):
        return self._matrices

    def propagate(self, belief, action):
        
        following = np.zeros_like(belief, dtype=float)
        if self.context.ACTIONS[action] == 'major_rehab':
            return self._matrices[action] @ belief
        else:
            following[0] = (1 - self.initiation[action]) * belief[0]
            blocks = belief[1:].reshape(6, self.grid.n_levels)
            following[1:] = (blocks @ self.kernel[action].T +
                belief[0] * self.initiation[action] * self.initial_density / 6).ravel()
        return following

    def coarse_matrices(self, belief=None):
        
        grid = self.grid
        conditional = grid.lift(np.ones(37), reference=belief)
        lifting = np.zeros((grid.n_states, 37))
        lifting[np.arange(grid.n_states), grid.coarse] = conditional
        transitions = np.zeros((len(self.context.ACTIONS), 37, 37))
        for action in range(len(self.context.ACTIONS)):
            following = self._matrices[action] @ lifting
            np.add.at(transitions[action], grid.coarse, following)
        failure = self.failure @ lifting
        return transitions, failure

class DigitalTransitionModel:
    





    def __init__(self, context, grid):
        self.grid = grid
        n_levels = len(DAMAGE_EDGES) - 1
        n_regions = (len(context.STATES) - 1) // n_levels
        n_actions = len(context.ACTIONS)
        self._coarse = np.zeros((n_actions, len(context.STATES), len(context.STATES)))
        steps = tuple(context.digital_major_steps)
        probabilities = np.asarray(context.digital_major_probabilities, dtype=float)
        if (len(steps) != len(probabilities) or any(int(s) != s or s < 0 for s in steps)
                or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.isclose(probabilities.sum(), 1.0)):
            raise ValueError('invalid digital major-rehabilitation probabilities')
        for action_id, action in enumerate(context.ACTIONS):
            if action == 'major_rehab':
                self._coarse[action_id, 0, 0] = 1.0
                for region in range(n_regions):
                    for level in range(1, n_levels + 1):
                        source = 1 + region * n_levels + level - 1
                        for improvement, probability in zip(steps, probabilities):
                            remaining = max(0, level - improvement)
                            target = 0 if remaining == 0 else region * n_levels + remaining
                            self._coarse[action_id, target, source] += probability
                continue
            probability = (context.digital_dn_probability if action == 'do_nothing'
                           else context.digital_re_probability)
            if not np.isfinite(probability) or not 0 <= probability <= 1 / n_regions:
                raise ValueError('invalid digital deterioration probability')
            self._coarse[action_id, 0, 0] = 1 - n_regions * probability
            for region in range(n_regions):
                first = 1 + region * n_levels
                self._coarse[action_id, first, 0] = probability
                for level in range(n_levels):
                    source = first + level
                    if level == n_levels - 1:
                        self._coarse[action_id, source, source] = 1.0
                    else:
                        self._coarse[action_id, source, source] = 1 - probability
                        self._coarse[action_id, source + 1, source] = probability

        self._matrices = np.zeros((n_actions, grid.n_states, grid.n_states))
        self.failure = np.zeros((n_actions, grid.n_states))
        for action in range(n_actions):
            for source, coarse_source in enumerate(grid.coarse):
                for target in np.flatnonzero(self._coarse[action, :, coarse_source]):
                    probability = self._coarse[action, target, coarse_source]
                    if target == coarse_source:
                        self._matrices[action, source, source] += probability
                    else:
                        indexes = np.flatnonzero(grid.coarse == target)
                        self._matrices[action, indexes, source] += probability / len(indexes)
        if not np.allclose(self._coarse.sum(axis=1), 1, atol=1e-12):
            raise ValueError('digital transition probabilities do not sum to one')
        self._stay = np.diagonal(self._coarse, axis1=1, axis2=2).copy()
        self._moving = self._coarse.copy()
        indexes = np.arange(len(context.STATES))
        self._moving[:, indexes, indexes] = 0.0

    def matrices(self):
        return self._matrices

    def propagate(self, belief, action):
        
        coarse_belief = self.grid.aggregate(belief)
        incoming = self._moving[action] @ coarse_belief
        return self._stay[action, self.grid.coarse] * belief + self.grid.lift(incoming)

    def coarse_matrices(self, belief=None):
        
        return self._coarse.copy(), np.zeros((len(self._coarse), self._coarse.shape[1]))


class QLearningPlanner:
    def __init__(self, context, gamma=.90, learning_rate=.10, damage_step=.01,
                 health_weight=1.0):
        self.context = context
        self.states = context.STATES
        self.actions = tuple(str(a) for a in context.ACTIONS)
        self.grid = DamageGrid(damage_step)
        self.n_states, self.n_actions = self.grid.n_states, len(self.actions)
        self.gamma, self.learning_rate = float(gamma), float(learning_rate)
        self.q_values = np.zeros((self.n_states, 2, self.n_actions), dtype=float)
        self.visits = np.zeros(self.q_values.shape, dtype=np.uint32)
        
        self.reward_source = context.Planner(health_weight)
        self.coarse_reward_table = np.array([[self.compute_reward(s, a) for a in self.actions]
                                             for s in self.states])
        
        self.reward_table = self.coarse_reward_table[self.grid.coarse]
        self.transition_model = DigitalTransitionModel(context, self.grid)
        self.physical_transition_model = PhysicalTransitionModel(context, self.grid)
        self.likelihood = None
        self.final_episode = 0
        self.stop_reason = 'not_started'
        self.training_history, self.episode_history = [], []

    def compute_reward(self, state, action):
        return self.reward_source.compute_reward(str(state), str(action))

    def action_values(self, observation):
        _, restricted, belief = observation
        belief = np.asarray(belief, dtype=float)
        if (belief.shape != (self.n_states,) or not np.isfinite(belief).all()
                or np.any(belief < 0) or belief.sum() <= 0):
            raise ValueError('invalid full-state posterior')
        return (belief / belief.sum()) @ self.q_values[:, int(bool(restricted))]

    def coarse_q_values(self, restricted=False):
        
        values = np.zeros((37, self.n_actions), dtype=float)
        np.add.at(values, self.grid.coarse, self.q_values[:, int(bool(restricted))])
        return values / self.grid.coarse_sizes[:, None]

    def select_action(self, observation, epsilon=0, rng=None):
        if rng is not None and rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self.action_values(observation)))

    def update(self, state, action, reward, next_state, terminal=False):
        index = (*state, int(action))
        
        
        n = self.grid.n_levels
        related = np.array([0]) if state[0] == 0 else np.arange(1 + (state[0] - 1) % n, self.n_states, n)
        count = int(self.visits[related, :, int(action)].sum())
        
        
        alpha = min(self.learning_rate, (count + 1) ** (-.6))
        target = float(reward)
        if not terminal:
            target += self.gamma * float(np.max(self.q_values[next_state]))
        value = self.q_values[index] + alpha * (target - self.q_values[index])
        
        self.q_values[related, :, int(action)] = value
        self.visits[index] += 1

    def parameter_visits(self):
        
        counts = self.visits.sum(axis=1)
        result = np.vstack([counts[0], counts[1:].reshape(
            6, self.grid.n_levels, self.n_actions).sum(axis=0)])
        return result

    def reward_distribution(self, belief, action):
        
        action_id = self.actions.index(str(action))
        model = self.transition_model
        probabilities = np.r_[model.propagate(belief, action_id),
                               model.failure[action_id] @ belief]
        rewards = np.r_[self.reward_table[:, action_id],
                        self.compute_reward('1;0.8', action)]
        return probabilities, rewards

    def reward_stats(self, belief, action):
        probabilities, rewards = self.reward_distribution(belief, action)
        order = np.argsort(rewards, kind='stable')
        cdf = np.cumsum(probabilities[order])
        cdf /= cdf[-1]
        lower, upper = np.searchsorted(cdf, [.025, .975], side='left')
        return (float(probabilities @ rewards), float(rewards[order[lower]]),
                float(rewards[order[upper]]))

    def bellman_residual(self):
        
        model = self.physical_transition_model
        transitions = model.matrices()
        reward = (transitions * self.reward_table.T[:, :, None]).sum(axis=1)
        reward += model.failure * np.array([
            self.compute_reward('1;0.8', a) for a in self.actions])[:, None]
        values = self.q_values[:, 0].max(axis=1)
        targets = reward + self.gamma * np.einsum('aji,j->ai', transitions, values)
        return np.abs(targets.T - self.q_values[:, 0])

    def update_policy(self):
        
        coarse_q = self.coarse_q_values(False)
        self.policy = np.eye(self.n_actions)[np.argmax(coarse_q, axis=1)]

    def forecast_action(self, belief, restricted, depth=3):
        
        transitions = self.transition_model.matrices()
        failure = self.transition_model.failure
        terminal_reward = np.array([self.compute_reward('1;0.8', a) for a in self.actions])
        expected_reward = (transitions * self.reward_table.T[:, :, None]).sum(axis=1)
        expected_reward += failure * terminal_reward[:, None]

        def action_values(mass, mode, remaining):
            if remaining == 0:
                return mass @ self.q_values[:, int(bool(mode))]
            scores = np.empty(self.n_actions)
            for action in range(self.n_actions):
                
                next_mass = self.transition_model.propagate(mass, action)
                next_mode = self.actions[action] == 'restrict_operations'
                continuation = np.max(action_values(next_mass, next_mode, remaining - 1))
                scores[action] = mass @ expected_reward[action] + self.gamma * continuation
            return scores

        scores = action_values(np.asarray(belief, dtype=float), restricted, int(depth))
        return int(np.argmax(scores))


class TrainingEnvironment:
    def __init__(self, context, planner, physical, digital, n_obs):
        self.context, self.planner = context, planner
        self.physical, self.digital = physical, digital
        self.n_obs = int(n_obs)
        self.estimate_cache = {}

    def make(self, seed, initial=0):
        np.random.seed(seed)
        self.context.rnd.seed(seed)
        asset = copy(self.physical)
        asset.degradation_steps = -1
        asset.update_physical_state('do_nothing')
        
        if initial:
            grid = self.planner.grid
            level = (initial - 1) % grid.n_levels
            damage = np.random.uniform(grid.edges[level], grid.edges[level + 1])
            asset.state = np.array([1 + (initial - 1) // grid.n_levels, damage])
        self._observe(asset)
        return asset

    def _observe(self, asset, prior=None, action=None):
        obs, operation = asset.get_observations(self.n_obs)
        rows = tuple(int(i) for i in asset.last_indexes)
        if rows not in self.estimate_cache:
            self.digital.estimate_state(obs, operation)
            self.estimate_cache[rows] = state_index(self.context.STATES, self.digital.state,
                                                   observation=True)
        estimated = self.estimate_cache[rows]
        if prior is None:
            grid = self.planner.grid
            belief = grid.lift(np.full(37, 1 / 37)) * self.planner.likelihood[grid.coarse, estimated]
            belief /= belief.sum()
        else:
            belief = update_posterior(prior, estimated, action, self.planner.transition_model,
                                       self.planner.likelihood)
        asset._q_observation = (estimated, int(bool(asset.restricted)), belief)
        return asset._q_observation

    def advance(self, asset, action_index):
        prior = asset._q_observation[2].copy()
        action = self.planner.actions[action_index]
        asset.update_physical_state(action)
        next_index = self.planner.grid.index(asset.state)
        terminal = next_index is None
        
        reward_index = state_index(self.context.STATES, asset.state, observation=True)
        reward = self.planner.coarse_reward_table[reward_index, action_index]
        if not terminal:
            self._observe(asset, prior, action_index)
        return next_index, terminal, reward

    def training_state(self, asset):
        index = self.planner.grid.index(asset.state)
        if index is None:
            raise ValueError('a failed trajectory has no next decision state')
        return int(index), int(bool(asset.restricted))


def _epsilon(episode, maximum, start, end):
    fraction = min(1.0, episode / max(1, int(maximum * .8)))
    return float(end if fraction == 1 else start + fraction * (end - start))


def evaluate(planner, environment, episodes, steps, seed):
    returns, trace, errors, failures, counts = [], [], [], 0, np.zeros(planner.n_actions, dtype=int)
    for episode in range(episodes):
        asset = environment.make(seed + episode)
        total, discount, actions = 0.0, 1.0, []
        for step in range(steps):
            observation = asset._q_observation
            true_damage = 0 if asset.state[0] == 0 else float(asset.state[1])
            damages = np.array([float(s.split(';')[1]) for s in planner.states])
            errors.append(abs(float(planner.grid.aggregate(observation[2]) @ damages) - true_damage))
            action = planner.select_action(observation)
            
            step_seed = seed + 1000000 + episode * steps + step
            np.random.seed(step_seed)
            environment.context.rnd.seed(step_seed)
            _, terminal, reward = environment.advance(asset, action)
            total += discount * reward
            discount *= planner.gamma
            actions.append(action)
            counts[action] += 1
            if terminal:
                failures += 1
                break
        returns.append(total)
        trace.append(actions + [-1] * (steps - len(actions)))
    return {'mean_return': float(np.mean(returns)), 'std_return': float(np.std(returns)),
            'policy_trace': np.array(trace), 'failure_rate': failures / episodes,
            'posterior_mae': float(np.mean(errors)), 'mean_actions': counts / episodes}


def train(planner, environment, seed=42, max_episodes=50000, rollout_steps=60,
          checkpoint_interval=1000, checkpoint_episodes=64, epsilon_start=1,
          epsilon_end=.05, tolerance=.01, stable_checkpoints=5):
    rng = np.random.default_rng(seed)
    previous_policy, previous_trace, previous_return, previous_q = None, None, None, None
    stable, block_rewards = 0, []
    started = time.perf_counter()
    for episode in range(1, max_episodes + 1):
        
        initial = 0 if episode % 2 else 1 + (episode // 2 - 1) % (planner.n_states - 1)
        
        initial_action = ((episode // 2 - 1) // (planner.n_states - 1)) % planner.n_actions
        asset = environment.make(seed * 1000003 + episode, initial)
        epsilon = _epsilon(episode, max_episodes, epsilon_start, epsilon_end)
        total, terminal = 0.0, False
        for step in range(rollout_steps):
            state = environment.training_state(asset)
            action = (initial_action if initial and step == 0 else
                      planner.select_action(asset._q_observation, epsilon, rng))
            _, terminal, reward = environment.advance(asset, action)
            next_state = None if terminal else environment.training_state(asset)
            planner.update(state, action, reward, next_state, terminal)
            total += reward
            if terminal:
                break
        planner.episode_history.append({'episode': episode, 'epsilon': epsilon,
            'training_reward': total, 'steps': step + 1, 'failed': terminal})
        block_rewards.append(total)
        if episode % checkpoint_interval and episode != max_episodes:
            continue
        result = evaluate(planner, environment, checkpoint_episodes, rollout_steps, seed + 900000000)
        
        policy = np.argmax(planner.q_values[:planner.grid.n_levels + 1, 0], axis=-1)
        latent_change = 1.0 if previous_policy is None else float(np.mean(policy != previous_policy))
        
        trace = result['policy_trace']
        change = 1.0 if previous_trace is None else float(np.mean(trace != previous_trace))
        return_change = np.inf if previous_return is None else abs(
            result['mean_return'] - previous_return) / max(1, abs(previous_return))
        q_change = np.inf if previous_q is None else float(np.max(
            np.abs(planner.q_values - previous_q))) / max(1, float(np.max(np.abs(planner.q_values))))
        residual = planner.bellman_residual()
        visits = planner.parameter_visits()
        covered = int(visits.min()) >= checkpoint_episodes
        steady = (change < tolerance and return_change < tolerance and q_change < tolerance
                  and covered and epsilon == epsilon_end)
        stable = stable + 1 if steady else 0
        planner.training_history.append({'episode': episode, 'epsilon': epsilon,
            'training_mean_reward': np.mean(block_rewards),
            'evaluation_mean_return': result['mean_return'],
            'evaluation_std_return': result['std_return'],
            'evaluation_failure_rate': result['failure_rate'],
            'evaluation_posterior_mae': result['posterior_mae'],
            'policy_change_fraction': change, 'relative_return_change': return_change,
            'latent_policy_change_fraction': latent_change,
            'relative_q_change': q_change, 'minimum_visits': int(visits.min()),
            'bellman_residual_mean': float(residual.mean()),
            'bellman_residual_max': float(residual.max()),
            'stable_checkpoints': stable, 'visited_state_action_pairs': np.count_nonzero(planner.visits),
            'elapsed_seconds': time.perf_counter() - started})
        print(f'Q-learning {episode}/{max_episodes}: return={result["mean_return"]:.3f}, '
              f'failure={result["failure_rate"]:.1%}, policy_change={change:.2%}', flush=True)
        previous_policy, previous_return = policy.copy(), result['mean_return']
        previous_trace = trace.copy()
        previous_q = planner.q_values.copy()
        block_rewards = []
        if stable >= stable_checkpoints:
            break
    planner.final_episode = episode
    planner.stop_reason = 'stable_policy_and_return' if stable >= stable_checkpoints else 'episode_limit'
    planner.update_policy()


def _table_stem(planner, settings):
    
    signature = sha256()
    for value in (planner.reward_table, planner.likelihood, planner.transition_model.matrices(),
                  planner.transition_model.failure, planner.physical_transition_model.matrices(),
                  planner.physical_transition_model.failure):
        signature.update(value.astype('<f8').tobytes())
    for implementation in (QLearningPlanner.action_values, QLearningPlanner.select_action,
                           QLearningPlanner.update, DamageGrid, TrainingEnvironment, state_index, train,
                           DigitalTransitionModel, PhysicalTransitionModel, _major_rehab_probabilities,
                           update_posterior, planner.context.PhysicalAsset.update_physical_state):
        signature.update(inspect.getsource(implementation).encode())
    signature.update(repr(sorted(settings.items())).encode())
    return (f'tunnel_q_h{planner.reward_source.health_weight:g}_dx{planner.grid.step}_'
            f's{settings["seed"]}_g{planner.gamma}_lr{planner.learning_rate}_'
            f'ep{settings["max_episodes"]}_n{settings["n_obs"]}_t{settings["rollout_steps"]}_'
            f'{signature.hexdigest()[:12]}')


def prepare_q_learning(context, physical, digital, output_dir, seed=42, gamma=.9,
                       learning_rate=.1, max_episodes=50000, rollout_steps=60,
                       checkpoint_interval=1000, checkpoint_episodes=64,
                       n_obs=1, epsilon_start=1, epsilon_end=.05,
                       tolerance=.01, stable_checkpoints=5, damage_step=.01,
                       health_weight=1.0):
    numpy_state, random_state = np.random.get_state(), context.rnd.getstate()
    torch_state = context.torch.get_rng_state()
    try:
        planner = QLearningPlanner(context, gamma, learning_rate, damage_step, health_weight)
        
        planner.likelihood = digital.cpd.copy()
        settings = dict(seed=seed, max_episodes=max_episodes, rollout_steps=rollout_steps,
                        checkpoint_interval=checkpoint_interval, checkpoint_episodes=checkpoint_episodes,
                        epsilon_start=epsilon_start, epsilon_end=epsilon_end,
                        tolerance=tolerance, stable_checkpoints=stable_checkpoints, n_obs=n_obs)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        planner.q_table_file = output_dir / (_table_stem(planner, settings) + '.npz')
        if planner.q_table_file.exists():
            with np.load(planner.q_table_file, allow_pickle=False) as saved:
                planner.q_values = saved['q_values'].copy()
                planner.visits = saved['visits'].copy()
                planner.final_episode = int(saved['final_episode'])
                planner.stop_reason = str(saved['stop_reason'])
            planner.update_policy()
            print(f' Q : {planner.q_table_file}', flush=True)
        else:
            environment = TrainingEnvironment(context, planner, physical, digital, n_obs)
            train(planner, environment, **{k: v for k, v in settings.items() if k != 'n_obs'})
            prefix = planner.q_table_file.with_suffix('')
            pd.DataFrame(planner.training_history).to_csv(f'{prefix}_training.csv', index=False)
            pd.DataFrame(planner.episode_history).to_csv(f'{prefix}_episodes.csv', index=False)
            np.savez_compressed(planner.q_table_file, q_values=planner.q_values, visits=planner.visits,
                health_weight=planner.reward_source.health_weight,
                final_episode=planner.final_episode, stop_reason=planner.stop_reason,
                internal_damage_step=planner.grid.step, internal_damage_centers=planner.grid.centers,
                fine_to_coarse=planner.grid.coarse,
                physical_repair_mean=planner.context.physical_repair_mean,
                physical_repair_std=planner.context.physical_repair_std,
                digital_transitions=planner.transition_model.coarse_matrices()[0],
                observation_likelihood=planner.likelihood, actions=planner.actions,
                observation_matrix_file=context.observation_matrix_file.name,
                states=planner.grid.states, external_states=planner.states)
        return planner
    finally:
        np.random.set_state(numpy_state)
        context.rnd.setstate(random_state)
        context.torch.set_rng_state(torch_state)


def plot_q_learning_reward(q_table_file, output_dir):
    data = pd.read_csv(Path(q_table_file).with_name(Path(q_table_file).stem + '_episodes.csv'))
    reward, episodes = data.training_reward.to_numpy(), data.episode.to_numpy()
    window = min(100, len(reward))
    average = pd.Series(reward).rolling(window, min_periods=window).mean()
    with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman']}):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(episodes, reward, color='#999999', alpha=.3, linewidth=.5, label='Episode reward')
        ax.plot(episodes, average, color='#1769aa', linewidth=1.3, label=f'{window}-episode moving average')
        ax.set(xlabel='Training episode', ylabel='Episode total reward', title='Q-learning reward')
        ax.grid(True, axis='y', color='#d9d9d9', linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(Path(output_dir) / 'q_learning_reward.png', dpi=180, bbox_inches='tight')
        plt.close(fig)

