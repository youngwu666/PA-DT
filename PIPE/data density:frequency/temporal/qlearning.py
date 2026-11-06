




from __future__ import annotations

import time
import inspect
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


class QLearningPlanner:
    def __init__(self, n_states, actions, df_values, max_age=70,
                 gamma=0.95, learning_rate=0.1):
        self.n_states = int(n_states)
        self.actions = tuple(str(action) for action in actions)
        self.df_values = np.asarray(sorted(set(float(value) for value in df_values)))
        self.df_indices = {float(value): i for i, value in enumerate(self.df_values)}
        self.max_age = int(max_age)
        self.gamma = float(gamma)
        self.learning_rate = float(learning_rate)
        self.n_actions = len(self.actions)
        shape = (self.n_states, self.max_age + 1, len(self.df_values), self.n_actions)
        self.q_values = np.zeros(shape, dtype=np.float32)
        self.visits = np.zeros(shape, dtype=np.uint32)
        self.training_history = []
        self.episode_history = []
        self.reward_table = None
        self.final_episode = 0
        self.final_validation_return = float('nan')

    def _df_index(self, df_value):
        return self.df_indices[float(df_value)]

    def _key(self, observation):
        state, age, df_value = observation[:3]
        state = min(max(int(state), 0), self.n_states - 1)
        age = min(max(int(age), 0), self.max_age)
        df_index = self._df_index(df_value)
        return state, age, df_index

    def _values_at(self, observation):
        _, age, df_index = self._key(observation)
        belief = np.asarray(observation[3], dtype=float).reshape(-1)
        if (belief.size != self.n_states or not np.isfinite(belief).all()
                or np.any(belief < 0) or belief.sum() <= 0):
            raise ValueError('posterior must be a finite probability vector over all states')
        belief = belief / belief.sum()
        
        return belief @ self.q_values[:, age, df_index, :]

    def select_action(self, observation, epsilon=0.0, rng=None):
        if rng is not None and epsilon > 0 and rng.random() < epsilon:
            return int(rng.integers(self.n_actions))
        return int(np.argmax(self._values_at(observation)))

    def update(self, training_state, action_index, reward, next_training_state):
        
        
        key = self._key(training_state)
        action_index = int(action_index)
        current = float(self.q_values[key][action_index])
        next_key = self._key(next_training_state)
        target = float(reward) + self.gamma * float(np.max(self.q_values[next_key]))

        self.q_values[key][action_index] = current + self.learning_rate * (target - current)
        self.visits[key][action_index] += 1

    def action_probabilities_by_damage(self):
        
        greedy_actions = np.argmax(self.q_values, axis=-1)
        n_states = self.q_values.shape[0]

        probabilities = np.zeros((n_states, self.n_actions), dtype=float)
        for state_index in range(self.n_states):
            actions = greedy_actions[state_index].reshape(-1)
            counts = np.bincount(actions, minlength=self.n_actions)
            probabilities[state_index] = counts / counts.sum()
        return probabilities


def _epsilon_at(episode, max_episodes, epsilon_start, epsilon_end):
    decay_episodes = max(1, int(max_episodes * 0.8))
    fraction = min(1.0, episode / decay_episodes)
    return float(epsilon_start + fraction * (epsilon_end - epsilon_start))


def evaluate_policy(agent, make_environment, observe, advance, reward,
                    episodes, max_steps, seed, filter_samples=0):
    returns = []
    action_counts = np.zeros(agent.n_actions, dtype=np.int64)
    final_states = []
    health_trajectories = []
    policy_traces = []
    unsupported_mass = []
    for episode in range(episodes):
        episode_seed = int(seed + episode)
        np.random.seed(episode_seed)
        environment = make_environment(episode_seed)
        environment._q_filter_samples = int(filter_samples)
        environment._q_filter_rng = np.random.default_rng(episode_seed + 333000000)
        total_return = 0.0
        discount = 1.0
        health = []
        actions = []

        for step in range(max_steps):
            observation = observe(environment)
            action_index = agent.select_action(observation)
            action = agent.actions[action_index]
            _, age, df_index = agent._key(observation)
            visits = agent.visits[:, age, df_index, :]
            unvisited = (visits == 0).any(axis=1)
            unsupported_mass.append(float(observation[3][unvisited].sum()))
            
            np.random.seed(int(seed + 10000000 + episode * (max_steps + 1) + step))
            next_true_state = advance(environment, action)
            total_return += discount * reward(next_true_state, action)
            action_counts[action_index] += 1
            discount *= agent.gamma
            health.append(float(environment.ratio))
            actions.append(action_index)

        returns.append(total_return)
        final_states.append(float(environment.ratio))
        health_trajectories.append(health)
        policy_traces.append(actions)

    return {
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0,
        'mean_final_remaining_ratio': float(np.mean(final_states)),
        'mean_remaining_ratio': float(np.mean(health_trajectories)),
        'mean_action_counts': (action_counts / max(1, episodes)).tolist(),
        'policy_trace': np.asarray(policy_traces, dtype=np.int8),
        'unvisited_action_posterior_mass': float(np.mean(unsupported_mass)),
    }


def train_q_learning(agent, make_environment, observe, advance, reward,
                     seed, learning_state, max_episodes=50000, rollout_steps=70,
                     epsilon_start=1.0, epsilon_end=0.05,
                     checkpoint_interval=1000, checkpoint_episodes=64,
                     stable_checkpoints=5, validation_samples=300,
                     stability_tolerance=0.01):
    rng = np.random.default_rng(seed)
    previous_trace = None
    previous_evaluation = None
    stable_count = 0
    started = time.perf_counter()
    steps = max(int(rollout_steps), agent.max_age + 1)
    block_rewards = []
    final_validation_return = np.nan

    for episode in range(1, max_episodes + 1):
        episode_seed = int(seed * 1000003 + episode)
        np.random.seed(episode_seed)
        environment = make_environment(episode_seed)
        epsilon = _epsilon_at(episode, max_episodes, epsilon_start, epsilon_end)
        episode_reward = 0.0

        
        
        for step in range(steps):
            observation = observe(environment)
            training_state = learning_state(environment)
            action_index = agent.select_action(observation, epsilon=epsilon, rng=rng)
            action = agent.actions[action_index]
            next_true_state = advance(environment, action)
            reward_value = reward(next_true_state, action)
            episode_reward += float(reward_value)
            next_training_state = learning_state(environment)
            agent.update(training_state, action_index, reward_value, next_training_state)

        agent.episode_history.append({
            'episode': episode,
            'epsilon': epsilon,
            'training_reward': episode_reward,
            'training_steps': steps,
        })
        block_rewards.append(episode_reward)
        if episode % checkpoint_interval != 0 and episode != max_episodes:
            continue

        evaluation = evaluate_policy(
            agent, make_environment, observe, advance, reward,
            episodes=checkpoint_episodes, max_steps=rollout_steps,
            seed=int(seed * 1000003 + 900000000), filter_samples=validation_samples
        )
        trace = evaluation['policy_trace']
        policy_change = (float(np.mean(trace != previous_trace))
                         if previous_trace is not None else 1.)
        return_change = (abs(evaluation['mean_return'] - previous_evaluation)
                         / max(1., abs(previous_evaluation))
                         if previous_evaluation is not None else np.inf)
        stable = (policy_change < stability_tolerance
                  and return_change < stability_tolerance)
        stable_count = stable_count + 1 if stable else 0

        training_mean_reward = float(np.mean(block_rewards))
        training_std_reward = float(np.std(block_rewards, ddof=1)) if len(block_rewards) > 1 else 0.0
        agent.training_history.append({
            'episode': episode,
            'epsilon': epsilon,
            'training_mean_reward': training_mean_reward,
            'training_std_reward': training_std_reward,
            'evaluation_mean_return': evaluation['mean_return'],
            'evaluation_std_return': evaluation['std_return'],
            'evaluation_final_remaining_ratio': evaluation['mean_final_remaining_ratio'],
            'evaluation_mean_remaining_ratio': evaluation['mean_remaining_ratio'],
            'policy_change_fraction': policy_change,
            'relative_return_change': return_change,
            'unvisited_action_posterior_mass': evaluation['unvisited_action_posterior_mass'],
            'stable_checkpoints': stable_count,
            'visited_state_action_pairs': int(np.count_nonzero(agent.visits)),
            'final_episode': episode,
            'final_validation_return': evaluation['mean_return'],
            'elapsed_seconds': time.perf_counter() - started,
        })
        final_validation_return = evaluation['mean_return']
        block_rewards = []
        print(f'Q-learning seed={seed} episode={episode}/{max_episodes} '
              f'return={evaluation["mean_return"]:.4f} '
              f'policy_change={policy_change:.3%}', flush=True)
        previous_trace = trace.copy()
        previous_evaluation = evaluation['mean_return']

        if (stable_count >= stable_checkpoints
                and episode >= max(1, int(max_episodes * 0.8))):
            break

    agent.completed_episodes = episode
    agent.stop_reason = ('stable_validation_policy_and_return'
                         if stable_count >= stable_checkpoints else 'episode_limit')
    agent.final_episode = episode
    agent.final_validation_return = float(final_validation_return)
    print(f'Q-learning : {agent.stop_reason} {episode} '
          f' Q ', flush=True)
    return agent


class RewardModel:
    
    def __init__(self, dbn, health_weight, transitions):
        self.corrosion_states = dbn.corrosion_states
        self.actions_list = dbn.possible_actions
        self.source = dbn.Planner(dbn.corrosion_states, dbn.possible_actions,
                                  dbn.gamma, transitions, health_weight=health_weight)
        self.reward_table = None

    def compute_reward(self, state, action):
        return self.source.compute_reward(state, action)

    def build_reward_table(self):
        self.reward_table = np.array([
            [self.compute_reward(state, action) for action in self.actions_list]
            for state in self.corrosion_states
        ])
        return self.reward_table


def _build_observation_likelihood(dbn, pinn_model, training_observations):
    




    indices = {str(state): i for i, state in enumerate(dbn.corrosion_states)}
    counts = np.zeros((len(indices), len(indices)), dtype=float)
    estimates = {}
    for depth, ratio in zip(dbn.y_dmax, dbn.labels_ratio):
        rows = tuple(np.argsort(np.abs(dbn.labels_ratio - ratio))[:training_observations])
        if rows not in estimates:
            prediction = np.mean(pinn_model.predict(dbn.X_regressors[list(rows)]))
            _, state = dbn.trans_dmax(prediction, dbn.corrosion_states, dbn.state_edges)
            estimates[rows] = indices[str(state)]
        _, state = dbn.trans_dmax(depth, dbn.corrosion_states, dbn.state_edges)
        counts[indices[str(state)], estimates[rows]] += 1

    
    likelihood = counts + 1e-6
    likelihood /= likelihood.sum(axis=1, keepdims=True)
    return likelihood, counts


def _update_posterior(prior, estimated_index, action_index, transitions, likelihood):
    
    prediction = transitions[action_index] @ prior
    posterior = prediction * likelihood[:, estimated_index]
    return posterior / posterior.sum()


def _q_table_stem(dbn, health_weight, planner, reward_model, training_observations,
                  transitions, training_options, pinn_model):
    
    def text(value):
        return str(value).removesuffix('.0').replace('.', 'p')

    action_codes = {'Do_nothing': 'DN', 'Adding_inhibitor': 'AI',
                    'Pigging': 'PG', 'Replacement': 'RP'}
    actions = '-'.join(action_codes[str(a)] for a in dbn.possible_actions)
    rewards = '-'.join(text(value) for value in reward_model.reward_table[0])
    reward_profile = sha256(reward_model.reward_table.astype('<f8').tobytes()).hexdigest()[:10]
    observation_profile = sha256(planner.observation_matrix.astype('<f8').tobytes()).hexdigest()[:10]
    likelihood_profile = sha256(planner.observation_likelihood.astype('<f8').tobytes()).hexdigest()[:10]
    transition_profile = sha256()
    setup = dbn.GetSetup(dbn.corrosion_states, dbn.possible_actions,
                         planner.observation_matrix)
    for age in (0, 1, 50, 70):
        for df_value in planner.df_values:
            transition_profile.update(setup.get_transitions(age, float(df_value)).astype('<f8').tobytes())
    perception_profile = sha256()
    for values in (dbn.X_regressors, dbn.y_dmax, dbn.state_edges):
        perception_profile.update(np.asarray(values, dtype='<f8').tobytes())
    perception_profile.update(inspect.getsource(type(pinn_model).predict).encode())
    perception_profile.update(inspect.getsource(dbn.trans_dmax).encode())
    perception_profile.update(str(pinn_model.nominal_dmax).encode())
    for model_path in pinn_model.model_paths:
        perception_profile.update(Path(model_path).read_bytes())
    physical_profile = sha256()
    asset = dbn.PhysicalAsset('power_function', dbn.X_regressors, dbn.wall_thickness)
    rng = np.random.default_rng(dbn.random_seed)
    
    for age in (0, 1, 50, 70):
        for df_value in planner.df_values:
            for state in dbn.corrosion_states.astype(float):
                for action in dbn.possible_actions:
                    asset.degradation_steps = age
                    asset.dmax = dbn.wall_thickness * (1.0 - state)
                    asset.pre_dmax = asset.dmax
                    asset.df = float(df_value)
                    asset.update_physical_state(str(action), rng=rng)
                    physical_profile.update(np.asarray(
                        [asset.dmax, asset.degradation_steps, asset.df], dtype='<f8').tobytes())
    dfs = '-'.join(text(v) for v in planner.df_values)
    stem = (
        f'qbasic_a{len(dbn.possible_actions)}_{actions}_h{text(health_weight)}'
        f'_r{rewards}_hr{reward_profile}'
        f'_s{dbn.random_seed}o{training_observations}_g{text(dbn.gamma)}'
        f'_ob{observation_profile}_lh{likelihood_profile}'
        f'_tm{transition_profile.hexdigest()[:10]}'
        f'_pm{physical_profile.hexdigest()[:10]}'
        f'_pd{perception_profile.hexdigest()[:10]}'
        f'_l{text(dbn.learning_rate)}'
        f'_ep{training_options["max_episodes"]}_a{dbn.max_age}t{max(int(training_options["rollout_steps"]), dbn.max_age + 1)}'
        f'v{training_options["rollout_steps"]}'
        f'_e{text(training_options["epsilon_start"])}-{text(training_options["epsilon_end"])}'
        f'_w{text(dbn.wall_thickness)}_df{dfs}'
        f'_ck{training_options["checkpoint_interval"]}n{training_options["checkpoint_episodes"]}k{training_options["stable_checkpoints"]}'
        f't{text(training_options["stability_tolerance"])}m{training_options["validation_samples"]}'
    )
    return stem


class _TrainingEnvironment:
    




    def __init__(self, dbn, pinn_model, observation_matrix, transitions,
                 training_observations, observation_likelihood=None):
        self.dbn = dbn
        self.training_observations = int(training_observations)
        self.pinn_model = pinn_model
        self.observation_matrix = np.asarray(observation_matrix, dtype=float)
        if observation_likelihood is None:
            observation_likelihood, _ = _build_observation_likelihood(
                dbn, pinn_model, self.training_observations)
        self.observation_likelihood = np.asarray(observation_likelihood, dtype=float)
        self.state_indices = {str(state): i for i, state in enumerate(dbn.corrosion_states)}
        self.action_indices = {str(action): i for i, action in enumerate(dbn.possible_actions)}
        self.setup = dbn.GetSetup(dbn.corrosion_states, dbn.possible_actions,
                                  self.observation_matrix)

    @lru_cache(maxsize=None)
    def _estimate(self, rows):
        observations = self.dbn.X_regressors[list(rows)]
        predictions = self.pinn_model.predict(observations)
        _, state = self.dbn.trans_dmax(np.mean(predictions), self.dbn.corrosion_states,
                                      self.dbn.state_edges)
        return self.state_indices[str(state)]

    def _prepare_observation(self, asset, prior=None, previous_action=None, transitions=None):
        dbn = self.dbn
        ratio, _ = dbn.trans_dmax(asset.dmax, dbn.corrosion_states, dbn.state_edges)
        distances = np.abs(dbn.labels_ratio - ratio)
        rows = tuple(int(i) for i in np.argsort(distances)[:self.training_observations])
        estimated = self._estimate(rows)
        if prior is None:
            
            belief = self.observation_likelihood[:, estimated].copy()
            belief /= belief.sum()
        else:
            action_index = self.action_indices[str(previous_action)]
            filter_samples = getattr(asset, '_q_filter_samples', 0)
            if filter_samples:
                prior = asset._q_filter_rng.multinomial(filter_samples, prior) / filter_samples
            belief = _update_posterior(prior, estimated, action_index,
                                       transitions, self.observation_likelihood)
        asset._q_belief = belief
        asset._q_observation = (
            estimated, min(max(0, int(asset.degradation_steps)), dbn.max_age),
            float(asset.df), belief,
        )

    def make(self, seed):
        dbn = self.dbn
        np.random.seed(int(seed))
        asset = dbn.PhysicalAsset(degradation_law='power_function',
                                  X_regressors=dbn.X_regressors,
                                  wall_thickness=dbn.wall_thickness)
        asset.generate_observations()
        asset.update_physical_state(action='Do_nothing')
        self._prepare_observation(asset)
        return asset

    def observe(self, asset):
        return asset._q_observation

    def advance(self, asset, action):
        prior = asset._q_belief.copy()
        transitions = self.setup.get_transitions(int(asset.degradation_steps), float(asset.df))
        asset.update_physical_state(action=action)
        self._prepare_observation(asset, prior, action, transitions)
        return self.state_indices[str(asset.state)]

    def learning_state(self, asset):
        return (self.state_indices[str(asset.state)],
                min(max(0, int(asset.degradation_steps)), self.dbn.max_age), float(asset.df))


def prepare_q_learning(dbn, health_weight, transitions, pinn_model, observation_matrix, output_dir,
                       training_observations):
    
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        return _train_or_load(dbn, health_weight, transitions, pinn_model,
                              observation_matrix, Path(output_dir), training_observations)
    finally:
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def plot_q_learning_reward(q_table_file, output_dir):
    q_table_file = Path(q_table_file)
    episode_file = q_table_file.with_name(f'{q_table_file.stem}_episodes.csv')
    if not episode_file.exists():
        print(f': {episode_file}', flush=True)
        return None

    data = pd.read_csv(episode_file)
    episode = data['episode'].to_numpy(dtype=float)
    reward = data['training_reward'].to_numpy(dtype=float)
    valid = np.isfinite(episode) & np.isfinite(reward)
    if not valid.any():
        print(f': {episode_file}', flush=True)
        return None

    episode = episode[valid]
    reward = reward[valid]
    window = min(100, len(reward))
    mean_reward = pd.Series(reward).rolling(window=window, min_periods=window).mean().to_numpy()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'q_learning_reward.png'

    with plt.rc_context({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 10,
        'axes.linewidth': 0.8,
    }):
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(episode, reward, color='#999999', linewidth=0.5, alpha=0.3,
                label='Episode reward')
        ax.plot(episode, mean_reward, color='#1769aa', linewidth=1.3,
                label=f'{window}-episode moving average')
        ax.set_xlabel('Training episode')
        ax.set_ylabel('Episode total reward')
        ax.set_title('Q-learning reward')
        ax.grid(True, axis='y', color='#d9d9d9', linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(output_file, dpi=180, bbox_inches='tight')
        plt.close(fig)

    print(f': {output_file}', flush=True)
    return output_file


def _train_or_load(dbn, health_weight, transitions, pinn_model, observation_matrix, output_dir,
                   training_observations):
    output_dir.mkdir(parents=True, exist_ok=True)
    training_options = {
        name: getattr(dbn, name)
        for name in ('max_episodes', 'epsilon_start', 'epsilon_end',
                     'checkpoint_interval', 'checkpoint_episodes',
                     'stable_checkpoints', 'validation_samples', 'stability_tolerance')
    }
    training_options['rollout_steps'] = dbn.training_rollout_years

    reward_model = RewardModel(dbn, health_weight, transitions)
    reward_model.build_reward_table()
    df_after_action = {
        'Do_nothing': 1.0, 'Adding_inhibitor': 0.1,
        'Pigging': 0.5, 'Replacement': 1.0,
    }
    df_values = sorted({df_after_action[str(action)] for action in dbn.possible_actions})
    planner = QLearningPlanner(len(dbn.corrosion_states), dbn.possible_actions, df_values,
                               max_age=dbn.max_age, gamma=dbn.gamma,
                               learning_rate=dbn.learning_rate)
    planner.reward_table = reward_model.reward_table.copy()
    planner.df_after_action = df_after_action
    planner.transitions = np.asarray(transitions, dtype=float)
    planner.observation_likelihood, planner.observation_calibration_counts = \
        _build_observation_likelihood(dbn, pinn_model, training_observations)
    planner.observation_matrix = np.asarray(observation_matrix, dtype=float).copy()
    table_name = _q_table_stem(dbn, health_weight, planner, reward_model,
                               training_observations, transitions, training_options,
                               pinn_model)
    q_table_file = output_dir / f'{table_name}.npz'
    planner.q_table_file = q_table_file
    if q_table_file.exists():
        with np.load(q_table_file) as saved:
            planner.q_values = saved['q_values'].copy()
            planner.visits = saved['visits'].copy()
            planner.final_episode = int(saved['final_episode'])
            planner.final_validation_return = float(saved['final_validation_return'])
            planner.completed_episodes = int(saved['completed_episodes'])
            planner.stop_reason = str(saved['stop_reason'])
        print(f' Q : {q_table_file}', flush=True)
        return planner

    environment = _TrainingEnvironment(dbn, pinn_model, observation_matrix, transitions,
                                       training_observations, planner.observation_likelihood)
    def reward(next_true_state, action):
        return reward_model.reward_table[next_true_state, environment.action_indices[str(action)]]

    print(f' Q-learning: {dbn.random_seed}', flush=True)
    train_q_learning(
        planner, environment.make, environment.observe, environment.advance, reward,
        seed=dbn.random_seed, learning_state=environment.learning_state,
        **training_options
    )
    rows = [{'seed': dbn.random_seed, **row} for row in planner.training_history]
    pd.DataFrame(rows).to_csv(output_dir / f'{table_name}_training.csv', index=False)
    pd.DataFrame(planner.episode_history).to_csv(output_dir / f'{table_name}_episodes.csv', index=False)
    evaluation_rows = []
    for years in dbn.evaluation_years:
        result = evaluate_policy(planner, environment.make, environment.observe,
                                  environment.advance, reward,
                                  episodes=dbn.evaluation_episodes, max_steps=years,
                                  seed=dbn.random_seed + 740000, filter_samples=training_options["validation_samples"])
        evaluation_rows.append({
            'model': 'Q-learning', 'seed': dbn.random_seed, 'evaluation_years': years,
            'mean_discounted_return': result['mean_return'],
            'std_discounted_return': result['std_return'],
            'mean_final_remaining_ratio': result['mean_final_remaining_ratio'],
            'mean_remaining_ratio': result['mean_remaining_ratio'],
            'unvisited_action_posterior_mass': result['unvisited_action_posterior_mass'],
            'final_episode': planner.final_episode,
            **{f'mean_actions_{name}': count for name, count in
               zip(planner.actions, result['mean_action_counts'])},
        })
    pd.DataFrame(evaluation_rows).to_csv(output_dir / f'{table_name}_evaluation.csv', index=False)
    np.savez_compressed(q_table_file, q_values=planner.q_values, visits=planner.visits,
                        observation_matrix=planner.observation_matrix,
                        observation_likelihood=planner.observation_likelihood,
                        observation_calibration_counts=planner.observation_calibration_counts,
                        final_episode=np.asarray(planner.final_episode),
                        final_validation_return=np.asarray(planner.final_validation_return),
                        seed=np.asarray(dbn.random_seed, dtype=np.int64),
                        corrosion_states=dbn.corrosion_states.astype('U'),
                        actions=np.asarray(dbn.possible_actions, dtype='U'),
                        df_categories=planner.df_values,
                        completed_episodes=np.asarray(planner.completed_episodes),
                        stop_reason=np.asarray(planner.stop_reason))
    print(f'Q : {q_table_file}', flush=True)
    return planner
