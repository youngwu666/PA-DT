







from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import random as rnd
import sys

sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from pgmpy.models import DynamicBayesianNetwork as DBN
    from pgmpy.factors.discrete import TabularCPD
except ImportError as error:
    raise SystemExit('Pythonpgmpypytorch') from error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plotter import Plot
from qlearning import prepare_q_learning, plot_q_learning_reward, state_index, update_posterior

STATES = np.array(['0;0.0']+[f'{q};{d/10:.1f}' for q in range(1,7) for d in range(3,9)])
ACTIONS = np.array(['do_nothing', 'major_rehab', 'restrict_operations'])
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

def read_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def preserve_rng():
    
    numpy_state, random_state = np.random.get_state(), rnd.getstate()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        np.random.set_state(numpy_state)
        rnd.setstate(random_state)


class ShardedRecords:
    
    def __init__(self, data_dir):
        self.shards = [np.load(Path(data_dir) / f'recordings_classifier_{i}of2.npy',
                               mmap_mode='r', allow_pickle=False) for i in (1, 2)]
        if any(x.ndim != 3 or x.shape[1:] != (600, 10) for x in self.shards):
            raise ValueError('[,600,10]')
        self.offsets = np.r_[0, np.cumsum([len(x) for x in self.shards])]

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, indexes):
        indexes = np.asarray(indexes, dtype=int).ravel()
        if np.any((indexes < 0) | (indexes >= len(self))):
            raise IndexError('')
        values = np.empty((len(indexes), 600, 10), dtype=np.float32)
        for i, shard in enumerate(self.shards):
            rows = (indexes >= self.offsets[i]) & (indexes < self.offsets[i + 1])
            values[rows] = shard[indexes[rows] - self.offsets[i]]
        if not np.isfinite(values).all():
            raise ValueError('')
        return values


def normalize_cpd(matrix):
    
    col_sums = matrix.sum(axis=0, keepdims=True)
    return matrix / col_sums


class GetSetup:
    
    def __init__(self, states_list, actions_list, observation_likelihood):
        self.states_list = states_list
        self.actions_list = actions_list
        self.observation_likelihood = observation_likelihood
        self.Ns = len(states_list)
        self.Na = len(actions_list)

    def get_combined_transitions(self, transitions):
        
        
        combined = []
        for transition in transitions:
            values = self.observation_likelihood[:, :, None] * transition[:, None, :]
            combined.append(normalize_cpd(values.reshape(self.Ns, -1)))
        return np.concatenate(combined, axis=1)

    def get_graph(self, cpd_d_to_u, transitions):
        Ns, Na = self.Ns, self.Na
        state_names = self.states_list.tolist()
        action_names = self.actions_list.tolist()

        graph_structure = [
            (('U^{A}_{-1}', 0), ('D', 0)),
            (('D_{NN}', 0), ('D', 0)),
            (('D', 0), ('U^{P}', 0)),
            (('D', 0), ('D', 1)),
            (('U^{A}_{-1}', 1), ('D', 1)),
            (('D_{NN}', 1), ('D', 1)),
            (('D', 1), ('U^{P}', 1)),
        ]
        digital_subgraph = [
            (('D', 0), ('U', 0)),
            (('D', 0), ('D', 1)),
            (('U', 0), ('D', 1)),
            (('D', 1), ('U', 1)),
        ]

        
        initial_posterior = normalize_cpd(self.observation_likelihood)
        initial_values = np.concatenate([initial_posterior] * Na, axis=1)
        combined = self.get_combined_transitions(transitions)
        list_cpd_graph = []
        for step in (0, 1):
            previous_action = ('U^{A}_{-1}', step)
            estimated_state = ('D_{NN}', step)
            digital_state = ('D', step)
            planned_action = ('U^{P}', step)
            list_cpd_graph.extend([
                TabularCPD(previous_action, Na, np.full((Na, 1), 1 / Na),
                    state_names={previous_action: action_names}),
                TabularCPD(estimated_state, Ns, np.full((Ns, 1), 1 / Ns),
                    state_names={estimated_state: state_names}),
            ])
            evidence = [previous_action, estimated_state]
            cardinalities = [Na, Ns]
            names = {digital_state: state_names, previous_action: action_names,
                     estimated_state: state_names}
            if step == 1:
                evidence.append(('D', 0))
                cardinalities.append(Ns)
                names[('D', 0)] = state_names
            list_cpd_graph.extend([
                TabularCPD(digital_state, Ns, initial_values if step == 0 else combined,
                    evidence=evidence, evidence_card=cardinalities, state_names=names),
                TabularCPD(planned_action, Na, cpd_d_to_u,
                    evidence=[digital_state], evidence_card=[Ns],
                    state_names={planned_action: action_names, digital_state: state_names}),
            ])

        
        survival_transitions = transitions / transitions.sum(axis=1, keepdims=True)
        list_cpd_subgraph = [
            TabularCPD(('D', 0), Ns, np.full((Ns, 1), 1 / Ns)),
            TabularCPD(('U', 0), Na, cpd_d_to_u,
                evidence=[('D', 0)], evidence_card=[Ns]),
            TabularCPD(('D', 1), Ns, np.concatenate(survival_transitions, axis=1),
                evidence=[('U', 0), ('D', 0)], evidence_card=[Na, Ns]),
            TabularCPD(('U', 1), Na, cpd_d_to_u,
                evidence=[('D', 1)], evidence_card=[Ns]),
        ]
        return graph_structure, digital_subgraph, list_cpd_graph, list_cpd_subgraph


class PhysicalAsset:
    





























    def __init__(self, path_data, degradation_law):
        self.path_data = path_data
        self.state = None
        self.restricted = None
        self.X_classifier = None
        self.X_regressors = None
        self.labels_class = None
        self.labels_level = None
        self.labels_pressure = None 
        self.labels_velocity = None 
        self.n_classes = None
        self.degradation_law = degradation_law
        self.degradation_steps = None

    def generate_observations(self):
        
        for name in ('class', 'level', 'pressure', 'velocity'):
            setattr(self, 'labels_' + name, np.load(
                Path(self.path_data) / ('labels_' + name + '.npy'), allow_pickle=False))
        self.X_classifier = ShardedRecords(self.path_data)
        n = len(self.X_classifier)
        for value in (self.labels_class, self.labels_level, self.labels_pressure, self.labels_velocity):
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError('')
        self.n_classes = np.unique(self.labels_class).shape[0]
        if not np.array_equal(np.unique(self.labels_class), np.arange(7)):
            raise ValueError('')
        self.degradation_steps = -1

    def get_observations(self, n_obs):
        










        
        
        check_class = self.labels_class == self.state[0]
        
        if self.restricted:
            check_also_train = np.logical_and(self.labels_class == self.state[0], self.labels_pressure <= 18.)
        if self.restricted:
            
            indexes = np.where(check_also_train)[0]
        else:
            
            indexes = np.where(check_class)[0]
        
        distances = np.abs(self.labels_level[indexes] - self.state[1])
        closest_indeces = np.argsort(distances)
        obs_idx = indexes[closest_indeces[:n_obs]]
        if len(obs_idx) != n_obs:
            raise ValueError('/')
        self.last_indexes = obs_idx
        
        return self.X_classifier[obs_idx], dict(
            speed_kmh=self.labels_velocity[obs_idx], axle_mass_t=self.labels_pressure[obs_idx])
    
    def update_physical_state(self, action, rng=None, discrete_rng=None):
        







        
        normal_rng = np.random if rng is None else rng
        category_rng = rnd if discrete_rng is None else discrete_rng
        
        self.degradation_steps += 1
        
        
        if action == "do_nothing" or action == "major_rehab":
            self.restricted = 0
        elif action == "restrict_operations":
            self.restricted = 1

        
        
        
        if action == "major_rehab":
            if self.state[0] != 0:
                repair_amount = max(0.0, normal_rng.normal(
                    physical_repair_mean, physical_repair_std))
                remaining_damage = max(0.0, float(self.state[1]) - repair_amount)
                if remaining_damage < 0.30:
                    self.degradation_steps = 0
                    self.state = np.array([0, 0.30])
                else:
                    self.state[1] = remaining_damage
            return

        
        if self.degradation_steps == 0:
            self.state = np.array([0, 0.30])
    
        else:
            if self.degradation_law == "custom":
                if action == "do_nothing" or action == "restrict_operations":
                    if self.state[0] == 0:
                        self.state[0] = 4
                        self.state[1] = 0.31
                    else:
                        self.state[1] = np.array([0.31, 0.31, 0.32, 0.33, 0.34, 0.36, 0.37, 0.38, 0.39, 
                                                  0.40, 0.41, 0.42, 0.43, 0.44, 0.46, 0.47, 0.48, 0.49,
                                                  0.50, 0.51, 0.52, 0.53, 0.54, 0.56, 0.57, 0.58, 0.59,
                                                  0.60, 0.61, 0.62, 0.63, 0.64, 0.66, 0.67, 0.68, 0.69,
                                                  0.70, 0.71, 0.72, 0.73, 0.74, 0.76, 0.77, 0.78])[self.degradation_steps]
    
            elif self.degradation_law == "stochastic":
                if action == "do_nothing":
                    if self.state[0] == 0:
                        
                        if category_rng.randrange(2):
                            self.state[0] = 1 + category_rng.randrange(self.n_classes-1) 
                            self.state[1] = normal_rng.uniform(low=0.3, high=0.35)
                    else:
                        self.state[1] += np.max((0.,normal_rng.normal(0.015, 0.01)))
                        
                elif action == "restrict_operations":
                    if self.state[0] == 0:
                        
                        if not category_rng.randrange(4):
                            self.state[0] = 1 + category_rng.randrange(self.n_classes-1) 
                            self.state[1] = normal_rng.uniform(low=0.3, high=0.35)
                    else:
                        self.state[1] += np.max((0.,normal_rng.normal(0.0095, 0.005)))

class DigitalAsset:
    



    def __init__(self, models_path, cpd, d_level_discr=.1):
        
        self.cpd = np.asarray(cpd, dtype=float)
        self.d_level_discr = d_level_discr
        self.state, self.qois = None, None
        path = Path(models_path)
        self.classifier_code = read_module('tunnel_reference_classifier',
                                          path / '02_classifier_code/classifier.py')
        self.regressor_code = read_module('tunnel_reference_regressor',
                                          path / '03_regressor_code/regressor.py')
        weights_dir = path / '05_regressor_weights'
        
        with preserve_rng():
            self.classifier, self.classifier_checkpoint = self.classifier_code.load_model(
                path / '04_classifier_weights/classifier_weights.pt')
            self.regressors = {}
            for q in range(1, 7):
                checkpoints = torch.load(weights_dir / f'region_{q}.pt',
                                         map_location='cpu', weights_only=False)['members']
                members = []
                for checkpoint in checkpoints:
                    model, stats = self.regressor_code.load_model(checkpoint)
                    members.append((model, stats))
                self.regressors[q] = members
        metadata = json.loads((path / '01_dataset/metadata.json').read_text())
        self.normalization = metadata['normalization']
        pool_scaler = json.loads((path.parent / 'splits/classifier/normalization.json').read_text())
        self.dt = metadata['dt']
        for key in ('mean', 'std'):
            np.testing.assert_allclose(self.normalization[key], metadata['normalization'][key], rtol=0, atol=0)
            np.testing.assert_allclose(self.normalization[key], pool_scaler[key], rtol=0, atol=0)

    def estimate_state(self, obs_classifier, obs_regressors):
        mean, std = [np.asarray(self.normalization[k]) for k in ('mean', 'std')]
        u = np.asarray(obs_classifier, dtype=float) * std + mean
        z = self.regressor_code.extract_features(u, dt=self.dt)
        x = np.column_stack((z, obs_regressors['speed_kmh'], obs_regressors['axle_mass_t']))
        scaled = ((x - self.classifier_checkpoint['feature_mean'])
                  / self.classifier_checkpoint['feature_std'])
        with torch.no_grad():
            logits = self.classifier(torch.tensor(scaled, dtype=torch.float32)).numpy()
        damage_class = np.argmax(logits, axis=1)
        values, counts = np.unique(damage_class, return_counts=True)
        damage_class = int(values[np.argmax(counts)])
        if damage_class == 0:
            per_record = np.zeros(len(obs_classifier))
            continuous = 0.
            damage_level = 0.
        else:
            members = self.regressors[damage_class]
            predictions = []
            with torch.no_grad():
                for model, stats in members:
                    normalized = (x - np.asarray(stats['feature_mean'])) / np.asarray(stats['feature_std'])
                    predicted = model.predict(torch.tensor(normalized, dtype=torch.float32)).numpy().ravel()
                    predictions.append(predicted * stats['target_std'] + stats['target_mean'])
            
            per_record = np.mean(np.stack(predictions), axis=0)
            continuous = float(np.mean(per_record))
            
            damage_level = np.around(continuous*0.1/self.d_level_discr, 1) *self.d_level_discr/0.1
        self.state = np.array([damage_class, damage_level])
        self.continuous_damage = continuous
        self.last_features = x

class Planner:
    


    def __init__(self, health_weight):
        self.health_weight = float(health_weight)

    def compute_reward(self, state, action):
        







        alpha_health = self.health_weight
        
        def r_health(state):
            






            delim = state.find(';')
            loc_damage = int(state[:delim])
            liv_damage = float(state[delim+1:])
            if loc_damage == 0:
                r_health = 0
            else:
                r_health = 4 - np.exp(5*liv_damage)
                if liv_damage >= 0.79:
                    r_health = -250
            return r_health
                    
        def r_action(action):
            






            if action == "do_nothing":
                r_action = 30.
            elif action == "major_rehab":
                r_action = -100.
            elif action == "restrict_operations":
                
                r_action = 27.

            return r_action
        
        return r_action(action) + alpha_health*r_health(state)

        
class Graph:
    





















    def __init__(self,
                 physical_asset,
                 digital_asset,
                 states_list, 
                 actions_list,
                 planner):
        self.physical_asset = physical_asset
        self.digital_asset = digital_asset
        self.states_list = states_list
        self.actions_list = actions_list
        self.planner = planner
        self.setup = GetSetup(states_list, actions_list, planner.likelihood)
        self.dbn = DBN()
        self.dbn_predict = DBN()
        self.hist_var = None
        self.hist_d_state_prob = None
        self.hist_actions_prob = None
        self.time_index = None
        self.state_names = self.states_list.tolist()
        self.action_names = self.actions_list.tolist()
        self.damage_values = np.asarray(
            [0.0 if i == 0 else float(state.split(';')[1])
             for i, state in enumerate(self.state_names)],
            dtype=float
        )
        self.hist_observations = []
        self.pred_var = None
        self.pred_d_state_prob = None
        self.pred_actions_prob = None
        self.prediction_start_step = None
        self.failure = False
        self._fine_history = []

    def assemble_graph(self,
                       graph_structure,
                       cpd_list,
                       digital_subgraph,
                       list_cpd_subgraph,
                       plot=False):
        self.dbn.add_edges_from(graph_structure)
        for cpd in cpd_list:
            self.dbn.add_cpds(cpd)
        self.dbn.check_model()

        self.dbn_predict.add_edges_from(digital_subgraph)
        for cpd in list_cpd_subgraph:
            self.dbn_predict.add_cpds(cpd)
        self.dbn_predict.check_model()
        self.time_index = -1

        if plot:
            self.dbn.to_daft().render()
            self.dbn_predict.to_daft().render()
            plt.show()

    def compute_confidence_intervals(self, probabilities, ci_level=0.95):
        
        probabilities = np.asarray(probabilities, dtype=float)
        if (
            probabilities.shape != self.damage_values.shape
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or probabilities.sum() <= 0
            or not 0 < ci_level < 1
        ):
            raise ValueError('state probabilities or interval level are invalid')

        order = np.argsort(self.damage_values, kind='stable')
        values = self.damage_values[order]
        sorted_probabilities = probabilities[order]
        cdf = np.cumsum(sorted_probabilities)
        cdf /= cdf[-1]

        lower_index = np.searchsorted(cdf, (1 - ci_level) / 2, side='left')
        upper_index = np.searchsorted(cdf, 1 - (1 - ci_level) / 2, side='left')
        return float(values[lower_index]), float(values[upper_index])

    def _posterior_mean(self, probabilities):
        probabilities = np.asarray(probabilities, dtype=float)
        return float((probabilities * self.damage_values).sum())


    def _record(self, belief, terminal=False, previous_action=None):
        asset = self.physical_asset
        index = state_index(STATES, asset.state)
        if terminal:
            action, reference = None, None
            digital_state, digital_value, low, high = None, np.nan, np.nan, np.nan
            reward = self.planner.compute_reward(f'{int(asset.state[0])};0.8',
                previous_action if previous_action is not None else 'do_nothing')
            r_low, r_high = reward, reward
        else:
            action_id = self.planner.select_action((int(np.argmax(belief)), asset.restricted, belief))
            action = self.planner.actions[action_id]
            coarse_belief = self.planner.grid.aggregate(belief)
            
            coarse_belief /= coarse_belief.sum()
            true_index = self.planner.grid.index(asset.state)
            reference = self.planner.actions[int(np.argmax(
                self.planner.q_values[true_index, int(bool(asset.restricted))]))]
            digital_state = self.states_list[int(np.argmax(coarse_belief))]
            digital_value = self._posterior_mean(coarse_belief)
            low, high = self.compute_confidence_intervals(coarse_belief)
            reward, r_low, r_high = self.planner.reward_stats(belief, action)
        region = int(asset.state[0])
        damage = 0.0 if region == 0 else float(asset.state[1])
        
        realized = np.nan
        if previous_action is not None:
            reward_index = state_index(STATES, asset.state, observation=True)
            realized = self.planner.coarse_reward_table[reward_index, self.action_names.index(previous_action)]
        row = dict(p_state=f'{region};{damage:.6f}',
            p_state_discrete=None if index is None else self.states_list[index],
            d_state=digital_state, current_action=action, best_action=reference,
            p_state_num=damage, d_state_num=digital_value, Lower=low, Upper=high,
            reward=reward, reward_Lower=r_low, reward_Upper=r_high,
            realized_reward=realized,
            failed=terminal, restricted=bool(asset.restricted))
        action_prob = np.zeros(len(self.action_names))
        if action is not None:
            action_prob[self.action_names.index(action)] = 1
        self.hist_var = pd.concat([self.hist_var, pd.DataFrame([row])], ignore_index=True)
        self.hist_d_state_prob = pd.concat([self.hist_d_state_prob,
            pd.DataFrame([np.zeros(37) if terminal else coarse_belief], columns=self.state_names)], ignore_index=True)
        self._fine_history.append(None if terminal else belief.copy())
        self.hist_actions_prob = pd.concat([self.hist_actions_prob,
            pd.DataFrame([action_prob], columns=self.action_names)], ignore_index=True)
        if not terminal:
            names = self.digital_asset.regressor_code.feature_names()
            for k, pool_index in enumerate(asset.last_indexes):
                region = int(asset.labels_class[pool_index])
                observation = dict(time=self.time_index, true_region=region,
                    true_stiffness_loss=0 if region == 0 else float(asset.labels_level[pool_index]),
                    estimated_region=int(self.digital_asset.state[0]),
                    estimated_stiffness_loss=self.digital_asset.continuous_damage)
                observation.update({name: float(self.digital_asset.last_features[k, j])
                                    for j, name in enumerate(names)})
                self.hist_observations.append(observation)

    def simulate(self, n_steps, n_samples, n_obs):
        




        if self.time_index != -1:
            raise RuntimeError('run a new graph for each online simulation')
        self.hist_var = pd.DataFrame()
        self.hist_d_state_prob = pd.DataFrame(columns=self.state_names)
        self.hist_actions_prob = pd.DataFrame(columns=self.action_names)
        self.physical_asset.update_physical_state('do_nothing')
        prior = None
        for step in range(n_steps + 1):
            self.time_index = step
            previous_action = None if step == 0 else str(self.hist_var.current_action.iloc[-1])
            if step:
                
                transitions, _ = self.planner.transition_model.coarse_matrices(prior)
                combined = self.setup.get_combined_transitions(transitions)
                self.dbn.get_cpds(('D', 1)).values[...] = combined.reshape(
                    len(self.states_list), len(self.actions_list),
                    len(self.states_list), len(self.states_list))
                self.physical_asset.update_physical_state(previous_action)
            if state_index(STATES, self.physical_asset.state) is None:
                self.failure = True
                self._record(None, terminal=True, previous_action=previous_action)
                print(f' {step} 80%', flush=True)
                return
            obs, operation = self.physical_asset.get_observations(n_obs)
            self.digital_asset.estimate_state(obs, operation)
            estimated = state_index(STATES, self.digital_asset.state, observation=True)
            if prior is None:
                grid = self.planner.grid
                belief = grid.lift(np.full(37, 1 / 37)) * self.planner.likelihood[grid.coarse, estimated]
                belief /= belief.sum()
            else:
                belief = update_posterior(prior, estimated, self.action_names.index(previous_action),
                    self.planner.transition_model, self.planner.likelihood)
            self._record(belief, previous_action=previous_action)
            prior = belief

    def predict(self, n_steps, n_samples):
        
        start = self.hist_var.iloc[self.time_index]
        self.prediction_start_step = self.time_index
        self.pred_d_state_prob = pd.DataFrame(columns=self.state_names)
        self.pred_actions_prob = pd.DataFrame(columns=self.action_names)
        self._fine_prediction = []
        columns = ['d_state', 'd_state_num', 'current_action', 'Lower', 'Upper',
                   'reward', 'reward_Lower', 'reward_Upper', 'failure_probability']
        if bool(start.failed):
            self.pred_var = pd.DataFrame(columns=columns)
            return
        belief = self._fine_history[self.time_index].copy()
        restricted = int(bool(start.restricted))
        states, actions, rows = [], [], []
        for step in range(n_steps + 1):
            action_id = self.planner.forecast_action(belief, restricted, prediction_lookahead_steps)
            action = self.planner.actions[action_id]
            coarse_belief = self.planner.grid.aggregate(belief)
            
            coarse_belief /= coarse_belief.sum()
            low, high = self.compute_confidence_intervals(coarse_belief)
            reward, r_low, r_high = self.planner.reward_stats(belief, action)
            rows.append(dict(d_state=self.states_list[int(np.argmax(coarse_belief))],
                d_state_num=self._posterior_mean(coarse_belief), current_action=action,
                Lower=low, Upper=high, reward=reward, reward_Lower=r_low,
                reward_Upper=r_high, failure_probability=np.nan))
            states.append(coarse_belief)
            self._fine_prediction.append(belief.copy())
            actions.append(np.eye(len(self.actions_list))[action_id])
            if step == n_steps:
                break
            
            transitions, _ = self.planner.transition_model.coarse_matrices(belief)
            self.dbn_predict.get_cpds(('D', 1)).values[...] = transitions.transpose(1, 0, 2)

            
            
            self.dbn_predict.get_cpds(('D', 0)).values[...] = coarse_belief
            
            
            action_cpd = self.dbn_predict.get_cpds(('U', 0))
            action_cpd.values[...] = np.eye(len(self.actions_list))[action_id, :, None]
            sampled = self.dbn_predict.simulate(n_time_slices=2, n_samples=n_samples,
                do={('U', 0): action_id},
                seed=int(np.random.randint(0, 2**31 - 1)), show_progress=False)
            counts = np.bincount(sampled[('D', 1)].to_numpy(dtype=int), minlength=37)
            continuous_prediction = self.planner.transition_model.propagate(belief, action_id)
            belief = self.planner.grid.lift(counts / counts.sum(), reference=continuous_prediction)
            restricted = int(action == 'restrict_operations')
        self.pred_var = pd.DataFrame(rows)
        self.pred_d_state_prob = pd.DataFrame(states, columns=self.state_names)
        self.pred_actions_prob = pd.DataFrame(actions, columns=self.action_names)



def build_twin(health_weight):
    
    rnd.seed(simulation_seed)
    np.random.seed(simulation_seed)
    conf_mat_dt = np.load(observation_matrix_file, allow_pickle=False)
    if (conf_mat_dt.shape != (37, 37) or not np.isfinite(conf_mat_dt).all()
            or np.any(conf_mat_dt <= 0)
            or not np.allclose(conf_mat_dt.sum(axis=1), 1.0, atol=1e-12, rtol=0)):
        raise ValueError('37x37')
    physical = PhysicalAsset(str(ROOT / 'data') + '/', 'stochastic')
    physical.generate_observations()
    digital = DigitalAsset(ROOT / 'models', conf_mat_dt)
    q_planner = prepare_q_learning(
        sys.modules[__name__], physical, digital, HERE / 'q_tables',
        seed=random_seed, gamma=gamma, learning_rate=learning_rate,
        max_episodes=max_episodes, rollout_steps=n_steps,
        checkpoint_interval=checkpoint_interval,
        checkpoint_episodes=checkpoint_episodes,
        n_obs=n_obs, epsilon_start=epsilon_start, epsilon_end=epsilon_end,
        tolerance=stability_tolerance, stable_checkpoints=stable_checkpoints,
        damage_step=internal_damage_step,
        health_weight=health_weight,
    )
    setup = GetSetup(STATES, ACTIONS, q_planner.likelihood)
    transitions, _ = q_planner.transition_model.coarse_matrices()
    structure, subgraph, tables, sub_tables = setup.get_graph(q_planner.policy.T, transitions)
    graph = Graph(physical, digital, STATES, ACTIONS, q_planner)
    graph.assemble_graph(structure, tables, subgraph, sub_tables, plot=False)
    return graph, q_planner


def save(excel_file):
    
    global graph_frame

    n_sim = len(graph_frame.hist_var)
    simulation_time = list(range(n_sim))
    prediction_time = list(range(
        graph_frame.prediction_start_step,
        graph_frame.prediction_start_step + len(graph_frame.pred_var)
    ))
    action_to_value = {
        str(action): i for i, action in enumerate(graph_frame.actions_list)
    }

    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        simulation = graph_frame.hist_var.copy()
        simulation['current_action_value'] = simulation['current_action'].map(action_to_value)
        simulation['best_action_value'] = simulation['best_action'].map(action_to_value)
        simulation.index = simulation_time
        simulation.index.name = 'time'
        simulation.to_excel(writer, sheet_name='Simulation', index=True)

        state_probability = graph_frame.hist_d_state_prob.copy()
        state_probability.index = simulation_time
        state_probability.index.name = 'time'
        state_probability.to_excel(writer, sheet_name='Simulation_State_Prob', index=True)

        action_probability = graph_frame.hist_actions_prob.copy()
        action_probability.index = simulation_time
        action_probability.index.name = 'time'
        action_probability.to_excel(writer, sheet_name='Simulation_Action_Prob', index=True)

        pd.DataFrame(graph_frame.hist_observations).to_excel(
            writer,
            sheet_name='Simulation_Observations',
            index=False
        )

        prediction = graph_frame.pred_var.copy()
        prediction['current_action_value'] = prediction['current_action'].map(action_to_value)
        prediction.index = prediction_time
        prediction.index.name = 'time'
        prediction.to_excel(writer, sheet_name='Prediction', index=True)

        predicted_state_probability = graph_frame.pred_d_state_prob.copy()
        predicted_state_probability.index = prediction_time
        predicted_state_probability.index.name = 'time'
        predicted_state_probability.to_excel(
            writer,
            sheet_name='Prediction_State_Prob',
            index=True
        )

        predicted_action_probability = graph_frame.pred_actions_prob.copy()
        predicted_action_probability.index = prediction_time
        predicted_action_probability.index.name = 'time'
        predicted_action_probability.to_excel(
            writer,
            sheet_name='Prediction_Action_Prob',
            index=True
        )


def analyze_annual_cost(graph_frame, n_steps=60, restore_healthy=False,
                        daily_inspection=False):
    







    if len(graph_frame.hist_var) < n_steps + 1:
        raise ValueError('60')
    normal_rng = np.random.RandomState(simulation_seed)
    category_rng = rnd.Random(simulation_seed)
    asset = PhysicalAsset(str(ROOT / 'data'), 'stochastic')
    asset.n_classes = 7
    asset.degradation_steps = -1
    asset.update_physical_state('do_nothing', normal_rng, category_rng)
    reward_model = graph_frame.planner.reward_source
    rows = []
    for step in range(n_steps + 1):
        region = int(asset.state[0])
        damage = 0.0 if region == 0 else float(asset.state[1])
        needs_repair = (damage >= traditional3_threshold if daily_inspection
                        else damage > traditional_threshold)
        action = 'major_rehab' if region != 0 and needs_repair else 'do_nothing'
        rows.append(dict(time=step, p_state=f'{region};{damage:.12f}',
            p_state_num=damage, current_action=action,
            current_action_value=int(np.flatnonzero(ACTIONS == action)[0]),
            action_executed=step < n_steps, threshold_exceeded=damage > traditional_threshold,
            outside_calibration=damage > 0.80, Lower=np.nan, Upper=np.nan,
            action_reward=np.nan, weighted_health_reward=np.nan,
            reward=np.nan, reward_Lower=np.nan, reward_Upper=np.nan, realized_reward=np.nan))
        if daily_inspection:
            
            inspected = step < n_steps and action == 'do_nothing'
            rows[-1].update(inspection_performed=inspected,
                inspection_reward=traditional3_inspection_reward if inspected else 0.0,
                maintenance_threshold_reached=region != 0 and needs_repair,
                repair_target_ratio=0.30 if step < n_steps and action == 'major_rehab' else np.nan)
        if step == n_steps:
            break
        asset.update_physical_state(action, normal_rng, category_rng)
        if (restore_healthy or daily_inspection) and action == 'major_rehab':
            
            
            asset.state = np.array([0, 0.30])
            asset.degradation_steps = 0
            asset.restricted = 0
        reward_index = state_index(STATES, asset.state, observation=True)
        action_reward = reward_model.compute_reward('0;0.0', action)
        weighted_health_reward = (
            reward_model.compute_reward(str(STATES[reward_index]), action) - action_reward)
        if daily_inspection and action == 'do_nothing':
            
            action_reward = traditional3_inspection_reward
        rows[-1]['action_reward'] = action_reward
        rows[-1]['weighted_health_reward'] = weighted_health_reward
        rows[-1]['reward'] = action_reward + weighted_health_reward
    traditional = pd.DataFrame(rows)
    traditional.loc[traditional.index[1:], 'realized_reward'] = traditional.reward.iloc[:-1].to_numpy()

    summaries = []
    traditional_name = ('Traditional 3' if daily_inspection else
                        'Traditional 2' if restore_healthy else 'Traditional >80%')
    for name, frame in (('DBN-Qlearning', graph_frame.hist_var),
                        (traditional_name, traditional)):
        actions = frame.current_action.iloc[:n_steps]
        reward_actions = actions.copy()
        if 'inspection_performed' in frame:
            inspected = frame.inspection_performed.iloc[:n_steps].fillna(False).astype(bool)
            reward_actions.loc[inspected & actions.eq('do_nothing')] = 'inspection'
        control_rewards = {str(a): reward_model.compute_reward('0;0.0', str(a))
                           for a in ACTIONS}
        control_rewards['inspection'] = traditional3_inspection_reward
        cumulative_action_reward = float(reward_actions.map(control_rewards).sum())
        actual_rewards = frame.realized_reward.iloc[1:n_steps + 1].to_numpy(dtype=float)
        damages = frame.p_state_num.iloc[:n_steps + 1].to_numpy(dtype=float)
        summaries.append(dict(method=name, health_weight=reward_model.health_weight,
            decision_steps=n_steps, do_nothing=int((reward_actions == 'do_nothing').sum()),
            major_rehab=int((reward_actions == 'major_rehab').sum()),
            restrict_operations=int((reward_actions == 'restrict_operations').sum()),
            inspection_count=int((reward_actions == 'inspection').sum()),
            cumulative_action_reward=cumulative_action_reward,
            average_action_reward=cumulative_action_reward / n_steps,
            realized_discounted_return=float(actual_rewards @ gamma ** np.arange(n_steps)),
            threshold_exceeded_count=int((damages > traditional_threshold).sum()),
            mean_stiffness_loss=float(damages.mean()),
            maximum_stiffness_loss=float(damages.max()), final_stiffness_loss=float(damages[-1])))
    return traditional, pd.DataFrame(summaries)


def save_traditional_results(excel_file, traditional, comparison, health_weight,
                             restore_healthy=False, daily_inspection=False):
    
    settings = pd.DataFrame([
        ('method', 'traditional_threshold'), ('health_weight', health_weight),
        ('decision_steps', n_steps), ('threshold', traditional_threshold),
        ('simulation_seed', simulation_seed),
        ('threshold_rule', 'do_nothing until true damage > 0.80, then major_rehab'),
        ('after_exceedance', 'repair allowed only in this comparator; main method still terminates'),
        ('reward_units', 'normalized action rewards; not calibrated currency'),
        ('action_reward_rule', 'do_nothing +30; restrict_operations +27; major_rehab -100; inspection +10; one accounted action per executed step'),
        ('reward_rule', 'R_action + health_weight * R_health'),
        ('prediction', 'none; actual simulation only'),
    ], columns=['setting', 'value'])
    if restore_healthy and not daily_inspection:
        settings = pd.concat([settings, pd.DataFrame([
            ('strategy_label', 'Traditional 2'),
            ('repair_rule', 'major_rehab guarantees damage below 0.30; mapped to healthy'),
            ('healthy_encoding', 'region 0; plotted damage 0 denotes the healthy class, not measured zero damage'),
            ('new_damage', 'original DN initiation probability 0.5; uniformly choose one of six regions, including the previous one'),
        ], columns=['setting', 'value'])], ignore_index=True)
    if daily_inspection:
        settings.loc[settings.setting.eq('threshold'), 'value'] = traditional3_threshold
        settings.loc[settings.setting.eq('threshold_rule'), 'value'] = (
            'inspect at each daily decision; true damage >= 0.60 triggers major_rehab')
        settings.loc[settings.setting.eq('after_exceedance'), 'value'] = (
            'repair at 60%; original >80% main-method termination remains unchanged')
        settings = pd.concat([settings, pd.DataFrame([
            ('strategy_label', 'Traditional 3'),
            ('repair_rule', 'major_rehab targets 0.30 damage, specially mapped to healthy in this comparator'),
            ('healthy_encoding', 'region 0; plotted 0 denotes the healthy class; repair target is 0.30, not measured zero damage'),
            ('inspection_interval_days', 1),
            ('inspection_reward', traditional3_inspection_reward),
            ('inspection_count', int(traditional.inspection_performed.sum())),
            ('inspection_accounting', 'non-MA steps count inspection +10 only; MA steps count -100 only; no DN benefit or extra inspection fee; final snapshot has no executed action'),
            ('structural_action_encoding', 'current_action do_nothing denotes no structural intervention; inspection_performed identifies the inspection reward action'),
            ('inspection_observation', 'exact true state for this comparator; no added measurement error'),
            ('time_assumption', 'one comparator decision step is one day; existing deterioration increments retained without calendar calibration'),
            ('new_damage', 'original DN probability 0.5; uniformly choose any of six regions after healthy recovery'),
        ], columns=['setting', 'value'])], ignore_index=True)
    prediction_columns = ['time', 'd_state_num', 'current_action', 'current_action_value',
        'Lower', 'Upper', 'reward', 'reward_Lower', 'reward_Upper', 'failure_probability']
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        traditional.to_excel(writer, sheet_name='Simulation', index=False)
        pd.DataFrame(columns=prediction_columns).to_excel(writer, sheet_name='Prediction', index=False)
        pd.DataFrame(columns=['time', 'estimated_region', 'estimated_stiffness_loss']).to_excel(
            writer, sheet_name='Simulation_Observations', index=False)
        comparison.to_excel(writer, sheet_name='Cost_Comparison', index=False)
        settings.to_excel(writer, sheet_name='Settings', index=False)


def run(output, health_weight):
    
    global graph_frame

    output_dir = Path(output).resolve()
    health_name = f'health_{health_weight:g}'.replace('.', 'p')
    simulation_results_dir = output_dir / 'simulation_results' / health_name
    plotters_dir = output_dir / 'plotters' / health_name
    simulation_results_dir.mkdir(parents=True, exist_ok=True)
    plotters_dir.mkdir(parents=True, exist_ok=True)

    graph_frame, planner = build_twin(health_weight)
    graph_frame.simulate(
        n_steps, n_samples, n_obs
    )

    start_step = (
        graph_frame.time_index if prediction_start_step is None else prediction_start_step
    )
    completed_step = graph_frame.time_index
    graph_frame.time_index = start_step

    try:
        graph_frame.predict(
            prediction_horizon, prediction_samples
        )
    finally:
        graph_frame.time_index = completed_step

    excel_file = simulation_results_dir / 'digital_twin_simulation_results.xlsx'
    save(excel_file)

    plotting_frame = Plot(excel_file, plotters_dir, ACTIONS)
    plotting_frame.plot_all_together()
    plotting_frame.plot_sensor_dmax()
    plot_q_learning_reward(planner.q_table_file, plotters_dir)

    traditional, comparison = analyze_annual_cost(graph_frame, n_steps)
    traditional_file = simulation_results_dir / 'traditional_strategy_results.xlsx'
    save_traditional_results(traditional_file, traditional, comparison, health_weight)
    traditional2, comparison2 = analyze_annual_cost(graph_frame, n_steps, restore_healthy=True)
    traditional2_file = simulation_results_dir / 'traditional_strategy2_results.xlsx'
    save_traditional_results(traditional2_file, traditional2, comparison2, health_weight,
                             restore_healthy=True)
    traditional3, comparison3 = analyze_annual_cost(graph_frame, n_steps, daily_inspection=True)
    traditional3_file = simulation_results_dir / 'traditional_strategy3_results.xlsx'
    save_traditional_results(traditional3_file, traditional3, comparison3, health_weight,
                             daily_inspection=True)
    
    with plt.rc_context():
        single_plot = read_module('traditional_single_plot', HERE / '' / 'plotterall.py')
        single_plot._configure_style()
        history, prediction, _ = single_plot._read_result(traditional_file)
        single_plot.plot_simulation_and_prediction(
            history, prediction, plotters_dir / 'traditional_strategy.svg')
        history2, prediction2, _ = single_plot._read_result(traditional2_file)
        single_plot.plot_simulation_and_prediction(
            history2, prediction2, plotters_dir / 'traditional_strategy2.svg')
        history3, prediction3, _ = single_plot._read_result(traditional3_file)
        single_plot.plot_simulation_and_prediction(
            history3, prediction3, plotters_dir / 'traditional_strategy3.svg')
        compare_plot = read_module('traditional_comparison_plot', HERE / '' / 'plotterall_duibi.py')
        compare_plot._configure_style()
        compare_plot.plot_simulation_and_prediction(
            [compare_plot._read_result(excel_file), compare_plot._read_result(traditional_file)],
            plotters_dir / 'simulation_and_prediction_duibi.svg')
        compare_plot.plot_simulation_and_prediction(
            [compare_plot._read_result(excel_file), compare_plot._read_result(traditional2_file)],
            plotters_dir / 'simulation_and_prediction_duibi_traditional2.svg')
        compare_plot.plot_simulation_and_prediction(
            [compare_plot._read_result(excel_file), compare_plot._read_result(traditional3_file)],
            plotters_dir / 'simulation_and_prediction_duibi_traditional3.svg')
    print(comparison.to_string(index=False), flush=True)
    print(comparison2.to_string(index=False), flush=True)
    print(comparison3.to_string(index=False), flush=True)

    print(json.dumps({
        'health_weight': health_weight,
        'q_table_file': str(planner.q_table_file),
        'history_count': len(graph_frame.hist_var),
        'prediction_count': len(graph_frame.pred_var),
        'excel_file': str(excel_file),
        'traditional_excel_file': str(traditional_file),
        'traditional2_excel_file': str(traditional2_file),
        'traditional3_excel_file': str(traditional3_file),
        'plots': [
            str(plotters_dir / 'simulation_and_prediction.png'),
            str(plotters_dir / 'sensor_dmax.png'),
        ],
    }, ensure_ascii=False, indent=2))



health_weights = (1, 2)  
traditional_threshold = 0.80  
traditional3_threshold = 0.60  
traditional3_inspection_reward = 10.0  
random_seed = 42
simulation_seed = 40  
n_steps = 60
n_obs = 1
n_samples = 100
prediction_horizon = 20
prediction_samples = 1000
prediction_lookahead_steps = 3  
prediction_start_step = None
gamma = 0.90
learning_rate = 0.10  
internal_damage_step = 0.01  
max_episodes = 50000
epsilon_start = 1.0
epsilon_end = 0.05
checkpoint_interval = 1000
checkpoint_episodes = 64
stability_tolerance = 0.01
stable_checkpoints = 5
observation_matrix_file = HERE / 'observation_likelihood_local.npy'  
physical_repair_mean = 0.35  
physical_repair_std = 0.02  
digital_dn_probability = 0.10  
digital_re_probability = 0.03
digital_major_steps = (0, 2, 4, 6)  
digital_major_probabilities = (0.05, 0.30, 0.40, 0.25)


def main():
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        for health_weight in health_weights:
            print(f'\n===  {health_weight} {random_seed} ===',
                flush=True)
            run(HERE, health_weight)


if __name__ == '__main__':
    main()


