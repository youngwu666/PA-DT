





import os
import sys
from pathlib import Path
from functools import lru_cache
from physical_transition import PhysicalTransition
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from plotter4 import Plot
import qlearning
from qlearning import plot_q_learning_reward, _update_posterior
import torch
import seaborn as sns
from pinn import PINN
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
os.environ["PYTHONHASHSEED"] = str(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
from config_gpu import config_gpu
device = config_gpu(1)

def normalize_cpd(matrix):
    col_sums = matrix.sum(axis=0, keepdims=True)
    return matrix / col_sums

def trans_dmax(dmax, corrosion_states, state_edges):

    ratio = (wall_thickness - dmax)/wall_thickness
    
    if ratio > state_edges[0]: 
        state = corrosion_states[0] 
    if ratio < state_edges[-1]:  
        state = corrosion_states[-1] 
    
    for i in range(len(state_edges) - 1):
        if state_edges[i] >= ratio >= state_edges[i + 1]:
            state = corrosion_states[i]

    return ratio, state



class GetSetup:
    def __init__(self, 
                 corrosion_states, 
                 actions_list, 
                 conf_mat_dt):

        self.corrosion_states = corrosion_states
        self.actions_list = actions_list
        self.conf_mat_dt = conf_mat_dt
        
        self.Ns = self.corrosion_states.shape[0]       
        self.Na = self.actions_list.shape[0]      
        self.state_names = self.corrosion_states.tolist()    
        self.action_names = self.actions_list.tolist()  
        self.transition_model = PhysicalTransition(
            PhysicalAsset, X_regressors, wall_thickness,
            self.corrosion_states, self.actions_list)


    def get_transitions(self, age=0, df_value=1.0):
        return self.transition_model.matrices(int(age), float(df_value))

    @lru_cache(maxsize=None)
    def get_combined_transitions(self, age, df_value):
        transitions = self.get_transitions(age, df_value)
        combined = []
        for transition in transitions:
            combined.append(self._get_combined_transition(self.conf_mat_dt, transition))
        return np.concatenate(combined, axis=1)

    def _get_combined_transition(self, t1, t2):

        n_states = t1.shape[0]
        card_1 = t1.shape[1]
        card_2 = t2.shape[1]
        comb = np.zeros((n_states, card_1 * card_2))

        for i in range(n_states):           
            for j in range(card_1):        
                for k in range(card_2):
                    idx = j * n_states + k
                    comb[i, idx] = t1[i, j] * t2[i, k] 
        comb = normalize_cpd(comb)
        return comb


    def get_graph(self, cpd_d_to_u, transitions):

        Ns = self.Ns
        Na = self.Na
        state_names = self.state_names
        action_names = self.action_names

        
        graph_structure = ([
            (('U^{A}_{-1}', 0), ('D', 0)),
            (('D_{NN}', 0), ('D', 0)),
            (('D', 0), ('U^{P}', 0)),
            (('D', 0), ('D', 1)),
            (('U^{A}_{-1}', 1), ('D', 1)),
            (('D_{NN}', 1), ('D', 1)),
            (('D', 1), ('U^{P}', 1))
        ])
        
        digital_subgraph = ([
            (('D', 0), ('U', 0)),
            (('D', 0), ('D', 1)),
            (('U', 0), ('D', 1)),
            (('D', 1), ('U', 1))
        ])
        
        cond_d_nothing = self._get_combined_transition(t1=self.conf_mat_dt, t2=transitions[0])
        cond_d_inhibitor = self._get_combined_transition(t1=self.conf_mat_dt, t2=transitions[1])
        cond_d_pigging = self._get_combined_transition(t1=self.conf_mat_dt, t2=transitions[2])
        cond_d_replacement = self._get_combined_transition(t1=self.conf_mat_dt, t2=transitions[3])
        
        cond_d = np.concatenate((cond_d_nothing, cond_d_inhibitor, cond_d_pigging, cond_d_replacement), axis=1)

        
        U_a_0_cpd = TabularCPD(
            ('U^{A}_{-1}', 0), 
            Na,
            np.ones((Na, 1)),
            state_names={('U^{A}_{-1}', 0): action_names}
        )
        
        D_NN_0_cpd = TabularCPD(
            ('D_{NN}', 0), 
            Ns,
            np.ones((Ns, 1)), 
            state_names={('D_{NN}', 0): state_names}
        )
        
        D_0_cpd = TabularCPD(
            ('D', 0), 
            Ns,
            np.concatenate((self.conf_mat_dt, 
                            transitions[1],
                            np.eye(Ns),
                            np.eye(Ns)), axis=1),
            evidence=[('U^{A}_{-1}', 0), ('D_{NN}', 0)],
            evidence_card=[Na, Ns],
            state_names={
                ('D', 0): state_names,
                ('U^{A}_{-1}', 0): action_names,
                ('D_{NN}', 0): state_names
            }
        )

        U_p_0_cpd = TabularCPD(
            ('U^{P}', 0), 
            Na,
            cpd_d_to_u,
            evidence=[('D', 0)],
            evidence_card=[Ns],
            state_names={('U^{P}', 0): action_names, 
                         ('D', 0): state_names}
        )
        
        U_a_1_cpd = TabularCPD(
           ('U^{A}_{-1}', 1), 
           Na,
           np.ones((Na, 1)),
           state_names={('U^{A}_{-1}', 1): action_names}
        )
        
        D_NN_1_cpd = TabularCPD(
            ('D_{NN}', 1),
            Ns,
            np.ones((Ns, 1)),
            state_names={('D_{NN}', 1): state_names}
        )
        
        D_1_cpd = TabularCPD(
            ('D', 1),
            Ns,
            cond_d,
            evidence=[('U^{A}_{-1}', 1), ('D_{NN}', 1), ('D', 0)],
            evidence_card=[Na, Ns, Ns],
            state_names={
                ('D', 1): state_names,
                ('U^{A}_{-1}', 1): action_names,
                ('D_{NN}', 1): state_names,
                ('D', 0): state_names
            }
        )
        
        U_p_1_cpd = TabularCPD(
            ('U^{P}', 1),
            Na,
            cpd_d_to_u,
            evidence=[('D', 1)],
            evidence_card=[Ns],
            state_names={('U^{P}', 1): action_names, 
                         ('D', 1): state_names}
        )
        
        
        
        D_0_sub_cpd = TabularCPD(
            ('D', 0), 
            Ns,
            np.ones((Ns, 1))
        )

        U_0_sub_cpd = TabularCPD(
            ('U', 0), 
            Na,
            cpd_d_to_u,
            evidence=[('D', 0)],
            evidence_card=[Ns]
        )

        D_1_sub_cpd = TabularCPD(
            ('D', 1),
            Ns,
            np.concatenate(np.array([transitions[i] for i
                                     in range(Na)]), 1),
            evidence=[('U', 0), ('D', 0)],
            evidence_card=[Na, Ns]
        )
        
        U_1_sub_cpd = TabularCPD(
            ('U', 1), 
            Na,
            cpd_d_to_u,
            evidence=[('D', 1)],
            evidence_card=[Ns]
        )         
        list_cpd_graph = [U_a_0_cpd, D_NN_0_cpd, D_0_cpd, U_p_0_cpd,
                          U_a_1_cpd, D_NN_1_cpd, D_1_cpd, U_p_1_cpd]
        list_cpd_subgraph = [D_0_sub_cpd, U_0_sub_cpd, D_1_sub_cpd, U_1_sub_cpd]
        return graph_structure, digital_subgraph, list_cpd_graph, list_cpd_subgraph



class PhysicalAsset:
    def __init__(self, 
                 degradation_law, 
                 X_regressors, 
                 wall_thickness,
                 ):
        
        self.degradation_law = degradation_law
        self.X_regressors = X_regressors          
        self.wall_thickness = wall_thickness
        self.state = None                         
        self.ratio = None
        self.dmax = None                          
        self.pre_dmax = None
        self.degradation_steps = -1               
        self.df= 1
        self.size = 3
        
    def generate_observations(self):   
        
        self.degradation_steps = -1
        self.dmax = 0.000       
        ratio, state = trans_dmax(self.dmax, corrosion_states, state_edges)
        self.state = state      
        self.ratio = ratio

    def get_observations(self, n_obs):

        ratio, state = trans_dmax(self.dmax, corrosion_states, state_edges)
        
        current_residual_ratio = ratio
        
        
        distances = np.abs(labels_ratio - current_residual_ratio)
        closest_indices = np.argsort(distances)[:n_obs]
        
        
        norm_obs = self.X_regressors[closest_indices]   
        observed_ratios = labels_ratio[closest_indices] 
        
        return norm_obs, observed_ratios
    
    def update_physical_state(self, action, rng=np.random):

        self.degradation_steps += 1

        if action == "Replacement":
            
            self.degradation_steps = 0
            self.dmax = 0.000
            self.pre_dmax = 0.000
            self.df = 1.0

        
        if self.degradation_steps == 0:
            self.dmax = 0.000  
            self.pre_dmax = 0.000
            ratio, state = trans_dmax(self.dmax, corrosion_states, state_edges)
            self.state = state  
            self.ratio = ratio
            return          
        else:
            if self.degradation_law == "custom":
                
                sequence = ([0.000, 0.087, 0.155, 0.210, 0.254, 0.291, 0.322, 0.348, 0.370, 0.390,
                    0.407, 0.422, 0.436, 0.448, 0.459, 0.469, 0.478, 0.487, 0.495, 0.502,
                    0.509, 0.515, 0.521, 0.527, 0.532, 0.537, 0.542, 0.546, 0.550, 0.554,
                    0.558, 0.562, 0.565, 0.569, 0.572, 0.575, 0.578, 0.581, 0.584, 0.587,
                    0.589, 0.592, 0.595, 0.597, 0.600, 0.602, 0.605, 0.607, 0.610, 0.612,
                    0.615, 0.617, 0.620, 0.625, 0.868])
                state_interval = self.wall_thickness * 0.946 / 12 
                
                if action == "Do_nothing":
                    self.df = 1
                    natural_dmax = self.wall_thickness * np.array(sequence)[self.degradation_steps]  
                    delta = (natural_dmax - self.pre_dmax) * (self.df)
                    natural_dmax = self.pre_dmax + delta
                    self.dmax = natural_dmax 
                    
                elif action == "Adding_inhibitor":
                    natural_dmax = self.wall_thickness * np.array(sequence)[self.degradation_steps]  
                    
                    
                    prob_retr1, prob_retr2, prob_retr3 = 0.30, 0.20, 0.10
                    mean_repair = (prob_retr1 * 1 + prob_retr2 * 2 + prob_retr3 * 3) * state_interval
                    self.pre_dmax = max(0, self.pre_dmax - np.random.normal(mean_repair, 0.01))
                    
                    
                    
                    
                    self.df = 0.3
                    delta = (natural_dmax - self.pre_dmax) * (self.df)
                    natural_dmax = self.pre_dmax + delta 
                    
                elif action == "Pigging":
                    natural_dmax = self.wall_thickness * np.array(sequence)[self.degradation_steps]
 
                    
                    prob_retr1, prob_retr2, prob_retr3 = 0.15, 0.05, 0.00
                    mean_repair = (prob_retr1 * 1 + prob_retr2 * 2 + prob_retr3 * 3) * state_interval
                    self.pre_dmax = max(0, self.pre_dmax - np.random.normal(mean_repair, 0.01))
                    
                    
                    
                    
                    self.df = 0.7
                    delta = (natural_dmax - self.pre_dmax) * (self.df)
                    natural_dmax = self.pre_dmax + delta                   
                 
            elif self.degradation_law == "power_function":  
                
                state_interval = corrosion_range / 12

                if action == "Do_nothing":  

                    natural_dmax = 0.164 * (self.degradation_steps) ** 0.780 * self.size
                    delta = (natural_dmax - self.pre_dmax) * (self.df)
                    natural_dmax = self.pre_dmax + delta
                    self.dmax = natural_dmax 
                    self.df = 1
                    
                elif action == "Adding_inhibitor":

                    
                    prob_retr1, prob_retr2, prob_retr3 = 0.30, 0.40, 0.25
                    mean_repair = state_interval * (prob_retr1 * 1 + prob_retr2 * 2 + prob_retr3 * 3)/(prob_retr1 + prob_retr2 + prob_retr3)
                    
                    repair_fraction = np.clip(rng.normal(mean_repair, 0.01) / corrosion_range,
                                              0.0, 1.0 - np.finfo(float).eps)
                    self.dmax *= 1.0 - repair_fraction
                    
                    
                    
                    
                    self.df = 0.1
                    natural_dmax = 0.164 * (self.degradation_steps) ** 0.780 * self.size
                    delta = (natural_dmax - self.pre_dmax) * (self.df)
                    natural_dmax = self.pre_dmax + delta                    
                    
                    self.dmax = self.dmax + delta
                    
                elif action == "Pigging":
                    
                    
                    prob_retr1, prob_retr2 = 0.25, 0.15
                    mean_repair = state_interval * (prob_retr1 * 1 + prob_retr2 * 2)/(prob_retr1 + prob_retr2)
                    repair_fraction = np.clip(rng.normal(mean_repair, 0.01) / corrosion_range,
                                              0.0, 1.0 - np.finfo(float).eps)
                    self.dmax *= 1.0 - repair_fraction
                    
                    
                    
                    
                    self.df = 0.5
                    natural_dmax = 0.164 * (self.degradation_steps) ** 0.780 * self.size
                    delta = (natural_dmax - self.pre_dmax) * (self.df)
                    natural_dmax = self.pre_dmax + delta
                    
                    self.dmax = self.dmax + delta

            elif self.degradation_law == "power_function_stochastic":
                state_interval = corrosion_range / 12

                if action == "Do_nothing":  

                    natural_dmax = 0.164 * (self.degradation_steps) ** 0.780 * self.size
                    delta = (natural_dmax - self.pre_dmax) * (self.df)
                    natural_dmax = self.pre_dmax + delta
                    self.dmax = natural_dmax 
                    self.df = 1
                    
                elif action == "Adding_inhibitor":

                    
                    prob_retr1, prob_retr2, prob_retr3 = 0.30, 0.40, 0.25
                    mean_repair = state_interval * (prob_retr1 * 1 + prob_retr2 * 2 + prob_retr3 * 3)/(prob_retr1 + prob_retr2 + prob_retr3)
                    repair_fraction = np.clip(rng.normal(mean_repair, 0.01) / corrosion_range,
                                              0.0, 1.0 - np.finfo(float).eps)
                    self.dmax *= 1.0 - repair_fraction
                    
                    
                    
                    
                    self.df = 0.1
                    natural_dmax = 0.164 * (self.degradation_steps) ** 0.780 * self.size
                    delta = (natural_dmax - self.pre_dmax) * (self.df)
                    natural_dmax = self.pre_dmax + delta                    
                    
                    self.dmax = self.dmax + delta
                    
                elif action == "Pigging":
                    
                    
                    prob_retr1, prob_retr2 = 0.25, 0.15
                    mean_repair = state_interval * (prob_retr1 * 1 + prob_retr2 * 2)/(prob_retr1 + prob_retr2)
                    repair_fraction = np.clip(rng.normal(mean_repair, 0.01) / corrosion_range,
                                              0.0, 1.0 - np.finfo(float).eps)
                    self.dmax *= 1.0 - repair_fraction
                    
                    
                    
                    
                    self.df = 0.5
                    natural_dmax = 0.164 * (self.degradation_steps) ** 0.780 * self.size
                    delta = (natural_dmax - self.pre_dmax) * (self.df)
                    natural_dmax = self.pre_dmax + delta
                    
                    self.dmax = self.dmax + delta
                    
                    if np.random.rand() < 0.5:  
                        delta = np.max((0.0, np.random.normal(0.02, 0.01)))
                    self.dmax = max(0, natural_dmax + delta)
            
            self.dmax = max(0, self.dmax)
            self.pre_dmax = self.dmax
            
            ratio, state = trans_dmax(self.dmax, corrosion_states, state_edges)
            self.state = state
            self.ratio = ratio

class DigitalAsset:
    def __init__(self, 
                 pinn_model, 
                 cpd, 
                 corrosion_states):

        self.pinn_model = pinn_model
        self.cpd = cpd                         
        self.state = None                      
        self.qois = None
        self.corrosion_states = corrosion_states

    def estimate_state(self, observations):
        
        
        dmax_pred = self.pinn_model.predict(observations) 

        ratio, state = trans_dmax(np.mean(dmax_pred), corrosion_states, state_edges)
        
        estimated_state = state                     
        print(f": {estimated_state}")
        
        return estimated_state

    def compute_qois(self):
        pass


class Planner:
    def __init__(self, 
                 corrosion_states, 
                 actions_list, 
                 gamma, 
                 transitions,
                 health_weight=3.5):

        self.corrosion_states = corrosion_states           
        self.actions_list = actions_list          
        self.gamma = gamma                        
        self.transitions = transitions            
        self.reward_table = None
        self.health_weight = health_weight

    def compute_reward(self, state, action):
         
        
        action = str(action)
        index = np.where(corrosion_states == state)[0][0]
        
        alpha_health = self.health_weight  
        alpha_action = 1.0  
        
        def r_health(State):
            b = 0.10
            a = 8
            
            damage_ratio = State / (len(self.corrosion_states) - 1)
            return -a * damage_ratio ** 2 * np.exp(b * State) / 5

        def r_action(action):
            


            if action == "Do_nothing":
                r_action = 0.5    
                    
            elif action == "Adding_inhibitor":
                r_action = -13      
                
            elif action == "Pigging":
                r_action = -3.5     
                
            elif action == "Replacement":
                r_action = -150     
                
            return r_action
        
        total_reward = alpha_health * r_health(index) + alpha_action * r_action(action)
        return total_reward


class Graph:
    def __init__(self, 
                 physical_asset, 
                 digital_asset, 
                 corrosion_states, 
                 actions_list, 
                 planner):

        self.physical_asset = physical_asset
        self.digital_asset = digital_asset
        self.corrosion_states = corrosion_states         
        self.actions_list = actions_list       
        self.planner = planner
        self.setup = GetSetup(corrosion_states, actions_list, planner.observation_matrix)
        self.dbn = DBN()    
        self.dbn_predict = DBN()  
        
        self.hist_var = None               
        self.hist_d_state_prob = None
        self.hist_actions_prob = None 

        
        self.pred_var = None
        self.pred_d_state_prob = None
        self.pred_actions_prob = None

        
        self.hist_observations = None
        
        self.time_index = None 
        
        self.state_names = self.corrosion_states.tolist()
        self.action_names = self.actions_list.tolist()  
          
    def assemble_graph(self, 
                       graph_structure,
                       cpd_list,
                       digital_subgraph,
                       list_cpd_subgraph,
                       plot=False):

        
        self.dbn.add_edges_from(graph_structure)
        print(f'\n: {self.dbn._nodes()}')
        
        print('\n:')
        for edge in self.dbn.get_intra_edges():
            print(f"  {edge[0]} -> {edge[1]}")
    
        print('\n:')
        for edge in self.dbn.get_inter_edges():
            print(f"  {edge[0]} -> {edge[1]}")

        for cpd_table in cpd_list:
            cpd_table.normalize()
            self.dbn.add_cpds(cpd_table)
        self.dbn.check_model()

        if plot:
            plt.rc('font', size=7)
            self.dbn.to_daft().render()
            plt.show()

        self.time_index = -1 
        
        
        
        self.dbn_predict.add_edges_from(digital_subgraph)
        for cpd_table in list_cpd_subgraph:
            cpd_table.normalize()
            self.dbn_predict.add_cpds(cpd_table)
        self.dbn_predict.check_model()
        
    def _find_permutation(self, state):
        
        State = float(state)
        
        index = np.where(corrosion_states == state)[0][0]
        
        if State >= self.corrosion_states.shape[0]:
            print("\nDT:")
            sys.exit()
        return index

    def compute_confidence_intervals(self, last_d_state_prob, ci_level=0.95):
        
        values = np.array([float(s) for s in self.state_names])
        order = np.argsort(values)
        sorted_vals = values[order]
        sorted_p = np.asarray(last_d_state_prob)[order]
        
        cdf = np.cumsum(sorted_p)
        cdf = cdf / cdf[-1]
        
        lo_idx = np.searchsorted(cdf, (1 - ci_level) / 2)
        hi_idx = np.searchsorted(cdf, 1 - (1 - ci_level) / 2)
        
        return sorted_vals[lo_idx], sorted_vals[hi_idx]

    def _posterior_mean(self, probabilities):
        
        probs = np.asarray(probabilities, dtype=float)
        values = np.array([float(s) for s in self.corrosion_states])
        return float((probs * values).sum())



            
    def _reference_action(self, true_state):
        
        state_index = self._find_permutation(true_state)
        age = min(max(0, self.physical_asset.degradation_steps), self.planner.max_age)
        df_index = self.planner._df_index(self.physical_asset.df)
        action = np.argmax(self.planner.q_values[state_index, age, df_index])
        return str(self.actions_list[action])


    def _q_action(self, estimated_state, posterior):
        observation = (
            self._find_permutation(estimated_state),
            max(0, self.physical_asset.degradation_steps),
            self.physical_asset.df,
            np.asarray(posterior, dtype=float),
        )
        action_index = self.planner.select_action(observation)
        return self.actions_list[action_index]


    def simulate(self, n_steps, n_samples, n_obs, observation_interval=1):
        
        for name, value in (('n_obs', n_obs), ('observation_interval', observation_interval),
                            ('n_samples', n_samples)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name}')
        if n_obs > len(self.physical_asset.X_regressors):
            raise ValueError('Nobs')

        if self.time_index == -1:
            
            self.time_index += 1
            self.physical_asset.update_physical_state(action="Do_nothing")
            
            
            obs, observed_ratios = self.physical_asset.get_observations(n_obs)
            d_state = self.digital_asset.estimate_state(obs)
            
            
            
            self.hist_observations = []
            for k in range(len(observed_ratios)):
                row = {
                    'time': self.time_index, 
                    'dmax': float(wall_thickness * (1 - observed_ratios[k])),
                }
                for j, name in enumerate(feature_names):
                    row[name] = float(obs[k, j])
                self.hist_observations.append(row)
            
            
            p_state = np.array('1.0')
            p_state_num = 1.0
            p_dmax = 0.0
            if self.physical_asset.dmax != 0:
                p_state = self.physical_asset.state
                p_state_num = self.physical_asset.ratio
                p_dmax = self.physical_asset.dmax
            
            
            last_d_state_prob = self.planner.observation_likelihood[:, self._find_permutation(d_state)].copy()
            last_d_state_prob /= last_d_state_prob.sum()
            
            current_action = str(self._q_action(d_state, last_d_state_prob))
            current_action_prob = np.eye(len(self.actions_list))[
                list(self.actions_list).index(current_action)]


            
            best_action = self._reference_action(p_state)

            r_mean, r_lower, r_upper = self._reward_stats(last_d_state_prob, current_action)
            ci_lo, ci_hi = self.compute_confidence_intervals(last_d_state_prob, ci_level=0.95)

            
            self.hist_var = {
                'p_dmax': p_dmax,
                'observed': True,
                'n_obs_used': n_obs,
                
                'p_state': p_state,                  
                'p_state_num': p_state_num,    
                     
                'd_state': self.corrosion_states[np.argmax(last_d_state_prob)],       
                'd_state_num': self._posterior_mean(last_d_state_prob),
                
                'Lower': [ci_lo],
                'Upper': [ci_hi],
                
                'current_action': current_action,    
                'best_action': best_action,  
                
                'reward': r_mean, 'reward_Lower': r_lower, 'reward_Upper': r_upper,
                        
            }
            self.hist_var = pd.DataFrame(self.hist_var, index=[0,])
            
            
            self.hist_d_state_prob = pd.DataFrame([last_d_state_prob.tolist()])
            self.hist_d_state_prob.columns = self.state_names

            self.hist_actions_prob = pd.DataFrame([current_action_prob.tolist()])
            self.hist_actions_prob.columns = self.action_names


        
        
        for i in range(n_steps):

            self.time_index += 1
            last_action = self.hist_var.current_action[self.time_index-1]
            last_d_state_prob = self.hist_d_state_prob.loc[self.time_index-1].to_numpy()

            
            transitions = self.setup.get_transitions(
                int(self.physical_asset.degradation_steps), float(self.physical_asset.df))
            combined = self.setup.get_combined_transitions(
                int(self.physical_asset.degradation_steps), float(self.physical_asset.df))
            n_states = len(self.corrosion_states)
            self.dbn.get_cpds(('D', 1)).values = combined.reshape(
                n_states, len(self.actions_list), n_states, n_states).copy()

            
            self.physical_asset.update_physical_state(action=last_action)
            observed = self.time_index % observation_interval == 0
            if observed:
                obs, observed_ratios = self.physical_asset.get_observations(n_obs)
                d_state = self.digital_asset.estimate_state(obs)

                
                for k in range(len(observed_ratios)):
                    row = {
                        'time': self.time_index,
                        'dmax': float(wall_thickness * (1 - observed_ratios[k])),
                    }
                    for j, name in enumerate(feature_names):
                        row[name] = float(obs[k, j])
                    self.hist_observations.append(row)
            
            
            p_state = np.array('1.0')
            p_state_num = 1.0
            p_dmax = 0.0
            if self.physical_asset.dmax != 0:
                p_state = self.physical_asset.state
                p_state_num = self.physical_asset.ratio
                p_dmax = self.physical_asset.dmax
            

            print(f': {i + 1}/{n_steps}, : {self.time_index}')
            
            action_index = list(self.actions_list).index(last_action)
            if observed:
                sampled = np.random.choice(self.corrosion_states.shape[0],
                                           size=n_samples,
                                           p=last_d_state_prob)
                sampled_prior = np.bincount(
                    sampled, minlength=len(self.corrosion_states)
                ) / n_samples
                last_d_state_prob = _update_posterior(
                    sampled_prior, self._find_permutation(d_state), action_index,
                    transitions, self.planner.observation_likelihood)
            else:
                
                last_d_state_prob = transitions[action_index] @ last_d_state_prob
                last_d_state_prob /= last_d_state_prob.sum()
                d_state = self.corrosion_states[int(np.argmax(last_d_state_prob))]

            
            current_action = str(self._q_action(d_state, last_d_state_prob))
            current_action_prob = np.eye(len(self.actions_list))[
                list(self.actions_list).index(current_action)]

            
            best_action = self._reference_action(p_state)

            r_mean, r_lower, r_upper = self._reward_stats(last_d_state_prob, current_action)
            ci_lo, ci_hi = self.compute_confidence_intervals(last_d_state_prob, ci_level=0.95)
            
            new_var = {
                'p_dmax': p_dmax,
                'observed': observed,
                'n_obs_used': n_obs if observed else 0,
                
                'p_state_num': p_state_num,         
                'p_state': p_state,       
                      
                'd_state': self.corrosion_states[np.argmax(last_d_state_prob)],    
                'd_state_num': self._posterior_mean(last_d_state_prob),       
                
                'Lower': [ci_lo],
                'Upper': [ci_hi],
                   
                'current_action': current_action,   
                'best_action': best_action,       
                
                'reward': r_mean, 'reward_Lower': r_lower, 'reward_Upper': r_upper,  
            }
            self.hist_var = pd.concat([self.hist_var, pd.DataFrame(new_var, index=[0,])], 
                                     ignore_index=True)
            
            new_var = pd.DataFrame([last_d_state_prob.tolist()])
            new_var.columns = self.state_names
            self.hist_d_state_prob = pd.concat([self.hist_d_state_prob, new_var], 
                                             ignore_index=True)

            new_var = pd.DataFrame([current_action_prob.tolist()])
            new_var.columns = self.action_names
            self.hist_actions_prob = pd.concat([self.hist_actions_prob, new_var], 
                                             ignore_index=True)


    def predict(self, n_steps, n_samples):
        print(f'\n=== () ===')
        print(f': {self.time_index},  {n_steps} ')

        prob_D = np.zeros((n_steps + 1, self.corrosion_states.shape[0]))
        prob_U = np.zeros((n_steps + 1, self.actions_list.shape[0]))
        belief = self.hist_d_state_prob.loc[self.time_index].to_numpy(dtype=float)
        belief = belief / belief.sum()
        
        age = max(0, int(self.physical_asset.degradation_steps))
        df_value = float(self.physical_asset.df)

        for step in range(n_steps + 1):
            prob_D[step] = belief
            observation = (int(np.argmax(belief)), age, df_value, belief)
            action_index = self.planner.select_action(observation)
            action = str(self.actions_list[action_index])
            prob_U[step, action_index] = 1.0

            if step == n_steps:
                break

            
            transitions = self.setup.get_transitions(age, df_value)
            self.dbn_predict.get_cpds(('D', 1)).values = transitions.transpose(1, 0, 2).copy()
            
            cpd_D_0 = TabularCPD(('D', 0), len(self.corrosion_states), belief[:, None])
            sampled = self.dbn_predict.simulate(
                n_time_slices=2,
                n_samples=n_samples,
                do={('U', 0): action_index},
                virtual_evidence=[cpd_D_0],
                seed=int(np.random.randint(0, 2**31 - 1)),
                show_progress=False
            )
            counts = np.bincount(sampled[('D', 1)].to_numpy(dtype=int),
                                 minlength=len(self.corrosion_states))
            belief = counts / counts.sum()
            age = 0 if action == 'Replacement' else age + 1
            df_value = float(self.planner.df_after_action[action])

        self.pred_d_state_prob = pd.DataFrame(prob_D.tolist(), columns=self.state_names)
        self.pred_actions_prob = pd.DataFrame(prob_U.tolist(), columns=self.action_names)

        values = np.array([float(s) for s in self.state_names])
        rows = []
        for i in range(n_steps + 1):
            action_i = str(self.actions_list[np.argmax(prob_U[i])])
            r_mean, r_lo, r_hi = self._reward_stats(prob_D[i], action_i)
            ci_lo, ci_hi = self.compute_confidence_intervals(prob_D[i], ci_level=0.95)
            rows.append({
                'd_state': self.corrosion_states[np.argmax(prob_D[i])],
                'd_state_num': float((prob_D[i] * values).sum()),
                'current_action': action_i,
                'Lower': ci_lo,
                'Upper': ci_hi,
                'reward': r_mean, 'reward_Lower': r_lo, 'reward_Upper': r_hi,
            })
        self.pred_var = pd.DataFrame(rows)


    def _reward_stats(self, prob, action):
        
        action_idx = list(self.actions_list).index(action)
        r_per_state = self.planner.reward_table[:, action_idx]   
        prob = np.asarray(prob, dtype=float)
        
        
        r_mean = float((prob * r_per_state).sum())
        
        
        order = np.argsort(r_per_state)
        r_sorted = r_per_state[order]
        p_sorted = prob[order]
        cdf = np.cumsum(p_sorted)
        cdf = cdf / cdf[-1]
        lo = float(r_sorted[np.searchsorted(cdf, 0.025)])
        hi = float(r_sorted[np.searchsorted(cdf, 0.975)])
        return r_mean, lo, hi

class PINNModel:
    
    def __init__(self, model_paths):

        self.model_paths = model_paths
        self.models = []
        self.nominal_dmax = 1.92

        for path in model_paths:
            if Path(path).exists():

                model = torch.load(path, map_location=device, weights_only=False)
                model.eval()
                self.models.append(model)
        if not self.models:
            print(": ")
    
    def predict(self, X):   
        n_samples = X.shape[0]
        
        if len(self.models) > 0:

            predictions = []
            for model in self.models:
                model.eval()
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                pred = model.predict(X_tensor)
                pred = pred.cpu().detach().numpy().flatten()
                predictions.append(pred)
            
            
            weights = [0.0070652, 0.92255964, 0.07037516]
            ensemble_pred = np.average(predictions, axis=0, weights=weights) 
            
            
            ensemble_pred = ensemble_pred * self.nominal_dmax
            pred_std = np.std(predictions, axis=0)
            
            dmax_pred = ensemble_pred
            
        else:
            
            dmax_pred = self._idealized_prediction(X)
            pred_std = np.full(n_samples, 0.1)
              
        return dmax_pred
    

class ProbabilityObservationModel:
    
    def __init__(self, 
                 pinn_model, 
                 wall_thickness, 
                 X_regressors, 
                 labels_ratio):

        self.pinn_model = pinn_model
        self.wall_thickness = wall_thickness
        self.X_regressors = X_regressors
        self.labels_ratio = labels_ratio

    def build_observation_matrix(self, corrosion_states, corrosion_edges, real_dmax_values):
        
        save_path = Path(__file__).resolve().parent / 'DATA' / 'confusion_matrix.npy'
        if os.path.exists(save_path):
            print(f"Loading observation matrix from {save_path}...")
            self.obs_prob_matrix = np.load(save_path)

        else:
    
            Ns = len(corrosion_states)
            confusion_matrix = np.zeros((Ns, Ns))
            
            
            state_to_idx = {state: idx for idx, state in enumerate(corrosion_states)}
            
            if len(self.pinn_model.models) > 0:
    
                
                all_predictions = self.pinn_model.predict(self.X_regressors)

                
                
                
                
                
                
                
                
                
                
                

                
                for i, (true_dmax, pred_dmax) in enumerate(zip(real_dmax_values, all_predictions)):
                    
                    _, true_state = trans_dmax(true_dmax, corrosion_states, state_edges)
                    
                    _, pred_state = trans_dmax(pred_dmax, corrosion_states, state_edges)

                    if true_state in state_to_idx and pred_state in state_to_idx:
                        true_idx = state_to_idx[true_state]
                        pred_idx = state_to_idx[pred_state]
                        confusion_matrix[true_idx,pred_idx] += 1
                
                for j in range(Ns):
                    col_sum = confusion_matrix[:, j].sum()
                    if col_sum < 3:
                        state_name = corrosion_states[j]
                        confusion_matrix[:, j] = 0
                        for i in range(Ns):
                            distance = abs(i - j)
                            noise_level = 0.4
                            prob = np.exp(-0.5 * (distance / noise_level) ** 2)
                            confusion_matrix[i, j] = prob
            
            confusion_matrix = confusion_matrix + 1e-6
            self.obs_prob_matrix = normalize_cpd(confusion_matrix)
            
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, self.obs_prob_matrix)
        
        
        return self.obs_prob_matrix

    def build_observation_matrix1(self, corrosion_states, corrosion_edges, real_dmax_values):
        
        save_path = Path(__file__).resolve().parent / 'DATA' / 'confusion_matrix.npy'
        Ns = len(corrosion_states)
        counts = np.zeros((Ns, Ns))
        state_to_idx = {state: idx for idx, state in enumerate(corrosion_states)}
        if len(self.pinn_model.models) == 0:
            raise ValueError(" PINN ")

        predictions = np.asarray(self.pinn_model.predict(self.X_regressors)).reshape(-1)
        real_dmax_values = np.asarray(real_dmax_values).reshape(-1)
        if len(predictions) == 0 or len(predictions) != len(real_dmax_values):
            raise ValueError("")
        if not (np.isfinite(predictions).all() and np.isfinite(real_dmax_values).all()):
            raise ValueError("")
        pred_indices = []
        
        for true_dmax, pred_dmax in zip(real_dmax_values, predictions):
            _, true_state = trans_dmax(true_dmax, corrosion_states, corrosion_edges)
            _, pred_state = trans_dmax(pred_dmax, corrosion_states, corrosion_edges)
            pred_idx = state_to_idx[pred_state]
            pred_indices.append(pred_idx)
            counts[state_to_idx[true_state], pred_idx] += 1

        pred_indices = np.array(pred_indices)
        errors = real_dmax_values - predictions
        depths = (1 - (corrosion_edges[:-1] + corrosion_edges[1:])/2)*self.wall_thickness
        col_sums = counts.sum(axis=0)
        valid = np.flatnonzero(col_sums > 0)
        confusion_matrix = counts.copy()
        strength = 1.0  

        for j in np.flatnonzero(col_sums < 3):
            neighbors = valid[valid != j]
            if len(neighbors) == 0:
                continue
            left = neighbors[neighbors < j]
            right = neighbors[neighbors > j]
            if len(left) > 0 and len(right) > 0:
                neighbors = np.array([left[-1], right[0]])
                a, b = neighbors
                weights = np.array([depths[b] - depths[j], depths[j] - depths[a]])
            else:
                neighbors = neighbors[np.argsort(abs(depths[neighbors] - depths[j]))[:2]]
                weights = 1/abs(depths[neighbors] - depths[j])
            weights = weights/weights.sum()

            
            prior = np.zeros(Ns)
            for k, weight in zip(neighbors, weights):
                local_errors = errors[pred_indices == k]
                for error in local_errors:
                    depth = np.clip(depths[j] + error, 0, self.wall_thickness)
                    _, state = trans_dmax(depth, corrosion_states, corrosion_edges)
                    prior[state_to_idx[state]] += weight/len(local_errors)
            confusion_matrix[:, j] += strength*prior

        self.obs_prob_matrix = normalize_cpd(confusion_matrix + 1e-6)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, self.obs_prob_matrix)
        return self.obs_prob_matrix

def analyze_annual_cost(graph_frame, n_years=50):
    



    
    
    actions = graph_frame.hist_var['current_action'].iloc[:n_years].to_numpy()
    
    total_cost = 0
    action_costs = {
        "Do_nothing": 0,
        "Adding_inhibitor": 13,
        "Pigging": 3.5,
        "Replacement": 150
    }
    for action in actions:
        total_cost += action_costs.get(action, 0)
    
    annual_cost = total_cost / n_years
    
    action_counts = pd.Series(actions).value_counts()
    
    
    traditional_asset = PhysicalAsset(
        degradation_law="power_function",
        X_regressors=graph_frame.physical_asset.X_regressors,
        wall_thickness=graph_frame.physical_asset.wall_thickness
    )
    traditional_asset.generate_observations()
    traditional_asset.update_physical_state(action="Do_nothing")  
    traditional_total_cost = 0
    replacement_threshold = 0.2  
    replacement_cost = 150
    traditional_dmax_values = []
    for step in range(n_years):
        
        action = "Replacement" if traditional_asset.ratio < replacement_threshold else "Do_nothing"
        traditional_asset.update_physical_state(action=action)
        traditional_dmax_values.append(traditional_asset.dmax)
        if action == "Replacement":
            traditional_total_cost += replacement_cost
    
    traditional_annual_cost = traditional_total_cost / n_years
    
    
    print(f"\n===  ===")
    print(f": {n_years}")
    print(f": {total_cost:.2f}")
    print(f": {annual_cost:.2f}")
    print(f"\n:")
    for action, count in action_counts.items():
        print(f"  {action}: {count} ({count/n_years:.2f})")
    
    print(f"\n===  ===")
    print(f"(<20%): {traditional_annual_cost:.2f}")
    print(f": {annual_cost:.2f}")
    print(f": {(traditional_annual_cost - annual_cost)/traditional_annual_cost * 100:.1f}%" 
          if traditional_annual_cost > 0 else "0")
    
    return annual_cost, traditional_dmax_values


def save(excel_file):
    
    action_to_value = {
        'Do_nothing': 0,
        'Pigging': 1,
        'Adding_inhibitor': 2,
        'Replacement': 3
    }
    
    n_sim = len(graph_frame.hist_var)               
    t_sim = list(range(n_sim))                      
    
    n_pred = len(graph_frame.pred_var)              
    t_pred = list(range(n_sim-1, n_sim -1 + n_pred))     
    
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        
        sim_combined = graph_frame.hist_var.copy()
        sim_combined['current_action_value'] = sim_combined['current_action'].map(action_to_value)
        sim_combined['best_action_value'] = sim_combined['best_action'].map(action_to_value)
        sim_combined.index = t_sim
        sim_combined.index.name = 'time'
        sim_combined.to_excel(writer, sheet_name='Simulation', index=True)
        
        df = graph_frame.hist_d_state_prob.copy()
        df.index = t_sim; df.index.name = 'time'
        df.to_excel(writer, sheet_name='Simulation_State_Prob', index=True)
        
        df = graph_frame.hist_actions_prob.copy()
        df.index = t_sim; df.index.name = 'time'
        df.to_excel(writer, sheet_name='Simulation_Action_Prob', index=True)
        
        pd.DataFrame(graph_frame.hist_observations).to_excel(
            writer, sheet_name='Simulation_Observations', index=False)
        
        
        pred_combined = graph_frame.pred_var.copy()
        pred_combined['current_action_value'] = pred_combined['current_action'].map(action_to_value)
        pred_combined.index = t_pred
        pred_combined.index.name = 'time'
        pred_combined.to_excel(writer, sheet_name='Prediction', index=True)
        
        df = graph_frame.pred_d_state_prob.copy()
        df.index = t_pred; df.index.name = 'time'
        df.to_excel(writer, sheet_name='Prediction_State_Prob', index=True)
        
        df = graph_frame.pred_actions_prob.copy()
        df.index = t_pred; df.index.name = 'time'
        df.to_excel(writer, sheet_name='Prediction_Action_Prob', index=True)
        
        
        pd.DataFrame({'dmax': traditional_dmax_values}).to_excel(
            writer, sheet_name='Traditional_Strategy', index=False)

        
        pd.DataFrame([
            {'setting': key, 'value': value}
            for key, value in graph_frame.experiment_settings.items()
        ]).to_excel(writer, sheet_name='Experiment_Settings', index=False)
        pd.DataFrame(graph_frame.experiment_metrics).to_excel(
            writer, sheet_name='Experiment_Metrics', index=False)



def load_reference_q_learning(health_weight, transitions, pinn_model,
                              observation_matrix, output_dir):
    
    reward_model = qlearning.RewardModel(
        sys.modules[__name__], health_weight, transitions
    )
    reward_model.build_reward_table()

    def text(value):
        return str(value).removesuffix('.0').replace('.', 'p')

    action_codes = {
        'Do_nothing': 'DN',
        'Adding_inhibitor': 'AI',
        'Pigging': 'PG',
        'Replacement': 'RP',
    }
    actions = '-'.join(action_codes[str(action)] for action in possible_actions)
    rewards = '-'.join(text(value) for value in reward_model.reward_table[0])
    prefix = (
        f'qbasic_a{len(possible_actions)}_{actions}_h{text(health_weight)}'
        f'_r{rewards}_*_s{random_seed}o2_g{text(gamma)}_'
    )
    candidates = sorted(Path(output_dir).glob(prefix + '*.npz'))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f'Q{prefix!r}{len(candidates)}'
            'q_tablesQ'
        )

    df_after_action = {
        'Do_nothing': 1.0,
        'Adding_inhibitor': 0.1,
        'Pigging': 0.5,
        'Replacement': 1.0,
    }
    df_values = sorted({df_after_action[str(action)] for action in possible_actions})
    planner = qlearning.QLearningPlanner(
        len(corrosion_states), possible_actions, df_values,
        max_age=max_age, gamma=gamma, learning_rate=learning_rate
    )
    planner.reward_table = reward_model.reward_table.copy()
    planner.df_after_action = df_after_action
    planner.transitions = np.asarray(transitions, dtype=float)
    planner.transition_model = GetSetup(
        corrosion_states, possible_actions, observation_matrix
    ).transition_model
    planner.observation_matrix = np.asarray(observation_matrix, dtype=float).copy()
    planner.q_table_file = candidates[0]

    with np.load(planner.q_table_file, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['corrosion_states'], corrosion_states)
        np.testing.assert_array_equal(saved['actions'], possible_actions)
        np.testing.assert_allclose(saved['df_categories'], planner.df_values)
        np.testing.assert_allclose(saved['observation_matrix'], planner.observation_matrix,
                                   rtol=0, atol=1e-12)
        if int(saved['seed']) != random_seed:
            raise ValueError('Q')
        q_values = saved['q_values'].copy()
        visits = saved['visits'].copy()
        if q_values.shape != planner.q_values.shape or visits.shape != planner.visits.shape:
            raise ValueError('Q')
        if not np.isfinite(q_values).all() or np.any(visits < 0):
            raise ValueError('Q')
        planner.final_episode = int(saved['final_episode'])
        planner.final_validation_return = float(saved['final_validation_return'])
        planner.completed_episodes = int(saved['completed_episodes'])
        planner.stop_reason = str(saved['stop_reason'])
        planner.observation_likelihood = saved['observation_likelihood'].copy()
        planner.observation_calibration_counts = saved[
            'observation_calibration_counts'
        ].copy()

    q_values.setflags(write=False)
    visits.setflags(write=False)
    planner.q_values = q_values
    planner.visits = visits
    print(f'Q{planner.q_table_file}', flush=True)
    return planner


def compute_experiment_metrics(graph):
    
    history = graph.hist_var
    truth_ratio = history['p_state_num'].to_numpy(dtype=float)
    estimate_ratio = history['d_state_num'].to_numpy(dtype=float)
    lower = history['Lower'].to_numpy(dtype=float)
    upper = history['Upper'].to_numpy(dtype=float)
    error = estimate_ratio - truth_ratio

    true_discrete = []
    for ratio in truth_ratio:
        _, state = trans_dmax(
            wall_thickness * (1.0 - ratio), corrosion_states, state_edges
        )
        true_discrete.append(float(state))
    true_discrete = np.asarray(true_discrete)
    discrete_covered = (lower <= true_discrete) & (true_discrete <= upper)
    continuous_covered = (lower <= truth_ratio) & (truth_ratio <= upper)

    observed = history['observed'].to_numpy(dtype=bool)
    actions = history['current_action'].iloc[:-1].astype(str).to_numpy()
    action_costs = {
        'Do_nothing': 0.0,
        'Adding_inhibitor': 13.0,
        'Pigging': 3.5,
        'Replacement': 150.0,
    }
    action_counts = {
        action: int(np.sum(actions == action))
        for action in possible_actions
    }
    realized_returns = []
    for step, action in enumerate(actions):
        true_state = str(true_discrete[step + 1])
        state_index = graph._find_permutation(true_state)
        action_index = list(possible_actions).index(action)
        realized_returns.append(
            (gamma ** step) * graph.planner.reward_table[state_index, action_index]
        )

    rows = [
        {'metric': 'continuous_state_RMSE', 'value': float(np.sqrt(np.mean(error ** 2))),
         'unit': 'remaining-wall-thickness ratio', 'scope': '51 online states'},
        {'metric': 'continuous_state_MAE', 'value': float(np.mean(np.abs(error))),
         'unit': 'remaining-wall-thickness ratio', 'scope': '51 online states'},
        {'metric': 'discrete_state_95pct_coverage',
         'value': float(np.mean(discrete_covered)), 'unit': 'fraction',
         'scope': 'true state after the configured damage-state quantization'},
        {'metric': 'continuous_state_95pct_coverage',
         'value': float(np.mean(continuous_covered)), 'unit': 'fraction',
         'scope': 'continuous truth against the discrete posterior interval'},
        {'metric': 'mean_posterior_interval_width',
         'value': float(np.mean(upper - lower)),
         'unit': 'remaining-wall-thickness ratio', 'scope': '51 online states'},
        {'metric': 'observation_time_steps',
         'value': int(observed.sum()), 'unit': 'steps', 'scope': 'including step 0'},
        {'metric': 'observation_records',
         'value': int(history['n_obs_used'].sum()), 'unit': 'database records',
         'scope': 'actual records passed to PIDM'},
        {'metric': 'missing_observation_time_steps',
         'value': int((~observed).sum()), 'unit': 'steps', 'scope': 'online steps'},
        {'metric': 'realized_discounted_return',
         'value': float(np.sum(realized_returns)), 'unit': 'model reward units',
         'scope': '50 executed actions and subsequent simulated states'},
        {'metric': 'total_maintenance_cost',
         'value': float(sum(action_costs[a] for a in actions)),
         'unit': 'model cost units', 'scope': '50 executed actions'},
    ]
    rows.extend(
        {'metric': f'actions_{action}', 'value': count, 'unit': 'actions',
         'scope': '50 executed actions'}
        for action, count in action_counts.items()
    )
    return rows


def run_experiment(health_weight, n_obs=1, observation_interval=1, *, save_results=True):
    global wall_thickness, feature_names, X_regressors, y_dmax, labels_ratio
    global corrosion_range, state_edges, corrosion_states, possible_actions
    global graph_frame, traditional_dmax_values

    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    experiment_dir = Path(__file__).resolve().parent
    project_dir = Path(__file__).resolve().parents[2]
    health_name = str(health_weight).replace('.', 'p')
    case_name = (f'Nobs_{n_obs}' if experiment_kind == 'Nobs'
                 else f'interval_{observation_interval}')
        
    
    wall_thickness = 11   
    
    
    path_data = Path(__file__).resolve().parent / 'DATA' 
    xlsx_path = path_data / 'DATA.xlsx'
    data_df = pd.read_excel(xlsx_path)

    
    feature_columns = data_df.columns.drop(['dmax'])
    feature_names = list(feature_columns)
    X_regressors = data_df[feature_columns].to_numpy()
    y_dmax = data_df['dmax'].to_numpy() * 1.92

    
    labels_ratio = (wall_thickness - y_dmax) / wall_thickness
    
    
    min_residual = np.min(labels_ratio)
    max_residual = np.max(labels_ratio)
    
    max_corrosion = np.max(y_dmax)
    min_corrosion = np.min(y_dmax)
    
    
    corrosion_range = max_corrosion - min_corrosion
    state_range = max_residual - min_residual
    
    corrosion_edges = np.append(0.000, np.round(np.linspace(min_corrosion, max_corrosion, 13),3))
    state_edges = np.append(1.00, np.round(np.linspace(min_residual, max_residual, 13),3))
    state_edges = np.sort(state_edges)[::-1]
    
    corrosion_ranges = []
    corrosion_states = []
    s = []
    for i in range(12): 
        start = min_corrosion + i * corrosion_range / 12
        end = min_corrosion + (i + 1) * corrosion_range / 12
        start_state = min_residual + i * state_range / 12
        end_state = min_residual + (i + 1) * state_range / 12      
        if i == 0:
            corrosion_ranges.append((0,0))
            corrosion_states.append(["1"])
        corrosion_ranges.append((np.round(start,3), np.round(end,3)))
        
        s.append((np.round(start_state,3), np.round(end_state,3)))
        s_mid = np.round ( (start_state + end_state)/2 ,3 )
        corrosion_states = np.append(corrosion_states, f"{s_mid}")
    corrosion_states = np.array([str(x) for x in sorted([float(x) for x in corrosion_states], reverse=True)])
    
    
    possible_actions = np.array(["Do_nothing", "Adding_inhibitor", "Pigging", "Replacement"])

    
    model_paths = [path_data / f"best_model_fold{i+1}.pt" for i in range(3)]
    pinn_model = PINNModel(model_paths)
    
    
    physical_asset = PhysicalAsset(
        degradation_law="power_function",
        X_regressors=X_regressors,
        wall_thickness=wall_thickness
    )
    physical_asset.generate_observations()
    
    
    obs_prob_matrix = np.load(path_data / 'confusion_matrix.npy', allow_pickle=False)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    
    
    
    
    

    digital_asset = DigitalAsset(
        pinn_model=pinn_model,
        cpd=obs_prob_matrix,
        corrosion_states=corrosion_states
    )

    
    setup_frame = GetSetup(
        corrosion_states=corrosion_states,
        actions_list=possible_actions,
        conf_mat_dt=obs_prob_matrix
    )
    
    
    transitions = setup_frame.get_transitions(age=0, df_value=1.0)

    
    planner_frame = load_reference_q_learning(
        health_weight, transitions, pinn_model, obs_prob_matrix,
        project_dir / 'q_tables'
    )

    
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        planner_frame.observation_likelihood, planner_frame.observation_calibration_counts = \
            qlearning._build_observation_likelihood(
                sys.modules[__name__], pinn_model, n_obs
            )
    finally:
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
    cpd_d_to_u = planner_frame.action_probabilities_by_damage().T

    
    graph_structure, digital_subgraph, list_cpd_graph, list_cpd_subgraph = \
        GetSetup(corrosion_states, possible_actions,
                 planner_frame.observation_matrix).get_graph(cpd_d_to_u, transitions)

    
    graph_frame = Graph(
        physical_asset, 
        digital_asset, 
        corrosion_states, 
        possible_actions, 
        planner_frame
    )
    
    
    graph_frame.assemble_graph(
        graph_structure, list_cpd_graph, 
        digital_subgraph, list_cpd_subgraph, 
        plot=False
    )
    
    
    print(f'\n{experiment_kind}Nobs={n_obs}={observation_interval}')
    graph_frame.simulate(
        n_steps=simulation_steps, n_samples=filter_samples, n_obs=n_obs,
        observation_interval=observation_interval
    )
    graph_frame.predict(n_steps=prediction_steps, n_samples=prediction_samples)
    graph_frame.experiment_settings = {
        'experiment': experiment_kind,
        'n_obs': n_obs,
        'observation_interval': observation_interval,
        'initial_observation': True,
        'observation_schedule': 'global decision step % observation_interval == 0',
        'health_weight': health_weight,
        'seed': random_seed,
        'simulation_steps': simulation_steps,
        'prediction_steps': prediction_steps,
        'filter_samples': filter_samples,
        'prediction_samples': prediction_samples,
        'reference_q_training_nobs': 2,
        'q_table': str(planner_frame.q_table_file),
        'q_retrained': False,
        'likelihood_nobs': n_obs,
        'decisions_between_observations': True,
        'interval_interpretation': 'discrete posterior equal-tail 95% range',
        'evaluation_scope': 'simulated closed loop using the existing observation database',
        'nobs_interpretation': 'database records, not sensor channels',
    }
    graph_frame.experiment_metrics = compute_experiment_metrics(graph_frame)
    if not save_results:
        return graph_frame
    
    
    output_dir = experiment_dir / 'simulation_results' / case_name / f'health_{health_name}'
    output_dir.mkdir(parents=True, exist_ok=True)    
    

    excel_file = output_dir / 'digital_twin_simulation_results_4.xlsx'
    annual_cost, traditional_dmax_values = analyze_annual_cost(graph_frame, n_years=simulation_steps)
    save(excel_file)
    
    
    plotting_frame = Plot(excel_file, experiment_dir / 'plotters_4' / case_name / f'health_{health_name}', wall_thickness=wall_thickness)
    plotting_frame.plot_all_together()      
    
    
    plotting_frame.plot_sensor_dmax()               
    plot_q_learning_reward(
        planner_frame.q_table_file,
        experiment_dir / 'plotters_4' / case_name / f'health_{health_name}'
    )
    return graph_frame




experiment_values = (1, 2, 3)
experiment_kind = 'temporal'
health_weights = (1.5, 3)
random_seed = 42

gamma = 0.95
learning_rate = 0.1
max_episodes = 100000
training_rollout_years = 70
max_age = 70
epsilon_start = 1.0
epsilon_end = 0.05
checkpoint_interval = 100
checkpoint_episodes = 64
stable_checkpoints = 5
stability_tolerance = 0.01
validation_samples = 300
evaluation_years = (70, 300)
evaluation_episodes = 200

simulation_steps = 50
prediction_steps = 20
filter_samples = 300
prediction_samples = 1000


if __name__ == '__main__':
    for health_weight in health_weights:
        for value in experiment_values:
            if experiment_kind == 'Nobs':
                n_obs, observation_interval = value, 1
            else:
                n_obs, observation_interval = 1, value
            print(
                f'\\n=== {experiment_kind}health={health_weight}, '
                f'Nobs={n_obs}, interval={observation_interval}, seed={random_seed} ===',
                flush=True
            )
            run_experiment(
                health_weight,
                n_obs=n_obs,
                observation_interval=observation_interval
            )
