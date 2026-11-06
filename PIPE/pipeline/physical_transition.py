





import numpy as np


class PhysicalTransition:
    def __init__(self, asset_factory, X_regressors, wall_thickness,
                 corrosion_states, actions_list):
        
        self.Ns = len(corrosion_states)
        self.actions_list = tuple(str(action) for action in actions_list)
        p_nothing = 0.33  
        action_transitions = {
            'Do_nothing': self._get_low_diag_transition(p_nothing),
            'Adding_inhibitor': self._get_inhibitor_transition(p_nothing),
            'Pigging': self._get_pigging_transition(p_nothing),
            'Replacement': self._get_replacement_transition(),
        }
        self.transitions = np.stack([action_transitions[action]
                                     for action in self.actions_list], axis=0)


    def _get_low_diag_transition(self, prob_adv):
        
        n_states = self.Ns
        trans = np.zeros((n_states, n_states))
        
        
        trans[0, 0] = 1 - prob_adv  
        trans[1, 0] = prob_adv      
        
        
        for i in range(1, n_states):
            if i == n_states - 1:   
                trans[i, i] = 1.0       
            else:
                trans[i, i] = 1 - prob_adv  
                trans[i + 1, i] = prob_adv  
        
        return trans

    def _get_maintenance_transition1(self, prob_retr1, prob_retr2, prob_retr3, deteriorate_rate=0.05):

        n_states = self.Ns
        trans = np.zeros((n_states, n_states))
        repair_rate = prob_retr1 + prob_retr2 + prob_retr3
        
        for i in range(n_states):
            
            if i == 0:  
                trans[0, i] = 1.0 - deteriorate_rate
                trans[1, i] = deteriorate_rate
                
            elif i == n_states - 1:  
                
                trans[i, i] = 1.0 - repair_rate  
                
                trans[i-1, i] = prob_retr1
                trans[i-2, i] = prob_retr2
                trans[i-3, i] = prob_retr3

            else:  
                remain_prob = 1.0 - repair_rate - deteriorate_rate
                trans[i, i] = remain_prob        
                trans[i+1, i] = deteriorate_rate 
                
                if i >= 3:
                    trans[i-1, i] = prob_retr1
                    trans[i-2, i] = prob_retr2
                    trans[i-3, i] = prob_retr3
                elif i == 2:
                    trans[i-1, i] = repair_rate - prob_retr2
                    trans[i-2, i] = prob_retr2
                elif i == 1:
                    trans[i-1, i] = repair_rate   
                    
        return trans

    def _get_maintenance_transition2(self, prob_retr1, prob_retr2, deteriorate_rate=0.05):

        n_states = self.Ns
        trans = np.zeros((n_states, n_states))
        repair_rate = prob_retr1 + prob_retr2
        
        for i in range(n_states):
            
            if i == 0:  
                trans[0, i] = 1.0 - deteriorate_rate
                trans[1, i] = deteriorate_rate
                
            elif i == n_states - 1:  
                
                trans[i, i] = 1.0 - repair_rate  
                
                trans[i-1, i] = prob_retr1                
                trans[i-2, i] = prob_retr2

            else:  
                remain_prob = 1.0 - repair_rate - deteriorate_rate
                trans[i, i] = remain_prob        
                trans[i+1, i] = deteriorate_rate 
                
                if i >= 2:
                    trans[i-1, i] = prob_retr1
                    trans[i-2, i] = prob_retr2
                elif i == 1: 
                    trans[i-1, i] = repair_rate

        return trans

    def _get_inhibitor_transition(self, prob_adv):

        return self._get_maintenance_transition1(
            prob_retr1=0.30,    
            prob_retr2=0.10,    
            prob_retr3=0.05,    
            deteriorate_rate = prob_adv * 0.1
            )

    def _get_pigging_transition(self, prob_adv):

        return self._get_maintenance_transition2(            
            prob_retr1=0.15,    
            prob_retr2=0.05,    
            deteriorate_rate = prob_adv * 0.5
            )

    def _get_replacement_transition(self):
        n_states = self.Ns
        trans = np.zeros((n_states, n_states))
        trans[0, :] = 1.0  
        return trans

    def matrices(self, age, df_value):
        
        return self.transitions.copy()
