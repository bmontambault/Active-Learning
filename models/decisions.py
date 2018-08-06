import numpy as np


class Deterministic:
    
    init_params = []
    def __call__(self, utility):
        p = np.array([1 if u == max(utility) else 0 for u in utility])
        return p / p.sum()


class Propto:
    
    init_params = []
    def __call__(self, utility):
        return utility / np.sum(utility)
    
    
class Softmax:
    
    init_params = [1.]
    bounds = [(.001, 10)]
    def __call__(self, utility, temperature):
        centered_utility = utility - np.nanmax(utility)
        exp_u = np.exp(centered_utility / temperature)
        return exp_u / np.nansum(exp_u)
    
    
class PhaseSoftmax:
    
    init_params = [1., 12]
    bounds = [(.001, 10), (1, 25)]
    
    def __init__(self, choices, best_action, trial):
        self.choices = choices
        self.best_action = best_action
        self.trial = trial
        
    def __call__(self, utility, temperature, phase):
        if self.trial + 1 > phase:
            utility = np.array([np.exp(-abs(self.best_action - x)) for x in self.choices])
            
        centered_utility = utility - np.nanmax(utility)
        exp_u = np.exp(centered_utility / temperature).ravel()
        return exp_u / np.nansum(exp_u)
    
    
class PhaseSoftmax2:
    
    init_params = [1., 12]
    bounds = [(.001, 10), (1, 25)]
    
    def __init__(self, choices, action_1, trial):
        self.choices = choices
        self.action_1 = action_1
        self.trial = trial
        
    def __call__(self, utility, temperature, phase):
        if self.trial + 1 > phase:
            utility = np.array([np.exp(-abs(self.action_1 - x)) for x in self.choices])
        
        centered_utility = utility - np.nanmax(utility)
        exp_u = np.exp(centered_utility / temperature).ravel()
        return exp_u / np.nansum(exp_u)
    
    
class StaySoftmax:
    
    init_params = [1., .5]
    bounds = [(.001, 10), (.001, .999)]
    
    def __init__(self, action_1):
        self.action_1 = action_1
    
    def __call__(self, utility, temperature, stay):
        centered_utility = utility - np.nanmax(utility)
        exp_u = np.exp(centered_utility / temperature)
        u = np.array([ui * stay if ui == self.action_1 else (1 - stay) * ui for ui in exp_u])  
        return u / np.sum(u)
    
    
class StayPhaseSoftmax:
    
    init_params = [1., .5, 12]
    bounds = [(.001, 10), (.001, .999), (1, 25)]
    
    def __init__(self, choices, best_action, trial, action_1):
        self.choices = choices
        self.best_action = best_action
        self.trial = trial
        self.action_1 = action_1
        
    def __call__(self, utility, temperature, stay, phase):
        if self.trial + 1 > phase:
            utility = np.array([np.exp(-abs(self.best_action - x)) for x in self.choices])
            
        centered_utility = utility - np.nanmax(utility)
        exp_u = np.exp(centered_utility / temperature).ravel()
        u = np.array([ui * stay if ui == self.action_1 else (1 - stay) * ui for ui in exp_u])  
        return u / np.sum(u)
