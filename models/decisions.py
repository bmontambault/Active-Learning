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
    
    
class RandomStaySoftmax:
    
    init_params = [1., .5]
    bounds = [(.001, 10), (.001, .999)]
    
    def __init__(self, action_1):
        self.action_1 = action_1
    
    def __call__(self, utility, temperature, stay):
        centered_utility = utility - np.nanmax(utility)
        exp_u = np.exp(centered_utility / temperature)
        u = exp_u / np.nansum(exp_u)
        u = np.array([ui + stay if ui == self.action_1 else ui for ui in u])
        return u / np.sum(u)
    
    
class PhaseSoftmax:
    
    init_params = [1., 12]
    bounds = [(.001, 10), (.1, 24)]
    
    def __init__(self, choices, best_action, mean, trial):
        self.choices = choices
        self.best_action = best_action
        self.mean = mean
        self.trial = trial
        
    def __call__(self, utility, temperature, phase):
        if self.trial > phase:
            if np.all(self.mean != None):
                utility = self.mean / np.sum(self.mean)
            else:
                utility = np.array([np.exp(-abs(self.best_action - x)) for x in self.choices])
            
        centered_utility = utility - np.nanmax(utility)
        exp_u = np.exp(centered_utility / temperature)
        return exp_u / np.nansum(exp_u)