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