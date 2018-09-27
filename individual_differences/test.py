from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
from theano import tensor as tt
import pandas as pd


class Acq():
        
    def __init__(self, choices):
        self.choices = choices
        
    def __call__(self, x):
        return np.array([np.exp(-abs(x - xi)) for xi in self.choices])
    
    
class Dec():
    
    def __init__(self, utility):
        self.utility = utility
    
    def __call__(self, temperature):
        centered_utility = self.utility - np.nanmax(self.utility)
        exp_u = np.exp(centered_utility / temperature)
        return exp_u / np.nansum(exp_u)
    
    
def run(choices, x, temperature, n):
    
    actions = []
    acq = Acq(choices)
    for i in range(n):
        utility = acq(x)
        dec = Dec(utility)
        p = dec(temperature)
        next_action = sp.stats.rv_discrete(values = (choices, p)).rvs()
        actions.append(next_action)
    return np.array(actions)


def likelihood(choices, x, temperature, actions):
    
    joint_prob = 1
    acq = Acq(choices)
    for i in range(len(actions)):
        action = actions[i]
        utility = acq(x)
        dec = Dec(utility)
        p = dec(temperature)[action]
        joint_prob *= p
    return joint_prob


def full_likelihood(choices, all_actions):
    
    pass
    


def generate_data(group_probs, x_means, x_variances, temp_means, temp_variances, nparticipants, n, choices):
    
    all_actions = []
    for i in range(nparticipants):
        group = sp.stats.rv_discrete(values = (range(len(group_probs)), group_probs)).rvs()
        
        x_mean = x_means[group]
        x_var = x_variances[group]
        x = int(sp.stats.norm(x_mean, x_var).rvs())
        
        temp_mean = temp_means[group]
        temp_var = temp_variances[group]
        temp = sp.stats.norm(temp_mean, temp_var).rvs()
        
        actions = run(choices, x, temp, n)
        all_actions.append(actions)
    return np.array(all_actions)
        

def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

K = 5
with pm.Model() as model:
    alpha = pm.Gamma('alpha', 1., 1.)
    beta = pm.Beta('beta', 1., alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking(beta))
    
    mu = pm.Normal('mu', 50, 10)
    temp = pm.Uniform('temp', 1., .1)
    
    likelihood = 
    

    tau = pm.Gamma('tau', 1., 1., shape=K)
    lambda_ = pm.Uniform('lambda', 0, 5, shape=K)
    mu = pm.Normal('mu', 0, tau=lambda_ * tau, shape=K)
    obs = pm.NormalMixture('obs', w, mu, tau=lambda_ * tau,
                           observed=old_faithful_df.std_waiting.values)
