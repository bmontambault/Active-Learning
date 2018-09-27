import pandas as pd
import pymc3 as pm
from theano import tensor as tt
import numpy as np
import GPy

from data.get_results import get_results
from likelihood import Likelihood, get_kernel

def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining

results = get_results('data/results.json').iloc[3:]
function_names = results['function_name'].unique()
results['function_index'] = results['function_name'].apply(lambda x: function_names.tolist().index(x))

kernel = GPy.kern.RBF(1)
#kernel_dict = {f: get_kernel(results, kernel, f) for f in function_names}
functions_dict = results[['function_name', 'function']].drop_duplicates(subset = ['function_name']).set_index('function_name').to_dict()['function']
normalized_functions_dict = {f: (np.array(functions_dict[f]) - np.mean(functions_dict[f])) / np.std(functions_dict[f]) for f in function_names}


kernels = [kernel_dict[f] for f in function_names]
rewards = [normalized_functions_dict[f] for f in function_names]
choices = np.arange(80)

all_actions = results['response'].tolist()
data = np.vstack((range(len(results)), results['function_index'])).T
#data = results[['response', 'function_index']].values
test_data = np.array([data[0]])

"""
test_participant = results.iloc[0]
test_kernel = kernels[test_participant['function_name']]
test_choices = np.arange(80)

test_rewards = np.array(test_participant['function'])
n_test_rewards = (test_rewards - test_rewards.mean()) / test_rewards.std()
    
likelihood = Likelihood(1., 13, 1., .1, [.5, .5], test_choices, test_rewards, test_kernel)
test_actions = test_participant['response']
"""


K = 30
if __name__ == '__main__':
    
    #alpha = pm.Gamma('alpha', 1., 1.)
    #beta = pm.Beta('beta', 1., alpha, shape=K)
    #w = pm.Deterministic('w', stick_breaking(beta))
    with pm.Model() as single_model:
        center = pm.Beta('center', 1., 1.)
        steepness = pm.Gamma('steepness', 1., 1.)
        temperature = pm.Gamma('temperature', 1., 1.)
        mixture = pm.Categorical('mixture', [.5, .5])
        
        likelihood = Likelihood(center, steepness, temperature, mixture, choices, all_actions, rewards, kernels)
        obs = pm.DensityDist('like', likelihood.logpdf, observed = test_data)
        
    with single_model:
        trace = pm.sample(100)
