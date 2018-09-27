import pandas as pd
import GPy
import numpy as np

from data.get_results import get_results
from likelihood import get_kernel


def get_mean_var(kernel, actions, rewards, choices):
    
    if len(actions) == 0:
        return np.zeros(len(choices))[:,None], np.ones(len(choices))[:,None]
    else:
        X = np.array(actions)[:,None]
        Y = np.array(rewards)[:,None]
        m = GPy.models.GPRegression(X, Y, kernel)
        m.Gaussian_noise.variance = .00001
        mean, var = m.predict(np.array(choices)[:,None])
        return mean, var
    
    
def get_all_means_vars(kernel, actions, rewards, choices):
    
    all_means = []
    all_vars = []
    for i in range(len(actions)):
        a = actions[:i]
        r = rewards[:i]
        mean, var = get_mean_var(kernel, a, r, choices)
        all_means.append(mean.ravel())
        all_vars.append(var.ravel())
    return np.array(all_means), np.array(all_vars)


def vectorize_actions(actions, choices):
    
    vec_actions = np.zeros(shape = (len(actions), len(choices)))
    for i in range(len(actions)):
        vec_actions[i][actions[i]] = 1
    return vec_actions


def vectorize(participant, kernel_dict, normalized_functions_dict):
    
    function_name = participant['function_name']
    kern = kernel_dict[function_name]
    function = normalized_functions_dict[function_name]
    
    actions = participant['response']
    rewards = [function[a] for a in actions]
    choices = np.arange(len(function))
    
    mean, var = get_all_means_vars(kern, actions, rewards, choices)
    vec_actions = vectorize_actions(actions, choices)
    
    return np.array(list(zip(mean.T, var.T, vec_actions.T))).T


def ucb(X, explore):
    
    utility = X[:,0] + X[:,1] * explore
    return utility


def softmax(utility, temperature):
    
    centered_utility = utility - np.nanmax(utility, axis = 1)
    exp_u = np.exp(centered_utility / temperature)
    return exp_u / np.nansum(exp_u)


results = get_results('data/results.json').iloc[3:]
function_names = results['function_name'].unique()
kernel = GPy.kern.RBF(1)
#kernel_dict = {f: get_kernel(results, kernel, f) for f in function_names}
functions_dict = results[['function_name', 'function']].drop_duplicates(subset = ['function_name']).set_index('function_name').to_dict()['function']
normalized_functions_dict = {f: (np.array(functions_dict[f]) - np.mean(functions_dict[f])) / np.std(functions_dict[f]) for f in function_names}

participant = results.iloc[0]
X = vectorize(participant, kernel_dict, normalized_functions_dict)