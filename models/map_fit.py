import sys
sys.path.append('./agent')

import json
import numpy as np
import seaborn as sns

from data.get_results import get_results
from agent.train_gp import train_gp
from tasks.tasks import max_score, find_max
from fit_config import agents, kernels, all_kernel_priors


def fit_participant(participant_idx, method = 'DE', uniform_priors = False):
    
    results = get_results('data/results.json')
    results = results[3:]
    subject = results.iloc[participant_idx]
    name = subject['somataSessionId']
    goal = subject['goal']
    actions = subject['response']
    #actions = [actions[0]] + [actions[i] for i in range(1, len(actions)) if actions[i] != actions[i - 1]]
    ntrials = len(actions)
  
    function = subject['function']
    function_n = (function - np.mean(function)) / np.std(function)
    
    function_samples = subject['function_samples']
    function_samples_X = np.array([[i for i in range(len(function_samples['0']))] for j in range(len(function_samples.keys()))]).ravel()[:,None]
    function_samples_y = np.array([function_samples[key] for key in function_samples.keys()]).ravel()[:,None]
    function_samples_y_n = (function_samples_y - np.mean(function_samples_y)) / np.std(function_samples_y)
    
    for kernel in kernels:
        
        kernel_name = kernel.name
        kernel_priors = all_kernel_priors[kernel_name]
        kernel = train_gp(kernel, kernel_priors, function_samples_X, function_samples_y_n)
        for agent in agents:
            agent.kernel = kernel
            
        if goal == 'max_score_last':
            acquisition, decision, acq_params, dec_params, function, total_score, total_probability, current_actions, all_mean, all_var, all_utility, all_probability = max_score(function_n, agents, actions = actions, optimize_joint = True, method = method, uniform_priors = uniform_priors)
            goal = 'Max Score'
        else:
            acquisition, decision, acq_params, dec_params, function, total_score, total_probability, current_actions, all_mean, all_var, all_utility, all_probability = find_max(function_n, agents, actions = actions, optimize_joint = True, method = method, uniform_priors = uniform_priors)
            goal = 'Find Max'
        
        c = sns.color_palette("hls", len(acquisition)).as_hex()
        data = {'id': name, 'acquisition': acquisition, 'decision': decision, 'acq_params': acq_params, 'dec_params': dec_params, 'function': function, 'score': [x.item() for x in total_score], 'actions': current_actions, 'mean': all_mean, 'var': all_var, 'utility': all_utility, 'probability': all_probability, 'total_probability': total_probability, 'ntrials': ntrials, 'goal': goal, 'kernel': kernel_name, 'colors': c}
        
        path = 'model_results/map/participant{}_{}.json'.format(participant_idx, kernel_name)
        with open(path, 'w') as f:
            json.dump(data, f)     