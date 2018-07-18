import json
import pandas as pd
import numpy as np
import inspect
import scipy.stats as st
import GPy

from data.get_results import get_results
from fit_responses import gp
from acquisitions import Random, RandomMove, SGD, Explore, Exploit, UCB
from decisions import Propto, Softmax


def task(kernel, function, acquisition_type, decision_type, acq_params, dec_params, ntrials):
    
    actions = []
    rewards = []
    all_x = np.arange(len(function))
    for i in range(ntrials):
        
        if acquisition_type.isGP:
            mean, var = gp(kernel, np.array(actions), np.array(rewards), all_x)
        else:
            mean = []
            var = []
        
        if i > 0:
            last_x = actions[i - 1]
            last_y = rewards[i - 1]
        else:
            last_x = None
            last_y = None
        if i > 1:
            second_last_x = actions[i - 2]
            second_last_y = rewards[i - 2]
            idx = [j for j in range(len(actions[:i])) if actions[j] != last_x]
            if len(idx) > 0:
                unique_second_last_x = actions[idx[-1]]
                unique_second_last_y = rewards[idx[-1]]
            else:
                unique_second_last_x = None
                unique_second_last_y = None
                second_last_x = None
                second_last_y = None
        else:
            unique_second_last_x = None
            unique_second_last_y = None
            second_last_x = None
            second_last_y = None
        
        args = {'all_x': all_x, 'mean': mean, 'var': var, 'trial': i,
                'last_x': last_x, 'last_y': last_y,
                'second_last_x': second_last_x, 'second_last_y': second_last_y,
                'unique_second_last_x': unique_second_last_x, 'unique_second_last_y': unique_second_last_y,
                'ntrials': len(actions), 'trial': i, 'actions': actions[:i]}
        acq_arg_names = list(inspect.signature(acquisition_type.__init__).parameters.keys())
        acq_args = {arg_name: args[arg_name] for arg_name in args.keys() if arg_name in acq_arg_names}
        acquisition = acquisition_type(**acq_args)
        utility = acquisition(*acq_params)
        
        decision = decision_type()
        probability = decision(utility, *dec_params)
        next_action = st.rv_discrete(values = (all_x, probability)).rvs()
        next_reward = function[next_action]
        actions.append(next_action)
        rewards.append(next_reward)
    return actions, rewards


def test_condition(results, function_name, strategies, nattempts):
    
    condition = results[results['function_name'] == function_name].iloc[0]
    function_samples = condition['function_samples'].values()
    function_samples_n = np.array([((f - np.mean(f)) / np.std(f)) for f in function_samples])
    stacked_function_samples_x = np.hstack([np.arange(len(f)) for f in function_samples_n])[:,None]
    stacked_function_samples_y = np.hstack(funcstion_samples_n)[:,None]
    m = GPy.models.GPRegression(X = stacked_function_samples_x, Y = stacked_function_samples_y, kernel = GPy.kern.RBF(1))
    m.optimize()
    
    kernel = m.kern
    function = np.array(condition['function'])
    fmean = np.mean(function)
    fstd = np.std(function)
    function_n = function - fmean / fstd
    ntrials = len(condition['response'])
    all_find_max = []
    all_max_score = []
    for acquisition_type, decision_type, acq_params, dec_params in strategies:
        total_find_max = 0
        total_max_score = 0
        for i in range(nattempts):
            actions, rewards = task(kernel, function_n, acquisition_type, decision_type, acq_params, dec_params, ntrials)
            print (actions)
            find_max = np.max([function[a] for a in actions])
            max_score = np.sum([function[a] for a in actions])
            total_find_max += find_max
            total_max_score += max_score
        all_find_max.append(total_find_max / nattempts)
        all_max_score.append(total_max_score / nattempts)
    return all_find_max, all_max_score
        
ucb_params = [.1, 1, 10, 100]
sgd_params = [10, 100, 500, 1000]
        

results = get_results('data/results.json').iloc[3:]
strategies = [
        (SGD, Softmax, [500], [.001])
        ]

all_find_max, all_max_score = test_condition(results, 'pos_linear', strategies, 100)
