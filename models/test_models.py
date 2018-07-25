import json
import pandas as pd
import numpy as np
import inspect
import scipy.stats as st
import GPy

from data.get_results import get_results
from fit_responses import gp
from acquisitions import Random, LocalMove, MaxSGD, SGD, Phase, Explore, Exploit, UCB, MinimumSimpleRegretSearch, MinimumCumulativeRegretSearch
from decisions import Propto, Softmax


def task(function, acquisition_type, decision_type, acq_params, dec_params, ntrials, kernel = None):
    
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
        
        args = {'all_x': all_x, 'mean': mean, 'var': var, 'trial': i, 'remaining_trials': ntrials - i,
                'last_x': last_x, 'last_y': last_y,
                'second_last_x': second_last_x, 'second_last_y': second_last_y,
                'unique_second_last_x': unique_second_last_x, 'unique_second_last_y': unique_second_last_y,
                'ntrials': ntrials, 'actions': actions[:i], 'rewards': rewards[:i],
                'kernel': kernel}
        acq_arg_names = list(inspect.signature(acquisition_type.__init__).parameters.keys())
        acq_args = {arg_name: args[arg_name] for arg_name in args.keys() if arg_name in acq_arg_names}
        acquisition = acquisition_type(**acq_args)
        utility = acquisition(*acq_params)
        
        decision = decision_type()
        probability = decision(utility, *dec_params)
        next_action = st.rv_discrete(values = (all_x, probability)).rvs()
        print (next_action)
        next_reward = function[next_action]
        actions.append(next_action)
        rewards.append(next_reward)
    return actions, rewards


def test_condition(results, function_name, strategies, nattempts):
    
    condition = results[results['function_name'] == function_name].iloc[0]
    function_samples = condition['function_samples'].values()
    function_samples_n = np.array([((f - np.mean(f)) / np.std(f)) for f in function_samples])
    
    if np.any([s[0].isGP for s in strategies]):
        stacked_function_samples_x = np.hstack([np.arange(len(f)) for f in function_samples_n])[:,None]
        stacked_function_samples_y = np.hstack(function_samples_n)[:,None]
        m = GPy.models.GPRegression(X = stacked_function_samples_x, Y = stacked_function_samples_y, kernel = GPy.kern.RBF(1))
        m.optimize()
        kernel = m.kern
    else:
        kernel = None
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
            actions, rewards = task(function_n, acquisition_type, decision_type, acq_params, dec_params, ntrials, kernel)
            #print (actions)
            #print (rewards)
            import matplotlib.pyplot as plt
            plt.plot(actions, rewards, 'bo')
            plt.show()
            find_max = np.max([function[a] for a in actions])
            max_score = np.sum([function[a] for a in actions])
            total_find_max += find_max
            total_max_score += max_score
        all_find_max.append(total_find_max / nattempts)
        all_max_score.append(total_max_score / nattempts)
    return all_find_max, all_max_score


sgd_params = [1, 10, 100, 500, 1000]        
#ucb_params = [.1, 1, 10, 100]
ucb_params = [100, 500, 1000, 5000, 10000]
phase_params = np.arange(2, 25)
softmax_params = [.001, .1, 1., 10]

#strategies = [(Random, Softmax, [], [.1])] + [(SGD, Softmax, [s], [t]) for s in sgd_params for t in softmax_params]+ [(Explore, Softmax, [], [t]) for t in softmax_params] + [(Exploit, Softmax, [], [t]) for t in softmax_params] + [(Phase, Softmax, [p], [t]) for p in phase_params for t in softmax_params] + [(UCB, Softmax, [u], [t]) for u in ucb_params for t in softmax_params]
results = get_results('data/results.json').iloc[3:]

#strategies = [(UCB, Softmax, [e], [s]) for e in ucb_params for s in softmax_params]
strategies = [(MinimumCumulativeRegretSearch, Softmax, [], [.1])]
all_find_max, all_max_score = test_condition(results, 'sinc_compressed', strategies, 1)

strategy_types = [s[0].__name__ for s in strategies]
df = pd.DataFrame(np.array([strategy_types, all_max_score, all_find_max]).T, columns = ['strategy', 'max_score', 'find_max'])
params = [tuple(s[2] + s[3]) for s in strategies]
df['params'] = params