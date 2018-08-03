import numpy as np
import json
import GPy

from acquisitions import LocalMove, SGD, SGDMax, RandomSGD, RandomSGDMax, Explore, Exploit, EI, UCB
from decisions import Softmax, PhaseSoftmax, StaySoftmax, StayPhaseSoftmax
from run import run, fit_strategy, add_single_plot, get_all_means_vars, add_all_plots
from data.get_results import get_results

def get_kernel(results, kernel, function_name):
    
    function_samples = list(results[results['function_name'] == 'sinc_compressed'].iloc[0]['function_samples'].values())
    function_samples_n = np.array([((f - np.mean(f)) / np.std(f)) for f in function_samples])
    stacked_function_samples_x = np.hstack([np.arange(len(f)) for f in function_samples_n])[:,None]
    stacked_function_samples_y = np.hstack(function_samples_n)[:,None]
    m = GPy.models.GPRegression(X = stacked_function_samples_x, Y = stacked_function_samples_y, kernel = kernel)
    m.optimize()
    return m.kern


def fit_participant(results, ID, strategies):
    
    participant = results[results['somataSessionId'] == ID].iloc[0]
    function = participant['function']
    function_n = [(f - np.mean(function)) / np.std(function) for f in function]
    kernel = get_kernel(results, GPy.kern.RBF(1), participant['function_name'])
    
    actions = participant['response']
    rewards = [function_n[a] for a in actions]
    choices = np.arange(len(function_n))
    
    all_means, all_vars = get_all_means_vars(kernel, actions, rewards, choices)
    fit_data = [fit_strategy(actions, rewards, choices, strategy[0], strategy[1], kernel, all_means, all_vars) for strategy in strategies]
    data = [run(function_n, d['acquisition_type'], d['decision_type'], d['acq_params'], d['dec_params'], d['ntrials'], d['kernel'], d['actions'], d['rewards'], ID = participant['somataSessionId']) for d in fit_data]
    plot_data = add_all_plots(data)
    return plot_data
    
    

results = get_results('data/results.json').iloc[3:]
strategies = [(EI, StayPhaseSoftmax)]
plot_data = fit_participant(results, "JpTkw0A5hejoJZn5U12X6qwjSrtweYFK", strategies)

print (plot_data['id'])
with open('test_plot_data.json', 'w') as f:
    json.dump(plot_data, f)

#data = run(sinc_compressed_n, SGD, Softmax, [60.], [.1], 25)
