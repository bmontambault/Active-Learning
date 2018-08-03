import numpy as np
import pandas as pd
import json
import GPy

from acquisitions import LocalMove, SGD, SGDMax, RandomSGD, RandomSGDMax, Explore, Exploit, EI, UCB
from decisions import Softmax, PhaseSoftmax, StaySoftmax, StayPhaseSoftmax
from run import run, fit_strategy, get_all_means_vars, add_all_plots
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
    
    actions_ = []
    rewards_ = []
    for i in range(len(actions)):
        if actions[i] not in actions_:
            actions_.append(actions[i])
            rewards_.append(rewards[i])
    
    all_means, all_vars = get_all_means_vars(kernel, actions, rewards, choices)
    fit_data = [fit_strategy(actions, rewards, choices, strategy[0], strategy[1], kernel, all_means, all_vars) for strategy in strategies]
    data = [run(function_n, d['acquisition_type'], d['decision_type'], d['acq_params'], d['dec_params'], d['ntrials'], d['kernel'], d['actions'], d['rewards'], ID = participant['somataSessionId']) for d in fit_data]
    plot_data = add_all_plots(data)
    plot_data['function_name'] = [participant['function_name']]
    plot_data['goal'] = [participant['goal']]
    plot_data['id'] = [participant['somataSessionId']]
    return plot_data


def fit_participants(results, IDs, strategies):
    
    all_data = []
    all_plot_data = []
    for ID in IDs:
        plot_data = fit_participant(results, ID, strategies)
        all_plot_data.append(plot_data)
        row = [plot_data['acquisition'], plot_data['decision'], plot_data['acq_params'], plot_data['dec_params'], plot_data['function_name'], plot_data['goal'], plot_data['actions'], plot_data['AIC'], [plot_data['score']], [plot_data['max_score']], plot_data['pseudo_r2'], plot_data['id']]
        stacked = [[d[i] if len(d) > 1 else d[0] for d in row] for i in range(len(row[0]))]
        all_data += stacked
    data = pd.DataFrame(all_data)
    data.columns = ['acq', 'dec', 'acq_params', 'dec_params', 'f', 'goal', 'actions', 'AIC', 'score', 'max_score', 'pseudo_r2', 'id']
    data['%score'] = data['score'] / data['max_score']
    return data, all_plot_data
    

results = get_results('data/results.json').iloc[3:]
strategies = [(LocalMove, Softmax), (Explore, Softmax)]
IDs = ["gV5RZpvDCQgGzfrdPaYQMJH1kbbNaJN3"]
data, plot_data = fit_participants(results, IDs, strategies)

#plot_data1 = fit_participant(results, "JpTkw0A5hejoJZn5U12X6qwjSrtweYFK", strategies)
#plot_data2 = fit_participant(results, "gV5RZpvDCQgGzfrdPaYQMJH1kbbNaJN3", strategies)

#print (plot_data['id'])
#with open('test_plot_data.json', 'w') as f:
    #json.dump(plot_data, f)

#data = run(sinc_compressed_n, SGD, Softmax, [60.], [.1], 25)
