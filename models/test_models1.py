import numpy as np
import json
import GPy

from acquisitions import LocalMove, SGD, SGDMax, RandomSGD, RandomSGDMax, UCB
from decisions import Softmax, PhaseSoftmax
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
    

results = get_results('data/results.json').iloc[3:]
pos_linear = results[results['function_name'] == 'pos_linear'].iloc[0]['function']
neg_quad = results[results['function_name'] == 'neg_quad'].iloc[0]['function']
sinc_compressed = results[results['function_name'] == 'sinc_compressed'].iloc[0]['function']

pos_linear_n = [(f - np.mean(pos_linear)) / np.std(pos_linear) for f in pos_linear]
neg_quad_n = [(f - np.mean(neg_quad)) / np.std(neg_quad) for f in neg_quad]
sinc_compressed_n = [(f - np.mean(sinc_compressed)) / np.std(sinc_compressed) for f in sinc_compressed]

pos_linear_rbf = get_kernel(results, GPy.kern.RBF(1), 'pos_linear')
neg_quad_rbf = get_kernel(results, GPy.kern.RBF(1), 'neg_quad')
sinc_compressed_rbf = get_kernel(results, GPy.kern.RBF(1), 'sinc_compressed')


ID = "JpTkw0A5hejoJZn5U12X6qwjSrtweYFK"
participant = results[results['somataSessionId'] == ID].iloc[0]
actions = participant['response']
rewards = [sinc_compressed_n[a] for a in actions]
choices = np.arange(len(sinc_compressed))
strategies = ((SGD, Softmax), (SGD, PhaseSoftmax), (UCB, Softmax), (UCB, PhaseSoftmax))

#all_means, all_vars = get_all_means_vars(sinc_compressed_rbf, actions, rewards, choices)
#fit_data = [fit_strategy(actions, rewards, choices, strategy[0], strategy[1], sinc_compressed_rbf, all_means, all_vars) for strategy in strategies]
#data = [run(sinc_compressed_n, d['acquisition_type'], d['decision_type'], d['acq_params'], d['dec_params'], d['ntrials'], d['kernel'], d['actions'], d['rewards']) for d in fit_data]
plot_data = add_all_plots(data)
print (plot_data['id'])
with open('test_plot_data.json', 'w') as f:
    json.dump(plot_data, f)

'''
fit_data = fit_strategy(actions, rewards, choices, SGD1, PhaseSoftmax, kernel = sinc_compressed_rbf, all_means = [], all_vars = [], method = 'DE', restarts = 5)
data = run(sinc_compressed_n, fit_data['acquisition_type'], fit_data['decision_type'], 
           fit_data['acq_params'], fit_data['dec_params'], fit_data['ntrials'], 
           fit_data['kernel'], fit_data['actions'], fit_data['rewards'])
'''
'''
data = run(sinc_compressed_n, SGD, Softmax, [60.], [.1], 25)
plot_data = add_single_plot(data)
with open('test_plot_data.json', 'w') as f:
    json.dump(plot_data, f)
print (data['id'])
'''