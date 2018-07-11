import json
import pandas as pd
import numpy as np
import GPy

from data.get_results import get_results
from test_acq import UCB, SGD_MaxScore, EI
from test_dec import Softmax
from fit_responses import get_mean_var, get_utility, get_likelihood, joint_log_likelihood


with open('good_results.json', 'r') as f:
    results = pd.read_json(json.load(f))
all_results = get_results('data/results.json').iloc[1:]

#lin_rl = results[(results['function_name'] == 'pos_linear') & (results['goal'] == 'max_score_last')]
#bad_lin_rl = all_results[(all_results['function_name'] == 'pos_linear') & (all_results['goal'] == 'max_score_last')]
#bad_lin_rl = bad_lin_rl.loc[[i for i in bad_lin_rl.index if i not in lin_rl.index]]
'''
subject = all_results.loc[6]

actions = subject['response']
function = subject['function']
function_n = ((function - np.mean(function)) / np.std(function))
function_samples = list(subject['function_samples'].values())
function_samples_n = np.array([((f - np.mean(f)) / np.std(f)) for f in function_samples])

function_samples = subject['function_samples']
function_samples_X = np.array([[i for i in range(len(function_samples['0']))] for j in range(len(function_samples.keys()))]).ravel()[:,None]
function_samples_y = np.array([function_samples[key] for key in function_samples.keys()]).ravel()[:,None]
function_samples_y_n = (function_samples_y - np.mean(function_samples_y)) / np.std(function_samples_y)
all_x = np.arange(len(function))
rewards = [function_n[a] for a in actions]

kernel = GPy.kern.Linear(1) + GPy.kern.Bias(1)
#kernel = GPy.kern.RBF(1)
m = GPy.models.GPRegression(X = function_samples_X, Y = function_samples_y_n, kernel = kernel)
m.Gaussian_noise.variance.constrain_fixed()
m.optimize()

acquisition_type = UCB
decision_type = Softmax
acq_params = [1.37]
dec_params = [.161]

all_means, all_vars = get_mean_var(kernel, function_n, actions)
utility_data = get_utility(all_x, actions, rewards, acquisition_type, acq_params, all_means = all_means, all_vars = all_vars, replace = False)


likelihood_data = get_likelihood(utility_data, actions, decision_type, dec_params)
log_likelihood = joint_log_likelihood(actions, likelihood_data)

u = [utility_data['trial_'+str(i + 1)][actions[i]] for i in range(len(actions))]
l = [likelihood_data['trial_'+str(i + 1)][actions[i]] for i in range(len(actions))]
#print(u)
print (log_likelihood)
'''