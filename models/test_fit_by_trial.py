import sys
sys.path.append('./agent')

import scipy.stats as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPy

from data.get_results import get_results
from agent.acquisitions import random, fixed_sgd, sgd
from agent.decisions import deterministic, propto, softmax, random_stay_softmax
from agent.agent import Agent

from tasks.tasks import max_score
from tasks.visualize_responses import run



results = get_results('data/results.json')
results = results[3:]
subject = results.iloc[0]

function = subject['function']
function_n = (function - np.mean(function)) / np.std(function)
all_gradients = [abs(x - y) for x in function_n for y in function_n]
kernel = GPy.kern.RBF(1)


acq_params = [1.]#, .01]
acq_priors = [st.gamma(5, scale = 200).logpdf]#, st.beta(1, 1).logpdf]
#acq_bounds = [(.001, 2000)]#, (.001, .999)]
acq_bounds = [st.gamma(5, scale = 200).interval(.99)]

dec_params = [.1, .5]
dec_priors = [st.gamma(.01, scale = 10).logpdf, st.beta(1, 1).logpdf]#st.gamma(.1, scale = .1).logpdf]
dec_bounds = [(.001, .5), (.001, .999)]

fixed_dec_idx = []

agent = Agent(kernel, sgd, random_stay_softmax, acq_params, dec_params, acq_priors = acq_priors, acq_bounds = acq_bounds, dec_priors = dec_priors, dec_bounds = dec_bounds, name = subject['somataSessionId'])
actions = subject['response']#[:3]

#actions = [58, 27, 27]
ntrials = len(actions)

name, acquisition, decision, acq_params, dec_params, function, total_score, total_probability, current_actions, all_mean, all_var, all_utility, all_probability = max_score(function_n, [agent], actions = actions, optimize_joint = True, method = 'DE', fixed_dec_idx = fixed_dec_idx)
run(name, acquisition, decision, acq_params, dec_params, function, total_score, current_actions, all_mean, all_var, all_utility, all_probability, ntrials)
