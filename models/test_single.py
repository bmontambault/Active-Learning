import sys
sys.path.append('./agent')

import numpy as np
import GPy
from agent.agent import Agent
from agent.acquisitions import random, ucb, fixed_sgd
from agent.decisions import deterministic, softmax
from tasks.tasks import max_score, find_max
from tasks.visualize_responses import run



kernel = GPy.kern.RBF(1)
function = np.arange(80)
ntrials = 10
agent = Agent(kernel, fixed_sgd, softmax, [1], [.1], 'agent')
name, acquisition, decision, acq_params, dec_params, function, total_score, current_actions, all_mean, all_var, all_utility, all_probability = max_score(function, [agent], ntrials = ntrials)

run(name, acquisition, decision, acq_params, dec_params, function, total_score, current_actions, all_mean, all_var, all_utility, all_probability, ntrials)