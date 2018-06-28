import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPy

from data.get_results import get_results
from agent.acquisitions import random, fixed_sgd, sgd
from agent.decisions import deterministic, softmax
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

random_agent = Agent(kernel, random, deterministic, [], [], subject.somataSessionId)
agent = Agent(kernel, sgd, softmax, [100], [1.], subject.somataSessionId)
#lr_prior = GPy.priors.gamma_from_EV(10, 10).lnpdf
temp_prior = GPy.priors.gamma_from_EV(10, 1).lnpdf
x = np.linspace(0, 100)
#plt.plot(x, lr_prior(x))

#acq_priors = [lr_prior]
dec_priors = [temp_prior]

#print (agent.joint_log_likelihood(function_n, subject.response))

#agent.map_decision(function_n, subject.response, dec_priors)
#print (agent.joint_log_likelihood(function_n, subject.response))


name, acquisition, decision, acq_params, dec_params, function, total_score, current_actions, all_mean, all_var, all_utility, all_probability = max_score(function_n, [agent], actions = subject['response'])
run(name, acquisition, decision, acq_params, dec_params, function, total_score, current_actions, all_mean, all_var, all_utility, all_probability, len(subject['response']))