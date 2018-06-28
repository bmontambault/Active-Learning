import sys
sys.path.append('./agent')

import scipy.stats as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPy

from data.get_results import get_results
from agent.acquisitions import random, sgd, decaying_sgd
from agent.decisions import deterministic, propto, softmax, random_stay_softmax, random_switch_softmax
from agent.agent import Agent
from agent.train_gp import train_gp

from tasks.tasks import max_score, find_max
#from tasks.visualize_responses import run


participant_num = 4

results = get_results('data/results.json')
results = results[3:]
subject = results.iloc[participant_num]
actions = subject['response']

function = subject['function']
function_n = (function - np.mean(function)) / np.std(function)
kernel = GPy.kern.RBF(1)
kernel_name = 'Squared Exponential'
variance_prior = GPy.priors.Gamma(.1, .1)
lengthscale_prior = GPy.priors.Gamma(.1, .1)
kernel_priors = [variance_prior, lengthscale_prior]

function_samples = subject['function_samples']
function_samples_X = np.array([[i for i in range(len(function_samples['0']))] for j in range(len(function_samples.keys()))]).ravel()[:,None]
function_samples_y = np.array([function_samples[key] for key in function_samples.keys()]).ravel()[:,None]
function_samples_y_n = (function_samples_y - np.mean(function_samples_y)) / np.std(function_samples_y)

kernel = train_gp(kernel, kernel_priors, function_samples_X, function_samples_y_n)

lr_init = 1.
lr_prior = st.gamma(5, scale = 200).logpdf
lr_bounds = (1., 3000)

decay_init = .9
decay_prior = st.beta(1, 5).logpdf
decay_bounds = (.001, .999)


sr_init = .1
sr_prior = st.beta(1, 5).logpdf
sr_bounds = (.001, .999)

temp_init = 1.
temp_prior = st.gamma(1.5, scale = 3).logpdf
temp_bounds = (.01, 10)

random_agent = Agent(kernel, random, propto, [], [], name = subject['somataSessionId'])

sgd_agent = Agent(kernel, sgd, softmax, [lr_init], [temp_init], acq_priors = [lr_prior], dec_priors = [temp_prior], acq_bounds = [lr_bounds], dec_bounds = [temp_bounds], name = subject['somataSessionId'])
random_stay_sgd_agent = Agent(kernel, sgd, random_stay_softmax, [lr_init], [temp_init, sr_init], acq_priors = [lr_prior], dec_priors = [temp_prior, sr_prior], acq_bounds = [lr_bounds], dec_bounds = [temp_bounds, sr_bounds], name = subject['somataSessionId'])
random_switch_sgd_agent = Agent(kernel, sgd, random_switch_softmax, [lr_init], [temp_init, sr_init], acq_priors = [lr_prior], dec_priors = [temp_prior, sr_prior], acq_bounds = [lr_bounds], dec_bounds = [temp_bounds, sr_bounds], name = subject['somataSessionId'])

decaying_sgd_agent = Agent(kernel, decaying_sgd, softmax, [lr_init, decay_init], [temp_init], acq_priors = [lr_prior, decay_prior], dec_priors = [temp_prior], acq_bounds = [lr_bounds, decay_bounds], dec_bounds = [temp_bounds], name = subject['somataSessionId'])
random_stay_decaying_sgd_agent = Agent(kernel, decaying_sgd, random_stay_softmax, [lr_init, decay_init], [temp_init, sr_init], acq_priors = [lr_prior, decay_prior], dec_priors = [temp_prior, sr_prior], acq_bounds = [lr_bounds, decay_bounds], dec_bounds = [temp_bounds, sr_bounds], name = subject['somataSessionId'])
random_switch_decaying_sgd_agent = Agent(kernel, decaying_sgd, random_switch_softmax, [lr_init, decay_init], [temp_init, sr_init], acq_priors = [lr_prior, decay_prior], dec_priors = [temp_prior, sr_prior], acq_bounds = [lr_bounds, decay_bounds], dec_bounds = [temp_bounds, sr_bounds], name = subject['somataSessionId'])


ntrials = len(actions)

if subject['goal'] == 'max_score_last':
    name, acquisition, decision, acq_params, dec_params, function, total_score, total_probability, current_actions, all_mean, all_var, all_utility, all_probability = max_score(function_n, [random_agent, sgd_agent, random_stay_sgd_agent, random_switch_sgd_agent, decaying_sgd_agent, random_stay_decaying_sgd_agent, random_switch_decaying_sgd_agent], actions = actions, optimize_joint = True, method = 'DE', fixed_dec_idx = [])
    goal = 'Max Score'
else:
    name, acquisition, decision, acq_params, dec_params, function, total_score, total_probability, current_actions, all_mean, all_var, all_utility, all_probability = find_max(function_n, [random_agent, sgd_agent, random_stay_sgd_agent, random_switch_sgd_agent, decaying_sgd_agent, random_stay_decaying_sgd_agent, random_switch_decaying_sgd_agent], actions = actions, optimize_joint = True, method = 'DE', fixed_dec_idx = [])
    goal = 'Find Max'
    
#run(name, acquisition, decision, acq_params, dec_params, function, total_score, total_probability, current_actions, all_mean, all_var, all_utility, all_probability, ntrials, goal, kernel_name)
