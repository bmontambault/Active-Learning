import numpy as np
import GPy
from agent.agent import Agent
from agent.acquisitions import random
from agent.decisions import deterministic
from tasks.tasks import max_score, find_max
from tasks.visualize_responses import run

kernel = GPy.kern.RBF(1)
function = np.arange(80)
ntrials = 10
agent = Agent(kernel, random, deterministic, None, None)
name, acquisition, decision, acq_params, dec_params, function, total_score, current_actions, all_mean, all_var, all_utility, all_probability = max_score(function, agent, ntrials = ntrials)


run(name, acquisition, decision, acq_params, dec_params, function, total_score, current_actions, all_mean, all_var, all_utility, all_probability, ntrials)