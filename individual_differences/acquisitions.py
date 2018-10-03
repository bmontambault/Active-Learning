import pandas as pd
import GPy
import numpy as np
import scipy.stats as st
from theano import tensor as tt
from inspect import signature


def get_mean_var(kernel, actions, rewards, choices):
    
    if len(actions) == 0:
        return np.zeros(len(choices))[:,None], np.ones(len(choices))[:,None]
    else:
        X = np.array(actions)[:,None]
        Y = np.array(rewards)[:,None]
        m = GPy.models.GPRegression(X, Y, kernel)
        m.Gaussian_noise.variance = .00001
        mean, var = m.predict(np.array(choices)[:,None])
        return mean, var
    
    
def get_all_means_vars(kernel, actions, rewards, choices):
    
    all_means = []
    all_vars = []
    for i in range(len(actions)):
        a = actions[:i]
        r = rewards[:i]
        mean, var = get_mean_var(kernel, a, r, choices)
        all_means.append(mean.ravel())
        all_vars.append(var.ravel())
    return np.array(all_means), np.array(all_vars)


def generate(model, params, kernel, function, ntrials, nparticipants):
    
    choices = np.arange(len(function))
    all_responses = []
    for participant in range(nparticipants):
        actions = []
        rewards = []
        participant_responses = []
        for trial in range(ntrials):
            mean, var = get_mean_var(kernel, actions, rewards, choices)
            params['mean'] = mean.T
            params['var'] = var.T
            parameter_names = list(signature(model).parameters)
            model_params = {p: params[p] for p in parameter_names}
            likelihood = model(**model_params).ravel()
            
            action = st.rv_discrete(values = (choices, likelihood)).rvs()
            actions.append(action)
            rewards.append(function[action])
            
            one_hot_actions = np.zeros(len(choices))
            one_hot_actions[action] = 1
            response = np.array([mean.ravel(), var.ravel(), one_hot_actions])
            participant_responses.append(response)
        all_responses.append(participant_responses)
    return np.array(all_responses)


def mixture_generate(model, params, kernel, function, ntrials, nparticipants, mixture, K):
    
    return np.vstack([generate(model, {p: params[p][i] for p in params.keys()}, kernel, function, ntrials, int(nparticipants * mixture[i])) for i in range(K)])


def likelihood(X, model, params):
    
    mean = X[:,:,0]
    var = X[:,:,1]
    params['mean'] = mean
    params['var'] = var
    parameter_names = list(signature(model).parameters)
    model_params = {p: params[p] for p in parameter_names}
    
    likelihood = model(**model_params)
    action = X[:,:,2]
    action_likelihood = (likelihood * action).sum(axis = 2)
    participant_likelihood = action_likelihood.sum(axis = 1)
    sample_likelihood = participant_likelihood.sum()
    return sample_likelihood


def mixture_likelihood(X, model, params, mixture, K):
    
    return np.array([likelihood(X, model, {p: params[p][i] for p in params.keys()}) * mixture[i] for i in range(K)]).sum()


def ucb(mean, var, explore, temperature):
    
    mean_dims = mean.ndim    
    utility = mean + var * explore
    return utility
    center = utility.max(axis = mean_dims - 1, keepdims = True)
    centered_utility = utility - center
    return centered_utility
    
    exp_u = np.exp(centered_utility / temperature)
    exp_u_row_sums = exp_u.sum(axis = mean_dims - 1, keepdims = True)
    likelihood = exp_u / exp_u_row_sums
    return likelihood


def multi_ucb(mean, var, explore, temperature):
    
    mean_dims = mean.ndim
    
    utility = ((mean + var)[:,:,None] * explore).T
    return utility
    center = utility.max(axis = mean_dims - 1, keepdims = True)
    centered_utility = utility - center
    return centered_utility
    
    exp_u = np.exp(centered_utility / temperature)
    exp_u_row_sums = exp_u.sum(axis = mean_dims - 1, keepdims = True)
    likelihood = exp_u / exp_u_row_sums
    return likelihood


def local(linear_interp, last_action, learning_rate):
    
    choices = np.arange(len(linear_interp))
    if last_action == -1:
        return choices**0
    gradient = np.gradient(linear_interp)
    next_action = last_action + gradient * learning_rate
    return np.exp(abs(next_action - choices))
    


    
    