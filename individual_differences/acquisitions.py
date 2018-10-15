import GPy
import numpy as np
import scipy.stats as st
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


def generate_cluster(model, params, kernel, function, ntrials, nparticipants):
    
    choices = np.arange(len(function))
    all_responses = []
    for participant in range(nparticipants):
        actions = []
        rewards = []
        participant_responses = []
        for trial in range(ntrials):
            mean, var = get_mean_var(kernel, actions, rewards, choices)
            params['mean'] = mean.T[None]
            params['var'] = var.T[None]
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
    return np.array(all_responses) #(nparticipants, ntrials, function variables, choices)


def generate(model, params, kernel, function, ntrials, nparticipants, K, mixture):
    
    cluster_sizes = [int(nparticipants * mixture[i]) for i in range(K)]
    clusters = np.hstack([np.ones(cluster_sizes[i]) * i for i in range(len(cluster_sizes))])
    results = np.vstack([generate_cluster(model, {p: params[p][i] if p not in ['mean', 'var'] else params[p] for p in params.keys()}, kernel, function, ntrials, cluster_sizes[i]) for i in range(K)])
    return results, clusters
    

def likelihood(X, model, params, K, mixture):
    
    mean = X[:,:,0] #(nparticipants, ntrials, choices)
    var = X[:,:,1] #(nparticipants, ntrials, choices)
    params['mean'] = mean
    params['var'] = var
    parameter_names = list(signature(model).parameters)
    model_params = {p: params[p] for p in parameter_names}
    
    likelihood = model(**model_params) #(nparticipants, ntrials, choices, K)
    action = X[:,:,2]
    action_likelihood = (likelihood * (action[:,:,:,None] * np.ones(K))).sum(axis = 2) #(nparticipants, ntrials, K)
    participant_likelihood = action_likelihood.sum(axis = 1) #(nparticipants)
    sample_likelihood = participant_likelihood.sum(axis = 0) #(K)
    return (sample_likelihood * mixture).sum()


def ucb(mean, var, explore):
    
    utility = (mean[:,:,:,None] * (1 - explore) + var[:,:,:,None] * explore) #(nparticipants, ntrials, choices, K)
    if np.count_nonzero(utility) == 0:
        utility += 1
    return utility / utility.sum()
    
"""
    center = utility.max(axis = 2, keepdims = True)  #(nparticipants, ntrials, 1, K)
    centered_utility = utility - center #(nparticipants, ntrials, choices, K)
    
    exp_u = np.exp(centered_utility / temperature) #(nparticipants, ntrials, choices)
    exp_u_row_sums = exp_u.sum(axis = 2, keepdims = True) #(nparticipants, ntrials, 1, K)
    likelihood = exp_u / exp_u_row_sums
    return likelihood #(nparticipants, ntrials, choices, K)
"""

def local(linear_interp, last_action, learning_rate):
    
    choices = np.arange(len(linear_interp))
    if last_action == -1:
        return choices**0
    gradient = np.gradient(linear_interp)
    next_action = last_action + gradient * learning_rate
    return np.exp(abs(next_action - choices))
    


    
    