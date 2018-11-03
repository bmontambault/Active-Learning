import GPy
import numpy as np
import scipy.stats as st
from inspect import signature


def get_mean_var(kernel, actions, rewards, choices):
    
    if len(actions) == 0:
        return np.ones(len(choices))[:,None], np.ones(len(choices))[:,None]
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


def generate_cluster(model, params, kernel, function, ntrials, nparticipants, deterministic=False):
    
    #print (params)
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
            
            if len(actions) == 0:
                linear_interp = np.zeros(len(choices))
            else:
                linear_interp = np.interp(choices, actions, rewards)
            params['linear_interp'] = linear_interp
            
            if trial == 0:
                one_hot_last_actions = np.zeros(len(choices))
            else:
                one_hot_last_actions = participant_responses[trial - 1][2]
            params['last_actions'] = one_hot_last_actions
            
            trial_array = np.ones(len(mean)) * trial
            params['trial'] = trial_array.mean()
            
            parameter_names = list(signature(model).parameters)
            model_params = {p: params[p] for p in parameter_names}
            likelihood = model(**model_params).ravel()
            
            if deterministic:
                best = [i for i in range(len(likelihood)) if likelihood[i] == likelihood.max()]
                action = np.random.choice(best)
            else:
                action = st.rv_discrete(values = (choices, likelihood)).rvs()            
            actions.append(action)
            rewards.append(function[action])
            
            one_hot_actions = np.zeros(len(choices))
            one_hot_actions[action] = 1
            
            response = np.array([mean.ravel(), var.ravel(), linear_interp, one_hot_last_actions, trial_array, one_hot_actions])
            participant_responses.append(response)
        all_responses.append(participant_responses)
    return np.array(all_responses) #(nparticipants, ntrials, function variables, choices)


def generate(model, params, kernel, function, ntrials, nparticipants, K, mixture, deterministic=False):
    
    cluster_sizes = [int(nparticipants * mixture[i]) for i in range(K)]
    clusters = np.hstack([np.ones(cluster_sizes[i]) * i for i in range(len(cluster_sizes))])
    results = np.vstack([generate_cluster(model, {p: np.array([params[p][i]]) if p not in ['mean', 'var'] else params[p] for p in params.keys()}, kernel, function, ntrials, cluster_sizes[i], deterministic=deterministic) for i in range(K)])
    return results, clusters
    

def likelihood(X, model, params, K, mixture, c=0):
    
    mean = X[:,:,0] #(nparticipants, ntrials, choices)
    var = X[:,:,1] #(nparticipants, ntrials, choices)
    linear_interp = X[:,:,2]
    last_actions = X[:,:,3]
    
    trial = X[:,:,4].mean(axis=2).mean(axis=0)
    params['mean'] = mean
    params['var'] = var
    params['linear_interp'] = linear_interp
    params['last_actions'] = last_actions
    params['trial'] = trial
    
    parameter_names = list(signature(model).parameters)
    model_params = {p: params[p] for p in parameter_names}
    
    likelihood = np.log(model(**model_params) * c) #(nparticipants, ntrials, choices, K)
    action = X[:,:,5]
    
    action_likelihood = (np.nan_to_num(likelihood) * (action[:,:,:,None] * np.ones(K))).sum(axis=2) #(nparticipants, ntrials, K)
    participant_likelihood = action_likelihood.sum(axis=1) #(nparticipants)
    sample_likelihood = participant_likelihood.sum(axis=0) #(K)
    return (sample_likelihood * mixture).sum()


def ucb(mean, var, explore):
    """
    Returns ucb function values where there is one explore paramter value for all trials
    Returns ucb function values when there are explore parameter values for each trial
    Patameters:
        mean: (participants, trials, choices)
        var: (participants, trials, choices)
        explore: K
    """
    utility = (mean[:,:,:,None] * (1 - explore) + var[:,:,:,None] * explore) #(nparticipants, ntrials, choices, K)
    return utility


def ucb_by_trial(mean, var, explore):
    """
    Returns ucb function values when there are explore parameter values for each trial
    Patameters:
        mean: (participants, trials, choices)
        var: (participants, trials, choices)
        explore: (trials, K)
    """
    utility = (mean[:,:,:,None] * (1 - explore[:,:,None]) + var[:,:,:,None] * explore[:,:,None]) #(participants, trials, choices, K)
    return utility


def propto(utility):
    """
    Returns normalized utility
    Parameters:
        utility: (participants, trials, choices, K)
    """
    normalized_utility = (utility / utility.sum(axis=2, keepdims=True)) #(participants, trials, choices, K)
    return normalized_utility


def softmax(utility, temperature):
    """
    Returns likelihood given utility and temperature
    Parameters:
        utility: (participants, trials, choices, K)
        temperature: (K)
    """
    normalized_utility = propto(utility)
    center = normalized_utility.max(axis = 2, keepdims = True)  #(nparticipants, ntrials, 1, K)
    centered_utility = normalized_utility - center #(nparticipants, ntrials, choices, K)
    
    exp_u = np.exp(centered_utility / temperature) #(nparticipants, ntrials, choices)
    exp_u_row_sums = exp_u.sum(axis = 2, keepdims = True) #(nparticipants, ntrials, 1, K)
    likelihood = exp_u / exp_u_row_sums #(nparticipants, ntrials, choices, K)
    return likelihood


def softmax_ucb_acq(mean, var, explore, temperature):
    
    utility = ucb(mean, var, explore)
    likelihood = softmax(utility, temperature)
    return likelihood


def phase_ucb_acq(mean, var, trial, steepness, midpoint, temperature):
    
    explore = (1 - 1. / (1 + np.exp(-steepness[:,None] * (trial - midpoint[:,None]))).T)
    utility = ucb_by_trial(mean, var, explore)    
    likelihood = softmax(utility, temperature)
    return likelihood
    
    

def local(linear_interp, last_actions, learning_rate):
    
    choices = np.arange(len(linear_interp))
    if np.count_nonzero(last_actions) == 0:
        return np.ones(len(choices)) / len(choices)
    else:
        gradient = np.gradient(linear_interp)
        last_action = np.argmax(last_actions, axis = 2)
        next_action = last_action + gradient * learning_rate
        return np.exp(abs(next_action - choices))
    


    
    