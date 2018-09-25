import numpy as np


def local(mean, last_action, choices, learning_rate):
    
    if last_action == -1:
        return np.ones(len(choices))
    full_gradient = np.gradient(mean)
    gradient = full_gradient[last_action]
    next_action = last_action + gradient * learning_rate
    return np.array([np.exp(-abs(next_action - x)) for x in choices]).ravel()


def ucb(mean, var, trial, choices, center, steepness):
    
    exploit = 1. / (1 + np.exp(-steepness * (trial - center)))
    return exploit * mean + (1 - exploit) * var


def mes(mean, var, choices):
    
    return


def acquisition(mean, var, last_action, trial, choices, learning_rate, center, steepness, w):
    
    local_u = local(mean, last_action, choices, learning_rate)
    ucb_u = ucb(mean, var, trial, choices, center, steepness)
    mes_u = mes(mean, var, choices)
    utility = np.array([local_u, ucb_u, mes_u])
    return w * utility


def softmax(utility, temperature):
    
    centered_utility = utility - np.nanmax(utility)
    exp_u = np.exp(centered_utility / temperature)
    return exp_u / np.nansum(exp_u)


def likelihood(mean, var, last_action, trial, choices, action, learning_rate, center, steepness, w, temperature):
    
    utility = acquisition(mean, var, last_action, trial, choices, learning_rate, center, steepness, w)
    all_likelihoods = softmax(utility, temperature)
    return all_likelihoods[action]