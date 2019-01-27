import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
import pandas as pd

from generate_data import vectorize_all_responses



def softmax(utility, temperature):

    center = utility.max(axis=1, keepdims=True)
    centered_utility = utility - center
    
    exp_u = np.exp(centered_utility / temperature[None,:,None,None])
    exp_u_row_sums = exp_u.sum(axis = 2, keepdims = True) 
    likelihood = exp_u / exp_u_row_sums
    return likelihood


def likelihood(actions, mean, var, trials, steepness, x_midpoint, yscale, temperature, weights):
    
    position = trials[:,None] - x_midpoint[None,:]
    growth = np.exp(-steepness * position)
    denom = 1 + growth * yscale
    explore_param = 1 - (1. / denom)
    
    all_utility = mean[:,None,:,:]*(1-explore_param)[:,:,None,None] + var[:,None,:,:]*explore_param[:,:,None,None]    
    all_likelihood = softmax(all_utility, temperature)
    
    action_likelihood_group = (all_likelihood * actions[:,None,:,:]).sum(axis=2)
    action_likelihood = (action_likelihood_group * weights[None,:,None]).sum(axis=1)   
    log_action_likelihood = action_likelihood.log()
    log_likelihood = log_action_likelihood.sum()
    return log_likelihood


def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining


def group_ucb_model(X, steepness_alpha=1., steepness_beta=1., x_midpoint_prec=4., yscale_alpha=1., yscale_beta=.5,
                     temperature_alpha=1., temperature_beta=1., maxk=10, samples=200):
    
    ntrials = X.shape[1]
    trials = np.arange(ntrials)
    actions = theano.shared(X[0])
    mean = theano.shared(X[1])
    var = theano.shared(X[2])
    with pm.Model() as model:
    
        steepness = pm.Gamma('steepness', steepness_alpha, steepness_beta, shape=maxk)
        BoundNormal = pm.Bound(pm.Normal, lower=0, upper=ntrials)
        x_midpoint = BoundNormal('x_midpoint', mu=ntrials/2., sd=ntrials/x_midpoint_prec, shape=maxk)
        yscale = pm.Gamma('yscale', yscale_alpha, yscale_beta, shape=maxk)
        temperature = pm.Gamma('temperature', temperature_alpha, temperature_beta, shape=maxk)
        
        alpha = pm.Gamma('alpha', 10**-10., 10**-10.)
        beta = pm.Beta('beta', 1., alpha, shape=maxk)
        weights = pm.Deterministic('w', stick_breaking(beta))
    
        obs = pm.Potential('obs', likelihood(actions, mean, var, trials, steepness, x_midpoint, yscale, temperature, weights))
        trace = pm.sample(samples, njobs=4)
        return trace