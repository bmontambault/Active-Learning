import numpy as np
import pymc3 as pm
import theano
import pandas as pd

from generate_data import vectorize_all_responses



def softmax(utility, temperature):

    center = utility.max(axis=1, keepdims=True)
    centered_utility = utility - center    
    exp_u = np.exp(centered_utility / temperature[:,None])
    exp_u_row_sums = exp_u.sum(axis = 1, keepdims = True) 
    likelihood = exp_u / exp_u_row_sums
    return likelihood


def ucb_likelihood(actions, mean, var, trials, steepness, x_midpoint, yscale, temperature):
    
    
    position = trials - x_midpoint
    growth = np.exp(-steepness * position)
    denom = 1 + growth * yscale
    explore_param = 1 - (1. / denom)
    
    all_utility = mean*(1-explore_param[:,None]) + var*explore_param[:,None]
    all_likelihood = softmax(all_utility, temperature)
    action_likelihood = (all_likelihood * actions).sum(axis=1)
    log_action_likelihood = np.log(action_likelihood)
    log_likelihood = log_action_likelihood.sum()
    return log_likelihood


def single_ucb_model(X, steepness_alpha=1., steepness_beta=1., x_midpoint_prec=4., yscale_alpha=1., yscale_beta=.5,
                     temperature_alpha=1., temperature_beta=1., samples=200):
    
    ntrials = X.shape[1]
    trials = np.arange(ntrials)
    actions = theano.shared(X[0])
    mean = theano.shared(X[1])
    var = theano.shared(X[2])
    with pm.Model() as model:
    
        steepness = pm.Gamma('steepness', steepness_alpha, steepness_beta, shape=1)
        BoundNormal = pm.Bound(pm.Normal, lower=0, upper=ntrials)
        x_midpoint = BoundNormal('x_midpoint', mu=ntrials/2., sd=ntrials/x_midpoint_prec, shape=1)
        yscale = pm.Gamma('yscale', yscale_alpha, yscale_beta, shape=1)
        temperature = pm.Gamma('temperature', temperature_alpha, temperature_beta, shape=1)
    
        obs = pm.Potential('obs', likelihood(actions, mean, var, trials, steepness, x_midpoint, yscale, temperature))
        trace = pm.sample(samples, njobs=1)
        return trace
    

def single_ucb_model_all(X, fmax, steepness_alpha=1., steepness_beta=1., x_midpoint_prec=4., yscale_alpha=1., yscale_beta=.5,
                     temperature_alpha=1., temperature_beta=1., samples=200):
    
    df_list = []
    actions = X[0].argmax(axis=1)
    for i in range(X.shape[3]):
        trace = single_ucb_model(X[:,:,:,i], steepness_alpha, steepness_beta, x_midpoint_prec, yscale_alpha, yscale_beta, temperature_alpha, temperature_beta, samples)
        df = pm.trace_to_dataframe(trace)
        df['participant'] = i
        df_list.append(df)
    all_dfs = pd.concat(df_list)
    all_dfs['actions'] = all_dfs['participant'].apply(lambda x: actions[:,x])
    all_dfs['fmax'] = all_dfs['participant'].apply(lambda x: fmax[x])
    return all_dfs
    
    
#X, functions, goals = vectorize_all_responses()
#fmax = functions.argmax(axis=1)
