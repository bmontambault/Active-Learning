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


def group_static_ucb_mes_likelihood(actions, mean, var, mes, var_explore, mes_explore, temperature, assignments, maxk):
    
    #explore_variable = var[:,None,:,:]*(1-explore_method[None,:,None,None]) + mes[:,None,:,:]*explore_method[None,:,None,None]
    #all_utility = mean[:,None,:,:] + var[:,None,:,:]*explore_variable
    all_utility = mean[:,None,:,:] + var[:,None,:,:]*var_explore[None,:,None,None] + mes[:,None,:,:]*mes_explore[None,:,None,None]   
    all_likelihood = softmax(all_utility, temperature)
    
    action_likelihood_group = (all_likelihood * actions[:,None,:,:]).sum(axis=2)
    assignments_one_hot = tt.extra_ops.to_one_hot(assignments, maxk)    
    action_likelihood = (action_likelihood_group * assignments_one_hot.T[None,:,:]).sum(axis=1)
    
    log_action_likelihood = action_likelihood.log()
    log_likelihood = log_action_likelihood.sum()
    return log_likelihood


def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining


def group_static_ucb_mes_model(X, explore_param_alpha=.1, explore_param_beta=.1,
                     temperature_alpha=1., temperature_beta=1., explore_method_p=.5, maxk=10, samples=200):
    
    nparticipants = X.shape[3]
    actions = theano.shared(X[0])
    mean = theano.shared(X[1])
    var = theano.shared(X[2])
    mes = theano.shared(X[3])
    with pm.Model() as model:
    
        #explore_param = pm.Gamma('explore_param', explore_param_alpha, explore_param_beta, shape=maxk)
        var_param = pm.Gamma('var_param', explore_param_alpha, explore_param_beta, shape=maxk)
        mes_param = pm.Gamma('mes_param', explore_param_alpha, explore_param_beta, shape=maxk)
        temperature = pm.Gamma('temperature', temperature_alpha, temperature_beta, shape=maxk)
        #explore_method = pm.Bernoulli('explore_method', p=explore_method_p, shape=maxk)
        
        alpha = pm.Gamma('alpha', 10**-10., 10**-10.)
        #alpha = pm.HalfNormal('alpha', sd = 1000000)
        beta = pm.Beta('beta', 1., alpha, shape=maxk)
        weights = pm.Deterministic('w', stick_breaking(beta))
        assignments = pm.Categorical('assignments', weights, shape=nparticipants)
    
        obs = pm.Potential('obs', group_static_ucb_mes_likelihood(actions, mean, var, mes, var_param, mes_param, temperature, assignments, maxk))
        #step = pm.Metropolis()
        trace = pm.sample(samples, njobs=4)#, step=step)
        return trace