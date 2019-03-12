import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
import pandas as pd

from generate_responses import vectorize_all_responses



def softmax(utility, temperature):

    center = utility.max(axis=1, keepdims=True)
    centered_utility = utility - center
    
    exp_u = np.exp(centered_utility / temperature[None,:,None,None])
    exp_u_row_sums = exp_u.sum(axis = 2, keepdims = True) 
    likelihood = exp_u / exp_u_row_sums
    return likelihood


def sparse_group_static_ucb_mes_likelihood(actions, mean, var, mes, random_likelihood, method, explore_param, temperature, assignments, maxk):
    
    ucb_utility = mean[:,None,:,:] + var[:,None,:,:]*explore_param[None,:,None,None]
    ucb_likelihood = softmax(ucb_utility, temperature)
    mes_likelihood = softmax(mes[:,None,:,:], temperature)
    
    mu_print = tt.printing.Print('assignments')(assignments.shape)
    mu_print = tt.printing.Print('random_likelihood')(random_likelihood.shape)
    mu_print = tt.printing.Print('ucb_likelihood')(ucb_likelihood.shape)
    mu_print = tt.printing.Print('mes_likelihood')(mes_likelihood.shape)
    
    stacked_likelihood = tt.stack([ucb_likelihood, mes_likelihood, random_likelihood])
    weighted_likelihood = stacked_likelihood * method.T[:,None,:,None,None]
    all_likelihood = weighted_likelihood.sum(axis=0)
        
    action_likelihood_group = (all_likelihood * actions[:,None,:,:]).sum(axis=2)
    assignments_one_hot = tt.extra_ops.to_one_hot(assignments, maxk)    
    action_likelihood = (action_likelihood_group * assignments_one_hot.T[None,:,:]).sum(axis=1)
        
    log_action_likelihood = action_likelihood.log()
    log_likelihood = log_action_likelihood.sum()
    
    mu_print = tt.printing.Print('log_likelihood')(log_likelihood)
    return log_likelihood
    


def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
    return beta * portion_remaining


def group_static_ucb_mes_model(X, explore_param_alpha=.01, explore_param_beta=.01,
                     temperature_alpha=1., temperature_beta=10., method_alpha=.001, maxk=10, samples=200):
    
    X = X.copy().transpose((2,1,3,0))
    nparticipants = X.shape[3]
    nchoices = X.shape[2]
    ntrials = X.shape[1]
    actions = theano.shared(X[-1])
    mean = theano.shared(X[0])
    var = theano.shared(X[1])
    mes = theano.shared(X[3])
    #mes = theano.shared(X[4])
    random_likelihood = theano.shared((1./nchoices)*np.ones(shape=(ntrials, maxk, nchoices, nparticipants)))
    with pm.Model() as model:
    
        explore_param = pm.Gamma('var_param', explore_param_alpha, explore_param_beta, shape=maxk)
        temperature = pm.Gamma('temperature', temperature_alpha, temperature_beta, shape=maxk)
        method = pm.Dirichlet('method', np.ones(3)*method_alpha, shape=(maxk,3))
        
        alpha = pm.Gamma('alpha', 10**-10., 10**-10.)
        beta = pm.Beta('beta', 1., alpha, shape=maxk)
        weights = pm.Deterministic('w', stick_breaking(beta))
        assignments = pm.Categorical('assignments', weights, shape=nparticipants)
    
        obs = pm.Potential('obs', sparse_group_static_ucb_mes_likelihood(actions, mean, var, mes, random_likelihood, method, explore_param, temperature, assignments, maxk))
        trace = pm.sample(samples, njobs=4)
        return trace