import numpy as np
import scipy.stats as st
from mes import mes, gumbel_pdf
from utils import gp


def sgd_acq(observed_x, observed_y, domain, gamma_loc, gamma_shape, initial_var, eps, trials_remaining):
    n = len(observed_x)
    domain_range = len(domain)
    if n == 0:
        p = np.ones(domain_range) / domain_range
    elif n == 1:
        p = np.array([st.norm(observed_x[0], initial_var).pdf(x) if x != observed_x[0] else 0 for x in domain])
    else:        
        dk = (observed_y[n - 1] - observed_y[n - 2]) / (observed_x[n - 1] - observed_x[n - 2])        
        learning_rate = [max((next_x - observed_x[-1]) / dk, 1 / abs(dk)) for next_x in domain]        
        p = np.array([st.gamma(gamma_shape, loc = gamma_loc).pdf(lr) for lr in learning_rate])
    if np.nan_to_num(p.sum()) == 0:
        p = np.array([eps if x != observed_x[-1] else 1. for x in domain])
    return {}, p / p.sum()


def mes_acq(observed_x, observed_y, domain, kern, precision, upper_bound, nsamples, decision, decision_params, trials_remaining):
    mean, var = gp(observed_x, observed_y, kern, domain)    
    std = np.sqrt(var)
    gumbel_params, utility = mes(mean, std, precision, upper_bound, nsamples)
    a, b = gumbel_params
    pymax = gumbel_pdf(a, b, np.linspace(0, upper_bound))
    return {'pymax':pymax, 'utility':utility, 'mean':mean, 'std':std}, decision(utility, *decision_params)


def var_greedy_acq(observed_x, observed_y, domain, kern, decision, decision_params, trials_remaining):
    mean, var = gp(observed_x, observed_y, kern, domain)    
    std = np.sqrt(var)
    return {'utility':std, 'mean':mean, 'std':std}, decision(std, *decision_params)


def mean_greedy_acq(observed_x, observed_y, domain, kern, decision, decision_params):
    mean, var = gp(observed_x, observed_y, kern, domain)    
    std = np.sqrt(var)
    p = decision(mean, *decision_params)
    
    return {'utility':mean, 'mean':mean, 'std':std}, p


def weighted_mes_mean_greedy(observed_x, observed_y, domain, kern, precision, upper_bound, nsamples, weight, decision, decision_params, trials_remaining):
    mean, var = gp(observed_x, observed_y, kern, domain)    
    std = np.sqrt(var)
    gumbel_params, mes_utility = mes(mean, std, precision, upper_bound, nsamples)
    a, b = gumbel_params
    pymax = gumbel_pdf(a, b, np.linspace(0, upper_bound))
    utility = mean.ravel() + weight * trials_remaining * mes_utility
    return {'pymax':pymax, 'utility':utility, 'mean':mean, 'std':std}, decision(utility, *decision_params)


def mes_mean_greedy_last(observed_x, observed_y, domain, kern, precision, upper_bound, nsamples, decision, decision_params, trials_remaining):
    if trials_remaining > 0:
        return mes_acq(observed_x, observed_y, domain, kern, precision, upper_bound, nsamples, decision, decision_params, trials_remaining)
    else:
        return mean_greedy_acq(observed_x, observed_y, domain, kern, decision, decision_params)