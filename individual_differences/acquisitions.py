import numpy as np
from theano import tensor as tt
import scipy.stats as st
from inspect import signature

from gp_utils import get_mean_var, get_mes_utility


def generate(model, params, kernel, function, nparticipants, ntrials, K, mixture):
    
    choices = np.arange(len(function))
    participants = (mixture * nparticipants).astype(int)
    participants[-1] = nparticipants - participants[:-1].sum()
    split_matrix = np.repeat(np.eye(K), participants, axis=0)
        
    actions = [[] for j in range(nparticipants)]
    rewards = [[] for j in range(nparticipants)]
    action_vec = np.zeros(shape=(nparticipants,len(function))) #last action
    all_actions_vec = np.zeros(shape=(nparticipants,len(function)))
    all_X = None
    for i in range(ntrials):
        
        #get information about the current trial
        trial = np.zeros(shape=(nparticipants, 1, len(function)))
        trial[:,:,i] = np.ones(nparticipants)[:,None]
        if i < 2:
            is_first_two = np.ones(shape=(nparticipants, 1, len(function)))
        else:
            is_first_two = np.zeros(shape=(nparticipants, 1, len(function)))
            
        #get gradient and participant choices (add participant dimmension to choices)
        if i == 0:
            gradient = np.zeros(shape=(nparticipants, 1, len(function)))
        else:
            sorted_actions = np.sort(actions, axis=1)
            sorted_rewards = function[sorted_actions]
            linear_interp = np.array([np.interp(choices, sorted_actions[j], sorted_rewards[j]) for j in range(nparticipants)])[:,None,:]
            gradient = np.gradient(linear_interp, axis=2)
            all_actions_vec += action_vec
        
        participant_choices = choices * np.ones((nparticipants,1))[:,None,:]
        
        #calculate values for gp acquisition functions
        mean_var = np.array([get_mean_var(kernel, actions[j], rewards[j], choices) for j in range(nparticipants)])#.transpose(0,2,1)
        mes_utility = np.array([get_mes_utility(*mv) for mv in mean_var])[:,None,:]
        
        #combine data
        X = np.hstack((trial, is_first_two, action_vec[:,None,:], all_actions_vec[:,None,:],
                       gradient, participant_choices,
                       mean_var, mes_utility))[:,:,:,None]
        
        likelihood, utility = get_likelihood_utility(X, model, params, K, mixture)
        likelihood = (likelihood[0] * split_matrix[:,:,None]).sum(axis=1)
        utility = (utility[0] * split_matrix[:,:,None]).sum(axis=1)
        
        action_vec = np.array([np.random.multinomial(1, likelihood[j]) for j in range(nparticipants)])
        action_idx = action_vec.argmax(axis=1)[:,None]
        reward = function[action_idx]
                
        # add actions and utility/likelihood
        X = np.hstack((X, action_vec[:,None,:,None]))
        X = np.hstack((X, utility[:,None,:,None]))
        X = np.hstack((X, likelihood[:,None,:,None]))

        if i == 0:
            actions = action_idx
            rewards = reward
            all_X = X
        else:
            actions = np.hstack((actions, action_idx))
            rewards = np.hstack((rewards, reward))
            all_X = np.vstack((all_X.T, X.T)).T
        
    return all_X


def get_params(X, model, params, k):
    
    #get information about trials and previous actions
    trial = X[:,0,:,:]
    is_first_two = X[:,1,:,:]
    last_actions = X[:,2,:,:]
    all_actions = X[:,3,:,:]
    params['trial'] = trial
    params['is_first_two'] = is_first_two
    params['last_actions'] = last_actions
    params['all_actions'] = all_actions
    
    #get linear interpolation and possible choices
    gradient = X[:,4,:,:]
    choices = X[:,5,:,:]
    params['gradient'] = gradient
    params['choices'] = choices
    
    #get values for gp acquisition functions
    mean = X[:,6,:,:]
    var = X[:,7,:,:]
    mes_utility = X[:,8,:,:]
    params['mean'] = mean
    params['var'] = var
    params['mes_utility'] = mes_utility
    params['k'] = k
    
    parameter_names = list(signature(model).parameters)
    model_params = {p: params[p] for p in parameter_names}
    return model_params
    
    
def get_likelihood_utility(X, model, params, K, mixture):
    
    params = get_params(X, model, params, K)
    return model(**params)


def log_likelihood(X, model, params, K, mixture):
    
    actions = X[:,8,:,:]
    likelihood, utility = get_likelihood_utility(X, model, params, K, mixture)
    mixture_likelihood = (likelihood.transpose(1,3,0,2) * actions[:,:,:,None]).sum(axis=1)
    action_likelihood = (mixture_likelihood * mixture[None,None,:]).sum(axis=2)
    
    participant_log_likelihood = np.log(action_likelihood).sum(axis=1)
    log_likelihood = participant_log_likelihood.sum()
    return log_likelihood
    


def local_acq(gradient, last_actions, all_actions, choices, is_first_two, learning_rate, stay_penalty, temperature):
    
    last_gradient = (gradient * last_actions).sum(axis=1)
    actions_x = last_actions.argmax(axis=1)
    next_action = actions_x[:,:,None] + last_gradient[:,:,None] * learning_rate
    distance = np.abs(next_action[:,None,:,:] - choices[:,:,:,None])
    dist_utility = np.exp(-distance) * (1 - is_first_two[:,:,:,None])
    
    move_utility = is_first_two[:,:,:,None] + dist_utility
    penalty_utility = all_actions[:,:,:,None] * stay_penalty
    utility = move_utility - penalty_utility
    
    #penalty = (all_actions[:,:,:,None] * stay_penalty)
    #utility = move_utility + penalty
    
    utility = utility.transpose(2,0,3,1)
    likelihood = softmax1(utility, temperature)
    #print (likelihood.shape)
    return likelihood, utility



def phase_ucb_acq(mean, var, trial, steepness, x_midpoint, yscale, temperature):
    
    trial = trial.argmax(axis=1)
    position = trial - x_midpoint[:,None,None] #(k, nparticipants, ntrials)
    growth = np.exp(-steepness[:,None, None] * position) #(k, nparticipants, ntrials)
    denom = 1 + growth * yscale[:,None,None] #(k, nparticipants, ntrials)
    explore_param = 1 - (1. / denom) #(k, nparticipants, ntrials)
    exploit = (mean.transpose(1, 0, 2)[:,None] * (1- explore_param)).T #(ntrials, nparticipants, k, nchoices)
    explore = (var.transpose(1, 0, 2)[:,None] * (explore_param)).T #(ntrials, nparticipants, k, nchoices)
    
    utility = exploit + explore
    likelihood = softmax1(utility, temperature)
    return likelihood, utility


def mes_acq(mes_utility, temperature, k):
    
    utility = mes_utility.transpose(2,0,1)[:,:,None,:]
    utility = np.repeat(utility, k, axis=2)
    likelihood = softmax1(utility, temperature)
    return likelihood, utility


def generate_mixture_acq(mean, var, trial, gradient, last_actions, all_actions, choices, is_first_two, mes_utility,
                steepness, x_midpoint, yscale, phase_ucb_temperature, learning_rate, stay_penalty, local_temperature,
                mes_temperature, k, mixture):
    
    phase_ucb_u, phase_ucb_l = phase_ucb_acq(mean, var, trial, steepness, x_midpoint, yscale, phase_ucb_temperature)
    local_u, local_l = local_acq(gradient, last_actions, all_actions, choices, is_first_two, learning_rate, stay_penalty, local_temperature)
    mes_u, mes_l = mes_acq(mes_utility, mes_temperature, k)
    
    all_utility = np.array([phase_ucb_u, local_u, mes_u])
    all_likelihood = np.array([phase_ucb_l, local_l, mes_l])
    
    utility = (all_utility * mixture.T[:,None,None,:,None]).sum(axis=0)
    likelihood = (all_likelihood * mixture.T[:,None,None,:,None]).sum(axis=0)
    return utility, likelihood


def mixture_acq(mean, var, trial, gradient, last_actions, all_actions, choices, is_first_two, mes_utility,
                steepness, x_midpoint, yscale, phase_ucb_temperature, learning_rate, stay_penalty, local_temperature,
                mes_temperature, k, mixture):
    
    phase_ucb_u, phase_ucb_l = phase_ucb_acq(mean, var, trial, steepness, x_midpoint, yscale, phase_ucb_temperature)
    local_u, local_l = local_acq(gradient, last_actions, all_actions, choices, is_first_two, learning_rate, stay_penalty, local_temperature)
    mes_u, mes_l = mes_acq(mes_utility, mes_temperature, k)
    
    all_utility = tt.stack([phase_ucb_u, local_u, mes_u])
    all_likelihood = tt.stack([phase_ucb_l, local_l, mes_l])
    
    utility = (all_utility * mixture.T[:,None,None,:,None]).sum(axis=0)
    likelihood = (all_likelihood * mixture.T[:,None,None,:,None]).sum(axis=0)
    return utility, likelihood    
    
    
def softmax1(utility, temperature):
    """
    Returns likelihood given utility and temperature
    Parameters:
        utility: (ntrials, nparticipants, k, nchoices)
        temperature: (K)
    """
    center = utility.max(axis = 3, keepdims = True)  #(ntrials, nparticipants, K, 1)
    centered_utility = utility - center #(ntrials, nparticipants, K, choices)
    
    exp_u = np.exp(centered_utility / temperature[:,None]) #(ntrials, nparticipants, K, choices)
    exp_u_row_sums = exp_u.sum(axis = 3, keepdims = True) #(ntrials, nparticipants, K, 1)
    likelihood = exp_u / exp_u_row_sums #(ntrials, nparticipants, K, choices)
    return likelihood