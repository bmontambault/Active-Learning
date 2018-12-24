import GPy
import numpy as np
import scipy.stats as st
from inspect import signature


def get_mean_var(kernel, actions, rewards, choices):
    
    if len(actions) == 0:
        return np.ones(len(choices)), np.ones(len(choices))
    else:
        X = np.array(actions)[:,None]
        Y = np.array(rewards)[:,None]
        m = GPy.models.GPRegression(X, Y, kernel)
        m.Gaussian_noise.variance = .00001
        mean, var = m.predict(np.array(choices)[:,None])
        return mean.ravel(), var.ravel()
    
    
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

def get_function_samples(kernel, observed_x, observed_y, all_x, size):
    
    X = np.array(observed_x)[:,None]
    Y = np.array(observed_y)[:,None]
    m = GPy.models.GPRegression(X, Y, kernel)
    m.Gaussian_noise.variance = .00001
    f = m.posterior_samples_f(np.array(all_x)[:,None], full_cov = True, size = size)
    return f
    

def z_score(y, mean, std):
    
    if type(y) == list:
        y = np.array(y)[None,:]
    mean = mean.reshape(len(mean), 1)
    std = std.reshape(len(std), 1)
    return (y - mean)/std  


def ymax_cdf(y, mean, std):
        return np.prod(st.norm.cdf(z_score(y, mean, std)), axis = 0)


def inv_ymax_cdf(p, mean, std, precision):
    
    upper_bound = precision
    while True:
        if ymax_cdf(upper_bound, mean, std) >= p:
            break
        else:
            upper_bound *= 2
    
    Z = np.arange(0, upper_bound + precision, precision)
    while len(Z) > 2:
        bisect = len(Z) // 2
        z = Z[bisect]
        prob = ymax_cdf(z, mean, std)
        if prob > p:
            Z = Z[:bisect + 1]
        else:
            Z = Z[bisect:]
    return np.mean(Z)
    

def fit_gumbel(mean, std, precision):
    r1 = .25
    r2 = .75
    y1 = inv_ymax_cdf(r1, mean, std, precision)
    y2 = inv_ymax_cdf(r2, mean, std, precision)
    if y1 == y2:
        y1 -= precision
    R = np.array([[1., np.log(-np.log(r1))], [1., np.log(-np.log(r2))]])
    Y = np.array([y1, y2])
    a, b = np.linalg.solve(R, Y)
    return a, b


def sample_gumbel(a, b, nsamples):
    return [a - (b * np.log(-np.log(r))) for r in np.random.uniform(0, 1, int(nsamples))]


def gumbel_pdf(a, b, y):
    z = (y - a) / b
    return - ((1 / b) * np.exp(-(z + np.exp(-z))))


def get_mes_utility(mean, var, samples=100):
    std = np.sqrt(var)
    a, b = fit_gumbel(mean, std, .01)
    ymax_samples = sample_gumbel(a, b, samples)
    z_scores = z_score(ymax_samples, mean, std)
    log_norm_cdf_z_scores = st.norm.logcdf(z_scores)
    log_norm_pdf_z_scores = st.norm.logpdf(z_scores)
    return np.mean(z_scores * np.exp(log_norm_pdf_z_scores - (np.log(2) + log_norm_cdf_z_scores)) - log_norm_cdf_z_scores, axis = 1)


def generate_cluster(model, params, kernel, function, ntrials, nparticipants, deterministic=False):
    
    choices = np.arange(len(function))
    all_responses = []
    for participant in range(nparticipants):
        actions = []
        rewards = []
        participant_responses = []
        for trial in range(ntrials):
            mean, var = get_mean_var(kernel, actions, rewards, choices)
            mes_utility = get_mes_utility(mean, var)
            params['mean'] = mean.T[None]
            params['var'] = var.T[None]
            params['mes_utility'] = mes_utility[None,None,:]
            params['choices'] = choices[None,None,:]
            
            if len(actions) == 0:
                linear_interp = np.zeros(len(choices))
            else:
                sorted_actions = sorted(list(set(actions)))
                sorted_rewards = [function[a] for a in sorted_actions]
                linear_interp = np.interp(choices, sorted_actions, sorted_rewards)
            params['linear_interp'] = linear_interp[None,None,:]
            
            if trial == 0:
                one_hot_last_actions = np.zeros(len(choices))
            else:
                one_hot_last_actions = participant_responses[trial - 1][5]
            params['last_actions'] = one_hot_last_actions[None,None,:]
            
            trial_array = np.ones(len(mean)) * trial
            params['trial'] = trial_array.mean()
            
            if trial < 2:
                is_first = np.ones(len(choices))
            else:
                is_first = np.zeros(len(choices))
            params['is_first'] = is_first[None,None,:]
            
            one_hot_all_actions = np.zeros(len(choices))
            for a in actions:
                one_hot_all_actions[a] = 1
            params['all_actions'] = one_hot_all_actions[None,None,:]
            
            parameter_names = list(signature(model).parameters)
            model_params = {p: params[p] for p in parameter_names}
            utility, likelihood = model(**model_params)
            utility = utility.ravel()
            likelihood = likelihood.ravel()
            
            if deterministic:
                best = [i for i in range(len(likelihood)) if likelihood[i] == likelihood.max()]
                action = np.random.choice(best)
            else:
                action = st.rv_discrete(values = (choices, likelihood)).rvs()            
            actions.append(action)
            rewards.append(function[action])
            
            one_hot_actions = np.zeros(len(choices))
            one_hot_actions[action] = 1
            
            response = np.array([mean.ravel(), var.ravel(), linear_interp, one_hot_last_actions, trial_array, one_hot_actions, is_first, choices, one_hot_all_actions, mes_utility, utility, likelihood])
            participant_responses.append(response)
        all_responses.append(participant_responses)
    return np.array(all_responses) #(nparticipants, ntrials, function variables, choices)


def generate(model, params, kernel, function, ntrials, nparticipants, K, mixture, deterministic=False):
    
    cluster_sizes = [int(nparticipants * mixture[i]) for i in range(K)]
    clusters = np.hstack([np.ones(cluster_sizes[i]) * i for i in range(len(cluster_sizes))])
    results = np.vstack([generate_cluster(model, {p: np.array([params[p][i]]) if p not in ['mean', 'var'] else params[p] for p in params.keys()}, kernel, function, ntrials, cluster_sizes[i], deterministic=deterministic) for i in range(K)])
    return results, clusters



def generate1(model, params, kernel, function, nparticipants, ntrials, K, mixture):
    
    choices = np.arange(len(function))
    participants = (mixture * nparticipants).astype(int)
    participants[-1] = nparticipants - participants[:-1].sum()
    split_matrix = np.repeat(np.eye(K), participants, axis=0)
        
    actions = [[] for j in range(nparticipants)]
    rewards = [[] for j in range(nparticipants)]
    action_vec = np.zeros(shape=(nparticipants,len(function)))
    all_X = None
    for i in range(ntrials):
        
        #get information about the current trial
        trial = np.zeros(shape=(nparticipants, 1, len(function)))
        trial[:,:,i] = np.ones(nparticipants)[:,None]
        
        #calculate values for gp acquisition functions
        mean_var = np.array([get_mean_var(kernel, actions[j], rewards[j], choices) for j in range(nparticipants)])#.transpose(0,2,1)
        mes_utility = np.array([get_mes_utility(*mv) for mv in mean_var])[:,None,:]
        
        #combine trial and gp data, along with actions from the previous trial
        X = np.hstack((trial, action_vec[:,None,:], mean_var, mes_utility))[:,:,:,None]
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
    
    
    
def get_likelihood_utility(X, model, params, K, mixture):
    
    params = get_params(X, model, params)
    return model(**params)


def log_likelihood(X, model, params, K, mixture):
    
    actions = X[:,5,:,:]
    likelihood, utility = get_likelihood_utility(X, model, params, K, mixture)
    action_likelihood = likelihood.transpose(1,3,0,2) * actions[:,:,:,None]
    kll = np.log(action_likelihood.sum(axis=1)).sum(axis=0).sum(axis=0)
    ll = kll * mixture
    return ll.sum()


def phase_ucb_acq1(mean, var, trial, steepness, x_midpoint, yscale, temperature):
    
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


def get_params(X, model, params):
    
    #get information about trials and previous actions
    trial = X[:,0,:,:]
    previous_actions = X[:,1,:,:]
    params['trial'] = trial
    params['previous_actions'] = previous_actions
    
    #get values for gp acquisition functions
    mean = X[:,2,:,:]
    var = X[:,3,:,:]
    mes_utility = X[:,4,:,:]
    params['mean'] = mean
    params['var'] = var
    params['mes_utility'] = mes_utility
    
    parameter_names = list(signature(model).parameters)
    model_params = {p: params[p] for p in parameter_names}
    return model_params
    
    

def likelihood(X, model, params, K, mixture, c=0):
    
    mean = X[:,:,0] #(nparticipants, ntrials, choices)
    var = X[:,:,1] #(nparticipants, ntrials, choices)
    linear_interp = X[:,:,2]
    last_actions = X[:,:,3]
    is_first = X[:,:,6]
    choices = X[:,:,7]
    all_actions = X[:,:,8]
    mes_utility = X[:,:,9]
    
    trial = X[:,:,4].mean(axis=2).mean(axis=0)
    params['mean'] = mean
    params['var'] = var
    params['choices'] = choices
    params['linear_interp'] = linear_interp
    params['last_actions'] = last_actions
    params['is_first'] = is_first
    params['trial'] = trial
    params['all_actions'] = all_actions
    params['mes_utility'] = mes_utility
    
    parameter_names = list(signature(model).parameters)
    model_params = {p: params[p] for p in parameter_names}
    
    action = X[:,:,5]
    utility, all_likelihood = model(**model_params)
    likelihood = (all_likelihood * (action[:,:,:,None] * np.ones(K))).sum(axis=2)
    log_likelihood = np.log(likelihood)
    
    avg_participant_log_likelihood = log_likelihood.sum(axis=1)
    avg_sample_log_likelihood = avg_participant_log_likelihood.sum(axis=0)
    mixture_log_likelihood = (avg_sample_log_likelihood * mixture).sum()
    return mixture_log_likelihood


def ucb(mean, var, explore):
    """
    Returns ucb function values where there is one explore paramter value for all trials
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
    center = utility.max(axis = 2, keepdims = True)  #(nparticipants, ntrials, 1, K)
    centered_utility = utility - center #(nparticipants, ntrials, choices, K)
    
    exp_u = np.exp(centered_utility / temperature) #(nparticipants, ntrials, choices)
    exp_u_row_sums = exp_u.sum(axis = 2, keepdims = True) #(nparticipants, ntrials, 1, K)
    likelihood = exp_u / exp_u_row_sums #(nparticipants, ntrials, choices, K)
    return likelihood


def ucb_acq(mean, var, explore):
    
    utility = ucb(mean, var, explore)
    likelihood = propto(utility)
    return utility, likelihood


def softmax_ucb_acq(mean, var, explore, temperature):
    
    utility = ucb(mean, var, explore)
    likelihood = softmax(utility, temperature)
    return likelihood


def phase_ucb_acq(mean, var, trial, steepness, x_midpoint, yscale, temperature):
    
    explore = (1 - 1. / (1 + yscale[:,None] * np.exp(-steepness[:,None] * (trial - x_midpoint[:,None]))).T)
    utility = ucb_by_trial(mean, var, explore)    
    likelihood = softmax(utility, temperature)
    return utility, likelihood
    

def local(linear_interp, last_actions, all_actions, choices, is_first, learning_rate, stay_penalty):
    
    gradient = (np.gradient(linear_interp, axis=2) * last_actions).sum(axis=2) #(nparticipant, ntrials)
    actions = last_actions.argmax(axis=2) #(nparticipant, ntrials)
    next_action = actions[:,:,None] + gradient[:,:,None] * learning_rate #(nparticipants, ntrials, K)
    
    distance = np.abs((next_action.T - choices[:,:,None,:].T).transpose(3,2,0,1)) #(participants, trials, choices, K)
    nonfirst_utility = (np.exp(-distance).T * (1 - is_first.T)).T #(participants, trials, choices, K)
    first_utility = ((distance**0).T * is_first.T).T #(participants, trials, choices, K)   
    move_utility = (nonfirst_utility + first_utility) #(participants, trials, choices, K)
        
    penalty = (all_actions[:,:,:,None] * stay_penalty)
    utility = move_utility + penalty
    #penalty = (last_actions[:,:,:,None] * stay_penalty) + (1 - last_actions[:,:,:,None]) #(participants, trials, choices, K)
    #utility = move_utility * penalty #(participants, trials, choices, K)
    return utility
    
    
def local_acq(linear_interp, last_actions, all_actions, choices, is_first, learning_rate, stay_penalty, temperature):
    
    utility = local(linear_interp, last_actions, all_actions, choices, is_first, learning_rate, stay_penalty)
    likelihood = softmax(utility, temperature)
    return utility, likelihood


def mes_acq(mes_utility, temperature):
    
    likelihood = softmax(mes_utility[:,:,:,None], temperature)
    return mes_utility[:,:,:,None], likelihood


def mixture_acq(mean, var, trial, linear_interp, last_actions, all_actions, choices, is_first, mes_utility,
                steepness, x_midpoint, yscale, phase_ucb_temperature, learning_rate, stay_penalty, local_temperature,
                mes_temperature, mixture):
    
    print (mixture)
    
    phase_ucb_u, phase_ucb_l = phase_ucb_acq(mean, var, trial, steepness, x_midpoint, yscale, phase_ucb_temperature)
    local_u, local_l = local_acq(linear_interp, last_actions, all_actions, choices, is_first, learning_rate,
                                                stay_penalty, local_temperature)
    mes_u, mes_l = mes_acq(mes_utility, mes_temperature)
    
    all_utility = np.array([phase_ucb_u, local_u, mes_u])
    all_likelihood = np.array([phase_ucb_l, local_l, mes_l])
    
    utility = (all_utility * mixture.T[:,:,None,None,None]).sum(axis=0)
    likelihood = (all_likelihood * mixture.T[:,:,None,None,None]).sum(axis=0)
    
    return utility, likelihood
    