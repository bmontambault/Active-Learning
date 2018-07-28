import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import GPy
import inspect


"""
Get means and variances for one trial given the set of actions and rewerds up to that trial and a kernel
Parameters:
    kernel: GPy kernel
    actions: observed X values
    rewards: observed y values
"""
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
    
"""
Given a full sequence of actions and rewards, return two (#trials x #choices) arrays of posterior means and variances,
with the ith row containing predictions leading up to the ith trial
Parameters:
    kernel: GPy.kern object
    actions: actions over all trials
    rewards: rewards over all trials
    choices: space of all possible actions
"""
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


"""
Get trial dependent features to be used in acquisition/decision functions
Parameters:
    actions: all actions over all trials
    rewards: all rewards over all trials
    trial: current trial (0 indexed)
    ntrials: total number of trials
"""
def get_trial_features(actions, rewards, trial, ntrials):
    
    if trial > 0:
        action_1 = actions[trial - 1]
        reward_1 = rewards[trial - 1]
    else:
        action_1 = None
        reward_1 = None
    if trial > 0:
        action_2 = actions[trial - 2]
        reward_2 = rewards[trial - 2]
        idx = [i for i in range(len(actions[:trial])) if actions[i] != action_1]
        if len(idx) > 0:
            unique_action_2 = actions[idx[-1]]
            unique_reward_2 = rewards[idx[-1]]
        else:
            unique_action_2 = None
            unique_reward_2 = None
    else:
        action_2 = None
        reward_2 = None
        unique_action_2 = None
        unique_reward_2 = None
        
    return {'action_1': action_1, 'reward_1': reward_1, 'action_2': action_2, 'reward_2': reward_2,
            'unique_action_2': unique_action_2, 'unique_reward_2': unique_reward_2,
            'trial': trial, 'actions': actions[:trial], 'rewards': rewards[:trial],
            'remaining_trials': ntrials - trial}

"""
Given a full sequence of actions and rewards, return a (#trials x #choices) array of utilities given previous observations
Parameters:
    actions: actions over all trials
    rewards: rewards over all trials
    choices: space of all possible actions
    acquisition_type: acquisition function class
    acq_params: acquistion function parameters
    all_means: (#trials x #choices) array of means across all trials
    all_vars: (#trials x #choices) array of means across all trials
    kernel: GPy kernel used by GP acquisition functions
"""
def get_all_utility(actions, rewards, choices, acquisition_type, acq_params, all_means = [], all_vars = [], kernel = None):
    
    all_utility = []
    ntrials = len(actions)
    for trial in range(ntrials):
        if len(all_means) > 0:
            mean = all_means[trial]
            var = all_vars[trial]
        else:
            mean = None
            var = None
        args = get_trial_features(actions, rewards, trial, ntrials)
        args['mean'] = mean
        args['var'] = var
        args['kernel'] = kernel
        args['choices'] = choices
        acq_arg_names = list(inspect.signature(acquisition_type.__init__).parameters.keys())
        acq_args = {arg_name: args[arg_name] for arg_name in args.keys() if arg_name in acq_arg_names}
        acquisition = acquisition_type(**acq_args)
        utility = acquisition(*acq_params)
        all_utility.append(utility)
    return np.array(all_utility)

"""
Given a full sequence of actions and rewards and a (#trial x #choices) array of utilities, return 1) a
(#trials x #choices) array of likelihoods given previous observations and utilities and 2) a #trials array
of likelihoods given the action on each trial
Parameters:
    actions: actions over all trials
    rewards: rewards over all trials
    decision_type: acquisition function class
    dec_params: acquistion function parameters
    all_utility: (#trial x #choices) array of utilities over all trials
"""
def get_all_likelihood(actions, rewards, decision_type, dec_params, all_utility):
    
    all_likelihood = []
    action_likelihood = []
    ntrials = len(actions)
    for trial in range(ntrials):
        utility = all_utility[trial]
        args = get_trial_features(actions, rewards, trial, ntrials)
        dec_arg_names = list(inspect.signature(decision_type.__init__).parameters.keys())
        dec_args = {arg_name: args[arg_name] for arg_name in args.keys() if arg_name in dec_arg_names}
        decision = decision_type(**dec_args)
        likelihood = np.log(decision(utility, *dec_params))
        all_likelihood.append(likelihood)
        action_likelihood.append(likelihood[actions[trial]])
    return np.array(all_likelihood), np.array(action_likelihood)


"""
Simulate actions given a acquisition and decision function and parameters
Parameters:
    kernel: GPy kernel
    function: array
    acquisition_type: acquisition function class
    decision_type: decision function class
    acq_params: acquistion function parameters
    dec_params: decision function parameters
    ntrials: int
"""
def run(kernel, function, acquisition_type, decision_type, acq_params, dec_params, ntrials):
    
    data = {'trial_data': {}}
    actions = []
    rewards = []
    all_means = []
    all_vars = []
    choices = np.arange(len(function))
    action_likelihood = []
    for trial in range(ntrials):
        a = actions[:trial]
        r = rewards[:trial]
        if kernel == None:
            mean = None
            var = None
        else:
            mean, var = get_mean_var(kernel, a, r, choices)
        args = get_trial_features(actions, rewards, trial, ntrials)
        args['mean'] = mean
        args['var'] = var
        args['kernel'] = kernel
        args['choices'] = choices
        
        acq_arg_names = list(inspect.signature(acquisition_type.__init__).parameters.keys())
        acq_args = {arg_name: args[arg_name] for arg_name in args.keys() if arg_name in acq_arg_names}
        acquisition = acquisition_type(**acq_args)
        utility = acquisition(*acq_params)
        decision = decision_type()
        likelihood = decision(utility, *dec_params)
        next_action = st.rv_discrete(values = (choices, likelihood)).rvs()
        action_likelihood.append(likelihood[next_action])
        next_reward = function[next_action]
        actions.append(next_action)
        rewards.append(next_reward)
        all_means.append(mean.ravel().tolist())
        all_vars.append(var.ravel().tolist())
        data['trial_data'][trial] = {'actions': a, 'rewards': r, 'utility': utility.tolist(), 'likelihood': utility.tolist(),
                                    'joint_log_likelihood': np.sum(action_likelihood), 'mean': mean.ravel().tolist(),
                                    'var': var.ravel().tolist(), 'next_action': next_action, 'next_reward': next_reward}
    data['kernel'] = type(kernel).__name__
    data['acquisition'] = acquisition_type.__name__
    data['decision'] = decision_type.__name__
    data['acq_params'] = acq_params
    data['dec_params'] = dec_params
    data['actions'] = actions
    data['rewards'] = rewards
    data['choices'] = choices
    data['all_means'] = all_means
    data['all_vars'] = all_vars
    data['joint_log_likelihood'] = np.sum(action_likelihood)
    return data


"""
return negative joint log likelihood given a strategy (acquisition/decision functions and parameters) and a set
of actions and rewards
Parameters:
    params: combined list of acquisition and decision function params
    actions: actions over all trials
    rewards: rewards over all trials
    choices: space of all possible actions
    kernel: GPy kernel
    acquisition_type: acquisition function class
    decision_type: decision function class
    all_means: (#trials x #choices) array of means across all trials
    all_vars: (#trials x #choices) array of means across all trials
    all_utility: (#trial x #choices) array of utilities over all trials
"""
def objective(params, actions, rewards, choices, kernel, acquisition_type, decision_type, all_means, all_vars, all_utility = []):
    
    nacq_params = len(acquisition_type.init_params)
    acq_params = params[:nacq_params]
    dec_params = params[nacq_params:]
    if len(all_utility) == 0:
        all_utility = get_all_utility(actions, rewards, choices, acquisition_type, acq_params, all_means = all_means, all_vars = all_vars, kernel = kernel)
    all_likelihood, action_likelihood = get_all_likelihood(actions, rewards, decision_type, dec_params, all_utility)
    joint_log_likelihood = np.sum(action_likelihood)
    return -joint_log_likelihood


"""
find the set of acquisition/decision parameters that minimize negative joint log likelihood for a set of
actions/rewards
Parameters:
    actions: actions over all trials
    rewards: rewards over all trials
    choices: space of all possible actions
    kernel: GPy kernel
    acquisition_type: acquisition function class
    decision_type: decision function class
    all_means: (#trials x #choices) array of means across all trials
    all_vars: (#trials x #choices) array of means across all trials
    method: optimization method
    restarts: number of optimization restarts
"""
def fit_strategy(actions, rewards, choices, kernel, acquisition_type, decision_type, all_means = [], all_vars = [], method = 'DE', restarts = 5):
    
    init_params = acquisition_type.init_params + decision_type.init_params
    bounds = acquisition_type.bounds + decision_type.bounds
    if len(all_means) == 0 and acquisition_type.isGP:
        all_means, all_vars = get_all_means_vars(kernel, actions, rewards, choices)
    if len(acquisition_type.init_params) == 0:
        all_utility = get_all_utility(actions, rewards, choices, acquisition_type, [], all_means = all_means, all_vars = all_vars, kernel = kernel)
    else:
        all_utility = []
    fun = np.inf
    final_x = None
    for i in range(restarts):
        if method == 'DE':
            x = opt.differential_evolution(lambda x: objective(x, actions, rewards, choices, kernel, acquisition_type, decision_type, all_means = all_means, all_vars = all_vars, all_utility = all_utility), bounds)
        else:
            x = opt.minimize(lambda x: objective(x, actions, rewards, choices, kernel, acquisition_type, decision_type, all_means = all_means, all_vars = all_vars, all_utility = all_utility), init_params, method = method, bounds = bounds)
        if x.fun < fun:
            fun = x.fun
            final_x = x
    print (final_x)
    params = final_x.x
    return params
        
    