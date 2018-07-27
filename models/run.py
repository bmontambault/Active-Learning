import numpy as np
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
    all_actions: actions over all trials
    all_rewards: rewards over all trials
    choices: space of all possible actions
"""
def get_all_means_vars(kernel, all_actions, all_rewards, choices):
    
    all_means = []
    all_vars = []
    for i in range(len(all_actions)):
        actions = all_actions[:i]
        rewards = all_rewards[:i]
        mean, var = get_mean_var(kernel, actions, rewards, choices)
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


def get_all_utilities(actions, rewards, choices, acquisition_type, acq_params, all_means = [], all_vars = [], kernel = None):
    
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


def get_all_likelihood(actions, rewards, decision_type, dec_params, all_utility):
    
    all_likelihood = []
    ntrials = len(actions)
    for trial in range(ntrials):
        utility = all_utility[trial]
        args = get_trial_features(actions, rewards, trial, ntrials)
        dec_arg_names = list(inspect.signature(decision_type.__init__).parameters.keys())
        dec_args = {arg_name: args[arg_name] for arg_name in args.keys() if arg_name in dec_arg_names}
        decision = decision_type(**dec_args)
        likelihood = np.log(decision(utility, *dec_params))
        all_likelihood.append(likelihood)
    return np.array(all_likelihood)

"""
If no actions given simulate
If actions given fit strategy
"""
def run(kernel, function, acquisition_type, decision_type, acq_params, dec_params, ntrials, all_actions = []):
    
    choices = np.arange(len(function))
    actions = []
    rewards = []
    for trial in range(len(ntrials)):
        if len(actions) >= trial:
            actions = all_actions[:trial]
            rewards = [function[a] for a in actions]
            
        else:
            pass
    