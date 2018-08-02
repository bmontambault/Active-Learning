import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import GPy
import inspect
import uuid

import seaborn as sns
import bokeh.plotting as bp
from bokeh.embed import components
from bokeh.models import Legend, Span
from bokeh.layouts import widgetbox
from bokeh.layouts import gridplot

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
        
    unique_actions = []
    unique_rewards = []
    for i in range(trial):
        if actions[i] not in unique_actions:
            unique_actions.append(actions[i])
            unique_rewards.append(rewards[i])
    
    if trial > 0:
        best_action = actions[np.argmax(rewards[:trial])]
    else:
        best_action = None
    return {'action_1': action_1, 'reward_1': reward_1, 'action_2': action_2, 'reward_2': reward_2,
            'unique_action_2': unique_action_2, 'unique_reward_2': unique_reward_2,
            'trial': trial, 'actions': actions[:trial], 'rewards': rewards[:trial],
            'remaining_trials': ntrials - trial, 'best_action': best_action, 'unique_actions': unique_actions,
            'unique_rewards': unique_rewards}

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
def get_all_likelihood(actions, rewards, choices, decision_type, dec_params, all_utility):
    
    all_likelihood = []
    action_likelihood = []
    ntrials = len(actions)
    for trial in range(ntrials):
        utility = all_utility[trial]
        args = get_trial_features(actions, rewards, trial, ntrials)
        args['choices'] = choices
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
def run(function, acquisition_type, decision_type, acq_params, dec_params, ntrials, kernel = None, actions = [], rewards = [], all_means = [], all_vars = []):
    
    data = {'trial_data': {}}
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
        elif len(all_means) == trial:
            mean, var = get_mean_var(kernel, a, r, choices)
        else:
            mean = all_means[trial]
            var = all_vars[trial]
        args = get_trial_features(actions, rewards, trial, ntrials)
        args['mean'] = mean
        args['var'] = var
        args['kernel'] = kernel
        args['choices'] = choices
        
        acq_arg_names = list(inspect.signature(acquisition_type.__init__).parameters.keys())
        acq_args = {arg_name: args[arg_name] for arg_name in args.keys() if arg_name in acq_arg_names}
        acquisition = acquisition_type(**acq_args)
        utility = acquisition(*acq_params)
        dec_arg_names = list(inspect.signature(decision_type.__init__).parameters.keys())
        dec_args = {dec_name: args[dec_name] for dec_name in args.keys() if dec_name in dec_arg_names}
        decision = decision_type(**dec_args)
        likelihood = np.log(decision(utility, *dec_params))
        
        if len(actions) == trial:
            next_action = st.rv_discrete(values = (choices, likelihood)).rvs()
            next_reward = function[next_action]
            actions.append(next_action)
            rewards.append(next_reward)
        else:
            next_action = actions[trial]
            next_reward = rewards[trial]
        action_likelihood.append(likelihood[next_action])
        if np.all(mean != None):
            mean = mean.ravel().tolist()
            var = var.ravel().tolist()
        all_means.append(mean)
        all_vars.append(var)
        data['trial_data'][trial] = {'actions': a, 'rewards': r, 'utility': utility.tolist(), 'likelihood': likelihood.ravel().tolist(),
                                    'joint_log_likelihood': np.sum(action_likelihood), 'mean': mean,
                                    'var': var, 'next_action': next_action, 'next_reward': next_reward,
                                    'score': np.sum(r) + next_reward}
    data['kernel'] = type(kernel).__name__
    data['acquisition'] = acquisition_type.__name__
    data['decision'] = decision_type.__name__
    data['acq_params'] = acq_params
    data['dec_params'] = dec_params
    data['actions'] = actions
    data['rewards'] = rewards
    data['choices'] = choices.tolist()
    data['all_means'] = all_means
    data['all_vars'] = all_vars
    data['joint_log_likelihood'] = np.sum(action_likelihood)
    data['function'] = function
    data['score'] = np.sum(r) + next_reward
    data['id'] = str(uuid.uuid4())
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
    all_likelihood, action_likelihood = get_all_likelihood(actions, rewards, choices, decision_type, dec_params, all_utility)
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
def fit_strategy(actions, rewards, choices, acquisition_type, decision_type, kernel = None, all_means = [], all_vars = [], method = 'DE', restarts = 5):
    
    if not acquisition_type.isGP:
        all_means = []
        all_vars = []
        kernel = None
    
    nacq_params = len(acquisition_type.init_params)
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
    params = final_x.x.tolist()
    acq_params = params[:nacq_params]
    dec_params = params[nacq_params:]
    return {'actions': actions, 'rewards': rewards, 'acquisition_type': acquisition_type,
            'decision_type': decision_type, 'acq_params': acq_params, 'dec_params': dec_params,
            'kernel': kernel, 'choices': choices, 'log_likelihood': -fun, 'ntrials': len(actions)}
    


def add_single_plot(data):
    
    function = data['function']
    all_plot_data = {'ntrials': len(data['actions']), 'acquisition': data['acquisition'], 'decision': data['decision'],
                     'acq_params': data['acq_params'], 'dec_params': data['dec_params'], 'trial_data': {},
                     'function': function, 'id': data['id']}
    for trial in range(len(data['trial_data'])):
        utility_plot = bp.figure(title = "Utility of Next Action", plot_width = 400, plot_height = 400, tools = "")
        utility_plot.toolbar.logo = None
        likelihood_plot = bp.figure(title = "Log Likelihood of Next Action", plot_width = 400, plot_height = 400, tools = "")
        likelihood_plot.toolbar.logo = None
        actions = data['trial_data'][trial]['actions']
        rewards = data['trial_data'][trial]['rewards']
        utility = data['trial_data'][trial]['utility']
        likelihood = data['trial_data'][trial]['likelihood']
        mean = data['trial_data'][trial]['mean']
        var = data['trial_data'][trial]['var']
        next_action = data['trial_data'][trial]['next_action']
        if len(actions) > 0:
            utility_plot.circle(actions, rewards, color = '#db5f57')
        utility_plot.line(list(range(len(function))), function, color = '#57d3db')
        utility_plot.line(list(range(len(utility))), utility, color = '#57d3db')
        likelihood_plot.line(list(range(len(likelihood))), likelihood, color = '#57d3db')
        utility_next_action = Span(location = next_action, dimension = 'height', line_color = 'black')
        likelihood_next_action = Span(location = next_action, dimension = 'height', line_color = 'black')
        utility_plot.add_layout(utility_next_action)
        likelihood_plot.add_layout(likelihood_next_action)
        utility_script, utility_div = components(utility_plot)
        likelihood_script, likelihood_div = components(likelihood_plot)
                             
        if np.all(mean != None):
            gp_plot = bp.figure(title = "Expected Reward", plot_width = 400, plot_height = 400, tools = "")
            gp_plot.toolbar.logo = None
            if len(actions) > 0:
                gp_plot.circle(actions, rewards, color = '#db5f57', legend = "Actions")
            std = np.sqrt(np.array(var))
            upper = np.array(mean) + 2 * std
            lower = np.array(mean) - 2 * std
            index = np.arange(len(mean))
            gp_plot.line(list(range(len(function))), function, color = '#57d3db')
            gp_plot.line(index, mean, color = '#db5f57')
            band_x = np.append(index, index[::-1])
            band_y = np.append(lower, upper[::-1])
            gp_plot.patch(band_x, band_y, color = '#db5f57', fill_alpha = 0.2)
            gp_script, gp_div = components(gp_plot)
        else:
            gp_script = None
            gp_div = None
        
        all_plot_data['trial_data'][trial] = {'utility_div': utility_div, 'utility_script': utility_script,
                                            'likelihood_div': likelihood_div, 'likelihood_script': likelihood_script,
                                            'gp_div': gp_div, 'gp_script': gp_script, 'score': data['trial_data'][trial]['score']}
    return all_plot_data


def add_all_plots(data):
    
    kernel_idx = [i for i in range(len(data)) if data[i]['kernel'] != 'NoneType']
    print (kernel_idx)
    
    colormap = np.array(sns.color_palette("hls", len(data)).as_hex())
    all_plot_data = {'ntrials': len(data[0]['actions']), 'acquisition': [], 'decision': [],
                     'acq_params': [], 'dec_params': [], 'trial_data': {},
                     'function': data[0]['function'], 'id': data[0]['id']}
    for i in range(len(data)):
        all_plot_data['acquisition'].append(data[i]['acquisition'])
        all_plot_data['decision'].append(data[i]['decision'])
        all_plot_data['acq_params'].append(data[i]['acq_params'])
        all_plot_data['dec_params'].append(data[i]['dec_params'])
        
    for trial in range(all_plot_data['ntrials']):
        utility_plot = bp.figure(title = "Utility of Next Action", plot_width = 400, plot_height = 400, tools = "")
        utility_plot.toolbar.logo = None
        likelihood_plot = bp.figure(title = "Log Likelihood of Next Action", plot_width = 400, plot_height = 400, tools = "")
        likelihood_plot.toolbar.logo = None
        actions = data[0]['trial_data'][trial]['actions']
        rewards = data[0]['trial_data'][trial]['rewards']
        next_action = data[0]['trial_data'][trial]['next_action']
        utility_next_action = Span(location = next_action, dimension = 'height', line_color = 'black')
        likelihood_next_action = Span(location = next_action, dimension = 'height', line_color = 'black')
        utility_plot.add_layout(utility_next_action)
        likelihood_plot.add_layout(likelihood_next_action)
        if len(actions) > 0:
            utility_plot.circle(actions, rewards, color = '#db5f57')
        utility_plot.line(list(range(len(all_plot_data['function']))), all_plot_data['function'], color = '#57d3db')
        
        gp_plot = bp.figure(title = "Expected Reward", plot_width = 400, plot_height = 400, tools = "")
        gp_plot.toolbar.logo = None
        if len(actions) > 0:
            gp_plot.circle(actions, rewards, color = '#db5f57')
        for j in range(len(data)):
            utility = data[j]['trial_data'][trial]['utility']
            likelihood = data[j]['trial_data'][trial]['likelihood']
            utility_plot.line(list(range(len(utility))), utility, color = colormap[j])
            likelihood_plot.line(list(range(len(likelihood))), likelihood, color = colormap[j])
                        
            if j in kernel_idx:
                mean = data[j]['trial_data'][trial]['mean']
                var = data[j]['trial_data'][trial]['var']
                std = np.sqrt(np.array(var))
                upper = np.array(mean) + 2 * std
                lower = np.array(mean) - 2 * std
                index = np.arange(len(mean))
                gp_plot.line(list(range(len(all_plot_data['function']))), all_plot_data['function'], color = '#57d3db')
                gp_plot.line(index, mean, color = '#db5f57')
                band_x = np.append(index, index[::-1])
                band_y = np.append(lower, upper[::-1])
                gp_plot.patch(band_x, band_y, color = '#db5f57', fill_alpha = 0.2)
                if len(actions) > 0:
                    utility_plot.circle(actions, rewards, color = '#db5f57')
                
                
        gp_script, gp_div = components(gp_plot)
        utility_script, utility_div = components(utility_plot)
        likelihood_script, likelihood_div = components(likelihood_plot)
        all_plot_data['trial_data'][trial] = {'utility_div': utility_div, 'utility_script': utility_script,
                                            'likelihood_div': likelihood_div, 'likelihood_script': likelihood_script,
                                            'gp_div': gp_div, 'gp_script': gp_script,
                                            'score': data[0]['trial_data'][trial]['score'],
                                            'likelihood': [], 'joint_log_likelihood': []}
        for j in range(len(data)):
            all_plot_data['trial_data'][trial]['likelihood'].append(data[j]['trial_data'][trial]['likelihood'][next_action])
            all_plot_data['trial_data'][trial]['joint_log_likelihood'].append(data[j]['trial_data'][trial]['joint_log_likelihood'])
            
        
    return all_plot_data