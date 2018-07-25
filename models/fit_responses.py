import numpy as np
import pandas as pd
import inspect
import GPy
import scipy.optimize as opt
import bokeh.plotting as bp
from bokeh.embed import components
from bokeh.models import Legend
from bokeh.layouts import widgetbox
from bokeh.layouts import gridplot

import seaborn as sns

from acquisitions import LocalMove, SGD, Explore, Exploit, Phase, UCB, EI
from decisions import Propto, Softmax

"""
Get mean/variances of GP posterior predictive given a kernel and observations
Parameters:
    kernel: GPy kernel
    observed_x: list of observed actions
    observed_y: list of observed function values
    all_x: range over possible actions
"""

def gp(kernel, observed_x, observed_y, all_x):
    
    if len(observed_x) == 0:
        return np.zeros(len(all_x))[:,None], np.ones(len(all_x))[:,None]
    else:
        X = observed_x[:,None]
        Y = observed_y[:,None]
        m = GPy.models.GPRegression(X, Y, kernel)
        m.Gaussian_noise.variance = .00001
        mean, var = m.predict(all_x[:,None])
        return mean, var
    
    
"""
Get means and variances for each trial given a kernel and a sequence of actions
Parameters:
    kernel: GPy kernel to be used
    function: Function on which the given responses were made
    actions: sequence of actions given by participant
"""
    
def get_mean_var(kernel, function, actions):
    
    all_mean = []
    all_var = []
    all_x = np.arange(len(function))
    for i in range(len(actions)):
        observed_x = np.array(actions[:i])
        if len(observed_x) == 0:
            observed_y = []
        else:
            observed_y = function[observed_x]
        mean, var = gp(kernel, observed_x, observed_y, all_x)
        all_mean.append(mean.ravel())
        all_var.append(var.ravel())
    return np.array(all_mean), np.array(all_var)        


"""
Get utilities for all actions on all trials given a sequence actions
Parameters:
    list actions: sequence of actions given by participant
    acquisition_type: acquisition function class
    list acq_params: list of paramaters to pass to acquisition function
    array all_means: trials x actions array of GP means. Pass None if acquisition
                     function does not use a GP
    array all_vars: trials x actions array of GP variances. Pass None if
                    acquisition function does not use a GP
"""

def get_utility(all_x, actions, rewards, acquisition_type, kernel, acq_params, all_means = [], all_vars = [], replace = True):
    
    
    utility_data = pd.DataFrame()
    for i in range(len(actions)):
        trial = 'trial_' + str(i + 1)
        if len(all_means) == 0:
            mean = None
        else:
            mean = all_means[i]
        if len(all_vars) == 0:
            var = None
        else:
            var = all_vars[i]
        
        if i > 0:
            last_x = actions[i - 1]
            last_y = rewards[i - 1]
        else:
            last_x = None
            last_y = None
        if i > 1:
            second_last_x = actions[i - 2]
            second_last_y = rewards[i - 2]
            idx = [j for j in range(len(actions[:i])) if actions[j] != last_x]
            if len(idx) > 0:
                unique_second_last_x = actions[idx[-1]]
                unique_second_last_y = rewards[idx[-1]]
            else:
                unique_second_last_x = None
                unique_second_last_y = None
                second_last_x = None
                second_last_y = None
        else:
            unique_second_last_x = None
            unique_second_last_y = None
            second_last_x = None
            second_last_y = None
        
        args = {'all_x': all_x, 'mean': mean, 'var': var, 'trial': i,
                'last_x': last_x, 'last_y': last_y,
                'second_last_x': second_last_x, 'second_last_y': second_last_y,
                'unique_second_last_x': unique_second_last_x, 'unique_second_last_y': unique_second_last_y,
                'ntrials': len(actions), 'trial': i, 'actions': actions[:i], 'rewards': rewards[:i],
                'kernel': kernel}
        acq_arg_names = list(inspect.signature(acquisition_type.__init__).parameters.keys())
        acq_args = {arg_name: args[arg_name] for arg_name in args.keys() if arg_name in acq_arg_names}
        acquisition = acquisition_type(**acq_args)
        utility = acquisition(*acq_params)
        if not replace:
            utility = [utility[j] if j not in actions[:i] else np.NaN for j in range(len(utility))]
        utility_data[trial] = utility
    return utility_data


"""
Get log likelihoods for all actions on all trials given a sequence actions
Parameters:
    DataFrame utility_data: action x trial data frame of utilities
    list actions: sequence of actions given by participant
    decision_type: decision function class
    list dec_params: list of paramaters to pass to acquisition function
"""

def get_likelihood(utility_data, actions, decision_type, dec_params):
    
    likelihood_data = pd.DataFrame()
    for i in range(len(actions)):
        trial = utility_data.columns[i]
        args = {'trial': i, 'x2': actions[i - 1] if i > 0 else None}
        dec_arg_names = list(inspect.signature(decision_type.__init__).parameters.keys())
        dec_args = {arg_name: args[arg_name] for arg_name in args.keys() if arg_name in dec_arg_names}
        decision = decision_type(**dec_args)
        utility = utility_data[trial]
        likelihood = np.log(decision(utility, *dec_params))
        
        likelihood_data[trial] = likelihood
    likelihood_data.index = utility_data.index
    return likelihood_data


"""
Get log likelihood given a set of actions and the log likelihoods associated with
a particular model
Parameters:
    list actions: sequence of actions given by participant
    DataFrame likelihood_data: action x trial data frame of log likelihoods
"""

def joint_log_likelihood(actions, likelihood_data):
    
    return np.nansum([likelihood_data[likelihood_data.columns[i]][actions[i]] for i in range(len(actions))])
        
        
"""
Find MLE acquisition and decision function parameters given a sequence of actions. Returns
a dictionary including all utilities/likelihoods across trials and values for model comparison.
Parameters:
    list actions: sequence of actions given by participant
    acquisition_type: acquisition function class
    decision_type: decision function class
    array all_means: trials x actions array of GP means. Pass None if acquisition
                     function does not use a GP
    array all_vars: trials x actions array of GP variances. Pass None if
                    acquisition function does not use a GP
"""

def fit_strategy(all_x, actions, rewards, acquisition_type, decision_type, replace, kernel, all_means = [], all_vars = [], method = 'DE', restarts = 5):
    
    init_acq_params = acquisition_type.init_params
    init_dec_params = decision_type.init_params
    nacq_params = len(init_acq_params)
    
    if nacq_params == 0:
        utility_data = get_utility(all_x, actions, rewards, acquisition_type, [], kernel, all_means = all_means, all_vars = all_vars, replace = replace)
        def obj(params):
            likelihood_data = get_likelihood(utility_data, actions, decision_type, params)
            jll = joint_log_likelihood(actions, likelihood_data)
            return -jll
        
    else:
        def obj(params):
            acq_params = params[:nacq_params]
            dec_params = params[nacq_params:]
            utility_data = get_utility(all_x, actions, rewards, acquisition_type, acq_params, kernel, all_means, all_vars, replace)
            likelihood_data = get_likelihood(utility_data, actions, decision_type, dec_params)
            jll = joint_log_likelihood(actions, likelihood_data)
            return -jll
    
    init_params = init_acq_params + init_dec_params
    bounds = acquisition_type.bounds + decision_type.bounds
    fun = np.inf
    final_x = None
    for i in range(restarts):
        if method == 'DE':
            x = opt.differential_evolution(obj, bounds)
        else:
            x = opt.minimize(obj, init_params, method = method, bounds = bounds)
        if x.fun < fun:
            fun = x.fun
            final_x = x
    print (final_x)
    params = final_x.x
    acq_params = params[:nacq_params]
    dec_params = params[nacq_params:]
    if nacq_params > 0:
        utility_data = get_utility(all_x, actions, rewards, acquisition_type, acq_params, kernel, all_means, all_vars, replace)
    likelihood_data = get_likelihood(utility_data, actions, decision_type, dec_params)
    n = len(actions)
    k = len(params)
    data = {'acquisition': acquisition_type.__name__, 'acquisition_params': acq_params.tolist(),
            'decision': decision_type.__name__, 'decision_params': dec_params.tolist(),
            'all_utilities': utility_data.to_dict(), 'all_likelihoods': likelihood_data.to_dict(),
            'MLE_likelihood': -fun, 'AIC': -2 * -fun + 2 * k, 'BIC': -2 * -fun + np.log(n) * k}
    random_ic = -2 * (np.log(1. / len(all_x)) * n)
    data['pseudo_r2'] = 1 - (data['AIC'] / random_ic)
    data['approx_bf'] = np.exp(random_ic - data['BIC'])
    return data

"""
fits combinations of kernels and acquisition/decision function strategies to the actions
of one participant
Parameters:
    list actions: sequence of actions given by participant
    Series function: Function on which the given responses were made
    list training_functions: List of functions on which to fit kernel parameters
    list kernels: List of GPy kernels to fit to actions
    list strategies: list of (acquisition_type, decision_type) tuples to fit to actions
""" 

def fit_participant(participant_id, goal, actions, function, function_samples, kernels, strategies, method = 'DE', restarts = 5):
    
    all_x = np.arange(len(function))[:,None]
    stacked_function_samples_x = np.hstack([np.arange(len(f)) for f in function_samples])[:,None]
    stacked_function_samples_y = np.hstack(function_samples)[:,None]
    rewards = [function[a] for a in actions]
    if goal == 'find_max_last':
        replace = False
    else:
        replace = True
    all_data = []
    
    gp_strategies = [strategy for strategy in strategies if strategy[0].isGP]
    non_gp_strategies = [strategy for strategy in strategies if not strategy[0].isGP]
    
    for i in range(len(kernels)):
        kernel = kernels[i]
        m = GPy.models.GPRegression(X = stacked_function_samples_x, Y = stacked_function_samples_y, kernel = kernel)
        m.optimize()
        all_means, all_vars = get_mean_var(kernel, function, actions)
        for acquisition_type, decision_type in gp_strategies:
            data = fit_strategy(all_x, actions, rewards, acquisition_type, decision_type, replace, all_means = all_means, all_vars = all_vars, method = method, restarts = restarts)
            data['kernel'] = type(kernel).__name__
            data['all_means'] = {'trial_' + str(i + 1): all_means[i].ravel().tolist() for i in range(len(all_means))}
            data['all_vars'] = {'trial_' + str(i + 1): all_vars[i].ravel().tolist() for i in range(len(all_vars))}
            all_data.append(data)
            
    for acquisition_type, decision_type in non_gp_strategies:
        data = fit_strategy(all_x, actions, rewards, acquisition_type, decision_type, replace, all_means = [], all_vars = [], method = method, restarts = restarts)
        all_data.append(data)
        
    participant_data = {'id': participant_id, 'function': function.ravel().tolist(), 'actions': actions, 'rewards': rewards}
    participant_data['models'] = all_data
    return participant_data


def add_viz_data(data):
    
    for i in range(len(data)):
        participant = data[i]
        colormap = np.array(sns.color_palette("hls", len(participant['models']) + 1).as_hex())
        plots = []
        for trial in range(1, len(participant['models'][0]['all_utilities'].keys()) + 1):
            u_plot = bp.figure(title = "Utility of Next Action", plot_width = 700, plot_height = 400, tools = "")
            u_plot.toolbar.logo = None
            l_plot = bp.figure(title = "Log Likelihood of Next Action", plot_width = 700, plot_height = 400, tools = "")
            l_plot.toolbar.logo = None
            legend_items = []
            
            actions = participant['actions'][:trial - 1]
            rewards = participant['rewards'][:trial - 1]
            if len(rewards) == 0:
                min_reward = 0
                max_reward = 0
            else:
                min_reward = np.min(rewards)
                max_reward = np.max(rewards)
            next_action = participant['actions'][trial - 1]
            if len(actions) > 0:
                u_plot.circle(actions, rewards, color = colormap[-1])
                a = l_plot.circle(actions, rewards, color = colormap[-1])
                legend_items.append(('Actions', [a]))
            utilities = [[model['all_utilities']['trial_' + str(trial)][k] for k in range(len(model['all_utilities']['trial_' + str(trial)]))] for model in participant['models']]
            likelihoods = [[model['all_likelihoods']['trial_' + str(trial)][k] for k in range(len(model['all_likelihoods']['trial_' + str(trial)]))] for model in participant['models']]
            u_valmin = min(np.nanmin(np.array(utilities).ravel()), min_reward)
            u_valmax = max(np.nanmax(np.array(utilities).ravel()), max_reward)
            l_valmin = min(np.nanmin([l for l in np.array(likelihoods).ravel() if not np.isinf(l)]), min_reward) - .01
            l_valmax = max(np.nanmax(np.array(likelihoods).ravel()), max_reward) + .01
            
            u_plot.vbar(x = next_action, width = 0.1, bottom = u_valmin, top = u_valmax, color = "black")
            b = l_plot.vbar(x = next_action, width = 0.1, bottom = l_valmin, top = l_valmax, color = "black")
            legend_items.append(('Next Action             | p(Next)   | p(1,...,Next - 1)', [b]))
            for j in range(len(utilities)):
                utility = utilities[j]
                likelihood = likelihoods[j]
                index = [utility[k] for k in range(len(utility)) if not np.isnan(utility[k])]
                utility = [u if not np.isnan(u) and not np.isinf(u) else np.nanmin([ui for ui in utility if not np.isinf(ui)]) for u in utility]
                likelihood = [l if not np.isnan(l) and not np.isinf(l) else np.nanmin([li for li in likelihood if not np.isinf(li)]) for l in likelihood]
                acq = participant['models'][j]['acquisition']
                if 'kernel' in participant['models'][j]:
                    kern = '/' + participant['models'][j]['kernel']
                else:
                    kern = ''
                params = '(' + ','.join([str(np.round(p, 2)) for p in participant['models'][j]['acquisition_params'] + participant['models'][j]['decision_params']]) + ')'
                
                u_plot.line(list(range(len(utility))), utility, color = colormap[j])
                c = l_plot.line(list(range(len(likelihood))), likelihood, color = colormap[j])
                
                legend_string = acq + kern + params
                trial_l = np.round(likelihood[next_action], 2)
                likelihood_data = pd.DataFrame(participant['models'][j]['all_likelihoods'])
                total_l = np.round(joint_log_likelihood(actions[:trial], likelihood_data), 2)
                legend_string = legend_string + (' ' * (14 - len(legend_string))) + ' | ' + str(trial_l) + '      | ' + str(total_l)
                
                legend_items.append((legend_string, [c]))
            u_script, u_div = components(u_plot)
            l_script, l_div = components(l_plot)
            legend = Legend(items = legend_items)
            l_plot.add_layout(legend, 'right')
            
            
            gp_plot = bp.figure(title = "Expected Reward", plot_width = 400, plot_height = 400, tools = "")
            gp_plot.toolbar.logo = None
            gp_plot.circle(actions, rewards, color = colormap[-1], legend = "Actions")
            for j in range(len(data[i]['models'])):
                if 'kernel' in data[i]['models'][j].keys():
                    kernel = data[i]['models'][j]['kernel']
                    mean = data[i]['models'][j]['all_means']['trial_' + str(trial)]
                    var = data[i]['models'][j]['all_vars']['trial_' + str(trial)]
                    std = np.sqrt(np.array(var))
                    upper = np.array(mean) + 2 * std
                    lower = np.array(mean) - 2 * std
                    index = np.arange(len(mean))
                    gp_plot.line(index, mean, color = colormap[j], legend = kernel)
                    
                    band_x = np.append(index, index[::-1])
                    band_y = np.append(lower, upper[::-1])
                    gp_plot.patch(band_x, band_y, color = colormap[j], fill_alpha = 0.2)
            gp_plot.legend.location = "top_right"
            f_script, f_div = components(gp_plot)
            
            p = gridplot([[gp_plot, u_plot], [l_plot]], toolbar_location=None)
            p_script, p_div = components(p)
            
            plots.append({'u_script': u_script, 'u_div': u_div, 'l_script': l_script, 'l_div': l_div,
                          'f_script': f_script, 'f_div': f_div, 'p_script': p_script, 'p_div': p_div})
        data[i]['plots'] = plots
    return data
            

def fit_all_participants(results, method = 'DE', restarts = 5):
    
    data = []
    for index, participant in results.iterrows():
        actions = participant['response']
        function = participant['function']
        function_n = ((function - np.mean(function)) / np.std(function))
        function_samples = list(participant['function_samples'].values())
        function_samples_n = np.array([((f - np.mean(f)) / np.std(f)) for f in function_samples])
        participant_id = participant['somataSessionId']
        function_name = participant['function_name']
        goal = participant['goal']
        score = participant['total_score']
        
        if function_name == 'pos_linear':
            kernels = [GPy.kern.RBF(1), GPy.kern.Linear(1) + GPy.kern.Bias(1)]
        elif function_name == 'neg_quad':
            kernels = [GPy.kern.RBF(1), GPy.kern.Poly(1, order = 2)]
        elif function_name == 'sinc_compressed':
            kernels = [GPy.kern.RBF(1), GPy.kern.StdPeriodic(1) + GPy.kern.RBF(1)]
        strategies = [(LocalMove, Softmax), (SGD, Softmax), (Phase, Softmax), (UCB, Softmax)]
        strategies = [(LocalMove, Softmax), (SGD, Softmax), (Explore, Softmax), (Exploit, Softmax), (Phase, Softmax), (UCB, Softmax)]
        kernels = [GPy.kern.RBF(1)]
        
        participant_data = fit_participant(participant_id, goal, actions, function_n, function_samples_n, kernels, strategies, method = method, restarts = restarts)
        participant_data['function_name'] = function_name
        participant_data['goal'] = goal
        participant_data['score'] = score
        data.append(participant_data)
    return data
        