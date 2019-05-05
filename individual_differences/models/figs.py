import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import GPy
import pymc3 as pm


def get_group_posterior_samples(trace_df, group):
    
    df = trace_df[[c for c in trace_df.columns if '__' in c and c.split('__')[0] in ['var_param', 'mes_param', 'temperature'] and c.split('__')[-1] == str(group)]]
    df.columns = [c.split('__')[0] for c in df.columns]
    return df


def get_condition_responses(vectorized_data, functions, goals, function=None, goal=None):
    
    X = vectorized_data.copy()
    f = functions.copy()
    g = goals.copy()
    if function != None:
        X = X[:,:,:,f == function]
        g = g[f == function]
    if goal != None:
        
        X = X[:,:,:,g == goal]
    return X


def softmax(utility, temperature):

    center = utility.max(axis=1, keepdims=True)
    centered_utility = utility - center
    
    exp_u = np.exp(centered_utility / temperature[None,:,None,None])
    exp_u_row_sums = exp_u.sum(axis = 2, keepdims = True) 
    likelihood = exp_u / exp_u_row_sums
    return likelihood


def group_static_ucb_mes_likelihood(actions, mean, var, mes, var_explore, mes_explore, temperature):
    
    all_utility = mean[:,None,:,:] + var[:,None,:,:]*var_explore[None,:,None,None] + mes[:,None,:,:]*mes_explore[None,:,None,None]   
    all_likelihood = softmax(all_utility, temperature)
    
    action_likelihood = (all_likelihood * actions[:,None,:,:]).sum(axis=2)
    likelihood = np.nanmean(action_likelihood, axis=1)
    log_action_likelihood = np.log(likelihood)
    log_likelihood = log_action_likelihood.sum()
    return log_likelihood


def posterior_likelihood_(X, samples):
    
    var_samples, mes_samples, temp_samples = samples.values.T
    actions, mean, var, mes = X
    return group_static_ucb_mes_likelihood(actions, mean, var, mes, var_samples, mes_samples, temp_samples)


def posterior_likelihood(vectorized_responses, trace_df, group, functions, goals, function=None, goal=None):
    
    X = get_condition_responses(vectorized_responses, functions, goals, function, goal)
    samples = get_group_posterior_samples(trace_df, group)
    return posterior_likelihood_(X, samples)


def generate_prior_samples(explore_param_alpha=.1, explore_param_beta=.1,
                     temperature_alpha=1., temperature_beta=1., explore_method_p=.5,
                     samples=4000):
    
    var_param = pm.Gamma.dist(explore_param_alpha, explore_param_beta).random(size=samples)
    mes_param = pm.Gamma.dist(explore_param_alpha, explore_param_beta).random(size=samples)
    temperature = pm.Gamma.dist(temperature_alpha, temperature_beta).random(size=samples)
    df = pd.DataFrame(np.array([var_param, mes_param, temperature]).T)
    df.columns = ['var_param', 'mes_param', 'temperature']
    return df


def null_posterior_likelihood(vectorized_responses, functions, goals, function=None, goal=None):
    
    X = get_condition_responses(vectorized_responses, functions, goals, function, goal)
    samples = generate_prior_samples()
    return posterior_likelihood_(X, samples)
    

def bayes_factor(vectorized_responses, trace_df, group, functions, goals, function=None, goal=None):
    
    model = posterior_likelihood(vectorized_responses, trace_df, group, functions, goals, function, goal)
    null = null_posterior_likelihood(vectorized_responses, functions, goals, function, goal)
    bf = np.exp(model - null)
    return bf


def all_bayes_factors(vectorized_responses, trace_df, functions, goals):
    
    function_conditions = ['pos_linear', 'neg_quad', 'sinc_compressed', None]
    goal_conditions = ['max_score_last', 'find_max_last', None]
    groups = np.arange(15)
    
    all_bf = {}
    for function in function_conditions:
        for goal in goal_conditions:
            for group in groups:
                bf = bayes_factor(vectorized_responses, trace_df, group, functions,
                                  goals, function=function, goal=goal)
                print (function, goal, group)
                all_bf[(function, goal, group)] = bf
    return all_bf
    

    

def get_assignments(trace_df):
    
    assignment_columns = [c for c in trace_df.columns if 'assignments' in c]
    index = list(set([a for b in [trace_df[c].unique() for c in assignment_columns] for a in b]))
    assignments = trace_df[assignment_columns]
    assignment_counts = pd.concat([assignments[c].value_counts().sort_index().reindex(index) for c in trace_df.columns if 'assignments' in c], axis=1)
    assignment_counts.columns = assignment_counts.columns
    return assignment_counts.fillna(0)


def avg_prev_dist(response, assignments, assignments_plot, names, figsize, bins=None):
    
    fig, axes = plt.subplots(1, len(assignments_plot), figsize=figsize)
    for i in range(len(assignments_plot)):
        assign = assignments_plot[i]
        a = response[(assignments == assign)]
        dist = np.abs(a[:,1:] - a[:,:-1]).ravel()
        sns.distplot(dist, ax=axes[i], kde=None, bins=bins, norm_hist=True)
        axes[i].set_xlim((0, 80))
        axes[i].set_ylim((0, .45))
        axes[i].set_title(names[i])
        if i == 0:
            axes[i].set_xlabel('Distance From Previous Action')
            axes[i].set_ylabel('Density')
    return fig
        
        
def dist_max_plot(response, assignments, assignments_plot, fmax,
                  smooth=None):
    
    #fig, axes = plt.subplots(1, len(assignments_plot), figsize=figsize)
    fig, ax = plt.subplots(1,1)
    for i in range(len(assignments_plot)):
        assign = assignments_plot[i]
        a = response[(assignments == assign)]
        f = fmax[(assignments == assign)]
        
        dist = np.abs(a - f[:,None])
        if smooth == None:        
            dist_mean = dist.mean(axis=0)
            dist_std = dist.std(axis=0)
        elif smooth == 'gp':
            X = (np.ones(shape=dist.shape).cumsum(axis=1)-1).ravel()[:,None]
            y = dist.ravel()[:,None]
            gp = GPy.models.GPRegression(X, y)
            gp.optimize()
            dist_mean, dist_var = gp.predict(np.arange(25)[:,None])
            dist_std = np.sqrt(dist_var)
        ax.plot(dist_mean)
        lower = dist_mean.ravel() - dist_std.ravel()
        upper = dist_mean.ravel() + dist_std.ravel()
        ax.fill_between(np.arange(len(dist_mean)), lower,
            upper, alpha=.5)
        ax.set_ymin(0)
        
        
def dist_prev_plot(response, assignments, assignments_plot, group_names,
                  smooth=None):
    
    #fig, axes = plt.subplots(1, len(assignments_plot), figsize=figsize)
    fig, ax = plt.subplots(1,1)
    for i in range(len(assignments_plot)):
        assign = assignments_plot[i]
        group_name = group_names[i]
        a = response[(assignments == assign)]
        
        dist = np.abs(a[:,1:] - a[:,:-1])
        if smooth == None:        
            dist_mean = dist.mean(axis=0)
            dist_std = dist.std(axis=0)
        elif smooth == 'gp':
            X = (np.ones(shape=dist.shape).cumsum(axis=1)-1).ravel()[:,None]
            y = dist.ravel()[:,None]
            gp = GPy.models.GPRegression(X, y)
            gp.optimize()
            dist_mean, dist_var = gp.predict(np.arange(25)[:,None])
            dist_std = np.sqrt(dist_var)
        ax.plot(dist_mean, label=group_name)
        lower = dist_mean.ravel() - dist_std.ravel()
        upper = dist_mean.ravel() + dist_std.ravel()
        ax.fill_between(np.arange(len(dist_mean)), lower,
            upper, alpha=.5)
        ax.set_ylim(0)
        plt.legend(loc='upper right')
    return fig
        
        
def conditional_dists(trace_df, function_names, goals, function_name=None, goal=None):
    
    participants = list(range(len(function_names)))
    if function_name != None:
        participants = [i for i in participants if function_names[i] == function_name]
    if goal != None:
        participants = [i for i in participants if goals[i] == goal]
    
    all_samples = []
    params = ['var_param__', 'mes_param__', 'temperature__']
    for index,row in trace_df.iterrows():
        
        groups = row[['assignments__'+str(p) for p in participants]].values
        samples = [[row[param + str(int(g))] for param in params] for g in groups]
        all_samples.append(samples)
    return np.vstack(all_samples)





def plot_conditional_dists(trace_df, function_names, goals):

    f = ['pos_linear', 'neg_quad', 'sinc_compressed']
    g = ['max_score', 'find_max']
    all_dfs = []
    for function_name in f:
        for goal in g:
            samples = conditional_dists(trace_df, function_names, goals, function_name=function_name, goal=goal)
            df  = pd.DataFrame(samples)
            df['f'] = function_name
            df['g'] = goal
            all_dfs.append(df)
    return pd.concat(all_dfs)
    
    

def count_groups(trace_df, function_names, goals):
    
    assignments = get_assignments(trace_df)
    max_assignments = assignments.idxmax(axis=0).reset_index()
    max_assignments['function'] = function_names
    max_assignments['goal'] = goals
    return max_assignments
        
        

def condition_posterior(trace_df, group, function_names, goals, function_name=None, goal=None):
    
    participants = list(range(len(function_names)))
    if function_name != None:
        participants = [i for i in participants if function_names[i] == function_name]
    if goal != None:
        participants = [i for i in participants if goals[i] == goal]
    return posterior(trace_df, participants, group)
        

def posterior(trace_df, participants, group):
    
    assignment_cols = [col for col in trace_df.columns if 'assignment' in col and int(col.split('__')[-1]) in participants]
    assignment_samples = trace_df[assignment_cols].values.ravel()
    return assignment_samples
