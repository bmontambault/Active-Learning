import numpy as np
import pandas as pd
import GPy
import matplotlib.pyplot as plt

from mes_utils import get_mes_utility, gumbel_pdf


def get_results(path):
    
    data = pd.read_json(path)
    results = pd.DataFrame([x for y in data['results'] for x in y])
    all_data = pd.concat([data, results], axis = 1)
    return all_data[1:]


def get_samples(results):
    
    function_names = results['function_name'].unique()
    function_samples = {f: list(results[results['function_name'] == f].iloc[0]['function_samples'].values()) for f in function_names}
    return function_samples


def fit_function_samples(samples):
    
    samples_array = np.array(samples)
    centered_samples = samples_array - samples_array.mean(axis=1)[:,None]
    X = np.arange(centered_samples.shape[1])[:,None]
    y = centered_samples.T
        
    kern = GPy.kern.RBF(1)
    model = GPy.models.GPRegression(X, y, kern)
    model.optimize()
    return model.kern
    
    
def vectorize_response(results_row):
    
    vector = []
    function_samples = np.array(list(results_row['function_samples'].values()))
    kern = fit_function_samples(function_samples)
    function = np.array(results_row['function'])
    centered_function = function - function.mean()
    response = np.array(results_row['response'])
    all_actions = np.arange(len(centered_function))
    
    for i in range(len(response)):
        if i == 0:
            actions = []
            rewards = []
            mean = np.zeros(len(centered_function))
            var = np.ones(len(centered_function))
        else:
            actions = response[:i]
            rewards = centered_function[actions]
            model = GPy.models.GPRegression(actions[:,None], rewards[:,None], kern)
            model.Gaussian_noise.variance = 10**-2
            mean, var = model.predict(all_actions[:,None])
        
        filled_actions = np.hstack((actions, np.zeros(len(all_actions) - len(actions))*np.nan))
        filled_rewards = np.hstack((rewards, np.zeros(len(all_actions) - len(actions))*np.nan))
        
        std = np.sqrt(var).ravel()
        lower = mean.ravel() - std
        upper = mean.ravel() + std
        
        mes_utility = get_mes_utility(mean.ravel(), var.ravel(), 10000, .0001, True)
        vector.append([mean.ravel(), var.ravel(), mes_utility, lower, upper, centered_function, filled_actions, filled_rewards])
    return np.array(vector)


def plot_response(vector_row, figsize=(10,10)):
    
    trials = vector_row.shape[0]
    nrows = max(1, int(trials/5.))
    ncols = min(5, trials)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, sharex=True)
    axes = axes.ravel()
    all_actions = np.arange(vector_row.shape[2])
    for i in range(len(axes)):
        mean, var, mes_utility, lower, upper, centered_function, actions, rewards = vector_row[i]
        axes[i].plot(all_actions, centered_function)
        axes[i].plot(all_actions, mean)
        axes[i].fill_between(all_actions, lower, upper, alpha=.5)
        axes[i].plot(all_actions, upper)
        
        mes_min = mes_utility.min()
        mes_max = mes_utility.max()
        fmin = centered_function.min()
        fmax = centered_function.max()
        scaled_mes = ((mes_utility - mes_min) / (mes_max - mes_min)) * (fmax - fmin) + fmin
        axes[i].plot(all_actions, scaled_mes)
        axes[i].plot(actions, rewards, 'rx')
    