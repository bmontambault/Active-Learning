import numpy as np
import pandas as pd
import GPy
import matplotlib.pyplot as plt

from mes_utils import get_mes_utility, gumbel_pdf


def get_results(path):
    
    data = pd.read_json(path)
    results = pd.DataFrame([x for y in data['results'] for x in y])
    all_data = pd.concat([data, results], axis = 1)
    return all_data[3:]


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
    
    
def vectorize_response(results_row, kern=None):
    
    vector = []
    function_samples = np.array(list(results_row['function_samples'].values()))
    if kern == None:
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
        
        mes_utility, info, entropy = get_mes_utility(mean.ravel(), var.ravel(), 100000, .0001, True)
        vector.append([mean.ravel(), var.ravel(), mes_utility, info, entropy, lower, upper, centered_function, filled_actions, filled_rewards])
    
    vector = np.array(vector)
    actions = np.array(results_row['response'])
    actions_vec = np.eye(vector.shape[2])[actions.astype(int)]
    vector = np.hstack((vector, actions_vec[:,None,:]))
    return vector


def vectorize_all_responses(results_df):
    
    function_samples = results_df.copy().groupby('function_name')['function_samples'].apply(lambda x: tuple(x)[0])
    function_samples = pd.DataFrame(function_samples.values.tolist(), index= function_samples.index)
    centered_samples = pd.DataFrame(function_samples.values - function_samples.mean(axis=1).values[:,None], index=function_samples.index)
    
    function_names = list(set(centered_samples.index.get_level_values(0).tolist()))
    kerns = {}
    for function_name in function_names:
        d = centered_samples.loc[function_name]
        X = np.arange(d.shape[1])[:,None]
        y = centered_samples.loc[function_name].values.T
        
        kern = GPy.kern.RBF(1)
        model = GPy.models.GPRegression(X, y, kern)
        model.optimize()
        kerns[function_name] = kern
    
    vec = []
    for index, row in results_df.iterrows():
        print (index)
        function_name = row['function_name']
        kern = kerns[function_name]
        v = vectorize_response(row, kern)
        vec.append(v)
    return np.array(vec)


def plot_response(vector_row, figsize=(10,10), plot_entropy='mes'):
    
    trials = vector_row.shape[0]
    nrows = max(1, int(trials/5.))
    ncols = min(5, trials)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True, sharex=True)
    axes = axes.ravel()
    all_actions = np.arange(vector_row.shape[2])
    for i in range(len(axes)):
        mean, var, mes_utility, info, entropy, lower, upper, centered_function, actions, rewards = vector_row[i]
        axes[i].plot(all_actions, mean)
        axes[i].fill_between(all_actions, lower, upper, alpha=.5)
        axes[i].plot(all_actions, centered_function)
        axes[i].plot(all_actions, upper)
        
        fmin = centered_function.min()
        fmax = centered_function.max()
        if plot_entropy == 'c2':
            mes_min = entropy.min()
            mes_max = entropy.max()
            scaled_mes = ((entropy - mes_min) / (mes_max - mes_min)) * (fmax - fmin) + fmin
        elif plot_entropy == 'c1':
            mes_min = info.min()
            mes_max = info.max()
            scaled_mes = ((info - mes_min) / (mes_max - mes_min)) * (fmax - fmin) + fmin
        else:
            mes_min = mes_utility.min()
            mes_max = mes_utility.max()
            scaled_mes = ((mes_utility - mes_min) / (mes_max - mes_min)) * (fmax - fmin) + fmin
        axes[i].plot(all_actions, scaled_mes)
        axes[i].plot(actions, rewards, 'rx')
    