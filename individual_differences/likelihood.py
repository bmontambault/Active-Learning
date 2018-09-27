import numpy as np
import GPy

from data.get_results import get_results


def local_search(last_action, choices):
    if last_action == None:
        return np.ones(len(choices)) / len(choices)
    else:
        return np.array([np.exp(-abs(last_action - x)) for x in choices]).ravel()


def sigmoid_ucb(trial, mean, var, center, steepness):
    w = 1. / (1 + np.exp(-steepness * (trial - center)))
    return w * mean + (1 - w) * var

def softmax(utility, temperature):
    centered_utility = utility - np.nanmax(utility)
    exp_u = np.exp(centered_utility / temperature)
    return exp_u / np.nansum(exp_u)


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


def get_kernel(results, kernel, function_name):
    
    function_samples = list(results[results['function_name'] == function_name].iloc[0]['function_samples'].values())
    function_samples_n = np.array([((f - np.mean(f)) / np.std(f)) for f in function_samples])
    stacked_function_samples_x = np.hstack([np.arange(len(f)) for f in function_samples_n])[:,None]
    stacked_function_samples_y = np.hstack(function_samples_n)[:,None]
    m = GPy.models.GPRegression(X = stacked_function_samples_x, Y = stacked_function_samples_y, kernel = kernel)
    m.optimize()
    return m.kern


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


class Likelihood():
    
    def __init__(self, center, steepness, temperature, mixture, choices, all_actions, rewards, kernels):
        self.center = center
        self.steepness = steepness
        self.temperature = temperature
        self.mixture = mixture
        self.choices = choices
        self.all_actions = all_actions
        self.rewards = rewards
        self.kernels = kernels
    
        
    def logpdf(self, x):
        
        print (x)
        print (type(x))
        action_idx = x[0]
        function_idx = x[1]
        #action_idx, function_index = x
        actions = self.all_actions[int(action_idx)]
        all_means, all_vars = get_all_means_vars(self.kernels[function_idx], actions, self.rewards[function_idx], self.choices)
        total_likelihood = 0
        for i in range(len(actions)):
            action = actions[i]
            if i == 0:
                last_action = None
            else:
                last_action = actions[i - 1]
            
            mean = all_means[i]
            var = all_vars[i]
            ls = local_search(last_action, self.choices)
            su = sigmoid_ucb(i, mean, var, self.explore, self.center, self.steepness)
            
            utility = ls * self.mixture[0] + su * self.mixture[1]
            all_likelihoods = softmax(utility, self.temperature)
            
            likelihood = all_likelihoods[action]
            log_likelihood = np.log(likelihood)
            total_likelihood += log_likelihood
        return total_likelihood
    
    
""" 
results = get_results('data/results.json').iloc[3:]
function_names = results['function_name'].unique()
kernel = GPy.kern.RBF(1)
#kernels = {f: get_kernel(results, kernel, f) for f in function_names}

test_participant = results.iloc[0]
test_kernel = kernels[test_participant['function_name']]
test_choices = np.arange(80)

test_rewards = np.array(test_participant['function'])
n_test_rewards = (test_rewards - test_rewards.mean()) / test_rewards.std()

likelihood = Likelihood(1., 13, 1., .1, [.5, .5], test_choices, test_rewards, test_kernel)
test_actions = test_participant['response']
"""
