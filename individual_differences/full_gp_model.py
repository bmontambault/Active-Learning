import numpy as np
import GPy
import matplotlib.pyplot as plt
import pymc3 as pm
import theano.tensor as tt
from theano import scan
import theano
from pymc3.gp.util import conditioned_vars, infer_shape, stabilize, cholesky, solve_lower, solve_upper

def get_mean_var(choices, actions, rewards, kern):
    
    if len(actions) > 0:
        X = np.array(actions)[:,None]
        y = np.array(rewards)[:,None]
        model = GPy.models.GPRegression(X, y, kernel=kern)
        model.Gaussian_noise.variance = .00001
            
        mean, var = model.predict(choices[:,None])
        mean = mean.ravel()
        var = var.ravel()
    else:
        mean = np.zeros(len(choices))
        var = np.ones(len(choices))
    return mean, var


def get_all_mean_var(choices, actions, rewards, kern):
    
    all_mean = []
    all_var = []
    for trial in range(len(actions)):
        a = actions[:trial]
        r = rewards[:trial]
        mean, var = get_mean_var(choices, a, r, kern)
        all_mean.append(mean)
        all_var.append(var)
    return np.array(all_mean), np.array(all_var)


def get_mean_var1(choices, actions, rewards, kern):
        Kxx = kern(actions)
        Kxs = kern(actions, choices)
        L = cholesky(stabilize(Kxx))
        A = solve_lower(L, Kxs)
        v = solve_lower(L, rewards)
        mu = tt.dot(tt.transpose(A), v)
        Kss = kern(choices, diag=True)
        var = Kss - tt.sum(tt.square(A), 0)
        return mu, var


def generate1(function, ntrials, steepness, x_midpoint, yscale, lengthscale, height):
    
    choices = np.arange(len(function))
    actions = []
    rewards = []
    kern = GPy.kern.RBF(1, height, lengthscale)
    for trial in range(ntrials):
        position = trial - x_midpoint
        growth = np.exp(-steepness * position)
        denom = 1 + growth * yscale
        explore_param = 1 - (1. / denom)
        
        mean, var = get_mean_var(choices, actions, rewards, kern)
            
        utility = (1-explore_param)*mean + explore_param*var
        
        plot_gp(mean, var, utility, actions, rewards)
        
        action = np.argmax(utility)
        actions.append(action)
        rewards.append(function[action])
    return actions



def plot_gp(mean, var, utility, actions, rewards):
    
    std = np.sqrt(var)
    plt.plot(actions, rewards, 'bo')
    plt.plot(mean)
    plt.plot(utility)
    plt.fill_between(np.arange(len(mean)), mean-2*std, mean+2*std, alpha=.5)
    plt.show()
    
    

def pymc3_get_mean_var(actions, rewards, choices, var, lengthscale):
    
    kern = var**2 * pm.gp.cov.ExpQuad(1, ls=lengthscale)
    gp = pm.gp.Marginal(cov_func=kern)
    gp.X = actions[:,None]
    gp.y = rewards
    gp.noise = .00001
    mu, var = gp.predict(choices[:,None], diag=True)
    return mu, var


def softmax(utility, temperature):

    center = utility.max()
    centered_utility = utility - center
    
    exp_u = np.exp(centered_utility / temperature)
    likelihood = exp_u / exp_u.sum()
    return likelihood



def action_likelihood(choices, actions, rewards, next_action_vec, explore_param, kern):
    
    mean, var = get_mean_var1(choices, actions, rewards, kern)
    utility = mean * (1 - explore_param) + var * explore_param
    allp = softmax(utility, temperature)
    p = (allp * next_action_vec).sum()
    logp = np.log(p)
    return logp


def single_participant_model(actions, rewards, choices, samples=200):
    
    trials = tt.as_tensor_variable(np.arange(2,len(actions)))
    actions_vec = np.zeros(shape=(len(actions)-2, len(choices)))
    for i in range(2,len(actions)-2):
        actions_vec[i][actions[i]] = 1
    
    actions_vec = tt.as_tensor_variable(actions_vec)
    actions = tt.as_tensor_variable(actions)
    rewards = tt.as_tensor_variable(rewards)
     
    with pm.Model() as model:
    
        steepness = pm.Gamma('steepness', 1., 1., shape=1)
        x_midpoint = pm.Uniform('x_midpoint', 0, 20, shape=1)
        yscale = pm.Gamma('yscale', 1., 1., shape=1)
        temperature = pm.Gamma('temperature', 1., 1., shape=1)
        height = pm.Gamma('height', 1., 1., shape=1)
        lengthscale = pm.Gamma('lengthscale', 1., 1., shape=1)
        
        position = trials - x_midpoint
        growth = np.exp(-steepness * position)
        denom = 1 + growth * yscale
        explore_param = 1 - (1. / denom)
        
        kern = height**2 * pm.gp.cov.ExpQuad(1, ls=lengthscale)
        
        def action_likelihood(trial, next_action_vec, explore_param):
            t1 = tt.cast(trial, 'int32')
            a = actions[:t1]
            r = rewards[:t1]
            mean, var = get_mean_var1(choices[:,None], a[:,None], r, kern)
            utility = mean * (1 - explore_param) + var * explore_param
            allp = softmax(utility, temperature)
            p = (allp * next_action_vec).sum()
            logp = np.log(p)
            return logp
        
        #TODO: -inf in logp
        logp, _ = scan(fn=action_likelihood,sequences=[trials, actions_vec, explore_param])
        obs = pm.Deterministic('obs', logp)
        trace = pm.sample(samples)
        return trace
        
    
    
def single_model(actions, rewards, choices):
    
    trials = np.arange(len(actions))
    actions_array = np.zeros(shape=(len(actions), len(choices)))
    for i in range(len(actions)):
        actions_array[i][actions[i]] = 1
    
    with pm.Model() as model:
        steepness = pm.Gamma('steepness', 1., 1., shape=1)
        x_midpoint = pm.Uniform('x_midpoint', 0, 20, shape=1)
        yscale = pm.Gamma('yscale', 1., 1., shape=1)
        height = pm.Gamma('height', 1., 1., shape=1)
        lengthscale = pm.Gamma('lengthscale', 1., 1., shape=1)
        
        position = trials - x_midpoint
        growth = np.exp(-steepness * position)
        denom = 1 + growth * yscale
        explore_param = 1 - (1. / denom)
        
        kern = height**2 * pm.gp.cov.ExpQuad(1, ls=lengthscale)
        
        def action_likelihood(actions, rewards, next_action_vec, explore_param):
            mean, var = get_mean_var1(choices, actions, rewards, kern)
            utility = mean * (1 - explore_param) + var * explore_param
            allp = utility / utility.sum()
            p = (allp * next_action_vec).sum()
            logp = np.log(p)
            return logp
        
        mean, var = get_mean_var1(choices[:,None], actions[:,None], rewards, kern)
        
        utility = mean * (1 - explore_param[:,None]) + var * explore_param[:,None]
        likelihood = utility / utility.sum()
        
        action_likelihood = (likelihood * actions_array).sum(axis=1)
        log_likelihood = np.log(action_likelihood).sum()
        obs = pm.Deterministic('obs', log_likelihood)
        trace = pm.sample(200)
        return trace
    
    
def likelihood(actions, rewards, function_size, steepness, x_midpoint, yscale, lengthscale, var):
    
    choices = np.arange(function_size)
    kernel = GPy.kern.RBF(1, var, lengthscale)
    print (kernel)
    for trial in range(len(actions)):
        a = actions[:,:trial]
        r = rewards[:,:trial]
        if trial > 0:
            mean_var = np.array([get_mean_var(choices, a[i], r[i], kernel) for i in range(len(actions))])
        else:
            mean = np.zeros(shape=(len(actions),len(choices)))
            var = np.ones(shape=(len(actions),len(choices)))
            mean_var = np.stack([mean, var], axis=1)
        
        