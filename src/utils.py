import numpy as np
import GPy
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-bright')


def soft_max(utility, temp):
    
    centered_utility = utility - utility.max()
    exp_u = np.exp(centered_utility / temp)
    return exp_u / exp_u.sum()
    

def get_next_gp(observed_x, observed_y, function, kern, acq, params):
    
    n = len(function)
    domain = range(n)
    if len(observed_x) == 0:
        mean = np.zeros(n)
        std = np.ones(n)
        utility = np.zeros(n)
        params = np.NaN
        next_x = np.random.choice(domain)
        
    else:
        mean, var = gp(observed_x, observed_y, kern, domain)
        std = np.sqrt(var)
        params, utility = acq(mean, std, *params)
        next_x = np.argmax(utility)
        
    return mean, std, next_x, utility, params


def fit_kern(functions):
    
    stacked = functions.unstack().reset_index()[['level_1', 0]]
    X = np.array(stacked['level_1'])[:,None]
    y = np.array(stacked[0])[:,None]
    m = GPy.models.GPRegression(X, y)
    m.optimize()
    kern = m.kern
    return kern


def gp(observed_x, observed_y, kern, domain):
    
    if len(observed_x) == 0:
        return np.zeros(len(domain))[:,None], np.ones(len(domain))[:,None]
    else:
        X = np.array(observed_x)[:,None]
        Y = np.array(observed_y)[:,None]
        gp = GPy.models.GPRegression(X, Y, kern)
        return gp.predict(np.array(domain)[:,None])
    

def scale(f, new_min, new_max):
    
    min_f = min(f)
    max_f = max(f)
    return (new_max - new_min) * (f - min_f)/(max_f - min_f) + new_min