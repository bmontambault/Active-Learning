import numpy as np
from scipy.stats import rv_discrete
import GPy
    

def z_score(y, mean, std):
    
    if type(y) == list:
        y = np.array(y)[None,:]
    return (y - mean)/std    


def softmax(utility, temp):
    
    centered_utility = utility - utility.max()
    exp_u = np.exp(centered_utility / temp)
    return exp_u / exp_u.sum()


def deterministic(utility):
    
    p = np.array([1. if r == np.max(utility) else 0. for r in utility])
    return p / p.sum()

def get_next(observed_x, observed_y, function, upper_bound, acq, acqArgs, dec = None, decArgs = None, isGP = False):
    
    n = len(function)
    domain = range(n)
    if not isGP or len(observed_x) > 0:
        params, p = acq(observed_x, observed_y, domain, *acqArgs, dec, decArgs)
    else:
        if len(observed_x) == 0:
            mean = np.ones(n) * (upper_bound / 2.)
            std = np.ones(n) * (upper_bound / 4.)
            utility = np.zeros(n)
            p = np.ones(n) / n
            params = {'utility':utility, 'mean':mean, 'std':std}
    next_x = rv_discrete(values = (domain, p)).rvs()
    return params, p, next_x


def get_next_gp(observed_x, observed_y, function, kern, acq, args):
    
    n = len(function)
    domain = range(n)
    if len(observed_x) == 0:
        mean = np.zeros(n)
        std = np.ones(n)
        utility = np.zeros(n)
        next_x = np.random.choice(domain)
        {'utility':utility, 'mean':mean, 'std':std}
        
    else:
        mean, var = gp(observed_x, observed_y, kern, domain)
        std = np.sqrt(var)
        params, p = acq(mean, std, *args)
        next_x = np.random.choice(range(len(function)), p = p)
        
    return params, p, next_x


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