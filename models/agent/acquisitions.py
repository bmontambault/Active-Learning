import numpy as np
import GPy

def gp(kernel, observed_x, observed_y, all_x):
    
    if len(observed_x) == 0:
        return np.zeros(len(all_x))[:,None], np.ones(len(all_x))[:,None]
    else:
        X = np.array(observed_x)[:,None]
        Y = np.array(observed_y)[:,None]
        m = GPy.models.GPRegression(X, Y, kernel)
        m.Gaussian_noise.variance = .00001
        return m.predict(np.array(all_x)[:,None])



def random(kernel, observed_x, observed_y, all_x, params = [], acq_other = {}):
    
    mean, var = gp(kernel, observed_x, observed_y, all_x)
    return np.ones(len(all_x)) / len(all_x), mean, var, {'last_x': observed_x[-1] if len(observed_x) > 0 else None}



def sgd(kernel, observed_x, observed_y, all_x, params, acq_other = {}):
    
    mean, var = gp(kernel, observed_x, observed_y, all_x)
    if len(set(observed_x)) < 2:
        u = np.ones(len(all_x)) / len(all_x)
    else:
        
        x2 = observed_x[-1]
        y2 = observed_y[-1]
        x1 = [x for x in observed_x if x != x2][-1]
        y1 = [y for y in observed_y if y != y2][-1]
        
        dk = (y2 - y1) / (x2 - x1)
        learning_rate = params[0]
        next_x = x2 + dk * learning_rate
        u = [np.exp(-abs(next_x - i)) for i in range(len(all_x))]
    return np.array(u), mean, var, {'last_x': observed_x[-1] if len(observed_x) > 0 else None}



def decaying_sgd(kernel, observed_x, observed_y, all_x, params, acq_other):
    
    learning_rate, decay_rate = params
    new_learning_rate = learning_rate * decay_rate**acq_other['trial']
    return sgd(kernel, observed_x, observed_y, all_x, [new_learning_rate], acq_other)


def ucb(kernel, observed_x, observed_y, all_x, params, acq_other = {}):
    
    mean, var = gp(kernel, observed_x, observed_y, all_x)
    b = params[0]
    return (mean + b * var).ravel(), mean, var, {'last_x': observed_x[-1] if len(observed_x) > 0 else None}