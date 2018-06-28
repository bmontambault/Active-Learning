import numpy as np


def deterministic(utility, params, dec_other):
    
    p = np.array([1 if u == max(utility) else 0 for u in utility])
    return p / p.sum()


def propto(utility, params, dec_other):
    
    return utility / np.sum(utility)


def softmax(utility, params, dec_other):
    
    temperature = params[0]
    centered_utility = utility - np.max(utility)
    exp_u = np.exp(centered_utility / temperature)
    return exp_u / np.sum(exp_u)


def random_stay_softmax(utility, params, dec_other):
    
    temperature = params[0]
    u = softmax(utility, [temperature], dec_other)
    last_x = dec_other['last_x']
    p = params[1]
    prob_stay = np.array([p if i == last_x else 0 for i in range(len(utility))])
    total_u = u + prob_stay
    return total_u / total_u.sum()


def random_switch_softmax(utility, params, dec_other):
    
    temperature = params[0]
    u = softmax(utility, [temperature], dec_other)
    last_x = dec_other['last_x']
    p = params[1]
    prob_switch = np.array([p if i != last_x else 0 for i in range(len(utility))])
    total_u = u + prob_switch
    return total_u / total_u.sum()