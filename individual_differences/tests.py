import numpy as np

def centered_softmax(utility, temperature):
    
    center = utility.max()
    centered_utility = utility - center
    
    exp_u = np.exp(centered_utility / temperature)
    exp_u_row_sums = exp_u.sum()
    likelihood = exp_u / exp_u_row_sums
    return likelihood


def softmax(utility, temperature):
    
    exp_u = np.exp(utility / temperature)
    return exp_u / exp_u.sum()


def test_softmax(utility, temperature):
    
    p = softmax(utility, temperature)
    cp = centered_softmax(utility, temperature)
    print (p-cp)
    print (p.sum())
    print (cp.sum())
    return np.all(p == cp)