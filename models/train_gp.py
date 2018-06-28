import GPy
from data.get_results import get_results



def train_gp(kernel, priors, data):
    
    pass


rbf = GPy.kern.RBF(1)
variance = GPy.priors.Gamma(.1, .1)
lengthscale = GPy.priors.Gamma(.1, .1)
priors = [variance, lengthscale]


results = get_results('data/results.json')
results = results[3:]