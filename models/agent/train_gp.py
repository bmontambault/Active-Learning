import sys
import numpy as np
import GPy

sys.path.append("..")
from data.get_results import get_results


def set_kernel_parameter(kernel, parameter, value):
    
    names = parameter.split('.')
    if len(names) == 1:
        setattr(kernel, names[0], value)
    else:
        kernel = getattr(kernel, names[0])
        set_kernel_parameter(kernel, '.'.join(names[1:]), value)
        
        
def fix_kernel_parameter(kernel, parameter):
    
    names = parameter.split('.')
    if len(names) == 1:
        kernel.constrain_fixed()
    else:
        kernel = getattr(kernel, names[0])
        fix_kernel_parameter(kernel, '.'.join(names[1:]))
        
        
def set_kernel_prior(kernel, parameter, prior):
    
    if prior != None:
        names = parameter.split('.')
        if len(names) == 1:
            kernel.set_prior(prior)
        else:
            kernel = getattr(kernel, names[0])
            set_kernel_prior(kernel, '.'.join(names[1:]), prior)


def train_gp(kernel, priors, X, y):
    
    parameters = kernel.parameter_names()
    for i in range(len(priors)):
        set_kernel_prior(kernel, parameters[i], priors[i])
    m = GPy.models.GPRegression(X, y, kernel = kernel)
    m.Gaussian_noise.variance = .0001
    m.Gaussian_noise.variance.constrain_fixed()
    m.optimize()
    return kernel
        
        
'''
rbf = GPy.kern.RBF(1)
variance = GPy.priors.Gamma(.1, .1)
lengthscale = GPy.priors.Gamma(.1, .1)
priors = [variance, lengthscale]

results = get_results('../data/results.json')
results = results[3:]
data = results['function_samples']
subject_data = data.iloc[0]
X = np.array([[i for i in range(len(subject_data['0']))] for j in range(len(subject_data.keys()))]).ravel()[:,None]
y = np.array([subject_data[key] for key in subject_data.keys()]).ravel()[:,None]
y_n = (y - np.mean(y)) / np.std(y)
'''