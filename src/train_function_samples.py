import numpy as np
import pandas as pd
import GPy
import matplotlib.pyplot as plt
import scipy.stats as st

def get_data(path):
    
    data = pd.read_json(path)
    results = pd.DataFrame([x for y in data['results'] for x in y])
    all_data = pd.concat([data, results], axis = 1)
    return all_data


def training(functions, kernel, num_samples, burn_in):
    
    for i in range(len(functions)):
        f = functions[i]
        y = np.array(f)[:,None]
        X = np.arange(len(y))[:,None]
        m = GPy.models.GPRegression(X, y, kernel = kernel)
        hmc = GPy.inference.mcmc.HMC(m, stepsize = 5e-2)
        new_samples = hmc.sample(num_samples = num_samples)
        new_samples = new_samples[burn_in:]
        
        if i == 0:
            samples = new_samples
        else:
            samples = np.vstack((samples, new_samples))
    return samples


def plot_samples(samples, kernel):
    
    labels = kernel.parameter_names() + ['noise variance']
    xmin = samples.min()
    xmax = samples.max()
    xs = np.linspace(xmin, xmax, 100)
    for i in range(samples.shape[1]):
        s = samples[:, i]
        kernel = st.gaussian_kde(s)
        p = GPy.priors.Gamma.from_EV(s.mean(), s.var())
        p.plot()
        plt.plot(xs, kernel(xs), label = labels[i])
        _ = plt.legend()
        plt.show()


def train_all(data, kernels, num_samples, burn_in, priors = None):
    
    functions = data['function_name'].unique()
    function_samples = {f: data[data['function_name'] == f]['function_samples'].iloc[0] for f in functions}
    function_samples = {f: [function_samples[f][str(i)] for i in range(1, 10)] for f in functions}
    
    if priors == None:
        gamma_prior = GPy.priors.Gamma.from_EV(1., 10.)
        priors = {f: [gamma_prior for i in range(len(kernels[f].parameter_names()))] for f in functions}
    
    all_parameters = {'#samples': num_samples, 'burn_in': burn_in}
    for f in functions:
        kern = kernels[f]
        kern_parameters = kern.parameter_names()
        kern_priors = priors[f]
        for i in range(len(kern_priors)):
            set_kernel_prior(kern, kern_parameters[i], kern_priors[i])
        fsamples = function_samples[f]
        scale = np.std(fsamples)
        scaled_fsamples = [[y/scale for y in f] for f in fsamples]
        samples = training(scaled_fsamples, kern, num_samples, burn_in)
        plot_samples(samples, kern)
        
        mean = samples.mean(axis = 0)
        kern_mean = mean[:-1]
        noise_mean = mean[-1]
        var = samples.var(axis = 0)
        kern_post = [GPy.priors.Gamma.from_EV(mean[i], var[i]) for i in range(len(kern_mean))]
        noise_post = GPy.priors.Gamma.from_EV(noise_mean, var[-1])
        
        parameters = {'kernel_posterior_means': kern_mean.tolist(), 'noise_posterior_mean': noise_mean.tolist(),
                      'kernel_posterior': [(p.a, p.b) for p in kern_post], 'noise_posterior': (noise_post.a, noise_post.b),
                      'samples': scaled_fsamples, 'scale': scale}
        all_parameters[f] = parameters
    
    return all_parameters
    
    
    
def train_data():
    data = get_data('../data.json')
    data = data[3:]
    
    kernels = {'pos_linear': GPy.kern.Linear(1) + GPy.kern.Bias(1),
               'neg_quad': GPy.kern.Poly(1, order = 2),
               'sinc_compressed': GPy.kern.RBF(1)}
    
    parameters = train_all(data, kernels, 400, 100)
    return parameters