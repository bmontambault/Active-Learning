import numpy as np
import GPy
from parameterized_function_class import ParameterizedFunction

import mrs

class Random(ParameterizedFunction):
    
    parameter_names = []
    
    def f(self, observed_x, observed_y):
        
        utility = np.ones(self.domain) / self.domain
        return {'utility': utility}


class GPAcquisitionFunction(ParameterizedFunction):
    
    def set_kernel(self, kernel):
    
        self.kernel = kernel
        
        
    def set_noise_variance(self, noise_variance):
        
        self.noise_variance = noise_variance
    
    
    def get_mean_var(self, observed_x, observed_y):
        
        if len(observed_x) == 0:
            return np.zeros(self.domain)[:,None], np.ones(self.domain)[:,None]
        else:
            X = np.array(observed_x)[:,None]
            Y = np.array(observed_y)[:,None]
            gp = GPy.models.GPRegression(X, Y, self.kernel)
            gp.Gaussian_noise.variance = self.noise_variance
            return gp.predict(np.array(range(self.domain))[:,None])
        
    def get_cov(self, observed_x, observed_y):
        
        if len(observed_x) == 0:
            return np.identity(self.domain)
        else:
            X = np.array(observed_x)[:,None]
            return self.kernel.K(X)

        
class MeanGreedy(GPAcquisitionFunction):
    
    parameter_names = []
    
    def f(self, observed_x, observed_y):
        
        mean, var = self.get_mean_var(observed_x, observed_y)
        std = np.sqrt(var)
        utility = mean
        return {'utility': utility, 'mean': mean, 'std': std}
        
        
class UCB(GPAcquisitionFunction):
    
    parameter_names = ['explore']
    
    def f(self, observed_x, observed_y):
        
        mean, var = self.get_mean_var(observed_x, observed_y)
        std = np.sqrt(var)
        utility = mean + self.explore * std
        return {'utility': utility, 'mean': mean, 'std': std}


class CoolingUCB(GPAcquisitionFunction):
    
    parameter_names = ['explore', 'explore_cooling']
    
    def f(self, observed_x, observed_y):
        
        mean, var = self.get_mean_var(observed_x, observed_y)
        std = np.sqrt(var)
        utility = mean + (self.explore * self.explore_cooling**self.current_trial) * std
        return {'utility': utility, 'mean': mean, 'std': std}


class MRSpoint(GPAcquisitionFunction):
    
    parameter_names = []
    fnsamples = 10
    ynsamples = 4
    
    def f(self, observed_x, observed_y):
        
        mean, var = self.get_mean_var(observed_x, observed_y)
        cov = self.get_cov(observed_x, observed_y)
        current_regret = self.expected_regret(cov, self.fnsamples, np.argmax(mean))
        
        ysamples = np.array([np.random.normal(mean[i], var[i], self.ynsamples) for i in range(len(mean))])
        print (ysamples)
        
    
    def expected_regret(cov, nsamples, x):
    
        samples = np.random.multivariate_normal(np.zeros(cov.shape[0]), cov, nsamples)
        return np.mean(np.max(samples, axis = 1) - samples.T[x])

    
    
