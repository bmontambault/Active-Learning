import numpy as np
from parameterized_function_class import ParameterizedFunction


class Deterministic(ParameterizedFunction):
    
    parameter_names = []
    
    def f(self, utility):
        
        return np.array(map(lambda x: 1. if x == utility.max() else 0., utility))


class PropToUtility(ParameterizedFunction):
    
    parameter_names = []
    
    def f(self, utility):
        
        return utility / utility.sum()


class Softmax(ParameterizedFunction):
    
    parameter_names = ['temperature']
    
    def f(self, utility):
        
        centered_utility = utility - utility.max()
        exp_u = np.exp(centered_utility / self.temperature)
        return exp_u / exp_u.sum()
    
    
class CoolingSoftmax(ParameterizedFunction):
    
    parameter_names = ['temperature', 'cooling']
    
    def f(self, utility):
        
        temperature = self.temperature * self.cooling**self.current_trial
        centered_utility = utility - utility.max()
        exp_u = np.exp(centered_utility / temperature)
        return exp_u / exp_u.sum()