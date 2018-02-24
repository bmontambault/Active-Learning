import numpy as np

class ParameterizedFunction(object):

    def __init__(self, **parameters):
        
        self.set_parameters(**parameters)
        self.priors = {}
        
        self.current_trial = 0
        self.total_trials = 0
        
        
    def set_parameters(self, **parameters):
        
        self.parameters = {}
        for param in parameters:
            if param in self.parameter_names:
                setattr(self, param, parameters[param])
                self.parameters[param] = parameters[param]
        
        
    def set_parameters_array(self, parameters):
        
        self.set_parameters(**{self.parameter_names[i]: parameters[i] for i in range(len(self.parameter_names))})
        
        
    def set_priors(self, **priors):
        
        for prior in priors:
            self.priors[prior] = priors[prior]
            
   
    def log_prior(self):
        
        return np.sum([self.priors[param](getattr(self, param)) for param in self.priors])