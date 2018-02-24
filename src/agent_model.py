import numpy as np
import scipy.optimize as op
import scipy.stats as st

class AgentModel:
    
    def __init__(self, acquisition_function, decision_function, **parameters):
        
        self.acquisition_function = acquisition_function(**parameters)
        self.decision_function = decision_function(**parameters)
        
    
    def set_domain(self, domain):
        
        self.acquisition_function.domain = domain
        self.decision_function.domain = domain
        
    
    def set_trials(self, total_trials, current_trial):
        
        self.acquisition_function.total_trials = total_trials
        self.acquisition_function.current_trial = current_trial
        self.decision_function.total_trials = total_trials
        self.decision_function.current_trial = current_trial
    
    
    def set_parameters_array(self, parameters):
        
        num_acquisition_params = len(self.acquisition_function.parameter_names)
        acquisition_params = parameters[:num_acquisition_params]
        decision_params = parameters[num_acquisition_params:]
        self.acquisition_function.set_parameters_array(acquisition_params)
        self.decision_function.set_parameters_array(decision_params)
        
        
    def simulate_responses(self, function, total_trials):
        
        self.set_trials(total_trials, 0)
        domain = len(function)
        self.set_domain(domain)
        
        all_responses = []
        observed_x = []
        observed_y = []
        for i in range(total_trials):
            next_response = self.acquisition_function.f(observed_x, observed_y)
            p = self.decision_function.f(next_response['utility'])
            next_x = st.rv_discrete(values = (range(domain), p)).rvs()
            next_y = function[next_x]
            next_response['p'] = p
            next_response['observed_x'] = tuple(observed_x)
            next_response['observed_y'] = tuple(observed_y)
            next_response['next_x'] = next_x
            next_response['next_y'] = next_y
            all_responses.append(next_response)
            observed_x.append(next_x)
            observed_y.append(next_y)
        
        return all_responses
            
    
    def match_responses(self, function, response):
        
        total_trials = len(response)
        self.set_trials(total_trials, 0)
        self.set_domain(len(function))
        
        all_responses = []
        observed_x = []
        observed_y = []
        for i in range(total_trials):
            self.set_trials(total_trials, i)
            next_x = response[i]
            next_y = function[next_x]
            next_response = self.acquisition_function.f(observed_x, observed_y)
            p = self.decision_function.f(next_response['utility'])
            next_response['p'] = p
            next_response['observed_x'] = tuple(observed_x)
            next_response['observed_y'] = tuple(observed_y)
            next_response['next_x'] = next_x
            next_response['next_y'] = next_y
            all_responses.append(next_response)
            observed_x.append(next_x)
            observed_y.append(next_y)
            
        return all_responses
    
    
    def log_likelihood(self, function, response, get_responses = False):
        
        all_responses = self.match_responses(function, response)
        ll = np.sum([all_responses[i]['p'][response[i]] for i in range(len(response))])
        if get_responses:
            return ll, all_responses
        else:
            return ll
        
        
    def log_posterior(self, function, response):
        
        lp = self.acquisition_function.log_prior() + self.decision_function.log_prior()
        ll = self.log_likelihood(function, response)
        return lp + ll
        
    
    def optimize(self, function, response):
        
        def objective(parameters):
            self.set_parameters_array(parameters)
            return -self.log_posterior(function, response)
        
        init_acquisition_params = [getattr(self.acquisition_function, param) for param in self.acquisition_function.parameter_names]
        init_decision_params = [getattr(self.decision_function, param) for param in self.decision_function.parameter_names]
        init_params = init_acquisition_params + init_decision_params
        
        result = op.minimize(objective, init_params)
        self.set_parameters_array(result.x)
        return result
        