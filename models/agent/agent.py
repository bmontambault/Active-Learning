import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import uuid

from decisions import propto

class Agent:
    
    def __init__(self, acquisition, decision, acq_params, dec_params, kernel = None, acq_priors = None, dec_priors = None, acq_bounds = None, dec_bounds = None):
        self.kernel = kernel
        self.acquisition = acquisition
        self.decision = decision
        self.acq_params = acq_params
        self.dec_params = dec_params
        self.observed_x = []
        self.observed_y = []

        if acq_priors == None:
            self.acq_priors = [lambda x: 0 for i in range(len(acq_params))]
        else:
            self.acq_priors = acq_priors
        if dec_priors == None:
            self.dec_priors = [lambda x: 0 for i in range(len(dec_params))]
        else:
            self.dec_priors = dec_priors
        if acq_bounds == None:
            self.acq_bounds = [(-np.inf, np.inf) for i in range(len(acq_params))]
        else:
            self.acq_bounds = acq_bounds
        if dec_bounds == None:
            self.dec_bounds = [(-np.inf, np.inf) for i in range(len(dec_params))]
        else:
            self.dec_bounds = dec_bounds
        
    
    def set_kernel(self, kernel):
        
        self.kernel = kernel
    
    
    def action(self, function, acq_other, action = None, utility = []):
        all_x = np.arange(len(function))
        if len(utility) == 0:
            utility, mean, var, dec_other = self.acquisition(self.kernel, self.observed_x, self.observed_y, all_x, self.acq_params, acq_other = acq_other)
        else:
            _, mean, var, dec_other = self.acquisition(self.kernel, self.observed_x, self.observed_y, all_x, self.acq_params, acq_other = acq_other)
            
        probability = self.decision(utility, self.dec_params, dec_other)
        if action == None:
            action = st.rv_discrete(values = (all_x, probability)).rvs()
        return mean, var, utility, probability, action
            
            
    def log_prior(self):
        
        lp = [self.acq_priors[i](self.acq_params[i]) for i in range(len(self.acq_params))] + [self.dec_priors[j](self.dec_params[j]) for j in range(len(self.dec_params))]
        joint_lp = np.sum(lp)
        if np.isnan(joint_lp):
            return -np.inf
        else:
            return joint_lp
        
            
    def log_likelihood(self, function, action, acq_other, utility = []):
        
        if len(utility) == 0:
            mean, var, utility, probability, action = self.action(function, acq_other, action)
        else:
            mean, var, utility, probability, action = self.action(function, acq_other, action, utility)
        return np.log(probability[action])
            
    
    def log_posterior(self, function, action, acq_other, utility = []):
        
        lp = self.log_prior()
        ll = self.log_likelihood(function, action, acq_other, utility)
        return lp + ll
    
    def MAP(self, function, actions, fixed_acq_idx = [], fixed_dec_idx = [], method = 'L-BFGS-B'):
        
        fixed_acq_len = len(fixed_acq_idx)
        opt_acq_len = len(self.acq_params) - fixed_acq_len
        if opt_acq_len == 0:
            utilities = [self.action(function, {'trial': i}, action = actions[i])[2] for i in range(len(actions))]
        else:
            utilities = []
        
        def obj(params, fixed = []):
            opt_acq_params = params[:opt_acq_len].tolist()
            opt_dec_params = params[opt_acq_len:].tolist()
            
            if type(fixed) == float:
                fixed = [fixed]
            
            fixed_acq_params = fixed[:fixed_acq_len]
            fixed_dec_params = fixed[fixed_acq_len:]
            
            acq_params = []
            for i in range(len(self.acq_params)):
                if i in fixed_acq_idx:
                    acq_params.append(fixed_acq_params.pop(0))
                else:
                    acq_params.append(opt_acq_params.pop(0))
            dec_params = []
            for j in range(len(self.dec_params)):
                if j in fixed_dec_idx:
                    dec_params.append(fixed_dec_params.pop(0))
                else:
                    dec_params.append(opt_dec_params.pop(0))
                
            self.acq_params = acq_params
            self.dec_params = dec_params
                        
            a = -self.joint_log_posterior(function, actions, utilities = utilities)
            return a
        
        params = [self.acq_params[i] for i in range(len(self.acq_params)) if i not in fixed_acq_idx] + [self.dec_params[j] for j in range(len(self.dec_params)) if j not in fixed_dec_idx]
        fixed = [self.acq_params[i] for i in range(len(self.acq_params)) if i in fixed_acq_idx] + [self.dec_params[j] for j in range(len(self.dec_params)) if j in fixed_dec_idx]
        bounds =  [self.acq_bounds[i] for i in range(len(self.acq_bounds)) if i not in fixed_acq_idx] + [self.dec_bounds[j] for j in range(len(self.dec_bounds)) if j not in fixed_dec_idx]
        
        if method == 'DE':
            x = opt.differential_evolution(obj, bounds, args = fixed)
        else:
            x = opt.minimize(obj, params, args = fixed, method = method, bounds = bounds)
        print (x)
        params = x.x.tolist()
        opt_acq_params = params[:opt_acq_len]
        opt_dec_params = params[opt_acq_len:]
        for i in range(len(self.acq_params)):
            if i not in fixed_acq_idx:
                self.acq_params[i] = opt_acq_params.pop(0)
        for j in range(len(self.dec_params)):
            if j not in fixed_dec_idx:
                self.dec_params[j] = opt_dec_params.pop(0)
        
    
    def joint_log_likelihood(self, function, actions, utilities = []):
        
        if len(utilities) == 0:
            utilities = [[] for i in range(len(actions))]
        all_ll = []
        joint_ll = 0
        for i in range(len(actions)):
            acq_other = {'trial': i}
            action = actions[i]
            utility = utilities[i]
            ll = self.log_likelihood(function, action, acq_other = acq_other, utility = utility)
            all_ll.append(ll)
            joint_ll += ll
        return joint_ll
    
    
    def joint_log_posterior(self, function, actions, utilities = []):
        
        lp = self.log_prior()
        ll = self.joint_log_likelihood(function, actions, utilities)
        return lp + ll