import numpy as np
import scipy.stats as st
from acquisition_utils import z_score, fit_gumbel, sample_gumbel, get_function_samples

import matplotlib.pyplot as plt

class Acq(object):
    
    isGP = False
    
    def __init__(self, choices):
        self.choices = choices


class GPAcq(Acq):
    
    isGP = True
    
    def __init__(self, choices, mean, var):
        super().__init__(choices)
        self.mean = np.array(mean)
        self.var = np.array(var)


class Random(Acq):
    
    init_params = []
    bounds = []
    
    def __call__(self):
        return np.zeros(len(self.choices))
    
    

class LocalMove(Acq):
    
    init_params = []
    bounds = []
    
    def __init__(self, choices, action_1):
        super().__init__(choices)
        self.action_1 = action_1
    
    def __call__(self):
        if self.action_1 == None:
            return np.ones(len(self.choices)) / len(self.choices)
        else:
            return np.array([np.exp(-abs(self.action_1 - x)) for x in self.choices]).ravel()
        
        
class SGD(Acq):
    
    init_params = [50.]
    bounds = [(1., 800)]
    
    def __init__(self, choices, unique_actions, unique_rewards):
        self.choices = choices
        self.unique_actions = unique_actions
        self.unique_rewards = unique_rewards

    def __call__(self, learning_rate):
        if len(self.unique_actions) > 1:
            dk = (self.unique_rewards[-1] - self.unique_rewards[-2]) / (self.unique_actions[-1] - self.unique_actions[-2])
            last_action = self.unique_actions[-1]
            next_action = max(0, min(last_action + dk * learning_rate, max(self.choices))) 
            return np.array([np.exp(-abs(next_action - x)) for x in self.choices])
        else:
            return np.ones(len(self.choices))
        
        
class SGDMax(SGD):
    
    def __call__(self, learning_rate):
        if len(self.unique_actions) > 1:
            max_idx = np.argmax(self.unique_rewards)
            if max_idx == len(self.unique_actions) - 1:
                dk = (self.unique_rewards[-1] - self.unique_rewards[-2]) / (self.unique_actions[-1] - self.unique_actions[-2])
            else:
                dk = (self.unique_rewards[-1] - self.unique_rewards[max_idx]) / (self.unique_actions[-1] - self.unique_actions[max_idx])
            last_action = self.unique_actions[-1]
            next_action = max(0, min(last_action + dk * learning_rate, max(self.choices))) 
            return np.array([np.exp(-abs(next_action - x)) for x in self.choices])
        else:
            return np.ones(len(self.choices))
        
        
class RandomSGD(SGD):
    
    def __call__(self, learning_rate):
        if len(self.unique_actions) > 1:
            dk = []
            for i in range(100):
                xi1 = np.random.choice(range(len(self.unique_actions) - 1))
                xi2 = np.random.choice(range(xi1 + 1, len(self.unique_actions)))
                action_1 = self.unique_actions[xi1]
                action_2 = self.unique_actions[xi2]
                reward_1 = self.unique_rewards[xi1]
                reward_2 = self.unique_rewards[xi2]
                dk.append((reward_2 - reward_1) / (action_2 - action_1)) 
            last_action = self.unique_actions[-1]
            next_actions = [max(0, min(last_action + dki * learning_rate, max(self.choices))) for dki in dk]
            return np.mean([[np.exp(-abs(next_action - x)) for x in self.choices] for next_action in next_actions], axis = 0)
        else:
            return np.ones(len(self.choices))
        
        
class RandomSGDMax(SGD):
    
    def __call__(self, learning_rate):
        if len(self.unique_actions) > 1:
            max_idx = np.argmax(self.unique_rewards)
            max_action = self.unique_actions[max_idx]
            d = np.array([np.exp(-abs(max_action - x)) for x in self.unique_actions[:-1]])
            dk = []
            for i in range(100):
                p1 = d / d.sum()
                xi1 = np.random.choice(range(len(self.unique_actions) - 1), p = p1)
                p2 = d[xi1:] / d[xi1:].sum()
                xi2 = np.random.choice(range(xi1 + 1, len(self.unique_actions)), p = p2)
                action_1 = self.unique_actions[xi1]
                action_2 = self.unique_actions[xi2]
                reward_1 = self.unique_rewards[xi1]
                reward_2 = self.unique_rewards[xi2]
                dk.append((reward_2 - reward_1) / (action_2 - action_1)) 
            last_action = self.unique_actions[-1]
            next_actions = [max(0, min(last_action + dki * learning_rate, max(self.choices))) for dki in dk]
            return np.mean([[np.exp(-abs(next_action - x)) for x in self.choices] for next_action in next_actions], axis = 0)
        else:
            return np.ones(len(self.choices))


class Exploit(GPAcq):
    
    init_params = []
    bounds = []
    
    def __call__(self):
        return self.mean
    

class Explore(GPAcq):
    
    init_params = []
    bounds = []
    
    def __call__(self):
        return self.var
        
        
class Phase(GPAcq):
    
    init_params = [12]
    bounds = [(.1, 24)]
    
    def __init__(self, all_x, mean, var, trial):
        super().__init__(all_x, mean, var)
        self.trial = trial
        
    def __call__(self, p):
       if self.trial < p:
           return np.ones(len(self.all_x)) / len(self.all_x)
       else:
           return self.mean
        
    
    
class UCB(GPAcq):
    
    init_params = [1.]
    bounds = [(.001, 100.)]
    
    def __call__(self, explore):
        u = self.mean + explore * self.var
        return u
    
    
class EI(GPAcq):
    
    init_params = []
    bounds = []
    
    def __call__(self):
        expected_max = np.max(self.mean)
        expected_gain = self.mean - expected_max
        z = expected_gain / np.sqrt(self.var)
        return expected_gain * st.norm.cdf(z) + np.sqrt(self.var) * st.norm.pdf(z)


class MES(GPAcq):
    
    init_params = []
    bounds = []
    
    def __init__(self, choices, mean, var, remaining_trials):
        super().__init__(choices, mean, var)
        self.remaining_trials = remaining_trials
    
    def __call__(self):
        if self.remaining_trials == 1:
            return self.mean
        else:
            std = np.sqrt(self.var)
            a, b = fit_gumbel(self.mean, std, .01)
            ymax_samples = sample_gumbel(a, b, 100)
            z_scores = z_score(ymax_samples, self.mean, std)
            log_norm_cdf_z_scores = st.norm.logcdf(z_scores)
            log_norm_pdf_z_scores = st.norm.logpdf(z_scores)
            return np.mean(z_scores * np.exp(log_norm_pdf_z_scores - (np.log(2) + log_norm_cdf_z_scores)) - log_norm_cdf_z_scores, axis = 1)
    

class MRS(GPAcq):

    init_params = []
    bounds = []

    def __init__(self, choices, mean, var, kernel, actions, rewards, remaining_trials):
        super().__init__(choices, mean, var)
        self.actions = actions
        self.rewards = rewards
        self.kernel = kernel
        self.remaining_trials = remaining_trials   

    def __call__(self):
        
        if len(self.actions) == 0:
            return np.ones(len(self.choices))
        elif self.remaining_trials == 1:
            return self.mean  
        else:
            current_regret = self.current_regret()
            ysamples = np.array([np.random.normal(self.mean[i], self.var[i], 10) for i in range(len(self.mean))])
            expected_regret = self.expected_regret(ysamples)
            return current_regret - expected_regret
        
    def minimum_regret(self, functions):
        regret = [np.mean([np.max(f) - f[x] for f in functions]) for x in range(len(functions))]
        return np.min(regret)
    
    def current_regret(self):
        pass
    
    def expected_regret(self, ysamples):
        pass
    
    
class MCRS(MRS):
    
    def expected_regret(self, ysamples):
        #u = np.array([np.mean([self.minimum_regret(get_function_samples(self.kernel, self.actions + [self.all_x[i]], self.rewards + [ysamples[i][j]], self.all_x, 5).T) * (self.remaining_trials - 1) + ( - ysamples[i][j]) for j in range(len(ysamples[i]))]) for i in range(len(self.all_x))])
        #print (u)
        #return u
        all_regret = []
        for i in range(len(self.choices)):
            ysample_regret = []
            for j in range(len(ysamples[i])):
                fsamples = get_function_samples(self.kernel, self.actions + [self.choices[i]], self.rewards + [ysamples[i][j]], self.choices, 5).T
                fsample_regret = []
                for f in fsamples:
                    regret = np.mean([np.max(f) - f[x] for x in self.choices])
                    min_regret = np.min(regret)
                    y_regret = np.max(f) - ysamples[i][j]
                    #print (min_regret, y_regret, self.all_x[i], ysamples[i][j])
                    #plt.plot(f)
                    #plt.show()
                    total_regret = min_regret * (self.remaining_trials - 1) + y_regret
                    fsample_regret.append(total_regret)
                ysample_regret.append(np.mean(fsample_regret))
            all_regret.append(np.mean(ysample_regret))
        return all_regret
                    
    def current_regret(self):
         return self.minimum_regret(get_function_samples(self.kernel, self.actions, self.rewards, self.choices, 5).T) * self.remaining_trials

class MSRS(MRS):
    
    def current_regret(self):
         return self.minimum_regret(get_function_samples(self.kernel, self.actions, self.rewards, self.choices, 5).T)
    
    def expected_regret(self, ysamples):
        u = np.array([np.mean([self.minimum_regret(get_function_samples(self.kernel, self.actions + [self.choices[i]], self.rewards + [ysamples[i][j]], self.choices, 5).T) for j in range(len(ysamples[i]))]) for i in range(len(self.choices))])
        #print(u)
        return u