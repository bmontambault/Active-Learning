import numpy as np
import scipy.stats as st
from acquisition_utils import z_score, fit_gumbel, sample_gumbel, get_function_samples

import matplotlib.pyplot as plt

class Acq(object):
    
    isGP = False
    
    def __init__(self, all_x):
        self.all_x = all_x


class GPAcq(Acq):
    
    isGP = True
    
    def __init__(self, all_x, mean, var):
        super().__init__(all_x)
        self.mean = mean
        self.var = var


class Random(Acq):
    
    init_params = []
    bounds = []
    
    def __call__(self):
        return np.zeros(len(self.all_x))
    

class LocalMove(Acq):
    
    init_params = []
    bounds = []
    
    def __init__(self, all_x, last_x):
        super().__init__(all_x)
        self.last_x = last_x
    
    def __call__(self):
        if self.last_x == None:
            return np.ones(len(self.all_x)) / len(self.all_x)
        else:
            return np.array([np.exp(-abs(self.last_x - x)) for x in self.all_x]).ravel()
        
        
        
class SGD(Acq):
    
    init_params = [100.]
    bounds = [(.001, 1000)]
    
    def __init__(self, all_x, last_x, last_y, second_last_x, second_last_y, unique_second_last_x):
        self.all_x = all_x
        self.last_x = last_x
        self.last_y = last_y
        self.second_last_x = second_last_x
        self.second_last_y = second_last_y
        self.unique_second_last_x = unique_second_last_x
        
        if self.last_x != None and self.unique_second_last_x != None:
            if self.last_x == self.second_last_x:
                self.dk = 0
            else:
                self.dk = (last_y - second_last_y) / (last_x - second_last_x)
                if (self.last_x == self.all_x[-1] and self.dk > 0) or (self.last_x == self.all_x[0] and self.dk < 0):
                    self.dk = 0
        else:
            self.dk = None
            
    def __call__(self, learning_rate):
        if self.second_last_x == None or self.dk == None:
            return np.ones(len(self.all_x)) / len(self.all_x)
        else:
            next_x = max(0, min(self.last_x + self.dk * learning_rate, max(self.all_x)))
            u = np.array([np.exp(-abs(next_x - x)) for x in self.all_x]).ravel()
            return u
        
        
class MaxSGD(Acq):
    
    init_params = [100.]
    bounds = [(.001, 1000)]
    
    def __init__(self, all_x, actions, rewards):
        self.all_x = all_x
        
        if len(actions) > 0:
            best_reward_idx = np.argmax(rewards)
            best_action = actions[best_reward_idx]
            best_reward = rewards[best_reward_idx]
            all_left_actions = sorted([x for x in actions if x < best_action])
            all_right_actions = sorted([x for x in actions if x > best_action])
            
            if len(all_right_actions) > 0:
                right_action = all_right_actions[0]
                right_index = actions.index(right_action)
                right_reward = rewards[right_index]
            else:
                right_action = None
                right_reward = None
                right_index = None
            if len(all_left_actions) > 0:
                left_action = all_left_actions[0]
                left_index = actions.index(left_action)
                left_reward = rewards[left_index]
            else:
                left_action = None
                left_reward = None
                left_index = None
                
            sample = [a for a in [(left_action, left_reward, left_index), (right_action, right_reward, right_index), (best_action, best_reward, best_reward_idx)] if a[-1] != None]
            if len(sample) > 1:
                x1, y1, _ = max(sample, key = lambda x: x[1])
                x2, y2, _ = min(sample, key = lambda x: x[1])
                if x1 == x2:
                    self.dk = 0
                else:
                    self.dk = (y1 - y2) / (x1 - x2)
                if (x1 == self.all_x[-1] and self.dk > 0) or (x1 == self.all_x[0] and self.dk < 0):
                    self.dk = 0
            else:
                self.dk = None
                x1 = None
            self.x1 = x1
        else:
            self.dk = None
            x1 = None
            self.x1 = x1
            
            
    def __call__(self, learning_rate):
        if self.dk == None:
            return np.ones(len(self.all_x)) / len(self.all_x)
        else:
            next_x = max(0, min(self.x1 + self.dk * learning_rate, max(self.all_x)))
            u = np.array([np.exp(-abs(next_x - x)) for x in self.all_x]).ravel()
            return u


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
    
    def __init__(self, all_x, mean, var, remaining_trials):
        super().__init__(all_x, mean, var)
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
    

class MimimumRegretSearch(GPAcq):

    init_params = []
    bounds = []

    def __init__(self, all_x, mean, var, kernel, actions, rewards, remaining_trials):
        super().__init__(all_x, mean, var)
        self.actions = actions
        self.rewards = rewards
        self.kernel = kernel
        self.remaining_trials = remaining_trials   

    def __call__(self):
        
        if len(self.actions) == 0:
            return np.ones(len(self.all_x))
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
    
    
class MinimumCumulativeRegretSearch(MimimumRegretSearch):
    
    def expected_regret(self, ysamples):
        #u = np.array([np.mean([self.minimum_regret(get_function_samples(self.kernel, self.actions + [self.all_x[i]], self.rewards + [ysamples[i][j]], self.all_x, 5).T) * (self.remaining_trials - 1) + ( - ysamples[i][j]) for j in range(len(ysamples[i]))]) for i in range(len(self.all_x))])
        #print (u)
        #return u
        all_regret = []
        for i in range(len(self.all_x)):
            ysample_regret = []
            for j in range(len(ysamples[i])):
                fsamples = get_function_samples(self.kernel, self.actions + [self.all_x[i]], self.rewards + [ysamples[i][j]], self.all_x, 5).T
                fsample_regret = []
                for f in fsamples:
                    regret = np.mean([np.max(f) - f[x] for x in self.all_x])
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
         return self.minimum_regret(get_function_samples(self.kernel, self.actions, self.rewards, self.all_x, 5).T) * self.remaining_trials

class MinimumSimpleRegretSearch(MimimumRegretSearch):
    
    def current_regret(self):
         return self.minimum_regret(get_function_samples(self.kernel, self.actions, self.rewards, self.all_x, 5).T)
    
    def expected_regret(self, ysamples):
        u = np.array([np.mean([self.minimum_regret(get_function_samples(self.kernel, self.actions + [self.all_x[i]], self.rewards + [ysamples[i][j]], self.all_x, 5).T) for j in range(len(ysamples[i]))]) for i in range(len(self.all_x))])
        #print(u)
        return u