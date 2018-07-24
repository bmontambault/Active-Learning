import numpy as np
import scipy.stats as st


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
            all_left_rewards = [rewards[i] for i in range(len(rewards)) if actions[i] < best_action]
            all_right_rewards = [rewards[i] for i in range(len(rewards)) if actions[i] > best_action]
            
            if len(all_right_actions) > 0:
                right_action = all_right_actions[0]
                right_reward = all_right_rewards[0]
            else:
                right_action = None
                right_reward = None
            if len(all_left_actions) > 0:
                left_action = all_left_actions[0]
                left_reward = all_left_rewards[0]
            else:
                left_action = None
                left_reward = None
            
            #print (left_action, right_action)
            if right_action == None and left_action != None:
                x1 = best_action
                y1 = best_reward
                x2 = left_action
                y2 = left_reward
            elif right_action != None and left_action == None:
                x1 = right_action
                y1 = right_reward
                x2 = best_action
                y2 = best_reward
            elif right_action != None and left_action != None:
                x1 = right_action
                y1 = right_reward
                x2 = left_action
                y2 = left_reward
            else:
                x1 = None
                x2 = None
                
            print (x2, x1)
            if x1 == None:
                self.dk = None
            elif x1 == x2:
                self.dk = 0
            else:
                self.dk = (y1 - y2) / (x1 - x2)
            if ((x1 == self.all_x[-1] and self.dk > 0) or (x1 == self.all_x[0] and self.dk < 0)) and self.dk != None:
                    self.dk = 0
        else:
            self.dk = None
            x1 = None
        self.x1 = x1
            
            
    def __call__(self, learning_rate):
        if self.dk == None:
            return np.ones(len(self.all_x)) / len(self.all_x)
        else:
            next_x = max(0, min(self.x1 + self.dk * learning_rate, max(self.all_x)))
            #print (next_x)
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
        return self.mean + explore * self.var
    
    
    
class EI(GPAcq):
    
    init_params = []
    bounds = []
    
    def __call__(self):
        expected_max = np.max(self.mean)
        expected_gain = self.mean - expected_max
        z = expected_gain / np.sqrt(self.var)
        return expected_gain * st.norm.cdf(z) + np.sqrt(self.var) * st.norm.pdf(z)
    
    
