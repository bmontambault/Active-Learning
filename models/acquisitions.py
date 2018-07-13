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
    

class RandomMove(Acq):
    
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
        
    
    
class FixedSGD(Acq):
    
    init_params = []
    bounds = []
    
    def __init__(self, all_x, last_x, last_y, second_last_x, second_last_y):
        self.all_x = all_x
        self.last_x = last_x
        self.last_y = last_y
        self.second_last_x = second_last_x
        self.second_last_y = second_last_y
        
        if self.last_x != None and self.second_last_x != None:
            if self.last_x == self.second_last_x:
                self.dk = 0
            else:
                self.dk = (last_y - second_last_y) / (last_x - second_last_x)
        else:
            self.dk = None
    
    
class FixedSGD_MaxScore(FixedSGD):
        
    def __call__(self):
        if self.second_last_x == None:
            return np.ones(len(self.all_x)) / len(self.all_x)
        else:
            next_x = self.last_x + self.dk
            return np.array([np.exp(-abs(next_x - x)) for x in self.all_x]).ravel()
        

class FixedSGD_FindMax(FixedSGD):
    
    def __call__(self):
        if self.second_last_x == None or self.dk == 0:
            return np.ones(len(self.all_x)) / len(self.all_x)
        else:
            next_x = self.last_x + self.dk
            return np.array([np.exp(-abs(next_x - x)) for x in self.all_x]).ravel()

    

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
            next_x = self.last_x + self.dk * learning_rate
            return np.array([np.exp(-abs(next_x - x)) for x in self.all_x]).ravel()


class SGD_MaxScore(SGD):
        
    def __call__(self, learning_rate):
        if self.second_last_x == None or self.dk == None:
            return np.ones(len(self.all_x)) / len(self.all_x)
        else:
            next_x = self.last_x + self.dk * learning_rate
            return np.array([np.exp(-abs(next_x - x)) for x in self.all_x]).ravel()
        

class SGD_FindMax(SGD):
    
    def __call__(self, learning_rate):
        if self.second_last_x == None or self.dk == 0 or self.dk == None:
            return np.ones(len(self.all_x)) / len(self.all_x)
        else:
            next_x = self.last_x + self.dk * learning_rate
            return np.array([np.exp(-abs(next_x - x)) for x in self.all_x]).ravel()
        
        
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
    
class ParameterizedEI(GPAcq):
    
    init_params = [1.]
    bounds = [(.001, 100)]
    
    def __call__(self, explore):
        expected_max = np.max(self.mean)
        expected_gain = self.mean - (expected_max + explore)
        z = expected_gain / np.sqrt(self.var)
        return expected_gain * st.norm.cdf(z) + np.sqrt(self.var) * st.norm.pdf(z)