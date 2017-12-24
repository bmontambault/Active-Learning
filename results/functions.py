# -*- coding: utf-8 -*-

import mpmath as mp
import numpy as np
import GPy
import inspect
from scipy.optimize import differential_evolution


def softmax(f,temp):
    """
    args:
        f: acquisition function, Nx1 numpy array
        temp: temperature parameter, float
    returns:
        softmax function, Nx1 numpy array
    """
    ex=mp.matrix(((f-f.max())/temp).tolist()).apply(mp.exp)
    return ex.apply(mp.log)-mp.log(sum(ex))
    
#mean,var,x,trials_remaining,explore, and temp are arguments in all decision functions, but not all of them are used in every function
def UCB(mean,var,x,trials_remaining,explore,temp):
    """
    args:
        mean: expected mean of latent function, Nx1 numpy array
        var: expected variance of latent function, Nx1 numpy array
        x: participants x choice at the current trial, int
        trials_remaining: trials left including the current trial, int
        explore: explore parameter, float
        temp: softmax temperature parameter, float
    returns:
        full decision function, Nx1 numpy array
    """
    return softmax(mean+var*explore,temp)[x]
    
def UCB_greedy(mean,var,x,trials_remaining,explore,temp):
    if trials_remaining==1:
        return mean_greedy(mean,var,x,trials_remaining,explore,temp)
    else:
        return UCB(mean,var,x,trials_remaining,explore,temp)
    
def UCB_decay(mean,var,x,trials_remaining,explore,temp,decay):
    decayed_explore=explore*np.exp(-decay*(1./trials_remaining))
    return UCB(mean,var,x,trials_remaining,decayed_explore,temp)
        
def var_greedy(mean,var,x,trials_remaining,temp):
    return softmax(var,temp)
    
def mean_greedy(mean,var,x,trials_remaining,temp):
    return softmax(mean,temp)

def log_like(df,sessionId,kern,acquisition,params,xi):
    """
    args:
        df: data loaded from json file, Pandas Data Frame
        sessionId: for one particular participant, str
        kern: RBF kernel optimized on all functions, GPy kern
        acquisition: choice of acquisition function
        params: for the acquisition function, list
        xi: next trial number (e.g. if it's the first trial, 0), int
    returns:
        output of the chosen acquisition function, Nx1 numpy array
    """
    participant=(df[df['sessionId']==sessionId]).iloc[0] #get row with passed sessionId
    test_response=participant['test_response']
    function=participant['function']
    
    previous_x=np.array(test_response)[:,None][:xi] #indices of all bars that have been chosen so far (excluding xi)
    previous_y=np.array(function)[previous_x] #observed heights of all bars that have been chosen so far
    gp=GPy.models.GPRegression(previous_x,previous_y,kern) #train GP on indices and observed heights
    mean,var=gp.predict(np.arange(len(function))[:,None]) #expected mean and variance of latent function
    trials_remaining=len(test_response)-xi
    return acquisition(mean,var,test_response[xi],trials_remaining,*params)

def step_ahead(df,sessionId,kern,acquisition,xi):
    """
    args:
        df: data loaded from json file, Pandas Data Frame
        sessionId: for one particular participant, str
        kern: RBF kernel optimized on all functions, GPy kern
        acquisition: choice of acquisition function
    returns:
        list of mle parameters
    """
    bounds_dict={'explore':(0.,10.),'temp':(0.,1000.),'decay':(.0001,100.)} #bounds for each free parameter
    args=inspect.getargspec(acquisition)[0]
    bounds=[bounds_dict[a] for a in args[1:] if a in bounds_dict] #put bounds in list
    if len(bounds)>0:
        result=differential_evolution(lambda params: -log_like(df,sessionId,kern,acquisition,params,xi),bounds)
        opt=result.x
        ll=-result.fun
    else:
        opt=[]
        ll=log_like(df,sessionId,kern,acquisition,[],xi)
    ll_random=np.log(1./len(df[df['sessionId']==sessionId]['function'].iloc[0]))
    pseudo_r2=1-(ll/ll_random)
    return opt,ll,pseudo_r2
    
    
    