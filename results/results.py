#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import mpmath as mp
import GPy
from scipy.optimize import differential_evolution
import scipy.stats as st
import inspect

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
    
def UCBtrunk(mean,var,x,trials_remaining,explore,temp):
    if trials_remaining==1:
        return mean_greedy(mean,var,x,trials_remaining,explore,temp)
    else:
        return UCB(mean,var,x,trials_remaining,explore,temp)
        
def var_greedy(mean,var,x,trials_remaining,temp):
    return softmax(var,temp)
    
def mean_greedy(mean,var,x,trials_remaining,temp):
    return softmax(mean,temp)
        
def participant_log_like(df,sessionId,kern,acquisition,params,xi):
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
    
def fit_participant(df,sessionId,kern,acquisition):
    """
    args:
        df: data loaded from json file, Pandas Data Frame
        sessionId: for one particular participant, str
        kern: RBF kernel optimized on all functions, GPy kern
        acquisition: choice of acquisition function
    returns:
        list of mle parameters
    """
    bounds_dict={'explore':(0.,10.),'temp':(0.,1000.)} #bounds for each free parameter
    args=inspect.getargspec(acquisition)[0]
    bounds=[bounds_dict[a] for a in args[1:] if a in bounds_dict] #put bounds in list
    test_response=(df[df['sessionId']==sessionId]).iloc[0]['test_response'] #list of responses for participant
    
    #log likelihood of a set of parameters is the sum of the likelihoods of each new trial given the parameters and all previous trials
    result=differential_evolution(lambda params: np.sum(-participant_log_like(df,sessionId,kern,acquisition,params,xi) for xi in xrange(1,len(test_response))),bounds)
    return result
    
def participant_posterior_samples(df,sessionId,kern,acquisition,N):
    """
    args:
        df: data loaded from json file, Pandas Data Frame
        sessionId: for one particular participant, str
        kern: RBF kernel optimized on all functions, GPy kern
        acquisition: choice of acquisition function
        N: number of samples
    returns:
        Nx(number of params) list of all posterior samples
    """
    priors_dict={'explore':st.gamma(1),'temp':st.gamma(10)} #priors for each parameter
    args=inspect.getargspec(acquisition)[0]
    priors=[priors_dict[a] for a in args[1:] if a in priors_dict] #put priors in list
    test_response=(df[df['sessionId']==sessionId]).iloc[0]['test_response'] #list of responses for participant

    samples=[]
    r=[priors[i].rvs() for i in xrange(len(priors))] #initial samples from priors
    p=np.sum([participant_log_like(df,sessionId,kern,acquisition,r,xi) for xi in xrange(1,len(test_response))]) #likelihoods of initial samples
    for i in xrange(N):
        rn=[priors[i].rvs() for i in xrange(len(priors))] #proposals
        pn=np.sum([participant_log_like(df,sessionId,kern,acquisition,rn,xi) for xi in xrange(1,len(test_response))]) #proposal log likelihoods
        if pn>=p:
            p=pn
            r=rn
        else:
            u=np.log(np.random.rand())
            if u<pn-p:
                p=pn
                r=rn
        samples.append(r)
    return samples
    
        
df=pd.read_json('algoals-all.json')
dfv7=df[df['experimentVersion'].str.contains('0.7.')] #include only 0.7.0 and later
dfv7=dfv7[dfv7['age'].apply(lambda x: int(x))<99]
dfv7=dfv7[~dfv7['describe_strategy'].str.contains('EXCLUDE')]
          
all_functions=dfv7['function'].apply(lambda x: tuple(x)).unique()
all_function_x=np.array([a for b in [np.arange(len(f)) for f in all_functions] for a in b])[:,None]
all_function_y=np.array([a for b in all_functions for a in b])[:,None]

kern=GPy.kern.RBF(1)
m=GPy.models.GPRegression(all_function_x,all_function_y,kern)
m.Gaussian_noise.variance=.001
m.Gaussian_noise.variance.fix()
m.optimize() #fit kernel parameters to all functions
kern=m.kern

#add mle parameter estimates to dataframe
#dfv7['mle']=[fit_participant(dfv7,participant,kern,UCB) for participant in dfv7['sessionId']]

#add posterior samples to dataframe
#dfv7['posterior_samples']=[participant_posterior_samples(dfv7,participant,kern,UCB,1000) for participant in dfv7['sessionId']]
        