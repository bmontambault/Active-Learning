# -*- coding: utf-8 -*-

import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import GPy

def se(x,length,var):
    return var*np.exp(-(((x-x.T)**2)/(2*length**2)))

def lin(x,var):
    return var*np.dot(x,x.T)

def quad(x,var):
    return lin(x,var)**2

def log_like(x,y,cov_func,params):
    cov=cov_func(x,*params)+(np.eye(len(x))+.001)
    return st.multivariate_normal(np.zeros(len(x)),cov).logpdf(y.reshape(1,len(x)))


def IS(x,y,cov_functs,priors,hyperprior,n=1):
    samples=[]
    weights=[]
    for i in xrange(n):
        p=hyperprior.rvs()
        ll=[]
        for j in xrange(len(cov_functs)):
            params=[priors[j][k].rvs() for k in xrange(len(priors[j]))]
            ll.append(log_like(x,y,cov_functs[j],params))
        weights.append(np.array(ll)/sum(ll))
        samples.append(p[0])
    weighted=np.array(samples)*np.array(weights)
    print np.array(weights).mean(axis=0)
    means=weighted.mean(axis=0)
    return means/means.sum()
            


x=np.random.uniform(0,1,size=(25,1))
y=x+np.random.normal(0,.0001,(25,1))


hyperprior=st.dirichlet([.01,.01,.01])
all_priors=[[st.gamma(1.001),st.gamma(1.001)],[st.gamma(1.001)],[st.gamma(1.001)]]

s=IS(x,y,[se,lin,quad],all_priors,hyperprior,1000)