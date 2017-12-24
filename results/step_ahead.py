# -*- coding: utf-8 -*-

import pandas as pd
from functions import *

def test_functions(df,kern,i,functions):
    sessionIds=df['sessionId']
    for function in functions:
        for j in xrange(1,len(df['test_response'].iloc[0])):
            if j%i==0:
                results=[]
                params=[]
                for sessionId in sessionIds:
                    opt,ll,pseudo_r2=step_ahead(df,sessionId,kern,function,j)
                    results.append(pseudo_r2)
                    params.append(opt)
                df[function.__name__+'_'+str(j)+'_pseudoR2']=results
                df[function.__name__+'_'+str(j)+'_fitted']=params
    return df
                
            

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

#results=test_functions(dfv7,kern,5,[UCB,UCB_greedy,UCB_decay,var_greedy])


#sessionId='kUWzkuR2v1UX7S7MMDt1SVsf7h6dDoAn'
#r=step_ahead(dfv7,sessionId,kern,UCB,5)
