#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import importlib
import json
import numpy as np
import scipy.stats as st

def write_functions(config_file,save=False):
    config=importlib.import_module(config_file)
    miny_tmp=np.random.uniform(0.,.1)*config.bar_height
    maxy=np.random.uniform(.9,1.)*config.bar_height
    x=np.arange(0,config.nbars)
    norm_func={}

    functions={'pos_linear':lambda x:x,
           'neg_linear':lambda x:-x,
           'pos_power':lambda x:x**2,
           'neg_power':lambda x:-(x**2),
           'pos_quad':lambda x:(x-config.pos_quad_offset-(config.nbars/2.))**2,
           'neg_quad':lambda x:-((x-config.neg_quad_offset-(config.nbars/2.))**2),
           'sin':lambda x:np.sin(x),
           'sinc':lambda x:np.sin(x-config.sinc_offset-(config.nbars/2.))/(x-config.sinc_offset-(config.nbars/2.)),
           'sinc_compressed': lambda x: np.sin(x/2.-30.000001)/(x/2.-30.000001),
           'se':lambda x: st.multivariate_normal.rvs(np.zeros(len(x)),np.array([[np.exp(-((xi-xj)**2/float(2*config.se_length**2))) for xi in x] for xj in x]))      
           } 

    for f in functions:
        if f in config.functions:
            if f=='sinc_compressed':
                miny=miny_tmp+75.
            else:
                miny=miny_tmp
            y=functions[f](x)
            norm_func[f]=[(maxy-miny)/(max(y)-min(y))*(yi-max(y))+maxy for yi in y]
    if save==True:
        with open('{0}_{1}_functions.json'.format(config.experiment,config.version), 'w') as fp:
            json.dump(norm_func, fp)
    return norm_func


