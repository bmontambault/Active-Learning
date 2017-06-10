import numpy as np
import scipy.stats as st
import json

version="0.2.0"
max_height=500
bar_width=15
nbars=80
trials=25
predict_trials=10

se_length=5
sinc_offset=15
neg_quad_offset=15
pos_quad_offset=-15


functions={'pos_linear':lambda x:x,
           'neg_linear':lambda x:-x,
           'pos_power':lambda x:x**2,
           'neg_power':lambda x:-(x**2),
           'pos_quad':lambda x:(x-pos_quad_offset-(nbars/2.))**2,
           'neg_quad':lambda x:-((x-neg_quad_offset-(nbars/2.))**2),
           'sin':lambda x:np.sin(x),
           'sinc':lambda x:np.sin(x-sinc_offset-(nbars/2.))/(x-sinc_offset-(nbars/2.)),
           'se':lambda x: st.multivariate_normal.rvs(np.zeros(len(x)),np.array([[np.exp(-((xi-xj)**2/float(2*se_length**2))) for xi in x] for xj in x]))      
}

def write_functions():
    miny=np.random.uniform(0.,.2)*max_height
    maxy=np.random.uniform(.8,1.)*max_height
    x=np.arange(0,nbars)
    y={f:[1 if np.isnan(yi) else yi for yi in functions[f](x)] for f in functions}
    norm_func={f:[(maxy-miny)/(max(y[f])-min(y[f]))*(val-max(y[f]))+maxy for val in y[f]] for f in functions}
    with open('functions.json', 'w') as fp:
        json.dump(norm_func, fp)

def load():
    with open('data.json') as json_data:
        data=json.load(json_data)
        json_data.close()
    return data
    
def init():
    keys=[u'function', u'describe_goal', u'test_response', u'goal', u'age', u'final_score', u'describe_unclear', u'describe_strategy', u'ID', u'function_name']
    data={key:[] for key in keys}
    with open('data.json', 'w') as json_data:
        json.dump(data,json_data)
        
