from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import json

from utils import fit_kern, gp, scale, get_next, softmax, deterministic
from acquisition_functions import ucb_acq, mes_acq

import GPy

upper_bound = 500
rewardFunctions = pd.read_json('active-learning_0.8.0_functions.json')
kern = fit_kern(rewardFunctions)
fkerns = {'pos_linear': GPy.kern.Linear(1, 10),
          'neg_quad': GPy.kern.Poly(1, 10, order = 2),
          'sinc_compressed': GPy.kern.RBF(1, 100000, 10)}

acqFunctions = {'ucb':[ucb_acq, 1, True, False, False], 'mes':[mes_acq, 2, True, False, True]}
decFunctions = {'softmax':softmax, 'deterministic':deterministic}
app=Flask(__name__)
app.secret_key=''
app.config['SESSION_TYPE']='filesystem'

@app.route('/', methods=['GET', 'POST'])
def main():
    
    data = request.args
    if 'action' not in data:
        return render_template("main2.html", upper_bound = upper_bound)
    
    elif data['action'] == 'next':
        observed_x = [float(x) for x in data['observed_x'].split(',') if len(x) > 0]
        observed_y = [float(y) for y in data['observed_y'].split(',') if len(y) > 0]
        
        trials_remaining = int(data['trials']) - int(data['trialsPassed'])
        function = rewardFunctions[data['rewardFunction']]
        kern = fkerns[data['rewardFunction']]
        acqFunction, acqArgCount, isGP, takesRemaining, takesUpperBound = acqFunctions[data['acq']]
        if 'dec' in data:
            decFunction = decFunctions[data['dec']]
        else:
            decFunction = None
        acqArgs = [float(data[d]) for d in list(data)[9:9 + acqArgCount]]
        decArgs = [float(data[d]) for d in list(data)[9 + acqArgCount:]]
        
        if isGP:
            acqArgs.append(kern)
        if takesRemaining:
            acqArgs.append(trials_remaining)
        if takesUpperBound:
            acqArgs.append(upper_bound)
        
        params, p, next_x = get_next(observed_x, observed_y, function, upper_bound, acqFunction, acqArgs, dec = decFunction, decArgs = decArgs, isGP = isGP)
        next_y = function[next_x]
        observed_x.append(next_x)
        observed_y.append(next_y)
        params = {key: params[key].ravel().tolist() if isinstance(params[key], np.ndarray) else params[key]  for key in params.keys()}
        
        return json.dumps({'function':function.tolist(), 'next_x':next_x, 'next_y':next_y, 'params':params, 'prob':p.ravel().tolist()})
        
if __name__=="__main__":
    app.run() 