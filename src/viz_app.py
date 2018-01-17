from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import json

from utils import fit_kern, gp, scale, get_next, softmax, deterministic
from acquisition_functions import ucb_acq, mes_acq


upper_bound = 600
rewardFunctions = pd.read_json('active-learning_0.8.0_functions.json')
kern = fit_kern(rewardFunctions)
acqFunctions = {'ucb':[ucb_acq, 1, True, False, False], 'mes':[mes_acq, 2, True, False, True]}
decFunctions = {'softmax':softmax, 'deterministic':deterministic}
app=Flask(__name__)

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
  
'''    
@app.route('/<acq>', methods=['GET', 'POST'])
def acqFunction(acq):   
    
    data = request.args
    if acq not in acquisitionFunctions:
        return render_template("main.html", acq = None)
    
    elif 'action' not in data:
        return render_template('{}.html'.format(acq), acq = acq)
    
    elif data['action'] == 'next':
        
        decisionFunction = decisionFunctions['softmax']
        
        #acquisitionFunction, isGP, needsDecFunc = acquisitionFunctions[acq]
        #decArgCount = len(signature(decisionFunction).parameters) - 1
        #decArgs = [data[d] for d in list(data)[8:9 + decArgCount - 1]]
        #acqArgs = [data[d] for d in list(data)[9 + decArgCount - 1:]]
        
        #observed_x = ast.literal_eval(data['observed_x'])
        #observed_y = ast.literal_eval(data['observed_y'])
        #trials_remaining = int(data['remaining_trials'])
        #function = rewardFunctions[data['function']]
        
        #params, next_x = get_next(acquisitionFunction, observed_x, observed_y, function, , isGP)
        
        
        #args = get_args(acquisitionFunction, **data)
        #args['domain'] = range(data['datapoints'])
        

        #print (data)
        #args = get_args(acquisitionFunction, **data)
        #print (args)
        #return json.dumps({'observedX':[], 'observedY':[], 'plot':''})
        #params, next_x = get_next(acquisitionFunction, *args)
        '''

'''
@app.route('/MES', methods=['GET', 'POST'])
def MES():
    
    data = request.args
    if 'action' not in data:
        return render_template('mes.html', strategy = "MES")
    
    elif data['action'] == 'next':
        function = functions[data['function']]
        observed_x = ast.literal_eval(json.loads(data['observedX']))
        observed_y = ast.literal_eval(json.loads(data['observedY']))
        
        precision = float(data['cdfPrecision'])
        nsamples = int(data['gumbelSamples'])
        
        mean, std, next_x, utility, params = get_next_gp(observed_x, observed_y, function, kern, mes, (precision, upper_bound, nsamples))
        next_x = int(next_x)
        next_y = float(function[next_x])
        observed_x.append(next_x)
        observed_y.append(next_y)
        
        if np.any(np.isnan(params)):
            a = np.NaN
            b = np.NaN
        else:
            a, b = params
            
        fig = plot_mes(mean, std, a, b, utility, upper_bound)
        ax = fig.axes[0]
        ax.plot(observed_x[:-1], observed_y[:-1], 'bo')
        ax.plot(observed_x[-1], observed_y[-1], 'ro')
        ax.plot([float(f) for f in function], 'g')
        plot = mpld3.fig_to_html(fig)
                
        return json.dumps({'observedX':observed_x, 'observedY':observed_y, 'plot':plot})
    
    else:
        return json.dumps({'observedX': [], 'observedY':[], 'plot':''}) 
    
    
@app.route('/SGD', methods=['GET', 'POST'])
def SGD():

    data = request.args
    if 'action' not in data:
        return render_template('sgd.html', strategy = "SGD")
    
    elif data['action'] == 'next':
        function = functions[data['function']]
        observed_x = ast.literal_eval(json.loads(data['observedX']))
        observed_y = ast.literal_eval(json.loads(data['observedY']))
        
        a = float(data['gammaShape'])
        b = float(data['gammaScale'])
        
        next_x, p = sgd(observed_x, observed_y, function, a, b)
        next_x = int(next_x)
        next_y = float(function[next_x])
        observed_x.append(next_x)
        observed_y.append(next_y)
        
        fig = plot_sgd(p, upper_bound / 2)
        ax = fig.axes[0]
        ax.plot(observed_x[:-1], observed_y[:-1], 'bo')
        ax.plot(observed_x[-1], observed_y[-1], 'ro')
        ax.plot([float(f) for f in function], 'g')
        plot = mpld3.fig_to_html(fig)
        
        return json.dumps({'observedX':observed_x, 'observedY':observed_y, 'plot':plot})
    
    else:
        return json.dumps({'observedX': [], 'observedY':[], 'plot':''})
'''
        
if __name__=="__main__":
    app.run() 