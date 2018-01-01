from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import json
import ast
import mpld3

from utils import fit_kern, gp, plot_gp, scale, get_next_gp
from mes import mes, plot_mes
from sgd import sgd, plot_sgd

upper_bound = 600
functions = pd.read_json('active-learning_0.8.0_functions.json')
kern = fit_kern(functions)
app=Flask(__name__)

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
        
        
if __name__=="__main__":
    app.run() 