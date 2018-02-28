from flask import Flask, render_template, request
import GPy
import numpy as np
import pandas as pd
import json

from train_function_samples import get_data
from acquisition_functions import set_kernel_parameter
from agent_model import AgentModel
from acquisition_functions2 import Random, MeanGreedy, UCB 
from decision_functions2 import Deterministic, PropToUtility, Softmax


data = get_data('../data.json')
functions = {f: data[data['function_name'] == f].iloc[0]['function'] for f in data['function_name'].unique()}
training_params = pd.read_json('model_params.json')
acqmap = {'ucb': UCB}
decmap = {'deterministic': Deterministic, 'softmax': Softmax}
kernels = {'pos_linear': GPy.kern.Linear(1) + GPy.kern.Bias(1), 'neg_quad': GPy.kern.Poly(1, order = 2), 'sinc_compressed': GPy.kern.RBF(1)}
function_names = {'pos_linear': 'Positive Linear', 'neg_quad': 'Negative Quadratic', 'sinc_compressed': 'Compressed Sinc'}

app=Flask(__name__)
app.secret_key=''
app.config['SESSION_TYPE']='filesystem'

@app.route('/', methods=['GET', 'POST'])
def new_simulation():
    
    data = request.args
    if len(data) > 0:
        dataDict = {key: float(data[key]) for key in data if key not in ['trials', 'acquisition', 'decision', 'reward']}
        trials = int(data['trials'])
        acquisition = acqmap[data['acquisition']]
        decision = decmap[data['decision']]
        reward = data['reward']
        
        agent = AgentModel(acquisition, decision, **dataDict)
        acquisition_parent_class = str(type(agent.acquisition_function).__mro__[1]).split('.')[1][:-2]
        if acquisition_parent_class == 'GPAcquisitionFunction':
            kern = kernels[reward]
            kern_param_names = kern.parameter_names()
            kern_params = training_params[reward]['kernel_posterior_means']
            for i in range(len(kern_params)):
                set_kernel_parameter(kern, kern_param_names[i], kern_params[i])
            noise_variance = training_params[reward]['noise_posterior_mean']
            agent.acquisition_function.set_kernel(kern)
            agent.acquisition_function.set_noise_variance(noise_variance)
        
        function = functions[reward]
        scale = training_params[reward]['scale']
        scaled_function = [f / scale  for f in function]
        responses = agent.simulate_responses(scaled_function, trials)
        for i in range(len(responses)):
            responses[i]['mean'] = (responses[i]['mean'] * scale).ravel().tolist()
            responses[i]['std'] = (responses[i]['std'] * scale).ravel().tolist()
            responses[i]['next_y'] = responses[i]['next_y'] * scale
            responses[i]['observed_y'] = (np.array(responses[i]['observed_y']) * scale).tolist()
            responses[i]['utility'] = responses[i]['utility'].ravel().tolist()
            responses[i]['p'] = responses[i]['p'].ravel().tolist()
            
        acquisition_name = str(type(agent.acquisition_function)).split('.')[1][:-2]
        decision_name = str(type(agent.decision_function)).split('.')[1][:-2]
        sim = {'function': function, 'responses': responses, 'Trials': trials, 'acquisition': acquisition_name, 'decision': decision_name, 'acquisition_params': agent.acquisition_function.parameters, 'decision_params': agent.decision_function.parameters, 'function_name': function_names[reward]}
        
        with open('static/simulations.json', 'r') as jf:
            all_sims = json.load(jf)
        all_sims.append(sim)
        with open('static/simulations.json', 'w') as jf:
            json.dump(all_sims, jf)
    
    return render_template('new_simulation.html')


@app.route('/view_simulation')
def view_simulation():
    
    with open('static/simulations.json', 'r') as jf:
        all_sims = json.load(jf)
    print (all_sims)
    print (len(all_sims))
            
    return render_template('view_simulation.html')

        

@app.route('/participants')
def participants():
    
    return render_template('fit_participant.html')
    
    
if __name__=="__main__":
    app.run()
    