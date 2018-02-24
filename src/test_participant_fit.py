import numpy as np
import pandas as pd
import GPy

from acquisition_functions import set_kernel_parameter
from train_function_samples import get_data

from agent_model import AgentModel
from acquisition_functions2 import UCB, MeanGreedy, CoolingUCB, Random
from decision_functions2 import Softmax, CoolingSoftmax, PropToUtility


def load_model_params():
    
    return pd.read_json('model_params.json')


def positive(v):
    
    if v > 0:
        return 0
    else:
        return -100
    

def zero_one(v):
    
    if v > 0 and v < 1:
        return 0
    else:
        return -np.inf


data = get_data('../data.json')
kernel_params = load_model_params()
kernels = {'pos_linear': GPy.kern.Linear(1) + GPy.kern.Bias(1),
           'neg_quad': GPy.kern.Poly(1, order = 2),
           'sinc_compressed': GPy.kern.RBF(1)}

sample_participant = data.iloc[18]
function_name = sample_participant['function_name']
function = sample_participant['function']
response = sample_participant['response']
response_up_to_max = response[:np.argmax(response) + 1]

model_parameters = load_model_params()
kern = kernels[function_name]
kern_param_names = kern.parameter_names()
kern_params = model_parameters[function_name]['kernel_posterior_means']
for i in range(len(kern_params)):
    set_kernel_parameter(kern, kern_param_names[i], kern_params[i])
noise_variance = model_parameters[function_name]['noise_posterior_mean']
scaled_function = [f / model_parameters[function_name]['scale']  for f in function]

'''
agent = AgentModel(CoolingUCB, PropToUtility, explore = 1., explore_cooling = 1.1)
agent.acquisition_function.set_kernel(kern)
agent.acquisition_function.set_noise_variance(noise_variance)

agent.acquisition_function.set_priors(explore = positive, explore_cooling = positive)
#agent.decision_function.set_priors(temperature = positive, cooling = positive)
result = agent.optimize(scaled_function, response_up_to_max)
'''