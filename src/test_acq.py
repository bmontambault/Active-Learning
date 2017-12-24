import pandas as pd

from sim import sim
from utils import fit_kern, soft_max

from acquisition_functions import mes_acq, sgd_acq, var_greedy_acq, mean_greedy_acq, weighted_mes_mean_greedy, mes_mean_greedy_last


functions = pd.read_json('active-learning_0.8.0_functions.json')
kern = fit_kern(functions)
pos_linear = functions['pos_linear']
neg_quad = functions['neg_quad']
sinc_compressed = functions['sinc_compressed']


upper_bound = 650

'''
acq = mes_acq
acq_params = (kern, .001, upper_bound, 1000, soft_max, (.01,)) #kern, precision, upper_bound, nsamples, decision, decision_params
sim('max_score', neg_quad, 2, upper_bound, acq, acq_params)
'''

'''
acq = sgd_acq
acq_params = (1., 1., 1., .001)
sim('min_error', pos_linear, 7, upper_bound, acq, acq_params)
'''

'''
acq = var_greedy_acq
acq_params = (kern, soft_max, (.1,)) #kern, decision, decision_params
sim('max_score', pos_linear, 10, upper_bound, acq, acq_params)
'''

'''
acq = weighted_mes_mean_greedy
acq_params = (kern, .001, upper_bound, 1000, 1000, soft_max, (.1,))
sim('max_score', neg_quad, 10, upper_bound, acq, acq_params)
'''


acq = mes_mean_greedy_last
acq_params = (kern, .001, upper_bound, 1000, soft_max, (.1,))
sim('find_max', neg_quad, 5, upper_bound, acq, acq_params)