import sys
sys.path.append('./agent')

import scipy.stats as st
import GPy

from agent.acquisitions import random, sgd, decaying_sgd, ucb
from agent.decisions import propto, softmax, random_stay_softmax, random_switch_softmax
from agent.agent import Agent


random_agent = Agent(random, propto, [], [])

lr_init = 1.
lr_prior = st.gamma(5, scale = 200).logpdf
lr_bounds = (1., 3000)

temp_init = 1.
temp_prior = st.gamma(1.5, scale = 3).logpdf
temp_bounds = (.01, 10)

sgd_agent = Agent(sgd, softmax, [lr_init], [temp_init], acq_priors = [lr_prior], dec_priors = [temp_prior], acq_bounds = [lr_bounds], dec_bounds = [temp_bounds])

decay_init = .9
decay_prior = st.beta(1, 5).logpdf
decay_bounds = (.000, .999)

decaying_sgd_agent = Agent(decaying_sgd, softmax, [lr_init, decay_init], [temp_init], acq_priors = [lr_prior, decay_prior], dec_priors = [temp_prior], acq_bounds = [lr_bounds, decay_bounds], dec_bounds = [temp_bounds])

sr_init = .1
sr_prior = st.beta(1, 5).logpdf
sr_bounds = (.001, .999)

random_stay_sgd_agent = Agent(sgd, random_stay_softmax, [lr_init], [temp_init, sr_init], acq_priors = [lr_prior], dec_priors = [temp_prior, sr_prior], acq_bounds = [lr_bounds], dec_bounds = [temp_bounds, sr_bounds])
random_switch_sgd_agent = Agent(sgd, random_switch_softmax, [lr_init], [temp_init, sr_init], acq_priors = [lr_prior], dec_priors = [temp_prior, sr_prior], acq_bounds = [lr_bounds], dec_bounds = [temp_bounds, sr_bounds])
random_stay_decaying_sgd_agent = Agent(decaying_sgd, random_stay_softmax, [lr_init, decay_init], [temp_init, sr_init], acq_priors = [lr_prior, decay_prior], dec_priors = [temp_prior, sr_prior], acq_bounds = [lr_bounds, decay_bounds], dec_bounds = [temp_bounds, sr_bounds])
random_switch_decaying_sgd_agent = Agent(decaying_sgd, random_switch_softmax, [lr_init, decay_init], [temp_init, sr_init], acq_priors = [lr_prior, decay_prior], dec_priors = [temp_prior, sr_prior], acq_bounds = [lr_bounds, decay_bounds], dec_bounds = [temp_bounds, sr_bounds])


explore_init = 1.
explore_prior = st.gamma(1, scale = 1).logpdf
explore_bounds = (0, 1000)

ucb_agent = Agent(ucb, softmax, [explore_init], [temp_init], acq_priors = [explore_prior], dec_priors = [temp_prior], acq_bounds = [explore_bounds], dec_bounds = [temp_bounds])


agents = [sgd_agent,
          decaying_sgd_agent,
          random_stay_sgd_agent,
          random_switch_sgd_agent,
          random_stay_decaying_sgd_agent,
          random_switch_decaying_sgd_agent]

agents = [sgd_agent, decaying_sgd_agent, random_stay_sgd_agent, random_stay_decaying_sgd_agent]


rbf_kern = GPy.kern.RBF(1)
lin_kern = GPy.kern.Linear(1)
quad_kern = GPy.kern.Poly(1, order = 2)
per_kern = GPy.kern.StdPeriodic(1)

kernels = [rbf_kern,
           lin_kern,
           quad_kern]#,
           #per_kern]
           
kernels = [rbf_kern]


variance_prior = GPy.priors.Gamma(.1, .1)
lengthscale_prior = GPy.priors.Gamma(.1, .1)
scale_prior = GPy.priors.Gamma(.1, .1)
bias_prior = GPy.priors.Gamma(.1, .1)
period_prior = GPy.priors.Gamma(.1, .1)

all_kernel_priors = {'rbf': [variance_prior, lengthscale_prior],
                     'linear': [variance_prior],
                     'poly': [variance_prior, scale_prior, bias_prior],
                     'std_periodic': [variance_prior, period_prior, lengthscale_prior]}