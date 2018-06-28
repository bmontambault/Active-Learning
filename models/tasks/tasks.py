import numpy as np

def task(function, agents, scoref, ntrials = None, actions = None, optimize_joint = False, optimize_by_trial = False, method = 'L-BFGS-B', fixed_acq_idx = [], fixed_dec_idx = [], uniform_priors = False):
    
    if optimize_joint and optimize_by_trial:
        raise ValueError('Can optimize either by trial likelihood or joint likelihood, not both')
    if ntrials == None:
        ntrials = len(actions)
    
    all_mean = []
    all_var = []
    current_actions = []
    all_utility = []
    all_probability = []
    total_score = []
    total_probability = []
    
    acquisition = []
    decision = []
    all_acq_params = []
    all_dec_params = []
    all_utility = []
    all_probability = []
    for agent in agents:
        print (agent.acquisition, agent.decision)
        if uniform_priors:
            agent.acq_priors = [lambda x: 0 for n in range(len(agent.acq_params))]
            agent.dec_priors = [lambda x: 0 for m in range(len(agent.dec_params))]
        agent_acq_params = []
        agent_dec_params = []
        agent_utility = []
        agent_probability = []
        
        if actions != None and optimize_joint and len(agent.acq_params + agent.dec_params) > 0:
            agent.MAP(function, actions, fixed_acq_idx = fixed_acq_idx, fixed_dec_idx = fixed_dec_idx, method = method)
        
        for i in range(ntrials):
            
            acq_params = agent.acq_params
            if i > 1 and actions != None and optimize_by_trial and len(agent.acq_params + agent.dec_params) > 0:
                agent.MAP(function, [actions[i]], fixed_acq_idx = fixed_acq_idx, fixed_dec_idx = fixed_dec_idx, method = method)
            
            acq_other = {'trial': i}
            mean, var, utility, probability, action = agent.action(function, acq_other, (lambda i: None if actions == None else actions[i])(i))
            all_mean.append(mean.ravel().tolist())
            all_var.append(var.ravel().tolist())
            agent_utility.append(utility.tolist())
            agent_probability.append(np.log(probability).tolist())
            current_actions.append(action)
            total_score.append(scoref(function[current_actions]))
            agent.observed_x.append(action)
            agent.observed_y.append(function[action])
            
            agent_acq_params.append(agent.acq_params)
            agent_dec_params.append(agent.dec_params)
            agent.acq_params = acq_params
        
        total_probability.append(np.sum([agent_probability[i][current_actions[i]] for i in range(len(agent_probability))]))
        all_acq_params.append(agent_acq_params)
        all_dec_params.append(agent_dec_params)
        acquisition.append(agent.acquisition.__name__)
        decision.append(agent.decision.__name__)
        all_utility.append(agent_utility)
        all_probability.append(agent_probability)
        
    return acquisition, decision, all_acq_params, all_dec_params, function.ravel().tolist(), total_score, total_probability, current_actions, all_mean, all_var, all_utility, all_probability


def max_score(function, agent, ntrials = None, actions = None, optimize_joint = False, optimize_by_trial = False, method = 'L-BFGS-B', fixed_acq_idx = [], fixed_dec_idx = [], uniform_priors = False):
    
    return task(function, agent, np.sum, ntrials = ntrials, actions = actions, optimize_joint = optimize_joint, optimize_by_trial = optimize_by_trial, method = method, fixed_acq_idx = fixed_acq_idx, fixed_dec_idx = fixed_dec_idx, uniform_priors = uniform_priors)


def find_max(function, agent, ntrials = None, actions = None, optimize_joint = False, optimize_by_trial = False, method = 'L-BFGS-B', fixed_acq_idx = [], fixed_dec_idx = [], uniform_priors = False):
    
    return task(function, agent, np.max, ntrials = ntrials, actions = actions, optimize_joint = optimize_joint, optimize_by_trial = optimize_by_trial, method = method, fixed_acq_idx = fixed_acq_idx, fixed_dec_idx = fixed_dec_idx, uniform_priors = uniform_priors)