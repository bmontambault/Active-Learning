import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_discrete
from utils import scale
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
    
    
def choose_next(observed_x, observed_y, domain, acq, acq_params):
    
    vals, prob = acq(observed_x, observed_y, domain, *acq_params)
    next_x = rv_discrete(values = (domain, prob)).rvs()
    return next_x, prob, vals


def plot(function, upper_bound ,observed_x, observed_y, next_x, next_y, prob, trials, running_regret, max_regret, mean = None, std = None, utility = None, pymax = None):
    
    subplots = 3
    if not utility is None:
        subplots += 1
    if not pymax is None:
        subplots += 1
    
    rows = np.ceil(subplots / 4)
    cols = np.ceil(subplots / rows)
    fig = plt.figure(figsize = (4 * cols, 4 * rows))

    main_ax = fig.add_subplot(rows, cols, 1)
    domain = range(len(function))
    
    plot_obs, = main_ax.plot(observed_x, observed_y, 'bo')
    plot_next, = main_ax.plot(next_x, next_y, 'ro', label = 'Next')
    plot_truth, = main_ax.plot(domain, function, color = 'g')
        
    prob_ax = fig.add_subplot(rows, cols, 2)
    plot_prob, = prob_ax.plot(domain, prob.tolist(), color = 'm')
    
    handles = [plot_obs, plot_next, plot_truth, plot_prob]
    labels = ['Observed', 'Next', 'Ground Truth', 'p(a)']
    
    if not mean is None:
        plot_mean, = main_ax.plot(domain, mean.tolist(), color = 'r')
        plot_var = main_ax.fill_between(domain, (mean - 2 * std).ravel().tolist(), (mean + 2 * std).ravel().tolist(), color = 'blue', alpha = .5)
        handles.insert(2, plot_mean)
        handles.insert(3, plot_var)
        labels.insert(2, 'Mean')
        labels.insert(3, 'Confidence Bound')
    
    if not utility is None:
        utility_ax = fig.add_subplot(rows, cols, 3)
        utility_plot, = utility_ax.plot(domain, utility.ravel().tolist(), color = 'teal')
        handles.append(utility_plot)
        labels.append('Utility')
    
    if not pymax is None:
        pymax_ax = fig.add_subplot(rows, cols, 4)
        pymax_plot, = pymax_ax.plot(pymax.tolist(), np.linspace(0, upper_bound), color = 'orange')
        handles.append(pymax_plot)
        labels.append('p(y = y*)')
        
    if len([x for x in running_regret if x !=None]) > 0:
        regret_ax = fig.add_subplot(rows, cols, subplots)
        r = running_regret + list(np.full(trials - len(running_regret), None))
        regret_plot, = regret_ax.plot(range(1, trials + 1), r, 'go')
        regret_ax.set_xlim(1, trials)
        regret_ax.set_ylim(0, max_regret)
        handles.append(regret_plot)
        labels.append('Regret')
    
    main_ax.set_ylim(0, upper_bound)
    fig.subplots_adjust(bottom = 0.3, wspace = 0.33)
    if subplots > 4:
        xanchor = 3.2
    else:
        xanchor = .8
    main_ax.legend(handles = handles, labels = labels, loc = 'upper center', bbox_to_anchor = (xanchor, -.2), ncol = 3)
    return fig

    
def sim(task, function, trials, upper_bound, acq, acq_params):
    
    ymax = max(function)
    if task == 'max_score':
        max_regret = trials
    elif task == 'find_max':
        max_regret = 1
    elif task == 'min_error':
        max_regret = None
    score = 0
    regret = 0
    running_regret = []
    trials_remaining = trials
    
    domain = function.index.tolist()
    observed_x = []
    observed_y = []
    for i in range(trials):
        trials_remaining -= 1 
        next_x, prob, vals = choose_next(observed_x, observed_y, domain, acq, list(acq_params) + [trials_remaining])
        next_y = function[next_x]
        
        score, regret = get_score(task, score, regret, next_y, ymax)
        running_regret.append(regret)
        
        fig = plot(function, upper_bound, observed_x, observed_y, next_x, next_y, prob, trials, running_regret, max_regret, **vals)
        observed_x.append(next_x)
        observed_y.append(next_y)
        plt.show()
        
        
def get_score(task, score, regret, y, ymax):
    
    if task == 'max_score':
        new_score = y
        total_score = score + new_score
        new_regret = (ymax - y) / ymax
        total_regret = regret + new_regret
        
    elif task == 'find_max':
        new_score = max(0, y - score)
        total_score = max(score, y)
        new_regret = (ymax - total_score) / ymax
        total_regret = new_regret
        
    elif task == 'min_error':
        new_score = None
        total_score = None
        new_regret = None
        total_regret = None
    
    return total_score, total_regret
        
        
        
    
    