import pandas as pd
import numpy as np
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.kde import gaussian_kde

from single_ucb import ucb_likelihood
from single_mes import mes_likelihood



def get_assignments(samples_df, X, fmax, goals):
    
    assignment_columns = [c for c in samples_df.columns if 'assignments' in c]
    nparticipants = len(assignment_columns)
    all_assignments = []
    for i in range(nparticipants):
        assignment = samples_df['assignments__'+str(i)].value_counts().index[0]
        all_assignments.append(assignment)
    
    unique_clusters = [str(x) for x in list(set(all_assignments))]
    cluster_columns = [c for c in samples_df.columns if c.split('__')[-1] in unique_clusters and 'beta' not in c and 'assignments' not in c]
    
    actions = X[0,:,:,:].argmax(axis=1)
    nassignments = []
    for i in range(len(unique_clusters)):
        cluster = unique_clusters[i]
        participants_idx = [j for j in range(len(all_assignments)) if all_assignments[j] == int(cluster)]
        #cluster_actions = actions[:,np.array(participants_idx)]
        #splot = sns.pairplot(samples_df[['explore_param__'+cluster, 'temperature__'+cluster]], diag_kind="kde")
        #for i, j in zip(*np.triu_indices_from(splot.axes, 1)):
        #    splot.axes[i, j].clear()
        
        fig, axes = plt.subplots(2, 3, figsize=(8,5))
        nfassignments = []
        for l in range(3):
            ngassignments = []
            for m in range(2):
                f = [79, 41, 32][l]
                g = ['max_score_last', 'find_max_last'][m]
                subplot_idx = [k for k in participants_idx if fmax[k] == f and goals[k] == g]
                
                ngassignments.append(len(subplot_idx))
                if len(subplot_idx) > 0:
                    subplot_actions = actions[:, np.array(subplot_idx)]
                    axes[m][l].plot(subplot_actions)
                axes[m][l].axhline(f, ls='--', c='black')
                axes[m][l].set_ylim([0, 82])
                axes[m][l].set_xlim([0, 25])
            nfassignments.append(ngassignments)
        nassignments.append(nfassignments)
            
    
    return all_assignments, samples_df[cluster_columns], nassignments, fig#, splot


def cluster_distributions(nassignments):
    
    cluster_frequencies = nassignments.sum(axis=1).sum(axis=1)
    cluster_probabilities = cluster_frequencies / cluster_frequencies.sum()
    
    


def plot_multi(samples_df, X):
    
    actions = X[0,:,:,:].argmax(axis=1).T 
    
    ntrials = X.shape[1]
    trials = np.arange(ntrials)
    clusters = [(samples_df[c].value_counts().index[0]) for c in samples_df.columns if 'assignments__' in c]
    unique_clusters = list(set(clusters))
    for cluster in unique_clusters:
        
        participant_indices = [i for i in range(len(clusters)) if clusters[i] == cluster]
        print (len(participant_indices))
        
        columns = [c for c in samples_df.columns if '__'+str(cluster) in c]
        df = samples_df[columns]
        df.columns = [c.replace('__'+str(cluster),'') for c in df.columns]
        
        position = trials[:,None] - df['x_midpoint'][None,:]
        growth = np.exp(-df['steepness'][None,:] * position)
        denom = 1 + growth * df['yscale'][None,:]
        explore_param = 1 - (1. / denom)
        mean = explore_param.mean(axis=1)
        std = explore_param.std(axis=1)
        upper = mean + 2*std
        lower = mean - 2*std
        
        fig, axes = plt.subplots(1, 2, figsize=(8,5))
        
        axes[0].plot(trials, mean)
        axes[0].fill_between(trials, upper, lower, alpha=.5)
        axes[0].plot(explore_param)
        axes[0].set_ylim([0, 1])
        axes[1].plot(actions[participant_indices].T)
        axes[1].set_ylim([0, 82])
        plt.show()
        
    


def plot_sample_explore(steepness_params=np.array([0, 100, .5, 100]), x_midpoint_params=np.array([10, 25, 10, 10]),
                        yscale_params=np.array([0, 1, 1, 1]), titles=['Pure Exploit', 'Pure Explore', 'Soft Phases', 'Hard Phases'], ntrials=25,
                        figsize=(20, 4), fontsize=14, rows=1):
    
    trials = np.arange(ntrials)
    explore = (1 - 1. / (1 + yscale_params[:,None] * np.exp(-steepness_params[:,None] * (trials - x_midpoint_params[:,None]))))
    fig, axes = plt.subplots(rows, int(len(steepness_params)/rows), sharex=True, sharey=True, figsize=figsize)
    axes = axes.ravel()
    for i in range(len(axes)):
        axes[i].plot(explore[i])
        axes[i].set_title(titles[i], fontsize=fontsize)
        #axes[i].text(14, .6, '$\\theta_{1}=' + str(yscale_params[i]) + '$ \n$\\theta_{2}=' + str(steepness_params[i]) + '$ \n$\\theta_{3}=' + str(x_midpoint_params[i]) + '$',
            #fontsize=fontsize)
        axes[i].tick_params(labelsize=fontsize)
        if i == 0:
            axes[i].set_ylabel('$\\beta$', fontsize=fontsize)
        axes[i].set_xlabel('t', fontsize=fontsize)
        axes[i].set_ylim([0, 1])
        
    return fig
    


def sample_explore(n=20, ntrials=25, steepness_alpha=1., steepness_beta=1., x_midpoint_prec=4., yscale_alpha=1., yscale_beta=.5):
    
    trials = np.arange(ntrials)
    steepness_dist = pm.Gamma.dist(steepness_alpha, steepness_beta)
    BoundNormal = pm.Bound(pm.Normal, lower=0, upper=ntrials)
    x_midpoint_dist = BoundNormal.dist(mu=ntrials/2., sd=ntrials/x_midpoint_prec)
    yscale_dist = pm.Gamma.dist(yscale_alpha, yscale_beta)
    
    steepness = steepness_dist.random(size=n)
    x_midpoint = x_midpoint_dist.random(size=n)
    yscale = yscale_dist.random(size=n)
    
    position = trials[:,None] - x_midpoint[None,:]
    growth = np.exp(-steepness * position)
    denom = 1 + growth * yscale
    explore_param = 1 - (1. / denom)
    
    x = np.linspace(0,30)
    steepness_pdf = np.exp(steepness_dist.logp(x).eval())
    x_midpoint_pdf = np.exp(x_midpoint_dist.logp(x).eval())
    yscale_pdf = np.exp(yscale_dist.logp(x).eval())
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,5))
    ax1.plot(x, steepness_pdf)
    ax1.plot(x, x_midpoint_pdf)
    ax1.plot(x, yscale_pdf)
    ax2.plot(explore_param)
    
    
def plot_explore(df, actions, fmax):
    
    ntrials = len(actions)
    trials = np.arange(ntrials)
        
    position = trials[:,None] - df['x_midpoint__0'][None,:]
    growth = np.exp(-df['steepness__0'][None,:] * position)
    denom = 1 + growth * df['yscale__0'][None,:]
    explore_param = 1 - (1. / denom)
    mean = explore_param.mean(axis=1)
    std = explore_param.std(axis=1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5))
    ax1.plot(mean);
    ax1.fill_between(np.arange(ntrials), mean - 2*std, mean + 2*std, alpha=.5)
    ax2.plot(actions)
    ax2.axhline(fmax, ls='--', c='black')
    ax1.set_ylim([0, 1])
    ax2.set_ylim([0, 82])
    plt.show()
    
    
def plot_explore_participants(samples_df):
    
    for name, df in samples_df.groupby('participant'):
        actions = np.array(df['actions'][0])
        ntrials = len(actions)
        trials = np.arange(ntrials)
        
        position = trials[:,None] - df['x_midpoint__0'][None,:]
        growth = np.exp(-df['steepness__0'][None,:] * position)
        denom = 1 + growth * df['yscale__0'][None,:]
        explore_param = 1 - (1. / denom)
        mean = explore_param.mean(axis=1)
        std = explore_param.std(axis=1)
        
        fmax = df['fmax'][0]
        
        color='blue'
        if 'explore_method' in df.columns:
            e = df['explore_method'].mean()
            if e > .5:
                color='red'
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13,5))
        ax1.plot(mean, color=color);
        ax1.fill_between(np.arange(ntrials), mean - 2*std, mean + 2*std, alpha=.5, color=color)
        ax2.plot(actions)
        ax2.axhline(fmax, ls='--', c='black')
        ax1.set_ylim([0, 1])
        ax2.set_ylim([0, 80])
        plt.show()
        
        
def assign_ucb_clusters(trace, X):
    
    trials = np.arange(X.shape[1])
    mean_weights = trace['w'].mean(axis=0)
    samples = np.array([[s['steepness'], s['x_midpoint'], s['temperature']] for s in np.array(trace)])#[:,:,idx]
    args = samples.mean(axis=0).T
    
    all_likelihoods = []
    for i in range(X.shape[3]):
        participant = X[:,:,:,i]
        actions = participant[0]
        mean = participant[1]
        var = participant[2]
        
        cluster_likelihoods = []
        for arg in args:
            steepness, x_midpoint, temperature = arg
            l = ucb_likelihood(actions, mean, var, trials, np.array([steepness]), np.array([x_midpoint]), np.array([temperature]))
            cluster_likelihoods.append(l)
        all_likelihoods.append(cluster_likelihoods)
    cluster_assignments = np.array(all_likelihoods + np.log(mean_weights[None,:])).argmax(axis=1)
    return cluster_assignments


def assign_mes_clusters(trace, X):
    
    trials = np.arange(X.shape[1])
    mean_weights = trace['w'].mean(axis=0)
    samples = np.array([[s['steepness'], s['x_midpoint'], s['temperature']] for s in np.array(trace)])#[:,:,idx]
    args = samples.mean(axis=0).T
    
    all_likelihoods = []
    for i in range(X.shape[3]):
        participant = X[:,:,:,i]
        actions = participant[0]
        mean = participant[1]
        mes = participant[3]
        
        cluster_likelihoods = []
        for arg in args:
            steepness, x_midpoint, yscale, temperature = arg
            l = mes_likelihood(actions, mean, mes, trials, np.array([steepness]), np.array([x_midpoint]),
                           np.array([yscale]), np.array([temperature]))
            cluster_likelihoods.append(l)
        all_likelihoods.append(cluster_likelihoods)
    cluster_assignments = np.array(all_likelihoods + np.log(mean_weights[None,:])).argmax(axis=1)
    return cluster_assignments


        

def assign_ucb_mes_clusters(trace, threshold, X):
    
    trials = np.arange(X.shape[1])
    mean_weights = trace['w'].mean(axis=0)
    idx = np.argwhere(mean_weights>=threshold).ravel()
    samples = np.array([[s['steepness'], s['x_midpoint'], s['yscale'], s['temperature'], s['explore_method']] for s in np.array(trace)])#[:,:,idx]
    args = samples.mean(axis=0).T
    
    all_likelihoods = []
    for i in range(X.shape[3]):
        participant = X[:,:,:,i]
        actions = participant[0]
        mean = participant[1]
        var = participant[2]
        mes = participant[3]
        
        cluster_likelihoods = []
        for arg in args:
            steepness, x_midpoint, yscale, temperature, explore_method = arg
            l = likelihood(actions, mean, var, mes, trials, np.array([steepness]), np.array([x_midpoint]),
                           np.array([yscale]), np.array([temperature]), np.array([explore_method]))
            cluster_likelihoods.append(l)
        all_likelihoods.append(cluster_likelihoods)
    cluster_assignments = np.array(all_likelihoods).argmax(axis=1)
    return cluster_assignments


        
def get_explore_clusters(trace, threshold, ntrials):
    
    mean_weights = trace['w'].mean(axis=0)
    idx = np.argwhere(mean_weights>=threshold).ravel()
    samples = np.array([[s['steepness'], s['x_midpoint'], s['temperature']] for s in np.array(trace)])#[:,:,idx]
    
    trials = np.arange(ntrials)
    position = trials[:,None,None] - samples[:,1,:]
    growth = np.exp(-samples[:,0,:] * position)
    denom = 1 + growth
    e = 1 - 1. / denom

    mean_e = e.mean(axis=1)
    std_e = e.std(axis=1)
    return mean_e, std_e, samples


def plot_explore_clusters(trace, threshold, ntrials):
    
    mean_e, std_e, samples = get_explore_clusters(trace, threshold, ntrials)
    mean_e = mean_e.T
    std_e = std_e.T
    colors = np.array(sns.color_palette("hls", mean_e.shape[0]).as_hex())
    upper = (mean_e + 2 * std_e).clip(min=0, max=1)
    lower = (mean_e - 2 * std_e).clip(min=0, max=1)
    for i in range(len(mean_e)):
        plt.plot(mean_e[i], color=colors[i])
        plt.fill_between(np.arange(len(mean_e[i])), lower[i], upper[i], color=colors[i], alpha=.5)
    
    
def plot_cluster_assignments(trace, cluster_f, X, fmax, param_range=25, figsize=(13,5)):
    
    actions = X[0].argmax(axis=1).T
    cluster_assignments = cluster_f(trace, X)
    mean_e, std_e, samples = get_explore_clusters(trace, actions.shape[1], actions.shape[1])
        
    #posterior_densities = [[gaussian_kde(samples[:,j,i]) for j in range(samples.shape[1])] for i in range(samples.shape[2])]
    posterior_means = np.array([[np.mean(samples[:,j,i]) for j in range(samples.shape[1])] for i in range(samples.shape[2])])
    posterior_std = np.array([[np.std(samples[:,j,i]) for j in range(samples.shape[1])] for i in range(samples.shape[2])])
    #x = np.linspace(0, param_range)
    
    populated_clusters = list(set(cluster_assignments))
    fs = np.sort(list(set(fmax)))[::-1]
    fig, axes = plt.subplots(len(populated_clusters), 4, figsize=figsize)
    #posterior_labels = ['steepness', 'x_midpoint', 'yscale', 'temperature']
    all_clustered_actions = []
    for i in range(len(populated_clusters)):
        
        axes[i][0].set_ylabel("$\mathbf{C="+str(populated_clusters[i])+"}$", rotation='horizontal', labelpad=20)
        k = populated_clusters[i]
        #dens = posterior_densities[k]
        """
        for d in range(len(dens)):
            pdf = dens[d].pdf(x)
            axes[i][0].plot(x, pdf, label=posterior_labels[d])
        """
        m = mean_e[:,k]
        s = std_e[:,k]
        axes[i][0].plot(m)
        axes[i][0].fill_between(np.arange(len(m)), m - 2*s, m + 2*s, alpha=.5)
        axes[i][0].set_ylim([0, 1])
        cluster_indices = np.argwhere(cluster_assignments==k).ravel()
        
        cluster_actions = []
        for j in range(3):
            idx = [x for x in cluster_indices if fmax[x] == fs[j]]
            a = actions[idx]
            cluster_actions.append(len(a))
            if len(a) > 0:
                axes[i][j+1].plot(a.T)
            axes[i][j+1].axhline(fs[j], ls='--', c='black')
            axes[i][j+1].set_ylim([0, 84])
        all_clustered_actions.append(cluster_actions)
        
    #axes[0][0].legend(loc="upper right")
    #axes[0][0].set_title("$p(\\theta|D)$")
    axes[0][0].set_title("$p(\\beta|D)$")
    axes[0][1].set_title("Linear")
    axes[0][2].set_title("Quadratic")
    axes[0][3].set_title("Sinc")
    return fig, all_clustered_actions, posterior_means[populated_clusters], posterior_std[populated_clusters]


def plot_cluster_assignments_goals(trace, cluster_f, X, fmax, goals, param_range=25, figsize=(13,5)):
    
    fm_X = np.array([X[:,:,:,i] for i in range(X.shape[3]) if goals[i] == 'find_max_last']).transpose(1,2,3,0)
    fm_fmax = np.array([fmax[i] for i in range(len(fmax)) if goals[i] == 'find_max_last'])
    ms_X = np.array([X[:,:,:,i] for i in range(X.shape[3]) if goals[i] == 'max_score_last']).transpose(1,2,3,0)
    ms_fmax = np.array([fmax[i] for i in range(len(fmax)) if goals[i] == 'max_score_last'])
    
    fm_fig, fm_actions, fm_posterior_means, fm_posterior_std = plot_cluster_assignments(trace, cluster_f, fm_X, fm_fmax, param_range, figsize)
    ms_fig, ms_actions, ms_posterior_means, ms_posterior_std = plot_cluster_assignments(trace, cluster_f, ms_X, ms_fmax, param_range, figsize)
    
    return fm_fig, ms_fig, np.array(fm_actions), np.array(ms_actions), fm_posterior_means, fm_posterior_std, ms_posterior_means, ms_posterior_std
    


def plot_components(df, figsize=(8,6)):
    
    values = (df.mean()[[c for c in df.columns if 'w' in c]]).values
    k = len(values)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(k), values)
    ax.set_xlabel('Component');
    ax.set_ylabel('Posterior expected mixture weight');
    return fig
        
