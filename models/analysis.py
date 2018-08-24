import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np


def get_best_aic(data):
    return data[data.groupby(['id'], sort = False)['AIC'].transform(min) == data['AIC']]


def get_strategy(data, col, val):
    
    data = data[data[col] == val].reset_index()
    params = pd.DataFrame(data[col + '_params'].values.tolist(), columns = [col + '_p' + str(i) for i in range(len(data[col + '_params'].iloc[0]))])
    return pd.concat([data, params], axis = 1)


def count_best_by_condition(data, condition = ['f', 'goal'], acqf = 'acq_type', field = None):
    
    df = get_best_aic(data)
    df = df[['f', 'goal', 'acq_type', 'acq', 'dec']]
    if field == None:
        for acq in data[acqf].unique():
            for dec in data['dec'].unique():
                df[(acq, dec)] = df.apply(lambda x: 1 if x[acqf] == acq and x['dec'] == dec else 0, axis = 1)
    else:
        for m in data[field].unique():
            df[m] = df.apply(lambda x: 1 if x[field] == m else 0, axis = 1)
    if condition != None:
        return df.groupby(condition).sum()
    else:
        return df


def count(data, groups):
    
    n = {'local': 3, 'localGP': 6, 'globalGP': 3}
    best = data[data.groupby(['id'], sort = False)['adjusted_pseudo_r2'].transform(max) == data['adjusted_pseudo_r2']]
    best['count'] = 1
    grouped = best.groupby(groups).sum()['count'].reset_index()
    if 'acq_type' in groups:
        grouped['avg_count'] = grouped.apply(lambda x: x['count'] / n[x['acq_type']], axis = 1)
        grouped = grouped.sort_values('avg_count', ascending = False)
    else:
        grouped = grouped.sort_values('count', ascending = False)
    return grouped


def avg_r2(data, groups):
    
    g = data.groupby(groups)
    mean = g.mean()['adjusted_pseudo_r2']
    std = g.std()['adjusted_pseudo_r2']
    d = pd.concat([mean, std], axis = 1)
    d.columns = ['mean', 'std']
    d = d.reset_index()
    d = d.sort_values('mean', ascending = False)
    return d


def plot_counts(data, groups, n = 3):
    
    c = count(data, groups)
    c['x'] = c.apply(lambda x: '\n'.join([x[col] for col in c.columns if col != 'count']), axis = 1)
    fig, ax = plt.subplots(1)
    top_c = c[c['count'] >= sorted(c['count'].tolist(), reverse = True)[n - 1]]
    top_c.plot(x = 'x', y = 'count', kind = 'bar', ax = ax, legend = False)
    return fig


def plot_all_counts(data, groups, n = 3):
    
    condition = groups[0]
    nconditions = len(data[condition].unique())
    c = count(data, groups)
    c['x'] = c.apply(lambda x: '\n'.join([x[col] for col in c.columns if col not in [condition, 'count']]), axis = 1)
    nth = c.groupby(condition).agg({'count': lambda x: tuple(sorted(x.tolist(), reverse = True))[n - 1]})
    top_c = c[c.apply(lambda x: x['count'] >= nth['count'][x[condition]], axis = 1)]
    
    fig, axes = plt.subplots(1, nconditions, figsize = (15, 5))
    for (cond, group), ax in zip(top_c.groupby(condition), axes.flatten()):
        group.plot(x = 'x', y = 'count', kind = 'bar', ax = ax, title = cond, legend = False)
    return fig


def plot_r2(data, groups, n = 3):
    
    r = avg_r2(data, groups)
    r['x'] = r.apply(lambda x: '\n'.join([x[col] for col in r.columns if col not in ['mean', 'std']]), axis = 1)
    top_r = r.head(n).reset_index()
    
    data['x'] = data.apply(lambda x: '\n'.join([x[col] for col in r.columns if col not in ['mean', 'std', 'x']]), axis = 1)
    data = data[data['x'].isin(top_r['x'].unique())]
    sorted_order = top_r['x'].unique()
    p = sns.boxplot(x = 'x', y = 'adjusted_pseudo_r2', data = data, order = sorted_order)
    plt.xticks(rotation = 90)
    return p


def plot_all_r2(data, groups, n = 3):
    
    condition = groups[0]
    nconditions = len(data[condition].unique())
    r = avg_r2(data, groups)
    r['x'] = r.apply(lambda x: '\n'.join([x[col] for col in r.columns if col not in [condition, 'mean', 'std']]), axis = 1)
    top_r = r.groupby(condition).head(n).reset_index()
    
    data['x'] = data.apply(lambda x: '\n'.join([x[col] for col in r.columns if col not in [condition, 'mean', 'std', 'x']]), axis = 1)
    #data = data[data['x'].isin(top_r['x'].unique())]
    fig, axes = plt.subplots(1, nconditions, figsize = (15, 5))
    for (cond, group), ax in zip(data.groupby(condition), axes.flatten()):
        group = group[group['x'].isin(top_r[top_r[condition] == cond]['x'].unique())]
        sorted_order = group.sort_values('adjusted_pseudo_r2')['x'].unique()
        p = sns.barplot(x = 'x', y = 'adjusted_pseudo_r2', data = group, estimator = np.mean, order = sorted_order, ax = ax)
        p.set_title(cond)
        p.set_xticklabels(p.get_xticklabels(), rotation=90)
    return top_r
        

def anova(formula, data):
    
    model = smf.ols(formula, data).fit()
    anova = sm.stats.anova_lm(model)
    return anova