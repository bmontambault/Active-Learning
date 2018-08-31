import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import GPy
import io
import csv
import itertools


clean_acq_names = {'LocalMove': 'Local-Move',
                   'SGD': 'SGD',
                   'SGDMax': 'SGD-Max',
                   'Explore': 'Explore',
                   'Exploit': 'Exploit',
                   'SoftExplorePhase': '$\\pi$-First',
                   'EI': 'EI',
                   'UCB': 'UCB',
                   'MES': 'MES',
                   'MCRS': 'MCRS',
                   'MSRS': 'MSRS'}

clean_dec_names = {'Softmax': 'Softmax',
                   'DecaySoftmax': 'Decay-Softmax',
                   'SoftMaxPhaseSoftmax': '$\\pi$-Greedy-Softmax',
                   'SoftRandomFirstSoftmax': '$\\pi$-First-Softmax',
                   'SoftRandomPhaseSoftmax': '$\\pi$-Last-Softmax',
                   'SoftMaxPhaseSoftmax': '$\\pi$-Greedy-Softmax',
                   'RepeatSoftmax': 'Repeat-Softmax'}


clean_goal_names = {'find_max_last': 'Find-Max',
                    'max_score_last': 'Max-Score'}

clean_f_names = {'neg_quad': 'Quadratic',
                 'pos_linear': 'Linear',
                 'sinc_compressed': 'Sinc'}

#f = {'neg_quad': [249.78174128119795, 262.46706602724856, 274.7496323145263, 286.6294401430313, 298.10648951276335, 309.1807804237226, 319.852312875909, 330.1210868693225, 339.98710240396315, 349.45035947983104, 358.510858096926, 367.1685982552481, 375.42357995479745, 383.27580319557387, 390.72526797757746, 397.77197430080827, 404.4159221652661, 410.6571115709512, 416.49554251786344, 421.93121500600284, 426.9641290353693, 431.594284605963, 435.8216817177838, 439.64632037083186, 443.06820056510696, 446.08732230060923, 448.70368557733866, 450.9172903952953, 452.728136754479, 454.13622465488993, 455.141554096528, 455.7441250793932, 455.94393760348555, 455.7409916688051, 455.13528727535174, 454.1268244231256, 452.7156031121266, 450.9016233423547, 448.68488511380997, 446.0653884264924, 443.043133280402, 439.61811967553876, 435.7903476119027, 431.55981708949366, 426.9265281083118, 421.8904806683572, 416.4516747696297, 410.6101104121294, 404.3657875958562, 397.71870632081016, 390.66886658699127, 383.21626839439944, 375.3609117430349, 367.1027966328975, 358.44192306398725, 349.3782910363042, 339.91190054984816, 330.0427516046193, 319.7708442006177, 309.0961783378432, 298.01875401629593, 286.5385712359757, 274.65562999688257, 262.3699302990167, 249.681472142378, 236.59025552696644, 223.09628045278205, 209.1995469198246, 194.9000549280945, 180.1978044775916, 165.09279556831575, 149.58502820026706, 133.67450237344553, 117.36121808785106, 100.64517534348397, 83.52637414034393, 66.00481447843094, 48.08049635774523, 29.753419778286798, 11.023584740055298], 'sinc_compressed': [63.76406179770396, 67.14604248872388, 75.03271999076935, 83.58655640007436, 88.42458599902938, 86.84249672512425, 79.2966072568953, 69.34608346964353, 62.000310876593005, 61.22654380611118, 67.84365610047183, 78.90211102086357, 88.9586114728188, 92.71394759045455, 87.7549887313902, 76.00681830890312, 63.063322964418376, 55.58491011776596, 57.94674420315613, 69.78845189741872, 85.78595053534508, 97.94691560681724, 99.48232440552766, 88.43225645271485, 69.18713708142116, 50.92960442177122, 43.46565971743746, 52.255468510156845, 75.03258382074773, 101.87203353071305, 119.0933400922475, 115.58658062620088, 88.84906395872946, 47.864707877697754, 11.1098002482305, 0, 30.149196818007226, 103.97431007857824, 207.88270124765356, 315.5704046497534, 396.5108722377619, 426.54370806289273, 396.5110333436044, 315.5706779891399, 207.88300670596155, 103.9745646372699, 30.149342616164006, 2.156540853093247e-05, 11.109725002711116, 47.86459087337636, 88.84896395162872, 115.58653753555183, 119.09336256786844, 101.87210054088018, 75.03265809737383, 52.25551498298563, 43.465660633249, 50.92956505742046, 69.18708059952144, 88.43221131979726, 99.48231074619741, 97.946936748918, 85.78599309973225, 69.78849361436238, 57.94676517413376, 55.58490200717222, 63.06329200197234, 76.00678118015486, 87.75496382730199, 92.71394608791724, 88.9586324854721, 78.90214284765835, 67.843682573291, 61.22655237522156, 62.000298451732874, 69.3460573475694, 79.29658097763564, 86.84248314405352, 88.42459107491175, 83.58657666486306], 'pos_linear': [41.97759512957924, 47.096746440851746, 52.21589775212419, 57.335049063396696, 62.4542003746692, 67.5733516859417, 72.6925029972142, 77.81165430848665, 82.93080561975916, 88.0499569310316, 93.1691082423041, 98.28825955357661, 103.40741086484911, 108.52656217612162, 113.64571348739406, 118.76486479866657, 123.88401610993907, 129.00316742121151, 134.12231873248402, 139.24147004375652, 144.36062135502902, 149.47977266630153, 154.59892397757397, 159.71807528884648, 164.83722660011892, 169.95637791139143, 175.07552922266393, 180.19468053393643, 185.31383184520894, 190.4329831564814, 195.5521344677539, 200.6712857790264, 205.79043709029887, 210.90958840157134, 216.02873971284384, 221.14789102411632, 226.26704233538882, 231.3861936466613, 236.5053449579338, 241.62449626920628, 246.74364758047878, 251.86279889175125, 256.98195020302376, 262.10110151429626, 267.22025282556876, 272.3394041368412, 277.4585554481137, 282.57770675938616, 287.69685807065866, 292.81600938193117, 297.93516069320367, 303.0543120044761, 308.1734633157486, 313.2926146270211, 318.4117659382936, 323.5309172495661, 328.6500685608386, 333.7692198721111, 338.8883711833836, 344.00752249465603, 349.12667380592853, 354.24582511720104, 359.36497642847354, 364.484127739746, 369.6032790510185, 374.722430362291, 379.84158167356344, 384.96073298483594, 390.07988429610845, 395.19903560738095, 400.3181869186534, 405.4373382299259, 410.5564895411984, 415.6756408524709, 420.79479216374335, 425.91394347501586, 431.03309478628836, 436.15224609756086, 441.2713974088333, 446.3905487201058]}
#f = {clean_f_names[key]: f[key] for key in f}


def add_explore(results):
    
    results['max_response'] = results['function'].apply(lambda x: np.argmax(x))
    results['max_trial'] = results.apply(lambda x: x['response'].index(x['max_response']) if x['max_response'] in x['response'] else len(x['response']), axis = 1)
    results['prop_reward'] = results.apply(lambda x: [x['function'][x['response'][i]] / max(x['function']) for i in range(len(x['response']))], axis = 1)
    results['reward'] = results.apply(lambda x: [x['function'][x['response'][i]] for i in range(len(x['response']))], axis = 1)
    results['distance'] = results.apply(lambda x: [abs(i - x['max_response']) for i in x['response']], axis = 1)
    return results


def average_max_trial(results, groups):
    
    return results.groupby(groups).mean()['max_trial']


def explore(results):
    
    max_trials = average_max_trial(results, ['function_name'])
    fig, axes = plt.subplots(1, 3, figsize = (17, 5))
    for i in range(3):
        function_name = ['pos_linear', 'neg_quad', 'sinc_compressed'][i]
        ax = axes[i]
        j = 0
        for goal in results['goal'].unique():
            condition = results[(results['function_name'] == function_name) & (results['goal'] == goal)]
            y = np.array([np.array(x) for x in condition['distance'].values])
            X = np.array([np.arange(len(yi)) for yi in y])
            y = y.ravel()[:,None]
            X = X.ravel()[:,None]
            m = GPy.models.GPRegression(X, y, GPy.kern.RBF(1))
            m.optimize()
            x_ = np.arange(25)[:,None]
            mean, var = m.predict(x_)
            std = np.sqrt(var)
            
            max_trial = max_trials[function_name]
            ax.axvline(x = max_trial, color = 'black')
            ax.plot(X.ravel() + 1, y.ravel(), ['ro', 'bo'][j], alpha = .1)
            ax.plot(x_.ravel() + 1, mean, color = ['red', 'blue'][j])
            ax.fill_between(x_.ravel() + 1, (mean - 2 * std).ravel(), (mean + 2 * std).ravel(), color = ['red', 'blue'][j], alpha = .2)
            ax.set_xlabel('Trial Number')
            ax.set_ylabel("Absolute Distance From Max")
            ax.set_ylim(bottom = 0)
            ax.set_title(clean_f_names[function_name])
            j += 1
    
    lines = [Line2D([0], [0], color = "red", lw=3), Line2D([0], [0], color="blue", lw=3), Line2D([0], [0], color="black", lw=3)]
    ax.legend(lines, ['Max-Score', 'Find-Max', 'Best Found'], bbox_to_anchor=(-2.01, 1))
    return fig



def participant_max(data):
    
    return data.groupby('id').agg({'f': lambda x: tuple(x)[0], 'goal': lambda x: tuple(x)[0], 'pseudo_r2': max})


def participant_avg(data, groups = []):
    
    if len(groups) > 0:
        return data.groupby(groups).mean()['pseudo_r2']
    else:
        return data.mean()['pseudo_r2']


def participant_avg_max(data, groups = []):
    
    p = participant_max(data)
    if len(groups) > 0:
        return p.groupby(groups).mean()
    else:
        return p.mean()
    
    
def max_strats(data):
    
    p = participant_max(data)
    return pd.merge(data[['id', 'pseudo_r2', 'acq', 'dec']], p, on = ['id', 'pseudo_r2'], how = 'right')
        

def count_max_strats(data):
    
    ms = max_strats(data)
    ms['count'] = 1
    return ms.groupby(['acq', 'dec']).sum()['count']
    

def load_data(path):
    
    data = pd.read_csv(path)
    data = data[data['acq'].isin(clean_acq_names)]
    data = data[data['dec'].isin(clean_dec_names)]
    
    data['acq'] = data['acq'].map(clean_acq_names)
    data['dec'] = data['dec'].map(clean_dec_names)
    data['strategy'] = data.apply(lambda x: (x['acq'], x['dec']), axis = 1)
    data['goal'] = data['goal'].map(clean_goal_names)
    data['f'] = data['f'].map(clean_f_names)
    return data


def get_pivot(data):
    
    conditions = data[['id', 'f', 'goal']].drop_duplicates()
    pivot = data.pivot(index ='id', columns = 'strategy', values = 'pseudo_r2')
    r2_data = pd.merge(conditions, pivot, on = 'id', how = 'left')
    return r2_data


def r2_pca(pivot, n_components):
    
    df = pivot.drop(['id', 'goal', 'f'], axis = 1)
    models = df.columns.tolist()
    X = df.values
    pca = PCA(n_components = n_components)
    pca.fit(X)
    X_ = pca.transform(X)
    components = pca.components_
    labels = pca.components_.argmax(axis = 0)
    print (labels)
    print (models)
    d = pd.DataFrame()
    d['models'] = models
    d['labels'] = labels
    return d
    return X_, pca.explained_variance_


def keep_best_r2(pivot):
    
    X = pivot.drop(['id', 'goal', 'f'], axis = 1).values
    mask = np.zeros_like(X)
    mask[np.arange(len(X)), X.argmax(1)] = 1
    df = pd.concat([pivot[['id', 'goal', 'f']], pd.DataFrame(mask)], axis = 1)
    df.columns = pivot.columns
    df = df.loc[:,(df != 0).any(axis = 0)]
    return df


def setup_decision_tree(pivot):
    
    strat = pivot.drop(['id', 'f', 'goal'], axis = 1).idxmax(1)
    new_pivot = pivot[['f', 'goal']]
    new_pivot['strat'] = strat
    s = '@RELATION "r2_test"\n\n@ATTRIBUTE f 	{Linear, Quadratic, Sinc}\n@ATTRIBUTE goal {Find-Max, Max-Score}\n'
    #variables = pivot.drop(['id', 'f', 'goal'], axis = 1).columns
    var_s = '@ATTRIBUTE strat [{}]\n'.format(', '.join(new_pivot['strat'].unique())).replace(']', '}').replace('[', '{') + '\n@DATA\n'
    #var_s = '\n'.join(['@ATTRIBUTE ' + str(col) + ' numeric' for col in variables]) + '\n\n@DATA\n'
    with open('clus/r2_part_test.arff', 'w') as file:
        file.write(s + var_s)
    #for col in pivot.columns:
        #if col not in ['id', 'f', 'goal']:
            #pivot[col] = pivot[col].apply(lambda x: int(x))
    new_pivot.to_csv('clus/data.csv', index = False, header = False)
    
    filenames = ['clus/r2_part_test.arff', 'clus/data.csv']
    with open('clus/r2_test.arff', 'w') as outfile:
        for line in itertools.chain.from_iterable(list(map(open, filenames))):
            outfile.write(line)
    
    ss = '[Attributes]\nDescriptive = 1-2\nTarget = 3\nClustering = 3\n\n[Tree]\nHeuristic = Default\nPruningMethod = M5Multi'
    #ss = '[Attributes]\nDescriptive = 1-2\nTarget = 3-{0}\nClustering = 3-{0}\n\n[Tree]\nHeuristic = Default\nPruningMethod = M5Multi'.format(len(variables) + 2)
    with open('clus/r2_test.s', 'w') as outfile:
        outfile.write(ss)
    

def plot_corr(pivot):
    
    df = pivot.drop(['id', 'goal', 'f'], axis = 1)
    corr = df.corr()
    mask = np.zeros_like(corr, dtype = np.bool)
    mask[np.triu_indices_from(mask)] = True
    cmap = sns.diverging_palette(220, 10, as_cmap = True)
    f, ax = plt.subplots(figsize = (9, 9))
    sns.heatmap(corr, mask = mask, cmap = cmap, vmax = .3, center = 0, square = True, linewidth = .5)
    return f


def clean_names(df):
    
    clean_df = df.copy()
    columns = ['id', 'goal', 'f'] + ['-'.join((col[0].replace('\\','').replace('$','').replace('-',''), col[1].replace('\\','').replace('$','').replace('-',''))) for col in clean_df.columns if col not in ['id', 'f', 'goal']]
    clean_df.columns = columns
    return clean_df



def tsne(X_):
    
    X_embedded = TSNE(n_components=2).fit_transform(X_)
    return X_embedded
    


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
    ax.set_xlabel('Acquisition/Decision Function')
    ax.set_ylabel('Maximum $R^{2}$ count')
    plt.tight_layout()
    return fig


def plot_all_counts(data, groups, n = 3):
    
    condition = groups[0]
    nconditions = len(data[condition].unique())
    c = count(data, groups)
    c['x'] = c.apply(lambda x: '\n'.join([x[col] for col in c.columns if col not in [condition, 'count']]), axis = 1)
    nth = c.groupby(condition).agg({'count': lambda x: tuple(sorted(x.tolist(), reverse = True))[n - 1]})
    top_c = c[c.apply(lambda x: x['count'] >= nth['count'][x[condition]] and x['count'] > 1, axis = 1)]
    
    fig, axes = plt.subplots(1, nconditions, figsize = (15, 5))
    for (cond, group), ax in zip(top_c.groupby(condition), axes.flatten()):
        group.plot(x = 'x', y = 'count', kind = 'bar', ax = ax, title = cond, legend = False)
        ax.set_xlabel('Acquisition/Decision Function')
        ax.set_ylabel('Maximum $R^{2}$ count')
    plt.tight_layout()
    return fig


def plot_r2(data, groups, n = 3):
    
    r = avg_r2(data, groups)
    r['x'] = r.apply(lambda x: '\n'.join([x[col] for col in r.columns if col not in ['mean', 'std']]), axis = 1)
    top_r = r.head(n).reset_index()
    
    data['x'] = data.apply(lambda x: '\n'.join([x[col] for col in r.columns if col not in ['mean', 'std', 'x']]), axis = 1)
    data = data[data['x'].isin(top_r['x'].unique())]
    sorted_order = data.groupby('x').sum().reset_index().sort_values('adjusted_pseudo_r2', ascending = False)['x'].unique()
    fig, ax = plt.subplots(1)
    p = sns.barplot(x = 'x', y = 'adjusted_pseudo_r2', data = data, estimator = np.mean, order = sorted_order, ax = ax)
    p.set_xlabel('Acquisition/Decision Function')
    p.set_ylabel('Average $R^{2}$')
    plt.xticks(rotation = 90)
    plt.tight_layout()
    return fig


def plot_all_r2(data, groups, n = 3):
    
    condition = groups[0]
    nconditions = len(data[condition].unique())
    r = avg_r2(data, groups)
    r['x'] = r.apply(lambda x: '\n'.join([x[col] for col in r.columns if col not in [condition, 'mean', 'std']]), axis = 1)
    top_r = r.groupby(condition).head(n).reset_index()
    
    data['x'] = data.apply(lambda x: '\n'.join([x[col] for col in r.columns if col not in [condition, 'mean', 'std', 'x']]), axis = 1)
    fig, axes = plt.subplots(1, nconditions, figsize = (15, 5))
    for (cond, group), ax in zip(data.groupby(condition), axes.flatten()):
        group = group[group['x'].isin(top_r[top_r[condition] == cond]['x'].unique())]
        sorted_order = group.groupby('x').sum().reset_index().sort_values('adjusted_pseudo_r2', ascending = False)['x'].unique()
        p = sns.barplot(x = 'x', y = 'adjusted_pseudo_r2', data = group, estimator = np.mean, order = sorted_order, ax = ax)
        p.set_title(cond)
        p.set_xlabel('Acquisition/Decision Function')
        p.set_ylabel('Average $R^{2}$')
        p.set_xticklabels(p.get_xticklabels(), rotation=90)
    plt.tight_layout()
    return fig
        

def anova(formula, data):
    
    model = smf.ols(formula, data).fit()
    anova = sm.stats.anova_lm(model)
    return anova