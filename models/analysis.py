import pandas as pd

def get_best_aic(data):
    return data[data.groupby(['id'], sort = False)['AIC'].transform(min) == data['AIC']]


def get_strategy(data, col, val):
    
    data = data[data[col] == val].reset_index()
    params = pd.DataFrame(data[col + '_params'].values.tolist(), columns = [col + '_p' + str(i) for i in range(len(data[col + '_params'].iloc[0]))])
    return pd.concat([data, params], axis = 1)


def count_best_by_condition(data, condition = ['f', 'goal']):
    
    df = get_best_aic(data)
    df = df[['f', 'goal', 'acq']]
    for acq in data['acq'].unique():
        df[acq] = df.apply(lambda x: 1 if x['acq'] == acq else 0, axis = 1)
    return df.groupby(condition).sum()