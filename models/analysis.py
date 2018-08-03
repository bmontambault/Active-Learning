import pandas as pd

def get_best_aic(data):
    return data[data.groupby(['id'], sort = False)['AIC'].transform(min) == data['AIC']]

def get_strategy(data, col, val):
    
    data = data[data[col] == val].reset_index()
    params = pd.DataFrame(data[col + '_params'].values.tolist(), columns = [col + '_p' + str(i) for i in range(len(data[col + '_params'].iloc[0]))])
    return pd.concat([data, params], axis = 1)