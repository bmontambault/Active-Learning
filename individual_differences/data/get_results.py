import pandas as pd


def get_results(path):
    
    data = pd.read_json(path)
    results = pd.DataFrame([x for y in data['results'] for x in y])
    all_data = pd.concat([data, results], axis = 1)
    return all_data