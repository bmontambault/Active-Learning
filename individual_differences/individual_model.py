if __name__ == '__main__':


    import pandas as pd
    import pymc3 as pm
    from theano import tensor as tt
    import numpy as np
    import GPy
    
    from data.get_results import get_results
    from likelihood import get_kernel
    from vectorized import vectorize
    from vectorized import participant_ucb_likelihood


    
    results = get_results('data/results.json').iloc[3:]
    function_names = results['function_name'].unique()
    kernel_dict = {f: get_kernel(results, GPy.kern.RBF(1), f) for f in function_names}
    functions_dict = results[['function_name', 'function']].drop_duplicates(subset = ['function_name']).set_index('function_name').to_dict()['function']
    normalized_functions_dict = {f: (np.array(functions_dict[f]) - np.mean(functions_dict[f])) / np.std(functions_dict[f]) for f in function_names}
    
    #participant = results.iloc[0]
    #x = vectorize(participant, kernel_dict, normalized_functions_dict)

    '''
    with pm.Model() as single_model:
        explore = pm.Beta('explore', 1., 1.)
        temperature = pm.Gamma('temperature', 1., 1.)        
        obs = pm.Potential('obs', participant_ucb_likelihood(x, explore, temperature))
        
    with single_model:
        trace = pm.sample(100)
    '''