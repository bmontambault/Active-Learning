

if __name__ == '__main__':
    
    import pymc3 as pm
    from theano import tensor as tt
    from acquisitions import likelihood, mixture_likelihood, ucb
    
    def stick_breaking(beta):
        portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
        return beta * portion_remaining

    def single_ucb(X, nsamples = 100):

        with pm.Model() as model:
                explore = pm.Gamma('explore', 1., 1.)
                temperature = pm.Gamma('temperature', 1., 1.)        
                obs = pm.Potential('obs', likelihood(X, ucb, {'explore': explore, 'temperature': temperature}))
                
        with model:
            trace = pm.sample(nsamples)
        return trace
    
    
    def mix_ucb(X, nsamples = 100, K = 30):
        
        with pm.Model() as model:
                alpha = pm.Gamma('alpha', 1., 1.)
                beta = pm.Beta('beta', 1., alpha, shape = K)
                w = pm.Deterministic('w', stick_breaking(beta))
            
                explore = pm.Gamma('explore', 1., 1., shape = K)
                temperature = pm.Gamma('temperature', 1., 1., shape = K)        
                obs = pm.Potential('obs', mixture_likelihood(X, explore, temperature, w, K))
                
        with model:
            trace = pm.sample(nsamples)
        return trace