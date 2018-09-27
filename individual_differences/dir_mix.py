import pandas as pd
import pymc3 as pm
from theano import tensor as tt


if __name__ == '__main__':

    old_faithful_df = pd.read_csv(pm.get_data('old_faithful.csv'))
    old_faithful_df['std_waiting'] = (old_faithful_df.waiting - old_faithful_df.waiting.mean()) / old_faithful_df.waiting.std()
    
    
    N = old_faithful_df.shape[0]
    K = 30
    
    def stick_breaking(beta):
        portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
        return beta * portion_remaining
    
    
    with pm.Model() as model:
        alpha = pm.Gamma('alpha', 1., 1.)
        beta = pm.Beta('beta', 1., alpha, shape=K)
        w = pm.Deterministic('w', stick_breaking(beta))
    
        tau = pm.Gamma('tau', 1., 1., shape=K)
        lambda_ = pm.Uniform('lambda', 0, 5, shape=K)
        mu = pm.Normal('mu', 0, tau=lambda_ * tau, shape=K)
        obs = pm.NormalMixture('obs', w, mu, tau=lambda_ * tau,
                               observed=old_faithful_df.std_waiting.values)
    
    
    with model:
        trace = pm.sample(100)