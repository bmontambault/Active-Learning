import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt
import GPy


def z_score(y, mean, std):
    
    if type(y) == list:
        y = np.array(y)[None,:]
    return (y - mean)/std  


def ymax_cdf(y, mean, std):
        return np.prod(st.norm.cdf(z_score(y, mean, std)), axis = 0)



def inv_ymax_cdf(p, mean, std, precision):
    
    upper_bound = precision
    while True:
        if ymax_cdf(upper_bound, mean, std) >= p:
            break
        else:
            upper_bound *= 2
    
    Z = np.arange(0, upper_bound + precision, precision)
    while len(Z) > 2:
        bisect = len(Z) // 2
        z = Z[bisect]
        prob = ymax_cdf(z, mean, std)
        if prob > p:
            Z = Z[:bisect + 1]
        else:
            Z = Z[bisect:]
    return np.mean(Z)
    

def fit_gumbel(mean, std, precision):
    r1 = .25
    r2 = .75
    y1 = inv_ymax_cdf(r1, mean, std, precision)
    y2 = inv_ymax_cdf(r2, mean, std, precision)
    if y1 == y2:
        y1 -= precision
    R = np.array([[1., np.log(-np.log(r1))], [1., np.log(-np.log(r2))]])
    Y = np.array([y1, y2])
    a, b = np.linalg.solve(R, Y)
    return a, b


def sample_gumbel(a, b, nsamples):
    return [a - (b * np.log(-np.log(r))) for r in np.random.uniform(0, 1, int(nsamples))]


def gumbel_pdf(a, b, y):
    z = (y - a) / b
    return - ((1 / b) * np.exp(-(z + np.exp(-z))))


def maxval_entropy(mean, std, ymax_samples):
    z_scores = z_score(ymax_samples, mean, std)
    log_norm_cdf_z_scores = st.norm.logcdf(z_scores)
    log_norm_pdf_z_scores = st.norm.logpdf(z_scores)        
    return np.mean(z_scores * np.exp(log_norm_pdf_z_scores - (np.log(2) + log_norm_cdf_z_scores)) - log_norm_cdf_z_scores, axis = 1)


X = np.array([1, 2, 3, 4, 8, 9])
y = np.array([4, 7, 8, 5, 1, 3])
m = GPy.models.GPRegression(X = X[:,None], Y = y[:,None], kernel = GPy.kern.RBF(1))
#m.optimize()

newX = np.arange(1, 10)[:,None]
mean, var = m.predict(newX)
a, b = fit_gumbel(mean, np.sqrt(var), .01)
y_samples = sample_gumbel(a, b, 100)

plt.plot(maxval_entropy(mean, np.sqrt(var), y_samples))
plt.plot(mean, var, 'bo')
