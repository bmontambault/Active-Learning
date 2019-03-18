import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt


def z_score(y, mean, std):
    
    return (y - mean[:,None]) / std[:,None]
    if type(y) == list:
        y = np.array(y)[None,:]
    mean = mean.reshape(len(mean), 1)
    std = std.reshape(len(std), 1)
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
    return np.array([a - (b * np.log(-np.log(r))) for r in np.random.uniform(0, 1, int(nsamples))])


def gumbel_pdf(a, b, y):
    z = (y - a) / b
    return - ((1 / b) * np.exp(-(z + np.exp(-z))))


def get_mes_utility(mean, var, samples=100, precision=.001, return_gumbel=False):
    std = np.sqrt(var)
    a, b = fit_gumbel(mean, std, precision)
    ymax_samples = sample_gumbel(a, b, samples)
    
    #plt.hist(ymax_samples, bins=50)
    #x = np.linspace(np.min(ymax_samples), np.max(ymax_samples))
    #plt.plot(x, [gumbel_pdf(a, b, y)*50000 for y in x])
    #plt.show()
    
    #z_scores = z_score(ymax_samples, mean, std)
    z = (ymax_samples[:,None] - mean) / std
    return np.mean(z * st.norm.pdf(z) / (2. * st.norm.cdf(z)) - st.norm.logcdf(z), axis=0)

    
    """
    log_norm_cdf_z = st.norm.logcdf(z)
    log_norm_pdf_z = st.norm.logpdf(z)
    
    info = z * np.exp(log_norm_pdf_z - np.log(2) + log_norm_cdf_z)
    entropy = log_norm_cdf_z
    return np.mean(info - entropy, axis=1), info.mean(axis=1), entropy.mean(axis=1)
    

    mes_utility = np.mean(z * np.exp(log_norm_pdf_z - (np.log(2) + log_norm_cdf_z)) - log_norm_cdf_z, axis = 1)
    if return_gumbel:
        return mes_utility, a, b
    else:
        return mes_utility
    """
    
def gpflowopt_mes(mean, var, samples=100, precision=.001):
    std = np.sqrt(var)
    a, b = fit_gumbel(mean, std, precision)
    ymax_samples = sample_gumbel(a, b, samples)
    
    gamma = (ymax_samples[:,None] - mean) / np.sqrt(var)
    print (gamma)
    return np.mean(gamma * st.norm.pdf(gamma) / (2. * st.norm.cdf(gamma)) - st.norm.logcdf(gamma), axis=0)
    
    
    