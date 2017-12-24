import numpy as np
import scipy.stats as st

def z_score(y, mean, std):
    if type(y) == list:
        y = np.array(y)[None,:]
    return (y - mean)/std    


def ymax_cdf(y, mean, std):
        return np.prod(st.norm.cdf(z_score(y, mean, std)), axis = 0)


def inv_ymax_cdf(p, mean, std, precision, upper_bound):
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


def fit_gumbel(mean, std, precision, upper_bound):
    r1 = .25
    r2 = .75
    y1 = inv_ymax_cdf(r1, mean, std, precision, upper_bound)
    y2 = inv_ymax_cdf(r2, mean, std, precision, upper_bound)
    R = np.array([[1., np.log(-np.log(r1))], [1., np.log(-np.log(r2))]])
    Y = np.array([y1, y2])
    a, b = np.linalg.solve(R, Y)
    return a, b


def sample_gumbel(a, b, nsamples):
    return [a - (b * np.log(-np.log(r))) for r in np.random.uniform(0, 1, nsamples)]


def gumbel_pdf(a, b, y):
    z = (y - a) / b
    return 1 - ((1 / b) * np.exp(-(z + np.exp(-z))))


def maxval_entropy(mean, std, ymax_samples):
    z_scores = z_score(ymax_samples, mean, std)
    log_norm_cdf_z_scores = st.norm.logcdf(z_scores)
    log_norm_pdf_z_scores = st.norm.logpdf(z_scores)        
    return np.mean(z_scores * np.exp(log_norm_pdf_z_scores - (np.log(2) + log_norm_cdf_z_scores)) - log_norm_cdf_z_scores, axis = 1)


def mes(mean, std, precision, upper_bound, nsamples):
    a, b = fit_gumbel(mean, std, precision, upper_bound)
    ymax_samples = sample_gumbel(a, b, nsamples)
    utility = maxval_entropy(mean, std, ymax_samples)
    return (a, b), utility