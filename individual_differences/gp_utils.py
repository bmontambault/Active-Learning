import GPy
import numpy as np
import scipy.stats as st


def get_mean_var(kernel, actions, rewards, choices):
    
    if len(actions) == 0:
        return np.ones(len(choices)), np.ones(len(choices))
    else:
        X = np.array(actions)[:,None]
        Y = np.array(rewards)[:,None]
        m = GPy.models.GPRegression(X, Y, kernel)
        m.Gaussian_noise.variance = .00001
        mean, var = m.predict(np.array(choices)[:,None])
        return mean.ravel(), var.ravel()
    
    
def get_all_means_vars(kernel, actions, rewards, choices):
    
    all_means = []
    all_vars = []
    for i in range(len(actions)):
        a = actions[:i]
        r = rewards[:i]
        mean, var = get_mean_var(kernel, a, r, choices)
        all_means.append(mean.ravel())
        all_vars.append(var.ravel())
    return np.array(all_means), np.array(all_vars)

def get_function_samples(kernel, observed_x, observed_y, all_x, size):
    
    X = np.array(observed_x)[:,None]
    Y = np.array(observed_y)[:,None]
    m = GPy.models.GPRegression(X, Y, kernel)
    m.Gaussian_noise.variance = .00001
    f = m.posterior_samples_f(np.array(all_x)[:,None], full_cov = True, size = size)
    return f
    

def z_score(y, mean, std):
    
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
    return [a - (b * np.log(-np.log(r))) for r in np.random.uniform(0, 1, int(nsamples))]


def gumbel_pdf(a, b, y):
    z = (y - a) / b
    return - ((1 / b) * np.exp(-(z + np.exp(-z))))


def get_mes_utility(mean, var, samples=100):
    std = np.sqrt(var)
    a, b = fit_gumbel(mean, std, .01)
    ymax_samples = sample_gumbel(a, b, samples)
    z_scores = z_score(ymax_samples, mean, std)
    log_norm_cdf_z_scores = st.norm.logcdf(z_scores)
    log_norm_pdf_z_scores = st.norm.logpdf(z_scores)
    return np.mean(z_scores * np.exp(log_norm_pdf_z_scores - (np.log(2) + log_norm_cdf_z_scores)) - log_norm_cdf_z_scores, axis = 1)
