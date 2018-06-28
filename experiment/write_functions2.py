import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import json


def sinc(loc, scale, height, x):
        
    return (np.sin(scale * x - (loc * scale + .000001))/(scale * x - (loc * scale + .000001))) * height


def lin(intercept, slope, x):
    
    return intercept + slope * x


def neg_quad(loc, scale, height, x):
    
    center = len(x)/2
    return - (scale * (x - loc - center)**2) + height
    

def squash(y, upper_bound):
    
    miny = max(0, min(y))
    maxy = min(upper_bound, max(y))
    return [(maxy-miny)/(max(y)-min(y))*(yi-max(y))+maxy for yi in y]
    

def generate_sinc(n, scale_mean, scale_var, height_var, upper_bound, x):
    
    locs = np.random.randint(0, 79, size = n)
    scales = np.random.lognormal(scale_mean, scale_var, size = n)
    heights = upper_bound - np.abs(np.random.normal(0, height_var, size = n)) * upper_bound
    
    y = [squash(sinc(locs[i], scales[i], heights[i], x), upper_bound) for i in range(n)]
    return y


def generate_lin(n, intercept_mean, intercept_var, slope_mean, slope_var, upper_bound, x):
    
    intercepts = np.random.lognormal(intercept_mean, intercept_var, size = n)
    slopes = np.random.lognormal(slope_mean, slope_var, size = n)
    y = [squash(lin(intercepts[i], slopes[i], x), upper_bound) for i in range(n)]
    return y


def sample_lin(mean, cov, x, upper_bound, m, n, plot = False):
    
    intercept, slope = st.multivariate_normal(mean, cov).rvs(m).T
    samples = {}
    for i in range(n):
        y = squash(slope[i]*x + intercept[i], upper_bound)
        samples[i] = y
        if plot:
            plt.plot(y)
    return samples
    
    
def sample_negquad_alt(mean, cov, x, upper_bound, m, n, plot = False):
    
    scale, loc, height = st.multivariate_normal(mean, cov).rvs(m).T
    a = scale
    b = scale*2*-loc
    c = scale*loc**2+height
    params = np.array([a, b, c]).T
    mean = np.mean(params, axis = 0)
    cov = np.cov(params, rowvar = False)
    samples = {}
    for i in range(n):
        y = squash(a[i]*x**2 + b[i]*x + c[i], upper_bound)
        samples[i] = y
        if plot:
            plt.plot(y)
            
    return mean, cov, samples


def sample_negquad(mean, cov, x, upper_bound, n, plot = False):
    a, b, c = st.multivariate_normal(mean, cov).rvs(n).T
    y = {i:squash(a[i]*x**2 + b[i]*x + c[i], upper_bound) for i in range(n)}
    if plot:
        for val in y.values():
            plt.plot(val)
    return y



    
def get_linear_samples():
    upper_bound = 500
    x = np.arange(80)
    
    mean = [upper_bound/2, 0]
    cov = [[16000, -400],
           [0, .01]]
    linear_samples = sample_lin(mean, cov, x, upper_bound, 100, 20, plot = True)
    
    with open('linear_samples.json', 'w') as f:
        json.dump(linear_samples, f)
    np.savetxt('linear_sample_params.csv', np.vstack((mean, cov)))
    

def get_negquad_samples():
    upper_bound = 500
    x = np.arange(80)
    
    negquad_alt_mean = [-.2, len(x)/2, upper_bound - 50]
    negquad_alt_cov = [[.00001, 0, 0],
           [0, 240, 0],
           [0, 0, 50]]
    negquad_mean, negquad_cov, negquad_samples = sample_negquad_alt(negquad_alt_mean, negquad_alt_cov, x, upper_bound, 1000, 20, True)
    with open('negquad_samples.json', 'w') as f:
        json.dump(negquad_samples, f)
    np.savetxt('negquad_sample_params.csv', np.vstack((negquad_mean, negquad_cov)))

'''
mean = [40, 500, 0]
cov = [[.000001, 0, 0],
       [0, .00001, 0],
       [0, 0, .000000001]]
sample_negquad_loc(mean, cov, x, 500, 1)


#print (get_vertex((-.3126, 25.008)))

mean = [-.3126, 25.008, 0]
cov = [[.0000000001, 0, 0],
       [0, .00000001, 0],
       [0, 0, .0000001]]
sample_negquad(mean, cov, x, upper_bound, 50)
'''
'''
mean = [upper_bound/2, 0]
cov = [[16000, -400],
       [0, .01]]
sample_lin(mean, cov, x, upper_bound, 100)
'''

'''
y1 = neg_quad(0, -1, 500, x)
y2 = neg_quad(10, 1, 500, x)
y3 = neg_quad(-20, -10, 500, x)
y4 = neg_quad(40, 20, 500, x)

y1s = squash(y1, upper_bound)
y2s = squash(y2, upper_bound)
y3s = squash(y3, upper_bound)
y4s = squash(y4, upper_bound)
'''