import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

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


def sample_lin(mean, cov, x):
    
    intercept, slope = st.multivariate_normal(mean, cov).rvs()
    y = intercept + slope * x
    plt.plot(x, y)


upper_bound = 500
x = np.arange(80)
sinc_samples = generate_sinc(15, -1.18, .5, .1, upper_bound, x)
lin_samples = generate_lin(10, 0, 5, 0, 500, upper_bound, x)

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