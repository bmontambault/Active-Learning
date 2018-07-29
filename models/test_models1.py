import numpy as np
import json

from acquisitions import SGD1, SGD2
from decisions import Softmax
from run import run, add_single_plot
from data.get_results import get_results

results = get_results('data/results.json').iloc[3:]
pos_linear = results[results['function_name'] == 'pos_linear'].iloc[0]['function']
neg_quad = results[results['function_name'] == 'neg_quad'].iloc[0]['function']
sinc_compressed = results[results['function_name'] == 'sinc_compressed'].iloc[0]['function']

pos_linear_n = [(f - np.mean(pos_linear)) / np.std(pos_linear) for f in pos_linear]
neg_quad_n = [(f - np.mean(neg_quad)) / np.std(neg_quad) for f in neg_quad]
sinc_compressed_n = [(f - np.mean(sinc_compressed)) / np.std(sinc_compressed) for f in sinc_compressed]

data = run(neg_quad_n, SGD1, Softmax, [40.], [.01], 25)
plot_data = add_single_plot(data)
with open('test_plot_data.json', 'w') as f:
    json.dump(plot_data, f)
print (data['id'])