import json
import pandas as pd

from data.get_results import get_results
from fit_responses import fit_all_participants


with open('good_results.json', 'r') as f:
    good_results = pd.read_json(json.load(f))
results = get_results('data/results.json').iloc[1:]

a = results[results['function_name'] == 'neg_quad'].iloc[3:6]

#lin_opt = good_results[(good_results['function_name'] == 'pos_linear') & (good_results['goal'] == 'find_max_last')]
#a = lin_opt[lin_opt['somataSessionId'] == 'GMGgVOhx4mAGGG4QXgCFDnygkI5tmqAQ']
#a = lin_rl.iloc[-1:]
#a = results.loc[[24, 34, 52, 54, 66, 69]]
new_data = fit_all_participants(a, method = 'DE')#method = 'L-BFGS-B')