from flask import Flask, render_template
import json
import numpy as np
import seaborn as sns

'''
app = Flask(__name__)
data_path = r'E:\Projects\Git\Public\Active-Learning-New\tasks\tmp.json'


@app.route("/")
def index():
    with open(data_path, 'r') as f:
        data = json.load(f)
    return render_template('index.html', **data)
'''

def visualize_map(path):
    
    app = Flask(__name__)
    
    @app.route("/")
    def index():
        with open(path, 'r') as f:
            data = json.load(f)
        return render_template('index.html', **data)
    
    app.run()


'''
def run(name, acquisition, decision, acq_params, dec_params, function, total_score, total_probability, current_actions, all_mean, all_var, all_utility, all_probability, ntrials, goal, kernel_name):
    
    c = sns.color_palette("hls", len(acquisition)).as_hex()
    data = {'id': name, 'acquisition': acquisition, 'decision': decision, 'acq_params': acq_params, 'dec_params': dec_params, 'function': function, 'score': [x.item() for x in total_score], 'actions': current_actions, 'mean': all_mean, 'var': all_var, 'utility': all_utility, 'probability': all_probability, 'total_probability': total_probability, 'ntrials': ntrials, 'goal': goal, 'kernel': kernel_name, 'colors': c}
    with open(data_path, 'w') as f:
        json.dump(data, f)        
    app.run()
'''