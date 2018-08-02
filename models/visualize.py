from flask import Flask, render_template, request
import json
import numpy as np


with open('test_plot_data.json', 'r') as f:
    all_plot_data = json.load(f)
    all_plot_data = {d['id']: d for d in [all_plot_data]}


app=Flask(__name__)

@app.route('/<ID>')
def simulation(ID):
    
    plot_data = all_plot_data[ID]
    data = request.args
    if len(data) == 0:
        ntrials = plot_data['ntrials']
        acq_params = [np.round(a, 2).tolist() for a in plot_data['acq_params']]
        dec_params = [np.round(a, 2).tolist() for a in plot_data['dec_params']]
        return render_template('simulation.html', ntrials = ntrials, acquisition = plot_data['acquisition'],
                               decision = plot_data['decision'], acq_params = acq_params, dec_params = dec_params,
                               max_score = np.max(plot_data['function']) * ntrials, ID = ID)
    else:
        return json.dumps(plot_data['trial_data'][data['trial']])

if __name__=="__main__":
    app.run()
