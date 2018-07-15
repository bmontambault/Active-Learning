from flask import Flask, render_template, request
import json
import numpy as np


app=Flask(__name__)
app.secret_key=''
app.config['SESSION_TYPE']='filesystem'

with open('all_model_results.json', 'r') as f:
    model_results = json.load(f)
    

@app.route('/')
def index():
    
    ids = [model_results[i]['id'] for i in range(len(model_results))]
    functions = [model_results[i]['function_name'] for i in range(len(model_results))]
    goals = [model_results[i]['goal'] for i in range(len(model_results))]
    scores = [np.round(model_results[i]['score'], 2) for i in range(len(model_results))]
    max_scores = [np.round(model_results[i]['max_score'], 2) for i in range(len(model_results))]
    
    for d in model_results:
        d['models'] = [m for m in d['models'] if m['acquisition'] != 'Phase']
    
    best_model = [max([m for m in d['models']], key = lambda x: x['pseudo_r2']) for d in model_results]
    model_name = []
    for m in best_model:
        if 'kernel' in m.keys():
            model_name.append(m['acquisition'] + '(' + m['kernel'] + ')')
        else:
            model_name.append(m['acquisition'])
    pseudo_r2 = [m['pseudo_r2'] for m in best_model]
    params = [np.round(m['acquisition_params'] + m['decision_params'], 2).tolist() for m in best_model]
    return render_template('index.html', ids = ids, functions = functions, goals = goals, scores = scores, max_scores = max_scores, model_name = model_name, params = params, pseudo_r2 = pseudo_r2)


@app.route('/<participant_id>')
def participant(participant_id):
    
    data = request.args
    if len(data) == 0:
        participant_plots = [d for d in model_results if d['id'] == participant_id][0]['plots']
        ntrials = len(participant_plots)
        return render_template('participant.html', participant_id = participant_id, ntrials = ntrials)
    else:
        participant_plots = [d for d in model_results if d['id'] == participant_id][0]['plots']
        return json.dumps(participant_plots[int(data['trial'])])


if __name__=="__main__":
    app.run()
    