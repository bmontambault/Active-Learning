#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
from flask import Flask,render_template,request
import importlib
import sys
import os
import json
from task_files import task_files

app=Flask(__name__)
app.secret_key='key'
app.config['SESSION_TYPE']='filesystem'
config_file='config'
config=importlib.import_module(config_file)
function_names=config.functions
tasks=config.tasks
data_path='data.json'

path=os.path.dirname(os.path.realpath(__file__))
data_path=path+'/'+data_path
with open(path+'/'+'{0}_{1}_functions.json'.format(config.experiment,config.version)) as json_data:
    functions=json.load(json_data)
    json_data.close()
    
@app.route('/',methods=['GET','POST'])
def start():
    
    data=request.get_json()
    if data!=None:
        with open(data_path) as json_data:
            all_data=json.load(json_data)
            json_data.close()
        all_data.append(data)
        with open(data_path,'w') as json_data:
            json.dump(all_data,json_data)

    if "location" in request.args:
        return render_template(request.args["location"])
        
    else:
        fi=int(request.args.get('fi'))
        ti=int(request.args.get('ti'))
        print (fi,file=sys.stderr)
        function_name=function_names[fi]
        goals=tasks[ti]
        function=functions[function_name]
        task=['start.html']+[a for b in [task_files[t] for t in goals] for a in b]+['last_page.html']
        
        all_args={
                 'function':function,
                 'task':task,
                 'experiment':config.experiment,
                 'version':config.version,
                 'function_name':function_name,
                 'goals':goals,
                 'bar_height':config.bar_height,
                 'bar_width':config.bar_width,
                 'nbars':config.nbars,
                 'trials':config.trials,
                 'predict_trials':config.predict_trials,
                 'se_length':config.se_length,
                 'sinc_offset':config.sinc_offset,
                 'neg_quad_offset':config.neg_quad_offset,
                 'pos_quad_offset':config.pos_quad_offset
                 }
        
        return render_template('index.html',**all_args)
        
if __name__=="__main__":
    app.run()    
