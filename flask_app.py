#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys,os
from flask import Flask,render_template,request,session
import json
import string
import random
import uuid
from config import nbars,max_height,bar_width,trials,predict_trials,version,se_length

path=os.path.dirname(os.path.realpath(__file__))
goals=['max_score','find_max','min_error']
functions=['pos_linear','neg_linear','pos_power','neg_power','pos_quad','neg_quad','sin','sinc','se']

funcmap={f:''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10)) for f in functions}
revfuncmap={value:key for key,value in funcmap.iteritems()}

app=Flask(__name__)

def find_max_score(goal,function,trials,predict_trials):
    if goal=='find_max':
        return max(function)
    elif goal=='max_score':
        return max(function)*trials
    elif goal=='min_error':
        return max(function)*(predict_trials)

@app.route('/',methods=['GET','POST'])
def start():
    if request.method=='GET':
        if 'id' not in session:
            session['id']=str(uuid.uuid4())
        return render_template('start.html')
    elif request.method=='POST':
        index=request.form['index']
        if index=='0':
            with open(path+'/'+'conditions.json') as json_data:
                conditions=json.load(json_data)
                json_data.close()
            prev_func=conditions['function_index']
            prev_goal=conditions['goal_index']
            if len(functions)-prev_func==1:
                if len(goals)-prev_goal==1:
                    curr_goal=0
                else:
                    curr_goal=prev_goal+1
                curr_function=0
            else:
                curr_goal=prev_goal
                curr_function=prev_goal+1
            goal=goals[curr_goal]
            function_name=functions[curr_function]
            conditions={'function_index':curr_function,'goal_index':curr_goal}
            with open('conditions.json', 'w') as json_data:
                json.dump(conditions,json_data)
            return task(goal,function_name,index)
        else:
            goal=request.form['goal']
            function_tag=request.form['function_name']
            if function_tag in functions:
                function_name=function_tag
            else:
                function_name=revfuncmap[function_tag]
            return task(goal,function_name,index)


@app.route('/admin',methods=['GET','POST'])
def start_admin():
    if request.method=='GET':
        return render_template('start_admin.html',function_names=functions)
    elif request.method=='POST':
        goal=request.form['goal']
        function_tag=request.form['function_name']
        if function_tag in functions:
            function_name=function_tag
        else:
            function_name=revfuncmap[function_tag]
        index=request.form['index']
        return task('test',goal,function_name,index)

def task(goal,function_name,index):
    with open(path+'/'+'functions.json') as json_data:
        function=json.load(json_data)[function_name]
        json_data.close()

    if index=='exit':
        participant={}
        for key in request.form.keys():
            participant[key]=request.form[key]
        participant['function_name']=revfuncmap[participant['function_name']]
        participant['id']=session['id']
        participant['function']=function
        participant['max_score']=int(max(function))
        participant['max_height']=max_height
        participant['bar_width']=bar_width
        participant['number_of_bars']=nbars
        participant['plot_width']=bar_width*nbars
        participant['experiment']='bmontambault/Active-Learning'
        participant['max_total_score']=int(find_max_score(goal,function,trials,predict_trials))
        participant['version']=version
        participant['se_function_lengthscale']=se_length
        print (participant,file=sys.stderr)
        return render_template('exit_survey.html',**participant)
        
    elif goal=='max_score':
        if index=='0':
            return render_template('max_score_instructions.html',trials=trials,goal=goal,function_name=funcmap[function_name])
        elif index=='1':
            return render_template('max_score.html',nbars=nbars,goal=goal,function_name=funcmap[function_name],function=function,trials=trials,bar_height=max_height,bar_width=bar_width)

    elif goal=='find_max':
        if index=='0':
            return render_template('find_max_instructions.html',goal=goal,function_name=funcmap[function_name])
        elif index=='1':
            return render_template('find_max.html',nbars=nbars,goal=goal,function_name=funcmap[function_name],function=function,trials=trials,bar_height=max_height,bar_width=bar_width)

    elif goal=='min_error':
        if index=='0':
            return render_template('min_error_instructions.html',goal=goal,function_name=funcmap[function_name])
        elif index=='1':
            return render_template('min_error_phase1.html',nbars=nbars,goal=goal,function_name=funcmap[function_name],function=function,trials=trials,bar_height=max_height,bar_width=bar_width)
        elif index=='2':
            phase1_response=request.form['test_response']
            test_start_time=request.form['test_start_time']
            test_response_time=request.form['test_response_time']
            return render_template('min_error_phase2.html',nbars=nbars,goal=goal,function_name=funcmap[function_name],function=function,trials=predict_trials,bar_height=max_height,bar_width=bar_width,phase1_response=phase1_response,test_start_time=test_start_time,test_response_time=test_response_time)

if __name__=="__main__":
    app.secret_key='key'
    app.config['SESSION_TYPE']='filesystem'
    app.run()
