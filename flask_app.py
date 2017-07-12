#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys,os
from flask import Flask,render_template,request,session,url_for
import json
import string
import random
import uuid
import ast
from config import nbars,max_height,bar_width,trials,predict_trials,version,se_length

path=os.path.dirname(os.path.realpath(__file__))
goals=['max_score','find_max','min_error']
functions=['pos_linear','neg_quad','sinc_compressed']

funcmap={f:''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10)) for f in functions}
revfuncmap={value:key for key,value in funcmap.iteritems()}

app=Flask(__name__)
app.secret_key='key'
app.config['SESSION_TYPE']='filesystem'


def find_phase2_set(phase1,function,trials):
    s=list(set(ast.literal_eval(phase1)))
    r=[]
    candidates=[i for i in xrange(len(function)) if i not in s]
    r.append(max(candidates,key=lambda x: function[x]))
    r.append(min(candidates,key=lambda x: function[x]))
    while len(r)<trials:
        candidates=[i for i in xrange(len(function)) if i not in s and i not in r]
        dist=[min([abs(ci-ri) for ri in r+s]) for ci in candidates]
        max_dist=[i for i in xrange(len(dist)) if dist[i]==max(dist)]
        r.append(candidates[max_dist[random.randint(0,len(max_dist)-1)]])
    return r

def find_max_score(goal,function,trials,predict_trials):
    if goal=='find_max':
        return max(function)
    elif goal=='max_score':
        return max(function)*trials
    elif goal=='min_error':
        return max(function)*predict_trials


@app.route('/',methods=['GET','POST'])
def start():
    if request.method=='GET':
        sessionId=request.args.get("sessionId")
        gi=request.args.get('gi')
        fi=request.args.get('fi')
        if sessionId is None:
            sessionId=str(uuid.uuid4())
        session["id"] = sessionId;
        return render_template('start.html',sessionId=sessionId,gi=gi,fi=fi)
    elif request.method=='POST':
        index=request.form['index']
        if index=='0':
            fi=int(request.args.get('fi'))
            gi=int(request.args.get('gi'))
            function_name=functions[fi]
            goal=goals[gi]
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
        participant['max_score']=float(max(function))
        participant['max_height']=max_height
        participant['bar_width']=bar_width
        participant['number_of_bars']=nbars
        participant['plot_width']=bar_width*nbars
        participant['experiment']='bmontambault/Active-Learning'
        participant['max_total_score']=float(find_max_score(goal,function,trials,predict_trials))
        participant['version']=version
        participant['se_function_lengthscale']=se_length
        
        for key in participant:
            if type(participant[key])==unicode:
                if len(participant[key])>0 and participant[key][0]=='[':
                    try:
                        participant[key]=ast.literal_eval(participant[key])
                    except:
                        raise ValueError('server error')
                elif ',' in participant[key]:
                    try:
                        participant[key]=[float(v) for v in participant[key].split(',')]
                    except:
                        print (participant[key],file=sys.stderr)
            if participant[key]=='':
                participant[key]=[]
        print (participant,file=sys.stderr)
        
        return render_template('exit_survey.html',**participant)
        
    elif index=='goal_prompt':
        return render_template('goal_prompt.html',trials=trials,goal=goal,function_name=funcmap[function_name])
    
    elif index=='1' or index=='2':
        describe_goal_pre=request.form['describe_goal_pre']

    if goal=='max_score':
        if index=='0':
            return render_template('max_score_instructions.html',trials=trials,goal=goal,function_name=funcmap[function_name])
        elif index=='1':
            return render_template('max_score.html',describe_goal_pre=describe_goal_pre,nbars=nbars,goal=goal,function_name=funcmap[function_name],function=function,trials=trials,bar_height=max_height,bar_width=bar_width)
    
    elif goal=='find_max':
        if index=='0':
            return render_template('find_max_instructions.html',trials=trials,goal=goal,function_name=funcmap[function_name])
        elif index=='1':
            return render_template('find_max.html',describe_goal_pre=describe_goal_pre,nbars=nbars,goal=goal,function_name=funcmap[function_name],function=function,trials=trials,bar_height=max_height,bar_width=bar_width)

    elif goal=='min_error':
        if index=='0':
            return render_template('min_error_instructions.html',trials=trials,predict_trials=predict_trials,goal=goal,function_name=funcmap[function_name])
        elif index=='1':
            return render_template('min_error_phase1.html',describe_goal_pre=describe_goal_pre,nbars=nbars,goal=goal,function_name=funcmap[function_name],function=function,trials=trials,bar_height=max_height,bar_width=bar_width)
        elif index=='2':
            max_score=float(max(function))
            phase1_response=request.form['test_response']
            test_start_time=request.form['test_start_time']
            test_response_time=request.form['test_response_time']
            phase2_prompts=find_phase2_set(phase1_response,function,predict_trials)
            random.shuffle(phase2_prompts)
            return render_template('min_error_phase2.html',describe_goal_pre=describe_goal_pre,max_score=max_score,nbars=nbars,goal=goal,function_name=funcmap[function_name],function=function,trials=predict_trials,bar_height=max_height,bar_width=bar_width,phase1_response=phase1_response,phase2_prompts=phase2_prompts,test_start_time=test_start_time,test_response_time=test_response_time)

if __name__=="__main__":
    app.run()
