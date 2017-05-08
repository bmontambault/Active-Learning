#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys,os
from flask import Flask,render_template,request
import json
import string
import random
from config import nbars,max_score,trials

goals=['max_score','find_max','min_error']
functions=['pos_linear','neg_linear','pos_power','neg_power','pos_quad','neg_quad','sin','se']

funcmap={f:''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10)) for f in functions}
revfuncmap={value:key for key,value in funcmap.iteritems()}

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def start():
    if request.method=='GET':
        return render_template('start.html')
    elif request.method=='POST':
        index=request.form['index']
        if index=='0':
            with open('data.json') as json_data:
                data=json.load(json_data)
                json_data.close()

            if data['goal']==[]:
                goal=goals[0]
                function_name=functions[0]
            else:
                prev_goal=goals.index(data['goal'][-1])
                prev_func=functions.index(data['function_name'][-1])

                if len(functions)-prev_func==1:
                    if len(goals)-prev_goal==1:
                        goal=goals[0]
                    else:
                        goal=goals[prev_goal+1]
                    function_name=functions[0]
                else:
                    goal=goals[prev_goal]
                    function_name=functions[prev_func+1]
            return task('',goal,function_name,index)
        else:
            goal=request.form['goal']
            function_tag=request.form['function_name']
            if function_tag in functions:
                function_name=function_tag
            else:
                function_name=revfuncmap[function_tag]
            return task('test',goal,function_name,index)


@app.route('/admin',methods=['GET','POST'])
def start_admin():
    if request.method=='GET':
        return render_template('start_admin.html')
    elif request.method=='POST':
        goal=request.form['goal']
        function_tag=request.form['function_name']
        if function_tag in functions:
            function_name=function_tag
        else:
            function_name=revfuncmap[function_tag]
        index=request.form['index']
        return task('test',goal,function_name,index)

def task(idtag,goal,function_name,index):
    with open('functions.json') as json_data:
            function=json.load(json_data)[function_name]
            json_data.close()

    if index=='exit':
        with open('data.json') as json_data:
            data=json.load(json_data)
            json_data.close()
        ID=idtag+str(len(data['ID'])+1)

        for key in request.form.keys():
            if key in data.keys():
                if key=='function_name':
                    data[key].append(revfuncmap[request.form[key]])
                else:
                    data[key].append(request.form[key])
        data['ID'].append(ID)

        data['function'].append(function)
        with open('data.json', 'w') as json_data:
            json.dump(data,json_data)
        return 'Thank you for participating. Your final score is {0}'.format(data['final_score'][-1].split('.')[0])

    elif goal=='max_score':
        if index=='0':
            return render_template('max_score_instructions.html',trials=trials,goal=goal,function_name=funcmap[function_name])
        elif index=='1':
            return render_template('max_score.html',nbars=nbars,goal=goal,function_name=funcmap[function_name],function=function,trials=trials,bar_height=max_score)
        elif index=='2':
            test_response=request.form['test_response']
            final_score=request.form['final_score']
            return render_template('exit_survey.html',goal=goal,function_name=funcmap[function_name],test_response=test_response,final_score=final_score)

    elif goal=='find_max':
        if index=='0':
            return render_template('find_max_instructions.html',goal=goal,function_name=funcmap[function_name])
        elif index=='1':
            return render_template('find_max.html',nbars=nbars,goal=goal,function_name=funcmap[function_name],function=function,trials=trials,bar_height=max_score)
        elif index=='2':
            test_response=request.form['test_response']
            final_score=request.form['final_score']
            return render_template('exit_survey.html',goal=goal,function_name=funcmap[function_name],test_response=test_response,final_score=final_score)

    elif goal=='min_error':
        if index=='0':
            return render_template('min_error_instructions.html',goal=goal,function_name=funcmap[function_name])
        elif index=='1':
            return render_template('min_error_phase1.html',nbars=nbars,goal=goal,function_name=funcmap[function_name],function=function,trials=trials,bar_height=max_score)
        elif index=='2':
            phase1_response=request.form['test_response']
            return render_template('min_error_phase2.html',nbars=nbars,goal=goal,function_name=funcmap[function_name],function=function,trials=trials,bar_height=max_score,phase1_response=phase1_response)
        elif index=='3':
            test_response=request.form['test_response']
            final_score=request.form['final_score']
            return render_template('exit_survey.html',goal=goal,function_name=funcmap[function_name],test_response=test_response,final_score=final_score)

if __name__=="__main__":
    app.run()
