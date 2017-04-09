#!/usr/bin/python2.7

from __future__ import print_function
import sys,os
from flask import Flask,render_template,request,url_for
import numpy as np
import pandas as pd

def pos_linear(x):
    return x
    
def neg_linear(x):
    return -x
    
def power(x):
    return x**2
    
def scale(y,min_score,max_score):
    a=(max_score-min_score)/(y.max()-y.min())
    b=max_score-a*y.max()
    return a*y+b
    
def reset_participants():
    df=pd.DataFrame(columns=['goal','function','y','train_response','test_response','final_score'])
    df.to_pickle('participants.pkl')
    
functions={'Positive Linear':pos_linear,'Negative Linear':neg_linear,'Power':power}
goals={'Maximize Score':['max_score/max_score_train_instructions.html','training.html','max_score/max_score_instructions.html','max_score/max_score.html'],
       'Find Max':['find_max/find_max_train_instructions.html','training.html','find_max/find_max_instructions.html','find_max/find_max.html'],
        'Minimize Error':['min_error/min_error_train_instructions.html','training.html','min_error/min_error_instructions.html','min_error/min_error.html']}

bars=50
bar_height=500
max_score=np.random.uniform(bar_height-(bar_height/10.),bar_height)
min_score=np.random.uniform(0,bar_height/10.)
train_trials=5
test_trials=5

app=Flask(__name__)
my_dir = os.path.dirname(__file__)
path=os.path.join(my_dir, 'participants.pkl')

@app.route('/',methods=['GET','POST'])
def start():
    if request.method=='GET':
        return render_template('start.html')
    elif request.method=='POST':
        tag=request.form['postTag']
        if tag=='test':                        
            function=request.form['function']
            f=functions[function]
            x=np.arange(bars)
            y=scale(f(x),min_score,max_score)
            goal=request.form['goal']
        
            participants=pd.read_pickle(path)
            participants.loc[len(participants)]=[goal,function,y,np.nan,np.nan,np.nan]
            participants.to_pickle(path)
            
            return render_template(goals[goal][0],train_trials=train_trials,test_trials=test_trials,instructions='instructions1')
        
        elif tag=='instructions1':
            participants=pd.read_pickle(path)
            participant=participants.ix[len(participants)-1]
            goal=participant['goal']
            y=participant['y']
            return render_template(goals[goal][1],train_trials=train_trials,y=list(y),bar_height=bar_height)
            
        elif tag=='training':
            participants=pd.read_pickle(path)
            participants['train_response'][len(participants)-1]=request.form['train_response']
            participants.to_pickle(path)
            participant=participants.ix[len(participants)-1]
            goal=participant['goal']
            return render_template(goals[goal][2],test_trials=test_trials,instructions='instructions2')
            
        elif tag=='instructions2':
            participants=pd.read_pickle(path)
            participant=participants.ix[len(participants)-1]
            goal=participant['goal']
            y=participant['y']
            train=[int(s) for s in participant['train_response'].split(',')]
            return render_template(goals[goal][3],test_trials=test_trials,y=list(y),bar_height=bar_height,train=train)
            
        elif tag=='testing':
            participants=pd.read_pickle(path)
            participants['test_response'][len(participants)-1]=request.form['test_response']
            final_score=request.form['final_score'].split('.')[0]
            participants['final_score'][len(participants)-1]=final_score
            participants.to_pickle('participants.pkl')
            return 'Thank you for participating. Your final score was '+final_score
            
            
            
if __name__=="__main__":
    app.run()