from flask import Flask, render_template, request
import json
import numpy as np
import os

app=Flask(__name__)
app.secret_key=''
app.config['SESSION_TYPE']='filesystem'

path = os.path.dirname(os.path.realpath(__file__))
with open(path + '/model_simulations.json', 'r') as f:
    model_results = json.load(f)