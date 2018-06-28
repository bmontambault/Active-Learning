import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt

def get_participant(df, i):
    
    testX = np.array(df['response'].iloc[i])
    testY = np.array(df['function'].iloc[i])
    a = testY[testX]
    
    plt.plot(testX, a, 'bo')
    plt.plot(testY, 'g')


df = pd.read_csv('20182901_summary.csv')
df['function'] = df['function'].apply(lambda x: ast.literal_eval(x))
df['response'] = df['response'].apply(lambda x: ast.literal_eval(x))

