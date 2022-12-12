#작동하는 놈 
#source /Users/yoon/Documents/python/FitML/RL_stable/bin/activate

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import random
from scipy.ndimage.filters import uniform_filter1d
import os
    

def errorfill(x, y, yerr, color=None, alpha_fill=0.2, ax=None,label=None):
    ax = ax if ax is not None else plt.gca()
    
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color,label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)

def graph_plot(filename, moving_average=0,color=None,label=None):
    _color = ['b','g','r','c','m','y','k']
    if color is None:
        #color = 'r' #'b','g','r','c','m','y','k','w'
        color = random.choice(_color) 
    _index=[]
    _mean=[]
    _std=[]
    #filename = "0.result/"+filename
    df = pd.read_csv(filename)
    #df = pd.read_csv('trpo_Mlp.csv')

    _index=df['index'].to_numpy()
    _mean=df['mean'].to_numpy()
    _std=df['std'].to_numpy()
    _std=_std*1.96

    #plt.errorbar(_index, _mean, yerr=_std)
    if moving_average==0:
        errorfill(_index, _mean, yerr=_std, color=color,label=label)
    else: 
        avg_mean = uniform_filter1d(_mean, size=moving_average)
        avg_std = uniform_filter1d(_std, size=moving_average)
        errorfill(_index, avg_mean, yerr=avg_std, color=color,label=label)
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.title('AntBulletEnv-v0')
    plt.legend()
    




if __name__ == "__main__":

    #_color = ['b','g','r','c','m','y','k']
    #graph_plot('ppo_Mlp.csv', moving_average=5, color='b',label='ppo')
    #graph_plot('sac_Mlp.csv', moving_average=5, color='y',label='sac')
    #graph_plot('trpo_Mlp.csv', moving_average=5, color='r',label='trpo')
    graph_plot('picker_200_9.csv', moving_average=5, color='m',label='picker')

    #df = pd.read_csv("trpo_Mlp.csv", header=None)
    #df = pd.read_csv('trpo_Mlp.csv', header=None)

    #print(df)
    #graph_plot('trpo_Mlp.csv', moving_average=0)
    plt.show()

