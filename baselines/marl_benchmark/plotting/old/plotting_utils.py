import pandas as pd
import os

from pathlib import Path
import numpy as np

# import matplotlib as mpl
import matplotlib.pyplot as plt

def str2list(s):
    slist = s[1:-1].split(',')
    return [float(x) for x in slist]


def load_progress_data(scenario, names):
    progress_paths = [os.path.join('../../baselines',
                                 'marl_benchmark',
                                 'log',
                                 'results',
                                 'run',
                                 scenario,
                                 name,
                                 'progress.csv') 
                      for name in names]
    return pd.concat([pd.read_csv(progress_path) for progress_path in progress_paths], ignore_index=True)


def plot_features(df, features, title="", xlabel="", ylabel="", xlim=None, ylim=None):
    fig = plt.figure()
    ax = plt.axes()
    
    y = df[features]
    
    for feature in features:
        ax.plot(df[feature], label=feature)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.legend()
    


    