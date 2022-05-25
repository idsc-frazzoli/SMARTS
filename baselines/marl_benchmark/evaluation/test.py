import argparse
from pathlib import Path
import pandas as pd
from time import gmtime, strftime
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

FIGSIZE = (16, 9)
LARGESIZE, MEDIUMSIZE, SMALLSIZE = 16, 13, 10
# LARGESIZE, MEDIUMSIZE, SMALLSIZE = 40, 30, 20

plt.rcParams.update({'font.size': LARGESIZE})
plt.rcParams.update({'axes.titlesize': LARGESIZE})
plt.rcParams.update({'axes.labelsize': MEDIUMSIZE})
plt.rcParams.update({'xtick.labelsize': SMALLSIZE})
plt.rcParams.update({'ytick.labelsize': SMALLSIZE})
plt.rcParams.update({'legend.fontsize': MEDIUMSIZE})
plt.rcParams.update({'figure.titlesize': LARGESIZE})

COLORS = {
    'blue': '#377eb8',
    'green': '#4daf4a',
    'purple': '#984ea3',
    'dark_orange': '#a65628',
    'orange': '#ff7f00',
    'red': '#e41a1c',
    'black': '#17202A',
}

AGENT_COLORS = {
    0: '#377eb8',
    1: '#4daf4a',
    2: '#984ea3',
    3: '#ff7f00',
    4: '#e41a1c'
}

PALETTE = ['#A93226', '#CB4335',  # red
           '#884EA0', '#7D3C98',  # purple
           '#2471A3', '#2E86C1',  # blue
           '#17A589', '#138D75',  # blue/green
           '#229954', '#28B463',  # green
           '#D4AC0D', '#D68910',  # yellow
           '#CA6F1E', '#BA4A00',  # orange
           ]


def main():
    # x = np.arange(0, 10, 0.001)
    # print(x)
    # y = 2*np.sin(x)
    # colors = np.abs(y)
    # fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
    # ax.scatter(x, y, c=colors, s=1500, cmap=plt.get_cmap("turbo"))
    # plt.savefig('test.png')

    # fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
    # lw = 1
    # ax.plot([0, 51.3], [10, 0], color='k', linewidth=lw)
    # ax.plot([-1.2, 33.5], [3.8, -3.3], color='k', linewidth=lw)
    # ax.plot([0, 33.5], [-10, -3.3], color='k', linewidth=lw)
    # ax.plot([1.2, 51.3], [-16.3, -6.2], color='k', linewidth=lw)
    # ax.plot([0, 51.3], [10, 0], color='k', linewidth=lw)
    # ax.plot([51.3, 110], [0, 0], color='k', linewidth=lw)
    # ax.plot([51.3, 110], [-6.2, -6.2], color='k', linewidth=lw)
    # ax.plot([-0.6, 32.5], [7, 0.4], color='grey', linewidth=lw, linestyle='--')
    # ax.plot([0.6, 32.5], [-13.2, -6.7], color='grey', linewidth=lw, linestyle='--')
    # ax.plot([51.3, 110], [-3.1, -3.1], color='grey', linewidth=lw, linestyle='--')
    # ax.set_aspect('equal', 'box')
    # plt.savefig("map_test.png")

    fig, axs = plt.subplots(3, 2, figsize=(15, 3*4), tight_layout=True, gridspec_kw={'width_ratios': [40, 1]})
    axs[0, 0].plot((1, 5), [2, 3])
    axs[0, 0].set_xlabel('xlabel')
    axs[0, 0].set_ylabel('ylabel')
    axs[0, 0].set_aspect('equal', 'box')
    axs[0, 0].set_title("title")
    axs[1, 0].plot([1, 3], [2, 3])
    axs[2, 0].plot([1, 3], [2, 3])

    cmap = LinearSegmentedColormap.from_list('rg', ["#008000", "#00FF00", "#ffff00", "#FF0000", "#800000"], N=256)
    norm = matplotlib.colors.Normalize(vmin=5, vmax=10)

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=axs[0, 1], orientation='vertical', label='Some Units')
    # fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
    #              cax=axs[1, 1], orientation='vertical', label='Some Units')
    axs[1, 1].set_visible(False)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=axs[2, 1], orientation='vertical', label='Some Units')
    # axs[0, 1].
    fig.tight_layout()
    plt.savefig("tst.png")


def test(empty):
    if empty:
        return
    print('Not empty.')


if __name__ == "__main__":
    main()
    # test(True)

