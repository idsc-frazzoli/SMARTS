import argparse
from pathlib import Path
import pandas as pd
from time import gmtime, strftime
import os

from typing import List, Tuple, Dict, Union, Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import shutil

from timeit import default_timer as timer


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


def get_min_max(checkpoint_path):

    info = {'max_episode_length': 0,
            'min_speed': 0.0,
            'max_speed': 16.0,
            'min_acceleration': 100.0,
            'max_acceleration': -100.0,
            'max_step_reward': 0.0,
            'min_step_reward': 0.0,
            'min_x_pos': 0.0,
            'max_x_pos': 0.0,
            'min_y_pos': 0.0,
            'max_y_pos': 0.0,
            'n_agents': None,
            }


    times = os.listdir(Path(checkpoint_path))
    for time in times:
        if time == "plots":
            continue
        episodes = os.listdir(Path(checkpoint_path, time))
        for episode in episodes:
            info['n_agents'] = len(os.listdir(Path(checkpoint_path, time, episode)))
            for agent in range(info['n_agents']):
                csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
                df = pd.read_csv(csv_path, index_col=0, header=None).T
                info['max_episode_length'] = max(info['max_episode_length'], df.shape[0])
                info['max_speed'] = max(info['max_speed'], max(df["Speed"]))
                info['min_speed'] = min(info['min_speed'], min(df["Speed"]))
                info['max_acceleration'] = max(info['max_acceleration'], max(df["Acceleration"]))
                info['min_acceleration'] = min(info['min_acceleration'], min(df["Acceleration"]))
                info['max_step_reward'] = max(info['max_step_reward'], max(df["Step_Reward"]))
                info['min_step_reward'] = min(info['min_step_reward'], min(df["Step_Reward"]))
                info['max_x_pos'] = max(info['max_x_pos'], max(df["Xpos"]))
                info['min_x_pos'] = min(info['min_x_pos'], min(df["Xpos"]))
                info['max_y_pos'] = max(info['max_y_pos'], max(df["Ypos"]))
                info['min_y_pos'] = min(info['min_y_pos'], min(df["Ypos"]))

    return info


def plot_positions(checkpoint_path):
    datetime = strftime("%Y%m%d_%H%M%S_", gmtime())
    matplotlib.rcParams['savefig.dpi'] = 800
    times = os.listdir(Path(checkpoint_path))
    fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
    for time in times:
        if time == "plots":
            continue
        episodes = os.listdir(Path(checkpoint_path, time))
        for episode in episodes:
            n_agents = len(os.listdir(Path(checkpoint_path, time, episode)))
            for agent in range(n_agents):
                csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
                df = pd.read_csv(csv_path, index_col=0, header=None).T

                # ax.plot(df['Xpos'], df['Ypos'],
                #         color=AGENT_COLORS[agent], label='Agent ' + str(agent))
                ax.scatter(df['Xpos'], df['Ypos'],
                           c=df["Speed"]/16, label='Agent ' + str(agent), s=2, cmap=plt.get_cmap("turbo"), alpha=0.1)

    plots_path = Path(checkpoint_path, 'plots')
    plots_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(plots_path, '{}agent_positions.png'.format(datetime)))


def plot_map(ax):
    lw = 1
    ax.plot([0, 51.3], [10, 0], color='k', linewidth=lw)
    ax.plot([-1.2, 33.5], [3.8, -3.3], color='k', linewidth=lw)
    ax.plot([0, 33.5], [-10, -3.3], color='k', linewidth=lw)
    ax.plot([1.2, 51.3], [-16.3, -6.2], color='k', linewidth=lw)
    ax.plot([0, 51.3], [10, 0], color='k', linewidth=lw)
    ax.plot([51.3, 110], [0, 0], color='k', linewidth=lw)
    ax.plot([51.3, 110], [-6.2, -6.2], color='k', linewidth=lw)
    ax.plot([-0.6, 32.5], [7, 0.4], color='grey', linewidth=lw, linestyle='--')
    ax.plot([0.6, 32.5], [-13.2, -6.7], color='grey', linewidth=lw, linestyle='--')
    ax.plot([51.3, 110], [-3.1, -3.1], color='grey', linewidth=lw, linestyle='--')

    ax.set_xlim([-5, 115])
    ax.set_ylim([-17, 11])



def make_video(
        tmp_path,
        save_path,
        fps=20,
):
    import moviepy.video.io.ImageSequenceClip
    image_files = [os.path.join(tmp_path, img)
                   for img in os.listdir(tmp_path)
                   if img.endswith(".png")]
    image_files.sort()
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(str(save_path))

    shutil.rmtree(tmp_path)


def plot_positions1(checkpoint_path: Union[str, Path],
                    info: Dict,
                    figure_name: str = 'test',
                    color: str = 'speed',
                    dpi: int = 100) -> None:

    x_min, x_max = info['min_x_pos'], info['max_x_pos']
    y_min, y_max = info['min_y_pos'], info['max_y_pos']
    speed_min, speed_max = info['min_speed'], info['max_speed']
    speed_range = speed_max - speed_min
    reward_min, reward_max = info['min_step_reward'], info['max_step_reward']
    reward_range = reward_max - reward_min

    datetime = strftime("%Y%m%d_%H%M%S_", gmtime())
    matplotlib.rcParams['savefig.dpi'] = dpi

    times = os.listdir(Path(checkpoint_path))
    fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
    plot_map(ax)
    ax.set_aspect('equal', 'box')
    for time in times:
        if time == "plots":
            continue
        episodes = os.listdir(Path(checkpoint_path, time))
        for episode in episodes:
            n_agents = len(os.listdir(Path(checkpoint_path, time, episode)))
            for agent in range(n_agents):
                csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
                df = pd.read_csv(csv_path, index_col=0, header=None).T

                # ax.plot(df['Xpos'], df['Ypos'],
                #         color=AGENT_COLORS[agent], label='Agent ' + str(agent))
                ax.scatter(df['Xpos'], df['Ypos'],
                           c=df["Speed"]/16,
                           label='Agent ' + str(agent),
                           s=2, cmap=plt.get_cmap("turbo"),
                           alpha=0.1)

    plots_path = Path(checkpoint_path, 'plots')
    plots_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(plots_path, '{}'.format(datetime) + figure_name + '.png'))
    plt.close('all')


def animate_positions(checkpoint_path):

    plots_path = Path(checkpoint_path, 'plots', 'tmp')
    plots_path.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams['savefig.dpi'] = 100

    info = get_min_max(checkpoint_path)
    print(info)

    dfs = {}
    for agent in range(info['n_agents']):
        dfs[agent] = []

    times = os.listdir(Path(checkpoint_path))
    for time in times:
        if time == "plots":
            continue
        episodes = os.listdir(Path(checkpoint_path, time))
        for episode in episodes:
            for agent in range(info['n_agents']):
                csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
                dfs[agent].append(pd.read_csv(csv_path, index_col=0, header=None).T)

    times = os.listdir(Path(checkpoint_path))
    for time_step in range(int(info['max_episode_length'])):
        print(time_step)
        fig, ax = plt.subplots(figsize=(15, 4), tight_layout=True)
        plot_map(ax)
        ax.set_aspect('equal', 'box')
        for agent in range(info['n_agents']):
            for df in dfs[agent]:
                ax.scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
                           c=df["Speed"][:time_step], vmin=info['min_speed'], vmax=info['max_speed'],
                           label='Agent ' + str(agent),
                           s=2,
                           cmap=plt.get_cmap("turbo"),
                           alpha=0.1)
        plt.xlim([-5, 115])
        plt.ylim([-17, 11])
        plt.savefig(Path(plots_path, '{:04d}.png'.format(time_step)))
        plt.close('all')

    datetime = strftime("%Y%m%d_%H%M%S_", gmtime())
    make_video(plots_path, Path(checkpoint_path, 'plots', '{}'.format(datetime) + 'video.mp4'), 20)


def animate(checkpoint_path):

    dot_size = 2
    alpha = 0.1

    plots_path = Path(checkpoint_path, 'plots', 'tmp')
    plots_path.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams['savefig.dpi'] = 100

    start = timer()

    info = get_min_max(checkpoint_path)
    print(info)

    end = timer()
    print('Time to get min max: {}'.format(end - start))

    start = timer()

    dfs = {}
    for agent in range(info['n_agents']):
        dfs[agent] = []

    times = os.listdir(Path(checkpoint_path))
    for time in times:
        if time == "plots":
            continue
        episodes = os.listdir(Path(checkpoint_path, time))
        for episode in episodes:
            for agent in range(info['n_agents']):
                csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
                dfs[agent].append(pd.read_csv(csv_path, index_col=0, header=None).T)

    end = timer()
    print('Time to set up dfs: {}'.format(end - start))

    times = os.listdir(Path(checkpoint_path))
    for time_step in range(int(info['max_episode_length'])):
        print(time_step)

        start = timer()

        fig, axs = plt.subplots(3, 2, figsize=(15, 3*4), tight_layout=True, gridspec_kw={'width_ratios': [40, 1]})
        # plot_map(ax)
        # ax.set_aspect('equal', 'box')
        for agent in range(info['n_agents']):
            start1 = timer()
            for df in dfs[agent]:
                # speed plot
                plot_map(axs[0, 0])
                cmap_speed = matplotlib.cm.turbo
                norm_speed = matplotlib.colors.Normalize(vmin=info['min_speed'], vmax=info['max_speed'])
                axs[0, 0].scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
                                  c=df["Speed"][:time_step], vmin=info['min_speed'], vmax=info['max_speed'],
                                  label='Agent ' + str(agent),
                                  s=dot_size,
                                  cmap=plt.get_cmap("turbo"),
                                  alpha=alpha)
                axs[0, 0].set_xlabel(r'x $[m]$')
                axs[0, 0].set_ylabel(r'y $[m]$')
                axs[0, 0].set_aspect('equal', 'box')
                axs[0, 0].set_title("Speed Distribution")
                fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_speed, cmap=cmap_speed),
                             cax=axs[0, 1], orientation='vertical', label=r'speed $[m/s]$')

                # acceleration plot
                plot_map(axs[1, 0])
                cmap_acc = matplotlib.cm.turbo
                norm_acc = matplotlib.colors.Normalize(vmin=info['min_acceleration'], vmax=10)
                axs[1, 0].scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
                                  c=df["Acceleration"][:time_step], vmin=info['min_acceleration'], vmax=10,
                                  label='Agent ' + str(agent),
                                  s=dot_size,
                                  cmap=plt.get_cmap("turbo"),
                                  alpha=alpha)
                axs[1, 0].set_xlabel(r'x $[m]$')
                axs[1, 0].set_ylabel(r'y $[m]$')
                axs[1, 0].set_aspect('equal', 'box')
                axs[1, 0].set_title("Acceleration Distribution")
                fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_acc, cmap=cmap_acc),
                             cax=axs[1, 1], orientation='vertical', label=r"acceleration $[m/s^2]$")

                # reward plot
                plot_map(axs[2, 0])
                cmap_acc = matplotlib.cm.turbo
                norm_acc = matplotlib.colors.Normalize(vmin=-5, vmax=0)
                axs[2, 0].scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
                                  c=df["Step_Reward"][:time_step], vmin=-5, vmax=0,
                                  label='Agent ' + str(agent),
                                  s=dot_size,
                                  cmap=plt.get_cmap("turbo"),
                                  alpha=alpha)
                axs[2, 0].set_xlabel(r'x $[m]$')
                axs[2, 0].set_ylabel(r'y $[m]$')
                axs[2, 0].set_aspect('equal', 'box')
                axs[2, 0].set_title("Step Reward Distribution")
                fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_acc, cmap=cmap_acc),
                             cax=axs[2, 1], orientation='vertical', label=r"reward")

            end1 = timer()
            print('Time to plot one agent: {}'.format(end1 - start1))

        start1 = timer()
        plt.savefig(Path(plots_path, '{:04d}.png'.format(time_step)))


        end = timer()

        # plt.imsave(img, name)

        plt.close('all')
        print('Time to save and close: {}'.format(end - start1))
        print('Time to render one frame: {}'.format(end - start))

    datetime = strftime("%Y%m%d_%H%M%S_", gmtime())
    make_video(plots_path, Path(checkpoint_path, 'plots', '{}'.format(datetime) + 'video.mp4'), 20)












# def plot_positions_test(checkpoint_path):
#     datetime = strftime("%Y%m%d_%H%M%S_", gmtime())
#     matplotlib.rcParams['savefig.dpi'] = 800
#     cmap = matplotlib.cm.get_cmap("turbo")
#     print(cmap)
#
#     (max_episode_length,
#      min_speed, max_speed,
#      min_step_reward, max_step_reward,
#      min_x_pos, max_x_pos,
#      min_y_pos, max_y_pos) = get_min_max(checkpoint_path)
#
#     times = os.listdir(Path(checkpoint_path))
#     fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
#     for time in times:
#         if time == "plots":
#             continue
#         episodes = os.listdir(Path(checkpoint_path, time))
#         for episode in episodes:
#             n_agents = len(os.listdir(Path(checkpoint_path, time, episode)))
#             for agent in range(n_agents):
#                 csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
#                 df = pd.read_csv(csv_path, index_col=0, header=None).T
#
#                 for t in range(df.shape[0]-1):
#                     color_value = (df['Speed'][t:t+1] - min_speed) / (max_speed - min_speed)
#                     ax.plot(df['Xpos'][t:t+1], df['Ypos'][t:t+1],
#                             color=cmap(color_value), label='Agent ' + str(agent))
#                 # ax.scatter(df['Xpos'], df['Ypos'],
#                 #            c=df["Speed"] / 16, label='Agent ' + str(agent), s=2, cmap=plt.get_cmap("turbo"), alpha=0.1)
#
#     plots_path = Path(checkpoint_path, 'plots')
#     plots_path.mkdir(parents=True, exist_ok=True)
#     plt.savefig(Path(plots_path, '{}agent_positions.png'.format(datetime)))