import argparse
from pathlib import Path
import pandas as pd
from time import gmtime, strftime
import os

from typing import List, Tuple, Dict, Union, Any
from pandas import DataFrame

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import shutil

from timeit import default_timer as timer

from scipy.stats import gaussian_kde
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

COLORS_LIST = ['#377eb8', '#4daf4a', '#984ea3', '#a65628', '#ff7f00', '#e41a1c', '#17202A', ]

ACTIONS = {0: 'keep_lane', 1: 'slow_down', 2: 'change_lane_left', 3: 'change_lane_right'}
ACTION_COLORS = {0: '#DFFF00', 1: '#DE3163', 2: '#6495ED', 3: '#0DE273'}

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
           '#229954', '#28B463',  # green
           '#D4AC0D', '#D68910',  # yellow
           '#CA6F1E', '#BA4A00',  # orange
           '#17A589', '#138D75',  # blue/green
           ]


class StraightLane:
    float_pair = Tuple[float, float]

    def __init__(self,
                 boundaries: List[Tuple[float_pair, float_pair]],
                 center_lines: List[Tuple[float_pair, float_pair]]):
        # notation: boundaries = [((x1, x2), (y1, y2)), ((x3, x4), (y3, y4))]
        # same for center_lines, but can be more than 2
        self.boundaries = boundaries
        self.center_lines = center_lines

    def plot(self, ax):
        for boundary in self.boundaries:
            ax.plot(boundary[0], boundary[1], color='k', linewidth=1)
        for center_line in self.center_lines:
            ax.plot(center_line[0], center_line[1], color='grey', linewidth=1, linestyle='--')


class Map:
    def __init__(self, lanes: Union[List[StraightLane], Any],
                 x_lim: Union[Tuple[float, float], Any],
                 y_lim: Union[Tuple[float, float], Any],
                 aspect_ratio: Union[Tuple[int, int], Any] = (1, 1),
                 empty: bool = False):
        self.lanes = lanes
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.aspect_ratio = aspect_ratio
        self.empty = empty

    def plot(self, ax):
        if self.empty:
            return
        for lane in self.lanes:
            lane.plot(ax)
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


def get_info(checkpoint_path: Union[str, Path]):
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
            'max_cost_com': -1e10,
            'min_cost_com': 1e10,
            'max_cost_per_acceleration': -1e10,
            'min_cost_per_acceleration': 1e10,
            'max_goal_improvement_reward': -1e10,
            'min_goal_improvement_reward': 1e10,
            'max_cost_per': -1e10,
            'min_cost_per': 1e10,
            }

    times = os.listdir(Path(checkpoint_path))
    for time in times:
        # print(time)
        if time == "plots":
            continue
        episodes = os.listdir(Path(checkpoint_path, time))
        print(episodes)
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
                # in some older evaluation run files, this information does not exist
                try:
                    cost_per = [list(df['cost_per_time'])[i] + list(df['cost_per_acceleration'])[i] -
                                list(df['goal_improvement_reward'])[i]
                                for i in range(len(list(df['cost_per_time'])))]
                    info['max_cost_com'] = max(info['max_cost_com'], max(df['cost_com']))
                    info['min_cost_com'] = min(info['min_cost_com'], min(df['cost_com']))
                    info['max_cost_per_acceleration'] = max(info['max_cost_per_acceleration'],
                                                            max(df['cost_per_acceleration']))
                    info['min_cost_per_acceleration'] = min(info['min_cost_per_acceleration'],
                                                            min(df['cost_per_acceleration']))
                    info['max_goal_improvement_reward'] = max(info['max_goal_improvement_reward'],
                                                              max(df['goal_improvement_reward']))
                    info['min_goal_improvement_reward'] = min(info['min_goal_improvement_reward'],
                                                              min(df['goal_improvement_reward']))
                    info['max_cost_per'] = max(info['max_cost_per'], max(cost_per))
                    info['min_cost_per'] = min(info['min_cost_per'], min(cost_per))
                except:
                    continue

    return info


def make_video(
        tmp_path,
        save_path,
        fps=20,
        remove_tmp: bool = True
):
    # import from smarts conda environment
    import moviepy.video.io.ImageSequenceClip
    image_files = [os.path.join(tmp_path, img)
                   for img in os.listdir(tmp_path)
                   if img.endswith(".png")]
    image_files.sort()
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    print(save_path)
    clip.write_videofile(str(save_path), codec="libx264")

    if remove_tmp:
        shutil.rmtree(tmp_path)


def get_rewards(dfs: Dict[str, List[DataFrame]],
                masks: Dict[str, List],
                goal_reached_reward: float = 300.0,
                ) -> Tuple[List[float], List[float]]:
    """
    :param goal_reached_reward: Reward an agent gets upon reaching the goal position/region.
    :param dfs: Lists of all episode dataframes in a dict with agent keys.
    :param masks: Masks containing information whether an off-road event happened, a collision happened, or all the
    agents reached the goals.
    :return: Two lists of unfiltered and filtered total episode rewards.
    """
    off_road_mask = masks["off_road_mask"]
    collision_mask = masks["collision_mask"]
    goal_reached_mask = masks["goal_reached_mask"]

    rewards = [0] * len(dfs[0])
    filtered_rewards = [np.nan] * len(dfs[0])

    for i in range(len(dfs[0])):
        for agent in dfs.keys():
            rewards[i] += sum(dfs[agent][i]["Step_Reward"])
        # if no car went off-road and no collision happened and all cars reached the goal
        if not off_road_mask[i] and not collision_mask[i] and goal_reached_mask[i]:
            for agent in dfs.keys():
                filtered_rewards[i] = 0.0
                # sum rewards and subtract goal_reached_reward
                filtered_rewards[i] += sum(dfs[agent][i]["Step_Reward"]) - goal_reached_reward

    filtered_rewards = [x for x in filtered_rewards if x is not np.nan]

    return rewards, filtered_rewards


def get_poa(cent_rewards_filtered: List[float], decent_rewards_filtered: List[float]) -> Tuple[float, list, list]:
    min_rew = min(min(cent_rewards_filtered), min(decent_rewards_filtered))
    max_rew = max(max(cent_rewards_filtered), max(decent_rewards_filtered))

    # print(min_rew)
    # print(max_rew)

    rew_range = max_rew - min_rew

    cent_rew_normalized = [(x - min_rew) / rew_range for x in cent_rewards_filtered]
    decent_rew_normalized = [(x - min_rew) / rew_range for x in decent_rewards_filtered]

    poa = np.mean(cent_rew_normalized) / np.mean(decent_rew_normalized)

    return poa, cent_rew_normalized, decent_rew_normalized


def load_checkpoint_dfs(checkpoint_path: Union[str, Path], info: dict) -> Tuple[dict, dict]:
    dfs = {}
    for agent in range(info['n_agents']):
        dfs[agent] = []

    # if off_road_mask[i] = 1, an off_road_event happened in episode i (in dfs)
    off_road_mask = []
    # if collision_mask[i] = 1, a collision happened in episode i (in dfs)
    collision_mask = []
    # if goal_reached_mask[i] = 1, every agent reached the goal in episode i (in dfs)
    goal_reached_mask = []

    times = os.listdir(Path(checkpoint_path))
    for time in times:
        if time == "plots":
            continue
        episodes = os.listdir(Path(checkpoint_path, time))
        for episode in episodes:
            off_road_mask.append(0)
            collision_mask.append(0)
            goal_reached_mask.append(1)
            for agent in range(info['n_agents']):
                csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
                dfs[agent].append(pd.read_csv(csv_path, index_col=0, header=None).T)
                if dfs[agent][-1]["Num_Collision"][1] > 0:
                    collision_mask[-1] = 1
                if dfs[agent][-1]["Num_Off_Road"][1] > 0:
                    off_road_mask[-1] = 1
                # if statement because Goal_Reached was added later to episode logs
                if "Goal_Reached" in dfs[agent][-1].keys():
                    if dfs[agent][-1]["Goal_Reached"][1] < 1:
                        goal_reached_mask[-1] = 0

    masks = {"off_road_mask": off_road_mask, "collision_mask": collision_mask, "goal_reached_mask": goal_reached_mask}

    return dfs, masks


def plot_positions(checkpoint_path: Union[str, Path],
                   info: dict,
                   scenario_map: Map,
                   save_path: Union[str, Path] = None,
                   coloring: str = 'speed',
                   dpi: int = 100,
                   ) -> None:
    assert coloring in ["speed", "agents", "reward", "acceleration", "control_input"]

    x_min, x_max = info['min_x_pos'], info['max_x_pos']
    y_min, y_max = info['min_y_pos'], info['max_y_pos']
    speed_min, speed_max = info['min_speed'], info['max_speed']
    speed_range = speed_max - speed_min
    reward_min, reward_max = info['min_step_reward'], info['max_step_reward']
    reward_range = reward_max - reward_min

    datetime = strftime("%Y%m%d_%H%M%S_", gmtime())
    matplotlib.rcParams['savefig.dpi'] = dpi

    if save_path is None:
        save_path = Path(checkpoint_path, 'plots')

    save_path.mkdir(parents=True, exist_ok=True)

    dfs, _ = load_checkpoint_dfs(checkpoint_path, info)

    dx_dy = (scenario_map.x_lim[1] - scenario_map.x_lim[0]) / (scenario_map.y_lim[1] - scenario_map.y_lim[0])
    aspect_ratio = scenario_map.aspect_ratio

    if coloring == "speed":
        fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
                                gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
        scenario_map.plot(axs[0])
        axs[0].set_aspect('equal', 'box')
        for agent in range(info['n_agents']):
            for df in dfs[agent]:
                axs[0].scatter(df['Xpos'], df['Ypos'],
                               c=df["Speed"], vmin=0, vmax=16,
                               s=2,
                               cmap=plt.get_cmap("turbo"),
                               alpha=0.1)
        axs[0].set_xlabel(r'x $[m]$')
        axs[0].set_ylabel(r'y $[m]$')
        axs[0].title.set_text('Checkpoint {:04d}'.format(int(str(checkpoint_path)[-6:])))
        cmap_speed = matplotlib.cm.turbo
        norm_speed = matplotlib.colors.Normalize(vmin=0, vmax=16)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_speed, cmap=cmap_speed),
                     cax=axs[1], orientation='vertical', label=r'speed $[m/s]$')
        plt.savefig(Path(save_path, 'speed_pos_cp{:04d}.png'.format(int(str(checkpoint_path)[-6:]))))
        plt.close('all')

    if coloring == "agents":
        fig, ax = plt.subplots(figsize=aspect_ratio, tight_layout=True)
        scenario_map.plot(ax)
        ax.set_aspect('equal', 'box')
        for agent in range(info['n_agents']):
            for df in dfs[agent]:
                ax.scatter(df['Xpos'], df['Ypos'],
                           c=AGENT_COLORS[agent],
                           s=2,
                           alpha=0.1)
            ax.scatter(1e5, 1e5, c=AGENT_COLORS[agent], label="agent {}".format(agent))
        ax.set_xlabel(r'x $[m]$')
        ax.set_ylabel(r'y $[m]$')
        plt.legend()
        ax.title.set_text('Checkpoint {:04d}'.format(int(str(checkpoint_path)[-6:])))
        plt.savefig(Path(save_path, 'agent_pos_cp{:04d}.png'.format(int(str(checkpoint_path)[-6:]))))
        plt.close('all')

    if coloring == "reward":
        fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
                                gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
        scenario_map.plot(axs[0])
        axs[0].set_aspect('equal', 'box')
        for agent in range(info['n_agents']):
            for df in dfs[agent]:
                axs[0].scatter(df['Xpos'], df['Ypos'],
                               c=df["Step_Reward"], vmin=-7, vmax=2,
                               s=2,
                               cmap=plt.get_cmap("turbo"),
                               alpha=0.1)
        axs[0].set_xlabel(r'x $[m]$')
        axs[0].set_ylabel(r'y $[m]$')
        axs[0].title.set_text('Checkpoint {:04d}'.format(int(str(checkpoint_path)[-6:])))
        cmap_rew = matplotlib.cm.turbo
        norm_rew = matplotlib.colors.Normalize(vmin=-7, vmax=2)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_rew, cmap=cmap_rew),
                     cax=axs[1], orientation='vertical', label=r'reward')
        plt.savefig(Path(save_path, 'reward_pos_cp{:04d}.png'.format(int(str(checkpoint_path)[-6:]))))
        plt.close('all')

    if coloring == "acceleration":
        fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
                                gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
        scenario_map.plot(axs[0])
        axs[0].set_aspect('equal', 'box')
        for agent in range(info['n_agents']):
            for df in dfs[agent]:
                axs[0].scatter(df['Xpos'], df['Ypos'],
                               c=df["Acceleration"], vmin=0, vmax=10,
                               s=2,
                               cmap=plt.get_cmap("turbo"),
                               alpha=0.1)
        axs[0].set_xlabel(r'x $[m]$')
        axs[0].set_ylabel(r'y $[m]$')
        axs[0].title.set_text('Checkpoint {:04d}'.format(int(str(checkpoint_path)[-6:])))
        cmap_acc = matplotlib.cm.turbo
        norm_acc = matplotlib.colors.Normalize(vmin=0, vmax=10)
        fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_acc, cmap=cmap_acc),
                     cax=axs[1], orientation='vertical', label=r'acceleration $[m/s^2]$')
        plt.savefig(Path(save_path, 'acc_pos_cp{:04d}.png'.format(int(str(checkpoint_path)[-6:]))))
        plt.close('all')

    if coloring == "control_input":
        fig, ax = plt.subplots(figsize=aspect_ratio, tight_layout=True)
        scenario_map.plot(ax)
        ax.set_aspect('equal', 'box')
        for agent in range(info['n_agents']):
            for df in dfs[agent]:
                ax.scatter(df['Xpos'], df['Ypos'],
                           c=[ACTION_COLORS[int(x)] for x in df['Operations']],
                           s=2,
                           alpha=0.1)
        for action in range(4):
            ax.scatter(1e5, 1e5, c=ACTION_COLORS[action], label=ACTIONS[action])
        ax.set_xlabel(r'x $[m]$')
        ax.set_ylabel(r'y $[m]$')
        plt.legend()
        ax.title.set_text('Checkpoint {:04d}'.format(int(str(checkpoint_path)[-6:])))
        plt.savefig(Path(save_path, 'ctrl_pos_cp{:04d}.png'.format(int(str(checkpoint_path)[-6:]))))
        plt.close('all')


def animate_positions(checkpoint_path: Path,
                      scenario_map: Map,
                      coloring: str = "speed"
                      ) -> None:
    assert coloring in ["speed", "agents", "reward", "acceleration", "control_input", "cost_com",
                        "time", "cost_per_acceleration", "goal_improvement_reward", "cost_per"]

    plots_path = Path(checkpoint_path, 'plots', 'tmp_{}'.format(coloring))
    plots_path.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams['savefig.dpi'] = 100

    print(checkpoint_path)
    info = get_info(checkpoint_path)


    dfs, _ = load_checkpoint_dfs(checkpoint_path, info)

    if scenario_map.empty:
        dx_dy = 5
    else:
        dx_dy = (scenario_map.x_lim[1] - scenario_map.x_lim[0]) / (scenario_map.y_lim[1] - scenario_map.y_lim[0])
    aspect_ratio = scenario_map.aspect_ratio

    cmap_green_red = LinearSegmentedColormap.from_list('rg', ["#008000", "#00FF00", "#ffff00", "#FF0000", "#800000"], N=256)
    cmap_red_green = LinearSegmentedColormap.from_list('rg', ["#800000", "#FF0000", "#ffff00", "#00FF00", "#008000"], N=256)

    if coloring == "speed":
        for time_step in range(int(info['max_episode_length'])):
            print('Current frame: {}'.format(time_step))
            fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
                                    gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
            scenario_map.plot(axs[0])
            axs[0].set_aspect('equal', 'box')
            for agent in range(info['n_agents']):
                for df in dfs[agent]:
                    axs[0].scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
                                   c=df["Speed"][:time_step], vmin=info['min_speed'], vmax=info['max_speed'],
                                   label='Agent ' + str(agent),
                                   s=2,
                                   cmap=plt.get_cmap("turbo"),
                                   alpha=0.1)
            axs[0].set_xlabel(r'x $[m]$')
            axs[0].set_ylabel(r'y $[m]$')
            cmap_speed = matplotlib.cm.turbo
            norm_speed = matplotlib.colors.Normalize(vmin=info['min_speed'], vmax=info['max_speed'])
            fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_speed, cmap=cmap_speed),
                         cax=axs[1], orientation='vertical', label=r'speed $[m/s]$')
            plt.savefig(Path(plots_path, '{:04d}.png'.format(time_step)))
            plt.close('all')

    if coloring == "agents":
        for time_step in range(int(info['max_episode_length'])):
            print('Current frame: {}'.format(time_step))
            fig, ax = plt.subplots(figsize=aspect_ratio, tight_layout=True)
            scenario_map.plot(ax)
            ax.set_aspect('equal', 'box')
            for agent in range(info['n_agents']):
                for df in dfs[agent]:
                    ax.scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
                               c=AGENT_COLORS[agent],
                               label='Agent ' + str(agent),
                               s=2,
                               alpha=0.1)
            ax.set_xlabel(r'x $[m]$')
            ax.set_ylabel(r'y $[m]$')
            # plt.legend()
            plt.savefig(Path(plots_path, '{:04d}.png'.format(time_step)))
            plt.close('all')

    if coloring == "reward":
        for time_step in range(int(info['max_episode_length'])):
            print('Current frame: {}'.format(time_step))
            fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
                                    gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
            scenario_map.plot(axs[0])
            axs[0].set_aspect('equal', 'box')
            for agent in range(info['n_agents']):
                for df in dfs[agent]:
                    cmap_rew = matplotlib.cm.turbo
                    norm_rew = matplotlib.colors.Normalize(vmin=-7, vmax=2)
                    axs[0].scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
                                   c=df["Step_Reward"][:time_step], vmin=-7, vmax=2,
                                   label='Agent ' + str(agent),
                                   s=2,
                                   cmap=plt.get_cmap("turbo"),
                                   alpha=0.1)
                    axs[0].set_xlabel(r'x $[m]$')
                    axs[0].set_ylabel(r'y $[m]$')
                    axs[0].set_aspect('equal', 'box')
                    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_rew, cmap=cmap_rew),
                                 cax=axs[1], orientation='vertical', label=r"step reward")
            plt.savefig(Path(plots_path, '{:04d}.png'.format(time_step)))
            plt.close('all')

    if coloring == "acceleration":
        for time_step in range(int(info['max_episode_length'])):
            print('Current frame: {}'.format(time_step))
            fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
                                    gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
            scenario_map.plot(axs[0])
            axs[0].set_aspect('equal', 'box')
            for agent in range(info['n_agents']):
                for df in dfs[agent]:
                    cmap_acc = matplotlib.cm.turbo
                    norm_acc = matplotlib.colors.Normalize(vmin=info['min_acceleration'], vmax=10)
                    axs[0].scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
                                   c=df["Acceleration"][:time_step], vmin=info['min_acceleration'], vmax=10,
                                   label='Agent ' + str(agent),
                                   s=2,
                                   cmap=plt.get_cmap("turbo"),
                                   alpha=0.1)
                    axs[0].set_xlabel(r'x $[m]$')
                    axs[0].set_ylabel(r'y $[m]$')
                    axs[0].set_aspect('equal', 'box')
                    # axs[0].set_title("Acceleration Distribution")
                    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_acc, cmap=cmap_acc),
                                 cax=axs[1], orientation='vertical', label=r"acceleration $[m/s^2]$")
            plt.savefig(Path(plots_path, '{:04d}.png'.format(time_step)))
            plt.close('all')

    if coloring == "control_input":
        for time_step in range(int(info['max_episode_length'])):
            print('Current frame: {}'.format(time_step))
            fig, ax = plt.subplots(figsize=aspect_ratio, tight_layout=True)
            scenario_map.plot(ax)
            ax.set_aspect('equal', 'box')
            for agent in range(info['n_agents']):
                for df in dfs[agent]:
                    # for ts in range(time_step):
                    ax.scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
                               c=[ACTION_COLORS[int(x)] for x in df['Operations'][:time_step]],
                               s=2,
                               alpha=0.1)
            for action in range(4):
                ax.scatter(1e5, 1e5, c=ACTION_COLORS[action], label=ACTIONS[action])
            ax.set_xlabel(r'x $[m]$')
            ax.set_ylabel(r'y $[m]$')
            plt.legend(loc='upper right')
            plt.savefig(Path(plots_path, '{:04d}.png'.format(time_step)))
            plt.close('all')

    if coloring == "cost_com":
        for time_step in range(int(info['max_episode_length'])):
            print('Current frame: {}'.format(time_step))
            fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
                                    gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
            scenario_map.plot(axs[0])
            axs[0].set_aspect('equal', 'box')
            for agent in range(info['n_agents']):
                for df in dfs[agent]:
                    norm_cc = matplotlib.colors.Normalize(vmin=info['min_cost_com'], vmax=info['max_cost_com'])
                    axs[0].scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
                                   c=df["cost_com"][:time_step], vmin=info['min_cost_com'], vmax=info['max_cost_com'],
                                   label='Agent ' + str(agent),
                                   s=2,
                                   cmap=cmap_green_red,
                                   alpha=0.1)
                    axs[0].set_xlabel(r'x $[m]$')
                    axs[0].set_ylabel(r'y $[m]$')
                    axs[0].set_aspect('equal', 'box')
                    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_cc, cmap=cmap_green_red),
                                 cax=axs[1], orientation='vertical', label=r"clearance cost")
            plt.savefig(Path(plots_path, '{:04d}.png'.format(time_step)))
            plt.close('all')

    if coloring == "cost_per_acceleration":
        for time_step in range(int(info['max_episode_length'])):
            print('Current frame: {}'.format(time_step))
            fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
                                    gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
            scenario_map.plot(axs[0])
            axs[0].set_aspect('equal', 'box')
            for agent in range(info['n_agents']):
                for df in dfs[agent]:
                    norm_pa = matplotlib.colors.Normalize(vmin=info['min_cost_per_acceleration'],
                                                          vmax=info['max_cost_per_acceleration'])
                    axs[0].scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
                                   c=df["cost_per_acceleration"][:time_step],
                                   vmin=info['min_cost_per_acceleration'], vmax=info['max_cost_per_acceleration'],
                                   label='Agent ' + str(agent),
                                   s=2,
                                   cmap=cmap_green_red,
                                   alpha=0.1)
                    axs[0].set_xlabel(r'x $[m]$')
                    axs[0].set_ylabel(r'y $[m]$')
                    axs[0].set_aspect('equal', 'box')
                    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_pa, cmap=cmap_green_red),
                                 cax=axs[1], orientation='vertical', label=r"personal acceleration cost")
            plt.savefig(Path(plots_path, '{:04d}.png'.format(time_step)))
            plt.close('all')

    if coloring == "goal_improvement_reward":
        for time_step in range(int(info['max_episode_length'])):
            print('Current frame: {}'.format(time_step))
            fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
                                    gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
            scenario_map.plot(axs[0])
            axs[0].set_aspect('equal', 'box')
            for agent in range(info['n_agents']):
                for df in dfs[agent]:
                    norm_gir = matplotlib.colors.Normalize(vmin=info['min_goal_improvement_reward'],
                                                           vmax=info['max_goal_improvement_reward'])
                    axs[0].scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
                                   c=df["goal_improvement_reward"][:time_step],
                                   vmin=info['min_goal_improvement_reward'], vmax=info['max_goal_improvement_reward'],
                                   label='Agent ' + str(agent),
                                   s=2,
                                   cmap=cmap_red_green,
                                   alpha=0.1)
                    axs[0].set_xlabel(r'x $[m]$')
                    axs[0].set_ylabel(r'y $[m]$')
                    axs[0].set_aspect('equal', 'box')
                    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_gir, cmap=cmap_red_green),
                                 cax=axs[1], orientation='vertical', label=r"goal improvement reward")
            plt.savefig(Path(plots_path, '{:04d}.png'.format(time_step)))
            plt.close('all')


    datetime = strftime("%Y%m%d_%H%M%S_", gmtime())
    make_video(tmp_path=plots_path,
               save_path=Path(checkpoint_path, 'plots', '{}'.format(datetime) + 'pos_{}_video.mp4'.format(coloring)),
               fps=20)


def animate(checkpoint_path: Union[str, Path], scenario_map: Map) -> None:
    dot_size = 2
    alpha = 0.1

    plots_path = Path(checkpoint_path, 'plots', 'tmp')
    plots_path.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams['savefig.dpi'] = 100

    start = timer()

    info = get_info(checkpoint_path)
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

        fig, axs = plt.subplots(3, 2, figsize=(15, 3 * 4), tight_layout=True, gridspec_kw={'width_ratios': [40, 1]})
        # plot_map(ax)
        # ax.set_aspect('equal', 'box')
        for agent in range(info['n_agents']):
            start1 = timer()
            for df in dfs[agent]:
                # speed plot
                scenario_map.plot(axs[0, 0])
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
                scenario_map.plot(axs[1, 0])
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
                scenario_map.plot(axs[2, 0])
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


# TODO: density plots for velocity, acceleration, etc.
def density_plot(checkpoint_path: Union[str, Path],
                 info: dict,
                 save_path: Union[str, Path] = None,
                 dpi: int = 100,
                 ) -> None:
    matplotlib.rcParams['savefig.dpi'] = dpi

    if save_path is None:
        save_path = Path(checkpoint_path, 'plots')

    save_path.mkdir(parents=True, exist_ok=True)

    dfs, masks = load_checkpoint_dfs(checkpoint_path, info)

    speeds = {}
    accelerations = {}

    density_speed = []
    density_acceleration = []
    for agent in range(info["n_agents"]):
        speeds[agent] = []
        accelerations[agent] = []
        for df in dfs[agent]:
            # list concatenation
            speeds[agent] += list(df["Speed"])
            accelerations[agent] += list(df["Acceleration"])
        density_speed.append(gaussian_kde(speeds[agent]))
        density_acceleration.append(gaussian_kde(accelerations[agent]))

    x_speed = np.arange(0, 16, .02)
    x_acceleration = np.arange(0, 20, .04)

    fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
    for agent in range(info["n_agents"]):
        ax.plot(x_speed, density_speed[agent](x_speed), label='agent {}'.format(agent), color=AGENT_COLORS[agent])
    plt.legend()
    plt.xlabel(r"speed $[m/s]$")
    plt.title("Speed Density")
    plt.savefig(Path(save_path, "speed_density.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
    for agent in range(info["n_agents"]):
        ax.plot(x_acceleration, density_acceleration[agent](x_acceleration),
                label='agent {}'.format(agent),
                color=AGENT_COLORS[agent])
    plt.legend()
    plt.xlabel(r"acceleration $[m/s^2]$")
    plt.title("Acceleration Density")
    plt.savefig(Path(save_path, "acceleration_density.png"), dpi=200)
    plt.close(fig)

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
#      min_y_pos, max_y_pos) = info(checkpoint_path)
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


# def plot_positions(checkpoint_path):
#     datetime = strftime("%Y%m%d_%H%M%S_", gmtime())
#     matplotlib.rcParams['savefig.dpi'] = 800
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
#                 # ax.plot(df['Xpos'], df['Ypos'],
#                 #         color=AGENT_COLORS[agent], label='Agent ' + str(agent))
#                 ax.scatter(df['Xpos'], df['Ypos'],
#                            c=df["Speed"] / 16, label='Agent ' + str(agent), s=2, cmap=plt.get_cmap("turbo"), alpha=0.1)
#
#     plots_path = Path(checkpoint_path, 'plots')
#     plots_path.mkdir(parents=True, exist_ok=True)
#     plt.savefig(Path(plots_path, '{}agent_positions.png'.format(datetime)))


# def plot_map(ax):
#     lw = 1
#     ax.plot([0, 51.3], [10, 0], color='k', linewidth=lw)
#     ax.plot([-1.2, 33.5], [3.8, -3.3], color='k', linewidth=lw)
#     ax.plot([0, 33.5], [-10, -3.3], color='k', linewidth=lw)
#     ax.plot([1.2, 51.3], [-16.3, -6.2], color='k', linewidth=lw)
#     ax.plot([0, 51.3], [10, 0], color='k', linewidth=lw)
#     ax.plot([51.3, 110], [0, 0], color='k', linewidth=lw)
#     ax.plot([51.3, 110], [-6.2, -6.2], color='k', linewidth=lw)
#     ax.plot([-0.6, 32.5], [7, 0.4], color='grey', linewidth=lw, linestyle='--')
#     ax.plot([0.6, 32.5], [-13.2, -6.7], color='grey', linewidth=lw, linestyle='--')
#     ax.plot([51.3, 110], [-3.1, -3.1], color='grey', linewidth=lw, linestyle='--')
#
#     ax.set_xlim([-5, 115])
#     ax.set_ylim([-17, 11])

# def animate_positions(checkpoint_path):
#     plots_path = Path(checkpoint_path, 'plots', 'tmp')
#     plots_path.mkdir(parents=True, exist_ok=True)
#
#     matplotlib.rcParams['savefig.dpi'] = 100
#
#     info = get_info(checkpoint_path)
#     print(info)
#
#     dfs = {}
#     for agent in range(info['n_agents']):
#         dfs[agent] = []
#
#     times = os.listdir(Path(checkpoint_path))
#     for time in times:
#         if time == "plots":
#             continue
#         episodes = os.listdir(Path(checkpoint_path, time))
#         for episode in episodes:
#             for agent in range(info['n_agents']):
#                 csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
#                 dfs[agent].append(pd.read_csv(csv_path, index_col=0, header=None).T)
#
#     times = os.listdir(Path(checkpoint_path))
#     for time_step in range(int(info['max_episode_length'])):
#         print('Current frame: {}'.format(time_step))
#         fig, axs = plt.subplots(1, 2, figsize=(15, 4), tight_layout=True, gridspec_kw={'width_ratios': [40, 1]})
#         plot_map(axs[0])
#         axs[0].set_aspect('equal', 'box')
#         for agent in range(info['n_agents']):
#             for df in dfs[agent]:
#                 axs[0].scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
#                                c=df["Speed"][:time_step], vmin=info['min_speed'], vmax=info['max_speed'],
#                                label='Agent ' + str(agent),
#                                s=2,
#                                cmap=plt.get_cmap("turbo"),
#                                alpha=0.1)
#         axs[0].set_xlim([-5, 115])
#         axs[0].set_ylim([-17, 11])
#         axs[0].set_xlabel(r'x $[m]$')
#         axs[0].set_ylabel(r'y $[m]$')
#         cmap_speed = matplotlib.cm.turbo
#         norm_speed = matplotlib.colors.Normalize(vmin=info['min_speed'], vmax=info['max_speed'])
#         fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_speed, cmap=cmap_speed),
#                      cax=axs[1], orientation='vertical', label=r'speed $[m/s]$')
#         plt.savefig(Path(plots_path, '{:04d}.png'.format(time_step)))
#         plt.close('all')
#
#     datetime = strftime("%Y%m%d_%H%M%S_", gmtime())
#     make_video(plots_path, Path(checkpoint_path, 'plots', '{}'.format(datetime) + 'pos_video.mp4'), 20)


# def animate_positions_agents(checkpoint_path):
#     plots_path = Path(checkpoint_path, 'plots', 'tmp')
#     plots_path.mkdir(parents=True, exist_ok=True)
#
#     matplotlib.rcParams['savefig.dpi'] = 100
#
#     info = get_info(checkpoint_path)
#     print(info)
#
#     dfs = {}
#     for agent in range(info['n_agents']):
#         dfs[agent] = []
#
#     times = os.listdir(Path(checkpoint_path))
#     for time in times:
#         if time == "plots":
#             continue
#         episodes = os.listdir(Path(checkpoint_path, time))
#         for episode in episodes:
#             for agent in range(info['n_agents']):
#                 csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
#                 dfs[agent].append(pd.read_csv(csv_path, index_col=0, header=None).T)
#
#     times = os.listdir(Path(checkpoint_path))
#     for time_step in range(int(info['max_episode_length'])):
#         print('Current frame: {}'.format(time_step))
#         fig, ax = plt.subplots(figsize=(15, 4), tight_layout=True)
#         plot_map(ax)
#         ax.set_aspect('equal', 'box')
#         for agent in range(info['n_agents']):
#             for df in dfs[agent]:
#                 ax.scatter(df['Xpos'][:time_step], df['Ypos'][:time_step],
#                            c=AGENT_COLORS[agent],
#                            label='Agent ' + str(agent),
#                            s=2,
#                            alpha=0.1)
#         ax.set_xlim([-5, 115])
#         ax.set_ylim([-17, 11])
#         ax.set_xlabel(r'x $[m]$')
#         ax.set_ylabel(r'y $[m]$')
#         # plt.legend()
#         plt.savefig(Path(plots_path, '{:04d}.png'.format(time_step)))
#         plt.close('all')
#
#     datetime = strftime("%Y%m%d_%H%M%S_", gmtime())
#     make_video(plots_path, Path(checkpoint_path, 'plots', '{}'.format(datetime) + 'pos_agents_video.mp4'), 20)
