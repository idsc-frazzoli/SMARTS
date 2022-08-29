from scenarios import *

import argparse
from pathlib import Path
import pandas as pd
from time import gmtime, strftime
import os

from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils import get_info, load_checkpoint_dfs
from matplotlib.colors import LinearSegmentedColormap

import subprocess

LARGESIZE, MEDIUMSIZE, SMALLSIZE = 16, 13, 10

#                 yellow      red       green      blue
AGENT_COLORS = ["#F9D923", "#EB5353", "#00C897", "#2155CD"]



# AGENT_COLORS = ["#F9D923", "#00C897", "#2155CD", "#EB5353"]


def get_corner_coordinates(x, y, alpha, w, l):
    if alpha < 0:
        alpha += np.pi
    x_corners_nonrot = [l / 2, l / 2, -l / 2, -l / 2]
    y_corners_nonrot = [w / 2, -w / 2, -w / 2, +w / 2]
    x_corners, y_corners = [], []
    for i in range(len(x_corners_nonrot)):
        dx = np.cos(alpha) * x_corners_nonrot[i] + np.sin(alpha) * y_corners_nonrot[i]
        dy = np.sin(alpha) * x_corners_nonrot[i] - np.cos(alpha) * y_corners_nonrot[i]
        x_corners.append(x + dx)
        y_corners.append(y + dy)
    return x_corners, y_corners


def position_plots(checkpoint_path, scenario_map):

    name = checkpoint_path.split("/")[2]
    # subprocess.check_call(["latex"])
    plots_path = Path(checkpoint_path, 'plots', 'report_plots')
    plots_path.mkdir(parents=True, exist_ok=True)

    dpi = 400
    matplotlib.rcParams['savefig.dpi'] = dpi
    matplotlib.rcParams["text.usetex"] = True

    plt.rcParams.update({'font.size': LARGESIZE})
    plt.rcParams.update({'axes.titlesize': LARGESIZE})
    plt.rcParams.update({'axes.labelsize': 22})
    plt.rcParams.update({'xtick.labelsize': 18})
    plt.rcParams.update({'ytick.labelsize': 18})
    plt.rcParams.update({'legend.fontsize': 20})
    plt.rcParams.update({'figure.titlesize': LARGESIZE})

    plt.rcParams.update({'font.size': LARGESIZE})
    plt.rcParams.update({'axes.titlesize': LARGESIZE})
    plt.rcParams.update({'axes.labelsize': 28})
    plt.rcParams.update({'xtick.labelsize': 28})
    plt.rcParams.update({'ytick.labelsize': 28})
    plt.rcParams.update({'legend.fontsize': 31})
    plt.rcParams.update({'figure.titlesize': LARGESIZE})

    info = get_info(checkpoint_path)

    print(info["max_episode_length"])

    min_datapoints = 10

    dfs, masks = load_checkpoint_dfs(checkpoint_path, info)

    x_positions = {agent: {ts: [] for ts in range(info["max_episode_length"])} for agent in dfs}
    y_positions = {agent: {ts: [] for ts in range(info["max_episode_length"])} for agent in dfs}

    for agent in dfs:
        for episode in range(len(dfs[agent])):
            if not masks["goal_reached_mask"]:
                continue
            for time_step in range(len(dfs[agent][episode]["Xpos"])):
                x_positions[agent][time_step].append(dfs[agent][episode]["Xpos"][time_step + 1])
                y_positions[agent][time_step].append(dfs[agent][episode]["Ypos"][time_step + 1])

    x_means = {agent: {ts: np.mean(x_positions[agent][ts]) for ts in range(info["max_episode_length"])} for agent in
               dfs}
    y_means = {agent: {ts: np.mean(y_positions[agent][ts]) for ts in range(info["max_episode_length"])} for agent in
               dfs}

    x_lens = {agent: {ts: len(x_positions[agent][ts]) for ts in range(info["max_episode_length"])} for agent in
              dfs}
    y_lens = {agent: {ts: len(y_positions[agent][ts]) for ts in range(info["max_episode_length"])} for agent in
              dfs}

    dx_dy = (scenario_map.x_lim[1] - scenario_map.x_lim[0]) / (scenario_map.y_lim[1] - scenario_map.y_lim[0])
    aspect_ratio = scenario_map.aspect_ratio
    fig, ax = plt.subplots(1, 1, figsize=aspect_ratio, tight_layout=True)

    # paint the grass
    ax.fill([100, 100, -100, -100], [100, -100, -100, 100], color="green", alpha=0.2)

    scenario_map.plot(ax)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel(r'$x \ [\textrm{m}]$')
    ax.set_ylabel(r'$y \ [\textrm{m}]$')

    time_step_split = 27

    for agent in x_means:
        x = [x_means[agent][ts] for ts in range(len(x_means[agent])) if x_lens[agent][ts] > min_datapoints]
        y = [y_means[agent][ts] for ts in range(len(y_means[agent])) if y_lens[agent][ts] > min_datapoints]
        x1, x2 = x[:min(len(x), time_step_split)], x[min(len(x), time_step_split):]
        y1, y2 = y[:min(len(y), time_step_split)], y[min(len(y), time_step_split):]
        ax.plot(x1, y1, color=AGENT_COLORS[agent], linestyle="-", linewidth=3.5)
        ax.plot(x2, y2, color=AGENT_COLORS[agent], linestyle="--", linewidth=3.5)

        dx = x[min(time_step_split, len(x) - 1)] - x[min(time_step_split, len(x) - 1) - 1]
        dy = y[min(time_step_split, len(y) - 1)] - y[min(time_step_split, len(y) - 1) - 1]
        alpha = np.arctan2(dy, dx)
        x_corners, y_corners = get_corner_coordinates(x[min(time_step_split, len(x) - 1)],
                                                      y[min(time_step_split, len(x) - 1)], alpha, 1.3, 2.5)
        ax.fill(x_corners, y_corners, color=AGENT_COLORS[agent])

        ax.scatter(x[-1], y[-1], s=150, color=AGENT_COLORS[agent], marker="x", linewidths=3)

    for agent in x_positions:
        for ts in x_positions[agent]:
            if not x_lens[agent][ts] > min_datapoints:
                continue
            ax.scatter(x_positions[agent][ts], y_positions[agent][ts],
                       color=AGENT_COLORS[agent], alpha=.025, marker='.', s=2)

    # intersection colors
    ax.scatter(100, 100, s=200, color=AGENT_COLORS[0], label=r'$\textrm{Agent 1}$')
    ax.scatter(100, 100, s=200, color=AGENT_COLORS[3], label=r'$\textrm{Agent 2}$')
    ax.scatter(100, 100, s=200, color=AGENT_COLORS[1], label=r'$\textrm{Agent 3}$')
    ax.scatter(100, 100, s=200, color=AGENT_COLORS[2], label=r'$\textrm{Agent 4}$')

    # # merge colors
    # ax.scatter(100, 100, s=200, color=AGENT_COLORS[0], label=r'$\textrm{Agent 1}$')
    # ax.scatter(100, 100, s=200, color=AGENT_COLORS[2], label=r'$\textrm{Agent 2}$')
    # ax.scatter(100, 100, s=200, color=AGENT_COLORS[3], label=r'$\textrm{Agent 3}$')
    # ax.scatter(100, 100, s=200, color=AGENT_COLORS[1], label=r'$\textrm{Agent 4}$')


    plt.legend(loc='upper right', ncol=1)
    plt.savefig('data/plots/{}_ts27.png'.format(name), dpi=dpi)
    # plt.savefig('data/plots/{}_2_ts27.pdf'.format(name))

def main(
        checkpoint_path,
        scenario_name
):
    if scenario_name == "merge110_lanes2":
        scenario_map = get_merge110_lanes2()
    elif scenario_name in ["merge40_lanes1", "merge40_lanes1_2", "merge40_lanes1_3", "merge40_lanes1_4"]:
        scenario_map = get_merge40_lanes1()
    elif scenario_name == "merge75_lanes321":
        scenario_map = get_merge75_lanes321()
    elif scenario_name == "merge90_lanes32":
        scenario_map = get_merge90_lanes32()
    elif scenario_name in ["merge65_lanes42", "merge65_lanes42_asym", "merge_2", "merge_3", "merge_4"]:
        scenario_map = get_merge65_lanes42()
    elif scenario_name == "straight_merge90_lanes7":
        scenario_map = get_straight_merge90_lanes7()
    elif scenario_name in ["int_4", "int_3", "int_2", "int_4_rand2"]:
        scenario_map = get_int_4()
    elif scenario_name in ["int_4_short", "int_3_short", "int_2_short", "int_4_rand2_short"]:
        scenario_map = get_int_4_short()
    else:
        scenario_map = get_empty()

    position_plots(checkpoint_path, scenario_map)


def parse_args():
    parser = argparse.ArgumentParser("")

    parser.add_argument('-p',
                        '--checkpoint_path',
                        type=str,
                        help='Path to the checkpoint for plotting.',
                        required=True)

    parser.add_argument('-n',
                        '--scenario_name',
                        type=str,
                        default="",
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        checkpoint_path=args.checkpoint_path,
        scenario_name=args.scenario_name
    )

# if __name__ == "__main__":
#     main(
#         checkpoint_path="./data/int_4_feae4/test",
#         scenario_name="int_4_short"
#     )
