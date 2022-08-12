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

ACTION_COLORS = ["#F9D923", "#EB5353", "#00C897", "#2155CD"]
ACTIONS = {0: r'$\textrm{keep lane}$', 1: r'$\textrm{slow down}$', 2: r'$\textrm{change lane left}$',
           3: r'$\textrm{change lane right}$'}


def position_plots(checkpoint_path, scenario_map, n_timesteps):
    cmap_green_red = LinearSegmentedColormap.from_list('rg', ["#008000", "#00FF00", "#ffff00", "#FF0000", "#800000"],
                                                       N=256)
    cmap_red_green = LinearSegmentedColormap.from_list('rg', ["#800000", "#FF0000", "#ffff00", "#00FF00", "#008000"],
                                                       N=256)

    # subprocess.check_call(["latex"])
    plots_path = Path(checkpoint_path, 'plots', 'report_plots')
    plots_path.mkdir(parents=True, exist_ok=True)

    dpi = 100
    matplotlib.rcParams['savefig.dpi'] = dpi
    matplotlib.rcParams["text.usetex"] = True

    plt.rcParams.update({'font.size': LARGESIZE})
    plt.rcParams.update({'axes.titlesize': LARGESIZE})
    plt.rcParams.update({'axes.labelsize': 22})
    plt.rcParams.update({'xtick.labelsize': 18})
    plt.rcParams.update({'ytick.labelsize': 18})
    plt.rcParams.update({'legend.fontsize': 20})
    plt.rcParams.update({'figure.titlesize': LARGESIZE})

    info = get_info(checkpoint_path)

    print(info["max_episode_length"])

    timesteps_to_evaluate = []
    for i in range(1, n_timesteps):
        timesteps_to_evaluate.append(int(np.round(i * info["max_episode_length"] / n_timesteps) - 1))
    timesteps_to_evaluate.append(info["max_episode_length"] - 1)

    dfs, _ = load_checkpoint_dfs(checkpoint_path, info)

    if scenario_map.empty:
        dx_dy = 5
    else:
        dx_dy = (scenario_map.x_lim[1] - scenario_map.x_lim[0]) / (scenario_map.y_lim[1] - scenario_map.y_lim[0])
    aspect_ratio = scenario_map.aspect_ratio

    save_path = Path(plots_path, 'speed_plots')
    save_path.mkdir(parents=True, exist_ok=True)
    # for k in timesteps_to_evaluate:
    #     fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
    #                             gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
    #     scenario_map.plot(axs[0])
    #     axs[0].set_aspect('equal', 'box')
    #     for agent in range(info['n_agents']):
    #         for df in dfs[agent]:
    #             axs[0].scatter(df['Xpos'][:k], df['Ypos'][:k],
    #                            c=df["Speed"][:k], vmin=info['min_speed'], vmax=info['max_speed'],
    #                            label='Agent ' + str(agent),
    #                            s=2,
    #                            cmap=plt.get_cmap("turbo"),
    #                            alpha=0.1)
    #     axs[0].set_xlabel(r'$x \ [m]$')
    #     axs[0].set_ylabel(r'$y \ [m]$')
    #     axs[0].text(50, 10, r"$\textrm{time step }$"+r"$k={}$".format(k), fontsize=24)
    #     cmap_speed = matplotlib.cm.turbo
    #     norm_speed = matplotlib.colors.Normalize(vmin=info['min_speed'], vmax=info['max_speed'])
    #     fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_speed, cmap=cmap_speed),
    #                  cax=axs[1], orientation='vertical', label=r' $\textrm{speed} \ \Big[\frac{m}{s}\Big]$')
    #     plt.savefig(Path(save_path, '{:04d}_{}.png'.format(k, dpi)))
    #     # plt.savefig(Path(save_path, '{:04d}.pdf'.format(k)))
    #     plt.close('all')

    # # colormaps = ["autumn", "cool", "hot", "hsv", "gist_ncar", "gnuplot", "gnuplot2", "jet", "viridis", "plasma"]
    # # cmaps = [matplotlib.cm.autumn ,matplotlib.cm.cool, matplotlib.cm.hot, matplotlib.cm.hsv, matplotlib.cm.gist_ncar,
    # #          matplotlib.cm.gnuplot, matplotlib.cm.gnuplot2, matplotlib.cm.jet, matplotlib.cm.viridis, matplotlib.cm.plasma]

    save_path = Path(plots_path, )
    plots_path.mkdir(parents=True, exist_ok=True)

    # for i, cm in enumerate(colormaps):
    fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
                            gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
    scenario_map.plot(axs[0])
    axs[0].set_aspect('equal', 'box')
    for agent in range(info['n_agents']):
        for df in dfs[agent]:
            axs[0].scatter(df['Xpos'], df['Ypos'],
                           c=list(range(len(df["Xpos"]))), vmin=0, vmax=80, #vmax=info["max_episode_length"],
                           label='Agent ' + str(agent),
                           s=2,
                           cmap=plt.get_cmap("gnuplot2"),
                           alpha=0.1)
    axs[0].set_xlabel(r'$x \ [m]$')
    axs[0].set_ylabel(r'$y \ [m]$')
    cmap = matplotlib.cm.gnuplot2
    norm = matplotlib.colors.Normalize(vmin=0, vmax=80)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                 cax=axs[1], orientation='vertical', label=r' $\textrm{time step}$')
    plt.savefig(Path(plots_path, 'time_plot_{}.png'.format(dpi)))
    # plt.savefig(Path(plots_path, 'time_plot.pdf'))
    plt.close('all')
    #
    fig, ax = plt.subplots(figsize=aspect_ratio, tight_layout=True)
    scenario_map.plot(ax)
    ax.set_aspect('equal', 'box')
    for agent in range(info['n_agents']):
        for df in dfs[agent]:
            # for ts in range(time_step):
            ax.scatter(df['Xpos'], df['Ypos'],
                       c=[ACTION_COLORS[int(x)] for x in df['Operations']],
                       s=2,
                       alpha=0.1)
    for action in range(4):
        ax.scatter(1e5, 1e5, c=ACTION_COLORS[action], label=ACTIONS[action])
    ax.set_xlabel(r'$x \ [m]$')
    ax.set_ylabel(r'$y \ [m]$')
    plt.legend(loc='upper right', ncol=2)
    plt.savefig(Path(plots_path, 'actions_plot_{}.png'.format(dpi)))
    # plt.savefig(Path(plots_path, 'actions_plot.pdf'))
    plt.close('all')


    fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
                            gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
    scenario_map.plot(axs[0])
    axs[0].set_aspect('equal', 'box')
    for agent in range(info['n_agents']):
        for df in dfs[agent]:
            axs[0].scatter(df['Xpos'][:], df['Ypos'][:],
                           c=df["Speed"][:], vmin=info['min_speed'], vmax=info['max_speed'],
                           label='Agent ' + str(agent),
                           s=2,
                           cmap=plt.get_cmap("turbo"),
                           alpha=0.1)
    axs[0].set_xlabel(r'$x \ [m]$')
    axs[0].set_ylabel(r'$y \ [m]$')
    cmap_speed = matplotlib.cm.turbo
    norm_speed = matplotlib.colors.Normalize(vmin=info['min_speed'], vmax=info['max_speed'])
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_speed, cmap=cmap_speed),
                 cax=axs[1], orientation='vertical', label=r' $\textrm{speed} \ \Big[\frac{m}{s}\Big]$')
    plt.savefig(Path(save_path, 'speed_plot_{}.png'.format(dpi)))
    # plt.savefig(Path(save_path, 'speed_plot.pdf'))
    plt.close('all')

    # fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
    #                         gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
    # scenario_map.plot(axs[0])
    # axs[0].set_aspect('equal', 'box')
    # for agent in range(info['n_agents']):
    #     for df in dfs[agent]:
    #         norm_cc = matplotlib.colors.Normalize(vmin=info['min_cost_com'], vmax=info['max_cost_com'])
    #         axs[0].scatter(df['Xpos'], df['Ypos'],
    #                        c=df["cost_com"], vmin=info['min_cost_com'], vmax=info['max_cost_com'],
    #                        label='Agent ' + str(agent),
    #                        s=2,
    #                        cmap=cmap_green_red,
    #                        alpha=0.1)
    #         axs[0].set_xlabel(r'$x \ [m]$')
    #         axs[0].set_ylabel(r'$y \ [m]$')
    #         axs[0].set_aspect('equal', 'box')
    #         fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_cc, cmap=cmap_green_red),
    #                      cax=axs[1], orientation='vertical', label=r"$\textrm{clearance cost}$")
    # plt.savefig(Path(plots_path, 'cost_com_plot.png'))
    # plt.savefig(Path(plots_path, 'cost_com_plot.pdf'))
    # plt.close('all')
    #
    # fig, axs = plt.subplots(1, 2, figsize=aspect_ratio, tight_layout=True,
    #                         gridspec_kw={'width_ratios': [int(dx_dy * 10), 1]})
    # scenario_map.plot(axs[0])
    # axs[0].set_aspect('equal', 'box')
    # for agent in range(info['n_agents']):
    #     for df in dfs[agent]:
    #         norm_cc = matplotlib.colors.Normalize(vmin=info['min_cost_per_acceleration'], vmax=info['max_cost_per_acceleration'])
    #         axs[0].scatter(df['Xpos'], df['Ypos'],
    #                        c=df["cost_per_acceleration"], vmin=info['min_cost_per_acceleration'], vmax=info['max_cost_per_acceleration'],
    #                        label='Agent ' + str(agent),
    #                        s=2,
    #                        cmap=cmap_green_red,
    #                        alpha=0.1)
    #         axs[0].set_xlabel(r'$x \ [m]$')
    #         axs[0].set_ylabel(r'$y \ [m]$')
    #         axs[0].set_aspect('equal', 'box')
    #         fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm_cc, cmap=cmap_green_red),
    #                      cax=axs[1], orientation='vertical', label=r"$\textrm{personal acceleration cost}$")
    # plt.savefig(Path(plots_path, 'cost_per_acc_plot.png'))
    # plt.savefig(Path(plots_path, 'cost_per_acc_plot.pdf'))
    # plt.close('all')


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
    elif scenario_name in ["merge65_lanes42", "merge65_lanes42_asym"]:
        scenario_map = get_merge65_lanes42()
    elif scenario_name == "straight_merge90_lanes7":
        scenario_map = get_straight_merge90_lanes7()
    elif scenario_name == "int_4":
        scenario_map = get_int_4()
    else:
        scenario_map = get_empty()

    position_plots(checkpoint_path, scenario_map, n_timesteps=4)


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
