import argparse
from pathlib import Path
import pandas as pd
from time import gmtime, strftime
import os

from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import shutil

from matplotlib.patches import Ellipse

from utils import load_checkpoint_dfs, get_rewards, get_lengths

NE_COLORS = [{"cent": "#DE3163", "decent": "#FF7F50"},
             {"cent": "#6495ED", "decent": "#40E0D0"},
             {"cent": "#FFBF00", "decent": "#DFFF00"},
             {"cent": "#CCCCFF", "decent": "#9FE2BF"}
             ]


# def extract_rew_len(checkpoint_path):
#     dfs, masks = load_checkpoint_dfs(checkpoint_path)
#     rewards, filtered_rewards = get_rewards(dfs, masks, goal_reached_reward=300)
#     lengths, filtered_lengths = get_lengths(dfs, masks)
#
#     return


def main(
        paths_cent,
        paths_decent,
):
    # TODO: clean up this code

    # clean this up later
    info = {'n_agents': 2}

    # ne1_paths_cent = os.listdir(Path(ne1_path_to_cent))
    # ne1_paths_cent = [ne1_path_to_cent + '/' + x for x in ne1_paths_cent]

    rewards_cent = dict.fromkeys([x.split('_')[-5] for x in paths_cent], [])
    rewards_decent = dict.fromkeys([x.split('_')[-5] for x in paths_decent], [])

    lengths_cent = dict.fromkeys([x.split('_')[-5] for x in paths_cent], [])
    lengths_decent = dict.fromkeys([x.split('_')[-5] for x in paths_decent], [])

    for path in paths_cent:
        checkpoints = os.listdir(Path(path))
        for checkpoint in checkpoints:
            checkpoint_path = Path(path, checkpoint)
            dfs, masks = load_checkpoint_dfs(checkpoint_path, info)
            _, rewards_cent[path.split('_')[-5]] = get_rewards(dfs, masks, goal_reached_reward=300)
            _, lengths_cent[path.split('_')[-5]] = get_lengths(dfs, masks)
            # print(lengths_ne1_cent[path.split('_')[-5]])

    for path in paths_decent:
        checkpoints = os.listdir(Path(path))
        for checkpoint in checkpoints:
            checkpoint_path = Path(path, checkpoint)
            dfs, masks = load_checkpoint_dfs(checkpoint_path, info)
            _, rewards_decent[path.split('_')[-5]] = get_rewards(dfs, masks, goal_reached_reward=300)
            _, lengths_decent[path.split('_')[-5]] = get_lengths(dfs, masks)

    fig, ax = plt.subplots(figsize=(12, 8), tight_layout=True)
    for key in rewards_cent:
        print(key)
        mean_rew = np.mean(rewards_cent[key])
        rew_std = np.std(rewards_cent[key])
        mean_len = np.mean(lengths_cent[key])
        len_std = np.std(lengths_cent[key])
        ax.scatter(mean_len, mean_rew, color=NE_COLORS[0]['cent'])
        ax.text(mean_len, mean_rew, key)
        e = Ellipse(xy=(mean_len, mean_rew),
                    width=len_std, height=rew_std,
                    color=NE_COLORS[0]['cent'], alpha=0.2)
        ax.add_artist(e)
    ax.scatter(1e3, 1e3, color=NE_COLORS[0]['cent'], label="centralized")

    for key in rewards_decent:
        print(key)
        mean_rew = np.mean(rewards_decent[key])
        rew_std = np.std(rewards_decent[key])
        mean_len = np.mean(lengths_decent[key])
        len_std = np.std(lengths_decent[key])
        ax.scatter(mean_len, mean_rew, color=NE_COLORS[1]['decent'])
        ax.text(mean_len, mean_rew, key)
        e = Ellipse(xy=(mean_len, mean_rew),
                    width=len_std, height=rew_std,
                    color=NE_COLORS[1]['decent'], alpha=0.2)
        ax.add_artist(e)
    ax.scatter(1e3, 1e3, color=NE_COLORS[1]['decent'], label="decentralized")

    plt.legend()
    plt.xlabel("episode length")
    plt.ylabel("episode reward")
    plt.ylim([-3000, 500])
    plt.xlim([20, 90])
    # plt.savefig("test_20220512.png", dpi=300)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser("Learning Metrics Plotter")

    parser.add_argument('--paths_cent',
                        nargs='+',
                        help='Paths of evaluations files',
                        required=True)

    parser.add_argument('--paths_decent',
                        nargs='+',
                        help='Paths of evaluations files',
                        required=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        paths_cent=args.paths_cent,
        paths_decent=args.paths_decent,
    )
