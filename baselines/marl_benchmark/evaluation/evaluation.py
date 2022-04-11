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

from utils import get_min_max, plot_positions, plot_positions1, animate_positions, animate

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





def min_max_reward_filtered(paths: List[str]) -> Tuple[float, float]:
    min_reward: float = 1e10
    max_reward: float = -1e10
    for path in paths:
        checkpoints = os.listdir(Path(path))
        for checkpoint in checkpoints:
            times = os.listdir(Path(path, checkpoint))
            for time in times:
                if time == "plots":
                    continue
                episodes = os.listdir(Path(path, checkpoint, time))
                for episode in episodes:
                    print(episode)
                    n_agents = len(os.listdir(Path(path, checkpoint, time, episode)))
                    for agent in range(n_agents):
                        csv_path = Path(path, checkpoint, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
                        df = pd.read_csv(csv_path, index_col=0, header=None).T
                        if df["Num_Collision"][1] < 1e-5 and df["Num_Off_Road"][1] < 1e-5:
                            min_reward = min(min_reward, sum(df["Step_Reward"]))
                            max_reward = max(max_reward, sum(df["Step_Reward"]))

    return min_reward, max_reward

def get_mean_reward_checkpoint_filtered(checkpoint_path: str) -> float:
    times = os.listdir(Path(checkpoint_path))
    n = 0
    reward = 0
    for time in times:
        if time == "plots":
            continue
        episodes = os.listdir(Path(checkpoint_path, time))
        for episode in episodes:
            print(episode)
            n_agents = len(os.listdir(Path(checkpoint_path, time, episode)))
            for agent in range(n_agents):
                csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
                df = pd.read_csv(csv_path, index_col=0, header=None).T
                if df["Num_Collision"][1] < 1e-5 and df["Num_Off_Road"][1] < 1e-5:
                    reward += sum(df["Step_Reward"])
                    n += 1
    reward /= n
    return reward


def price_of_anarchy(cent_path: str, decent_path: str, cent_checkpoint: int, decent_checkpoint: int) -> float:
    paths = [cent_path, decent_path]
    min_reward, max_reward = min_max_reward_filtered(paths)
    # cent_dirs = [x[0] for x in os.walk(cent_path)]
    # decent_dirs = [x[0] for x in os.walk(decent_path)]
    cent_cps = os.listdir(Path(cent_path))
    decent_cps = os.listdir(Path(decent_path))

    cent_reward, decent_reward = None, None

    for cent_cp in cent_cps:
        if cent_cp == "checkpoint_{:06d}".format(cent_checkpoint):
            cent_reward = get_mean_reward_checkpoint_filtered(cent_path + "/" + cent_cp)

    for decent_cp in decent_cps:
        if decent_cp == "checkpoint_{:06d}".format(decent_checkpoint):
            decent_reward = get_mean_reward_checkpoint_filtered(decent_path + "/" + decent_cp)

    assert cent_reward is not None, "Centralized checkpoint does not exist."
    assert decent_reward is not None, "Decentralized checkpoint does not exist."

    cent_reward_norm = (cent_reward - min_reward) / (max_reward - min_reward)
    decent_reward_norm = (decent_reward - min_reward) / (max_reward - min_reward)

    poa = cent_reward_norm / decent_reward_norm

    return poa




def main(
        paths,
):
    print(paths)

    cent_path = paths[0]
    decent_path = paths[1]

    # checkpoint_path = Path(paths[0], 'checkpoint_000560')
    # limits = get_min_max(checkpoint_path)
    # plot_positions1(checkpoint_path, limits)







    # # cent_cp = 340
    # # decent_cp = 560
    # cent_cp = 400
    # decent_cp = 400
    #
    # poa = price_of_anarchy(cent_path, decent_path, cent_cp, decent_cp)
    #
    # print(poa)



    checkpoint_path = Path(paths[0], 'checkpoint_000300')
    # animate_positions(checkpoint_path)
    animate(checkpoint_path)

    # for path in paths:
    #     checkpoints = os.listdir(Path(path))
    #     for checkpoint in checkpoints:
    #         print(checkpoint)
    #         checkpoint_path = Path(path, checkpoint)
    #         plot_positions(checkpoint_path)

            # times = os.listdir(Path(path, checkpoint))
            # for time in times:
            #     episodes = os.listdir(Path(path, checkpoint, time))
            #     for episode in episodes:
            #         n_agents = len(os.listdir(Path(path, checkpoint, time, episode)))
            #         for agent in range(n_agents):
            #             csv_path = Path(path, checkpoint, time, episode, 'agent_AGENT-{}'.format(str(agent)))
            #             df = pd.read_csv(csv_path)



def parse_args():
    parser = argparse.ArgumentParser("Learning Metrics Plotter")

    parser.add_argument('-p',
                        '--paths',
                        nargs='+',
                        help='Paths of evaluations files',
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        paths=args.paths,
    )
