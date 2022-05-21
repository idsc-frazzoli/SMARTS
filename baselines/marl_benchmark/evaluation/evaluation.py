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

from baselines.marl_benchmark.evaluation.utils import animate_positions, animate, plot_positions, get_info, \
    make_video, get_rewards, get_poa, load_checkpoint_dfs, set_box_color, density_plot

from baselines.marl_benchmark.evaluation.scenarios import get_empty, get_merge40_lanes1, get_merge110_lanes2, \
    get_merge75_lanes321, get_merge90_lanes32, get_merge65_lanes42, get_straight_merge90_lanes7

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
    0: '#e41a1c',  # red
    1: '#377eb8',  # blue
    2: '#4daf4a',  # green
    3: '#984ea3',  # purple
    4: '#ff7f00',  # orange

}

PALETTE = ['#A93226', '#CB4335',  # red
           '#884EA0', '#7D3C98',  # purple
           '#2471A3', '#2E86C1',  # blue
           '#17A589', '#138D75',  # blue/green
           '#229954', '#28B463',  # green
           '#D4AC0D', '#D68910',  # yellow
           '#CA6F1E', '#BA4A00',  # orange
           ]


# def min_max_reward_filtered(paths: List[str]) -> Tuple[float, float]:
#     min_reward: float = 1e10
#     max_reward: float = -1e10
#     for path in paths:
#         checkpoints = os.listdir(Path(path))
#         for checkpoint in checkpoints:
#             times = os.listdir(Path(path, checkpoint))
#             for time in times:
#                 if time == "plots":
#                     continue
#                 episodes = os.listdir(Path(path, checkpoint, time))
#                 for episode in episodes:
#                     print(episode)
#                     n_agents = len(os.listdir(Path(path, checkpoint, time, episode)))
#                     for agent in range(n_agents):
#                         csv_path = Path(path, checkpoint, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
#                         df = pd.read_csv(csv_path, index_col=0, header=None).T
#                         if df["Num_Collision"][1] < 1e-5 and df["Num_Off_Road"][1] < 1e-5:
#                             min_reward = min(min_reward, sum(df["Step_Reward"]))
#                             max_reward = max(max_reward, sum(df["Step_Reward"]))
#
#     return min_reward, max_reward
#
#
# def get_mean_reward_checkpoint_filtered(checkpoint_path: str) -> float:
#     times = os.listdir(Path(checkpoint_path))
#     n = 0
#     reward = 0
#     for time in times:
#         if time == "plots":
#             continue
#         episodes = os.listdir(Path(checkpoint_path, time))
#         for episode in episodes:
#             print(episode)
#             n_agents = len(os.listdir(Path(checkpoint_path, time, episode)))
#             for agent in range(n_agents):
#                 csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
#                 df = pd.read_csv(csv_path, index_col=0, header=None).T
#                 if df["Num_Collision"][1] < 1e-5 and df["Num_Off_Road"][1] < 1e-5:
#                     reward += sum(df["Step_Reward"])
#                     n += 1
#     reward /= n
#     return reward
#
#
# def price_of_anarchy(cent_path: str, decent_path: str, cent_checkpoint: int, decent_checkpoint: int) -> float:
#     paths = [cent_path, decent_path]
#     min_reward, max_reward = min_max_reward_filtered(paths)
#     # cent_dirs = [x[0] for x in os.walk(cent_path)]
#     # decent_dirs = [x[0] for x in os.walk(decent_path)]
#     cent_cps = os.listdir(Path(cent_path))
#     decent_cps = os.listdir(Path(decent_path))
#
#     cent_reward, decent_reward = None, None
#
#     for cent_cp in cent_cps:
#         if cent_cp == "checkpoint_{:06d}".format(cent_checkpoint):
#             cent_reward = get_mean_reward_checkpoint_filtered(cent_path + "/" + cent_cp)
#
#     for decent_cp in decent_cps:
#         if decent_cp == "checkpoint_{:06d}".format(decent_checkpoint):
#             decent_reward = get_mean_reward_checkpoint_filtered(decent_path + "/" + decent_cp)
#
#     assert cent_reward is not None, "Centralized checkpoint does not exist."
#     assert decent_reward is not None, "Decentralized checkpoint does not exist."
#
#     cent_reward_norm = (cent_reward - min_reward) / (max_reward - min_reward)
#     decent_reward_norm = (decent_reward - min_reward) / (max_reward - min_reward)
#
#     poa = cent_reward_norm / decent_reward_norm
#
#     return poa


def main(
        paths,
        scenario_name,
        checkpoints=None,
        training_progress_video=False,
        checkpoint_video=False,
        coloring=None,
        poa=False,
        decentralized_cp=None,
        centralized_cp=None,
        density_plots=False,
):
    print(paths)

    cps = []
    if checkpoints is not None:
        cps = ["checkpoint_{:06d}".format(int(x)) for x in checkpoints]

    if not coloring:
        coloring = ["control_input", "speed", "reward", "acceleration", "agents"]

    # for PoA calculations
    dec_cp = []
    if decentralized_cp is not None:
        dec_cp = ["checkpoint_{:06d}".format(int(decentralized_cp))]
    c_cp = []
    if centralized_cp is not None:
        c_cp = ["checkpoint_{:06d}".format(int(centralized_cp))]

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
    else:
        scenario_map = get_empty()

    if training_progress_video:
        for path in paths:
            checkpoints_dir = os.listdir(Path(path))
            for i, checkpoint in enumerate(checkpoints_dir):
                if checkpoint != "videos":
                    print(i)
                    checkpoint_path = Path(path, checkpoint)
                    save_path = Path(path, 'videos/tmp_acc')
                    info = get_info(checkpoint_path)
                    plot_positions(checkpoint_path, info, scenario_map, save_path=save_path, coloring="acceleration")

            save_path = Path(path, 'videos/tmp_acc')
            make_video(save_path, path + "/videos/acc_cp_video.mp4", fps=15, remove_tmp=False)

    if checkpoint_video:
        for path in paths:
            checkpoints_dir = os.listdir(Path(path))
            for i, checkpoint in enumerate(checkpoints_dir):
                if checkpoints is None or checkpoint in cps:
                    print('Current checkpoint: {}'.format(checkpoint))
                    checkpoint_path = Path(path, checkpoint)
                    if "control_input" in coloring:
                        animate_positions(checkpoint_path, scenario_map, "control_input")
                    if "speed" in coloring:
                        animate_positions(checkpoint_path, scenario_map, "speed")
                    if "reward" in coloring:
                        animate_positions(checkpoint_path, scenario_map, "reward")
                    if "acceleration" in coloring:
                        animate_positions(checkpoint_path, scenario_map, "acceleration")
                    if "agents" in coloring:
                        animate_positions(checkpoint_path, scenario_map, "agents")

    if poa:
        assert len(paths) == 2, "Enter exactly two paths; decentralized followed by centralized."
        print("Will take first path as decentralized.")
        checkpoints_decent = os.listdir(Path(paths[0]))
        checkpoints_cent = os.listdir(Path(paths[1]))
        for checkpoint_decent in checkpoints_decent:
            if decentralized_cp is None or checkpoint_decent in dec_cp:
                for checkpoint_cent in checkpoints_cent:
                    if centralized_cp is None or checkpoint_cent in c_cp:
                        info_decent = get_info(Path(paths[0], checkpoint_decent))
                        info_cent = get_info(Path(paths[0], checkpoint_cent))
                        dfs_decent, masks_decent = load_checkpoint_dfs(Path(paths[0], checkpoint_decent), info_decent)
                        dfs_cent, masks_cent = load_checkpoint_dfs(Path(paths[1], checkpoint_cent), info_cent)
                        rewards_decent, rewards_filtered_decent = get_rewards(dfs_decent, masks_decent)
                        rewards_cent, rewards_filtered_cent = get_rewards(dfs_cent, masks_cent)
                        p_o_a, cent_rew_norm, decent_rew_norm = get_poa(rewards_filtered_cent, rewards_filtered_decent)
                        print(
                            "PoA of decentralized {} and centralized {} is {}   (reward_decent = {}, reward_cent = {})"
                            .format(checkpoint_decent,
                                    checkpoint_cent,
                                    p_o_a,
                                    np.mean(rewards_filtered_decent),
                                    np.mean(rewards_filtered_cent)))
                        if centralized_cp is not None and decentralized_cp is not None:
                            fig = plt.figure(figsize=(6, 9))
                            cent = plt.boxplot(cent_rew_norm, positions=[-0.2], widths=0.3, sym='b+',
                                               showfliers=True)
                            decent = plt.boxplot(decent_rew_norm, positions=[0.2], widths=0.3, sym='r+',
                                                 showfliers=True)
                            set_box_color(cent, COLORS['blue'])
                            set_box_color(decent, COLORS['red'])
                            plt.plot([], c=COLORS['blue'], label='centralized')
                            plt.plot([], c=COLORS['red'], label='decentralized')
                            plt.legend()
                            plt.title('PoA = {0:.3f}'.format(p_o_a))
                            plt.xticks([-0.2, 0.2], ["cp {}".format(centralized_cp), "cp {}".format(decentralized_cp)])
                            plt.ylabel("normalized reward")

                            plt.savefig('test2.png', dpi=200)

    if density_plots:
        for path in paths:
            checkpoints_dir = os.listdir(Path(path))
            for i, checkpoint in enumerate(checkpoints_dir):
                if checkpoints is None or checkpoint in cps:
                    print('Current checkpoint: {}'.format(checkpoint))
                    checkpoint_path = Path(path, checkpoint)
                    info = get_info(checkpoint_path)
                    density_plot(checkpoint_path, info)


def parse_args():
    parser = argparse.ArgumentParser("Learning Metrics Plotter")

    parser.add_argument('-p',
                        '--paths',
                        nargs='+',
                        help='Paths of evaluations files',
                        required=True)

    parser.add_argument('-n',
                        '--scenario_name',
                        type=str,
                        default="",
                        required=False)

    parser.add_argument('--checkpoints',
                        nargs='*',
                        help='List of checkpoints to run evaluation on. Leave empty for all.',
                        required=False,
                        default=None)

    parser.add_argument('--coloring',
                        nargs='*',
                        help='List of colorings. Valid values: control_input, speed, reward, acceleration, agents. '
                             'Leave empty for all.',
                        required=False,
                        default=None)

    parser.add_argument(
        "--training_progress_video", default=False, action="store_true",
        help="Make video of training progress. Need to change coloring manually in evaluation.py"
    )

    parser.add_argument(
        "--checkpoint_video", default=False, action="store_true",
        help="Make video of evaluated checkpoint."
    )

    parser.add_argument(
        "--poa", default=False, action="store_true",
        help="Calculate the price of anarchy."
    )

    parser.add_argument('--poa_checkpoints',
                        nargs=2,
                        metavar=('decent_cp', 'cent_cp'),
                        help='Two checkpoints for the Price of Anarchy. Decentralized first.',
                        default=(None, None))

    parser.add_argument(
        "--density_plots", default=False, action="store_true",
        help="Make density plots."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    decent_cp, cent_cp = args.poa_checkpoints

    main(
        paths=args.paths,
        scenario_name=args.scenario_name,
        checkpoints=args.checkpoints,
        training_progress_video=args.training_progress_video,
        checkpoint_video=args.checkpoint_video,
        coloring=args.coloring,
        poa=args.poa,
        decentralized_cp=decent_cp,
        centralized_cp=cent_cp,
        density_plots=args.density_plots,
    )
