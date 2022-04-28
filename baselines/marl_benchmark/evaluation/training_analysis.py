import argparse
from pathlib import Path
import pandas as pd
from time import gmtime, strftime
import os

from typing import List, Tuple
from pandas import DataFrame
from pandas.errors import ParserError, EmptyDataError

from scipy.ndimage.filters import uniform_filter1d

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

import shutil


PALETTE = ['#A93226',  # red
           '#884EA0',  # purple
           '#2471A3',  # blue
           '#D4AC0D',  # yellow
           '#229954',  # green
           '#CA6F1E',  # orange
           '#17A589',  # blue/green
           '#CB4335',  # red 2
           '#7D3C98',  # purple 2
           '#2E86C1',  # blue 2
           '#D68910',  # yellow 2
           '#28B463',  # green 2
           '#BA4A00',  # orange 2
           '#138D75',  # #blue/green 2
           ] * 10


def extract_date_time(name: str) -> Tuple[str, str]:
    time = name[-8:]
    date = name[-19:-9]
    return date, time


def get_paradigm(df_progress: DataFrame) -> str:
    cols = list(df_progress.columns)
    decent = False
    for col in cols:
        if "AGENT" in col:
            decent = True
            break
    return "decentralized" if decent else "centralized"


def str2list(s):
    slist = s[1:-1].split(',')
    return [float(x) for x in slist]


def get_reward_stats(df_progress: DataFrame, checkpoint: int) -> Tuple[float, float]:
    mean_reward = df_progress["episode_reward_mean"][checkpoint - 1]
    episode_rewards = str2list(df_progress["hist_stats/episode_reward"][checkpoint - 1])
    standard_deviation = float(np.std(episode_rewards))

    return mean_reward, standard_deviation


def get_length_stats(df_progress: DataFrame, checkpoint: int) -> Tuple[float, float]:
    mean_len = df_progress["episode_len_mean"][checkpoint - 1]
    episode_len = str2list(df_progress["hist_stats/episode_lengths"][checkpoint - 1])
    standard_deviation = float(np.std(episode_len))

    return mean_len, standard_deviation


def main(
        training_path,
):
    """
    Generate a log-file and qualitative plots to automate convergence detection of multiple training runs.

    :param training_path: path to a folder full of training runs (sub-folders typically starting with PPO_FrameStack...)
    :return:
    """
    save_path = Path(training_path, "logs_and_plots")
    save_path.mkdir(parents=True, exist_ok=True)

    # first, set up log dict and convert to DataFrame in tend
    log_keys = {"date", "time", "run", "paradigm", "name", "converged", "stable_checkpoint", "first_stable",
                "total_checkpoints", "converged_len"}
    logs = dict([(key, []) for key in log_keys])

    # get list of only folders (excluding .json, etc.)
    runs_dirs = [name for name in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, name)) and
                 name != "logs_and_plots"]

    for run, d in enumerate(runs_dirs):
        logs["run"].append(run)
        # run_str = "{:03d}".format(run)
        logs["name"].append(d)
        progress_path = Path(training_path, d, "progress.csv")
        date, time = extract_date_time(d)
        logs["date"].append(date)
        logs["time"].append(time)

        # somtimes there is an error in the progress.csv file (file could be cleaned up manually) -> ignore and log
        try:
            df_progress = pd.read_csv(progress_path)
        except ParserError:
            logs["paradigm"].append("CSV_PARSE_ERROR")
            logs["converged"].append("CSV_PARSE_ERROR")
            logs["stable_checkpoint"].append("CSV_PARSE_ERROR")
            logs["first_stable"].append("CSV_PARSE_ERROR")
            logs["total_checkpoints"].append("CSV_PARSE_ERROR")
            logs["converged_len"].append("CSV_PARSE_ERROR")
            continue
        except EmptyDataError:
            logs["paradigm"].append("CSV_EMPTY_DATA_ERROR")
            logs["converged"].append("CSV_EMPTY_DATA_ERROR")
            logs["stable_checkpoint"].append("CSV_EMPTY_DATA_ERROR")
            logs["first_stable"].append("CSV_EMPTY_DATA_ERROR")
            logs["total_checkpoints"].append("CSV_EMPTY_DATA_ERROR")
            logs["converged_len"].append("CSV_EMPTY_DATA_ERROR")
            continue

        logs["paradigm"].append(get_paradigm(df_progress))
        cols = list(df_progress.columns)
        vf_expl_var_cols = [col for col in cols if "vf_explained_var" in col]
        vf_expl_var = [list(df_progress[col]) for col in vf_expl_var_cols]

        n_agents = len(vf_expl_var)

        # converged detection parameters
        var_thres = 1e-3  # switched back to not using variance as a detection metric
        # vf_expl_var mean has to be above this threshold
        # mean_thres = 0.97
        mean_thres = 0.97 - 0.01 * (n_agents - 1)
        # window size for mean filter
        filter_window = 5
        reward_derivative_threshold = 100  # currently not used -> TODO

        min_converged_len = 17

        # make filter_window even
        filter_window += filter_window % 2

        # list of checkpoints for the current run
        checkpoints = list(range(1, len(vf_expl_var[0]) + 1))
        logs["total_checkpoints"].append(checkpoints[-1])

        # take mean over the agents
        mean_vf_expl_var = np.mean([x for x in vf_expl_var], axis=0)

        # apply mean filter to vf_expl_var data
        filt_mn = uniform_filter1d(mean_vf_expl_var, size=filter_window)

        # calculate variance
        var_mn = [np.var(mean_vf_expl_var[j:j + filter_window]) for j in range(len(mean_vf_expl_var) - filter_window)]

        # convergence according to mean criterion
        conv_mn = [filt_mn[k] > mean_thres for k in range(len(mean_vf_expl_var))]
        # conv_mn = [filt_mn[int(k+filter_window/2)] > mean_thres and var_mn[k] < var_thres
        #            for k in range(len(mean_vf_expl_var) - filter_window)]

        converged_len = 0
        for c in conv_mn[::-1]:
            if c:
                converged_len += 1
            else:
                break

        logs["converged_len"].append(converged_len)

        # TODO: stable checkpoint selection should include the reward directly
        if converged_len >= min_converged_len:
            logs["converged"].append(1)
            logs["first_stable"].append(checkpoints[-1] - converged_len + 1)

            cp = checkpoints[-1] - converged_len
            # checkpoints are only saved each 5 checkpoints -> go to next and add 5
            cp_stable = cp + (5 - cp % 5) + 5
            logs["stable_checkpoint"].append(cp_stable)
        else:
            if converged_len > 0:
                logs["first_stable"].append(checkpoints[-1] - converged_len + 1)
            else:
                logs["first_stable"].append(0)
            logs["stable_checkpoint"].append(0)
            logs["converged"].append(0)

        # qualitative plotting for sanity check
        make_plots = True

        if make_plots:
            # plotting for my own understanding
            fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
            ax2 = ax.twinx()

            ax.plot(checkpoints, mean_vf_expl_var, color='k', label="mean_vf_expl_var")
            ax.plot(checkpoints, filt_mn, color='k', linestyle='--', label='mean_filtered_{}'.format(filter_window))
            # ax.plot([x + filter_window/2 for x in range(len(mean_vf_expl_var) - filter_window)],
            #         [100 * x for x in var_mn],
            #         color='k', linestyle='-.', alpha=0.3)

            mean_reward = df_progress["episode_reward_mean"]
            hist_rewards = df_progress["hist_stats/episode_reward"]
            upper = [np.percentile(str2list(x), 75) for x in hist_rewards]
            lower = [np.percentile(str2list(x), 25) for x in hist_rewards]
            median_reward = [np.median(str2list(x)) for x in hist_rewards]
            reward_derivative = [0] + [np.abs(mean_reward[i-1] - mean_reward[i]) for i in range(1, len(mean_reward))]
            ax2.plot(checkpoints, median_reward, color='r')
            ax2.plot(checkpoints, reward_derivative, color='b', alpha=0.2)
            ax2.fill_between(checkpoints, [0 for _ in checkpoints], [reward_derivative_threshold for _ in checkpoints],
                             color='b', alpha=0.05)
            ax2.fill_between(checkpoints, lower, upper,
                             color='r', alpha=0.2)
            ax2.set_ylabel('median reward with IQR')

            # only used for legend
            ax.plot([-1, -0.9], [-1, -1], color='r', label="median episode reward")
            ax.plot([-1, -0.9], [-1, -1], color='b', alpha=0.2, label="absolute reward derivative")
            ax.set_xlabel("checkpoint")
            ax.set_ylabel("vf_expl_var")

            for k in range(len(mean_vf_expl_var)):
                c = 'g' if conv_mn[k] else 'r'
                ax.scatter(checkpoints[k], 0.3, color=c, s=200, alpha=1, marker='s')

            ax.text(1, 0.3, "mean converged", color='white')

            # conv = []
            for i in range(len(vf_expl_var)):
                filt = uniform_filter1d(vf_expl_var[i], size=filter_window)
                ax.plot(checkpoints, filt, color=PALETTE[i], linestyle='--', label="filtered_agent_{}".format(i))
                ax.plot(checkpoints, vf_expl_var[i], color=PALETTE[i], label="vf_expl_var_agent_{}".format(i))
                # var = [np.var(vf_expl_var[i][j:j+filter_window]) for j in range(len(vf_expl_var[i]) - filter_window)]
                # ax.plot([x+filter_window/2 for x in range(len(vf_expl_var[i]) - filter_window)],
                #         [100*x for x in var],
                #         color=PALETTE[i], linestyle='-.', alpha=0.3)
                conv = [filt[k] > mean_thres for k in range(len(vf_expl_var[i]))]
                # conv = [filt[k+5] > mean_thres and var[k] < var_thres for k in range(len(vf_expl_var[i]) - filter_window)]
                for k in range(len(vf_expl_var[i])):
                    c = 'g' if conv[k] else 'r'
                    ax.scatter(checkpoints[k], 0.15+(i*0.04), color=c, s=200, alpha=1, marker='s')
                ax.text(1, 0.15+(i*0.04), "agent {} converged".format(i), color='white')

            ax.set_ylim([0, 1])
            ax.fill_between(checkpoints, [mean_thres for _ in checkpoints], [1 for _ in checkpoints],
                            color='g', alpha=0.2)

            plt.grid()
            ax.legend()
            ax.legend()
            plt.savefig(Path(training_path, "logs_and_plots/convergence_analysis_{}.png".format(d)))

    df_logs = pd.DataFrame.from_dict(logs)
    df_logs.to_csv(Path(training_path, "logs_and_plots/convergence_logs.csv"))

    # plot lengths and rewards of converged stable checkpoints
    fig, ax = plt.subplots(figsize=(10, 10), tight_layout=True)
    max_len, max_rew = 0, -1e6
    for run, d in enumerate(logs["name"]):
        if logs["converged"][run] == 1:
            progress_path = Path(training_path, d, "progress.csv")
            df_progress = pd.read_csv(progress_path)
            mean_reward, std_dev_rew = get_reward_stats(df_progress, logs["stable_checkpoint"][run])
            mean_len, std_dev_len = get_length_stats(df_progress, logs["stable_checkpoint"][run])
            if mean_len + std_dev_len/2 > max_len:
                max_len = mean_len + std_dev_len/2
            if mean_reward + std_dev_rew/2 > max_rew:
                max_rew = mean_reward + std_dev_rew/2
            e = Ellipse(xy=(mean_len, mean_reward),
                        width=std_dev_len, height=std_dev_rew,
                        color=PALETTE[run], alpha=0.2)
            ax.scatter(mean_len, mean_reward, color=PALETTE[run], label=logs["name"][run])
            ax.text(mean_len, mean_reward,
                    "cp {}, ({}, {})".format(logs["stable_checkpoint"][run], int(mean_len), int(mean_reward)))
            ax.add_artist(e)

    plt.legend()
    plt.grid()
    plt.xlabel("episode length")
    plt.ylabel("episode reward")
    plt.xlim([30, max_len])
    plt.ylim([-300, max_rew])
    plt.savefig(Path(training_path, "logs_and_plots/converged_len_reward.png"))


def parse_args():
    parser = argparse.ArgumentParser("Learning Metrics Plotter")

    parser.add_argument('-p',
                        '--path',
                        type=str,
                        help='Path to PPO_FrameStack training folders.',
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(training_path=args.path)
