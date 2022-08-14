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

import sys
import subprocess

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


def get_run_paths(path,
                  identification_prefix: str = "PPO_FrameStack"):

    run_paths = []
    for x in list(os.walk(Path(path), topdown=True)):
        if "evaluation" not in x[0] and identification_prefix in x[0].split('/')[-1]:
            run_paths.append('/'.join(x[0].split('/')[:-1]))

    # remove paths containing "logs_and_plots"
    run_paths = [x for x in run_paths if "logs_and_plots" not in x]

    return list(set(run_paths))


def main(
        path,
        save_path=None,
):
    """
    Generate a log-file and qualitative plots to automate convergence detection of multiple training runs.

    :param save_path:
    :param training_path: path to a folder full of training runs (sub-folders typically starting with PPO_FrameStack...)
    :return:
    """

    checkpoint_frequency = 5

    run_paths = get_run_paths(path)

    if save_path is None:
        save_path = Path(path, "evaluation")
    save_path.mkdir(parents=True, exist_ok=True)

    # first, set up log dict and convert to DataFrame in tend
    log_keys = {"date", "time", "run", "paradigm", "name",
                "total_checkpoints", "max_reward", "max_reward_checkpoint", "run_path"}
    logs = dict([(key, []) for key in log_keys])

    for training_path in run_paths:

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
            logs["run_path"].append(str(training_path) + '/' + d)

            # somtimes there is an error in the progress.csv file (file could be cleaned up manually) -> ignore and log
            try:
                df_progress = pd.read_csv(progress_path)
            except ParserError:
                logs["paradigm"].append("CSV_PARSE_ERROR")
                logs["total_checkpoints"].append("CSV_PARSE_ERROR")
                logs["max_reward_checkpoint"].append("CSV_PARSE_ERROR")
                logs["max_reward"].append("CSV_PARSE_ERROR")
                continue
            except EmptyDataError:
                logs["paradigm"].append("CSV_EMPTY_DATA_ERROR")
                logs["total_checkpoints"].append("CSV_EMPTY_DATA_ERROR")
                logs["max_reward_checkpoint"].append("CSV_EMPTY_DATA_ERROR")
                logs["max_reward"].append("CSV_EMPTY_DATA_ERROR")
                continue

            logs["paradigm"].append(get_paradigm(df_progress))
            cols = list(df_progress.columns)
            vf_expl_var_cols = [col for col in cols if "vf_explained_var" in col]
            vf_expl_var = [list(df_progress[col]) for col in vf_expl_var_cols]

            mean_checkpoint_rewards = list(df_progress["episode_reward_mean"])
            max_reward_checkpoint = (np.argmax(mean_checkpoint_rewards[checkpoint_frequency-1::checkpoint_frequency]) + 1) * checkpoint_frequency
            max_reward = mean_checkpoint_rewards[max_reward_checkpoint - 1]

            logs["max_reward"].append(max_reward)
            logs["max_reward_checkpoint"].append(max_reward_checkpoint)

            # list of checkpoints for the current run
            checkpoints = list(range(1, len(vf_expl_var[0]) + 1))
            logs["total_checkpoints"].append(checkpoints[-1])

    df_logs = pd.DataFrame.from_dict(logs)
    df_logs.to_csv(Path(save_path, "info_logs.csv"))


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
