import argparse
from pathlib import Path
import pandas as pd
from time import gmtime, strftime
import os

from typing import List, Tuple
from pandas import DataFrame

from scipy.ndimage.filters import uniform_filter1d

import matplotlib
import matplotlib.pyplot as plt
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
           ]


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


def main(
        training_path,
):
    save_path = Path(training_path, "logs_and_plots")
    save_path.mkdir(parents=True, exist_ok=True)

    log_keys = {"date", "time", "run", "paradigm", "name", "converged", "stable_checkpoint", "first_stable", "total_checkpoints"}
    logs = dict([(key, []) for key in log_keys])
    # get list of only folders (excluding .json, etc.)
    runs_dirs = [name for name in os.listdir(training_path) if os.path.isdir(os.path.join(training_path, name)) and
                 name != "logs_and_plots"]
    for run, dir in enumerate(runs_dirs):
        logs["run"].append(run)
        # run_str = "{:03d}".format(run)
        logs["name"].append(dir)
        progress_path = Path(training_path, dir, "progress.csv")
        date, time = extract_date_time(dir)
        logs["date"].append(date)
        logs["time"].append(time)
        df_progress = pd.read_csv(progress_path)
        logs["paradigm"].append(get_paradigm(df_progress))
        cols = list(df_progress.columns)
        vf_expl_var_cols = [col for col in cols if "vf_explained_var" in col]
        vf_expl_var = [list(df_progress[col]) for col in vf_expl_var_cols]

        n_agents = len(vf_expl_var)

        var_thres = 1e-3
        mean_thres = 0.97 - 0.01 * (n_agents - 1)
        filter_window = 14

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

        if converged_len >= min_converged_len:
            logs["converged"].append(1)
            logs["first_stable"].append(checkpoints[-1] - converged_len + 1)

            cp = checkpoints[-1] - converged_len
            cp_stable = cp + (5 - cp % 5) + 5
            logs["stable_checkpoint"].append(cp_stable)
        else:
            if converged_len > 0:
                logs["first_stable"].append(checkpoints[-1] - converged_len + 1)
            else:
                logs["first_stable"].append(0)
            logs["stable_checkpoint"].append(0)
            logs["converged"].append(0)

        make_plots = True

        if make_plots:
            # plotting for my own understanding
            fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)

            ax.plot(checkpoints, mean_vf_expl_var, color='k')
            ax.plot(checkpoints, filt_mn, color='k', linestyle='--')
            ax.plot([x + filter_window/2 for x in range(len(mean_vf_expl_var) - filter_window)],
                    [100 * x for x in var_mn],
                    color='k', linestyle='-.', alpha=0.3)

            for k in range(len(mean_vf_expl_var)):
                c = 'g' if conv_mn[k] else 'r'
                plt.scatter(checkpoints[k], 0.3, color=c, s=200, alpha=1, marker='s')

            plt.text(1, 0.3, "mean converged", color='white')

            # conv = []
            for i in range(len(vf_expl_var)):
                filt = uniform_filter1d(vf_expl_var[i], size=filter_window)
                ax.plot(checkpoints, filt, color=PALETTE[i], linestyle='--')
                ax.plot(checkpoints, vf_expl_var[i], color=PALETTE[i], label='{}'.format(i))
                var = [np.var(vf_expl_var[i][j:j+filter_window]) for j in range(len(vf_expl_var[i]) - filter_window)]
                ax.plot([x+filter_window/2 for x in range(len(vf_expl_var[i]) - filter_window)],
                        [100*x for x in var],
                        color=PALETTE[i], linestyle='-.', alpha=0.3)
                conv = [filt[k] > mean_thres for k in range(len(vf_expl_var[i]))]
                # conv = [filt[k+5] > mean_thres and var[k] < var_thres for k in range(len(vf_expl_var[i]) - filter_window)]
                for k in range(len(vf_expl_var[i])):
                    c = 'g' if conv[k] else 'r'
                    plt.scatter(checkpoints[k], 0.15+(i*0.04), color=c, s=200, alpha=1, marker='s')
                plt.text(1, 0.15+(i*0.04), "agent {} converged".format(i), color='white')

            plt.ylim([0, 1])

            # ax.fill_between(checkpoints, [0.95 for _ in checkpoints], [1 for _ in checkpoints],
            #                 color='r', alpha=0.2)
            ax.fill_between(checkpoints, [mean_thres for _ in checkpoints], [1 for _ in checkpoints],
                            color='g', alpha=0.2)

            plt.grid()
            plt.savefig(Path(training_path, "logs_and_plots/convergence_analysis_{}.png".format(dir)))

    df_logs = pd.DataFrame.from_dict(logs)
    df_logs.to_csv(Path(training_path, "logs_and_plots/convergence_logs.csv"))


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
