import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd

import argparse

LARGESIZE, MEDIUMSIZE, SMALLSIZE = 16, 13, 10

PALETTE_REP = ['#A93226',  # red
               '#884EA0',  # purple
               '#2471A3',  # blue
               '#D4AC0D',  # yellow
               '#229954',  # green
               ] * 10

PALETTE_REP = ["#4B8673", "#5FD068", "#D61C4E", "#F77E21"]

LINESTYLES = ['solid'] * 5 + ['dashed'] * 5 + ['dashdot'] * 5 + ['dotted'] * 5


def str2list(s):
    slist = s[1:-1].split(',')
    return [float(x) for x in slist]


def main():
    csv_paths = ["./report_plotting/4_merge/logs/PPO_FrameStack_3b54a_00000_0_2022-04-06_12-36-51",
                 "./report_plotting/4_merge/logs/PPO_FrameStack_d3992_00000_0_2022-04-04_19-02-51",
                 "./report_plotting/4_merge/logs/PPO_FrameStack_7b4bb_00000_0_2022-04-05_15-10-08",
                 "./report_plotting/4_merge/logs/PPO_FrameStack_fada6_00000_0_2022-04-04_08-34-02"]

    save_path = "./report_plotting/4_merge/logs"
    names = [r"$\textrm{centralized run 1}$",
             r"$\textrm{centralized run 2}$",
             r"$\textrm{decentralized run 1}$",
             r"$\textrm{decentralized run 2}$"]

    matplotlib.rcParams['savefig.dpi'] = 800
    matplotlib.rcParams["text.usetex"] = True

    plt.rcParams.update({'font.size': LARGESIZE})
    plt.rcParams.update({'axes.titlesize': 26})
    plt.rcParams.update({'axes.labelsize': 24})
    plt.rcParams.update({'xtick.labelsize': 20})
    plt.rcParams.update({'ytick.labelsize': 20})
    plt.rcParams.update({'legend.fontsize': 24})
    plt.rcParams.update({'figure.titlesize': LARGESIZE})

    dfs = []
    for path in csv_paths:
        df = pd.read_csv(Path(path, "progress.csv"))
        dfs.append(df)

    hist = 'hist_stats/episode_reward'

    xax = [df['episodes_total'] for df in dfs]

    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    for i, df in enumerate(dfs):
        medians = [np.median(str2list(x)) for x in df[hist]]
        upper = [np.percentile(str2list(x), 75) for x in df[hist]]
        lower = [np.percentile(str2list(x), 25) for x in df[hist]]
        # maximum = [np.max(str2list(x)) for x in df[hist]]
        # minimum = [np.min(str2list(x)) for x in df[hist]]
        ax.plot(xax[i], medians, color=PALETTE_REP[i], linestyle=LINESTYLES[i], label=names[i])
        ax.fill_between(xax[i], lower, upper,
                        color=PALETTE_REP[i], linestyle=LINESTYLES[i], alpha=0.2)
        # ax.fill_between(xax[i], minimum, maximum,
        #                 color=PALETTE_REP[i], linestyle=LINESTYLES[i], alpha=0.05)

    ax.grid("on")
    plt.legend()
    plt.ylabel(r'$\textrm{median reward}$')
    plt.xlabel(r'$\textrm{total simulated episodes}$')
    plt.title(r'$\textrm{Four-Way Merge Scenario, Training Progress}$')

    plt.savefig(Path(save_path, 'training_progress.png'))
    plt.savefig(Path(save_path, 'training_progress.pdf'))


    xax = [np.arange(1, len(df['done']) + 1) for df in dfs]

    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    for i, df in enumerate(dfs):
        medians = [np.median(str2list(x)) for x in df[hist]]
        upper = [np.percentile(str2list(x), 75) for x in df[hist]]
        lower = [np.percentile(str2list(x), 25) for x in df[hist]]
        # maximum = [np.max(str2list(x)) for x in df[hist]]
        # minimum = [np.min(str2list(x)) for x in df[hist]]
        ax.plot(xax[i], medians, color=PALETTE_REP[i], linestyle=LINESTYLES[i], label=names[i])
        ax.fill_between(xax[i], lower, upper,
                        color=PALETTE_REP[i], linestyle=LINESTYLES[i], alpha=0.2)
        # ax.fill_between(xax[i], minimum, maximum,
        #                 color=PALETTE_REP[i], linestyle=LINESTYLES[i], alpha=0.05)

    ax.grid("on")
    plt.legend()
    plt.ylabel(r'$\textrm{median reward}$')
    plt.xlabel(r'$\textrm{checkpoint}$')
    plt.title(r'$\textrm{Four-Way Merge Scenario, Training Progress}$')

    plt.savefig(Path(save_path, 'training_progress_cps.png'))
    plt.savefig(Path(save_path, 'training_progress_cps.pdf'))


# def parse_args():
#     parser = argparse.ArgumentParser("")
#
#     parser.add_argument('-p',
#                         '--paths',
#                         nargs='+',
#                         help='Paths of progress.csv files relative to ./baselines/marl_benchmark/',
#                         required=True)
#
#     parser.add_argument('-s',
#                         '--save_path',
#                         type=str,
#                         required=True)
#
#     return parser.parse_args()


if __name__ == "__main__":
    # args = parse_args()

    # main(
    #     csv_paths=args.paths,
    #     save_path=args.save_path,
    # )

    main()
