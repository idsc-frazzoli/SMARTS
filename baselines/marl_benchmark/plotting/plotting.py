import argparse
from pathlib import Path
import pandas as pd
from time import gmtime, strftime
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from utils import str2list, get_number_agents, plot_mean

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


def main(
        paths,
        scenario_name,
        title="",
        mean_reward=False,
        mean_len=False,
        learner_stats=False,
        legend=None,
        agent_wise=False,
        png=False,
        pdf=False,
        high_res=False,
        x_axis="checkpoints",
):
    if high_res:
        matplotlib.rcParams['savefig.dpi'] = 300

    if title == "":
        title = scenario_name

    datetime = strftime("%Y%m%d_%H%M%S_", gmtime())
    log_path = Path('plotting/plots', datetime + scenario_name)
    log_path.mkdir(parents=True, exist_ok=True)

    abs_paths = []
    for path in paths:
        abs_paths.append(Path("log/results/run", path, "progress.csv"))

    dfs = []
    for path in abs_paths:
        dfs.append(pd.read_csv(path))

    agents_info = [get_number_agents(df.columns) for df in dfs]
    n_agents = [x[0] for x in agents_info]
    paradigms = [x[1] for x in agents_info]

    xaxes = []
    xlabels = []
    xnames = []
    if 'checkpoints' in x_axis:
        xaxes.append([np.arange(1, len(df['done'])+1) for df in dfs])
        xlabels.append('checkpoint')
        xnames.append('checkpoint')
    if 'time_total_s' in x_axis:
        xaxes.append([df['time_total_s'] for df in dfs])
        xlabels.append('total time [s]')
        xnames.append('time_total_s')
    if 'episodes_total' in x_axis:
        xaxes.append([df['episodes_total'] for df in dfs])
        xlabels.append('total episodes')
        xnames.append('episodes_total')

    if legend is None:
        legend = [''] * len(dfs)

    # Remark: This could be done similarly as for learner_stats, but I'm too lazy to rewrite the code.
    if mean_reward:
        ylabel, yname = 'episode reward', 'episode_reward_mean'
        plot_mean(x_axis, dfs, ylabel, yname, legend, title, png, pdf, log_path)

    if mean_len:
        ylabel, yname = 'episode length', 'episode_len_mean'
        plot_mean(x_axis, dfs, ylabel, yname, legend, title, png, pdf, log_path)

    if learner_stats:
        learner_stats_prefix = 'info/learner/'
        learner_stats_postfix = {'cur_kl_coef': '/learner_stats/cur_kl_coeff',
                                 'cur_lr': '/learner_stats/cur_lr',
                                 'Total Loss': '/learner_stats/total_loss',
                                 'Policy Loss': '/learner_stats/policy_loss',
                                 'vf Loss': '/learner_stats/vf_loss',
                                 'vf Explained Variance': '/learner_stats/vf_explained_var',
                                 'kl': '/learner_stats/kl',
                                 'Entropy': '/learner_stats/entropy',
                                 'Entropy Coefficient': '/learner_stats/entropy_coeff'}
        learner_path = Path(log_path, 'learner_stats')
        learner_path.mkdir(parents=True, exist_ok=True)
        for i, df in enumerate(dfs):
            scenario_path = Path(learner_path, paths[i].split('/')[2])
            scenario_path.mkdir(parents=True, exist_ok=True)
            if paradigms[i] == "decentralized":
                for key, item in learner_stats_postfix.items():
                    for j, xaxis in enumerate(xaxes):
                        fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
                        for agent in range(n_agents[i]):
                            ax.plot(xaxis[i],
                                    df[learner_stats_prefix + 'AGENT-' + str(agent) + item],
                                    color=AGENT_COLORS[agent], label='Agent ' + str(agent))
                        # plt.title(title)
                        plt.legend()
                        plt.ylabel(key)
                        plt.xlabel(xlabels[j])
                        if png:
                            plt.savefig(Path(scenario_path, '{}'.format(item[15:]) + '_' + xnames[j] + '.png'))
                        if pdf:
                            plt.savefig(Path(scenario_path, '{}'.format(item[15:]) + '_' + xnames[j] + '.pdf'))
            if paradigms[i] == 'centralized':
                for key, item in learner_stats_postfix.items():
                    for j, xaxis in enumerate(xaxes):
                        fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
                        ax.plot(xaxis[i],
                                df[learner_stats_prefix + 'default_policy' + item],
                                color=PALETTE[2])
                        # plt.title(title)
                        plt.ylabel(key)
                        plt.xlabel(xlabels[j])
                        if png:
                            plt.savefig(Path(scenario_path, '{}'.format(item[15:]) + '_' + xnames[j] + '.png'))
                        if pdf:
                            plt.savefig(Path(scenario_path, '{}'.format(item[15:]) + '_' + xnames[j] + '.pdf'))

    if mean_reward and agent_wise:
        agent_wise_path = Path(Path(log_path, 'agent_wise'))
        agent_wise_path.mkdir(parents=True, exist_ok=True)
        for i, df in enumerate(dfs):
            scenario_path = Path(agent_wise_path, paths[i].split('/')[2])
            scenario_path.mkdir(parents=True, exist_ok=True)
            if paradigms[i] == 'decentralized':
                prefix = 'hist_stats/policy_AGENT-'
                postfix = '_reward'
                for j, xaxis in enumerate(xaxes):
                    fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
                    for agent in range(n_agents[i]):
                        ax.plot(xaxis[i],
                                df['policy_reward_mean/AGENT-' + str(agent)],
                                color=AGENT_COLORS[agent], label='Agent ' + str(agent))
                        std_devs = [np.std(str2list(x)) for x in df[prefix + str(agent) + postfix]]
                        ax.fill_between(xaxis[i],
                                        df['policy_reward_mean/AGENT-' + str(agent)] - std_devs,
                                        df['policy_reward_mean/AGENT-' + str(agent)] + std_devs,
                                        color=AGENT_COLORS[agent], alpha=0.2)
                    ax.plot(xaxis[i], df['episode_reward_mean']/n_agents[i],
                            color=COLORS['black'], label='Average Reward',
                            linewidth=3)
                    # plt.title(title)
                    plt.legend()
                    plt.ylabel('mean reward')
                    plt.xlabel(xlabels[j])
                    if png:
                        plt.savefig(Path(scenario_path, 'mean_reward' + '_' + xnames[j] + '.png'))
                    if pdf:
                        plt.savefig(Path(scenario_path, 'mean_reward' + '_' + xnames[j] + '.pdf'))
            # TODO: Also implement for centralized.
            if paradigms[i] == 'centralized':
                pass

    plt.close()


def parse_args():
    parser = argparse.ArgumentParser("Learning Metrics Plotter")

    parser.add_argument('-p',
                        '--paths',
                        nargs='+',
                        help='Paths of progress.csv file relative to ./log/results/run',
                        required=True)

    parser.add_argument('--legend',
                        nargs='*',
                        help='Legends of the files',
                        required=False,
                        default=None)

    parser.add_argument("--scenario_name",
                        type=str,
                        default="",
                        required=True)

    parser.add_argument("--title",
                        type=str,
                        default="",
                        required=False)

    parser.add_argument(
        "--mean_reward", default=False, action="store_true",
        help="Make plot of mean episode reward."
    )

    parser.add_argument(
        "--mean_len", default=False, action="store_true",
        help="Make plot of mean episode length."
    )

    parser.add_argument(
        "--learner_stats", default=False, action="store_true",
        help="Make plots for learner statistics. (requires number of agents)"
    )

    parser.add_argument(
        "--agent_wise", default=False, action="store_true",
        help="Additionally, save plots of mean reward and for each scenario agent-wise."
    )

    parser.add_argument(
        "--png", default=False, action="store_true",
        help="Save plots as png."
    )

    parser.add_argument(
        "--pdf", default=False, action="store_true",
        help="If I also want to save a .pdf of each plot."
    )

    parser.add_argument(
        "--high_res", default=False, action="store_true",
        help="Saved png is of higher resolution."
    )

    parser.add_argument(
        "--x_axis",
        nargs='+',
        type=str,
        default="checkpoints",
        help="x-axis values. can be checkpoints (default) or time_total_s"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        paths=args.paths,
        scenario_name=args.scenario_name,
        title=args.title,
        mean_reward=args.mean_reward,
        mean_len=args.mean_len,
        learner_stats=args.learner_stats,
        legend=args.legend,
        agent_wise=args.agent_wise,
        png=args.png,
        pdf=args.pdf,
        high_res=args.high_res,
        x_axis=args.x_axis
    )
