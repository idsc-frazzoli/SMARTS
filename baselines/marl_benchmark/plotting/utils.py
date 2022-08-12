import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGSIZE = (16, 9)

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

PALETTE_REP = ['#A93226',  # red
               '#884EA0',  # purple
               '#2471A3',  # blue
               '#D4AC0D',  # yellow
               '#229954',  # green
               ] * 10

LINESTYLES = ['solid'] * 5 + ['dashed'] * 5 + ['dashdot'] * 5 + ['dotted'] * 5


def str2list(s):
    slist = s[1:-1].split(',')
    return [float(x) for x in slist]


def get_number_agents(cols):
    finished = False
    n_agents = 0
    while not finished:
        finished = True
        for label in cols:
            if bool(re.match("(info/learner/AGENT-" + str(n_agents) + "/learner_stats/entropy)", label)):
                finished = False
                n_agents += 1
    if n_agents > 0:
        paradigm = "decentralized"
    else:
        paradigm = "centralized"
        n_agents = 1

    return n_agents, paradigm


def plot_mean(x_axis, dfs, ylabel, yname, legend, title, png, pdf, log_path, boxplot, y_lim):
    def plt_mn(dfs, xax, xlab, ylab, xna, yna, leg, tit, pn, pd, log_p, bxplt, y_lm):
        if yna == 'episode_reward_mean':
            hist = 'hist_stats/episode_reward'
        if yna == 'episode_len_mean':
            hist = 'hist_stats/episode_lengths'
        fig, ax = plt.subplots(figsize=FIGSIZE, tight_layout=True)
        for i, df in enumerate(dfs):
            if not bxplt:
                std_devs = [np.std(str2list(x)) for x in df[hist]]
                ax.plot(xax[i], df[yna], color=PALETTE_REP[i], linestyle=LINESTYLES[i], label=leg[i])
                ax.fill_between(xax[i],
                                df[yna] - std_devs,
                                df[yna] + std_devs,
                                color=PALETTE_REP[i], linestyle=LINESTYLES[i], alpha=0.2)
            else:
                medians = [np.median(str2list(x)) for x in df[hist]]
                upper = [np.percentile(str2list(x), 75) for x in df[hist]]
                lower = [np.percentile(str2list(x), 25) for x in df[hist]]
                maximum = [np.max(str2list(x)) for x in df[hist]]
                minimum = [np.min(str2list(x)) for x in df[hist]]
                ax.plot(xax[i], medians, color=PALETTE_REP[i], linestyle=LINESTYLES[i], label=leg[i])
                ax.fill_between(xax[i], lower, upper,
                                color=PALETTE_REP[i], linestyle=LINESTYLES[i], alpha=0.2)
                ax.fill_between(xax[i], minimum, maximum,
                                color=PALETTE_REP[i], linestyle=LINESTYLES[i], alpha=0.05)

        if leg != [''] * len(dfs):
            ax.legend()
        plt.title(tit)
        plt.grid()
        plt.ylabel(ylab)
        plt.xlabel(xlab)
        if y_lm:
            plt.ylim(y_lm[0], y_lm[1])

        if pn:
            plt.savefig(Path(log_p, yna + '_' + xna + '.png'))
        if pd:
            plt.savefig(Path(log_p, yna + '_' + xna + '.pdf'))

    if 'checkpoints' in x_axis:
        xaxis = [np.arange(1, len(df['done']) + 1) for df in dfs]
        xlabel, xname = 'checkpoint', 'checkpoint'
        plt_mn(dfs, xaxis, xlabel, ylabel, xname, yname, legend, title, png, pdf, log_path, boxplot, y_lim)
    if 'time_total_s' in x_axis:
        xaxis = [df['time_total_s'] for df in dfs]
        xlabel, xname = 'total time [s]', 'total_time_s'
        plt_mn(dfs, xaxis, xlabel, ylabel, xname, yname, legend, title, png, pdf, log_path, boxplot, y_lim)
    if 'episodes_total' in x_axis:
        xaxis = [df['episodes_total'] for df in dfs]
        xlabel, xname = 'total episodes', 'episodes_total'
        plt_mn(dfs, xaxis, xlabel, ylabel, xname, yname, legend, title, png, pdf, log_path, boxplot, y_lim)
    if 'timesteps_total' in x_axis:
        xaxis = [df['timesteps_total'] for df in dfs]
        xlabel, xname = 'total time steps', 'timesteps_total'
        plt_mn(dfs, xaxis, xlabel, ylabel, xname, yname, legend, title, png, pdf, log_path, boxplot, y_lim)
