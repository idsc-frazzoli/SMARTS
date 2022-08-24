import argparse
from pathlib import Path
import os

import pandas as pd
import math
import numpy as np
import yaml
import pickle

import copy

# from baselines.marl_benchmark.evaluation.utils import get_info, get_poa, get_rewards, load_checkpoint_dfs
# from collections import defaultdict

import matplotlib.pyplot as plt

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


def main(eval_path):

    fig_save_folder = Path(eval_path, "figures_report")
    fig_save_folder.mkdir(parents=True, exist_ok=True)

    with open(Path(eval_path, "data.pickle"), 'rb') as handle:
        data = pickle.load(handle)
    df_stats = pd.read_csv(Path(eval_path, "stats.csv"))

    goal_reached_threshold = 0.75

    costs, costs_com, costs_per_acc, costs_per_time, rews_per_goal = [], [], [], [], []
    paradigms, names, run_paths, eval_paths = [], [], [], []

    for i, row in df_stats.iterrows():
        if row["goal_reached_perc"] < goal_reached_threshold:
            continue

        if row["paradigm"] not in ["centralized", "decentralized"]:
            continue

        paradigm = row["paradigm"]
        name = row["name"]

        run_data = data[paradigm][name]
        cost, cost_com, cost_per_acc, cost_per_time, rew_per_goal = 0, 0, 0, 0, 0

        for agent in run_data.keys():
            cost_com += sum(run_data[agent]["episode_cost_com"]) / \
                        len(run_data[agent]["episode_cost_com"])
            cost_per_acc += sum(run_data[agent]["episode_cost_per_acceleration"]) / \
                            len(run_data[agent]["episode_cost_per_acceleration"])
            cost_per_time += sum(run_data[agent]["episode_cost_per_time"]) / \
                             len(run_data[agent]["episode_cost_per_time"])
            rew_per_goal += sum(run_data[agent]["episode_goal_improvement_reward"]) / \
                            len(run_data[agent]["episode_goal_improvement_reward"])
            cost += (sum(run_data[agent]["episode_cost_com"]) +
                     sum(run_data[agent]["episode_cost_per_acceleration"]) +
                     sum(run_data[agent]["episode_cost_per_time"]) -
                     sum(run_data[agent]["episode_goal_improvement_reward"])) / (
                        len(run_data[agent]["episode_cost_com"]))

        names.append(row["name"])
        run_paths.append(row["run_path"])
        eval_paths.append(row["evaluation_path"])
        costs.append(cost)
        paradigms.append(row["paradigm"])
        costs_com.append(cost_com)
        costs_per_acc.append(cost_per_acc)
        costs_per_time.append(cost_per_time)
        rews_per_goal.append(rew_per_goal)

    min_cost_index = np.argmin(costs)
    min_cost = costs[min_cost_index]
    dummy_costs = costs[:]
    for i in range(len(dummy_costs)):
        if paradigms[i] == "centralized":
            dummy_costs[i] = -float("inf")

    # find max NE cost
    max_ne_cost_index = np.argmax(np.array(dummy_costs))
    max_ne_cost = costs[max_ne_cost_index]

    print(paradigms)
    print(costs)
    print(max_ne_cost_index)
    print(min_cost_index)

    poa = max_ne_cost / min_cost
    print(poa)

    # poas = dict([(deg, []) for deg in degrees])
    # cost_com_high = dict([(deg, []) for deg in degrees])
    # cost_com_low = dict([(deg, []) for deg in degrees])
    # cost_per_acc_high = dict([(deg, []) for deg in degrees])
    # cost_per_acc_low = dict([(deg, []) for deg in degrees])
    # cost_per_time_high = dict([(deg, []) for deg in degrees])
    # cost_per_time_low = dict([(deg, []) for deg in degrees])
    #
    # for degree in degrees:
    #     for alpha in alphas:
    #         if (alpha, degree) not in alpha_degree_pairs:
    #             continue
    #         selector = (df_stats["degree"] == degree) & (df_stats["alpha"] == alpha)
    #         df = df_stats[selector]
    #         cent_selector = (df["paradigm"] == "centralized")
    #         decent_selector = (df["paradigm"] == "decentralized")
    #
    #         costs = []
    #         costs_com = []
    #         costs_per_acc = []
    #         costs_per_time = []
    #         paradigms = []
    #         names = []
    #         for i, row in df.iterrows():
    #             if row["goal_reached_perc"] < goal_reached_threshold or \
    #                     str(row["evaluation_path"]) == "nan":
    #                 continue
    #             run_data = data[degree][alpha][row["name"]]
    #             cost = 0
    #             cost_com = 0
    #             cost_per_acc = 0
    #             cost_per_time = 0
    #             for agent in run_data.keys():
    #                 cost_com += sum(run_data[agent]["episode_cost_com"]) / \
    #                             len(run_data[agent]["episode_cost_com"])
    #                 cost_per_acc += sum(run_data[agent]["episode_cost_per_acceleration"]) / \
    #                                 len(run_data[agent]["episode_cost_per_acceleration"])
    #                 cost_per_time += sum(run_data[agent]["episode_cost_per_time"]) / \
    #                                  len(run_data[agent]["episode_cost_per_time"])
    #                 cost += (sum(run_data[agent]["episode_cost_com"]) +
    #                          sum(run_data[agent]["episode_cost_per_acceleration"]) +
    #                          sum(run_data[agent]["episode_cost_per_time"])) / (
    #                                 3 * len(run_data[agent]["episode_cost_com"]))
    #
    #             names.append(row["name"])
    #             costs.append(cost)
    #             paradigms.append(row["paradigm"])
    #             costs_com.append(cost_com)
    #             costs_per_acc.append(cost_per_acc)
    #             costs_per_time.append(cost_per_time)
    #
    #         costs_sorted = []
    #         names_sorted = []
    #         paradigms_sorted = []
    #         costs_sorted_decent = []
    #         names_sorted_decent = []
    #         costs_sorted_cent = []
    #         names_sorted_cent = []
    #         costs_com_sorted = []
    #         costs_per_acc_sorted = []
    #         costs_per_time_sorted = []
    #         costs_com_sorted_decent = []
    #         costs_per_acc_sorted_decent = []
    #         costs_per_time_sorted_decent = []
    #         costs_com_sorted_cent = []
    #         costs_per_acc_sorted_cent = []
    #         costs_per_time_sorted_cent= []
    #         for _ in range(len(costs)):
    #             index = np.argmin(costs)
    #             costs_sorted.append(costs[index])
    #             names_sorted.append(names[index])
    #             paradigms_sorted.append(paradigms[index])
    #             costs_com_sorted.append((costs_com[index]))
    #             costs_per_acc_sorted.append(costs_per_acc[index])
    #             costs_per_time_sorted.append(costs_per_time[index])
    #             if paradigms[index] == "centralized":
    #                 costs_sorted_cent.append(costs[index])
    #                 names_sorted_cent.append(names[index])
    #                 costs_com_sorted_cent.append(costs_com[index])
    #                 costs_per_acc_sorted_cent.append(costs_per_acc[index])
    #                 costs_per_time_sorted_cent.append(costs_per_time[index])
    #             else:
    #                 costs_sorted_decent.append(costs[index])
    #                 names_sorted_decent.append(names[index])
    #                 costs_com_sorted_decent.append(costs_com[index])
    #                 costs_per_acc_sorted_decent.append(costs_per_acc[index])
    #                 costs_per_time_sorted_decent.append(costs_per_time[index])
    #             costs[index] = 1e20
    #
    #         poas[degree].append(costs_sorted_decent[-1] / costs_sorted[0])
    #         cost_com_high[degree].append(costs_com_sorted_decent[-1])
    #         cost_com_low[degree].append((costs_com_sorted[0]))
    #         cost_per_acc_high[degree].append(costs_per_acc_sorted_decent[-1])
    #         cost_per_acc_low[degree].append((costs_per_acc_sorted[0]))
    #         cost_per_time_high[degree].append(costs_per_time_sorted_decent[-1])
    #         cost_per_time_low[degree].append((costs_per_time_sorted[0]))
    #
    #         print(poas[degree][-1])


    # fig, ax = plt.subplots(figsize=(16,9), tight_layout=True)
    # for i, degree in enumerate(degrees):
    #     ax.plot(alphas, poas[degree],
    #             color=PALETTE[i], marker="x", markersize=10, label="degree = {}".format(degree),
    #             linestyle=(0, (5, 10)))
    # ax.legend()
    # plt.title(r"PoA for different $\alpha$'s")
    # plt.grid()
    # plt.ylabel("PoA")
    # plt.xlabel(r"$\alpha$")
    # fig.show()
    # plt.savefig(Path(fig_save_folder, "PoA_alpha_plot.png"), dpi=500)
    # plt.savefig(Path(fig_save_folder, "PoA_alpha_plot.pdf"))
    #
    # for degree in degrees:
    #     fig, ax = plt.subplots(figsize=(14, 7), tight_layout=True)
    #     # plt.xticks(rotation=45)
    #     N = len(alphas)
    #     ind = np.arange(N)
    #     width = 0.25
    #     locations = [-0.15, 0.15]
    #     ax.bar(ind+locations[1], cost_com_high[degree], width, color=PALETTE[0])
    #     ax.bar(ind+locations[1], cost_per_time_high[degree], width, bottom=cost_com_high[degree], color=PALETTE[1])
    #     ax.bar(ind+locations[1], cost_per_acc_high[degree], width, bottom=np.add(cost_com_high[degree], cost_per_time_high[degree]), color=PALETTE[2])
    #     ax.legend(labels=[r'$J^{com}(\gamma)$', r'$J^{per, time}(\gamma)$', r'$J^{per, acc}(\gamma)$'])
    #     ax.bar(ind+locations[0], cost_com_low[degree], width, color=PALETTE[0])
    #     ax.bar(ind+locations[0], cost_per_time_low[degree], width, bottom=cost_com_low[degree], color=PALETTE[1])
    #     ax.bar(ind+locations[0], cost_per_acc_low[degree], width, bottom=np.add(cost_com_low[degree], cost_per_time_low[degree]), color=PALETTE[2])
    #     xticks = [i+loc for i in ind for loc in locations]
    #     xlabels = [r"$\alpha$ = {}, {}".format(a, x) for a in alphas for x in [r"$\gamma^*$", r"$\gamma_{NE}$"]]
    #     plt.xticks(xticks, xlabels, rotation=45)
    #     plt.ylabel("cost")
    #
    #     plt.savefig(Path(fig_save_folder, "costs_plot_degree_{}.png".format(degree)), dpi=500)
    #     plt.savefig(Path(fig_save_folder, "costs_plot_degree_{}.pdf".format(degree)))
    #


def parse_args():
    parser = argparse.ArgumentParser("")

    parser.add_argument("-p",
                        "--path",
                        type=str,
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(eval_path=args.path)
