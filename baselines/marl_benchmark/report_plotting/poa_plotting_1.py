import argparse
from pathlib import Path
import os

import pandas as pd
import math
import numpy as np
import yaml
import pickle

import copy

# from baselines.marl_benchmark.evaluation.utils import load_checkpoint_dfs
from collections import defaultdict

import matplotlib.pyplot as plt

PALETTE = ['#A93226',  # red
           '#884EA0',  # purple
           '#2471A3',  # blue
           '#229954',  # green
           '#D4AC0D',  # yellow
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


def load_checkpoint_dfs(checkpoint_path, info):
    dfs = {}
    for agent in range(info['n_agents']):
        dfs[agent] = []

    # if off_road_mask[i] = 1, an off_road_event happened in episode i (in dfs)
    off_road_mask = []
    # if collision_mask[i] = 1, a collision happened in episode i (in dfs)
    collision_mask = []
    # if goal_reached_mask[i] = 1, every agent reached the goal in episode i (in dfs)
    goal_reached_mask = []

    times = os.listdir(Path(checkpoint_path))
    for time in times:
        if time == "plots":
            continue
        episodes = os.listdir(Path(checkpoint_path, time))
        for episode in episodes:
            off_road_mask.append(0)
            collision_mask.append(0)
            goal_reached_mask.append(1)
            for agent in range(info['n_agents']):
                csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
                dfs[agent].append(pd.read_csv(csv_path, index_col=0, header=None).T)
                if dfs[agent][-1]["Num_Collision"][1] > 0:
                    collision_mask[-1] = 1
                if dfs[agent][-1]["Num_Off_Road"][1] > 0:
                    off_road_mask[-1] = 1
                # if statement because Goal_Reached was added later to episode logs
                if "Goal_Reached" in dfs[agent][-1].keys():
                    if dfs[agent][-1]["Goal_Reached"][1] < 1:
                        goal_reached_mask[-1] = 0

    masks = {"off_road_mask": off_road_mask, "collision_mask": collision_mask, "goal_reached_mask": goal_reached_mask}

    return dfs, masks


def get_rewards(dfs,
                masks,
                goal_reached_reward: float = 300.0,
                ):
    """
    :param goal_reached_reward: Reward an agent gets upon reaching the goal position/region.
    :param dfs: Lists of all episode dataframes in a dict with agent keys.
    :param masks: Masks containing information whether an off-road event happened, a collision happened, or all the
    agents reached the goals.
    :return: Two lists of unfiltered and filtered total episode rewards.
    """
    off_road_mask = masks["off_road_mask"]
    collision_mask = masks["collision_mask"]
    goal_reached_mask = masks["goal_reached_mask"]

    rewards = [0] * len(dfs[0])
    filtered_rewards = [np.nan] * len(dfs[0])

    for i in range(len(dfs[0])):
        for agent in dfs.keys():
            rewards[i] += sum(dfs[agent][i]["Step_Reward"])
        # if no car went off-road and no collision happened and all cars reached the goal
        if not off_road_mask[i] and not collision_mask[i] and goal_reached_mask[i]:
            for agent in dfs.keys():
                filtered_rewards[i] = 0.0
                # sum rewards and subtract goal_reached_reward
                filtered_rewards[i] += sum(dfs[agent][i]["Step_Reward"]) - goal_reached_reward

    filtered_rewards = [x for x in filtered_rewards if x is not np.nan]

    return rewards, filtered_rewards


def get_poa(cent_rewards_filtered, decent_rewards_filtered):
    min_rew = min(min(cent_rewards_filtered), min(decent_rewards_filtered))
    max_rew = max(max(cent_rewards_filtered), max(decent_rewards_filtered))

    rew_range = max_rew - min_rew

    cent_rew_normalized = [(x - min_rew) / rew_range for x in cent_rewards_filtered]
    decent_rew_normalized = [(x - min_rew) / rew_range for x in decent_rewards_filtered]

    poa = np.mean(cent_rew_normalized) / np.mean(decent_rew_normalized)

    return poa, cent_rew_normalized, decent_rew_normalized


def get_info(checkpoint_path):
    info = {'max_episode_length': 0,
            'min_speed': 0.0,
            'max_speed': 16.0,
            'min_acceleration': 100.0,
            'max_acceleration': -100.0,
            'max_step_reward': 0.0,
            'min_step_reward': 0.0,
            'min_x_pos': 0.0,
            'max_x_pos': 0.0,
            'min_y_pos': 0.0,
            'max_y_pos': 0.0,
            'n_agents': None,
            'max_cost_com': -1e10,
            'min_cost_com': 1e10,
            'max_cost_per_acceleration': -1e10,
            'min_cost_per_acceleration': 1e10,
            'max_goal_improvement_reward': -1e10,
            'min_goal_improvement_reward': 1e10,
            'max_cost_per': -1e10,
            'min_cost_per': 1e10,
            }

    times = os.listdir(Path(checkpoint_path))
    for time in times:
        # print(time)
        if time == "plots":
            continue
        episodes = os.listdir(Path(checkpoint_path, time))
        print(episodes)
        for episode in episodes:
            info['n_agents'] = len(os.listdir(Path(checkpoint_path, time, episode)))
            for agent in range(info['n_agents']):
                csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
                df = pd.read_csv(csv_path, index_col=0, header=None).T
                info['max_episode_length'] = max(info['max_episode_length'], df.shape[0])
                info['max_speed'] = max(info['max_speed'], max(df["Speed"]))
                info['min_speed'] = min(info['min_speed'], min(df["Speed"]))
                info['max_acceleration'] = max(info['max_acceleration'], max(df["Acceleration"]))
                info['min_acceleration'] = min(info['min_acceleration'], min(df["Acceleration"]))
                info['max_step_reward'] = max(info['max_step_reward'], max(df["Step_Reward"]))
                info['min_step_reward'] = min(info['min_step_reward'], min(df["Step_Reward"]))
                info['max_x_pos'] = max(info['max_x_pos'], max(df["Xpos"]))
                info['min_x_pos'] = min(info['min_x_pos'], min(df["Xpos"]))
                info['max_y_pos'] = max(info['max_y_pos'], max(df["Ypos"]))
                info['min_y_pos'] = min(info['min_y_pos'], min(df["Ypos"]))
                # in some older evaluation run files, this information does not exist
                try:
                    cost_per = [list(df['cost_per_time'])[i] + list(df['cost_per_acceleration'])[i] -
                                list(df['goal_improvement_reward'])[i]
                                for i in range(len(list(df['cost_per_time'])))]
                    info['max_cost_com'] = max(info['max_cost_com'], max(df['cost_com']))
                    info['min_cost_com'] = min(info['min_cost_com'], min(df['cost_com']))
                    info['max_cost_per_acceleration'] = max(info['max_cost_per_acceleration'],
                                                            max(df['cost_per_acceleration']))
                    info['min_cost_per_acceleration'] = min(info['min_cost_per_acceleration'],
                                                            min(df['cost_per_acceleration']))
                    info['max_goal_improvement_reward'] = max(info['max_goal_improvement_reward'],
                                                              max(df['goal_improvement_reward']))
                    info['min_goal_improvement_reward'] = min(info['min_goal_improvement_reward'],
                                                              min(df['goal_improvement_reward']))
                    info['max_cost_per'] = max(info['max_cost_per'], max(cost_per))
                    info['min_cost_per'] = min(info['min_cost_per'], min(cost_per))
                except:
                    continue

    return info


def main(path):
    plt.rcParams['savefig.dpi'] = 800
    plt.rcParams["text.usetex"] = True

    plt.rcParams.update({'font.size': 14})
    plt.rcParams.update({'axes.titlesize': 18})
    plt.rcParams.update({'axes.labelsize': 16})
    plt.rcParams.update({'xtick.labelsize': 14})
    plt.rcParams.update({'ytick.labelsize': 14})
    plt.rcParams.update({'legend.fontsize': 16})
    plt.rcParams.update({'figure.titlesize': 14})

    eval_path = Path(path, "evaluation")
    fig_save_folder = Path(eval_path, "figures_report")
    fig_save_folder.mkdir(parents=True, exist_ok=True)

    with open(Path(eval_path, "data.pickle"), 'rb') as handle:
        data = pickle.load(handle)
    df_stats = pd.read_csv(Path(eval_path, "stats.csv"))

    alphas = []
    degrees = []
    alpha_degree_pairs = []
    for i, row in df_stats.iterrows():
        alpha_degree_pairs.append((row["alpha"], row["degree"]))
        alphas.append(row["alpha"])
        degrees.append(row["degree"])
    alpha_degree_pairs = list(set(alpha_degree_pairs))
    alphas = list(set(alphas))
    alphas = np.sort(alphas)
    degrees = list(set(degrees))

    goal_reached_threshold = 0.75

    poas = dict([(deg, []) for deg in degrees])
    cost_high = dict([(deg, []) for deg in degrees])
    cost_low = dict([(deg, []) for deg in degrees])
    cost_com_high = dict([(deg, []) for deg in degrees])
    cost_com_low = dict([(deg, []) for deg in degrees])
    cost_per_acc_high = dict([(deg, []) for deg in degrees])
    cost_per_acc_low = dict([(deg, []) for deg in degrees])
    cost_per_time_high = dict([(deg, []) for deg in degrees])
    cost_per_time_low = dict([(deg, []) for deg in degrees])
    rew_per_goal_high = dict([(deg, []) for deg in degrees])
    rew_per_goal_low = dict([(deg, []) for deg in degrees])

    opt_name = dict([(deg, []) for deg in degrees])
    worst_ne_name = dict([(deg, []) for deg in degrees])
    opt_run_path = dict([(deg, []) for deg in degrees])
    worst_ne_run_path = dict([(deg, []) for deg in degrees])
    opt_eval_path = dict([(deg, []) for deg in degrees])
    worst_ne_eval_path = dict([(deg, []) for deg in degrees])

    for degree in degrees:
        for alpha in alphas:
            if (alpha, degree) not in alpha_degree_pairs:
                continue
            selector = (df_stats["degree"] == degree) & (df_stats["alpha"] == alpha)
            df = df_stats[selector]
            cent_selector = (df["paradigm"] == "centralized")
            decent_selector = (df["paradigm"] == "decentralized")

            costs = []
            costs_com = []
            costs_per_acc = []
            costs_per_time = []
            rews_per_goal = []
            paradigms = []
            names = []
            run_paths = []
            eval_paths = []
            for i, row in df.iterrows():
                print(row["manual_exclude"] == 1.0)
                print(row["manual_exclude"])
                if row["goal_reached_perc"] < goal_reached_threshold or \
                        str(row["evaluation_path"]) == "nan" or row["manual_exclude"] == 1.0:
                    continue
                run_data = data[degree][alpha][row["name"]]
                cost = 0
                cost_com = 0
                cost_per_acc = 0
                cost_per_time = 0
                rew_per_goal = 0
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





            # sorted stuff
            costs_sorted = []
            names_sorted = []
            run_paths_sorted = []
            eval_paths_sorted = []
            paradigms_sorted = []
            costs_com_sorted = []
            costs_per_acc_sorted = []
            costs_per_time_sorted = []
            rews_per_goal_sorted = []

            # decentralized
            costs_sorted_decent = []
            names_sorted_decent = []
            run_paths_sorted_decent = []
            eval_paths_sorted_decent = []
            costs_com_sorted_decent = []
            costs_per_acc_sorted_decent = []
            costs_per_time_sorted_decent = []
            rews_per_goal_sorted_decent = []

            # centralized
            costs_sorted_cent = []
            names_sorted_cent = []
            run_paths_sorted_cent = []
            eval_paths_sorted_cent = []
            costs_com_sorted_cent = []
            costs_per_acc_sorted_cent = []
            costs_per_time_sorted_cent = []
            rews_per_goal_sorted_cent = []

            for _ in range(len(costs)):
                index = np.argmin(costs)
                costs_sorted.append(costs[index])
                names_sorted.append(names[index])
                run_paths_sorted.append(run_paths[index])
                eval_paths_sorted.append(eval_paths[index])
                paradigms_sorted.append(paradigms[index])
                costs_com_sorted.append((costs_com[index]))
                costs_per_acc_sorted.append(costs_per_acc[index])
                costs_per_time_sorted.append(costs_per_time[index])
                rews_per_goal_sorted.append(rews_per_goal[index])
                if paradigms[index] == "centralized":
                    costs_sorted_cent.append(costs[index])
                    names_sorted_cent.append(names[index])
                    run_paths_sorted_cent.append(run_paths[index])
                    eval_paths_sorted_cent.append(eval_paths[index])
                    costs_com_sorted_cent.append(costs_com[index])
                    costs_per_acc_sorted_cent.append(costs_per_acc[index])
                    costs_per_time_sorted_cent.append(costs_per_time[index])
                    rews_per_goal_sorted_cent.append(rews_per_goal[index])
                else:
                    costs_sorted_decent.append(costs[index])
                    names_sorted_decent.append(names[index])
                    run_paths_sorted_decent.append(run_paths[index])
                    eval_paths_sorted_decent.append(eval_paths[index])
                    costs_com_sorted_decent.append(costs_com[index])
                    costs_per_acc_sorted_decent.append(costs_per_acc[index])
                    costs_per_time_sorted_decent.append(costs_per_time[index])
                    rews_per_goal_sorted_decent.append(rews_per_goal[index])
                costs[index] = 1e20

            poas[degree].append(costs_sorted_decent[-1] / costs_sorted[0])
            cost_com_high[degree].append(costs_com_sorted_decent[-1])
            cost_com_low[degree].append(costs_com_sorted[0])
            cost_per_acc_high[degree].append(costs_per_acc_sorted_decent[-1])
            cost_per_acc_low[degree].append(costs_per_acc_sorted[0])
            cost_per_time_high[degree].append(costs_per_time_sorted_decent[-1])
            cost_per_time_low[degree].append(costs_per_time_sorted[0])
            rew_per_goal_high[degree].append(rews_per_goal_sorted_decent[-1])
            rew_per_goal_low[degree].append(rews_per_goal_sorted[0])
            cost_high[degree].append(costs_sorted_decent[-1])
            cost_low[degree].append(costs_sorted[0])

            opt_name[degree].append(names_sorted[0])
            worst_ne_name[degree].append(names_sorted_decent[-1])
            opt_run_path[degree].append(run_paths_sorted[0])
            worst_ne_run_path[degree].append(run_paths_sorted_decent[-1])
            opt_eval_path[degree].append(eval_paths_sorted[0])
            worst_ne_eval_path[degree].append(eval_paths_sorted_decent[-1])

            print(poas[degree][-1])

    for degree in degrees:
        infos = {"alpha": alphas, "opt_name": opt_name[degree], "worst_ne_name": worst_ne_name[degree],
                 "opt_run_path": opt_run_path[degree], "worst_ne_run_path": worst_ne_run_path[degree],
                 "opt_eval_path": opt_eval_path[degree], "worst_ne_eval_path": worst_ne_eval_path[degree]}
        print(infos)
        df_names = pd.DataFrame.from_dict(infos)
        df_names.to_csv(Path(fig_save_folder, "infos.csv"))

    better_cost_low = copy.deepcopy(cost_low)
    better_cost_com_low = copy.deepcopy(cost_com_low)
    better_cost_per_acc_low = copy.deepcopy(cost_per_acc_low)
    better_cost_per_time_low = copy.deepcopy(cost_per_time_low)
    better_rew_per_goal_low = copy.deepcopy(rew_per_goal_low)
    better_alphas = copy.deepcopy(alphas)
    better_poas = copy.deepcopy(poas)
    for degree in degrees:
        for i, alpha_eval in enumerate(alphas):
            for j, alpha_pol in enumerate(alphas):
                # print("---------------------------------------------")
                # print(cost_com_low[degree][i])
                # print(alpha_pol*cost_com_low[degree][j]/alpha_eval)
                # print("---------------------------------------------")
                cost_com_other_policy = alpha_pol / alpha_eval * cost_com_low[degree][j]
                cost_with_other_policy = cost_com_other_policy + cost_per_acc_low[degree][j] + \
                                         cost_per_time_low[degree][j] - rew_per_goal_low[degree][j]

                if cost_with_other_policy < better_cost_low[degree][i]:
                    better_cost_low[degree][i] = cost_with_other_policy
                    better_cost_com_low[degree][i] = alpha_pol / alpha_eval * cost_com_low[degree][j]
                    better_cost_per_acc_low[degree][i] = cost_per_acc_low[degree][j]
                    better_cost_per_time_low[degree][i] = cost_per_time_low[degree][j]
                    better_rew_per_goal_low[degree][i] = rew_per_goal_low[degree][j]
                    better_alphas[i] = alpha_pol
        better_poas[degree] = [cost_high[degree][i] / cl for i, cl in enumerate(better_cost_low[degree])]

    fig, ax = plt.subplots(figsize=(16, 9), tight_layout=True)
    for i, degree in enumerate(degrees):
        ax.plot(alphas, poas[degree],
                color=PALETTE[i], marker="x", markersize=10, label="degree = {}".format(degree),
                linestyle=(0, (5, 10)))
        ax.plot(alphas, better_poas[degree],
                color=PALETTE[i + 1], marker="x", markersize=10, label="with all optimal policies".format(degree),
                linestyle=(0, (5, 10)))

        # for i, alpha in enumerate(alphas):
        #     ax.text(1.05*alpha, poas[degree][i], r"$\alpha = {}$".format(alpha), fontsize=9)
    ax.legend()
    ax.set_xticks(alphas, minor=False)
    ax.xaxis.grid(True, which="major")
    # plt.xticks(alphas, alphas)
    plt.title(r"PoA for different $\alpha$'s")
    plt.grid()
    plt.ylabel("PoA")
    plt.xlabel(r"$\alpha$")
    fig.show()
    ax.set_xscale('log')
    plt.savefig(Path(fig_save_folder, "PoA_alpha_plot_w_better.png"), dpi=500)
    # plt.savefig(Path(fig_save_folder, "PoA_alpha_plot.pdf"))

    # for degree in degrees:
    #     fig, ax = plt.subplots(figsize=(14, 7), tight_layout=True)
    #     # plt.xticks(rotation=45)
    #     N = len(alphas)
    #     ind = np.arange(N)
    #     width = 0.25
    #     locations = [-0.15, 0.15]
    #     ax.bar(ind + locations[1], cost_com_high[degree], width, color=PALETTE[0])
    #     ax.bar(ind + locations[1], cost_per_time_high[degree], width, bottom=cost_com_high[degree], color=PALETTE[1])
    #     ax.bar(ind + locations[1], cost_per_acc_high[degree], width,
    #            bottom=np.add(cost_com_high[degree], cost_per_time_high[degree]), color=PALETTE[2])
    #     ax.legend(labels=[r'$J^{com}(\gamma)$', r'$J^{per, time}(\gamma)$', r'$J^{per, acc}(\gamma)$'])
    #     ax.bar(ind + locations[0], cost_com_low[degree], width, color=PALETTE[0])
    #     ax.bar(ind + locations[0], cost_per_time_low[degree], width, bottom=cost_com_low[degree], color=PALETTE[1])
    #     ax.bar(ind + locations[0], cost_per_acc_low[degree], width,
    #            bottom=np.add(cost_com_low[degree], cost_per_time_low[degree]), color=PALETTE[2])
    #     xticks = [i + loc for i in ind for loc in locations]
    #     xlabels = [r"$\alpha$ = {}, {}".format(a, x) for a in alphas for x in [r"$\gamma^*$", r"$\gamma_{NE}$"]]
    #     plt.xticks(xticks, xlabels, rotation=45)
    #     plt.ylabel("cost")
    #
    #     plt.savefig(Path(fig_save_folder, "costs_plot_degree_{}.png".format(degree)), dpi=500)
    #     # plt.savefig(Path(fig_save_folder, "costs_plot_degree_{}.pdf".format(degree)))

    for degree in degrees:
        fig, ax = plt.subplots(figsize=(14, 7), tight_layout=True)
        # plt.xticks(rotation=45)
        N = len(alphas)
        ind = np.arange(N)
        width = 0.05
        distance = 0.0
        locations = [-0.20, 0.20]
        # print(rew_per_goal_high[degree])

        ax.bar(ind + locations[1] - 2 * width - 2 * distance, cost_com_high[degree], width, color=PALETTE[0])
        ax.bar(ind + locations[1] - width - distance, cost_per_time_high[degree], width, color=PALETTE[1])
        ax.bar(ind + locations[1], cost_per_acc_high[degree], width, color=PALETTE[2])
        ax.bar(ind + locations[1] + width + distance, -np.array(rew_per_goal_high[degree]), width, color=PALETTE[3])
        ax.bar(ind + locations[1] + 2 * width + 2 * distance, cost_high[degree], width, color="grey")

        ax.bar(ind + locations[0] - 2 * width - 2 * distance, cost_com_low[degree], width, color=PALETTE[0])
        ax.bar(ind + locations[0] - width - distance, cost_per_time_low[degree], width, color=PALETTE[1])
        ax.bar(ind + locations[0], cost_per_acc_low[degree], width, color=PALETTE[2])
        ax.bar(ind + locations[0] + width + distance, -np.array(rew_per_goal_low[degree]), width, color=PALETTE[3])
        ax.bar(ind + locations[0] + 2 * width + 2 * distance, cost_low[degree], width, color="grey")

        # ax.bar(ind + locations[0], cost_com_low[degree], width, color=PALETTE[0])
        # ax.bar(ind + locations[0], cost_per_time_low[degree], width, bottom=cost_com_low[degree], color=PALETTE[1])
        # ax.bar(ind + locations[0], cost_per_acc_low[degree], width,
        #        bottom=np.add(cost_com_low[degree], cost_per_time_low[degree]), color=PALETTE[2])
        xticks = [i + loc for i in ind for loc in locations]
        xlabels = [r"$\alpha$ = {}, {}".format(a, x) for a in alphas for x in [r"$\gamma^*$", r"$\gamma_{NE}$"]]
        ax.legend(labels=[r'$J^{com}$', r'$J^{per, time}$', r'$J^{per, acc}$', r'$J^{per, prog}$', r'$J^{total}$'])
        plt.xticks(xticks, xlabels, rotation=90)
        # ax.set_yticks([0], minor=False)
        # ax.yaxis.grid(True, which="major")
        ax.axhline(0, linestyle='-', linewidth=1, color='k')  # horizontal line
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)

        plt.ylabel(r"total cost  $J(\gamma)$")

        plt.savefig(Path(fig_save_folder, "costs_plot_degree_{}_1.png".format(degree)), dpi=500)
        # plt.savefig(Path(fig_save_folder, "costs_plot_degree_{}.pdf".format(degree)))

    for degree in degrees:
        fig, ax = plt.subplots(figsize=(14, 7), tight_layout=True)
        # plt.xticks(rotation=45)
        N = len(alphas)
        ind = np.arange(N)
        width = 0.05
        distance = 0.0
        locations = [-0.20, 0.20]
        # print(rew_per_goal_high[degree])

        for i in ind:
            ax.text(i + locations[0] - 2 * width - 2 * distance, -20, "{}".format(better_alphas[i]), fontsize=7)

        ax.bar(ind + locations[1] - 2 * width - 2 * distance, cost_com_high[degree], width, color=PALETTE[0])
        ax.bar(ind + locations[1] - width - distance, cost_per_time_high[degree], width, color=PALETTE[1])
        ax.bar(ind + locations[1], cost_per_acc_high[degree], width, color=PALETTE[2])
        ax.bar(ind + locations[1] + width + distance, -np.array(rew_per_goal_high[degree]), width, color=PALETTE[3])
        ax.bar(ind + locations[1] + 2 * width + 2 * distance, cost_high[degree], width, color="grey")

        ax.bar(ind + locations[0] - 2 * width - 2 * distance, better_cost_com_low[degree], width, color=PALETTE[0])
        ax.bar(ind + locations[0] - width - distance, better_cost_per_time_low[degree], width, color=PALETTE[1])
        ax.bar(ind + locations[0], better_cost_per_acc_low[degree], width, color=PALETTE[2])
        ax.bar(ind + locations[0] + width + distance, -np.array(better_rew_per_goal_low[degree]), width,
               color=PALETTE[3])
        ax.bar(ind + locations[0] + 2 * width + 2 * distance, better_cost_low[degree], width, color="grey")

        # ax.bar(ind + locations[0], cost_com_low[degree], width, color=PALETTE[0])
        # ax.bar(ind + locations[0], cost_per_time_low[degree], width, bottom=cost_com_low[degree], color=PALETTE[1])
        # ax.bar(ind + locations[0], cost_per_acc_low[degree], width,
        #        bottom=np.add(cost_com_low[degree], cost_per_time_low[degree]), color=PALETTE[2])
        xticks = [i + loc for i in ind for loc in locations]
        xlabels = [r"$\alpha$ = {}, {}".format(a, x) for a in alphas for x in [r"$\gamma^*$", r"$\gamma_{NE}$"]]
        ax.legend(labels=[r'$J^{com}$', r'$J^{per, time}$', r'$J^{per, acc}$', r'$J^{per, prog}$', r'$J^{total}$'])
        plt.xticks(xticks, xlabels, rotation=90)
        # ax.set_yticks([0], minor=False)
        # ax.yaxis.grid(True, which="major")
        ax.axhline(0, linestyle='-', linewidth=1, color='k')  # horizontal line
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)

        plt.ylabel(r"total cost  $J(\gamma)$")

        plt.savefig(Path(fig_save_folder, "costs_plot_degree_{}_better_opt_policies.png".format(degree)), dpi=500)


    ################################# plot for report #####################################
    for degree in degrees:
        fig, ax = plt.subplots(figsize=(14, 7), tight_layout=True)
        # plt.xticks(rotation=45)
        N = len(alphas)
        ind = np.arange(N)
        width = 0.05
        distance = 0.0
        locations = [-0.20, 0.20]
        # print(rew_per_goal_high[degree])

        print(better_poas[degree])
        for i in ind:
            # ax.text(i + locations[0] - 2 * width - 2 * distance, -20, "{}".format(better_alphas[i]), fontsize=7)
            ax.text(i + locations[0] - 2 * width - 2 * distance - 0.05, -160,
                    r"$\textrm{PoA}\approx \ $"+r"${}$".format(np.round(better_poas[degree][i], 3)), fontsize=12)

        ax.bar(ind + locations[1] - 2 * width - 2 * distance, cost_com_high[degree], width, color=PALETTE[0])
        ax.bar(ind + locations[1] - width - distance, cost_per_time_high[degree], width, color=PALETTE[1])
        ax.bar(ind + locations[1], cost_per_acc_high[degree], width, color=PALETTE[2])
        ax.bar(ind + locations[1] + width + distance, -np.array(rew_per_goal_high[degree]), width, color=PALETTE[3])
        ax.bar(ind + locations[1] + 2 * width + 2 * distance, cost_high[degree], width, color="grey")

        ax.bar(ind + locations[0] - 2 * width - 2 * distance, better_cost_com_low[degree], width, color=PALETTE[0])
        ax.bar(ind + locations[0] - width - distance, better_cost_per_time_low[degree], width, color=PALETTE[1])
        ax.bar(ind + locations[0], better_cost_per_acc_low[degree], width, color=PALETTE[2])
        ax.bar(ind + locations[0] + width + distance, -np.array(better_rew_per_goal_low[degree]), width,
               color=PALETTE[3])
        ax.bar(ind + locations[0] + 2 * width + 2 * distance, better_cost_low[degree], width, color="grey")

        # ax.bar(ind + locations[0], cost_com_low[degree], width, color=PALETTE[0])
        # ax.bar(ind + locations[0], cost_per_time_low[degree], width, bottom=cost_com_low[degree], color=PALETTE[1])
        # ax.bar(ind + locations[0], cost_per_acc_low[degree], width,
        #        bottom=np.add(cost_com_low[degree], cost_per_time_low[degree]), color=PALETTE[2])
        xticks = [i + loc for i in ind for loc in locations]
        xlabels = [r"$\alpha = {}$, {}".format(a, x) for a in alphas for x in [r"$\gamma^*$", r"$\gamma_{NE}$"]]
        ax.legend(labels=[r'$C^{com}$', r'$C^{per, time}$', r'$C^{per, acc}$', r'$C^{per, prog}$', r'$C^{total}$'])
        plt.xticks(xticks, xlabels, rotation=90)
        # ax.set_yticks([0], minor=False)
        # ax.yaxis.grid(True, which="major")
        ax.axhline(0, linestyle='-', linewidth=1, color='k')  # horizontal line
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)

        plt.ylim([-200, 700])

        plt.ylabel(r"$\textrm{total social cost} \ C(\gamma)$")

        plt.savefig(Path(fig_save_folder, "costs_plot_degree_report.png"), dpi=800)
        plt.savefig(Path(fig_save_folder, "costs_plot_degree_report.pdf"))


def parse_args():
    parser = argparse.ArgumentParser("")

    parser.add_argument("-p",
                        "--path",
                        type=str,
                        required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(path=args.path)
