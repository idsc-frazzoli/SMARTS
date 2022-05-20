
import argparse
from pathlib import Path
import os

import pandas as pd
import math
import numpy as np
import yaml
import pickle

import copy

from baselines.marl_benchmark.evaluation.utils import get_info, get_poa, get_rewards, load_checkpoint_dfs
from collections import defaultdict

def main(eval_path):
    df_all_conv = pd.read_csv(Path(eval_path, "stats.csv"))
    # delete unnamed columns
    df_all_conv = df_all_conv.loc[:, ~df_all_conv.columns.str.contains('^Unnamed')]

    data = defaultdict(dict)
    data_list = ["mean_step_reward",
                 "episode_step_reward",
                 "mean_cost_com",
                 "episode_cost_com",
                 "mean_cost_per_time",
                 "episode_cost_per_time",
                 "mean_cost_per_acceleration",
                 "episode_cost_per_acceleration",
                 "mean_goal_improvement_reward",
                 "episode_goal_improvement_reward",
                 ]
    alpha_degree_pairs = []
    for i, row in df_all_conv.iterrows():
        alpha_degree_pairs.append((row["alpha"], row["degree"]))
    alpha_degree_pairs = list(set(alpha_degree_pairs))
    for adp in alpha_degree_pairs:
        data[adp[1]][adp[0]] = dict()

    episode_agent_dict = dict([(x, []) for x in data_list])

    for i, row in df_all_conv.iterrows():
        if str(row["evaluation_path"]) == "nan":
            continue

        alpha = float(row["alpha"])
        degree = float(row["degree"])

        run_stats = dict([(agent, copy.deepcopy(episode_agent_dict)) for agent in range(int(row["n_agents"]))])
        for checkpoint in os.listdir(row["evaluation_path"]):
            checkpoint_path = Path(row["evaluation_path"], checkpoint)
            info = get_info(checkpoint_path)
            dfs, masks = load_checkpoint_dfs(checkpoint_path, info)
            print(checkpoint_path)

            goal_reached_mask = masks["goal_reached_mask"]

            for agent in dfs.keys():
                for episode in range(len(goal_reached_mask)):
                    if goal_reached_mask[episode]:
                        run_stats[agent]["mean_step_reward"].append(np.mean(dfs[agent][episode]["Step_Reward"]))
                        run_stats[agent]["episode_step_reward"].append(sum(dfs[agent][episode]["Step_Reward"]))
                        run_stats[agent]["mean_cost_com"].append(np.mean(dfs[agent][episode]["cost_com"]))
                        run_stats[agent]["episode_cost_com"].append(sum(dfs[agent][episode]["cost_com"]))
                        run_stats[agent]["mean_cost_per_time"].append(np.mean(dfs[agent][episode]["cost_per_time"]))
                        run_stats[agent]["episode_cost_per_time"].append(sum(dfs[agent][episode]["cost_per_time"]))
                        run_stats[agent]["mean_cost_per_acceleration"].append(np.mean(dfs[agent][episode]["cost_per_acceleration"]))
                        run_stats[agent]["episode_cost_per_acceleration"].append(sum(dfs[agent][episode]["cost_per_acceleration"]))
                        run_stats[agent]["mean_goal_improvement_reward"].append(np.mean(dfs[agent][episode]["goal_improvement_reward"]))
                        run_stats[agent]["episode_goal_improvement_reward"].append(sum(dfs[agent][episode]["goal_improvement_reward"]))

        data[degree][alpha][row["name"]] = dict(run_stats)
    data = dict(data)
    # with open(Path(eval_path, "data.yaml"), 'w') as outfile:
    #     yaml.dump(data, outfile, default_flow_style=False)

    # save data as pickle
    with open(Path(eval_path, "data.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
