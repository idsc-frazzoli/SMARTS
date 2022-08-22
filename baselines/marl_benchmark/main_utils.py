import baselines.marl_benchmark.plotting.utils as plotting_utils
from typing import List, Tuple, Union
import numpy as np
from pathlib import Path
import os
import pandas as pd
from collections import defaultdict
import pickle
import copy

import warnings

from baselines.marl_benchmark.evaluation.utils import get_info, get_poa, get_rewards, load_checkpoint_dfs


# cost 07
# pseudo-lexicographic cost (additional clearance cost): <com_cost, per_cost>, where
# per_cost = <off-road, goal reached, (time cost, closer to goal)>
# includes proposed changes from 14.04.2022 meeting (clearance cost shape, acceleration cost shape)
def get_detailed_reward_adapter(**kwargs):
    com_cost_coef = kwargs.get("com_cost_coef", 0.05)
    acc_cost_coef = kwargs.get("acc_cost_coef", 5.0)
    safety_dist = kwargs.get("safety_dist", 15.0)
    acc_thres = kwargs.get("acc_thres", 5.0)
    acc_cost_flatness = kwargs.get("acc_cost_flatness", 0.007)
    goal_reached_reward = kwargs.get("goal_reached_reward", 300.0)
    off_road_cost = kwargs.get("off_road_cost", 500.0)
    collision_cost = kwargs.get("collision_cost", 1000.0)

    if com_cost_coef == 0.05:
        warnings.warn("Using default value for com_cost_coef")
    if acc_cost_coef == 5.0:
        warnings.warn("Using default value for acc_cost_coef")
    if safety_dist == 15.0:
        warnings.warn("Using default value for safety_dist")
    if acc_thres == 5.0:
        warnings.warn("Using default value for acc_thres")
    if acc_cost_flatness == 0.007:
        warnings.warn("Using default value for acc_cost_flatness")
    if goal_reached_reward == 300.0:
        warnings.warn("Using default value for goal_reached_reward")
    if off_road_cost == 500.0:
        warnings.warn("Using default value for off_road_cost")
    if collision_cost == 1000.0:
        warnings.warn("Using default value for collision_cost")

    # this is better since we want an error (problem: doesn't work with older config files)
    alpha = kwargs["alpha"]
    degree = kwargs["degree"]
    asym_cost = kwargs["asym_cost"]


    def func(position: List[float],
             other_positions: Union[List[List[float]]],
             agent_id: int,
             acceleration: float,
             goal_distance: float,
             last_goal_distance: Union[float, None]) -> dict:

        cost_com, cost_per_time, cost_per_acceleration = 0.0, 0.0, 0.0

        goal_improvement_reward = 0.0

        if asym_cost:
            if agent_id == 0:
                time_penalty = 1.0
            else:
                time_penalty = 5.0
        else:
            time_penalty = 2.0

        cost_per_time += time_penalty

        if last_goal_distance is not None:
            goal_improvement = last_goal_distance - goal_distance
            if goal_improvement > 0:
                # lexicost3
                goal_improvement_reward += 1 * min(goal_improvement, 2)

        if other_positions:
            # safety_dist = float(15)  # [m]
            for pos in other_positions:
                # calculate distance to neighbor vehicle
                dist = float(np.linalg.norm(np.array(position) - np.array(pos)))
                cost_com += com_cost_coef * (np.abs(float(dist) - safety_dist) ** degree) if dist < safety_dist else 0.0

        acc_penalty = acc_cost_coef * (1 - np.exp(-acc_cost_flatness * (acceleration - acc_thres) ** 2)) \
            if acceleration > acc_thres else 0.0
        cost_per_acceleration += acc_penalty

        cost_com /= alpha

        # # alternatively
        # cost_per_acceleration *= alpha

        costs = {"cost_com": cost_com,
                 "cost_per_time": cost_per_time,
                 "cost_per_acceleration": cost_per_acceleration,
                 "goal_improvement_reward": goal_improvement_reward,
                 }

        return costs

    return func


def add_rewards_to_csv(episode_path, cost):
    # get episode dataframes
    dfs = {agent: pd.read_csv(Path(episode_path, agent), index_col=0, header=None).T
           for agent in os.listdir(Path(episode_path))}
    lens = {agent: dfs[agent].shape[0] for agent in dfs.keys()}
    positions = {agent: [[x, y] for x, y in zip(dfs[agent]["Xpos"], dfs[agent]["Ypos"])]
                 for agent in dfs.keys()}

    cost_com = dict([(agent, []) for agent in dfs.keys()])
    cost_per_time = dict([(agent, []) for agent in dfs.keys()])
    cost_per_acceleration = dict([(agent, []) for agent in dfs.keys()])
    goal_improvement_reward = dict([(agent, []) for agent in dfs.keys()])

    agents = list(dfs.keys())
    for agent in agents:
        for time_step in range(1, lens[agent] + 1):
            position = positions[agent][time_step - 1]
            acceleration = dfs[agent]["Acceleration"][time_step]
            agent_id = int(agent[-5])
            goal_distance = dfs[agent]["GDistance"][time_step]
            if time_step != 1:
                last_goal_distance = dfs[agent]["GDistance"][time_step - 1]
            else:
                last_goal_distance = None
            other_positions = []
            for other_agent in agents:
                if other_agent != agent and time_step < lens[other_agent]:
                    other_positions.append(positions[other_agent][time_step])

            costs = cost(position,
                         other_positions,
                         agent_id,
                         acceleration,
                         goal_distance,
                         last_goal_distance)

            cost_com[agent].append(costs["cost_com"])
            cost_per_time[agent].append(costs["cost_per_time"])
            cost_per_acceleration[agent].append(costs["cost_per_acceleration"])
            goal_improvement_reward[agent].append(costs["goal_improvement_reward"])

        dfs[agent]["cost_com"] = cost_com[agent]
        dfs[agent]["cost_per_time"] = cost_per_time[agent]
        dfs[agent]["cost_per_acceleration"] = cost_per_acceleration[agent]
        dfs[agent]["goal_improvement_reward"] = goal_improvement_reward[agent]

        df_transpose = dfs[agent].T
        df_transpose.to_csv(Path(episode_path, agent))


def get_config_yaml_path(config_path, alpha_degree, paradigm):
    alpha = alpha_degree.split("_")[0][5:]
    degree = alpha_degree.split("_")[1][6:]
    yaml_name = "baseline-lane-control_" + paradigm + "_" + alpha + "_" + degree + ".yaml"
    yaml_path = Path(config_path, yaml_name)
    return yaml_path


def get_plotting_paths(path):
    paradigms = {}
    paths = {}
    for x in list(os.walk(Path(path), topdown=True)):
        if "evaluation" not in x[0] and x[0].split('/')[-1] in ('decent', 'cent'):
            paths[x[0]] = list_all_run_paths(x[0])
            paradigms[x[0]] = x[0].split('/')[-1]

    # remove paths containing "logs_and_plots"
    paths = {key: [x for x in paths[key] if "logs_and_plots" not in x] for key in paths.keys()}

    return paths, paradigms


def get_convergence_paths(path,
                          identification_prefix: str = "PPO_FrameStack"):
    run_paths = []
    for x in list(os.walk(Path(path), topdown=True)):
        if "evaluation" not in x[0] and identification_prefix in x[0].split('/')[-1]:
            run_paths.append('/'.join(x[0].split('/')[:-1]))

    # remove paths containing "logs_and_plots"
    run_paths = [x for x in run_paths if "logs_and_plots" not in x]

    return list(set(run_paths))


def list_all_run_paths(path: str,
                       identification_prefix: str = "PPO_FrameStack") -> List[str]:
    all_paths = [x[0] for x in list(os.walk(Path(path), topdown=True))
                 if identification_prefix in x[0].split('/')[-1]]

    return all_paths


def get_alpha_degree(evaluation_path):
    a = evaluation_path.split("alpha")[-1]
    d = evaluation_path.split("degree")[-1]
    a = a.split("_")[0]
    d = d.split("/")[0]
    alpha = float(".".join(a.split("p")))
    degree = float(".".join(d.split("p")))
    return alpha, degree


def add_evaluation_paths(eval_path):
    df_all_conv = pd.read_csv(Path(eval_path, "convergence", "all_convergence.csv"))
    # delete unnamed columns
    df_all_conv = df_all_conv.loc[:, ~df_all_conv.columns.str.contains('^Unnamed')]

    evaluation_paths = []
    alphas = []
    degrees = []
    evaluation_runs_path = Path(eval_path, "evaluation_runs")
    for i, row in df_all_conv.iterrows():
        name = row["name"]
        evaluation_path = False
        for x in os.walk(evaluation_runs_path):
            if not row["human_converged"] or row["first_stable"] in ["CSV_PARSE_ERROR", "CSV_EMPTY_DATA_ERROR"]:
                break
            if x[0].split("/")[-1] == name:
                evaluation_path = x[0]
                break
        alpha, degree = get_alpha_degree(row["run_path"])
        alphas.append(alpha)
        degrees.append(degree)
        # evaluation_path = [x[0] for x in list(os.walk(evaluation_runs_path)) if x[0].split("/")[-1] == name]
        if not evaluation_path:
            evaluation_path = None
        # evaluation_path = evaluation_path[0]

        evaluation_paths.append(evaluation_path)
        print(name)
        # print(i)

    df_all_conv = df_all_conv.assign(evaluation_path=evaluation_paths)
    df_all_conv = df_all_conv.assign(alpha=alphas)
    df_all_conv = df_all_conv.assign(degree=degrees)

    df_all_conv.to_csv(Path(eval_path, "convergence", "all_convergence_eval_paths.csv"))


def make_stats(eval_path):
    df_all_conv = pd.read_csv(Path(eval_path, "convergence", "all_convergence_eval_paths.csv"))
    # delete unnamed columns
    df_all_conv = df_all_conv.loc[:, ~df_all_conv.columns.str.contains('^Unnamed')]

    goal_reached_perc = []
    collision_perc = []
    off_road_perc = []

    n_agents = []

    for i, row in df_all_conv.iterrows():
        if str(row["evaluation_path"]) != "nan":
            print(row["evaluation_path"])
            for checkpoint in os.listdir(row["evaluation_path"]):
                checkpoint_path = Path(row["evaluation_path"], checkpoint)
                info = get_info(checkpoint_path)
                dfs, masks = load_checkpoint_dfs(checkpoint_path, info)
                goal_reached_perc.append(sum(masks["goal_reached_mask"]) / len(masks["goal_reached_mask"]))
                collision_perc.append(sum(masks["collision_mask"]) / len(masks["collision_mask"]))
                off_road_perc.append(sum(masks["off_road_mask"]) / len(masks["off_road_mask"]))
                n_agents.append(info["n_agents"])
        else:
            goal_reached_perc.append(None)
            collision_perc.append(None)
            off_road_perc.append(None)
            n_agents.append(None)

    df_all_conv = df_all_conv.assign(goal_reached_perc=goal_reached_perc)
    df_all_conv = df_all_conv.assign(collision_perc=collision_perc)
    df_all_conv = df_all_conv.assign(off_road_perc=off_road_perc)
    df_all_conv = df_all_conv.assign(n_agents=n_agents)

    df_all_conv.to_csv(Path(eval_path, "stats.csv"))

def make_data_pickle(eval_path):
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
                        run_stats[agent]["mean_cost_per_acceleration"].append(
                            np.mean(dfs[agent][episode]["cost_per_acceleration"]))
                        run_stats[agent]["episode_cost_per_acceleration"].append(
                            sum(dfs[agent][episode]["cost_per_acceleration"]))
                        run_stats[agent]["mean_goal_improvement_reward"].append(
                            np.mean(dfs[agent][episode]["goal_improvement_reward"]))
                        run_stats[agent]["episode_goal_improvement_reward"].append(
                            sum(dfs[agent][episode]["goal_improvement_reward"]))

        data[degree][alpha][row["name"]] = dict(run_stats)
    data = dict(data)
    # with open(Path(eval_path, "data.yaml"), 'w') as outfile:
    #     yaml.dump(data, outfile, default_flow_style=False)

    # save data as pickle
    with open(Path(eval_path, "data.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


############################################

def add_evaluation_paths_new(eval_path):
    df_info_logs = pd.read_csv(Path(eval_path, "info_logs.csv"))
    # delete unnamed columns
    df_info_logs = df_info_logs.loc[:, ~df_info_logs.columns.str.contains('^Unnamed')]

    evaluation_paths = []
    evaluation_runs_path = Path(eval_path, "evaluation_runs")
    for i, row in df_info_logs.iterrows():
        name = row["name"]
        evaluation_path = False
        for x in os.walk(evaluation_runs_path):
            if row["paradigm"] in ["CSV_PARSE_ERROR", "CSV_EMPTY_DATA_ERROR"]:
                break
            if x[0].split("/")[-1] == name:
                evaluation_path = x[0]
                break
        # evaluation_path = [x[0] for x in list(os.walk(evaluation_runs_path)) if x[0].split("/")[-1] == name]
        if not evaluation_path:
            evaluation_path = None
        # evaluation_path = evaluation_path[0]

        evaluation_paths.append(evaluation_path)
        # print(i)

    df_info_logs = df_info_logs.assign(evaluation_path=evaluation_paths)

    df_info_logs.to_csv(Path(eval_path, "all_eval_paths.csv"))


def make_stats_new(eval_path):
    df_info_logs = pd.read_csv(Path(eval_path, "all_eval_paths.csv"))
    # delete unnamed columns
    df_info_logs = df_info_logs.loc[:, ~df_info_logs.columns.str.contains('^Unnamed')]

    goal_reached_perc = []
    collision_perc = []
    off_road_perc = []

    n_agents = []

    for i, row in df_info_logs.iterrows():
        if str(row["evaluation_path"]) != "nan":
            # print(row["evaluation_path"])
            for checkpoint in os.listdir(row["evaluation_path"]):
                checkpoint_path = Path(row["evaluation_path"], checkpoint)
                info = get_info(checkpoint_path)
                dfs, masks = load_checkpoint_dfs(checkpoint_path, info)
                goal_reached_perc.append(sum(masks["goal_reached_mask"]) / len(masks["goal_reached_mask"]))
                collision_perc.append(sum(masks["collision_mask"]) / len(masks["collision_mask"]))
                off_road_perc.append(sum(masks["off_road_mask"]) / len(masks["off_road_mask"]))
                n_agents.append(info["n_agents"])
        else:
            goal_reached_perc.append(None)
            collision_perc.append(None)
            off_road_perc.append(None)
            n_agents.append(None)

    df_info_logs = df_info_logs.assign(goal_reached_perc=goal_reached_perc)
    df_info_logs = df_info_logs.assign(collision_perc=collision_perc)
    df_info_logs = df_info_logs.assign(off_road_perc=off_road_perc)
    df_info_logs = df_info_logs.assign(n_agents=n_agents)

    df_info_logs.to_csv(Path(eval_path, "stats.csv"))


def make_data_pickle_new(eval_path):
    df_stats = pd.read_csv(Path(eval_path, "stats.csv"))
    # delete unnamed columns
    df_stats = df_stats.loc[:, ~df_stats.columns.str.contains('^Unnamed')]

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

    data["decentralized"], data["centralized"] = dict(), dict()

    episode_agent_dict = dict([(x, []) for x in data_list])

    for i, row in df_stats.iterrows():
        if str(row["evaluation_path"]) == "nan":
            continue

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
                        run_stats[agent]["mean_cost_per_acceleration"].append(
                            np.mean(dfs[agent][episode]["cost_per_acceleration"]))
                        run_stats[agent]["episode_cost_per_acceleration"].append(
                            sum(dfs[agent][episode]["cost_per_acceleration"]))
                        run_stats[agent]["mean_goal_improvement_reward"].append(
                            np.mean(dfs[agent][episode]["goal_improvement_reward"]))
                        run_stats[agent]["episode_goal_improvement_reward"].append(
                            sum(dfs[agent][episode]["goal_improvement_reward"]))

        data[row["paradigm"]][row["name"]] = dict(run_stats)
    data = dict(data)
    # with open(Path(eval_path, "data.yaml"), 'w') as outfile:
    #     yaml.dump(data, outfile, default_flow_style=False)

    # save data as pickle
    with open(Path(eval_path, "data.pickle"), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

