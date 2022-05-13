import baselines.marl_benchmark.plotting.utils as plotting_utils
from typing import List, Tuple, Union
import numpy as np
from pathlib import Path
import os
import pandas as pd


# cost 07
# pseudo-lexicographic cost (additional clearance cost): <com_cost, per_cost>, where
# per_cost = <off-road, goal reached, (time cost, closer to goal)>
# includes proposed changes from 14.04.2022 meeting (clearance cost shape, acceleration cost shape)
def get_detailed_reward_adapter(**kwargs):
    alpha = kwargs.get("alpha", 1.0)
    degree = kwargs.get("degree", 2.0)
    asym_cost = kwargs.get("asym_cost", False)
    com_cost_coef = kwargs.get("com_cost_coef", 0.05)
    acc_cost_coef = kwargs.get("acc_cost_coef", 5.0)
    safety_dist = kwargs.get("safety_dist", 15.0)
    acc_thres = kwargs.get("acc_thres", 5.0)
    acc_cost_flatness = kwargs.get("acc_cost_flatness", 0.007)

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
