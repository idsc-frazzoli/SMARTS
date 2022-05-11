from pathlib import Path
import pandas as pd
import yaml
import os

from typing import List, Union

import numpy as np


# cost 07
# pseudo-lexicographic cost (additional clearance cost): <com_cost, per_cost>, where
# per_cost = <off-road, goal reached, (time cost, closer to goal)>
# includes proposed changes from 14.04.2022 meeting (clearance cost shape, acceleration cost shape)
def get_detailed_reward_adapter(**kwargs):
    alpha = kwargs.get("alpha", 1.0)
    degree = kwargs.get("degree", 2.0)
    asym_cost = kwargs.get("asym_cost", True)

    def func(position: List[float],
             other_positions: Union[List[List[float]]],
             agent_id: int,
             acceleration: float,
             goal_distance: float,
             last_goal_distance: Union[float, None]) -> dict:

        cost_com, cost_per = 0.0, 0.0

        goal_improvement_reward = 0.0

        if asym_cost:
            if agent_id == 0:
                time_penalty = 1.0
            else:
                time_penalty = 5.0
        else:
            time_penalty = 2.0

        cost_per += time_penalty

        if last_goal_distance is not None:
            goal_improvement = last_goal_distance - goal_distance
            if goal_improvement > 0:
                # lexicost3
                goal_improvement_reward += 1 * min(goal_improvement, 2)

        if other_positions:
            safety_dist = float(25)  # [m]
            for pos in other_positions:
                # calculate distance to neighbor vehicle
                dist = float(np.linalg.norm(np.array(position) - np.array(pos)))
                print(dist)
                cost_com += 0.05 * (np.abs(float(dist) - safety_dist) ** degree) if dist < safety_dist else 0.0

        acc_penalty = 5.0 * (1 - np.exp(-0.007 * (acceleration - 5) ** 2)) if acceleration > 5 else 0.0
        cost_per += acc_penalty

        cost_com /= alpha

        costs = {"cost_com": cost_com,
                 "cost_per": cost_per,
                 "goal_improvement_reward": goal_improvement_reward,
                 }

        return costs

    return func

def add_rewards_to_csv(time_path, cost):

    # get episode dataframes
    dfs = {agent: pd.read_csv(Path(time_path, agent), index_col=0, header=None).T
           for agent in os.listdir(Path(time_path))}
    lens = {agent: dfs[agent].shape[0] for agent in dfs.keys()}
    positions = {agent: [[x, y] for x, y in zip(dfs[agent]["Xpos"], dfs[agent]["Ypos"])]
                 for agent in dfs.keys()}

    cost_com = dict([(agent, []) for agent in dfs.keys()])
    cost_per = dict([(agent, []) for agent in dfs.keys()])
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
            cost_per[agent].append(costs["cost_per"])
            goal_improvement_reward[agent].append(costs["goal_improvement_reward"])

        dfs[agent]["cost_com"] = cost_com[agent]
        dfs[agent]["cost_per"] = cost_per[agent]
        dfs[agent]["goal_improvement_reward"] = goal_improvement_reward[agent]

        df_transpose = dfs[agent].T
        df_transpose.to_csv(Path(time_path, agent))


def main():
    path = Path(
        "./log/results/run/testfolder_merge65_lanes42/evaluation/evaluation_runs/alpha01_degree2/decent/PPO_FrameStack_71fde_00000_0_2022-05-05_05-51-45/checkpoint_000160/1652111554/episode_0")
    dfs = {agent: pd.read_csv(Path(path, agent), index_col=0, header=None).T
           for agent in os.listdir(Path(path))}
    lens = {agent: dfs[agent].shape[0] for agent in dfs.keys()}
    positions = {agent: [[x, y] for x, y in zip(dfs[agent]["Xpos"], dfs[agent]["Ypos"])]
                 for agent in dfs.keys()}

    conf = {'alpha': 0.001, 'degree': 1.5, "asym_cost": False}

    cost = get_detailed_reward_adapter(**conf)

    # costs = ["cost_com", "cost_per", "goal_improvement_reward"]
    cost_com = dict([(agent, []) for agent in dfs.keys()])
    cost_per = dict([(agent, []) for agent in dfs.keys()])
    goal_improvement_reward = dict([(agent, []) for agent in dfs.keys()])

    # dists = dict([(agent, []) for agent in dfs.keys()])

    agents = list(dfs.keys())
    for agent in agents:
        for time_step in range(1, lens[agent]+1):
            position = positions[agent][time_step-1]
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
            cost_per[agent].append(costs["cost_per"])
            goal_improvement_reward[agent].append(costs["goal_improvement_reward"])

            # if len(agents) == 2 and other_positions and :
            #     dists[agent].append(np.linalg.norm(positions[agent][time_step-1], other_positions[0]))

        dfs[agent]["cost_com"] = cost_com[agent]
        dfs[agent]["cost_per"] = cost_per[agent]
        dfs[agent]["goal_improvement_reward"] = goal_improvement_reward[agent]

        dfT = dfs[agent].T
        dfT.to_csv(Path(path, agent))

    df = pd.read_csv(path, index_col=0, header=None).T

    with open("./agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_cent_1_1.yaml", 'r') as stream:
        config_yaml = yaml.safe_load(stream)
        print()

    ones = [1 for _ in range(len(df["Speed"]))]

    df["ones"] = ones

    dfT = df.T
    dfT.to_csv(path)

    pass


if __name__ == "__main__":
    main()
