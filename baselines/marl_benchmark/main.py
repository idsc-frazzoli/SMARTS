import pandas as pd
import numpy as np

import sys
# sys.path.append("~/SMARTS/baselines/marl_benchmark")
# sys.path.append("~/SMARTS/baselines")
# sys.path.append("/home/nando/SMARTS")
import yaml

from baselines.marl_benchmark.plotting import plotting
from baselines.marl_benchmark.evaluation import training_analysis
from baselines.marl_benchmark import evaluate
from baselines.marl_benchmark.evaluation import evaluation

import argparse
import os
import shutil
import baselines.marl_benchmark.plotting.utils as plotting_utils
from typing import List, Tuple, Union
from pathlib import Path

from baselines.marl_benchmark.wrappers.rllib.frame_stack import FrameStack

from timeit import default_timer as timer

PARADIGM_MAP = {"cent": "centralized", "decent": "decentralized"}


# start = timer()
# end = timer()
# print('Time to get paths: {}'.format(end - start))

# def get_alpha_degree(alpha_degree: str) -> Tuple[float, float]:
#     pass


# TODO: throw all these functions in utils somewhere...

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
            safety_dist = float(15)  # [m]
            for pos in other_positions:
                # calculate distance to neighbor vehicle
                dist = float(np.linalg.norm(np.array(position) - np.array(pos)))
                cost_com += 0.05 * (np.abs(float(dist) - safety_dist) ** degree) if dist < safety_dist else 0.0

        acc_penalty = 5.0 * (1 - np.exp(-0.007 * (acceleration - 5) ** 2)) if acceleration > 5 else 0.0
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


def main(path,
         do_plotting=False,
         do_convergence=False,
         do_evaluation_runs=False,
         do_videos=False
         ):
    eval_path = Path(path, "evaluation")
    eval_path.mkdir(parents=True, exist_ok=True)

    print("###########################")
    print(sys.path)

    if do_plotting:
        all_paths = list_all_run_paths(path)
        plotting_paths, paradigms = get_plotting_paths(path)
        overview_path = Path(path, "evaluation", "plots", "overview")
        overview_path.mkdir(parents=True, exist_ok=True)
        for key in plotting_paths.keys():
            print("plotting: {}".format(key))
            legend = [paradigms[key] + ' ' + x[-19:] for x in plotting_paths[key]]
            scenario_name = '_'.join(key.split('/')[-3:])
            title = ' '.join(key.split('/')[-3:])
            log_path = Path(eval_path, "plots", key.split('/')[-2], key.split('/')[-1])
            plotting.main(paths=plotting_paths[key],
                          scenario_name=scenario_name,
                          title=title,
                          mean_reward=True,
                          mean_len=True,
                          learner_stats=True,
                          legend=legend,
                          agent_wise=True,
                          png=True,
                          pdf=False,
                          boxplot=True,
                          log_path=log_path,
                          )

            # copy and rename reward, checkpoint plot to plots/overview
            plot_name = Path(log_path, os.listdir(log_path)[0], "episode_reward_mean_checkpoint.png")
            new_plot_name = Path(overview_path, "-".join(key.split('/')[-2:]) + ".png")
            shutil.copy(plot_name, new_plot_name)

    if do_convergence:
        conv_paths = get_convergence_paths(path)
        for i, c_path in enumerate(conv_paths):
            print("training analysis {}/{}: {}".format(i + 1, len(conv_paths), c_path))
            log_path = Path(eval_path, "convergence", c_path.split('/')[-4], c_path.split('/')[-3])

            training_analysis.main(c_path, log_path)

    if do_evaluation_runs:
        eval_runs_path = Path(path, "evaluation", "evaluation_runs")
        eval_runs_path.mkdir(parents=True, exist_ok=True)

        df_info = pd.read_csv(Path(path, "info"), sep=", ", engine="python")

        convergence_path = Path(path, "evaluation", "convergence")
        alpha_degree_dirs = os.listdir(Path(convergence_path))
        for ad in alpha_degree_dirs:
            alpha_degree_path = Path(convergence_path, ad)
            paradigm_dirs = os.listdir(alpha_degree_path)
            for paradigm in paradigm_dirs:
                config_path = get_config_yaml_path(df_info["config_path"][0], ad, paradigm)
                paradigm_path = Path(alpha_degree_path, paradigm)
                if not os.path.isfile(Path(paradigm_path, "convergence_logs_manual.csv")):
                    conv = "converged"
                    df_convergence_logs = pd.read_csv((Path(paradigm_path, "convergence_logs.csv")))
                else:
                    conv = "human_converged"
                    df_convergence_logs = pd.read_csv((Path(paradigm_path, "convergence_logs_manual.csv")))

                print(paradigm_path)
                for index, row in df_convergence_logs.iterrows():
                    if row[conv]:
                        print(paradigm_path)
                        print(PARADIGM_MAP[paradigm])
                        print(row["paradigm"])
                        # assert PARADIGM_MAP[paradigm] == row["paradigm"], "something went wrong with the paradigm log"

                        max_reward_checkpoint = row["max_reward_checkpoint"]
                        run_path = './' + row["run_path"]
                        checkpoint_path = run_path + \
                                          "/checkpoint_{:06d}".format(max_reward_checkpoint) + \
                                          "/checkpoint-{}".format(max_reward_checkpoint)
                        log_dir = Path(eval_runs_path, ad, paradigm, row["name"],
                                       "checkpoint_{:06d}".format(max_reward_checkpoint))
                        print(checkpoint_path)
                        # evaluate.main(df_info["scenario"][0],
                        #               [config_path],
                        #               log_dir,
                        #               num_steps=df_info["num_steps"][0],
                        #               num_episodes=200,
                        #               paradigm=PARADIGM_MAP[paradigm],
                        #               headless=True,
                        #               show_plots=False,
                        #               checkpoint=checkpoint_path,
                        #               data_replay_path=None,
                        #               )


                        # add detailed rewards to episode files
                        config_path_no_marl_benchmark = Path("/".join(str(config_path).split('/')[1:]))
                        with open(config_path_no_marl_benchmark, 'r') as stream:
                            config_yaml = yaml.safe_load(stream)
                            reward_config = config_yaml["agent"]["state"]["wrapper"]

                        get_detailed_rewards = get_detailed_reward_adapter(**reward_config)

                        times = os.listdir(log_dir)
                        for time in times:
                            if time != "plots":
                                episodes = os.listdir(Path(log_dir, time))
                                for episode in episodes:
                                    add_rewards_to_csv(Path(log_dir, time, episode), get_detailed_rewards)



    if do_videos:
        df_info = pd.read_csv(Path(path, "info"), sep=", ", engine="python")
        scenario_name = df_info["scenario"][0].split('/')[-1]
        evaluation_runs_path = Path(path, "evaluation", "evaluation_runs")
        evaluation_runs_paths = list_all_run_paths(str(evaluation_runs_path))
        evaluation.main(evaluation_runs_paths,
                        scenario_name,
                        checkpoints=None,  # leave None for all available checkpoints
                        training_progress_video=False,
                        checkpoint_video=True,
                        coloring=["control_input", "speed", "reward", "acceleration"],
                        poa=False,
                        decentralized_cp=None,
                        centralized_cp=None,
                        density_plots=False,
                        )

        # FrameStack.get_reward_adapter()


def parse_args():
    parser = argparse.ArgumentParser("")

    parser.add_argument("-p",
                        "--path",
                        type=str,
                        required=True)

    parser.add_argument(
        "--do_plotting", default=False, action="store_true"
    )

    parser.add_argument(
        "--do_convergence", default=False, action="store_true"
    )

    parser.add_argument(
        "--do_evaluation_runs", default=False, action="store_true"
    )

    parser.add_argument(
        "--do_videos", default=False, action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(path=args.path,
         do_plotting=args.do_plotting,
         do_convergence=args.do_convergence,
         do_evaluation_runs=args.do_evaluation_runs,
         do_videos=args.do_videos,
         )
