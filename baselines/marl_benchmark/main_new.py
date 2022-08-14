import pandas as pd
import numpy as np

import sys
# sys.path.append("~/SMARTS/baselines/marl_benchmark")
# sys.path.append("~/SMARTS/baselines")
# sys.path.append("/home/nando/SMARTS")
import yaml

from baselines.marl_benchmark.plotting import plotting
from baselines.marl_benchmark.evaluation import generate_info_logs
from baselines.marl_benchmark import evaluate
from baselines.marl_benchmark.evaluation import evaluation

import argparse
import os
import shutil

from pathlib import Path

from baselines.marl_benchmark.main_utils import get_plotting_paths, get_convergence_paths, get_config_yaml_path, \
    get_detailed_reward_adapter, list_all_run_paths, add_rewards_to_csv, add_evaluation_paths, make_stats, \
    make_data_pickle, add_evaluation_paths_new, make_stats_new


from timeit import default_timer as timer

PARADIGM_MAP = {"cent": "centralized", "decent": "decentralized"}


# start = timer()
# end = timer()
# print('Time to get paths: {}'.format(end - start))

# def get_alpha_degree(alpha_degree: str) -> Tuple[float, float]:
#     pass


def main(path,
         do_plotting=False,
         do_evaluation_runs=False,
         add_eval_paths=False,
         do_videos=False
         ):
    eval_path = Path(path, "evaluation")
    eval_path.mkdir(parents=True, exist_ok=True)

    # print("###########################")
    # print(sys.path)

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
            plot_name = Path(log_path, os.listdir(log_path)[0], "episode_reward_mean_timesteps_total.png")
            new_plot_name = Path(overview_path, "-".join(key.split('/')[-2:]) + ".png")
            shutil.copy(plot_name, new_plot_name)


    if do_evaluation_runs:

        generate_info_logs.main(path)
        df_info_logs = pd.read_csv(Path(eval_path, "info_logs.csv"))

        eval_runs_path = Path(path, "evaluation", "evaluation_runs")
        df_info = pd.read_csv(Path(path, "info"), sep=", ", engine="python")

        for index, row in df_info_logs.iterrows():
            if not row["paradigm"] in ["CSV_PARSE_ERROR", "CSV_EMPTY_DATA_ERROR"]:
                print(row["run_path"])
                # print(paradigm_path)
                # print(PARADIGM_MAP[paradigm])
                # print(row["paradigm"])
                # assert PARADIGM_MAP[paradigm] == row["paradigm"], "something went wrong with the paradigm log"

                if row["paradigm"] == "centralized":
                    config_path = Path(df_info["config_path"][0], "baseline-lane-control_cent.yaml")
                else:
                    config_path = Path(df_info["config_path"][0], "baseline-lane-control_decent.yaml")

                max_reward_checkpoint = int(row["max_reward_checkpoint"])
                run_path = './' + row["run_path"]
                checkpoint_path = run_path + \
                                  "/checkpoint_{:06d}".format(max_reward_checkpoint) + \
                                  "/checkpoint-{}".format(max_reward_checkpoint)
                log_dir = Path(eval_runs_path, row["paradigm"], row["name"],
                               "checkpoint_{:06d}".format(max_reward_checkpoint))
                print(log_dir)
                if not os.path.isdir(log_dir):
                    evaluate.main(df_info["scenario"][0],
                                  [config_path],
                                  log_dir,
                                  num_steps=df_info["num_steps"][0],
                                  num_episodes=200,
                                  paradigm=row["paradigm"],
                                  headless=True,
                                  show_plots=False,
                                  checkpoint=checkpoint_path,
                                  data_replay_path=None,
                                  )

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


    if add_eval_paths:
        add_evaluation_paths_new(eval_path)
        make_stats_new(eval_path)
        # make_data_pickle_new(eval_path)


    if do_videos:
        df_info = pd.read_csv(Path(path, "info"), sep=", ", engine="python")
        scenario_name = df_info["scenario"][0].split('/')[-1]
        evaluation_runs_path = Path(path, "evaluation", "evaluation_runs")
        evaluation_runs_paths = list_all_run_paths(str(evaluation_runs_path))
        coloring = ["control_input"]
        # ["control_input", "speed", "reward", "acceleration", "agents", "cost_com", "cost_per_acceleration", "goal_improvement_reward"]
        evaluation.main(evaluation_runs_paths,
                        scenario_name,
                        checkpoints=None,  # leave None for all available checkpoints
                        training_progress_video=False,
                        checkpoint_video=True,
                        coloring=coloring,
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
        "--do_evaluation_runs", default=False, action="store_true"
    )

    parser.add_argument(
        "--add_eval_paths", default=False, action="store_true"
    )

    parser.add_argument(
        "--do_videos", default=False, action="store_true"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(path=args.path,
         do_plotting=args.do_plotting,
         do_evaluation_runs=args.do_evaluation_runs,
         add_eval_paths=args.add_eval_paths,
         do_videos=args.do_videos,
         )
