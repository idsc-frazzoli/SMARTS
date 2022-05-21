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

from pathlib import Path

from baselines.marl_benchmark.main_utils import get_plotting_paths, get_convergence_paths, get_config_yaml_path, \
    get_detailed_reward_adapter, list_all_run_paths, add_rewards_to_csv, add_evaluation_paths, make_stats, \
    make_data_pickle


from timeit import default_timer as timer

PARADIGM_MAP = {"cent": "centralized", "decent": "decentralized"}


# start = timer()
# end = timer()
# print('Time to get paths: {}'.format(end - start))

# def get_alpha_degree(alpha_degree: str) -> Tuple[float, float]:
#     pass


def main(path,
         do_plotting=False,
         do_convergence=False,
         concat_convergence=False,
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

            # # copy and rename reward, checkpoint plot to plots/overview
            # plot_name = Path(log_path, os.listdir(log_path)[0], "episode_reward_mean_timesteps_total.png")
            # new_plot_name = Path(overview_path, "-".join(key.split('/')[-2:]) + ".png")
            # shutil.copy(plot_name, new_plot_name)

    if do_convergence:
        conv_paths = get_convergence_paths(path)
        for i, c_path in enumerate(conv_paths):
            print("training analysis {}/{}: {}".format(i + 1, len(conv_paths), c_path))
            log_path = Path(eval_path, "convergence", c_path.split('/')[-4], c_path.split('/')[-3])

            training_analysis.main(c_path, log_path)

    # should do manual convergence first
    if concat_convergence:
        manual_convergence_paths = [x[0] + "/convergence_logs_manual.csv" for x in
                                    os.walk(Path(path, "evaluation", "convergence"))
                                    if "convergence_logs_manual.csv" in x[2]]

        dfs = []
        for mcp in manual_convergence_paths:
            dfs.append(pd.read_csv(mcp))

        df_concat = pd.concat(dfs, ignore_index=True)
        df_concat.to_csv(Path(path, "evaluation", "convergence", "all_convergence.csv"))

    if do_evaluation_runs:
        eval_runs_path = Path(path, "evaluation", "evaluation_runs")
        eval_runs_path.mkdir(parents=True, exist_ok=True)

        df_info = pd.read_csv(Path(path, "info"), sep=", ", engine="python")

        convergence_path = Path(path, "evaluation", "convergence")
        alpha_degree_dirs = [x for x in os.listdir(Path(convergence_path))
                             if not os.path.isfile(Path(convergence_path, x))]
        print(alpha_degree_dirs)
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
                    if row[conv] and not row["converged"] in ["CSV_PARSE_ERROR", "CSV_EMPTY_DATA_ERROR"]:
                        print(row["run_path"])
                        # print(paradigm_path)
                        # print(PARADIGM_MAP[paradigm])
                        # print(row["paradigm"])
                        # assert PARADIGM_MAP[paradigm] == row["paradigm"], "something went wrong with the paradigm log"

                        max_reward_checkpoint = int(row["max_reward_checkpoint"])
                        run_path = './' + row["run_path"]
                        checkpoint_path = run_path + \
                                          "/checkpoint_{:06d}".format(max_reward_checkpoint) + \
                                          "/checkpoint-{}".format(max_reward_checkpoint)
                        log_dir = Path(eval_runs_path, ad, paradigm, row["name"],
                                       "checkpoint_{:06d}".format(max_reward_checkpoint))
                        print(log_dir)
                        if not os.path.isdir(log_dir):
                            evaluate.main(df_info["scenario"][0],
                                          [config_path],
                                          log_dir,
                                          num_steps=df_info["num_steps"][0],
                                          num_episodes=200,
                                          paradigm=PARADIGM_MAP[paradigm],
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
        add_evaluation_paths(eval_path)
        make_stats(eval_path)
        make_data_pickle(eval_path)

    if do_videos:
        df_info = pd.read_csv(Path(path, "info"), sep=", ", engine="python")
        scenario_name = df_info["scenario"][0].split('/')[-1]
        evaluation_runs_path = Path(path, "evaluation", "evaluation_runs")
        evaluation_runs_paths = list_all_run_paths(str(evaluation_runs_path))
        coloring = ["control_input", "speed", "reward", "acceleration", "agents"]  # ["control_input", "speed", "reward", "acceleration", "agents"]
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
        "--do_convergence", default=False, action="store_true"
    )

    parser.add_argument(
        "--concat_convergence", default=False, action="store_true"
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
         do_convergence=args.do_convergence,
         concat_convergence=args.concat_convergence,
         do_evaluation_runs=args.do_evaluation_runs,
         add_eval_paths=args.add_eval_paths,
         do_videos=args.do_videos,
         )
