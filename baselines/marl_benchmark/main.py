from baselines.marl_benchmark.plotting import plotting
from baselines.marl_benchmark.evaluation import training_analysis
from baselines.marl_benchmark import evaluate

import argparse
import os
import baselines.marl_benchmark.plotting.utils as plotting_utils
from typing import List
from pathlib import Path

import sys

from timeit import default_timer as timer


# start = timer()
# end = timer()
# print('Time to get paths: {}'.format(end - start))

def get_plotting_paths(path):

    paradigms = {}
    paths = {}
    for x in list(os.walk(Path(path), topdown=True)):
        if "evaluation" not in x[0] and x[0].split('/')[-1] in ('decent', 'cent'):
            paths[x[0]] = list_all_run_paths(x[0])
            paradigms[x[0]] = x[0].split('/')[-1]

    return paths, paradigms



def list_all_run_paths(path: str,
                       identification_prefix: str = "PPO_FrameStack") -> List[str]:

    all_paths = [x[0] for x in list(os.walk(Path(path), topdown=True))
                 if identification_prefix in x[0].split('/')[-1]]

    return all_paths


def main(path):

    eval_path = Path(path, "evaluation")
    eval_path.mkdir(parents=True, exist_ok=True)

    # all_paths = list_all_run_paths(path)
    plotting_paths, paradigms = get_plotting_paths(path)
    for key in plotting_paths.keys():
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




    # evaluate.main(scenario,
    #               config_files,
    #               log_dir,
    #             num_steps=1000,
    #             num_episodes=10,
    #             paradigm="decentralized",
    #             headless=False,
    #             show_plots=False,
    #             checkpoint=None,
    #             data_replay_path=None,
    #             )

    #     pass



    # legend = []
    # scenario_name = ""
    # paths = []
    # plotting.main(paths=paths,
    #               scenario_name=scenario_name,
    #               title=args.title,
    #               mean_reward=True,
    #               mean_len=True,
    #               learner_stats=True,
    #               legend=legend,
    #               agent_wise=True,
    #               png=True,
    #               pdf=True,
    #               boxplot=True
    #               )


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
