#!/usr/bin/env python -W ignore::DeprecationWarning
# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import argparse
import os
from pathlib import Path

import ray
from ray import tune

from baselines.marl_benchmark import gen_config
from baselines.marl_benchmark.common import SimpleCallbacks

import yaml  # NK
# working with centralized

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
RUN_NAME = Path(__file__).stem
EXPERIMENT_NAME = "{scenario}-{n_agent}"


def main(
    scenario,
    config_file,
    log_dir,
    restore_path=None,
    num_workers=1,
    horizon=1000,
    paradigm="decentralized",
    headless=False,
    cluster=False,
    stop_time=14400,
    scenario1="",
    scenario2="",
    scenario3="",
    scenario4="",
    data_replay_path=None,
):
    if cluster:
        ray.init(address="auto", redis_password="5241590000000000")
        print(
            "--------------- Ray startup ------------\n{}".format(
                ray.state.cluster_resources()
            )
        )

    # use for debugging
    # print("WARNING: local mode on")
    # ray.init(local_mode=True)

    scenarios = [scenario]
    if scenario1 != "":
        scenarios.append(scenario1)
    if scenario2 != "":
        scenarios.append(scenario2)
    if scenario3 != "":
        scenarios.append(scenario3)
    if scenario4 != "":
        scenarios.append(scenario4)

    config = gen_config(
        scenarios=scenarios, config_file=config_file, paradigm=paradigm, headless=headless
    )

    tune_config = config["run"]["config"]
    tune_config.update(
        {
            "env_config": config["env_config"],
            "callbacks": SimpleCallbacks,
            "num_workers": num_workers,
            "horizon": horizon,
            # "envision_record_data_replay_path": data_replay_path,
        }
    )

    # TODO(ming): change scenario name (not path)
    experiment_name = EXPERIMENT_NAME.format(
        scenario=scenario.split("/")[-1],
        n_agent=4,
    )

    log_dir = Path(log_dir).expanduser().absolute() / RUN_NAME
    log_dir.mkdir(parents=True, exist_ok=True)

    if restore_path is not None:
        restore_path = Path(restore_path).expanduser()
        print(f"Loading model from {restore_path}")

    # run experiments
    config["run"].update(
        {
            "run_or_experiment": config["trainer"],
            "name": experiment_name,
            "local_dir": str(log_dir),
            "restore": restore_path,
        }
    )

    # if paradigm == 'centralized':
    #     # config['policy'] = config['policy'][:3]
    #     del config['run']['config']['custom_preprocessor']

    # with open('config.yaml', 'w') as outfile:
    #     yaml.dump(config, outfile, default_flow_style=False)
    #
    # config["run"]["config"]["env"].update(
    #     {
    #         "observation_space": config["run"]["config"]["env_config"]["custom_config"]["observation_space"],
    #         "action_space": config["run"]["config"]["env_config"]["custom_config"]["action_space"],
    #     }
    # )

    config["run"]["stop"]["time_total_s"] = stop_time
    config["run"]["config"]["env_config"]["envision_record_data_replay_path"] = data_replay_path

    analysis = tune.run(**config["run"])

    print(analysis.dataframe().head())


def parse_args():
    parser = argparse.ArgumentParser("Benchmark learning")
    parser.add_argument(
        "scenario",
        type=str,
        help="Scenario name",
    )
    parser.add_argument(
        "--scenario1",
        type=str,
        default="",
        help="Additional scenario name. Important: Number of agents for all scenarios has to be the same!",
    )
    parser.add_argument(
        "--scenario2",
        type=str,
        default="",
        help="Additional scenario name. Important: Number of agents for all scenarios has to be the same!",
    )
    parser.add_argument(
        "--scenario3",
        type=str,
        default="",
        help="Additional scenario name. Important: Number of agents for all scenarios has to be the same!",
    )
    parser.add_argument(
        "--scenario4",
        type=str,
        default="",
        help="Additional scenario name. Important: Number of agents for all scenarios has to be the same!",
    )

    parser.add_argument(
        "--paradigm",
        type=str,
        default="decentralized",
        help="Algorithm paradigm, decentralized (default) or centralized",
    )
    parser.add_argument(
        "--headless", default=False, action="store_true", help="Turn on headless mode"
    )
    parser.add_argument(
        "--log_dir",
        default="./log/results",
        type=str,
        help="Path to store RLlib log and checkpoints, default is ./log/results",
    )
    parser.add_argument("--config_file", "-f", type=str, required=True)
    parser.add_argument("--restore_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=1, help="RLlib num workers")
    parser.add_argument("--cluster", action="store_true")
    parser.add_argument("--horizon", type=int, default=1000, help="Horizon for a episode")
    parser.add_argument("--stop_time", type=int, default=14400, help="Max. number of seconds of training.")
    parser.add_argument("--data_replay_path", type=str, default=None, help="Path to store envision replay data.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        scenario=args.scenario,
        config_file=args.config_file,
        log_dir=args.log_dir,
        restore_path=args.restore_path,
        num_workers=args.num_workers,
        horizon=args.horizon,
        paradigm=args.paradigm,
        headless=args.headless,
        cluster=args.cluster,
        stop_time=args.stop_time,
        scenario1=args.scenario1,
        scenario2=args.scenario2,
        scenario3=args.scenario3,
        scenario4=args.scenario4,
        data_replay_path=args.data_replay_path,
    )
