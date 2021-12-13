import argparse
import os
from pathlib import Path

import ray
from ray import tune

from baselines.marl_benchmark import gen_config
from baselines.marl_benchmark.common import SimpleCallbacks

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
):
    if cluster:
        ray.init(address="auto", redis_password="5241590000000000")
        print(
            "--------------- Ray startup ------------\n{}".format(
                ray.state.cluster_resources()
            )
        )
    # does not fix memory issue, just makes it slower
    # else: #NK
    #     ray.init(local_mode=True) #NK
        
    config = gen_config(
        scenario=scenario, config_file=config_file, paradigm=paradigm, headless=headless
    )

    tune_config = config["run"]["config"]
    tune_config.update(
        {
            "env_config": config["env_config"],
            "callbacks": SimpleCallbacks,
            "num_workers": num_workers,
            "horizon": horizon,
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
    analysis = tune.run(**config["run"])

    print(analysis.dataframe().head())


# def parse_args():
#     parser = argparse.ArgumentParser("Benchmark learning")
#     parser.add_argument(
#         "scenario",
#         type=str,
#         help="Scenario name",
#     )
#     parser.add_argument(
#         "--paradigm",
#         type=str,
#         default="decentralized",
#         help="Algorithm paradigm, decentralized (default) or centralized",
#     )
#     parser.add_argument(
#         "--headless", default=False, action="store_true", help="Turn on headless mode"
#     )
#     parser.add_argument(
#         "--log_dir",
#         default="./log/results",
#         type=str,
#         help="Path to store RLlib log and checkpoints, default is ./log/results",
#     )
#     parser.add_argument("--config_file", "-f", type=str, required=True)
#     parser.add_argument("--restore_path", type=str, default=None)
#     parser.add_argument("--num_workers", type=int, default=1, help="RLlib num workers")
#     parser.add_argument("--cluster", action="store_true")
#     parser.add_argument(
#         "--horizon", type=int, default=1000, help="Horizon for a episode"
#     )

#     return parser.parse_args()


if __name__ == "__main__":
    scenario = os.path('scenarios/double_merge/cross')
    config_file = os.path('marl_benchmark/agents/ppo/baseline-lane-control.yaml')
    log_dir = os.path('.log/results')
    num_workers = 7
    paradigm = 'decentralized'
    headless = True
    
    main(
        scenario=scenario,
        config_file=config_file,
        log_dir=log_dir,
        restore_path=None,
        num_workers=num_workers,
        horizon=1000,
        paradigm=paradigm,
        headless=headless,
        cluster=False,
    )
