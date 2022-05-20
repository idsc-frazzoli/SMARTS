import os

from baselines.marl_benchmark import evaluate
from pathlib import Path


config_paths = Path("./agents/ppo/20220513_merge65_lanes42")
cp = "/agents/ppo/20220513_merge65_lanes42/"

log_dr = "./log/results/run/test/"

for config_file in os.listdir(config_paths):
    config_path = "./marl_benchmark" + cp + config_file
    if "_decent" in config_file:
        paradigm = "decentralized"
    elif "_cent" in config_file:
        paradigm = "centralized"

    folder = "_".join(config_file.split("_")[-3:-1])

    if paradigm == "centralized":
        evaluate.main("./scenarios/custom/merge/merge65_lanes42",
                      [config_path],
                      log_dr + folder,
                      num_steps=80,
                      num_episodes=200,
                      paradigm=paradigm,
                      headless=True,
                      show_plots=False,
                      checkpoint="./log/results/run/20220510_merge65_lanes42/alpha0p5_degree2/cent/run/merge65_lanes42-4/PPO_FrameStack_b5eb1_00000_0_2022-05-11_19-29-44/checkpoint_000265/checkpoint-265",
                      data_replay_path=None,
                      )

    if paradigm == "decentralized":
        evaluate.main("./scenarios/custom/merge/merge65_lanes42",
                      [config_path],
                      log_dr + folder,
                      num_steps=80,
                      num_episodes=200,
                      paradigm=paradigm,
                      headless=True,
                      show_plots=False,
                      checkpoint="./log/results/run/20220510_merge65_lanes42/alpha0p5_degree2/decent/run/merge65_lanes42-4/PPO_FrameStack_b3173_00000_0_2022-05-11_00-02-51/checkpoint_000250/checkpoint-250",
                      data_replay_path=None,
                      )