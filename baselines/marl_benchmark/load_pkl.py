
import pickle
checkpoint_path = "./log/log_old/results/run/cross-4/PPO_FrameStack_9c0e9_00000_0_2021-12-09_00-21-45/checkpoint_300/checkpoint-300"

extra_data = pickle.load(open(checkpoint_path, "rb"))

with open(checkpoint_path + ".tune_metadata", "rb") as f:
    metadata = pickle.load(f)