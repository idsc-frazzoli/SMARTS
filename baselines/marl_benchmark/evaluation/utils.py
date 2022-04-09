import re
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


def get_min_max(checkpoint_path):
    max_episode_length = 0.0
    max_speed = 0.0
    min_speed = 16.0
    max_step_reward = 0.0
    min_step_reward = 0.0
    min_x_pos = 0.0
    max_x_pos = 0.0
    min_y_pos = 0.0
    max_y_pos = 0.0

    times = os.listdir(Path(checkpoint_path))
    for time in times:
        if time == "plots":
            continue
        episodes = os.listdir(Path(checkpoint_path, time))
        for episode in episodes:
            n_agents = len(os.listdir(Path(checkpoint_path, time, episode)))
            for agent in range(n_agents):
                csv_path = Path(checkpoint_path, time, episode, 'agent_AGENT-{}.csv'.format(str(agent)))
                df = pd.read_csv(csv_path, index_col=0, header=None).T
                max_episode_length = max(max_episode_length, df.shape[0])
                max_speed = max(max_speed, max(df["Speed"]))
                min_speed = min(min_speed, min(df["Speed"]))
                max_step_reward = max(max_step_reward, max(df["Step_Reward"]))
                min_step_reward = min(min_step_reward, min(df["Step_Reward"]))
                max_x_pos = max(max_x_pos, max(df["Xpos"]))
                min_x_pos = min(min_x_pos, min(df["Xpos"]))
                max_y_pos = max(max_y_pos, max(df["Ypos"]))
                min_y_pos = min(min_y_pos, min(df["Ypos"]))

    return (max_episode_length,
            min_speed, max_speed,
            min_step_reward, max_step_reward,
            min_x_pos, max_x_pos,
            min_y_pos, max_y_pos)
