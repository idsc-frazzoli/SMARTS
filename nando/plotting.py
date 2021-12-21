
import pandas as pd
import os

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# progress_path = Path('../../../nando/ray_results/rllib_example_multi/PG_RLlibHiWayEnv_78d5d_00000_0_seed=125_2021-12-02_09-54-34/progress.csv')
# progress_path = Path('../../../nando/ray_results/rllib_example_multi/PG_RLlibHiWayEnv_15b92_00000_0_seed=140_2021-12-02_11-32-01/progress.csv')

scenario = 'nocross-4'
# name = 'PPO_FrameStack_9c0e9_00000_0_2021-12-09_00-21-45'
# name = 'PPO_FrameStack_42dda_00000_0_2021-12-14_11-23-28'
name = 'PPO_FrameStack_64782_00000_0_2021-12-15_10-47-26'

progress_path = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                             scenario,
                             name,
                             'progress.csv')


df_progress = pd.read_csv(progress_path)


df = df_progress[['episode_reward_mean','episode_reward_max','episode_reward_min']]
mean_reward = df_progress[['episode_reward_mean']]
ram = df_progress[['perf/ram_util_percent']]
cpu = df_progress[['perf/cpu_util_percent']]

#df.plot()
#mean_reward.plot()
#ram.plot()
cpu.plot()
print('done')
plt.savefig("plots/nocross_cpu.png")
# plt.show()



