
import pandas as pd
import os

from pathlib import Path
import numpy as np

# import matplotlib as mpl
import matplotlib.pyplot as plt

def str2list(s):
    slist = s[1:-1].split(',')
    return [float(x) for x in slist]


#%% old

# progress_path = Path('../../../nando/ray_results/rllib_example_multi/PG_RLlibHiWayEnv_78d5d_00000_0_seed=125_2021-12-02_09-54-34/progress.csv')
# progress_path = Path('../../../nando/ray_results/rllib_example_multi/PG_RLlibHiWayEnv_15b92_00000_0_seed=140_2021-12-02_11-32-01/progress.csv')

# scenario = 'nocross-4'

# name = 'PPO_FrameStack_9c0e9_00000_0_2021-12-09_00-21-45'
# name = 'PPO_FrameStack_42dda_00000_0_2021-12-14_11-23-28'
# name = 'PPO_FrameStack_64782_00000_0_2021-12-15_10-47-26'

# progress_path = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
#                              scenario,
#                              name,
#                              'progress.csv')


#%% 13.01.2022: nocross decentralized vs. centralized tests
# rudolf with num_workers = 30

scenario = 'nocross-4'
name_cen = 'PPO_FrameStack_9af5d_00000_0_2022-01-12_09-54-17'
name_decen = 'PPO_FrameStack_2630c_00000_0_2022-01-12_15-27-27'

# continued centralized training from last checkpoint on 14.01.2022
name_cen2 = 'PPO_FrameStack_e4676_00000_0_2022-01-14_14-11-40'

progress_path_cen = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                 scenario,
                                 name_cen,
                                 'progress.csv')

progress_path_decen = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                   scenario,
                                   name_decen,
                                   'progress.csv')

progress_path_cen2 = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                  scenario,
                                  name_cen2,
                                  'progress.csv')

df_progress_cen = pd.concat([pd.read_csv(progress_path_cen), pd.read_csv(progress_path_cen2)])
df_progress_decen = pd.read_csv(progress_path_decen)

df_ep_rew_cen = df_progress_cen['hist_stats/episode_reward']
df_ep_rew_decen = df_progress_decen['hist_stats/episode_reward']

rewards_cen = df_progress_cen[['episode_reward_mean','episode_reward_max','episode_reward_min']]
rewards_decen = df_progress_decen[['episode_reward_mean','episode_reward_max','episode_reward_min']]

rew_std_up_cen = [np.percentile(str2list(x),84) for x in df_ep_rew_cen]
rew_std_lo_cen = [np.percentile(str2list(x),16) for x in df_ep_rew_cen]
rew_std_up_decen = [np.percentile(str2list(x),84) for x in df_ep_rew_decen]
rew_std_lo_decen = [np.percentile(str2list(x),16) for x in df_ep_rew_decen]

rew_2std_up_cen = [np.percentile(str2list(x),97.5) for x in df_ep_rew_cen]
rew_2std_lo_cen = [np.percentile(str2list(x),2.5) for x in df_ep_rew_cen]
rew_2std_up_decen = [np.percentile(str2list(x),97.5) for x in df_ep_rew_decen]
rew_2std_lo_decen = [np.percentile(str2list(x),2.5) for x in df_ep_rew_decen]

rew_median_cen = [np.median(str2list(x)) for x in df_ep_rew_cen]
rew_median_decen = [np.median(str2list(x)) for x in df_ep_rew_decen]

# rew_75_decen = [np.percentile(str2list(x),75) for x in df_ep_rew_decen]
# rew_25_decen = [np.percentile(str2list(x),25) for x in df_ep_rew_decen]

# Plotting

fig = plt.figure()
ax = plt.axes()

# ax.plot(rewards_cen['episode_reward_mean'], color=[0,0.3,1,1], label='centralized')
# ax.plot(rewards_decen['episode_reward_mean'], color=[1,0.3,0,1], label='decentralized')

ax.fill_between(np.arange(1,len(rew_std_up_cen)+1), rew_std_up_cen, rew_std_lo_cen, color=[0,0.3,1,0.2])
ax.fill_between(np.arange(1,len(rew_std_up_decen)+1), rew_std_up_decen, rew_std_lo_decen, color=[1,0.3,0,0.2])

# ax.fill_between(np.arange(1,len(rew_2std_up_cen)+1), rew_2std_up_cen, rew_2std_lo_cen, color=[0,0.3,1,0.2])
# ax.fill_between(np.arange(1,len(rew_2std_up_decen)+1), rew_2std_up_decen, rew_2std_lo_decen, color=[1,0.3,0,0.2])
ax.plot(rew_median_cen, color=[0,0.3,1,1], label='centralized')
ax.plot(rew_median_decen, color=[1,0.3,0,1], label='decentralized')

plt.xlabel('iterations')
plt.ylabel('median reward')
plt.title('Decentralized vs. Centralized for nocross')

plt.legend()

plt.savefig("plots/nocross_cen_decen_cont.pdf")
plt.show()


plt.plot(np.arange(1,len(rewards_decen["episode_reward_mean"])+1), 
         rewards_cen["episode_reward_mean"][:len(rewards_decen["episode_reward_mean"])],
         color=[0,0.3,1,1], label='centralized')
plt.plot(np.arange(1,len(rewards_decen["episode_reward_mean"])+1),
         rewards_decen["episode_reward_mean"],
         color=[1,0.3,0,1], label='decentralized')

plt.xlabel('iterations')
plt.ylabel('mean reward')
plt.title('Decentralized vs. Centralized for nocross')
plt.legend()
plt.savefig("plots/nocross_cen_decen_short.pdf")
plt.savefig("plots/nocross_cen_decen_short.png")
plt.show()

plt.plot(np.arange(1,len(rewards_cen["episode_reward_mean"])+1), 
         rewards_cen["episode_reward_mean"],
         color=[0,0.3,1,1], label='centralized')
plt.plot(np.arange(1,len(rewards_decen["episode_reward_mean"])+1),
         rewards_decen["episode_reward_mean"],
         color=[1,0.3,0,1], label='decentralized')

plt.xlabel('iterations')
plt.ylabel('mean reward')
plt.title('Decentralized vs. Centralized for nocross')
plt.legend()
# plt.savefig("plots/nocross_cen_decen_long.pdf")
# plt.savefig("plots/nocross_cen_decen_long.png")
plt.show()

# df = df_progress[['episode_reward_mean','episode_reward_max','episode_reward_min']]
# mean_reward = df_progress[['episode_reward_mean']]
# ram = df_progress[['perf/ram_util_percent']]
# cpu = df_progress[['perf/cpu_util_percent']]

#df.plot()
#mean_reward.plot()
#ram.plot()
# cpu.plot()
# print('done')
# plt.savefig("plots/nocross_cpu.png")
# plt.show()


#%% 14.01.2022 cross decentralized vs. centralized tests
# rudolf with num_workers = 30


scenario = 'nocross-4'
name_decen = 'PPO_FrameStack_2630c_00000_0_2022-01-12_15-27-27'

decen_sample = pd.read_csv(os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                          scenario,
                                          name_decen,
                                         'progress.csv'))

decen_cols = decen_sample.columns

scenario = 'cross-4'
name_cen = 'PPO_FrameStack_d4e0a_00000_0_2022-01-13_09-54-43'
name_decen = 'PPO_FrameStack_a29a5_00000_0_2022-01-13_14-03-51'

progress_path_cen = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                             scenario,
                             name_cen,
                             'progress.csv')

progress_path_decen = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                             scenario,
                             name_decen,
                             'progress.csv')

df_progress_cen = pd.read_csv(progress_path_cen)
df_progress_decen = pd.read_csv(progress_path_decen)

# for some reason it didn't save the column names
df_progress_decen.columns = decen_cols

df_ep_rew_cen = df_progress_cen['hist_stats/episode_reward']
df_ep_rew_decen = df_progress_decen['hist_stats/episode_reward']

rewards_cen = df_progress_cen[['episode_reward_mean','episode_reward_max','episode_reward_min']]
rewards_decen = df_progress_decen[['episode_reward_mean','episode_reward_max','episode_reward_min']]

rew_std_up_cen = [np.percentile(str2list(x),84) for x in df_ep_rew_cen]
rew_std_lo_cen = [np.percentile(str2list(x),16) for x in df_ep_rew_cen]
rew_std_up_decen = [np.percentile(str2list(x),84) for x in df_ep_rew_decen]
rew_std_lo_decen = [np.percentile(str2list(x),16) for x in df_ep_rew_decen]

rew_2std_up_cen = [np.percentile(str2list(x),97.5) for x in df_ep_rew_cen]
rew_2std_lo_cen = [np.percentile(str2list(x),2.5) for x in df_ep_rew_cen]
rew_2std_up_decen = [np.percentile(str2list(x),97.5) for x in df_ep_rew_decen]
rew_2std_lo_decen = [np.percentile(str2list(x),2.5) for x in df_ep_rew_decen]

rew_median_cen = [np.median(str2list(x)) for x in df_ep_rew_cen]
rew_median_decen = [np.median(str2list(x)) for x in df_ep_rew_decen]

# rew_75_decen = [np.percentile(str2list(x),75) for x in df_ep_rew_decen]
# rew_25_decen = [np.percentile(str2list(x),25) for x in df_ep_rew_decen]

# Plotting

fig = plt.figure()
ax = plt.axes()

# ax.plot(rewards_cen['episode_reward_mean'], color=[0,0.3,1,1], label='centralized')
# ax.plot(rewards_decen['episode_reward_mean'], color=[1,0.3,0,1], label='decentralized')

ax.fill_between(np.arange(1,len(rew_std_up_cen)+1), rew_std_up_cen, rew_std_lo_cen, color=[0,0.3,1,0.2])
ax.fill_between(np.arange(1,len(rew_std_up_decen)+1), rew_std_up_decen, rew_std_lo_decen, color=[1,0.3,0,0.2])

# ax.fill_between(np.arange(1,len(rew_2std_up_cen)+1), rew_2std_up_cen, rew_2std_lo_cen, color=[0,0.3,1,0.2])
# ax.fill_between(np.arange(1,len(rew_2std_up_decen)+1), rew_2std_up_decen, rew_2std_lo_decen, color=[1,0.3,0,0.2])
ax.plot(rew_median_cen, color=[0,0.3,1,1], label='centralized')
ax.plot(rew_median_decen, color=[1,0.3,0,1], label='decentralized')

plt.xlabel('iterations')
plt.ylabel('median reward')
plt.title('Decentralized vs. Centralized for cross')

plt.legend()

# plt.savefig("plots/cross_cen_decen.pdf")
plt.show()





# df = df_progress[['episode_reward_mean','episode_reward_max','episode_reward_min']]
# mean_reward = df_progress[['episode_reward_mean']]
# ram = df_progress[['perf/ram_util_percent']]
# cpu = df_progress[['perf/cpu_util_percent']]

#df.plot()
#mean_reward.plot()
#ram.plot()
# cpu.plot()
# print('done')
# plt.savefig("plots/nocross_cpu.png")
# plt.show()


#%% 18.01.2022: nocross decentralized vs. centralized tests with only 2 actors
# rudolf with num_workers = 30

scenario = 'nocross_2-4'
name_cen = 'PPO_FrameStack_9e0a8_00000_0_2022-01-18_13-43-28'
name_decen = 'PPO_FrameStack_dda2a_00000_0_2022-01-18_08-23-08'


progress_path_cen = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                 scenario,
                                 name_cen,
                                 'progress.csv')

progress_path_decen = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                   scenario,
                                   name_decen,
                                   'progress.csv')

df_progress_cen = pd.read_csv(progress_path_cen)
df_progress_decen = pd.read_csv(progress_path_decen)

df_ep_rew_cen = df_progress_cen['hist_stats/episode_reward']
df_ep_rew_decen = df_progress_decen['hist_stats/episode_reward']

rewards_cen = df_progress_cen[['episode_reward_mean','episode_reward_max','episode_reward_min']]
rewards_decen = df_progress_decen[['episode_reward_mean','episode_reward_max','episode_reward_min']]

rew_std_up_cen = [np.percentile(str2list(x),84) for x in df_ep_rew_cen]
rew_std_lo_cen = [np.percentile(str2list(x),16) for x in df_ep_rew_cen]
rew_std_up_decen = [np.percentile(str2list(x),84) for x in df_ep_rew_decen]
rew_std_lo_decen = [np.percentile(str2list(x),16) for x in df_ep_rew_decen]

rew_2std_up_cen = [np.percentile(str2list(x),97.5) for x in df_ep_rew_cen]
rew_2std_lo_cen = [np.percentile(str2list(x),2.5) for x in df_ep_rew_cen]
rew_2std_up_decen = [np.percentile(str2list(x),97.5) for x in df_ep_rew_decen]
rew_2std_lo_decen = [np.percentile(str2list(x),2.5) for x in df_ep_rew_decen]

rew_median_cen = [np.median(str2list(x)) for x in df_ep_rew_cen]
rew_median_decen = [np.median(str2list(x)) for x in df_ep_rew_decen]

# rew_75_decen = [np.percentile(str2list(x),75) for x in df_ep_rew_decen]
# rew_25_decen = [np.percentile(str2list(x),25) for x in df_ep_rew_decen]

# Plotting

fig = plt.figure()
ax = plt.axes()

# ax.plot(rewards_cen['episode_reward_mean'], color=[0,0.3,1,1], label='centralized')
# ax.plot(rewards_decen['episode_reward_mean'], color=[1,0.3,0,1], label='decentralized')

ax.fill_between(np.arange(1,len(rew_std_up_cen)+1), rew_std_up_cen, rew_std_lo_cen, color=[0,0.3,1,0.2])
ax.fill_between(np.arange(1,len(rew_std_up_decen)+1), rew_std_up_decen, rew_std_lo_decen, color=[1,0.3,0,0.2])

# ax.fill_between(np.arange(1,len(rew_2std_up_cen)+1), rew_2std_up_cen, rew_2std_lo_cen, color=[0,0.3,1,0.2])
# ax.fill_between(np.arange(1,len(rew_2std_up_decen)+1), rew_2std_up_decen, rew_2std_lo_decen, color=[1,0.3,0,0.2])
ax.plot(rew_median_cen, color=[0,0.3,1,1], label='centralized')
ax.plot(rew_median_decen, color=[1,0.3,0,1], label='decentralized')

plt.xlabel('iterations')
plt.ylabel('median reward')
plt.title('Decentralized vs. Centralized for noross_2')

plt.legend()

# plt.savefig("plots/nocross_cen_decen.pdf")
plt.show()





# df = df_progress[['episode_reward_mean','episode_reward_max','episode_reward_min']]
# mean_reward = df_progress[['episode_reward_mean']]
# ram = df_progress[['perf/ram_util_percent']]
# cpu = df_progress[['perf/cpu_util_percent']]

#df.plot()
#mean_reward.plot()
#ram.plot()
# cpu.plot()
# print('done')
# plt.savefig("plots/nocross_cpu.png")
# plt.show()

#%% 31.01.2022: two_ways/bid tests with default and modified cost functions
# rudolf with num_workers = 30

# NOTE: the two rewards are not comparable, we have to look at the behavior!

title = "Modified Cost Test"

scenario = 'bid-4'
# 1. default cost (from benchmark)
name_1 = 'PPO_FrameStack_00695_00000_0_2022-01-31_13-16-29'
label_1 = "default cost"
# 2. modified cost
name_2 = 'PPO_FrameStack_2d1a5_00000_0_2022-01-31_17-42-36'
label_2 = "modified cost"

progress_path_1 = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                scenario,
                                name_1,
                                'progress.csv')

progress_path_2 = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                scenario,
                                name_2,
                                'progress.csv')


df_progress_1 = pd.read_csv(progress_path_1)
df_progress_2 = pd.read_csv(progress_path_2)

df_ep_rew_1 = df_progress_1['hist_stats/episode_reward']
df_ep_rew_2 = df_progress_2['hist_stats/episode_reward']

rewards_1 = df_progress_1[['episode_reward_mean','episode_reward_max','episode_reward_min']]
rewards_2 = df_progress_2[['episode_reward_mean','episode_reward_max','episode_reward_min']]

rew_std_up_1 = [np.percentile(str2list(x),84) for x in df_ep_rew_1]
rew_std_lo_1 = [np.percentile(str2list(x),16) for x in df_ep_rew_1]
rew_std_up_2 = [np.percentile(str2list(x),84) for x in df_ep_rew_2]
rew_std_lo_2 = [np.percentile(str2list(x),16) for x in df_ep_rew_2]

rew_2std_up_1 = [np.percentile(str2list(x),97.5) for x in df_ep_rew_1]
rew_2std_lo_1 = [np.percentile(str2list(x),2.5) for x in df_ep_rew_1]
rew_2std_up_2 = [np.percentile(str2list(x),97.5) for x in df_ep_rew_2]
rew_2std_lo_2 = [np.percentile(str2list(x),2.5) for x in df_ep_rew_2]

rew_median_1 = [np.median(str2list(x)) for x in df_ep_rew_1]
rew_median_2 = [np.median(str2list(x)) for x in df_ep_rew_2]

# Plotting

fig = plt.figure()
ax = plt.axes()

# ax.plot(rewards_cen['episode_reward_mean'], color=[0,0.3,1,1], label='centralized')
# ax.plot(rewards_decen['episode_reward_mean'], color=[1,0.3,0,1], label='decentralized')

# ax.fill_between(np.arange(1,len(rew_std_up_1)+1), rew_std_up_1, rew_std_lo_1, color=[0,0.3,1,0.2])
# ax.fill_between(np.arange(1,len(rew_std_up_2)+1), rew_std_up_2, rew_std_lo_2, color=[1,0.3,0,0.2])

# ax.fill_between(np.arange(1,len(rew_2std_up_cen)+1), rew_2std_up_cen, rew_2std_lo_cen, color=[0,0.3,1,0.2])
# ax.fill_between(np.arange(1,len(rew_2std_up_decen)+1), rew_2std_up_decen, rew_2std_lo_decen, color=[1,0.3,0,0.2])
# ax.plot(rew_median_1, color=[0,0.3,1,1], label=label_1)
ax.plot(rew_median_2, color=[1,0.3,0,1], label=label_2)

plt.xlabel('iterations')
plt.ylabel('median reward')
plt.title(title)

plt.legend()

# plt.savefig("plots/nocross_cen_decen.pdf")
plt.show()



#%% 01.02.2022: nocross with modified cost, decentralized
# rudolf with num_workers = 30


title = "Modified Cost Test"

scenario = 'nocross-4'
name_1 = 'PPO_FrameStack_96464_00000_0_2022-02-01_13-48-08'
label_1 = "modified cost"

progress_path_1 = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                scenario,
                                name_1,
                                'progress.csv')

df_progress_1 = pd.read_csv(progress_path_1)

df_ep_rew_1 = df_progress_1['hist_stats/episode_reward']

rewards_1 = df_progress_1[['episode_reward_mean','episode_reward_max','episode_reward_min']]

rew_std_up_1 = [np.percentile(str2list(x),84) for x in df_ep_rew_1]
rew_std_lo_1 = [np.percentile(str2list(x),16) for x in df_ep_rew_1]

rew_2std_up_1 = [np.percentile(str2list(x),97.5) for x in df_ep_rew_1]
rew_2std_lo_1 = [np.percentile(str2list(x),2.5) for x in df_ep_rew_1]

rew_median_1 = [np.median(str2list(x)) for x in df_ep_rew_1]

# Plotting

fig = plt.figure()
ax = plt.axes()

# ax.plot(rewards_cen['episode_reward_mean'], color=[0,0.3,1,1], label='centralized')

# ax.fill_between(np.arange(1,len(rew_std_up_1)+1), rew_std_up_1, rew_std_lo_1, color=[0,0.3,1,0.2])

# ax.fill_between(np.arange(1,len(rew_2std_up_cen)+1), rew_2std_up_cen, rew_2std_lo_cen, color=[0,0.3,1,0.2])
ax.plot(rew_median_1, color=[0,0.3,1,1], label=label_1)

plt.xlabel('iterations')
plt.ylabel('median reward')
plt.title(title)

plt.legend()

# plt.savefig("plots/nocross_cen_decen.pdf")
plt.show()



#%% 02.02.2022: cross_modified with sparse cost, decentralized
# rudolf with num_workers = 30


title = "Sparse Cost Test"

scenario = 'cross_modified-4'
name_1 = 'PPO_FrameStack_27e30_00000_0_2022-02-02_17-54-24'
label_1 = "sparse cost"

progress_path_1 = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                scenario,
                                name_1,
                                'progress.csv')

df_progress_1 = pd.read_csv(progress_path_1)

df_ep_rew_1 = df_progress_1['hist_stats/episode_reward']

rewards_1 = df_progress_1[['episode_reward_mean','episode_reward_max','episode_reward_min']]

rew_std_up_1 = [np.percentile(str2list(x),84) for x in df_ep_rew_1]
rew_std_lo_1 = [np.percentile(str2list(x),16) for x in df_ep_rew_1]

rew_2std_up_1 = [np.percentile(str2list(x),97.5) for x in df_ep_rew_1]
rew_2std_lo_1 = [np.percentile(str2list(x),2.5) for x in df_ep_rew_1]

rew_median_1 = [np.median(str2list(x)) for x in df_ep_rew_1]

# Plotting

fig = plt.figure()
ax = plt.axes()

ax.fill_between(np.arange(1,len(rew_std_up_1)+1), rew_std_up_1, rew_std_lo_1, color=[0,0.3,1,0.2])
ax.plot(rew_median_1, color=[0,0.3,1,1], label=label_1)

plt.xlabel('iterations')
plt.ylabel('median reward')
plt.title(title)


#%% 09.02.2022: merge with 1/d^2 cost, decentralized, the same experiment run 3 times
# rudolf with num_workers = 30

title = "Random Policy Initialization Test"

scenario = 'merge-4'
name_1 = 'PPO_FrameStack_ac834_00000_0_2022-02-09_13-17-47'
name_2 = 'PPO_FrameStack_3f9c8_00000_0_2022-02-09_14-19-10'
name_3 = 'PPO_FrameStack_36332_00000_0_2022-02-09_16-20-35'
label_1 = "run 1"
label_2 = "run 2"
label_3 = "run 3"

progress_path_1 = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                scenario,
                                name_1,
                                'progress.csv')

progress_path_2 = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                scenario,
                                name_2,
                                'progress.csv')

progress_path_3 = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                scenario,
                                name_3,
                                'progress.csv')

df_progress_1 = pd.read_csv(progress_path_1)
df_progress_2 = pd.read_csv(progress_path_2)
df_progress_3 = pd.read_csv(progress_path_3)

df_ep_rew_1 = df_progress_1['hist_stats/episode_reward']

# rewards_[run]_[agent]
rewards_1_0 = df_progress_1['policy_reward_mean/AGENT-0']
rewards_1_1 = df_progress_1['policy_reward_mean/AGENT-1']
rewards_2_0 = df_progress_2['policy_reward_mean/AGENT-0']
rewards_2_1 = df_progress_2['policy_reward_mean/AGENT-1']
rewards_3_0 = df_progress_3['policy_reward_mean/AGENT-0']
rewards_3_1 = df_progress_3['policy_reward_mean/AGENT-1']

# Plotting

fig = plt.figure()
ax = plt.axes()
ax.plot(rewards_1_0, color=[0,0,1], label='run 1, agent 0')
ax.plot(rewards_1_1, color=[1,0,0], label='run 1, agent 1')
ax.plot(rewards_2_0, color=[0,0.3,1], label='run 2, agent 0')
ax.plot(rewards_2_1, color=[1,0.3,0], label='run 2, agent 1')
ax.plot(rewards_3_0, color=[0,0.6,1], label='run 3, agent 0')
ax.plot(rewards_3_1, color=[1,0.6,0], label='run 3, agent 1')

plt.legend()
plt.xlabel('iterations')
plt.ylabel('median reward')
plt.title(title)

plt.savefig("plots/policy_initialization_1.pdf")


#%% 09.02.2022: training with 2 scenarios: merge and cross_modified 
# with 1/d^2 cost, decentralized
# rudolf with num_workers = 30



title = "Training with 2 Scenarios"

scenario = 'merge-4'
name_1 = 'PPO_FrameStack_c78eb_00000_0_2022-02-09_18-33-30'

progress_path_1 = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                scenario,
                                name_1,
                                'progress.csv')

df_progress_1 = pd.read_csv(progress_path_1)

# rewards_[run]_[agent]
episode_lengths = df_progress_1['hist_stats/episode_lengths']

all_episode_lengths = []

for l in episode_lengths:
    for x in str2list(l):
        all_episode_lengths.append(x)

fig = plt.figure()
ax = plt.axes()
plt.hist(all_episode_lengths, range=(0,150), bins=20)


plt.legend()
plt.ylabel('# occurances')
plt.xlabel('episode lenght')
plt.title(title)

plt.savefig("plots/two_scenarios_hist.pdf")


#%% 01.03.2022: training with 2 lanes, desired speed quadratic cost
# rudolf woth num_workers=20

title = "[title]"

scenario = 'straight100_lanes2-4'
name_1 = 'PPO_FrameStack_9b8d5_00000_0_2022-02-27_17-35-08'

progress_path_1 = os.path.join('/media/nando/Extreme SSD/MT/log/results/run',
                                scenario,
                                name_1,
                                'progress.csv')

df_progress_1 = pd.read_csv(progress_path_1)

df_ep_rew_1 = df_progress_1['hist_stats/episode_reward']

rewards_1 = df_progress_1[['episode_reward_mean','episode_reward_max','episode_reward_min']]

rew_std_up_1 = [np.percentile(str2list(x),84) for x in df_ep_rew_1]
rew_std_lo_1 = [np.percentile(str2list(x),16) for x in df_ep_rew_1]

rew_2std_up_1 = [np.percentile(str2list(x),97.5) for x in df_ep_rew_1]
rew_2std_lo_1 = [np.percentile(str2list(x),2.5) for x in df_ep_rew_1]

rew_median_1 = [np.median(str2list(x)) for x in df_ep_rew_1]

# Plotting

fig = plt.figure()
ax = plt.axes()

ax.plot(rew_median_1, color=[0,0.3,1,1])

plt.xlabel('iterations')
plt.ylabel('median reward')
plt.title(title)


#%% 01.03.2022: straight200_lanes1 decentralized vs. centralized tests with 2 actors
# rudolf with num_workers = 20

scenario = 'straight200_lanes1-4'
name_cen = 'PPO_FrameStack_d40be_00000_0_2022-02-28_06-36-58'
name_decen = 'PPO_FrameStack_4effe_00000_0_2022-02-27_20-24-48'

title = ''


progress_path_cen = os.path.join('/media/nando/Extreme SSD/MT/log/results/run',
                                 scenario,
                                 name_cen,
                                 'progress.csv')

progress_path_decen = os.path.join('/media/nando/Extreme SSD/MT/log/results/run',
                                   scenario,
                                   name_decen,
                                   'progress.csv')

df_progress_cen = pd.read_csv(progress_path_cen)
df_progress_decen = pd.read_csv(progress_path_decen)

df_ep_rew_cen = df_progress_cen['hist_stats/episode_reward']
df_ep_rew_decen = df_progress_decen['hist_stats/episode_reward']

rewards_cen = df_progress_cen[['episode_reward_mean','episode_reward_max','episode_reward_min']]
rewards_decen = df_progress_decen[['episode_reward_mean','episode_reward_max','episode_reward_min']]

rew_std_up_cen = [np.percentile(str2list(x),84) for x in df_ep_rew_cen]
rew_std_lo_cen = [np.percentile(str2list(x),16) for x in df_ep_rew_cen]
rew_std_up_decen = [np.percentile(str2list(x),84) for x in df_ep_rew_decen]
rew_std_lo_decen = [np.percentile(str2list(x),16) for x in df_ep_rew_decen]

rew_2std_up_cen = [np.percentile(str2list(x),97.5) for x in df_ep_rew_cen]
rew_2std_lo_cen = [np.percentile(str2list(x),2.5) for x in df_ep_rew_cen]
rew_2std_up_decen = [np.percentile(str2list(x),97.5) for x in df_ep_rew_decen]
rew_2std_lo_decen = [np.percentile(str2list(x),2.5) for x in df_ep_rew_decen]

rew_median_cen = [np.median(str2list(x)) for x in df_ep_rew_cen]
rew_median_decen = [np.median(str2list(x)) for x in df_ep_rew_decen]

# rew_75_decen = [np.percentile(str2list(x),75) for x in df_ep_rew_decen]
# rew_25_decen = [np.percentile(str2list(x),25) for x in df_ep_rew_decen]

# Plotting

fig = plt.figure()
ax = plt.axes()

# ax.plot(rewards_cen['episode_reward_mean'], color=[0,0.3,1,1], label='centralized')
# ax.plot(rewards_decen['episode_reward_mean'], color=[1,0.3,0,1], label='decentralized')

ax.fill_between(np.arange(1,len(rew_std_up_cen)+1), rew_std_up_cen, rew_std_lo_cen, color=[0,0.3,1,0.2])
ax.fill_between(np.arange(1,len(rew_std_up_decen)+1), rew_std_up_decen, rew_std_lo_decen, color=[1,0.3,0,0.2])

# ax.fill_between(np.arange(1,len(rew_2std_up_cen)+1), rew_2std_up_cen, rew_2std_lo_cen, color=[0,0.3,1,0.2])
# ax.fill_between(np.arange(1,len(rew_2std_up_decen)+1), rew_2std_up_decen, rew_2std_lo_decen, color=[1,0.3,0,0.2])
ax.plot(rew_median_cen, color=[0,0.3,1,1], label='centralized')
ax.plot(rew_median_decen, color=[1,0.3,0,1], label='decentralized')

plt.ylim([0, 1200])

plt.xlabel('iterations')
plt.ylabel('median reward')
plt.title(title)

plt.legend()

# plt.savefig("plots/nocross_cen_decen.pdf")
plt.show()

#%% 10.03.2022: straight100_lanes1 decentralized vs. centralized tests with 2 actors
# rudolf with num_workers = 20

scenario = 'straight100_lanes1-4'
# centralized
name_1 = 'PPO_FrameStack_fd7e5_00000_0_2022-03-09_21-36-33'
#decentralized
name_2 = 'PPO_FrameStack_5c9c6_00000_0_2022-03-09_20-34-47'

title = ''



progress_path_cen = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                scenario,
                                name_1,
                                'progress.csv')

progress_path_decen = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                scenario,
                                name_2,
                                'progress.csv')

df_progress_cen = pd.read_csv(progress_path_cen)
df_progress_decen = pd.read_csv(progress_path_decen)

df_ep_rew_cen = df_progress_cen['hist_stats/episode_reward']
df_ep_rew_decen = df_progress_decen['hist_stats/episode_reward']

rewards_cen = df_progress_cen[['episode_reward_mean','episode_reward_max','episode_reward_min']]
rewards_decen = df_progress_decen[['episode_reward_mean','episode_reward_max','episode_reward_min']]

rew_std_up_cen = [np.percentile(str2list(x),84) for x in df_ep_rew_cen]
rew_std_lo_cen = [np.percentile(str2list(x),16) for x in df_ep_rew_cen]
rew_std_up_decen = [np.percentile(str2list(x),84) for x in df_ep_rew_decen]
rew_std_lo_decen = [np.percentile(str2list(x),16) for x in df_ep_rew_decen]

rew_2std_up_cen = [np.percentile(str2list(x),97.5) for x in df_ep_rew_cen]
rew_2std_lo_cen = [np.percentile(str2list(x),2.5) for x in df_ep_rew_cen]
rew_2std_up_decen = [np.percentile(str2list(x),97.5) for x in df_ep_rew_decen]
rew_2std_lo_decen = [np.percentile(str2list(x),2.5) for x in df_ep_rew_decen]

rew_median_cen = [np.median(str2list(x)) for x in df_ep_rew_cen]
rew_median_decen = [np.median(str2list(x)) for x in df_ep_rew_decen]

# rew_75_decen = [np.percentile(str2list(x),75) for x in df_ep_rew_decen]
# rew_25_decen = [np.percentile(str2list(x),25) for x in df_ep_rew_decen]

# Plotting

fig = plt.figure()
ax = plt.axes()

# ax.plot(rewards_cen['episode_reward_mean'], color=[0,0.3,1,1], label='centralized')
# ax.plot(rewards_decen['episode_reward_mean'], color=[1,0.3,0,1], label='decentralized')

ax.fill_between(np.arange(1,len(rew_std_up_cen)+1), rew_std_up_cen, rew_std_lo_cen, color=[0,0.3,1,0.2])
ax.fill_between(np.arange(1,len(rew_std_up_decen)+1), rew_std_up_decen, rew_std_lo_decen, color=[1,0.3,0,0.2])

# ax.fill_between(np.arange(1,len(rew_2std_up_cen)+1), rew_2std_up_cen, rew_2std_lo_cen, color=[0,0.3,1,0.2])
# ax.fill_between(np.arange(1,len(rew_2std_up_decen)+1), rew_2std_up_decen, rew_2std_lo_decen, color=[1,0.3,0,0.2])
ax.plot(rew_median_cen, color=[0,0.3,1,1], label='centralized')
ax.plot(rew_median_decen, color=[1,0.3,0,1], label='decentralized')

plt.ylim([-2000, 1000])

plt.xlabel('iterations')
plt.ylabel('median reward')
plt.title(title)

plt.legend()

# plt.savefig("plots/nocross_cen_decen.pdf")
plt.show()

#%% 10.03.2022: straight50_lanes1 decentralized vs. centralized tests with 2 actors
# rudolf with num_workers = 20

scenario = 'straight50_lanes1-4'
# centralized
name_1 = 'PPO_FrameStack_ba552_00000_0_2022-03-09_19-32-59'
#decentralized
name_2 = 'PPO_FrameStack_12fd6_00000_0_2022-03-09_18-31-02'

title = ''



progress_path_cen = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                scenario,
                                name_1,
                                'progress.csv')

progress_path_decen = os.path.join('../baselines', 'marl_benchmark', 'log', 'results', 'run',
                                scenario,
                                name_2,
                                'progress.csv')

df_progress_cen = pd.read_csv(progress_path_cen)
df_progress_decen = pd.read_csv(progress_path_decen)

df_ep_rew_cen = df_progress_cen['hist_stats/episode_reward']
df_ep_rew_decen = df_progress_decen['hist_stats/episode_reward']

rewards_cen = df_progress_cen[['episode_reward_mean','episode_reward_max','episode_reward_min']]
rewards_decen = df_progress_decen[['episode_reward_mean','episode_reward_max','episode_reward_min']]

rew_std_up_cen = [np.percentile(str2list(x),84) for x in df_ep_rew_cen]
rew_std_lo_cen = [np.percentile(str2list(x),16) for x in df_ep_rew_cen]
rew_std_up_decen = [np.percentile(str2list(x),84) for x in df_ep_rew_decen]
rew_std_lo_decen = [np.percentile(str2list(x),16) for x in df_ep_rew_decen]

rew_2std_up_cen = [np.percentile(str2list(x),97.5) for x in df_ep_rew_cen]
rew_2std_lo_cen = [np.percentile(str2list(x),2.5) for x in df_ep_rew_cen]
rew_2std_up_decen = [np.percentile(str2list(x),97.5) for x in df_ep_rew_decen]
rew_2std_lo_decen = [np.percentile(str2list(x),2.5) for x in df_ep_rew_decen]

rew_median_cen = [np.median(str2list(x)) for x in df_ep_rew_cen]
rew_median_decen = [np.median(str2list(x)) for x in df_ep_rew_decen]

# rew_75_decen = [np.percentile(str2list(x),75) for x in df_ep_rew_decen]
# rew_25_decen = [np.percentile(str2list(x),25) for x in df_ep_rew_decen]

# Plotting

fig = plt.figure()
ax = plt.axes()

# ax.plot(rewards_cen['episode_reward_mean'], color=[0,0.3,1,1], label='centralized')
# ax.plot(rewards_decen['episode_reward_mean'], color=[1,0.3,0,1], label='decentralized')

ax.fill_between(np.arange(1,len(rew_std_up_cen)+1), rew_std_up_cen, rew_std_lo_cen, color=[0,0.3,1,0.2])
ax.fill_between(np.arange(1,len(rew_std_up_decen)+1), rew_std_up_decen, rew_std_lo_decen, color=[1,0.3,0,0.2])

# ax.fill_between(np.arange(1,len(rew_2std_up_cen)+1), rew_2std_up_cen, rew_2std_lo_cen, color=[0,0.3,1,0.2])
# ax.fill_between(np.arange(1,len(rew_2std_up_decen)+1), rew_2std_up_decen, rew_2std_lo_decen, color=[1,0.3,0,0.2])
ax.plot(rew_median_cen, color=[0,0.3,1,1], label='centralized')
ax.plot(rew_median_decen, color=[1,0.3,0,1], label='decentralized')

# plt.ylim([-2000, 1000])

plt.xlabel('iterations')
plt.ylabel('median reward')
plt.title(title)

plt.legend()

plt.savefig("plots/{}_cen_decen.pdf".format(scenario))
plt.show()