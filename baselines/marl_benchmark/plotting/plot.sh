#!/bin/bash

# 23.03.2022
# 24h decent vs cent merge110_lanes2 fully lexicographic cost (2)
# 
#python plotting/plotting.py -p ./merge110_lanes2-4/PPO_FrameStack_93b35_00000_0_2022-03-18_16-00-48 ./merge110_lanes2-4/PPO_FrameStack_1728f_00000_0_2022-03-19_17-07-43 --scenario_name merge110_lanes2_lexicost2 --title "Decentralized vs. Centralized" --mean_reward --mean_len --agent_wise --learner_stats --png --legend decentralized centralized --x_axis checkpoints time_total_s episodes_total timesteps_total

# 10h/10h/20h decent vs 20 h cent merge110_lanes2 lexicographic cost (1)
# 
#python plotting/plotting.py -p ./merge110_lanes2-4/PPO_FrameStack_4f155_00000_0_2022-03-16_08-44-36 ./merge110_lanes2-4/PPO_FrameStack_84551_00000_0_2022-03-15_17-08-21 ./merge110_lanes2-4/PPO_FrameStack_a0024_00000_0_2022-03-22_20-21-16 ./merge110_lanes2-4/PPO_FrameStack_f8f98_00000_0_2022-03-17_11-39-58 --scenario_name merge110_lanes2_lexicost1 --title "Decentralized vs. Centralized" --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decentralized run 1" "decentralized run 2" "decentralized run 3" "centralized" --x_axis checkpoints time_total_s episodes_total timesteps_total

# 2h test for intersection
# 
#python plotting/plotting.py -p ./int_4_rand2-4/PPO_FrameStack_1d4ea_00000_0_2022-03-21_14-13-43 --scenario_name int_4_rand2_lexicost2_test --title "Intersection Decentralized Test" --mean_reward --mean_len --agent_wise --learner_stats --png --x_axis checkpoints time_total_s episodes_total timesteps_total

# 10h decent vs. cent for intersection
# 
#python plotting/plotting.py -p ./int_4_rand2-4/PPO_FrameStack_fedd0_00000_0_2022-03-21_20-46-34 ./int_4_rand2-4/PPO_FrameStack_5fb1d_00000_0_2022-03-22_06-50-34 --scenario_name int_4_rand2_lexicost2 --title "Intersection Decentralized vs. Centralized" --mean_reward --mean_len --png --x_axis checkpoints time_total_s episodes_total timesteps_total --legend "decentralized" "centralized" --agent_wise --learner_stats

# 20h decent vs 20 h cent merge110_lanes2 lexicographic cost (1)
# 
#python plotting/plotting.py -p ./merge110_lanes2-4/PPO_FrameStack_a0024_00000_0_2022-03-22_20-21-16 ./merge110_lanes2-4/PPO_FrameStack_f8f98_00000_0_2022-03-17_11-39-58 --scenario_name merge110_lanes2_lexicost1 --title "Decentralized vs. Centralized" --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decentralized" "centralized" --x_axis checkpoints time_total_s episodes_total timesteps_total

# 23.03.2022
# int31 tests decent
#python plotting/plotting.py -p ./int31-4/PPO_FrameStack_f334d_00000_0_2022-03-24_07-28-07 --scenario_name int31_lexicost2 --title "Intersection Tests" --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decentralized" --x_axis checkpoints time_total_s episodes_total timesteps_total

# 28.03.2022
#python plotting/plotting.py -p ./merge100_lanes2-4/PPO_FrameStack_a627e_00000_0_2022-03-26_11-12-40 ./merge100_lanes2-4/PPO_FrameStack_dbf10_00000_0_2022-03-27_07-52-33 --scenario_name merge110_lanes2_lr_schedule_tests --title "Learning Rate Tests" --mean_reward --mean_len --agent_wise --learner_stats --png --legend "lr_schedule 1" "lr_schedule 2" --x_axis checkpoints time_total_s episodes_total timesteps_total

# 29.03.2022
#python plotting/plotting.py -p ./merge100_lanes2-4/PPO_FrameStack_2e54c_00000_0_2022-03-28_10-38-18 --scenario_name merge110_lanes2_constant_lr --mean_reward --mean_len --agent_wise --learner_stats --png --legend "constant learning rate" --x_axis checkpoints time_total_s episodes_total timesteps_total

# 30.03.2022
#python plotting/plotting.py -p ./merge100_lanes2-4/PPO_FrameStack_2e54c_00000_0_2022-03-28_10-38-18 ./merge100_lanes2-4/PPO_FrameStack_8ba13_00000_0_2022-03-30_02-10-18 ./merge100_lanes2-4/PPO_FrameStack_3254c_00000_0_2022-03-29_16-06-30 --scenario_name merge110_lanes2_different_lr_schedules --mean_reward --mean_len --agent_wise --learner_stats --png --legend "constant" "linear decline" "drop" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# 05.04.2022 
#python plotting/plotting.py -p ./merge110_lanes2-4/PPO_FrameStack_fada6_00000_0_2022-04-04_08-34-02 ./merge110_lanes2-4/PPO_FrameStack_d3992_00000_0_2022-04-04_19-02-51 --scenario_name merge110_lanes2_decent_vs_cent_lr_schedule --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decentralized" "centralized" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# 07.04.2022
#python plotting/plotting.py -p ./merge110_lanes2-4/PPO_FrameStack_7b4bb_00000_0_2022-04-05_15-10-08 ./merge110_lanes2-4/PPO_FrameStack_3b54a_00000_0_2022-04-06_12-36-51 --scenario_name merge110_lanes2_decent_vs_cent_lr_schedule --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decentralized" "centralized" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# 12.04.2022 
#python plotting/plotting.py -p ./merge40_lanes1-4/PPO_FrameStack_40abf_00000_0_2022-04-12_06-10-29 ./merge40_lanes1-4/PPO_FrameStack_d7035_00000_0_2022-04-12_08-23-32 --scenario_name merge40_lanes1_decent_test --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decentralized 1" "decentralized 2" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# 13.04.2022
#python plotting/plotting.py -p ./merge40_lanes1-4/PPO_FrameStack_3d116_00000_0_2022-04-12_15-35-53 ./merge40_lanes1-4/PPO_FrameStack_9e9de_00000_0_2022-04-12_13-08-17 ./merge40_lanes1-4/PPO_FrameStack_40abf_00000_0_2022-04-12_06-10-29 ./merge40_lanes1-4/PPO_FrameStack_48e37_00000_0_2022-04-12_20-36-52 ./merge40_lanes1-4/PPO_FrameStack_69ef9_00000_0_2022-04-12_20-02-00 ./merge40_lanes1-4/PPO_FrameStack_66982_00000_0_2022-04-12_19-26-07 ./merge40_lanes1-4/PPO_FrameStack_72221_00000_0_2022-04-12_21-13-49 ./merge40_lanes1-4/PPO_FrameStack_80346_00000_0_2022-04-12_18-00-56 ./merge40_lanes1-4/PPO_FrameStack_a0b72_00000_0_2022-04-12_16-35-56 ./merge40_lanes1-4/PPO_FrameStack_a3747_00000_0_2022-04-12_18-37-42 ./merge40_lanes1-4/PPO_FrameStack_d4e34_00000_0_2022-04-12_14-35-42 ./merge40_lanes1-4/PPO_FrameStack_d9b8f_00000_0_2022-04-12_17-13-19 ./merge40_lanes1-4/PPO_FrameStack_d7035_00000_0_2022-04-12_08-23-32 ./merge40_lanes1-4/PPO_FrameStack_f00a2_00000_0_2022-04-12_22-00-17 --scenario_name merge40_lanes1_decent_vs_cent_everything --mean_reward --mean_len --agent_wise --learner_stats --png --legend "cent" "cent" "decent" "decent" "cent" "decent" "cent" "decent" "decent" "cent" "decent" "cent" "decent" "decent" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# only decent 1 converged
#python plotting/plotting.py -p ./merge40_lanes1-4/PPO_FrameStack_40abf_00000_0_2022-04-12_06-10-29 ./merge40_lanes1-4/PPO_FrameStack_48e37_00000_0_2022-04-12_20-36-52  ./merge40_lanes1-4/PPO_FrameStack_66982_00000_0_2022-04-12_19-26-07 ./merge40_lanes1-4/PPO_FrameStack_80346_00000_0_2022-04-12_18-00-56 ./merge40_lanes1-4/PPO_FrameStack_a0b72_00000_0_2022-04-12_16-35-56 ./merge40_lanes1-4/PPO_FrameStack_d4e34_00000_0_2022-04-12_14-35-42 ./merge40_lanes1-4/PPO_FrameStack_f00a2_00000_0_2022-04-12_22-00-17 --scenario_name merge40_lanes1_only_decent --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decent 1" "decent 2" "decent 3" "decent 4" "decent 5" "decent 6" "decent 7" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# contains runs that didn't converge (cent 3)
#python plotting/plotting.py -p ./merge40_lanes1-4/20220412/PPO_FrameStack_3d116_00000_0_2022-04-12_15-35-53 ./merge40_lanes1-4/20220412/PPO_FrameStack_9e9de_00000_0_2022-04-12_13-08-17 ./merge40_lanes1-4/20220412/PPO_FrameStack_69ef9_00000_0_2022-04-12_20-02-00 ./merge40_lanes1-4/20220412/PPO_FrameStack_72221_00000_0_2022-04-12_21-13-49 ./merge40_lanes1-4/20220412/PPO_FrameStack_a3747_00000_0_2022-04-12_18-37-42 ./merge40_lanes1-4/20220412/PPO_FrameStack_d9b8f_00000_0_2022-04-12_17-13-19 --scenario_name merge40_lanes1_only_cent --mean_reward --mean_len --agent_wise --learner_stats --png --legend "cent 1" "cent 2" "cent 3" "cent 4" "cent 5" "cent 6" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# contains trainings that didn't converge (decent 1, 2, 3)
#python plotting/plotting.py -p ./merge40_lanes1-4/20220413/PPO_FrameStack_4f086_00000_0_2022-04-13_12-36-15 ./merge40_lanes1-4/20220413/PPO_FrameStack_39b32_00000_0_2022-04-13_11-38-23 ./merge40_lanes1-4/20220413/PPO_FrameStack_91fc2_00000_0_2022-04-13_10-00-38 ./merge40_lanes1-4/20220413/PPO_FrameStack_382d4_00000_0_2022-04-13_15-56-02 ./merge40_lanes1-4/20220413/PPO_FrameStack_486b1_00000_0_2022-04-13_10-41-32 --scenario_name merge40_lanes1_only_decent --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decent 1" "decent 2" "decent 3" "decent 4" "decent 5" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# decent runs that converged
#python plotting/plotting.py -p ./merge40_lanes1-4/20220412/PPO_FrameStack_40abf_00000_0_2022-04-12_06-10-29 ./merge40_lanes1-4/20220413/PPO_FrameStack_382d4_00000_0_2022-04-13_15-56-02 ./merge40_lanes1-4/20220413/PPO_FrameStack_486b1_00000_0_2022-04-13_10-41-32 ./merge40_lanes1-4/20220412/PPO_FrameStack_d7035_00000_0_2022-04-12_08-23-32 --scenario_name merge40_lanes1_only_converged_decent --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decent 1" "decent 2" "decent 3" "decent 4" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# 21.04.2022 PoA evaluation
python plotting/plotting.py -p ./merge40_lanes1-4/20220412/PPO_FrameStack_d7035_00000_0_2022-04-12_08-23-32 ./merge40_lanes1-4/20220412/PPO_FrameStack_d9b8f_00000_0_2022-04-12_17-13-19 --scenario_name merge40_lanes1_PoA --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decentralized" "centralized" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot




