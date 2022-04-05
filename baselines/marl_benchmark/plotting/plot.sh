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
python plotting/plotting.py -p ./merge110_lanes2-4/PPO_FrameStack_fada6_00000_0_2022-04-04_08-34-02 ./merge110_lanes2-4/PPO_FrameStack_d3992_00000_0_2022-04-04_19-02-51 --scenario_name merge110_lanes2_decent_vs_cent_lr_schedule --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decentralized" "centralized" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot
