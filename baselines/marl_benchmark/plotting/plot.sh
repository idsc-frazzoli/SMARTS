#!/bin/bash

# 23.03.2022
# 24h decent vs cent merge110_lanes2 fully lexicographic cost
# 20220323_172214_merge110_lanes2
python plotting/plotting.py -p ./merge110_lanes2-4/PPO_FrameStack_93b35_00000_0_2022-03-18_16-00-48 ./merge110_lanes2-4/PPO_FrameStack_1728f_00000_0_2022-03-19_17-07-43 --scenario_name merge110_lanes2 --title "Decentralized vs. Centralized" --mean_reward --mean_len --agent_wise --learner_stats --png --legend decentralized centralized --x_axis checkpoints time_total_s episodes_total

# 10h decent vs 20 hcent merge110_lanes2 lexicographic cost
# 20220323_172223_merge110_lanes2
python plotting/plotting.py -p ./merge110_lanes2-4/PPO_FrameStack_4f155_00000_0_2022-03-16_08-44-36 ./merge110_lanes2-4/PPO_FrameStack_84551_00000_0_2022-03-15_17-08-21 ./merge110_lanes2-4/PPO_FrameStack_f8f98_00000_0_2022-03-17_11-39-58 --scenario_name merge110_lanes2 --title "Decentralized vs. Centralized" --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decentralized run 1" "decentralized run 2"  "centralized" --x_axis checkpoints time_total_s episodes_total


# 2h test for intersection
# 20220323_172238_int_4_rand2
python plotting/plotting.py -p ./int_4_rand2-4/PPO_FrameStack_1d4ea_00000_0_2022-03-21_14-13-43 --scenario_name int_4_rand2 --title "Intersection Decentralized Test" --mean_reward --mean_len --agent_wise --learner_stats --png --x_axis checkpoints time_total_s episodes_total

# 10h decent vs. cent for intersection
# 20220323_172244_int_4_rand2
python plotting/plotting.py -p ./int_4_rand2-4/PPO_FrameStack_fedd0_00000_0_2022-03-21_20-46-34 ./int_4_rand2-4/PPO_FrameStack_5fb1d_00000_0_2022-03-22_06-50-34 --scenario_name int_4_rand2 --title "Intersection Decentralized vs. Centralized" --mean_reward --mean_len --png --x_axis checkpoints time_total_s episodes_total --legend "decentralized" "centralized" --agent_wise --learner_stats
