#!/bin/bash

# 21.03.2022
# 24h decent vs cent merge110_lanes2 fully lexicographic cost
# 20220321_202101_merge110_lanes2
#python plotting/plotting.py -p ./merge110_lanes2-4/PPO_FrameStack_93b35_00000_0_2022-03-18_16-00-48 ./merge110_lanes2-4/PPO_FrameStack_1728f_00000_0_2022-03-19_17-07-43 --scenario_name merge110_lanes2 --title "Decentralized vs. Centralized" --mean_reward --agent_wise --learner_stats --png --legend decentralized centralized

# 10h decent vs 20 hcent merge110_lanes2 lexicographic cost
# 20220321_203245_merge110_lanes2
#python plotting/plotting.py -p ./merge110_lanes2-4/PPO_FrameStack_4f155_00000_0_2022-03-16_08-44-36 ./merge110_lanes2-4/PPO_FrameStack_84551_00000_0_2022-03-15_17-08-21 ./merge110_lanes2-4/PPO_FrameStack_f8f98_00000_0_2022-03-17_11-39-58 --scenario_name merge110_lanes2 --title "Decentralized vs. Centralized" --mean_reward --agent_wise --learner_stats --png --legend "decentralized run 1" "decentralized run 2"  "centralized"

# 2h test for intersection
# 20220321_204134_int_4_rand2
#python plotting/plotting.py -p ./int_4_rand2-4/PPO_FrameStack_1d4ea_00000_0_2022-03-21_14-13-43 --scenario_name int_4_rand2 --title "Intersection Decentralized Test" --mean_reward --mean_len --agent_wise --learner_stats --png
