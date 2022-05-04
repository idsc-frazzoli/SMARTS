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
#python plotting/plotting.py -p ./merge40_lanes1-4/20220412/PPO_FrameStack_d7035_00000_0_2022-04-12_08-23-32 ./merge40_lanes1-4/20220412/PPO_FrameStack_d9b8f_00000_0_2022-04-12_17-13-19 --scenario_name merge40_lanes1_PoA --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decentralized" "centralized" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# 23.04.2022 
# 3/6 runs decent 3/6 runs cent alpha=1, degree=2, first half
#python plotting/plotting.py -p ./merge40_lanes1/alpha1_degree2/decent/PPO_FrameStack_9c927_00000_0_2022-04-22_23-47-47 ./merge40_lanes1/alpha1_degree2/decent/PPO_FrameStack_28ca3_00000_0_2022-04-22_20-16-57 ./merge40_lanes1/alpha1_degree2/decent/PPO_FrameStack_743ce_00000_0_2022-04-22_16-30-00 ./merge40_lanes1/alpha1_degree2/cent/PPO_FrameStack_57e0b_00000_0_2022-04-22_19-06-41 ./merge40_lanes1/alpha1_degree2/cent/PPO_FrameStack_69b42_00000_0_2022-04-23_00-36-28 ./merge40_lanes1/alpha1_degree2/cent/PPO_FrameStack_26889_00000_0_2022-04-22_23-01-32 --scenario_name merge40_lanes1_alpha1_degree2_first_half_ylim --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decent 2022-04-22_23-47-47" "decent 2022-04-22_20-16-57" "decent 2022-04-22_16-30-00" "cent 2022-04-22_19-06-41" "cent 2022-04-23_00-36-28" "cent 2022-04-22_23-01-32" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot --y_lim 300 500

# 3/6 runs decent 3/6 runs cent alpha=1, degree=2 second half
#python plotting/plotting.py -p ./merge40_lanes1/alpha1_degree2/decent/PPO_FrameStack_ecc46_00000_0_2022-04-22_22-09-48 ./merge40_lanes1/alpha1_degree2/decent/PPO_FrameStack_ee937_00000_0_2022-04-23_01-37-27 ./merge40_lanes1/alpha1_degree2/decent/PPO_FrameStack_f0551_00000_0_2022-04-22_18-06-31 ./merge40_lanes1/alpha1_degree2/cent/PPO_FrameStack_57563_00000_0_2022-04-23_02-37-38 ./merge40_lanes1/alpha1_degree2/cent/PPO_FrameStack_69345_00000_0_2022-04-22_17-12-38 ./merge40_lanes1/alpha1_degree2/cent/PPO_FrameStack_b5ab0_00000_0_2022-04-22_21-11-00 --scenario_name merge40_lanes1_alpha1_degree2_second_half_ylim --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decent 2022-04-22_22-09-48" "decent 2022-04-23_01-37-27" "decent 2022-04-22_18-06-31" "cent 2022-04-23_02-37-38" "cent 2022-04-22_17-12-38" "cent 2022-04-22_21-11-00" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot --y_lim 300 500

# asym time cost tests decent
#python plotting/plotting.py -p ./merge40_lanes1_asym_time_test/alpha1_degree2/decent/PPO_FrameStack_2fa67_00000_0_2022-04-23_10-00-21 ./merge40_lanes1_asym_time_test/alpha1_degree2/decent/PPO_FrameStack_febfa_00000_0_2022-04-23_10-41-55 --scenario_name merge40_lanes1_saym_time_cost --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decent 2022-04-23_10-00-21" "decent 2022-04-23_10-41-55" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# 03.05.2022 merge90_lanes32 tests
#python plotting/plotting.py -p ./20220502_merge90_lanes32/alpha1_degree2/cent/run/merge90_lanes32-4/PPO_FrameStack_2ff16_00000_0_2022-05-02_18-03-36 ./20220502_merge90_lanes32/alpha1_degree2/cent/run/merge90_lanes32-4/PPO_FrameStack_23195_00000_0_2022-05-03_00-15-28 ./20220502_merge90_lanes32/alpha1_degree2/decent/run/merge90_lanes32-4/PPO_FrameStack_8b6a8_00000_0_2022-05-03_00-18-23 ./20220502_merge90_lanes32/alpha1_degree2/cent/run/merge90_lanes32-4/PPO_FrameStack_79ec0_00000_0_2022-05-03_06-08-39 ./20220502_merge90_lanes32/alpha1_degree2/decent/run/merge90_lanes32-4/PPO_FrameStack_10565_00000_0_2022-05-02_18-02-43 ./20220502_merge90_lanes32/alpha1_degree2/decent/run/merge90_lanes32-4/PPO_FrameStack_ffe12_00000_0_2022-05-03_05-22-18 --scenario_name merge90_lanes32_cent_decent_2_runs_each --mean_reward --mean_len --agent_wise --learner_stats --png --legend "cent 2022-05-02_18-03-36" "cent 2022-05-03_00-15-28" "cent 2022-05-03_06-08-39" "decent 2022-05-03_00-18-23" "decent 2022-05-02_18-02-43" "decent 2022-05-03_05-22-18" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# 03.05.2022 merge65_lanes2 tests
#python plotting/plotting.py -p ./20220503_merge65_lanes42/alpha1_degree2/cent/run/merge65_lanes42-4/PPO_FrameStack_7be5c_00000_0_2022-05-03_12-35-15 ./20220503_merge65_lanes42/alpha1_degree2/cent/run/merge65_lanes42-4/PPO_FrameStack_9ee66_00000_0_2022-05-03_14-37-56 ./20220503_merge65_lanes42/alpha1_degree2/decent/run/merge65_lanes42-4/PPO_FrameStack_219bd_00000_0_2022-05-03_12-32-44 ./20220503_merge65_lanes42/alpha1_degree2/decent/run/merge65_lanes42-4/PPO_FrameStack_9295f_00000_0_2022-05-03_14-44-44 --scenario_name merge65_lanes42_cent_decent_tests --mean_reward --mean_len --agent_wise --learner_stats --png --legend "cent 2022-05-03_12-35-15" "cent 2022-05-03_14-37-56" "decent 2022-05-03_12-32-44" "decent 2022-05-03_14-44-44" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# 03.05.2022 merge65_lanes2 cost 07 tests cent
python plotting/plotting.py -p ./20220503_1_merge65_lanes42/alpha1_degree2/cent/run/merge65_lanes42-4/PPO_FrameStack_73bc9_00000_0_2022-05-03_21-03-16 ./20220503_1_merge65_lanes42/alpha1_degree2/cent/run/merge65_lanes42-4/PPO_FrameStack_80f50_00000_0_2022-05-04_01-21-20 ./20220503_1_merge65_lanes42/alpha1_degree2/cent/run/merge65_lanes42-4/PPO_FrameStack_1717d_00000_0_2022-05-03_18-51-50 ./20220503_1_merge65_lanes42/alpha1_degree2/cent/run/merge65_lanes42-4/PPO_FrameStack_ca073_00000_0_2022-05-03_23-14-32 ./20220503_1_merge65_lanes42/alpha1_degree2/cent/run/merge65_lanes42-4/PPO_FrameStack_e7c55_00000_0_2022-05-03_16-34-30 --scenario_name merge65_lanes42_cent_tests --mean_reward --mean_len --agent_wise --learner_stats --png --legend "cent 2022-05-03_21-03-16" "cent 2022-05-04_01-21-20" "cent 2022-05-03_18-51-50" "cent 2022-05-03_23-14-32" "cent 2022-05-03_16-34-30" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot

# 03.05.2022 merge65_lanes2 cost 07 tests decent
python plotting/plotting.py -p ./20220503_1_merge65_lanes42/alpha1_degree2/decent/run/merge65_lanes42-4/PPO_FrameStack_5ba24_00000_0_2022-05-03_21-16-55 ./20220503_1_merge65_lanes42/alpha1_degree2/decent/run/merge65_lanes42-4/PPO_FrameStack_6dbeb_00000_0_2022-05-03_23-33-25 ./20220503_1_merge65_lanes42/alpha1_degree2/decent/run/merge65_lanes42-4/PPO_FrameStack_aa792_00000_0_2022-05-04_01-51-08 ./20220503_1_merge65_lanes42/alpha1_degree2/decent/run/merge65_lanes42-4/PPO_FrameStack_bbf6e_00000_0_2022-05-03_16-33-16 ./20220503_1_merge65_lanes42/alpha1_degree2/decent/run/merge65_lanes42-4/PPO_FrameStack_ffa84_00000_0_2022-05-03_18-58-20 --scenario_name merge65_lanes42_decent_tests --mean_reward --mean_len --agent_wise --learner_stats --png --legend "decent 2022-05-03_21-16-55" "decent 2022-05-03_23-33-25" "decent 2022-05-04_01-51-08" "decent 2022-05-03_16-33-16" "decent 2022-05-03_18-58-20" --x_axis checkpoints time_total_s episodes_total timesteps_total --boxplot













