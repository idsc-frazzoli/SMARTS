#!/bin/bash

# 15.02.2022
#python run.py scenarios/double_merge/merge_asym2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm decentralized --stop_time 4000
#python run.py scenarios/double_merge/merge_asym2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm centralized --stop_time 4000

#python run.py scenarios/custom/long_merge -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm decentralized --stop_time 7200
#python run.py scenarios/custom/long_merge -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm centralized --stop_time 7200

#python run.py scenarios/custom/long_merge -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --restore_path ./log/results/run/long_merge-4/PPO_FrameStack_8fc32_00000_0_2022-02-15_13-31-20/checkpoint_000232/checkpoint-232 --headless --paradigm decentralized --stop_time 36000 

#python run.py scenarios/custom/long_merge -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --restore_path ./log/results/run/long_merge-4/PPO_FrameStack_7febb_00000_0_2022-02-15_15-32-35/checkpoint_000464/checkpoint-464 --headless --paradigm centralized --stop_time 36000 

# 16.02.2022
#python run.py scenarios/custom/merge_asym3 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm decentralized --stop_time 7200
#python run.py scenarios/custom/merge_asym3 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm centralized --stop_time 7200

#python run.py scenarios/custom/merge_asym4 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm decentralized --stop_time 21600
#python run.py scenarios/custom/merge_asym4 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm centralized --stop_time 21600

# 20.02.2022
#python run.py scenarios/custom/straight/straight100_lanes4 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm decentralized --stop_time 10000
#python run.py scenarios/custom/merge_asym5 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm decentralized --stop_time 10000

# 25.02.2022
#python run.py scenarios/custom/straight/straight100_lanes4 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm decentralized --stop_time 36000

# 26.02.2022
#python run.py scenarios/custom/straight/straight100_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 20 --headless --paradigm decentralized --stop_time 9000

# 27.02.2022
#python run.py scenarios/custom/straight/straight100_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 20 --headless --paradigm decentralized --stop_time 10000
#python run.py scenarios/custom/straight/straight200_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 20 --headless --paradigm decentralized --stop_time 36000
#python run.py scenarios/custom/straight/straight200_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 20 --headless --paradigm centralized --stop_time 36000

# 08.03.2022 / 09.03.2022
#python run.py scenarios/custom/straight/straight200_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 20 --headless --paradigm decentralized --stop_time 7200 --horizon 200
#python run.py scenarios/custom/straight/straight200_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 20 --headless --paradigm centralized --stop_time 7200 --horizon 200

# 09.03.2022
#python run.py scenarios/custom/straight/straight200_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 20 --headless --paradigm decentralized --stop_time 3600 --horizon 200

#scl scenario build-all scenarios/custom/straight/straight50_lanes1
#scl scenario build-all scenarios/custom/straight/straight100_lanes1
#python run.py scenarios/custom/straight/straight50_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 20 --headless --paradigm decentralized --stop_time 3600 --horizon 60
#python run.py scenarios/custom/straight/straight50_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 20 --headless --paradigm centralized --stop_time 3600 --horizon 60
#python run.py scenarios/custom/straight/straight100_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 20 --headless --paradigm decentralized --stop_time 3600 --horizon 120
#python run.py scenarios/custom/straight/straight100_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 20 --headless --paradigm centralized --stop_time 3600 --horizon 120

# 10.03.2022 (14.03.2022)
#scl scenario build-all scenarios/custom/straight/straight100_lanes1
#python run.py scenarios/custom/straight/straight100_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 3600 --horizon 140
#python run.py scenarios/custom/straight/straight100_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm centralized --stop_time 7200 --horizon 140

# 14.03.2022
#scl scenario build-all scenarios/custom/merge110_lanes2
#python run.py scenarios/custom/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 18000 --horizon 300
##python run.py scenarios/custom/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm centralized --stop_time 36000 --horizon 300

# 15.03.2022
#scl scenario build-all scenarios/custom/merge110_lanes2
#python run.py scenarios/custom/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 18000 --horizon 150
# run for an other 5 hours
#python run.py scenarios/custom/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 18000 --horizon 150 --restore_path ./log/results/run/merge110_lanes2-4/PPO_FrameStack_67136_00000_0_2022-03-15_10-33-50/checkpoint_000290/checkpoint-290
# run centralized for 10 hours
#python run.py scenarios/custom/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 36000 --horizon 150

# 17.03.2022 run centralized for 20 hrs
#python run.py scenarios/custom/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm centralized --stop_time 72000 --horizon 150

# 18.03.2022 decentralized vs. centralized for 24 hrs each with lexi cost
#python run.py scenarios/custom/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 86400 --horizon 150
#python run.py scenarios/custom/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm centralized --stop_time 86400 --horizon 150

# 21.03.2022 decentralized training on new wintersection scenario 2hrs
#scl scenario build-all scenarios/custom/intersection/int_4_rand2
#python run.py scenarios/custom/intersection/int_4_rand2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 7200 --horizon 150
#python run.py scenarios/custom/intersection/int_4_rand2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 36000 --horizon 150
#python run.py scenarios/custom/intersection/int_4_rand2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm centralized --stop_time 36000 --horizon 150

# 22.03.2022 repeat the experiment from 15.03.2022 for 20 hrs this time
#python run.py scenarios/custom/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 72000 --horizon 150

# 23.03.2022 int31 experiment
#scl scenario build-all scenarios/custom/intersection/int31
#python run.py scenarios/custom/intersection/int31 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 36000 --horizon 80

# 26.03.2022 merge100_lanes2 (randomized starting positions) decent training with different lr_schedules (20 hrs each)
#scl scenario build-all scenarios/custom/merge/merge100_lanes2
#python run.py scenarios/custom/merge/merge100_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 72000 --horizon 150
#python run.py scenarios/custom/merge/merge100_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule1.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 72000 --horizon 150

# 28.03.2022 merge100_lanes2 (randomized starting positions) decent training with different lr_schedules (constant and linear decline), 10 hrs each
#python run.py scenarios/custom/merge/merge100_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 36000 --horizon 150
#python run.py scenarios/custom/merge/merge100_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule2.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 36000 --horizon 150
#python run.py scenarios/custom/merge/merge100_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule3.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 36000 --horizon 150

# 30.03.2022 merge100_lanes2, continue training from checkpoints
#python run.py scenarios/custom/merge/merge100_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule2.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 54000 --horizon 150 --restore_path ./log/results/run/merge100_lanes2-4/PPO_FrameStack_3254c_00000_0_2022-03-29_16-06-30/checkpoint_000500/checkpoint-500
#python run.py scenarios/custom/merge/merge100_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule3.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 54000 --horizon 150 --restore_path ./log/results/run/merge100_lanes2-4/PPO_FrameStack_8ba13_00000_0_2022-03-30_02-10-18/checkpoint_000512/checkpoint-512

# 04.04.2022 merge110_lanes2 training cent vs. decent with lr_schedule4 (timesteps_total: 4.5 mio), cost 04
#python run.py scenarios/custom/merge/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule4.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 150000 --horizon 150
#python run.py scenarios/custom/merge/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule4.yaml --num_workers 19 --headless --paradigm centralized --stop_time 150000 --horizon 150

# 05.04.2022 merge110_lanes2 training cent vs. decent with lr_schedule5/6 (episodes_total: 50 k), pseudolexicost with ttc cost (cost 05)
#python run.py scenarios/custom/merge/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule5.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 150000 --horizon 150
#python run.py scenarios/custom/merge/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule6.yaml --num_workers 19 --headless --paradigm centralized --stop_time 150000 --horizon 150

# 11.04.2022 merge40_lanes1 training decent test with lr_schedule7 (episodes_total: 10k), cost 06
#scl scenario build-all scenarios/custom/merge/merge40_lanes1
#python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule7.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 150000 --horizon 60

# 12.04.2022 merge40_lanes1 training decent vs cent with lr_schedule7/8 (episodes_total: 13k), cost 06, run locally
#for (( i=0; i<5; i++ ))
#do
#  python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule8.yaml --num_workers 3 --headless --paradigm centralized --stop_time 150000 --horizon 60
#  python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule7.yaml --num_workers 3 --headless --paradigm decentralized --stop_time 150000 --horizon 60
#done


# 13.04.2022 merge40_lanes1 training decent with lr_schedule9 (episodes_total: 16k), cost 06, run locally
#for (( i=0; i<1; i++ ))
#do
#  python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule9.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60
#done

#python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_lr_schedule10.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60

# 22.04.2022 merge40_lanes1, 15k episodes per run, 6 runs for each decent, cent
#for (( i=0; i<6; i++ ))
#do
#  python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_decent_1_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/merge40_lanes1/alpha1_degree2/decent
#  python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_cent_1_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/merge40_lanes1/alpha1_degree2/cent
#done

## 23.04.2022 merge40_lanes1, 15k episodes per run, 10 runs for decent, asym time cost tests (1, 3)
#for (( i=0; i<10; i++ ))
#do
#  python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_decent_1_2.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/merge40_lanes1_asym_time_test/alpha1_degree2/decent
#done

# 25.04.2022
# merge40_lanes1, 20k episodes per run, 10 runs for decent and cent, asym time cost tests (1, 5)
#for (( i=0; i<20; i++ ))
#do
#  python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_decent_1_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220425_merge40_lanes1_asym_time_test_1_5/alpha1_degree2/decent
##  python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_cent_1_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220425_merge40_lanes1_asym_time_test_1_5/alpha1_degree2/cent
#done

# merge40_lanes1, 20k episodes per run, 10 runs for decent and cent, asym time cost tests (1, 5)
#for (( i=0; i<20; i++ ))
#do
##  python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_decent_1_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220425_merge40_lanes1_asym_time_test_1_3/alpha1_degree2/decent
#  python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_cent_1_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220425_merge40_lanes1_asym_time_test_1_3/alpha1_degree2/cent
#done


# 28.04.2022 tests with different starting positions for agent 0
##scl scenario build-all scenarios/custom/merge/merge40_lanes1_2
##scl scenario build-all scenarios/custom/merge/merge40_lanes1_3
##scl scenario build-all scenarios/custom/merge/merge40_lanes1_4
#for (( i=0; i<7; i++ ))
#do
##  python run.py scenarios/custom/merge/merge40_lanes1_2 -f marl_benchmark/agents/ppo/baseline-lane-control_decent_1_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220428_merge40_lanes1_2_asym_time_test_1_5/alpha1_degree2/decent
##  python run.py scenarios/custom/merge/merge40_lanes1_2 -f marl_benchmark/agents/ppo/baseline-lane-control_cent_1_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220428_merge40_lanes1_2_asym_time_test_1_5/alpha1_degree2/cent
##  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/baseline-lane-control_decent_1_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220428_merge40_lanes1_3_asym_time_test_1_5/alpha1_degree2/decent
##  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/baseline-lane-control_cent_1_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220428_merge40_lanes1_3_asym_time_test_1_5/alpha1_degree2/cent
#  python run.py scenarios/custom/merge/merge40_lanes1_4 -f marl_benchmark/agents/ppo/baseline-lane-control_decent_1_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220428_merge40_lanes1_4_asym_time_test_1_5/alpha1_degree2/decent
#  python run.py scenarios/custom/merge/merge40_lanes1_4 -f marl_benchmark/agents/ppo/baseline-lane-control_cent_1_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220428_merge40_lanes1_4_asym_time_test_1_5/alpha1_degree2/cent
#
#done


# test merge75_lanes321
#scl scenario build-all scenarios/custom/merge/merge75_lanes321
#python run.py scenarios/custom/merge/merge75_lanes321 -f marl_benchmark/agents/ppo/baseline-lane-control_decent_1_2_test.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220429_merge75_lanes321/alpha1_degree2/decent

## 5 runs decent and cent with merge90_lanes32 symmetric cost 08, 20k episodes per run
#scl scenario build-all scenarios/custom/merge/merge90_lanes32
#for (( i=0; i<5; i++ ))
#do
##  python run.py scenarios/custom/merge/merge90_lanes32 -f marl_benchmark/agents/ppo/baseline-lane-control_decent_1_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220502_merge90_lanes32/alpha1_degree2/decent
#  python run.py scenarios/custom/merge/merge90_lanes32 -f marl_benchmark/agents/ppo/baseline-lane-control_cent_1_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220502_merge90_lanes32/alpha1_degree2/cent
#done

## 5 runs decent and cent with merge65_lanes42 symmetric cost 07, 13k episodes per run, alpha = 1, degree = 2
#scl scenario build-all scenarios/custom/merge/merge65_lanes42
#for (( i=0; i<5; i++ ))
#do
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha/baseline-lane-control_decent_1_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220503_merge65_lanes42/alpha1_degree2/decent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha/baseline-lane-control_cent_1_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220503_merge65_lanes42/alpha1_degree2/cent
#done

## 5 runs decent and cent with merge65_lanes42 symmetric cost 07, 13k episodes per run, alpha = 0.1, degree = 2
#scl scenario build-all scenarios/custom/merge/merge65_lanes42
#for (( i=0; i<5; i++ ))
#do
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha/baseline-lane-control_decent_01_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220504_merge65_lanes42/alpha01_degree2/decent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha/baseline-lane-control_cent_01_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220504_merge65_lanes42/alpha01_degree2/cent
#done

## 5 runs decent and cent with merge65_lanes42 symmetric cost 07, 13k episodes per run, alpha = 2, degree = 2
#scl scenario build-all scenarios/custom/merge/merge65_lanes42
#for (( i=0; i<5; i++ ))
#do
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha/baseline-lane-control_decent_2_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220504_merge65_lanes42/alpha2_degree2/decent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha/baseline-lane-control_cent_2_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220504_merge65_lanes42/alpha2_degree2/cent
#done

## 5 runs decent and cent with merge65_lanes42 symmetric cost 07, 13k episodes per run, alpha = 10, degree = 2
#scl scenario build-all scenarios/custom/merge/merge65_lanes42
#for (( i=0; i<5; i++ ))
#do
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha/baseline-lane-control_decent_10_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220504_merge65_lanes42/alpha10_degree2/decent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha/baseline-lane-control_cent_10_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220504_merge65_lanes42/alpha10_degree2/cent
#done


## 10.05.2022 runs
#for (( i=0; i<5; i++ ))
#do
##  # container 1: cent, decent alpha=0p5, degree=2; cent alpha=0p5, degree=1
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_cent_0p5_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha0p5_degree2/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_decent_0p5_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha0p5_degree2/decent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_cent_0p5_1.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha0p5_degree1/cent
##  # container 2: cent, decent alpha=1, degree=2; cent alpha=1, degree=1
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_cent_1_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha1_degree2/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_decent_1_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha1_degree2/decent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_cent_1_1.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha1_degree1/cent
##  # container 3: cent, decent alpha=2, degree=2; cent alpha=2, degree=1
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_cent_2_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha2_degree2/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_decent_2_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha2_degree2/decent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_cent_2_1.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha2_degree1/cent
##  # container 4: cent, decent alpha=0p5, degree=1p5; decent alpha=0p5, degree=1
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_cent_0p5_1p5.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha0p5_degree1p5/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_decent_0p5_1p5.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha0p5_degree1p5/decent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_decent_0p5_1.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha0p5_degree1/decent
##  # container 5: cent, decent alpha=1, degree=1p5; decent alpha=1, degree=1
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_cent_1_1p5.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha1_degree1p5/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_decent_1_1p5.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha1_degree1p5/decent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_decent_1_1.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha1_degree1/decent
#  # container 6: cent, decent alpha=2, degree=1p5; decent alpha=2, degree=1
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_cent_2_1p5.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha2_degree1p5/cent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_decent_2_1p5.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha2_degree1p5/decent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/merge65_lanes42_alpha_degree/baseline-lane-control_decent_2_1.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220510_merge65_lanes42/alpha2_degree1/decent
#
#done


## 13.05.2022 runs
#for (( i=0; i<5; i++ ))
#do
##  # container 1
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_cent_0p100_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p100_degree2/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_0p100_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p100_degree2/decent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_cent_0p154_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p154_degree2/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_0p154_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p154_degree2/decent
##  # container 2
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_cent_0p239_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p239_degree2/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_0p239_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p239_degree2/decent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_cent_0p368_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p368_degree2/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_0p368_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p368_degree2/decent
##  # container 3
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_cent_0p570_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p570_degree2/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_0p570_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p570_degree2/decent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_cent_0p879_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p879_degree2/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_0p879_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p879_degree2/decent
##  # container 4
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_cent_1p36_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha1p36_degree2/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_1p36_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha1p36_degree2/decent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_cent_2p10_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha2p10_degree2/cent
##  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_2p10_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha2p10_degree2/decent
#  # container 5
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_cent_3p24_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha3p24_degree2/cent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_3p24_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha3p24_degree2/decent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_cent_5p00_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha5p00_degree2/cent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_5p00_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha5p00_degree2/decent
#done


# 22.05.2022 7 additional decent runs, 25.05.2022 3 additional decent runs
#for (( i=0; i<3; i++ ))
#do
#  # container 11
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_0p100_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p100_degree2/decent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_0p154_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p154_degree2/decent
#  # container 12
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_0p239_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p239_degree2/decent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_0p368_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p368_degree2/decent
#  # container 13
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_0p570_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p570_degree2/decent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_0p879_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha0p879_degree2/decent
#  # container 14
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_1p36_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha1p36_degree2/decent
#  python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_2p10_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha2p10_degree2/decent
  # container 15
  #python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_3p24_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha3p24_degree2/decent
  #python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/20220513_merge65_lanes42/baseline-lane-control_decent_5p00_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220513_merge65_lanes42/alpha5p00_degree2/decent
#done

########################################################################################################################################################

# 12.08.2022

#scl scenario build-all scenarios/custom/intersection/int_4

#python run.py scenarios/custom/intersection/int_4 -f marl_benchmark/agents/ppo/20220812_4_intersection/baseline-lane-control_decent.yaml --num_workers 14 --headless --paradigm decentralized --stop_time 150000 --horizon 100 --log_dir ./log/results/run/20220812_4_intersection/decent

#python run.py scenarios/custom/intersection/int_4 -f marl_benchmark/agents/ppo/20220812_4_intersection/baseline-lane-control_cent.yaml --num_workers 14 --headless --paradigm centralized --stop_time 150000 --horizon 100 --log_dir ./log/results/run/20220812_4_intersection/cent



# 13.08.2022

# 3 runs
#for (( i=0; i<3; i++ ))
#do
#  python run.py scenarios/custom/intersection/int_4 -f marl_benchmark/agents/ppo/20220813_4_intersection/baseline-lane-control_decent.yaml --num_workers 14 --headless --paradigm decentralized --stop_time 150000 --horizon 100 --log_dir ./log/results/run/20220813_4_intersection/decent
 # python run.py scenarios/custom/intersection/int_4 -f marl_benchmark/agents/ppo/20220813_4_intersection/baseline-lane-control_cent.yaml --num_workers 14 --headless --paradigm centralized --stop_time 150000 --horizon 100 --log_dir ./log/results/run/20220813_4_intersection/cent
#done


# 3 runs

#scl scenario build-all scenarios/custom/intersection/int_3

#for (( i=0; i<3; i++ ))
#do
#  python run.py scenarios/custom/intersection/int_3 -f marl_benchmark/agents/ppo/20220814_3_intersection/baseline-lane-control_decent.yaml --num_workers 14 --headless --paradigm decentralized --stop_time 150000 --horizon 110 --log_dir ./log/results/run/20220814_3_intersection/decent
#  python run.py scenarios/custom/intersection/int_3 -f marl_benchmark/agents/ppo/20220814_3_intersection/baseline-lane-control_cent.yaml --num_workers 14 --headless --paradigm centralized --stop_time 150000 --horizon 110 --log_dir ./log/results/run/20220814_3_intersection/cent
#done


# 15.08.2022
#for (( i=0; i<3; i++ ))
#do
#  python run.py scenarios/custom/intersection/int_3 -f marl_benchmark/agents/ppo/20220815_3_intersection/baseline-lane-control_decent.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 150000 --horizon 120 --log_dir ./log/results/run/20220815_3_intersection/decent
#  python run.py scenarios/custom/intersection/int_3 -f marl_benchmark/agents/ppo/20220815_3_intersection/baseline-lane-control_cent.yaml --num_workers 19 --headless --paradigm centralized --stop_time 150000 --horizon 120 --log_dir ./log/results/run/20220815_3_intersection/cent
#done


# 16.08.2022
#scl scenario build-all scenarios/custom/intersection/int_2
#for (( i=0; i<2; i++ ))
#do
#  python run.py scenarios/custom/intersection/int_2 -f marl_benchmark/agents/ppo/20220816_2_intersection/baseline-lane-control_decent.yaml --num_workers 9 --headless --paradigm decentralized --stop_time 150000 --horizon 100 --log_dir ./log/results/run/20220816_2_intersection/decent
#  python run.py scenarios/custom/intersection/int_2 -f marl_benchmark/agents/ppo/20220816_2_intersection/baseline-lane-control_cent.yaml --num_workers 9 --headless --paradigm centralized --stop_time 150000 --horizon 100 --log_dir ./log/results/run/20220816_2_intersection/cent
#done


# 17.08.2022
#scl scenario build-all scenarios/custom/intersection/int_4_rand2
#for (( i=0; i<2; i++ ))
#do
#  python run.py scenarios/custom/intersection/int_4_rand2 -f marl_benchmark/agents/ppo/20220817_4_intersection_rand2/baseline-lane-control_decent.yaml --num_workers 19 --headless --paradigm decentralized --stop_time 150000 --horizon 120 --log_dir ./log/results/run/20220817_4_intersection_rand2/decent
#  python run.py scenarios/custom/intersection/int_4_rand2 -f marl_benchmark/agents/ppo/20220817_4_intersection_rand2/baseline-lane-control_cent.yaml --num_workers 19 --headless --paradigm centralized --stop_time 150000 --horizon 120 --log_dir ./log/results/run/20220817_4_intersection_rand2/cent
#done



# 20.08.2022
scl scenario build-all scenarios/custom/merge/merge_2
for (( i=0; i<3; i++ ))
do
  python run.py scenarios/custom/merge/merge_2 -f marl_benchmark/agents/ppo/20220820_2_merge/baseline-lane-control_decent.yaml --num_workers 9 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220820_2_merge/decent
  python run.py scenarios/custom/merge/merge_2 -f marl_benchmark/agents/ppo/20220820_2_merge/baseline-lane-control_cent.yaml --num_workers 9 --headless --paradigm centralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220820_2_merge/cent
done


scl scenario build-all scenarios/custom/merge/merge_3
for (( i=0; i<3; i++ ))
do
  python run.py scenarios/custom/merge/merge_3 -f marl_benchmark/agents/ppo/20220820_3_merge/baseline-lane-control_decent.yaml --num_workers 9 --headless --paradigm decentralized --stop_time 150000 --horizon 90 --log_dir ./log/results/run/20220820_3_merge/decent
  python run.py scenarios/custom/merge/merge_3 -f marl_benchmark/agents/ppo/20220820_3_merge/baseline-lane-control_cent.yaml --num_workers 9 --headless --paradigm centralized --stop_time 150000 --horizon 90 --log_dir ./log/results/run/20220820_3_merge/cent
done





















