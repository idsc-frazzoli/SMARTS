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
for (( i=0; i<20; i++ ))
do
#  python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_decent_1_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220425_merge40_lanes1_asym_time_test_1_5/alpha1_degree2/decent
  python run.py scenarios/custom/merge/merge40_lanes1 -f marl_benchmark/agents/ppo/baseline-lane-control_cent_1_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220425_merge40_lanes1_asym_time_test_1_5/alpha1_degree2/cent
done







