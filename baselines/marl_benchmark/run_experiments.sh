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
python run.py scenarios/custom/merge110_lanes2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 19 --headless --paradigm centralized --stop_time 72000 --horizon 150












