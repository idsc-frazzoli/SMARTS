#!/bin/bash

# This script allows for the queueing of multiple evaluations that can be run in the background. 

num_runs=300
scenario_path="scenarios/custom/merge/merge40_lanes1"


#runs=("log/results/run/20220425_merge40_lanes1_asym_time_test_1_5/alpha1_degree2/cent/PPO_FrameStack_65478_00000_0_2022-04-27_13-10-22" "log/results/run/20220425_merge40_lanes1_asym_time_test_1_5/alpha1_degree2/decent_1/PPO_FrameStack_88cf7_00000_0_2022-04-25_20-23-14" "log/results/run/20220425_merge40_lanes1_asym_time_test_1_5/alpha1_degree2/decent_1/PPO_FrameStack_146fc_00000_0_2022-04-26_10-10-21" "log/results/run/20220425_merge40_lanes1_asym_time_test_1_5/alpha1_degree2/decent_1/PPO_FrameStack_195c1_00000_0_2022-04-25_22-07-30")
#paradigms=("centralized" "decentralized" "decentralized" "decentralized")
#checkpoints=(120 125 115 100)

runs=("log/results/run/20220425_merge40_lanes1_asym_time_test_1_3/alpha1_degree2/cent/PPO_FrameStack_aed17_00000_0_2022-04-27_11-25-03" "log/results/run/20220425_merge40_lanes1_asym_time_test_1_3/alpha1_degree2/decent1/PPO_FrameStack_ffad3_00000_0_2022-04-26_03-50-22" "log/results/run/20220425_merge40_lanes1_asym_time_test_1_3/alpha1_degree2/decent1/PPO_FrameStack_fb070_00000_0_2022-04-26_08-29-25")

paradigms=("centralized" "decentralized" "decentralized")

checkpoints=(170 105 130)

num_steps=60
len=${#runs[@]}
for (( i=0; i<$len; i++ ))
  do
  echo "Doing evaluation for ${runs[$i]} (${paradigms[$i]})."
#  for cp in ${checkpoints[@]}
  cp=${checkpoints[$i]}
  cpf=$(printf "%06d" $cp)
  #cp_dir="./log/results/run/merge40_lanes1-4/${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
  cp_dir="./${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
  log_dir="./evaluation/evaluation_data/${runs[$i]}/checkpoint_${cpf}"
  echo "cp folder: ${cp_dir}"
  python evaluate.py ${scenario_path} -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --log_dir ${log_dir} --checkpoint ${cp_dir} --paradigm ${paradigms[$i]} --headless --num_runs ${num_runs} --num_steps ${num_steps}
done













