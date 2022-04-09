#!/bin/bash

# This script allows for the queueing of multiple evaluations that can be run in the background. 

num_runs=100
scenario_path="scenarios/custom/merge/merge110_lanes2"

#checkpoints=(20 50 100 200 300 340)
checkpoints=(260)
runs=("PPO_FrameStack_d3992_00000_0_2022-04-04_19-02-51")
#checkpoints=(400)
#runs=("PPO_FrameStack_3b54a_00000_0_2022-04-06_12-36-51")
paradigms=("centralized")
len=${#runs[@]}
for (( i=0; i<$len; i++ ))
  do
  echo "Doing evaluation for ${runs[$i]} (${paradigms[$i]})."
  for cp in ${checkpoints[@]}
  do
    cpf=$(printf "%06d" $cp)
    cp_dir="./log/results/run/merge110_lanes2-4/${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
    log_dir="./evaluation/${runs[$i]}/checkpoint_${cpf}"
    echo "cp folder: ${cp_dir}"
    python evaluate.py ${scenario_path} -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --log_dir ${log_dir} --checkpoint ${cp_dir} --paradigm ${paradigms[$i]} --headless --num_runs ${num_runs} --num_steps 150
  done
done

#checkpoints=(20 50 100 200 300 400 500 560)
checkpoints=(260)
runs=("PPO_FrameStack_fada6_00000_0_2022-04-04_08-34-02")
#checkpoints=(400)
#runs=("PPO_FrameStack_7b4bb_00000_0_2022-04-05_15-10-08")

paradigms=("decentralized")
len=${#runs[@]}
for (( i=0; i<$len; i++ ))
  do
  echo "Doing evaluation for ${runs[$i]} (${paradigms[$i]})."
  for cp in ${checkpoints[@]}
  do
    cpf=$(printf "%06d" $cp)
    cp_dir="./log/results/run/merge110_lanes2-4/${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
    log_dir="./evaluation/${runs[$i]}/checkpoint_${cpf}"
    echo "cp folder: ${cp_dir}"
    python evaluate.py ${scenario_path} -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --log_dir ${log_dir} --checkpoint ${cp_dir} --paradigm ${paradigms[$i]} --headless --num_runs ${num_runs} --num_steps 150
  done
done














