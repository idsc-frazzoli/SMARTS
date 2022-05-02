#!/bin/bash

# This script allows for the queueing of multiple evaluations that can be run in the background. 



#"scenarios/custom/merge/merge75_lanes321"
#"scenarios/custom/merge/merge40_lanes1"
#"scenarios/custom/merge/merge40_lanes1_2"
#"scenarios/custom/merge/merge40_lanes1_3"
#"scenarios/custom/merge/merge40_lanes1_4"


#runs=("log/results/run/20220425_merge40_lanes1_asym_time_test_1_5/alpha1_degree2/cent/PPO_FrameStack_65478_00000_0_2022-04-27_13-10-22" "log/results/run/20220425_merge40_lanes1_asym_time_test_1_5/alpha1_degree2/decent_1/PPO_FrameStack_88cf7_00000_0_2022-04-25_20-23-14" "log/results/run/20220425_merge40_lanes1_asym_time_test_1_5/alpha1_degree2/decent_1/PPO_FrameStack_146fc_00000_0_2022-04-26_10-10-21" "log/results/run/20220425_merge40_lanes1_asym_time_test_1_5/alpha1_degree2/decent_1/PPO_FrameStack_195c1_00000_0_2022-04-25_22-07-30")
#paradigms=("centralized" "decentralized" "decentralized" "decentralized")
#checkpoints=(120 125 115 100)

#runs=("log/results/run/20220429_merge75_lanes321/alpha1_degree2/decent/PPO_FrameStack_8b3ab_00000_0_2022-04-29_15-32-14")
#names=("PPO_FrameStack_8b3ab_00000_0_2022-04-29_15-32-14")
#paradigms=("decentralized")
#checkpoints=(200)
#num_steps=80

#num_runs=300
#scenario_path="scenarios/custom/merge/merge40_lanes1_3"
#runs=("log/results/run/20220428_merge40_lanes1_3_asym_time_test_1_5/alpha1_degree2/decent/PPO_FrameStack_df1ee_00000_0_2022-04-28_16-18-43" "log/results/run/20220428_merge40_lanes1_3_asym_time_test_1_5/alpha1_degree2/decent/PPO_FrameStack_00ee0_00000_0_2022-04-28_23-00-31" "log/results/run/20220428_merge40_lanes1_3_asym_time_test_1_5/alpha1_degree2/cent/PPO_FrameStack_3e452_00000_0_2022-04-28_13-43-53")
#names=("PPO_FrameStack_df1ee_00000_0_2022-04-28_16-18-43" "PPO_FrameStack_00ee0_00000_0_2022-04-28_23-00-31" "PPO_FrameStack_3e452_00000_0_2022-04-28_13-43-53")
#paradigms=("decentralized" "decentralized" "centralized")
#checkpoints=(100 130 200)
#num_steps=60
#
#len=${#runs[@]}
#for (( i=0; i<len; i++ ))
#  do
#  echo "Doing evaluation for ${runs[$i]} (${paradigms[$i]})."
##  for cp in ${checkpoints[@]}
#  cp=${checkpoints[$i]}
#  cpf=$(printf "%06d" "$cp")
#  #cp_dir="./log/results/run/merge40_lanes1-4/${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
#  cp_dir="./${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
#  #log_dir="./evaluation/evaluation_data/${runs[$i]}/checkpoint_${cpf}"
#  log_dir="./${runs[$i]}/../logs_and_plots/evaluation_data/${names[$i]}/checkpoint_${cpf}"
#  echo "cp folder: ${cp_dir}"
#  python evaluate.py ${scenario_path} -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --log_dir ${log_dir} --checkpoint ${cp_dir} --paradigm ${paradigms[$i]} --headless --num_runs ${num_runs} --num_steps ${num_steps}
#done
#
#num_runs=300
#scenario_path="scenarios/custom/merge/merge40_lanes1_2"
#runs=("log/results/run/20220428_merge40_lanes1_2_asym_time_test_1_5/alpha1_degree2/decent/PPO_FrameStack_bd5ea_00000_0_2022-04-28_16-53-33" "log/results/run/20220428_merge40_lanes1_2_asym_time_test_1_5/alpha1_degree2/decent/PPO_FrameStack_dfb49_00000_0_2022-04-28_21-55-10" "log/results/run/20220428_merge40_lanes1_2_asym_time_test_1_5/alpha1_degree2/cent/PPO_FrameStack_0581e_00000_0_2022-04-29_05-34-21")
#names=("PPO_FrameStack_bd5ea_00000_0_2022-04-28_16-53-33" "PPO_FrameStack_dfb49_00000_0_2022-04-28_21-55-10" "PPO_FrameStack_0581e_00000_0_2022-04-29_05-34-21")
#paradigms=("decentralized" "decentralized" "centralized")
#checkpoints=(125 150 170)
#num_steps=60
#
#len=${#runs[@]}
#for (( i=0; i<len; i++ ))
#  do
#  echo "Doing evaluation for ${runs[$i]} (${paradigms[$i]})."
##  for cp in ${checkpoints[@]}
#  cp=${checkpoints[$i]}
#  cpf=$(printf "%06d" "$cp")
#  #cp_dir="./log/results/run/merge40_lanes1-4/${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
#  cp_dir="./${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
#  #log_dir="./evaluation/evaluation_data/${runs[$i]}/checkpoint_${cpf}"
#  log_dir="./${runs[$i]}/../logs_and_plots/evaluation_data/${names[$i]}/checkpoint_${cpf}"
#  echo "cp folder: ${cp_dir}"
#  python evaluate.py ${scenario_path} -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --log_dir ${log_dir} --checkpoint ${cp_dir} --paradigm ${paradigms[$i]} --headless --num_runs ${num_runs} --num_steps ${num_steps}
#done
#
#num_runs=300
#scenario_path="scenarios/custom/merge/merge40_lanes1_4"
#runs=("log/results/run/20220428_merge40_lanes1_4_asym_time_test_1_5/alpha1_degree2/decent/PPO_FrameStack_45f54_00000_0_2022-04-28_16-28-45" "log/results/run/20220428_merge40_lanes1_4_asym_time_test_1_5/alpha1_degree2/cent/PPO_FrameStack_cf27c_00000_0_2022-04-29_06-37-15" "log/results/run/20220428_merge40_lanes1_4_asym_time_test_1_5/alpha1_degree2/cent/PPO_FrameStack_d450a_00000_0_2022-04-28_13-55-14")
#names=("PPO_FrameStack_45f54_00000_0_2022-04-28_16-28-45" "PPO_FrameStack_cf27c_00000_0_2022-04-29_06-37-15" "PPO_FrameStack_d450a_00000_0_2022-04-28_13-55-14")
#paradigms=("decentralized" "centralized" "centralized")
#checkpoints=(170 210 240)
#num_steps=60
#
#len=${#runs[@]}
#for (( i=0; i<len; i++ ))
#  do
#  echo "Doing evaluation for ${runs[$i]} (${paradigms[$i]})."
##  for cp in ${checkpoints[@]}
#  cp=${checkpoints[$i]}
#  cpf=$(printf "%06d" "$cp")
#  #cp_dir="./log/results/run/merge40_lanes1-4/${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
#  cp_dir="./${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
#  #log_dir="./evaluation/evaluation_data/${runs[$i]}/checkpoint_${cpf}"
#  log_dir="./${runs[$i]}/../logs_and_plots/evaluation_data/${names[$i]}/checkpoint_${cpf}"
#  echo "cp folder: ${cp_dir}"
#  python evaluate.py ${scenario_path} -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --log_dir ${log_dir} --checkpoint ${cp_dir} --paradigm ${paradigms[$i]} --headless --num_runs ${num_runs} --num_steps ${num_steps}
#done



num_runs=300
scenario_path="scenarios/custom/merge/merge40_lanes1_3"
runs=("log/results/run/20220428_merge40_lanes1_3_asym_time_test_1_5/alpha1_degree2/decent/PPO_FrameStack_ccd32_00000_0_2022-04-28_12-14-49" "log/results/run/20220428_merge40_lanes1_3_asym_time_test_1_5/alpha1_degree2/cent/PPO_FrameStack_46e94_00000_0_2022-04-29_03-48-48" "log/results/run/20220428_merge40_lanes1_3_asym_time_test_1_5/alpha1_degree2/cent/PPO_FrameStack_334b3_00000_0_2022-04-28_21-14-33" "log/results/run/20220428_merge40_lanes1_3_asym_time_test_1_5/alpha1_degree2/cent/PPO_FrameStack_24978_00000_0_2022-04-29_00-48-53" "log/results/run/20220428_merge40_lanes1_3_asym_time_test_1_5/alpha1_degree2/cent/PPO_FrameStack_ede3a_00000_0_2022-04-29_06-38-07")
names=("PPO_FrameStack_ccd32_00000_0_2022-04-28_12-14-49" "PPO_FrameStack_46e94_00000_0_2022-04-29_03-48-48" "PPO_FrameStack_334b3_00000_0_2022-04-28_21-14-33" "PPO_FrameStack_24978_00000_0_2022-04-29_00-48-53" "PPO_FrameStack_ede3a_00000_0_2022-04-29_06-38-07")
paradigms=("decentralized" "centralized" "centralized" "centralized" "centralized")
checkpoints=(105 160 170 190 200)
num_steps=60

len=${#runs[@]}
for (( i=0; i<len; i++ ))
  do
  echo "Doing evaluation for ${runs[$i]} (${paradigms[$i]})."
#  for cp in ${checkpoints[@]}
  cp=${checkpoints[$i]}
  cpf=$(printf "%06d" "$cp")
  #cp_dir="./log/results/run/merge40_lanes1-4/${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
  cp_dir="./${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
  #log_dir="./evaluation/evaluation_data/${runs[$i]}/checkpoint_${cpf}"
  log_dir="./${runs[$i]}/../logs_and_plots/evaluation_data/${names[$i]}/checkpoint_${cpf}"
  echo "cp folder: ${cp_dir}"
  python evaluate.py ${scenario_path} -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --log_dir ${log_dir} --checkpoint ${cp_dir} --paradigm ${paradigms[$i]} --headless --num_runs ${num_runs} --num_steps ${num_steps}
done




