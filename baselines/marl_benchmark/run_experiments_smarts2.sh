# 05.05.2022

## 5 runs decent and cent with merge40_lanes1_3 asymmetric cost 07, 13k episodes per run, alpha = 2, degree = 2
#scl scenario build-all scenarios/custom/merge/merge40_lanes1_3
#for (( i=0; i<5; i++ ))
#do
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_decent_2_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha2_degree2/decent
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_cent_2_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha2_degree2/cent
#done

## 5 runs decent and cent with merge40_lanes1_3 asymmetric cost 07, 13k episodes per run, alpha = 0.1, degree = 2
#for (( i=0; i<5; i++ ))
#do
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_decent_01_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha01_degree2/decent
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_cent_01_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha01_degree2/cent
#done

## 5 runs decent and cent with merge40_lanes1_3 asymmetric cost 07, 13k episodes per run, alpha = 10, degree = 2
#for (( i=0; i<5; i++ ))
#do
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_decent_10_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha10_degree2/decent
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_cent_10_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha10_degree2/cent
#done

## 5 runs decent and cent with merge40_lanes1_3 asymmetric cost 07, 13k episodes per run, alpha = 5, degree = 2
#for (( i=0; i<5; i++ ))
#do
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_decent_5_2.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha5_degree2/decent
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_cent_5_2.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha5_degree2/cent
#done

## 5 runs decent and cent with merge40_lanes1_3 asymmetric cost 07, 13k episodes per run, alpha = 1, degree = 1
#for (( i=0; i<5; i++ ))
#do
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_decent_1_1.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha1_degree1/decent
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_cent_1_1.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha1_degree1/cent
#done
#
## 5 runs decent and cent with merge40_lanes1_3 asymmetric cost 07, 13k episodes per run, alpha = 1, degree = 3
#for (( i=0; i<5; i++ ))
#do
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_decent_1_3.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha1_degree3/decent
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_cent_1_3.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha1_degree3/cent
#done
#
## 5 runs decent and cent with merge40_lanes1_3 asymmetric cost 07, 13k episodes per run, alpha = 1, degree = 4
#for (( i=0; i<5; i++ ))
#do
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_decent_1_4.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha1_degree4/decent
#  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_cent_1_4.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha1_degree4/cent
#done
#
# 5 runs decent and cent with merge40_lanes1_3 asymmetric cost 07, 13k episodes per run, alpha = 1, degree = 5
for (( i=0; i<5; i++ ))
do
  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_decent_1_5.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha1_degree5/decent
  python run.py scenarios/custom/merge/merge40_lanes1_3 -f marl_benchmark/agents/ppo/merge40_lanes1_3_alpha/baseline-lane-control_cent_1_5.yaml --num_workers 5 --headless --paradigm centralized --stop_time 150000 --horizon 60 --log_dir ./log/results/run/20220505_merge40_lanes1_3/alpha1_degree5/cent
done


