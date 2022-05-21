#!/bin/bash

python run.py scenarios/custom/merge/merge65_lanes42 -f marl_benchmark/agents/ppo/baseline-lane-control_test.yaml --num_workers 5 --headless --paradigm decentralized --stop_time 150000 --horizon 80 --log_dir ./log/results/run/20220521_merge65_lanes42_ol_test/alpha1_degree2/decent
