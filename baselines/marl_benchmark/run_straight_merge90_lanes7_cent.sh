#!/bin/bash

scl scenario build-all scenarios/custom/merge/straight_merge90_lanes7
python run.py scenarios/custom/merge/straight_merge90_lanes7 -f marl_benchmark/agents/ppo/20220520_straight_merge90_lanes7/baseline-lane-control_cent_1_2.yaml --num_workers 14 --headless --paradigm centralized --stop_time 150000 --horizon 130 --log_dir ./log/results/run/20220520_straight_merge90_lanes7/alpha1_degree2/cent
