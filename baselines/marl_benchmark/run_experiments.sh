#!/bin/bash

python run.py scenarios/double_merge/merge_asym2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm decentralized --stop_time 4000

python run.py scenarios/double_merge/merge_asym2 -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --num_workers 30 --headless --paradigm centralized --stop_time 4000

