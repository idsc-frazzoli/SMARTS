#!/bin/bash

# This script allows for the queueing of multiple evaluations that can be run in the background. 

num_runs=10
#scenario_path="scenarios/custom/merge/merge110_lanes2"
scenario_path="scenarios/custom/merge/merge40_lanes1"

#checkpoints=(20 50 100 200 300 340)
#runs=("PPO_FrameStack_d3992_00000_0_2022-04-04_19-02-51")
#checkpoints=(20 50 100 250)
#checkpoints=(50)
#checkpoints=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 270 271)


#runs=("PPO_FrameStack_40abf_00000_0_2022-04-12_06-10-29" "PPO_FrameStack_d7035_00000_0_2022-04-12_08-23-32")
#runs=("20220412/PPO_FrameStack_3d116_00000_0_2022-04-12_15-35-53" "20220412/PPO_FrameStack_9e9de_00000_0_2022-04-12_13-08-17" "20220412/PPO_FrameStack_72221_00000_0_2022-04-12_21-13-49" "20220412/PPO_FrameStack_a3747_00000_0_2022-04-12_18-37-42" "20220412/PPO_FrameStack_d9b8f_00000_0_2022-04-12_17-13-19")
#paradigms=("centralized" "centralized" "centralized" "centralized" "centralized")

#runs=("20220412/PPO_FrameStack_69ef9_00000_0_2022-04-12_20-02-00")
#paradigms=("centralized")

#runs=("20220413/PPO_FrameStack_382d4_00000_0_2022-04-13_15-56-02" "20220413/PPO_FrameStack_486b1_00000_0_2022-04-13_10-41-32")
#paradigms=("decentralized" "decentralized")

#runs=("PPO_FrameStack_a6b8a_00000_0_2022-04-13_19-05-15")
#paradigms=("centralized")

runs=("alpha1_degree2/decent/PPO_FrameStack_febfa_00000_0_2022-04-23_10-41-55")
paradigms=("decentralized")



checkpoints=(80)

num_steps=60
len=${#runs[@]}
for (( i=0; i<$len; i++ ))
  do
  echo "Doing evaluation for ${runs[$i]} (${paradigms[$i]})."
  for cp in ${checkpoints[@]}
  do
    cpf=$(printf "%06d" $cp)
    #cp_dir="./log/results/run/merge40_lanes1-4/${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
    cp_dir="./log/results/run/merge40_lanes1_asym_time_test/${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
    log_dir="./evaluation/evaluation_data/${runs[$i]}/checkpoint_${cpf}"
    echo "cp folder: ${cp_dir}"
    python evaluate.py ${scenario_path} -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --log_dir ${log_dir} --checkpoint ${cp_dir} --paradigm ${paradigms[$i]} --headless --num_runs ${num_runs} --num_steps ${num_steps}
  done
done
#
##checkpoints=(20 50 100 200 300 400 500 560)
##checkpoints=(260)
##runs=("PPO_FrameStack_fada6_00000_0_2022-04-04_08-34-02")
#checkpoints=(20 40 100 200 500 680)
#runs=("PPO_FrameStack_7b4bb_00000_0_2022-04-05_15-10-08")
#
#paradigms=("decentralized")
#len=${#runs[@]}
#for (( i=0; i<$len; i++ ))
#  do
#  echo "Doing evaluation for ${runs[$i]} (${paradigms[$i]})."
#  for cp in ${checkpoints[@]}
#  do
#    cpf=$(printf "%06d" $cp)
#    cp_dir="./log/results/run/merge110_lanes2-4/${runs[$i]}/checkpoint_${cpf}/checkpoint-${cp}"
#    log_dir="./evaluation/${runs[$i]}/checkpoint_${cpf}"
#    echo "cp folder: ${cp_dir}"
#    python evaluate.py ${scenario_path} -f marl_benchmark/agents/ppo/baseline-lane-control.yaml --log_dir ${log_dir} --checkpoint ${cp_dir} --paradigm ${paradigms[$i]} --headless --num_runs ${num_runs} --num_steps 150
#  done
#done














