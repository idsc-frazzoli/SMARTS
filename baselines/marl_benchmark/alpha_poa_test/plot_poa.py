import os
from pathlib import Path
import matplotlib.pyplot as plt


from baselines.marl_benchmark.evaluation.utils import get_info, get_poa, get_rewards, load_checkpoint_dfs

alphas = {"0p100": 0.100,
          "0p154": 0.154,
          "0p239": 0.239,
          "0p368": 0.368,
          "0p570": 0.570,
          "0p879": 0.879,
          "1p36": 1.36,
          "2p10": 2.10,
          "3p24": 3.24,
          "5p00": 5.00,
          }

path = "evaluation_test_poa_alpha"

prefixes = ["cent_", "decent_"]

poas = []

for alpha in alphas.keys():
    print(alpha)
    filtered_rewards = []
    for pf in prefixes:
        folder = pf + alpha
        checkpoint_path = Path(path, folder)
        info = get_info(checkpoint_path)
        dfs, masks = load_checkpoint_dfs(checkpoint_path, info)
        _, fr = get_rewards(dfs, masks)
        filtered_rewards.append(fr)
    poa, _, _ = get_poa(filtered_rewards[0],filtered_rewards[1])
    poas.append(poa)

    print(poas)
plt.plot(list(alphas.values()), poas)
plt.show()
plt.savefig('test.png')
