import pandas as pd
import os

from collections import defaultdict

from pathlib import Path
import numpy as np

# import matplotlib as mpl
import matplotlib.pyplot as plt
import json

import re

# with open('./1642442374/2022011717593473.jsonl', 'r') as json_file:
#     json_list = list(json_file)
    # data = json.load(json_file)
    
with open('./1642500065/2022011810010588.jsonl', 'r') as json_file:
    json_list = list(json_file)
#     data = json.load(json_file)
    
    
# a0_pos = []
# a1_pos = []
# pattern = '\"position\": {\"AGENT-[^}]*'
# for i, x in enumerate(json_list[2:]):
#     # print(i+2)
#     try:
#         result = '{' + re.findall(pattern, x)[0] + '}}'
#         d = json.loads(result)
#         a0_pos.append(d['position']['AGENT-0'])
#         a1_pos.append(d['position']['AGENT-1'])
#     except:
#         print('{}'.format(i+2))


# a0_pos_x = [x[0] for x in a0_pos]
# a0_pos_y = [x[1] for x in a0_pos]

# a1_pos_x = [x[0] for x in a1_pos]
# a1_pos_y = [x[1] for x in a1_pos]

# n = 500

# for n in range(1,3000,100):
#     plt.plot(a0_pos_x[n-100:n], a0_pos_y[n-100:n], '.')
#     plt.plot(a1_pos_x[n-100:n], a1_pos_y[n-100:n], '.')
#     plt.title('{}'.format(n))
#     plt.show()




positions = defaultdict()
episode = 0

for i, x in enumerate(json_list[3:]):
    print(i)
    try:
        d = json.loads(x)
    except:
        positions[episode] = defaultdict()
        for key in d['traffic'].keys():
            positions[episode][key] = d['traffic'][key]['driven_path']
        episode += 1
            
# remove empty dicts
positions = {i:j for i,j in positions.items() if len(j.keys()) != 0}
        
#%%


for i in range(len(positions.keys())):
    key0 = list(positions.keys())[i]
    
    traj = defaultdict()
    
    for key in positions[key0].keys():
        traj[key] = defaultdict()
        traj[key]['x'] = []
        traj[key]['y'] = []
        for tra in positions[key0][key]:
            traj[key]['x'].append(tra[0])
            traj[key]['y'].append(tra[1])
    
    for key in traj.keys():
        plt.plot(traj[key]['x'], traj[key]['y'],)
        
        
        
    # plt.plot(a1_pos_x[n-100:n], a1_pos_y[n-100:n], '.')
    # plt.title('{}'.format(n))
    plt.xlim([5,100])
    plt.ylim([0,40])
    plt.show()

