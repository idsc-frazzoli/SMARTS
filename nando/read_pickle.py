import pandas as pd
import os


scenario = 'cross-4'
name = 'PPO_FrameStack_9c0e9_00000_0_2021-12-09_00-21-45'
checkpoint_nr = 10

pickle_path = os.path.join('baselines', 'marl_benchmark', 'log', 'results', 'run',
                           scenario,
                           name,
                           'checkpoint_' + str(checkpoint_nr),
                           'checkpoint_' + str(checkpoint_nr)
                           )

df_checkpoint = pd.read_pickle(pickle_path)

