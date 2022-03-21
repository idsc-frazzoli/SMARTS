import argparse
import pathlib
import time
import warnings
from typing import Dict, List, Any, Callable, Tuple

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting.custom_evaluations import get_custom_evaluation
from plotting.utils import get_files, parse_env_task, parse_reward


FIGSIZE = (15, 4)
LARGESIZE, MEDIUMSIZE, SMALLSIZE = 16, 13, 10

plt.rcParams.update({'font.size': LARGESIZE})
plt.rcParams.update({'axes.titlesize': LARGESIZE})
plt.rcParams.update({'axes.labelsize': MEDIUMSIZE})
plt.rcParams.update({'xtick.labelsize': SMALLSIZE})
plt.rcParams.update({'ytick.labelsize': SMALLSIZE})
plt.rcParams.update({'legend.fontsize': MEDIUMSIZE})
plt.rcParams.update({'figure.titlesize': LARGESIZE})

COLORS = {
    'default': '#377eb8',
    'tltl': '#4daf4a',
    'bhnr': '#984ea3',
    'morl_uni': '#a65628',
    'morl_dec': '#ff7f00',
    'hrs_pot': '#e41a1c'
}






