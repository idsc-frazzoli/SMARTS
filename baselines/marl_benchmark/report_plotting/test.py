
from pathlib import Path
import pickle



with open(Path("report_plotting", "data.pickle"), 'rb') as handle:
    data = pickle.load(handle)


print("done")