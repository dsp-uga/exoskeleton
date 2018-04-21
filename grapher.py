import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sys

if len(sys.argv) < 3:
    print("Usage: python grapher.py <datapath> <index>")

data = pd.read_csv(sys.argv[1])
arr = data.as_matrix()
plt.plot(arr[:, int(sys.argv[2])])
plt.show()
