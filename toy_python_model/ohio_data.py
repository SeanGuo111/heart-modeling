import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animator
from tqdm import tqdm
from math_functions import *
from parameters import *

ohio_data = np.genfromtxt("data_100.csv", delimiter=",")
print(ohio_data.shape)
fig, axes = plt.subplots(2, 3)
for i in range(axes.size-1):
    x_i, y_i = i // 3, i % 3
    plt.sca(axes[x_i, y_i])
    plt.plot(ohio_data[i,0:10000])


plt.show()