import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animator
from tqdm import tqdm
from math_functions import *
from parameters import *

def show_data(data, start_time, end_time):
    print(data.shape)
    fig, axes = plt.subplots(2, 3)
    for i in range(axes.size-1):
        x_i, y_i = i // 3, i % 3
        plt.sca(axes[x_i, y_i])
        plt.plot(ohio_data[i,start_time:end_time])
    
    plt.suptitle(f"t = {start_time} to {end_time}")
    plt.show()



ohio_data = np.genfromtxt("C:\\Users\\swguo\\VSCode Projects\\Heart Modeling\\Ohio\\data_101.csv", delimiter=",")
print(ohio_data.shape)

show_data(ohio_data, 0, 10000)
show_data(ohio_data, 0, 2500)

