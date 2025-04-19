import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animator
from tqdm import tqdm



def load_data(data_num: int, npy_or_csv: str = "npy", save_file: bool = False):
    path_start = "C:\\Users\\swguo\\VSCode Projects\\Heart Modeling\\Ohio\\first_data_batch"
    data_path = path_start + f"data_{data_num}"

    data = 0
    if npy_or_csv == "npy":
        data = np.load(data_path + ".npy")
    elif npy_or_csv == "csv":
        data = np.genfromtxt(data_path + ".csv", delimiter=",")
        if save_file: np.save(data_path, data)
    else:
        print("Specify txt or csv to load")
        return

    return data

def on_press(event):
    global current_channel
    current_channel += 1
    plt.imshow(data_100[current_channel])

    plt.draw()
    print(plt.xlim())
    print(plt.ylim())

    sys.stdout.flush()


def show_data(data, start_time, end_time):
    print(data.shape)
    fig, axes = plt.subplots(2, 3)
    for i in range(axes.size-1):
        x_i, y_i = i // 3, i % 3
        plt.sca(axes[x_i, y_i])
        plt.plot(data[i,start_time:end_time])
    
    plt.suptitle(f"t = {start_time} to {end_time}")
    plt.show()

save_file = True

data_100 = load_data(100, "npy")
print(f"Maxs, mins: {np.max(data_100, axis=0)}, {np.min(data_100, axis=0)}")
data_100 = data_100[:,:-1]
data_100 = data_100.reshape((5,1024,1024))
current_channel = 0

fig, ax = plt.subplots()
fig.canvas.mpl_connect('key_press_event', on_press)


# plt.xlim(0, data_100.shape[2]-1)
# plt.ylim(0, data_100.shape[2]-1)
plt.imshow(data_100[0], extent=(0, 1023, 0, 1023))
plt.show()



