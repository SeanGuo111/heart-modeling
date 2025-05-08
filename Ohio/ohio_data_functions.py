import sys
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animator
import optimap as om
import monochrome as mc
from numba import jit
from boxsdk import DevelopmentClient
from tqdm import tqdm
# import cv2; installing opencv may cause matplotlib issues with qt; remove opencv to fix this

def load_individual_dataset(data_id: int, npy_or_csv: str = "npy", 
                            data_path = "C:\\Users\\swguo\\VSCode Projects\\Heart Modeling\\Ohio\\first_data_batch\\",
                            save_path = "C:\\Users\\swguo\\VSCode Projects\\Heart Modeling\\Ohio\\first_data_batch\\",
                            save_file: bool = False):
    data_path_full = data_path + f"data_{data_id}"

    length = 1024*1024
    data = None

    if npy_or_csv == "npy":
        data = np.load(data_path_full + ".npy")
        
    elif npy_or_csv == "csv":
        data = np.loadtxt(data_path_full + ".csv", delimiter=",", usecols=range(length)) 
        #usecols=range(length) ensures np ignores last element of each row, which it was interpreting as nan in genfromtxt
        #loadtxt wayyyy faster than genfromtxt (0.5s vs. ~6-7s)
        data = data.reshape((5,1024,1024))
        if save_file: np.save(save_path, data)
    else:
        print("Specify txt or csv to load")
        return

    return data

def download_npys_from_disk(start_data_id, end_data_id, channels=[0,1,2,3,4],
                            data_path = "/mnt/data2/sean/data/dataset01/",
                            save_path = "/mnt/data2/sean/data/dataset01/",
                            save_file: bool = False):
    """
    Saves box csvs with given ids and specified channels as .npy files, each with shape (channels, dim, dim)
    Save_file false by default as a safety
    """
    dim = 1024
    channel_count = len(channels)
    channel_str = "".join(map(str, channels))
    
    print(f"Downloading csv files {start_data_id}-{end_data_id}")
    for id in tqdm(range(start_data_id, end_data_id+1)):
        data_path_full = data_path + f"data_{id}.csv"
        data = np.loadtxt(data_path_full, delimiter=",", usecols=range(dim*dim))

        channeled = data[channels][:] 
        reshaped = channeled.reshape((channel_count,dim,dim))
        file_save_name = save_path + f"npydata_c{channel_str}_{id}.npy"
        if save_file: np.save(file_save_name, reshaped)
    

def download_npys_from_box(start_data_id, end_data_id, channels=[0,1,2,3,4], 
                           save_path = "C:\\Users\\swguo\\VSCode Projects\\Heart Modeling\\Ohio\\second_data_batch\\", save_file: bool = False):
    """
    Saves box csvs with given ids and specified channels as .npy files, each with shape (channels, dim, dim)
    Save_file false by default as a safety
    """

    # Developer token: V2wREhqwIsFEueXWhqI4CTkcfvyWZ49X
    client = DevelopmentClient()
    # me = client.user(user_id='me').get()
    # print(f"User: {me.name}, Email: {me.login}")
    root_folder = client.folder(folder_id='292696197350')  # '0' is the ID for the root folder
    all_files = root_folder.get_items()
    
    id_list = list(range(start_data_id, end_data_id+1))
    file_name_list = [f"data_{id}.csv" for id in id_list]

    dim = 1024
    channel_count = len(channels)
    channel_str = "".join(map(str, channels))
    counter = 1

    for file_box in all_files:
        file_name = file_box.name
        if file_name in file_name_list:
            print(f"\nFile {counter}: {file_name}")
            file_bytes = file_box.content()
            #print(f"File type 1: {type(file_bytes)}")
            
            file = io.BytesIO(file_bytes).readlines()
            #print(f"File type 2: {type(file)}")

            data = np.loadtxt(file, delimiter=",", usecols=range(dim*dim))
            channeled = data[channels][:] 
            reshaped = channeled.reshape((channel_count,dim,dim))
            file_save_name = save_path + f"npydata_c{channel_str}_{file_name[5:-4]}.npy"
            if save_file: np.save(file_save_name, reshaped)

            counter += 1


def load_from_npy(start_data_id, end_data_id, filename_channels=[0,1,2,3,4], subset_channels=None, 
                  data_path = "C:\\Users\\swguo\\VSCode Projects\\Heart Modeling\\Ohio\\second_data_batch\\",
                  save_path = "C:\\Users\\swguo\\VSCode Projects\\Heart Modeling\\Ohio\\second_data_batch\\",
                  save_subsets=False):
    """Returns single ndarray with shape (channels, time, dim, dim), from filenames with certain channels and an optional
    subset of those filename channels that are actually selected to be in the loaded dataset. 
    Subset defaults to being equal to filename_channels.
    Dimensions (dim1, dim2) are 1. rows, top->bottom and then 2. columns, left->right, like matrix indexing. (Not (x,y).)"""

    if subset_channels is None or subset_channels == []:
        subset_channels = filename_channels
    elif len(subset_channels) > len(filename_channels):
        print("Subset channels should be a subset of the filename channels")
        return

    channel_str = "".join(map(str, filename_channels))
    subset_str = "".join(map(str, subset_channels))
    
    dim=1024
    channel_count = len(subset_channels)
    time_axis_len = end_data_id - start_data_id + 1
    dataset = np.zeros((time_axis_len, channel_count, dim, dim))
    
    index = 0
    for id in tqdm(range(start_data_id, end_data_id+1)):  
        data_path_full = data_path + f"npydata_c{channel_str}_{id}.npy"    
        raw = np.load(data_path_full)
        channeled = raw[subset_channels][:] 
        dataset[index] = channeled

        if save_subsets:
            file_save_name = save_path + f"npydata_c{subset_str}_{id}.npy"
            np.save(file_save_name, channeled)

        index += 1
        
    dataset = np.swapaxes(dataset, 0, 1)
    return dataset

def on_press(event):
    global current_time
    global dataset_c0
    current_time += 1
    plt.imshow(dataset_c0[current_time])
    plt.title(f"T = {current_time}")

    plt.draw()

    sys.stdout.flush()

def slideshow(channel_dataset):
    global current_time
    current_time = 0
    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_press)

    plt.imshow(channel_dataset[0], extent=(0, 1023, 0, 1023))
    plt.title("T = 0")
    plt.show()

def animation(channel_dataset, time_skip = 1, interval = 10, save=False): #Animation of just u
    """NOTE: this function was modified from code mentor @ UCSB wrote. (all other functions were written originally)"""

    fig, ax_1 = plt.subplots(1, 1, figsize=(5, 5))
    img = ax_1.imshow(channel_dataset[0], cmap="Blues", interpolation="none")
    #img_v = ax_2.imshow(v_sequence[0], cmap="Greys", interpolation="none", vmin=0, vmax=1)

    def lilly(t):
        current_time = t * time_skip
        img.set_data((channel_dataset[current_time]))
        #img_v.set_data((v_sequence[current_time]))
        plt.title(f"Time = {current_time}")

        img.autoscale()
        #img_v.autoscale()
        return img#, img_v

    anim = animator.FuncAnimation(
        fig, lilly, range(len(channel_dataset) // time_skip), interval=interval #display duration for each frame in milliseconds
    )
    plt.show()

def chunk(dataset, lower_dim, flatten=True):
    """Takes a single channel dataset with shape (time, dim, dim), with dim = 1024, 
    and returns smaller-dimensional samples from the center grid cells as an ndarray with one of two shapes:
        flatten = False: (row, column, time, lower_dim, lower_dim).
        flatten = True: (row*column, time, lower_dim, lower_dim) 
    Lower dim should be a factor of 1000."""

    og_dim = 1024
    num_subarrays = og_dim // lower_dim
    margin = (og_dim - (num_subarrays * lower_dim)) // 2
    cropped = dataset[:, margin:og_dim-margin, margin:og_dim-margin]

    col_split = np.array(np.split(cropped, num_subarrays, 2))
    col_row_split = np.array(np.split(col_split, num_subarrays, 2))
    if not flatten: 
        return col_row_split
    time_axis = col_row_split.shape[2]
    return np.reshape(col_row_split, (num_subarrays ** 2, time_axis, lower_dim, lower_dim))

def rotate(dataset, flatten=True):
    """Takes a flattened dataset with shape (samples, time, dim, dim) and returns rotated samples with one of two shapes:
            flatten = False: (samples, 4, time, dim, dim).
            flatten = True: (samples*4, time, dim, dim) 
    """
    rotate_once = lambda ndarray : np.array([om.video.rotate_left(video) for video in ndarray])
    rotated_1 = rotate_once(dataset)
    rotated_2 = rotate_once(rotated_1)
    rotated_3 = rotate_once(rotated_2)
    dataset = np.array([dataset, rotated_1, rotated_2, rotated_3])
    dataset = np.swapaxes(dataset, 0, 1)
    if flatten:
        samples, num_rotations, time, lower_dim, lower_dim_2 = dataset.shape
        return np.reshape(dataset, (4*samples, time, lower_dim, lower_dim))
    return dataset

def splice(dataset: np.ndarray, frames=5, flatten=True):
    """Takes a flattened dataset with shape (samples, time, dim, dim) and returns spliced samples with one of two shapes:
            flatten = False: (samples, time//frames, frames, dim, dim)
            flatten = True: (samples*(time//frames), frames, dim, dim) 
    """
    time = dataset.shape[1]
    split_count, remainder = time // frames, time % frames
    cropped = dataset[:,:time-remainder,:,:]
    cropped_spliced = np.array(np.split(cropped, split_count, 1))
    if not flatten:
        return np.swapaxes(cropped_spliced, 0, 1)
    
    variations, dim = dataset.shape[0], dataset.shape[-1]
    return np.reshape(dataset, (variations*split_count, frames, dim, dim))