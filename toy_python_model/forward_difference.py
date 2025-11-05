import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animator
from tqdm import tqdm
from math_functions import *
from parameters import *
from numba import jit

def initial_conditions(u_sequence, v_sequence, type: str):
    # u_sequence[0] = np.random.rand(x_grid_size, y_grid_size)
    # v_sequence[0] = np.random.rand(x_grid_size, y_grid_size)

    if type == "center":
        u_sequence[0] = np.zeros_like(u_sequence[0])
        v_sequence[0] = np.zeros_like(v_sequence[0])
        u_sequence[0, x_grid_size // 2, y_grid_size // 2] = 1
    elif type == "random":
        u_sequence[0] = np.random.rand(u_sequence.shape[0], u_sequence.shape[1])
        v_sequence[0] = np.random.rand(v_sequence.shape[0], v_sequence.shape[1])
    else:
        u_sequence[0] = np.zeros_like(u_sequence[0])
        v_sequence[0] = np.zeros_like(v_sequence[0])

    return u_sequence, v_sequence

def produce_random_waves(u, wave_count, wave_thickness = 1):
    for i in range(wave_count):
        x_margin, y_margin = x_grid_size // 10, y_grid_size // 10
        x_i, y_i = np.random.randint(x_margin, x_grid_size-x_margin), np.random.randint(y_margin, y_grid_size-y_margin)
        u[x_i: x_i+wave_thickness, y_i: y_i+wave_thickness] = 1
    
    return u

def compute(timesteps = default_timesteps, last_disturbance_time = 1500, disturbance_count = 0, max_wave_count = 1, wave_thickness = 1, 
            mechanical_coupling = False, initial_condition_type: str = "center", laplacian_method = laplacian_standard, 
            save: bool = False, u_filename = "", v_filename = ""):
    """Returns a u and v sequence primed with a given initial condition function (which takes u_seq and v_seq and returns u_seq, v_seq)
    and calculated using a given laplacian method."""
    u_sequence = np.zeros((timesteps, x_grid_size, y_grid_size))
    v_sequence = np.zeros((timesteps, x_grid_size, y_grid_size))
    T_a_sequence = np.zeros((timesteps, x_grid_size, y_grid_size))

    u_sequence, v_sequence = initial_conditions(u_sequence, v_sequence, initial_condition_type)

    # disturbances_times = np.random.randint(0, timesteps, size=disturbance_count)
    # disturbances_dict = {x: np.random.randint(1, max_wave_count) for x in disturbances_times}
    
    disturbances_array = np.zeros(timesteps, dtype=int)
    counter = 1
    while counter <= disturbance_count:
        index = np.random.randint(0,last_disturbance_time+1)
        if disturbances_array[index] != 0:
            continue
        disturbances_array[index] = np.random.randint(1, max_wave_count+1)
        counter += 1
    nonzero_indices = np.nonzero(disturbances_array)[0]
    disturbances_dict = {index : disturbances_array[index] for index in nonzero_indices}
    print(disturbances_dict)


    for t in tqdm(range(0, len(u_sequence) - 1)):
        u_t, v_t = u_sequence[t], v_sequence[t]

        if disturbances_array[t]:
            u_t = produce_random_waves(u_t, disturbances_array[t], wave_thickness)

        F, G = reactions_aliev_panfilov(u_t, v_t)

        #u_t = enforce_boundary_conditions(u_t)
        #v_t = enforce_boundary_conditions(v_t) not needed: no diffusion for v.
        diffusion = D * laplacian_method(u_t)

        u_sequence[t+1] = u_t + (dt * (diffusion + F))
        test = u_sequence[t+1]
        v_sequence[t+1] = v_t + dt * G

        if mechanical_coupling:
            T_a_t = T_a_sequence[t]
            T_a_sequence[t+1] = T_a_t + dt * active_stress(u_t, T_a_t)


    if save:
        np.save(u_filename, u_sequence)
        np.save(v_filename, v_sequence)            
    
    return_value = [u_sequence, v_sequence, disturbances_dict]
    if mechanical_coupling:
        return_value.append(T_a_sequence)
    return return_value

def generate_random_sequences(sequence_count: int = 10, timesteps = 3000, last_disturbance_time = 1500, disturbance_count=5, max_wave_count=2, wave_thickness=1,
                                initial_condition_type: str ="center", laplacian_method=laplacian_standard):
    
    u_sequences = np.zeros((sequence_count, timesteps, x_grid_size, y_grid_size))
    v_sequences = np.zeros((sequence_count, timesteps, x_grid_size, y_grid_size))
    disturbances_dict_list = []

    for i in range(sequence_count):
        print(f"\nSimulation {i+1}.")
        u_sequences[i], v_sequences[i], disturbances_dict = compute(timesteps = timesteps, last_disturbance_time = last_disturbance_time, disturbance_count = disturbance_count, max_wave_count = max_wave_count, wave_thickness = wave_thickness,
                                                                    initial_condition_type = initial_condition_type, laplacian_method = laplacian_method)
        disturbances_dict_list.append(disturbances_dict)
    
    return u_sequences, v_sequences, disturbances_dict_list


def animation(variable_sequence, disturbances_dict, save=False): #Animation of just u
    """NOTE: this function was modified from code mentor @ UCSB wrote. (all other functions were written originally)"""
    variable_sequence = variable_sequence[:,1:x_grid_size-2,1:y_grid_size-2] #Filter out borders, which are affected by Neumann conditions
    #v_sequence = v_sequence[:,1:x_grid_size-2,1:y_grid_size-2] #Filter out borders, which are affected by Neumann conditions
    time_skip = 10 #intervals between displayed frame

    fig, ax_1 = plt.subplots(1, 1, figsize=(5, 5))
    img = ax_1.imshow(variable_sequence[0], cmap="Blues", interpolation="none")
    #img_v = ax_2.imshow(v_sequence[0], cmap="Greys", interpolation="none", vmin=0, vmax=1)

    def lilly(t):
        current_time = t * time_skip
        img.set_data((variable_sequence[current_time]))
        #img_v.set_data((v_sequence[current_time]))
        plt.title(f"Time = {current_time}\nDisturbances: {disturbances_dict}")

        img.autoscale()
        #img_v.autoscale()
        return img#, img_v

    anim = animator.FuncAnimation(
        fig, lilly, range(len(variable_sequence) // time_skip), interval=10 #display duration for each frame in milliseconds
    )
    plt.show()
    # if(save):
    #     print(f"Saving to {fol}/RGB_{nowString}.mp4")
    #     anim.save(f"{fol}/animation_{nowString}.mp4", writer=animator.FFMpegWriter())
    # ANIMATION ENDS HERE

#COMPUTATION ------------------------------------------------------

# Diffusion test: successful
# u = np.random.rand(x_grid_size, y_grid_size)
# l1 = laplacian_standard(u)
# print(l1)
# l2 = laplacian_barkley(u)
# print(l2)

# u_sequence, v_sequence, disturbances_dict, T_a_sequence = compute(timesteps = 3000, disturbance_count=0, max_wave_count=2, wave_thickness=1,
#                                 mechanical_coupling=True, initial_condition_type="center", laplacian_method=laplacian_standard, save=False, 
#                                 u_filename="u_innerspiral_4000t", v_filename="v_innerspiral_4000t")
# animation(T_a_sequence, disturbances_dict)

# u_file_name = "u_innerspiral_4000t.npy"
# v_file_name = "v_innerspiral_4000t.npy"
# u_sequence, v_sequence = np.load(u_file_name), np.load(v_file_name)


#GENERATE RANDOM SEQUENCES ------------------------------------------------------

sequence_count = 2
u_sequences, v_sequences, disturbances_dict_list = generate_random_sequences(sequence_count=sequence_count, initial_condition_type="zero",
                                                                             timesteps=5000, last_disturbance_time=1500, disturbance_count=10, max_wave_count=5, wave_thickness=3)
for i in range(len(u_sequences)):
    animation(u_sequences[i], v_sequences[i], disturbances_dict_list[i])

# Go back at look at stimulation of a single cell (1D) -> 2D
# Try action potential model for single cell

# video of each ohio data frame