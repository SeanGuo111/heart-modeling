import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animator
from tqdm import tqdm

# simulation parameters:
dx, dy, dt = 1, 1, 0.05 #NOTE: dx and dy must be 1 for success. Why?
x_grid_size, y_grid_size = 83, 83 #81 resolution to allow for exact midpoint. +2 to account for neumann conditions on both sides
timesteps = 4000

u_sequence = np.zeros((timesteps, x_grid_size, y_grid_size))
v_sequence = np.zeros((timesteps, x_grid_size, y_grid_size))

# equation parameters:
D = 0.3 #0.1?
k = 8
a = 0.15
epsilon_0 = 0.002 # 
mu_1 = 1 # parameters to fix later
mu_2 = 1 # parameters to fix later

mu_1 = 0.1
mu_2 = 0.2

def epsilon(u: np.ndarray, v: np.ndarray):
    return epsilon_0 + ((mu_1*v) / (mu_2 + u))

def reactions_aliev_panfilov(u: np.ndarray, v: np.ndarray):
    F = -(k * u * (u - a) * (u - 1)) - (u * v)
    G = epsilon(u,v) * (-v - (k * u * (u - a - 1)))
    return F, G

def determine_diffusion_indices(x,y):
    """Given the x and y indices for a point on the grid, it returns the proper indices
    so as to allow for Neumann boundary conditions. Specifically, any index outside of
    the border is set to the index just inside of the border, so values accessed outside
    the border will be equal to those adjacent.
    """
    x_new, y_new = x, y
    if (x_new == -1):
        x_new = 0
    if (x_new == x_grid_size):
        x_new = x_grid_size - 1
    if (y_new == -1):
        y_new = 0
    if (y_new == y_grid_size):
        y_new = y_grid_size - 1
    return x_new, y_new

def enforce_boundary_conditions(u: np.ndarray):
    for x_index in range(1, x_grid_size-1):
        u[x_index,0] = u[x_index,1]
        u[x_index,y_grid_size-1] = u[x_index,y_grid_size-2]
    for y_index in range(1, y_grid_size-1):
        u[0,y_index] = u[1,y_index]
        u[x_grid_size-1,y_index] = u[x_grid_size-2,y_index]
    
    u[0,0] = u[1,1]
    u[x_grid_size-1,0] = u[x_grid_size-2,1]
    u[0,y_grid_size-1] = u[1,y_grid_size-2]
    u[x_grid_size-1,y_grid_size-1] = u[x_grid_size-2,y_grid_size-2]

    return u

def laplacian_standard (u: np.ndarray):
    laplacian = np.zeros_like(u)

    # boundary conditions already taken care of

    # for x_i in range(0, x_grid_size):
    #     for y_i in range(0, y_grid_size):
    #         prev_x_i, prev_y_i = determine_diffusion_indices(x_i-1, y_i-1)
    #         next_x_i, next_y_i = determine_diffusion_indices(x_i+1, y_i+1)
    #         laplacian[x_i][y_i] = (u[prev_x_i][y_i] + u[next_x_i][y_i] + u[x_i][prev_y_i] + u[x_i][next_y_i] - 4*u[x_i][y_i]) / (dx**2)
    
    for x_i in range(1, x_grid_size-1):
        for y_i in range(1, y_grid_size-1):
            prev_x_i, prev_y_i = x_i - 1, y_i - 1
            next_x_i, next_y_i = x_i + 1, y_i + 1
            laplacian[x_i][y_i] = (u[prev_x_i][y_i] + u[next_x_i][y_i] + u[x_i][prev_y_i] + u[x_i][next_y_i] - 4*u[x_i][y_i]) / (dx**2)

    return laplacian

def laplacian_barkley(u: np.ndarray):
    laplacian = np.zeros_like(u)

    for x_i in range(0, x_grid_size):
        for y_i in range(0, y_grid_size):
            current_u = u[x_i][y_i]
            prev_x_i, prev_y_i = determine_diffusion_indices(x_i-1, y_i-1)
            next_x_i, next_y_i = determine_diffusion_indices(x_i+1, y_i+1)
            # if prev/next x/y index is off the grid, it'll get set to the adjacent on-grid index.
            # So the next statements will add the on-grid value to itself, which is what we want (i.e. current_u will = the laplacian to be assigned to)

            laplacian[prev_x_i][y_i] = laplacian[prev_x_i][y_i] + current_u
            laplacian[next_x_i][y_i] = laplacian[next_x_i][y_i] + current_u
            laplacian[x_i][prev_y_i] = laplacian[x_i][prev_y_i] + current_u
            laplacian[x_i][next_y_i] = laplacian[x_i][next_y_i] + current_u

            laplacian[x_i][y_i] = laplacian[x_i][y_i] - (4 * current_u)

    return laplacian / (dx ** 2)

def set_initial_conditions(u_sequence, v_sequence):
    # u_sequence[0] = np.random.rand(x_grid_size, y_grid_size)
    # v_sequence[0] = np.random.rand(x_grid_size, y_grid_size)
    initial_u = u_sequence[0]
    initial_v = v_sequence[0]

    initial_u = np.zeros_like(initial_u)
    initial_u[41,41] = 1
    initial_u[29,29] = 1
    initial_u[10,10] = 1



    initial_v = np.zeros_like(initial_v)

    print(initial_u)
    print(initial_v)

    u_sequence[0] = initial_u
    v_sequence[0] = initial_v
    return u_sequence, v_sequence

def compute(u_sequence, v_sequence, save: bool = False, u_filename = "", v_filename = ""):
    """Given a u and v sequence primed with initial conditions, fills the rest of the sequence."""
    for t in tqdm(range(0, len(u_sequence) - 1)):
        u_t, v_t = u_sequence[t], v_sequence[t]

        # if t < 50:
        #     v_t[0:50,0:50] = 1
        if t == 1525:
            u_t[30:32, 30:32] = 1

        F, G = reactions_aliev_panfilov(u_t, v_t)

        u_t = enforce_boundary_conditions(u_t)
        v_t = enforce_boundary_conditions(v_t)
        diffusion = D * laplacian_standard(u_t)

        u_sequence[t+1]= u_t + (dt * (diffusion + F))
        #test = u_sequence[t+1]
        v_sequence[t+1]= v_t + dt * G

    if save:
        np.save(u_filename, u_sequence)
        np.save(v_filename, v_sequence)
    
    return u_sequence, v_sequence

def animation(variable_sequence, save=False, fol=""): #Animation of just u
    """NOTE: this function was modified from something mentor @ UCSB wrote. (all other functions were written originally)"""
    # Artists
    time_skip = 10 #intervals between displayed frame

    fig, axis = plt.subplots(1, 1, figsize=(5, 5))
    img = axis.imshow(variable_sequence[0], cmap="Blues", interpolation="none", vmin=0, vmax=1)

    def lilly(t):
        current_time = t * time_skip
        img.set_data((variable_sequence[current_time]))
        plt.title(f"Time = {current_time}")

        img.autoscale()
        return img

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

# u_sequence, v_sequence = set_initial_conditions(u_sequence, v_sequence)
# u_sequence, v_sequence = compute(u_sequence, v_sequence, save=False, u_filename="u_innerspiral_4000t", v_filename="v_innerspiral_4000t")

u_file_name = "u_innerspiral_4000t.npy"
v_file_name = "v_innerspiral_4000t.npy"
u_sequence, v_sequence = np.load(u_file_name), np.load(v_file_name)

# print(f"U print:\n{u_sequence}")
# print(f"V print:\n{v_sequence}")

animation(u_sequence[:,1:x_grid_size-2,1:y_grid_size-2])

# 1, start randomly putting in excitations: eventually -> spiral waves
# 2--go back at look at stimulation of a single cell (1D)
# 3 -- read inverse mechanic paper
