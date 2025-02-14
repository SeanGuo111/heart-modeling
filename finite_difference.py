import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animator
from tqdm import tqdm

# simulation parameters:
dx, dy, dt = 0.5, 0.5, 0.05
x_grid_size, y_grid_size = 100, 100
timesteps = 1000

u_sequence = np.zeros((timesteps, x_grid_size, y_grid_size))
v_sequence = np.zeros((timesteps, x_grid_size, y_grid_size))

# equation parameters:
D = 0.4 #0.1?
k = 8
a = 0.09
epsilon_0 = 0.001 # 
mu_1 = 1 # parameters to fix later
mu_2 = 1 # parameters to fix later

mu_1 = 0.1
mu_2 = 0.1

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

def epsilon(u: np.ndarray, v: np.ndarray):
    return epsilon_0 + ((mu_1*v) / (mu_2 + u))

def reactions(u: np.ndarray, v: np.ndarray):
    F = -(k * u * (u - a) * (u - 1)) - (u * v)
    G = epsilon(u,v) * (-v - (k * u * (u - a - 1)))
    return F, G

def laplacian_standard (u: np.ndarray):
    laplacian = np.zeros_like(u)

    for x_i in range(0, x_grid_size):
        for y_i in range(0, y_grid_size):
            prev_x_i, prev_y_i = determine_diffusion_indices(x_i-1, y_i-1)
            next_x_i, next_y_i = determine_diffusion_indices(x_i+1, y_i+1)
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
    initial_u[50][50] = 1
    initial_u[10][20] = 1

    initial_v = np.zeros_like(initial_v)

    u_sequence[0] = initial_u
    v_sequence[0] = initial_v
    return u_sequence, v_sequence

def compute(u_sequence, v_sequence):
    """Given a u and v sequence primed with initial conditions, fills the rest of the sequence."""
    for t in tqdm(range(0, len(u_sequence) - 1)):
        u_t, v_t = u_sequence[t], v_sequence[t]

        F, G = reactions(u_t, v_t)
        diffusion = D * laplacian_barkley(u_t)

        u_sequence[t+1]= u_t + (dt * (diffusion + F))
        v_sequence[t+1]= v_t + (dt * G)

    return u_sequence, v_sequence

def animation(sol_p, save=False, fol=""): #Animation of just u
    """NOTE: this function was modified from something mentor @ UCSB wrote. (all other functions were written originally)"""
    # Artists
    scrubby = 10 #intervals between displayed frame

    fig, axis = plt.subplots(1, 1, figsize=(5, 5))
    img = axis.imshow(sol_p[0], cmap="jet", interpolation="gaussian")

    def lilly(s):
        probe = s * scrubby
        img.set_data((sol_p[probe]))
        # Add colorbar
        #img.set_title(f"At frame {probe}")

        img.autoscale()
        return img

    anim = animator.FuncAnimation(
        fig, lilly, range(len(sol_p) // scrubby), interval=10 #time period for each frame
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

u_sequence, v_sequence = set_initial_conditions(u_sequence, v_sequence)
# print("u: ", u_sequence[0])
# print("\nv: ", v_sequence[0])
u_sequence, v_sequence = compute(u_sequence, v_sequence)
print(f"U print:\n{u_sequence}")
print(f"V print:\n{v_sequence}")

animation(u_sequence)