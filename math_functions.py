from parameters import *
import numpy as np

def epsilon(u: np.ndarray, v: np.ndarray):
    return epsilon_0 + ((mu_1*v) / (mu_2 + u))

def reactions_aliev_panfilov(u: np.ndarray, v: np.ndarray):
    F = -(k * u * (u - a) * (u - 1)) - (u * v)
    G = epsilon(u,v) * (-v - (k * u * (u - a - 1)))
    return F, G

def active_stress(u: np.ndarray, T_a: np.ndarray):
    epsilon_u = np.where(u < 0.05, 10, 1)
    return epsilon_u * ((k_T * u) - T_a)

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

def laplacian_8_ways(u: np.ndarray):
    laplacian = np.zeros_like(u)
    # boundary conditions already taken care of

    for x_i in range(1, x_grid_size-1):
        for y_i in range(1, y_grid_size-1):
            prev_x_i, prev_y_i = x_i - 1, y_i - 1
            next_x_i, next_y_i = x_i + 1, y_i + 1
            laplacian_current = (u[prev_x_i][y_i] + u[next_x_i][y_i] + u[x_i][prev_y_i] + u[x_i][next_y_i])
            laplacian_current = laplacian_current + (u[prev_x_i][prev_y_i] + u[prev_x_i][next_y_i] + u[next_x_i][prev_y_i] + u[next_x_i][next_y_i])
            laplacian_current = (laplacian_current - 8*u[x_i][y_i]) / (dx**2)

            laplacian[x_i][y_i] = laplacian_current
            

    return laplacian