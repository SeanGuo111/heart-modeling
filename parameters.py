# simulation parameters:
dx, dy, dt = 1, 1, 0.05 #NOTE: dx and dy must be 1 for success. Why?
x_grid_size, y_grid_size = 83, 83 #81 resolution to allow for exact midpoint. +2 to account for neumann conditions on both sides
default_timesteps = 4000

# equation parameters:
D = 0.3 #0.2?
k = 8
a = 0.15
epsilon_0 = 0.002 # 
mu_1 = 1 # parameters to fix later
mu_2 = 1 # parameters to fix later

mu_1 = 0.2
mu_2 = 0.1