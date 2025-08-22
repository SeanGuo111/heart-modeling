#%%

import numpy as np
import matplotlib.pyplot as plt
import optimap as om
from tqdm import tqdm


# parameters
nt = 1000         # total number of integration steps
dt = 0.001         # integration time step
eps = 0.08          # recovery rate
a = 0.5             # excitation threshold
b = 0.8             # recovery sensitivity
I_ext = 0.0         # excitable
D = 0.001             # diffusion coefficient

nx = 21
ny = 21

V = np.zeros((nt,nx,ny))
R = np.zeros((nt,nx,ny))

lap = np.zeros((nx,ny,2))

V[0,:,:] = -1.3  
R[0,:,:] = -1.0 

# Integration using Euler's method
for t in tqdm(range(1, nt)):

    lap[:,:,0] = V[t-1,:,:]

    for x in range(1,nx-1):
        for y in range(1,ny-1):
            lap[x,y,1] = lap[x+1,y,0] + lap[x-1,y,0] + lap[x,y+1,0] + lap[x,y-1,0] - 4.0 * lap[x,y,0]

    for x in range(1,nx-1):
        for y in range(1,ny-1):
    
            v = V[t-1,x,y]
            r = R[t-1,x,y]

            if(t==50000):
                V[t-1,5,5] = 2
                V[t-1,4,5] = 2
                V[t-1,5,4] = 2
                V[t-1,4,4] = 2
                #V[t-1,1,1] = -0.2
    
            # FitzHugh-Nagumo model equations
            dv = (v - (v**3) / 3.0 - r + I_ext) * dt
            dr = eps * (v + a - b * r) * dt

            V[t,x,y] = V[t-1,x,y] + dv + lap[x,y,1]*D
            R[t,x,y] = R[t-1,x,y] + dr
    
    # boundaries
    for x in range(1,nx-1):
        V[t,x,0] = V[t,x,1]
        V[t,x,ny-1] = V[t,x,ny-2]

    for y in range(1,ny-1):
        V[t,0,y] = V[t,1,y]
        V[t,nx-1,y] = V[t,nx-2,y]

    V[t,0,0] = V[t,1,1]
    V[t,nx-1,0] = V[t,nx-2,1]
    V[t,0,ny-1] = V[t,1,ny-2]
    V[t,nx-1,ny-1] = V[t,nx-2,ny-2]


    

#%%

om.show_video(V, skip_frame=500,vmin=-2, vmax=2, cmap='gray')




#%%

# minimal code


for t in range(1, nt):

    lap[:,:,0] = V[t-1,:,:]

    for x in range(1,nx-1):
        for y in range(1,ny-1):
            lap[x,y,1] = lap[x+1,y,0] + lap[x-1,y,0] + lap[x,y+1,0] + lap[x,y-1,0] - 4.0 * lap[x,y,0]

    for x in range(1,nx-1):
        for y in range(1,ny-1):
    
            v = V[t-1,x,y]
            r = R[t-1,x,y]
    
            # FitzHugh-Nagumo model equations
            dv = (v - (v**3) / 3.0 - r + I_ext) * dt
            dr = eps * (v + a - b * r) * dt

            V[t,x,y] = V[t-1,x,y] + dv + lap[x,y,1]*D
            R[t,x,y] = R[t-1,x,y] + dr




            