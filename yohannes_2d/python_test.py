import numpy as np
import matplotlib.pyplot as plt
import ca_cardiac_model

# python -m numpy.f2py -c -m ca_cardiac_model cardiac_m.f90 ca-currents.f v-currents.f

def localized_random_stimulus(ix, iy, time, radius=5, stim_duration=1.0, time_prob=0.01):
    global last_time_checked, stimulus_centers, stimulus_active, stimulus_start_time
    
    # Check if we're at a new time point
    if abs(time - last_time_checked) > 1e-6:  # Floating point comparison
        last_time_checked = time
        
        # If stimulus is active, check if it should end
        if stimulus_active and (time - stimulus_start_time > stim_duration):
            stimulus_active = False
        
        # Randomly decide to start a new stimulus
        if not stimulus_active and random.random() < time_prob:
            stimulus_active = True
            stimulus_start_time = time
            # Choose a random center point
            center_x = random.randint(1, lx)
            center_y = random.randint(1, ly)
            stimulus_centers = [(center_x, center_y)]
    
    # Check if current point is within the stimulus radius of any center
    
    if stimulus_active:
        #print(stimulus_active)
        for center_x, center_y in stimulus_centers:
            distance = ((ix - center_x)**2 + (iy - center_y)**2)**0.5
            if distance <= radius:
                return 80.0  # Return stimulus current
    
    return 0.0  # No stimulus

def run_cardiac_simulation():
    # Set parameters from the provided defaults
    lx = 10  # Assuming a small grid size, adjust as needed
    ly = 10  # Assuming a small grid size, adjust as needed
    nstim = 10  # Number of stimuli, adjust as needed
    iseed = 823323  # Initial random number seed
    rbcl = 1000.0  # Pacing rate
    dfu = 0.0001  # Effective voltage diffusion coefficient
    
    # Ionic current parameters
    gicai = 2.20  # Strength of LCC
    gtos = 0.04  # Strength of ito slow
    gtof = 0.15  # Strength of ito fast
    gnacai = 1.5  # Strength of NCX
    
    zxr = 0.09  # Controls degree of Ca-induced inactivation
    
    nbt = 4000  # Total number of RyR2 clusters
    cxinit = 1200.0  # Initial SR load
    
    # Sodium concentration calculation
    xmx = -2.0/250.0
    xnai = xmx * 520.0 + 16.0  # Constant Na concentration
    
    # Constants
    xnao = 136.0  # External Na (mM)
    xki = 140.0  # Internal K (mM)
    xko = 5.40  # External K (mM)
    cao = 1.8  # External Ca (mM)
    
    temp = 308.0  # Temperature (K)
    xxr = 8.314  # Gas constant
    xf = 96.485  # Faraday's constant
    
    dt = 0.1  # Time step
    
    # Define buffer size for time steps - control from Python
    max_buffer_size = 20000  # Adjust based on your expected simulation length
    parallel = False # RK4
    mod_output = 10 
    
    # Run the simulation with Python-controlled buffer size
    v_out, cb_out, csrb_out, ci_out, t_out, num_steps = ca_cardiac_model.cardiac_simulation(
        lx, ly, nstim, iseed, rbcl, dfu, gicai, gtos, gtof, gnacai, zxr, nbt, 
        cxinit, xnai, xnao, xki, xko, cao, temp, xxr, xf, dt,
        max_buffer_size,mod_output,parallel,localized_random_stimulus
    )
    
    print(f"Simulation completed with {num_steps} steps (buffer size: {max_buffer_size})")
    
    return v_out, cb_out, csrb_out, ci_out, t_out, num_steps

def plot_cell_output(v_out, cb_out, t_out, num_steps, cell_x=1, cell_y=1):
    """Plot the voltage and calcium concentration for a specific cell"""
    # Extract data for the specified cell
    cell_x_idx = cell_x - 1 if cell_x > 0 else cell_x
    cell_y_idx = cell_y - 1 if cell_y > 0 else cell_y
    
    # Extract data only up to the number of steps actually computed
    voltage = v_out[cell_x_idx, cell_y_idx, :num_steps]
    calcium = cb_out[cell_x_idx, cell_y_idx, :num_steps]
    time = t_out[:num_steps]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot voltage
    ax1.plot(time, voltage, 'b-', linewidth=2)
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title(f'Cardiac Cell Simulation Results at Position ({cell_x}, {cell_y})')
    ax1.grid(True)
    
    # Plot calcium concentration
    ax2.plot(time, calcium, 'r-', linewidth=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Ca²⁺ Concentration (μM)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('cardiac_simulation_results.png')
    plt.show()

if __name__ == "__main__":
    print("Running cardiac simulation...")
    v_out, cb_out, csrb_out, ci_out, t_out, num_steps = run_cardiac_simulation()
    
    print(f"Time range: {t_out[0]} to {t_out[num_steps-1]} ms")
    
    # Plot results for cell at position (1,1)
    plot_cell_output(v_out, cb_out, t_out, num_steps, cell_x=1, cell_y=1)
