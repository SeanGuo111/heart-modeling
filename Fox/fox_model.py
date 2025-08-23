import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import ode
from parameters import *
import ionic_currents as ic

def set_initial_conditions():
    """
    Key:
    0: V
    1. Ca2+i
    2. Ca2+SR
    3. f
    4. d
    5. m
    6. h
    7. j
    8. fCa
    9. XKr
    10. XKs
    11. Xto
    12. Yto
    """

    var = np.zeros(13,dtype=np.float64)
    var[0] = -94.7 
    var[1] = 0.0472 
    var[2] = 320 
    var[3] = 0.983
    var[4] = 0.0001
    var[5] = 2.4676 * (10**-4)
    var[6] = 0.99869
    var[7] = 0.99887
    var[8] = 0.942
    var[9] = 0.229
    var[10] = 0.0001
    var[11] = 3.742 * (10**-5)
    var[12] = 1
    
    return var

njit()
def function(t,var):
    df = np.zeros(13,dtype=np.float64)
    
    df = Istim(t,var,df)
    
    df, ENa = ic.INa(t,var,df)
    # df = np.zeros(13,dtype=np.float64)

    df, EK = ic.IK1(t,var,df)
    # df = np.zeros(13,dtype=np.float64)

    df = ic.IKr(t,var,df,EK)

    df = ic.IKs(t,var,df)

    df = ic.Ito(t,var,df,EK)

    df = ic.IKp(t,var,df,EK)

    df = ic.INaK(t,var,df)

    df, INaCa_val, VFRT = ic.INaCa(t,var,df)
    # df = np.zeros(13,dtype=np.float64)

    df = ic.INab(t, var, df, ENa)

    df, ICab_val = ic.ICab(t, var, df)
    # df = np.zeros(13,dtype=np.float64)

    df, IpCa_val = ic.IpCa(t, var, df)    
    # df = np.zeros(13,dtype=np.float64)

    df, ICa_val, ICa_max = ic.ICa(t, var, df, VFRT)
    # df = np.zeros(13,dtype=np.float64)

    df = ic.ICaK(t, var, df, VFRT, ICa_max)

    df = ic.calcium_handling(t, var, df, ICa_val, ICab_val, IpCa_val, INaCa_val)

    # Flip sum of voltage sum -------------
    df[0] = -df[0]
    return df

def Istim(t,var,df):
    #NOTE: RECONFIGURE
    if t > 10 and t < 11:
        print(t)
        df[0] += -80
    return df

# Variables
var = set_initial_conditions()
print("Initial conditions:")
print(var)

df_test = function(0, var)
print("\nDerivatives at time = 0, with initial conditions and given parameters:")
print(df_test)

# Initialise constants and state variables
init_states = set_initial_conditions()

# Set timespan to solve over
start = 0
end = 300
h = 10
num_points = (end-start)*h + 1
t = np.linspace(start, end, num_points)

# Construct ODE object to solve
r = ode(function)
r.set_integrator('vode', method='bdf', atol=1e-06, rtol=1e-06, max_step=1)
r.set_initial_value(init_states, t[0])

initial_rates = function(t[0], init_states)
print(f"Initial rates: {initial_rates}")

# Solve model
states = np.zeros((13,num_points))
states[:,0] = init_states
for (i,t_end) in tqdm(enumerate(t[1:])):
    if r.successful():
        r.integrate(t_end)
        states[:,i+1] = r.y
    else:
        break
voltage = states[0,:]
ca_i = states[1,:]


fig, (ax1, ax2) = plt.subplots(1,2)
plt.sca(ax1)
plt.title("Voltage")
plt.plot(t, voltage)
plt.sca(ax2)
plt.title("Intracellular calcium")
plt.plot(t, ca_i)
plt.show()

plt.clf()
voltage_normalized = (voltage-np.min(voltage))/(np.max(voltage)-np.min(voltage))
ca_i_normalized = (ca_i-np.min(ca_i))/(np.max(ca_i)-np.min(ca_i))
ca_SR = states[2,:]
ca_SR_normalized = (ca_SR-np.min(ca_SR))/(np.max(ca_SR)-np.min(ca_SR))

plt.plot(t, voltage_normalized, label="V")
plt.plot(t, ca_i_normalized, label = "Ca_i")
plt.plot(t, ca_SR_normalized, label = "Ca_SR")
plt.legend()
plt.show()