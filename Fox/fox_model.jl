
include("ionic_currents.jl")
using Main.ionic_currents #IMPORTS MUST BE DONE IN THIS FORMAT: Main.package_name
import DifferentialEquations as DE, ProgressLogging
import Plots
import Logging: global_logger
import TerminalLoggers: TerminalLogger
# Above packages must all be installed into julia environment.

function set_initial_conditions()
    """
    Key:
    1: V
    2. Ca2+i
    3. Ca2+SR
    4. f
    5. d
    6. m
    7. h
    8. j
    9. fCa
    10. XKr
    11. XKs
    12. Xto
    13. Yto
    14. w (Omichi)
    """
    
    var = zeros(N,N,14)
    var[:,:,1] .= -94.7 
    var[:,:,2] .= 0.0472 
    var[:,:,3] .= 320 
    var[:,:,4] .= 0.983
    var[:,:,5] .= 0.0001
    var[:,:,6] .= 2.4676 * (10^-4)
    var[:,:,7] .= 0.99869
    var[:,:,8] .= 0.99887
    var[:,:,9] .= 0.942
    var[:,:,10] .= 0.229
    var[:,:,11] .= 0.0001
    var[:,:,12] .= 3.742 * (10^-5)
    var[:,:,13] .= 1
    var[:,:,14] .= 0.9 #Initial value???? Only chosen because ICa inactivation gates (f and fCa) start high, around 0.9-1.
    # var[14] is also unused for fox's model, only used for omichi.
   
    return var
end

function ionic_functions(var,t,i,j)
    # i, j only necessary for Istim.
    df = zeros(14)
    
    df = Istim(t,var,df,i,j)
    
    df, ENa = INa(t,var,df)
    # df = np.zeros(13,dtype=np.float64)

    df, EK = IK1(t,var,df)
    # df = np.zeros(13,dtype=np.float64)

    df = IKr(t,var,df,EK)

    df = IKs(t,var,df)

    df = Ito(t,var,df,EK)

    df = IKp(t,var,df,EK)

    df = INaK(t,var,df)

    df, INaCa_val, VFRT = INaCa(t,var,df)
    # df = np.zeros(13,dtype=np.float64)

    df = INab(t, var, df, ENa)

    df, ICab_val = ICab(t, var, df)
    # df = np.zeros(13,dtype=np.float64)

    df, IpCa_val = IpCa(t, var, df)    
    # df = np.zeros(13,dtype=np.float64)

    df, ICa_val, ICa_max = ICa(t, var, df, VFRT)
    # df = np.zeros(13,dtype=np.float64)

    df = ICaK(t, var, df, VFRT, ICa_max)

    df = calcium_handling_fox(t, var, df, ICa_val, ICab_val, IpCa_val, INaCa_val)

    # Flip sum of voltage -------------
    df[1] = -df[1]
    return df
end

function determine_diffusion_indices(x,y)
    """Given the x and y indices for a point on the grid, it returns the proper indices
    so as to allow for Neumann boundary conditions. Specifically, any index outside of
    the border is set to the index just inside of the border, so values accessed outside
    the border will be equal to those adjacent.
    """
    x_new, y_new = x, y
    if x_new == 0
        x_new = 1
    elseif x_new == N+1
        x_new = N
    end

    if y_new == 0
        y_new = 1
    elseif y_new == N+1
        y_new = N
    end

    return x_new, y_new
end

function laplacian(var)
    V = var[:,:,1]
    laplacian = zeros(N,N)
    for x_i = 1:N
        for y_i = 1:N
            prev_x_i, prev_y_i = determine_diffusion_indices(x_i-1, y_i-1)
            next_x_i, next_y_i = determine_diffusion_indices(x_i+1, y_i+1)
            laplacian[x_i, y_i] = (V[prev_x_i, y_i] + V[next_x_i, y_i] + V[x_i, prev_y_i] + V[x_i, next_y_i] - 4*V[x_i, y_i]) / (dx^2)
        end
    end

    return laplacian
end

function func(var,parameters,t)
    df = zeros(N,N,14)
    for i = 1:N
        for j = 1:N
            var_point = var[i,j,:]
            df[i,j,:] = ionic_functions(var_point,t,i,j)
        end
    end

    laplacian(var)
    df[:,:,1] .= df[:,:,1] .+ (D .* laplacian(var))

    return df
end

# function plots(t_eval, states)
#     voltage = states[0,:]
#     ca_i = states[1,:]
#     ca_SR = states[2,:]

#     fig, (ax1, ax2) = plt.subplots(1,2)
#     plt.sca(ax1)
#     plt.title("Voltage")
#     plt.plot(t_eval, voltage)
#     plt.sca(ax2)
#     plt.title("Intracellular calcium")
#     plt.plot(t_eval, ca_i)
#     plt.show()
#     plt.clf()

#     voltage_normalized = (voltage-np.min(voltage))/(np.max(voltage)-np.min(voltage))
#     ca_i_normalized = (ca_i-np.min(ca_i))/(np.max(ca_i)-np.min(ca_i))
#     ca_SR_normalized = (ca_SR-np.min(ca_SR))/(np.max(ca_SR)-np.min(ca_SR))

#     plt.plot(t_eval, voltage_normalized, label="V")
#     plt.plot(t_eval, ca_i_normalized, label = "Ca_i")
#     plt.plot(t_eval, ca_SR_normalized, label = "Ca_SR")
#     plt.legend()
#     plt.show()
# end

global_logger(TerminalLogger())

# Variables
initial_conditions = set_initial_conditions()
println("Initial conditions:")
println(initial_conditions[1,1,:])
params = 1.0

df_test = func(initial_conditions, params, 0)
println("\nDerivatives at time = 0, with initial conditions and given parameters:")
println(df_test[1,1,:])

# Set timespan to solve over
start_t = 0
end_t = 500
h = 10
num_points = (end_t-start_t)*h + 1
t_span = (start_t, end_t)
t_eval = range(start_t, end_t, length = num_points)

prob = DE.ODEProblem(func, initial_conditions, (Float64(start_t),Float64(end_t)))
sol = DE.solve(prob, DE.Tsit5(), saveat=0.1, maxiters=1e8, progress=true)
#states = sol.u
states = DE.stack(sol.u)
println("\nShape of states:")
println(size(states))

# Dimensions: i, j, variable, time
voltage_trace = states[3,3,1,:]
ca_i_trace = states[1,1,2,:]
ca_SR_trace = states[1,1,3,:]

voltage = states[:,:,1,:]

println()
println(size(voltage_trace))
println(size(t_eval))

Plots.plot(t_eval, voltage_trace, show=true)

#plots(t_eval, states)
println("Press the enter key to quit:")
readline()

# Plots.(t_eval, voltage_trace, show=true)
