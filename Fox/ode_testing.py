import numpy as np
import scipy.integrate as int
import matplotlib.pyplot as plt

f = lambda t,y: y
sol = int.solve_ivp(f,[0,2],[0,2,4],max_step=0.1)
t = sol.t
y = sol.y
print(t)
print(y[1])
plt.plot(t,y[1])
plt.show()

t = np.linspace(0, 0.1, 100)
y0 = 1
sol2 = int.odeint(f, y0, t)
print(sol2.shape)
plt.plot(t,sol2)
plt.show()

