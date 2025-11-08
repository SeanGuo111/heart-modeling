import monochrome as mc
import optimap as om
import numpy as np
import matplotlib.pyplot as plt


voltage = np.load("C:\\Users\\swguo\\VSCode Projects\\Heart Modeling\\voltage_11x11_2000_Cm1_D0.1.npy")
voltage = np.moveaxis(voltage, -1, 0)
print(voltage.shape)

mc.show(voltage)
mc.export_video("C:\\Users\\swguo\\VSCode Projects\\Heart Modeling\\voltage_11x11_2000_Cm1_D0.1.mp4")

fig, axs = plt.subplots(1,3)
plt.sca(axs[0])
plt.plot(voltage[:,6,6])
plt.title("Trace (3,3)")
lim = plt.ylim()
plt.sca(axs[1])
plt.plot(voltage[:,7,7])
plt.title("Trace (4,4)")
plt.ylim(lim)
plt.sca(axs[2])
plt.plot(voltage[:,8,8])
plt.title("Trace (5,5)")
plt.ylim(lim)
plt.show()