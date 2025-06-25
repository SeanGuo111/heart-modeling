import numpy as np
import optimap as om
import monochrome as mc
import ohio_data_functions as od
import correlation_functions as cf
from matplotlib import pyplot as plt

save_file = True
data_path = "/run/media/mason/967a45a1-7b0d-48db-8570-8e4aa7aba5ed/sean/data/dataset01/"
channels = [0,1,2,3,4]

#od.download_npys_from_disk(100, 2000, channels, data_path, data_path, save_file)
data = od.load_from_npy(1000, 1899, channels, [0,1,2,3,4], data_path, data_path, save_subsets=False)
voltage = data[0]
calcium = data[2]
print(voltage.shape)
print(calcium.shape)

# voltage = od.chunk(voltage, 200)
# calcium = od.chunk(calcium, 200)
# print(voltage.shape)
# print(calcium.shape)

# voltage = od.rotate(voltage)
# calcium = od.rotate(calcium)
# print(voltage.shape)
# print(calcium.shape)

# voltage = od.splice(voltage, flatten=True)
# calcium = od.splice(voltage, flatten=True)
# print(voltage.shape)
# print(calcium.shape)

# loc_x, loc_y = 0,0
# flag = True
# while flag == True:
#     i_x, i_y = np.random.randint(0, 1024), np.random.randint(0, 1024)
#     calcium_pixel = calcium[:,i_x,i_y]
#     voltage_pixel = voltage[:,i_x,i_y]
#     calcium_pixel_norm = cf.normalize(calcium_pixel)
#     voltage_pixel_norm = cf.normalize(voltage_pixel)

#     global_time_delay, window_correlations, cb_shifted = cf.process_pixel(voltage_pixel_norm, calcium_pixel_norm)
#     print(f"Pixel ({i_x}, {i_y}:")
#     print(global_time_delay)
#     print(window_correlations)

#     plt.plot(voltage_pixel_norm, label="v")
#     plt.plot(calcium_pixel_norm, label="ca")
#     plt.title(f"({i_x}, {i_y}, time delay: {global_time_delay})")
#     plt.legend()
#     plt.show()

#     if (abs(global_time_delay) > 10):
#         loc_x, loc_y = i_x, i_y
#         flag = False

global_delay_matrix, window_correlations_matrix, ca_delayed_matrix = cf.process_matrices(voltage, calcium)
vals,counts = np.unique(global_delay_matrix, return_counts=True)
index = np.argmax(counts)
mode = vals[index]

plt.imshow(global_delay_matrix)
plt.colorbar()
#plt.scatter(loc_x, loc_y, c="red")
plt.title(f"Spatial global time delay over t=1000-1899. Mode={mode}")
plt.show()

# for i in range(0,3):
#     random_coords = [rand_coord for rand_coord in np.random.randint(0,7200,5)]
#     random_videos = [spliced_data[rand_coord] for rand_coord in random_coords]
#     om.video.show_videos(random_videos, titles=random_coords, cmaps="Grays", interval=20)


# train_size = int(len(voltage) * 0.75)
# test_size = int(len(voltage) * 0.2)

# train_samples = voltage[:train_size]
# test_samples = voltage[-test_size:]
# print(train_samples.shape)
# print(test_samples.shape)

# om.show_videos(train_samples[-3:])
# om.show_videos(test_samples[:3])