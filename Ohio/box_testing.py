import numpy as np
from scipy import stats 
import optimap as om
import monochrome as mc
import ohio_data_functions as od
import matplotlib.pyplot as plt
import correlation_functions as cf

media_path_start = "C:\\Users\\swguo\\VSCode Projects\\Heart Modeling\\Ohio\\media\\"
data_path = "C:\\Users\\swguo\\VSCode Projects\\Heart Modeling\\Ohio\\second_data_batch\\"
start_id = 200
end_id = 299
filename_channels = [0,1,2]
subset_channels = []
save_subsets = False

#download_npys_from_box(200,299,channels=[0,1,2],save_file=True)
dataset: np.ndarray = od.load_from_npy(200, 299, [0,1,2], subset_channels, data_path, data_path, save_subsets)
#dataset_100s: np.ndarray = load_from_npy(100, 199, [0,1,2,3,4], subset_channels, save_subsets)

# VOLTAGE FIRST. CALCIUM SECOND
voltage = dataset[0]
calcium = dataset[2]

loc_x, loc_y = 0,0
flag = True
while flag == True:
    i_x, i_y = np.random.randint(0, 1024), np.random.randint(0, 1024)
    calcium_pixel = calcium[:,i_x,i_y]
    voltage_pixel = voltage[:,i_x,i_y]
    calcium_pixel_norm = cf.normalize(calcium_pixel)
    voltage_pixel_norm = cf.normalize(voltage_pixel)

    global_time_delay, window_correlations, cb_shifted = cf.process_pixel(voltage_pixel_norm, calcium_pixel_norm)
    print(f"Pixel ({i_x}, {i_y}:")
    print(global_time_delay)
    print(window_correlations)

    plt.plot(voltage_pixel_norm, label="v")
    plt.plot(calcium_pixel_norm, label="ca")
    plt.title(f"({i_x}, {i_y}, time delay: {global_time_delay})")
    plt.legend()
    plt.show()

    if (abs(global_time_delay) > 10):
        loc_x, loc_y = i_x, i_y
        flag = False

global_delay_matrix, window_correlations_matrix, ca_delayed_matrix = cf.process_matrices(voltage, calcium)
vals,counts = np.unique(global_delay_matrix, return_counts=True)
index = np.argmax(counts)
mode = vals[index]

plt.imshow(global_delay_matrix)
plt.colorbar()
plt.scatter(loc_x, loc_y, c="red")
plt.title(f"Spatial global time delay over t=200-299. Mode={mode}")
plt.show()

print(f"Original channel 0 shape: {calcium.shape}")
chunked_dataset = od.chunk(calcium, 200, flatten=False)
chunked_dataset_flattened = od.chunk(calcium, 200, flatten=True)
print(chunked_dataset_flattened.shape)
chunked_dataset_rotated = od.rotate(chunked_dataset_flattened, flatten=True)
print(chunked_dataset_rotated.shape)
final = od.splice(chunked_dataset_rotated, 5, flatten=True)
print(final.shape)
#Shuffle:
np.random.shuffle(final)

mc.show(final[0])
om.video.show_videos(final[0:5], interval=20)
#om.video.show_videos(final[0][:5], titles=["0", "90", "180", "270"], cmaps="Grays", interval=20)

collage_video = om.video.collage([ndarray for ndarray in chunked_dataset_flattened], ncols=5, padding=20, padding_value=0)
om.video.show_video(collage_video, interval=20)
#om.video.export_video(media_path_start + "dataset_c0_200-299_collage.mp4", collage_video)

for video in chunked_dataset_flattened:
    om.video.show_video(video, interval=50)
# random_coords = [(int(rand_coord[0]), int(rand_coord[1])) for rand_coord in np.random.randint(0, 2, size=(3,2))]
# random_videos = [chunked_dataset[rand_coord] for rand_coord in random_coords]
# om.video.show_videos(random_videos, titles=random_coords, cmaps="Grays", interval=20)