import numpy as np
import optimap as om
import monochrome as mc
import ohio_data_functions as od

save_file = True
data_path = "/mnt/data2/sean/data/dataset01/"
channels = [0,1,2,3,4]

#od.download_npys_from_disk(100, 2000, channels, data_path, data_path, save_file)
data = od.load_from_npy(1, 1000, channels, [0,1,2,3,4], data_path, data_path, save_subsets=False)[0]
print(data.shape)

chunked_data = od.chunk(data, 250)
print(chunked_data.shape)

rotated_data = od.rotate(chunked_data)
print(rotated_data.shape)

spliced_data = od.splice(rotated_data, flatten=False)
print(spliced_data.shape)

# for i in range(0,3):
#     random_coords = [rand_coord for rand_coord in np.random.randint(0,7200,5)]
#     random_videos = [spliced_data[rand_coord] for rand_coord in random_coords]
#     om.video.show_videos(random_videos, titles=random_coords, cmaps="Grays", interval=20)

print("Probing one timseries:")
time_series_ex = spliced_data[0]
print(time_series_ex.shape)

train_size = int(len(time_series_ex) * 0.75)
test_size = int(len(time_series_ex) * 0.2)

train_samples = time_series_ex[:train_size]
test_samples = time_series_ex[-test_size:]
print(train_samples.shape)
print(test_samples.shape)

om.show_videos(train_samples[-3:])
om.show_videos(test_samples[:3])