import numpy as np
import optimap as om
import monochrome as mc
import ohio_data_functions as od

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

dataset_c0 = dataset[2]
# dataset_100s_c0 = dataset_100s[0]
# mc.show(dataset_c0)
# mc.show(dataset_100s_c0)

print(f"Original channel 0 shape: {dataset_c0.shape}")
chunked_dataset = od.chunk(dataset_c0, 200, flatten=False)
chunked_dataset_flattened = od.chunk(dataset_c0, 200, flatten=True)
print(chunked_dataset_flattened.shape)
chunked_dataset_rotated = od.rotate(chunked_dataset_flattened, flatten=True)
print(chunked_dataset_rotated.shape)
final = od.splice(chunked_dataset_rotated, 5, flatten=True)
print(final.shape)

mc.show(final[0])
om.video.show_videos(final[0:5], cmaps="Grays", interval=20)
#om.video.show_videos(final[0][:5], titles=["0", "90", "180", "270"], cmaps="Grays", interval=20)

collage_video = om.video.collage([ndarray for ndarray in chunked_dataset_flattened], ncols=5, padding=20, padding_value=0)
om.video.show_video(collage_video, cmap="Grays", interval=20)
om.video.export_video(media_path_start + "dataset_c2_200-299_collage.mp4", collage_video)

for video in chunked_dataset_flattened:
    om.video.show_video(video, interval=50)
# random_coords = [(int(rand_coord[0]), int(rand_coord[1])) for rand_coord in np.random.randint(0, 2, size=(3,2))]
# random_videos = [chunked_dataset[rand_coord] for rand_coord in random_coords]
# om.video.show_videos(random_videos, titles=random_coords, cmaps="Grays", interval=20)