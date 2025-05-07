import numpy as np
import optimap as om
import monochrome as mc
import ohio_data_functions as od

save_file = True
data_path = "/mnt/data2/sean/data/dataset01/"
#od.download_npys_from_disk(1, 99, [0,1,2,3,4], data_path, data_path, save_file)
data = od.load_from_npy(1, 99, [0,1,2,3,4], [0,1,2], data_path, data_path, save_subsets=True)
print(data.shape)