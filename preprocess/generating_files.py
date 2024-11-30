import os
from tqdm import tqdm

data_path = '/home/yz/dataset/SSCBenchKITTI360'
sequences = ['2013_05_28_drive_0000_sync', '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync', '2013_05_28_drive_0004_sync',
             '2013_05_28_drive_0005_sync', '2013_05_28_drive_0006_sync', '2013_05_28_drive_0007_sync', '2013_05_28_drive_0009_sync',
             '2013_05_28_drive_0010_sync'
             ]

for sequence in sequences:
    left_images = os.listdir(os.path.join(data_path, 'data_2d_raw', sequence, 'image_00/data_rect'))
    left_images.sort()
    left_image_files = []
    right_image_files = []
    for left_image in tqdm(left_images):
        left_image_files.append(os.path.join('image_00/data_rect', left_image))
        right_image_files.append(os.path.join('image_01/data_rect', left_image))
    
    with open('filenames_kitti360/' + sequence + '.txt', "w", encoding="utf-8") as file:
        for i in range(len(left_image_files)):
            file.write(f"{left_image_files[i]} {right_image_files[i]}\n")