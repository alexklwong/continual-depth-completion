import os
import argparse
import random
import shutil

# Replay filepaths
replay_image_paths = "training2/void/unsupervised/void_train_image_1500.txt"
replay_sparse_depth_paths = "training2/void/unsupervised/void_train_sparse_depth_1500.txt"
replay_intrinsics_paths = "training2/void/unsupervised/void_train_intrinsics_1500.txt"

# Settings
resize_factor = 10

# New filepaths
new_image_paths = "training2/void/unsupervised/void_train_image_1500_extended.txt"
new_sparse_depth_paths = "training2/void/unsupervised/void_train_sparse_depth_1500_extended.txt"
new_intrinsics_paths = "training2/void/unsupervised/void_train_intrinsics_1500_extended.txt"



# Read filepaths into lists
image_paths = []
with open(replay_image_paths, 'r') as replay_file: 
     for line in replay_file:   
         image_paths.append(line)

sparse_depth_paths = []
with open(replay_sparse_depth_paths, 'r') as replay_file: 
     for line in replay_file:   
         sparse_depth_paths.append(line)

intrinsics_paths = []
with open(replay_intrinsics_paths, 'r') as replay_file: 
     for line in replay_file:   
         intrinsics_paths.append(line)

# Copy files
files = [replay_image_paths, replay_sparse_depth_paths, replay_intrinsics_paths]
new_files = [new_image_paths, new_sparse_depth_paths, new_intrinsics_paths]
for i, replay_file in enumerate(files):
    shutil.copyfile(replay_file, new_files[i])

# Extend files
all_lines = list(zip(image_paths, sparse_depth_paths, intrinsics_paths))

for i in range(resize_factor-1):
    # Copy and shuffle
    all_lines2 = all_lines.copy()
    random.shuffle(all_lines2)
    all_lines2 = list(zip(*all_lines2))
    # Add new lines
    for j, lines in enumerate(all_lines2):
        with open(new_files[j], 'a') as replay_file:
            for line in lines:
                replay_file.write(line)
