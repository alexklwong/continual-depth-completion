#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory> <filename>"
    exit 1
fi

# Assign arguments to variables
base_dir=$1
filename=$2
source_dir="$base_dir/outputs"
target_dir="$base_dir/to_visualize"

# Array of subdirectories
subdirs=("ground_truth" "image" "output_depth" "sparse_depth")

# Loop over each subdirectory in subdirs array
for subdir in "${subdirs[@]}"; do
    # Create target subdirectory if it doesn't exist
    mkdir -p "$target_dir/$subdir"
    
    # Copy the file from source to target
    cp "$source_dir/$subdir/$filename.png" "$target_dir/$subdir/$filename.png"
    
    # Check if the copy was successful
    if [ $? -eq 0 ]; then
        echo "Copied $source_dir/$subdir/$filename.png to $target_dir/$subdir/$filename.png"
    else
        echo "Failed to copy $source_dir/$subdir/$filename.png"
    fi
done
