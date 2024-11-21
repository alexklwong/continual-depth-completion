#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory_name>"
    exit 1
fi

# Assign the argument to a variable
directory_name=$1

# Set the CUDA device
export CUDA_VISIBLE_DEVICES=3

# Run the Python script with the specified directory name
python tools/visualize_output.py \
--output_root_dirpath \
    "$directory_name"/to_visualize \
--visualization_dirpath \
    "$directory_name"/visualize_output \
--task depth_completion \
--visualize_error \
--vmin 1.0 \
--vmax 60.0
