#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python depth_completion/src/run_depth_completion.py \
--image_path validation/kitti/kitti_val_image.txt \
--sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--intrinsics_path validation/kitti/kitti_val_intrinsics.txt \
--ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--restore_paths \
    pretrained_models/depth_completion/kbnet/kitti/kbnet-kitti.pth \
--model_name kbnet_kitti \
--network_modules depth \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--min_evaluate_depth 1e-3 \
--max_evaluate_depth 100.0 \
--output_path \
    pretrained_models/depth_completion/kbnet/kitti/evaluation_results/kitti-val \
--device gpu

