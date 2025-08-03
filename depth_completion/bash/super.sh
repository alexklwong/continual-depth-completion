#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

python depth_completion/src/train_depth_completion.py \
--train_image_paths \
    training_v2/training/waymo/waymo_train_day_image.txt \
--train_sparse_depth_path \
    training_v2/training/waymo/waymo_train_day_lidar.txt \
--train_intrinsics_path \
    training_v2/training/waymo/waymo_train_day_intrinsics.txt \
--val_image_paths \
    validation/kitti/kitti_val_image.txt \
    tesing_v2/validation/waymo/waymo_val_day_image.txt \
--val_sparse_depth_paths \
    validation/kitti/kitti_val_sparse_depth.txt \
    tesing_v2/validation/waymo/waymo_val_day_lidar.txt \
--val_intrinsics_paths \
    validation/kitti/kitti_val_intrinsics.txt \
    tesing_v2/validation/waymo/waymo_val_day_intrinsics.txt \
--val_ground_truth_paths \
    validation/kitti/kitti_val_ground_truth.txt \
    tesing_v2/validation/waymo/waymo_val_day_ground_truth.txt \
--model_name msg_chn_kitti \
--network_modules depth pose \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--train_batch_size 4 \
--train_crop_shapes \
        800 640 \
--learning_rates 5e-5 1e-4 2e-4 1e-4 5e-5 \
--learning_schedule 2 8 30 45 60 \
--augmentation_probabilities 1.0 \
--augmentation_schedule -1 \
--augmentation_random_brightness 0.50 1.50 \
--augmentation_random_contrast 0.50 1.50 \
--augmentation_random_gamma -1 -1 \
--augmentation_random_hue -0.1 0.1 \
--augmentation_random_saturation 0.50 1.50 \
--augmentation_random_noise_type none \
--augmentation_random_noise_spread -1 \
--augmentation_padding_mode edge \
--augmentation_random_crop_type horizontal bottom \
--augmentation_random_flip_type horizontal \
--augmentation_random_rotate_max -1 \
--augmentation_random_crop_and_pad -1 -1 \
--augmentation_random_resize_and_pad 0.60 1.00 \
--augmentation_random_resize_and_crop -1 -1 \
--augmentation_random_remove_patch_percent_range_image -1 -1 \
--augmentation_random_remove_patch_size_image -1 -1 \
--augmentation_random_remove_patch_percent_range_depth -1 -1 \
--augmentation_random_remove_patch_size_depth -1 -1 \
--supervision_type unsupervised \
--w_losses \
    w_color=0.15 \
    w_structure=0.95 \
    w_sparse_depth=2.0 \
    w_smoothness=2.0 \
    w_weight_decay_depth=0.0 \
    w_weight_decay_pose=0.0 \
--min_evaluate_depth 1e-3 1e-3 1e-3 \
--max_evaluate_depth 100.0 80.0 100.0 \
--evaluation_protocols kitti waymo \
--n_step_per_summary 100 \
--n_step_per_checkpoint 100 \
--start_step_validation 100 \
--restore_paths \
    /media/home/xechen/continual-depth-completion/trained_completion/final.pth.tar \
--frozen_model_paths \
    /media/home/xechen/continual-depth-completion/trained_completion/final.pth.tar \
--checkpoint_path \
    trained_completion/rebutal/mg \
--device gpu \
--n_thread 8
