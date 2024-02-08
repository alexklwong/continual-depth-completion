#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python depth_completion/src/train_depth_completion.py \
--train_image_paths training/synthia-kitti-person/suprvised_subsample/synthia_train_image-subsample.txt \
--train_sparse_depth_paths training/synthia-kitti-person/suprvised_subsample/synthia_train_sparse_depth-subsample.txt \
--train_ground_truth_paths training/synthia-kitti-person/suprvised_subsample/synthia_train_ground_truth-subsample.txt \
--train_intrinsics_paths training/synthia-kitti-person/suprvised_subsample/synthia_train_intrinsics-subsample.txt \
--val_image_path testing/synthia-kitti-person/synthia_test_image.txt \
--val_sparse_depth_path testing/synthia-kitti-person/synthia_test_sparse_depth.txt \
--val_intrinsics_path testing/synthia-kitti-person/synthia_test_intrinsics.txt \
--val_ground_truth_path testing/synthia-kitti-person/synthia_test_ground_truth.txt \
--n_batch 16 \
--n_height 320 \
--n_width 640 \
--model_name costdcnet_synthia \
--min_predict_depth 0.0 \
--max_predict_depth 90.0 \
--learning_rates 5e-4 2.5e-4 1.25e-4 6.25e-5 3.125e-5 1.5e-5 \
--learning_schedule 1 2 3 4 5 6 \
--n_step_grad_acc 1 \
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
--augmentation_random_crop_and_pad 0.90 1.00 \
--augmentation_random_resize_and_pad 0.60 1.00 \
--augmentation_random_resize_and_crop -1 -1 \
--augmentation_random_remove_patch_percent_range_image -1 -1 \
--augmentation_random_remove_patch_size_image -1 -1 \
--augmentation_random_remove_patch_percent_range_depth -1 -1 \
--augmentation_random_remove_patch_size_depth -1 -1 \
--supervision_type supervised \
--w_losses w_l1=1.0 w_l2=1.0 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 90.0 \
--evaluation_protocol default \
--n_step_per_summary 1000 \
--n_step_per_checkpoint 10000 \
--start_step_validation 10000 \
--checkpoint_path \
trained_completion/costdcnet_synthia/ \
--device gpu \
--n_thread 8 \
--port 8888