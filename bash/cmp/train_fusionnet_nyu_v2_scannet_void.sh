#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python depth_completion/src/train_depth_completion.py \
--train_image_paths training/void/void_train_image.txt \
--train_sparse_depth_path training/void/void_train_sparse_depth.txt \
--train_intrinsics_path training/void/void_train_intrinsics.txt \
--val_image_paths \
    testing/nyu_v2/nyu_v2_test_image.txt \
    testing/scannet/scannet_test_image.txt \
    testing/void/void_test_image.txt \
--val_sparse_depth_paths \
    testing/nyu_v2/nyu_v2_test_sparse_depth.txt \
    testing/scannet/scannet_test_sparse_depth.txt \
    testing/void/void_test_sparse_depth.txt \
--val_intrinsics_paths \
    testing/nyu_v2/nyu_v2_test_intrinsics.txt \
    testing/scannet/scannet_test_intrinsics.txt \
    testing/void/void_test_intrinsics.txt \
--val_ground_truth_paths \
    testing/nyu_v2/nyu_v2_test_ground_truth.txt \
    testing/scannet/scannet_test_ground_truth.txt \
    testing/void/void_test_ground_truth.txt \
--replay_image_paths \
--replay_sparse_depth_paths \
--replay_intrinsics_paths \
--model_name fusionnet_nyu_v2_scannet \
--network_modules depth pose spatial_pyramid_pool \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--train_batch_size 4 \
--train_crop_shapes 416 512 \
--replay_batch_size 4 \
--replay_crop_shapes \
    416 512 \
    416 512 \
--replay_dataset_size 64 \
--replay_seed 526 \
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
--create_optimized_replay_buffer true \
--min_evaluate_depth 0.2 0.2 0.2 \
--max_evaluate_depth 5.0 5.0 5.0 \
--evaluation_protocols nyu_v2 scannet void \
--n_step_per_summary 5000 \
--n_step_per_checkpoint 5000 \
--start_step_validation 5000 \
--restore_paths \
--frozen_model_paths \
--checkpoint_path trained_completion/replay/fusionnet_nyu_v2_scannet_void \
--device gpu \
--n_thread 8
