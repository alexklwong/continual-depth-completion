#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python depth_completion/src/train_depth_completion.py \
--train_image_paths \
    training/nyu_v2/unsupervised/nyu_v2_train_image_corner.txt \
--train_sparse_depth_path \
    training/nyu_v2/unsupervised/nyu_v2_train_sparse_depth_corner.txt \
--train_intrinsics_path \
    training/nyu_v2/unsupervised/nyu_v2_train_intrinsics_corner.txt \
--val_image_paths \
    validation/nyu_v2/nyu_v2_val_image_corner.txt \
    testing/void/void_test_image_1500.txt \
--val_sparse_depth_paths \
    validation/nyu_v2/nyu_v2_val_sparse_depth_corner.txt \
    testing/void/void_test_sparse_depth_1500.txt \
--val_intrinsics_paths \
    validation/nyu_v2/nyu_v2_val_intrinsics_corner.txt \
    testing/void/void_test_intrinsics_1500.txt \
--val_ground_truth_paths \
    validation/nyu_v2/nyu_v2_val_ground_truth_corner.txt \
    testing/void/void_test_ground_truth_1500.txt \
--model_name kbnet_void \
--network_modules depth pose ewc fisher \
--min_predict_depth  0.1 \
--max_predict_depth 8.0 \
--train_batch_size 12 \
--train_crop_shapes \
    416 512 \
--learning_rates 1e-4 5e-5 2.5e-5 \
--learning_schedule 3 6 9 \
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
--augmentation_random_crop_type horizontal vertical \
--augmentation_random_flip_type horizontal vertical \
--augmentation_random_rotate_max 25 \
--augmentation_random_crop_and_pad 0.90 1.00 \
--augmentation_random_resize_and_pad 0.60 1.00 \
--augmentation_random_resize_and_crop -1 -1 \
--augmentation_random_remove_patch_percent_range_image 1e-3 5e-3 \
--augmentation_random_remove_patch_size_image 5 5 \
--augmentation_random_remove_patch_percent_range_depth 0.60 0.70 \
--augmentation_random_remove_patch_size_depth 1 1 \
--supervision_type unsupervised \
--w_losses \
    w_color=0.15 \
    w_structure=0.95 \
    w_sparse_depth=2.0 \
    w_smoothness=2.0 \
    w_weight_decay_depth=0.0 \
    w_weight_decay_pose=0.0 \
    w_ewc=0.05 \
--min_evaluate_depths 0.2 0.2 \
--max_evaluate_depths 5.0 5.0 \
--evaluation_protocol default \
--n_step_per_summary 1000 \
--n_step_per_checkpoint 1000 \
--start_step_validation 10000 \
--restore_paths \
    trained_completion/kbnet/void/kbnet_void_fisher/checkpoints_kbnet_void-70740/kbnet-70740.pth \
    trained_completion/kbnet/void/kbnet_void_fisher/checkpoints_kbnet_void-70740/posenet-70740.pth \
    trained_completion/kbnet/void/kbnet_void_fisher/checkpoints_kbnet_void-70740/fisher-info_70740.pth \
--frozen_model_paths \
    trained_completion/kbnet/void/kbnet_void_fisher/checkpoints_kbnet_void-70740/kbnet-70740.pth \
    trained_completion/kbnet/void/kbnet_void_fisher/checkpoints_kbnet_void-70740/posenet-70740.pth \
--checkpoint_path \
    trained_completion/kbnet/void_nyuv2_ewc0.05 \
--device gpu \
--n_thread 8
