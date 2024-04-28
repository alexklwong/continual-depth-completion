#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python depth_completion/src/train_depth_completion.py \
--train_image_paths \
    training/kitti/unsupervised/kitti_train_nonstatic_images.txt \
--train_sparse_depth_path \
    training/kitti/unsupervised/kitti_train_nonstatic_sparse_depth.txt \
--train_intrinsics_path \
    training/kitti/unsupervised/kitti_train_nonstatic_intrinsics.txt \
--val_image_path testing/void/void_test_image_1500.txt \
--val_sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--val_intrinsics_path testing/void/void_test_intrinsics_1500.txt \
--val_ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--model_name voiced \
--network_modules depth pose ewc \
--min_predict_depth 0.1 \
--max_predict_depth 100.0 \
--train_batch_size 12 \
--train_crop_shapes \
    320 768 \
--learning_rates 0 \
--learning_schedule 1 \
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
    w_ewc=1.0 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--evaluation_protocol default \
--n_step_per_summary 1000 \
--n_step_per_checkpoint 1000 \
--start_step_validation 100000 \
--restore_paths \
    external_models/voiced/void/voiced-void1500.pth \
    external_models/voiced/void/posenet-void1500.pth \
    trained_completion/voiced/void1500/voiced_void1500/checkpoints_voiced-77740/fisher-info_77740.pth \
--checkpoint_path \
    trained_completion/voiced/kitti/voiced_void1500_kitti_ewc \
--frozen_model_paths \
    external_models/voiced/void/voiced-void1500.pth \
    external_models/voiced/void/posenet-void1500.pth \
--device gpu \
--n_thread 8
