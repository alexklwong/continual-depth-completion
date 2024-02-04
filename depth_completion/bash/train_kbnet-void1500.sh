#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python depth_completion/src/train_depth_completion.py \
--train_image_paths training/void/unsupervised/void_train_image_1500.txt \
--train_sparse_depth_path training/void/unsupervised/void_train_sparse_depth_1500.txt \
--train_intrinsics_path training/void/unsupervised/void_train_intrinsics_1500.txt \
--val_image_path testing/void/void_test_image_1500.txt \
--val_sparse_depth_path testing/void/void_test_sparse_depth_1500.txt \
--val_intrinsics_path testing/void/void_test_intrinsics_1500.txt \
--val_ground_truth_path testing/void/void_test_ground_truth_1500.txt \
--n_batch 12 \
--n_height 416 \
--n_width 512 \
--model_name kbnet_void \
--network_modules depth pose \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--learning_rates 1e-4 5e-5 \
--learning_schedule 20 40 \
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
--w_losses w_color=0.15 w_structure=0.95 w_sparse_depth=2.0 w_smoothness=2.0 \
--w_weight_decay_depth 0.00 \
--w_weight_decay_pose 0.00 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--evaluation_protocol default \
--n_step_per_summary 1000 \
--n_step_per_checkpoint 1000 \
--start_step_validation 1000 \
--checkpoint_path \
trained_completion/kbnet/void1500/kbnet_12x416x512_min01max80_lr0-1e4_20-5e5_40_co015_st095_sz200_sm200_bri050-150_con050-150_hue10_sat050-150_hvflip_rot25_cap090-100_rap060-100_rac000-000_rmim1e3-5e3_5x5_rmpt060-070 \
--device gpu \
--n_thread 8 \
--port 8888