#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

python depth_completion/src/train_depth_completion.py \
--train_image_paths training/virtual_kitti-kitti-person/supervised/vkitti_train_image-clone.txt \
--train_sparse_depth_paths training/virtual_kitti-kitti-person/supervised/vkitti_train_sparse_depth-clone.txt \
--train_ground_truth_paths training/virtual_kitti-kitti-person/supervised/vkitti_train_ground_truth-clone.txt \
--train_intrinsics_paths training/virtual_kitti-kitti-person/supervised/vkitti_train_intrinsics-clone.txt \
--val_image_path testing/virtual_kitti-kitti-person/vkitti_test_image-clone.txt \
--val_sparse_depth_path testing/virtual_kitti-kitti-person/vkitti_test_sparse_depth-clone.txt \
--val_intrinsics_path testing/virtual_kitti-kitti-person/vkitti_test_intrinsics-clone.txt \
--val_ground_truth_path testing/virtual_kitti-kitti-person/vkitti_test_ground_truth-clone.txt \
--n_batch 12 \
--n_height 224 \
--n_width 1216 \
--model_name costdcnet_vkitti \
--min_predict_depth 0.0 \
--max_predict_depth 90.0 \
--learning_rates 1e-3 5e-4 2.5e-4 1.25e-4 6.25e-5 3.125e-5 1.5e-5 7.5e-6 3.75e-6 1.875e-6 \
--learning_schedule 5 10 15 20 25 30 35 40 45 50 \
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
--augmentation_random_flip_type horizontal \
--augmentation_random_rotate_max -1 \
--augmentation_random_crop_and_pad 0.90 1.00 \
--augmentation_random_resize_and_pad 0.60 1.00 \
--augmentation_random_resize_and_crop -1 -1 \
--augmentation_random_remove_patch_percent_range_image 1e-3 5e-3 \
--augmentation_random_remove_patch_size_image 5 5 \
--augmentation_random_remove_patch_percent_range_depth 0.60 0.70 \
--augmentation_random_remove_patch_size_depth 1 1 \
--supervision_type supervised \
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
trained_completion/msg_chn_vkitti/ \
--device gpu \
--n_thread 8 \
--port 8888