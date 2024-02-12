#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=58888
L=1e-4

python depth_completion/src/ddp_train_depth_completion.py \
--train_image_paths training/synthia-kitti-person/suprvised_subsample/synthia_train_image-subsample.txt \
--train_sparse_depth_paths training/synthia-kitti-person/suprvised_subsample/synthia_train_sparse_depth-subsample.txt \
--train_ground_truth_paths training/synthia-kitti-person/suprvised_subsample/synthia_train_ground_truth-subsample.txt \
--train_intrinsics_paths training/synthia-kitti-person/suprvised_subsample/synthia_train_intrinsics-subsample.txt \
--val_image_path testing/synthia-kitti-person/synthia_test_image.txt \
--val_sparse_depth_path testing/synthia-kitti-person/synthia_test_sparse_depth.txt \
--val_intrinsics_path testing/synthia-kitti-person/synthia_test_intrinsics.txt \
--val_ground_truth_path testing/synthia-kitti-person/synthia_test_ground_truth.txt \
--n_batch 32 \
--n_height 320 \
--n_width 640 \
--model_name msg_chn \
--min_predict_depth 0.0 \
--max_predict_depth 90.0 \
--learning_rates 1e-3  \
--learning_schedule 20 \
--lr_warmup_iter -1 \
--augmentation_probabilities 1.00 \
--augmentation_schedule -1 \
--augmentation_random_brightness 0.5 1.5 \
--augmentation_random_contrast 0.5 1.5 \
--augmentation_random_saturation 0.5 1.5 \
--augmentation_random_gamma -1 -1 \
--augmentation_random_hue -0.1 0.1 \
--augmentation_random_noise_type none \
--augmentation_random_noise_spread -1 \
--augmentation_random_crop_type horizontal bottom \
--augmentation_random_crop_to_shape -1 -1 \
--augmentation_random_flip_type horizontal \
--augmentation_random_rotate_max -1 \
--augmentation_random_crop_and_pad 0.9 1.0 \
--augmentation_random_resize_and_pad -1 -1 \
--augmentation_random_resize_and_crop -1 -1 \
--augmentation_random_remove_patch_percent_range_image -1 -1 \
--augmentation_random_remove_patch_size_image -1 -1 \
--augmentation_random_remove_patch_percent_range_depth -1 -1 \
--augmentation_random_remove_patch_size_depth -1 -1 \
--w_losses w_l1=1.0 w_l2=1.0 w_smoothness=0.00 \
--w_weight_decay 0.0 \
--min_evaluate_depth 0.0 \
--max_evaluate_depth 90.0 \
--evaluation_protocol default \
--checkpoint_path \
    model_save/msg_chn_synthia/ \
--n_step_per_checkpoint 5000 \
--n_step_per_summary 1000 \
--n_image_per_summary 4 \
--validation_start_step 5000 \
--device gpu \
--n_thread 8

#256 640 320 768 \
# --augmentation_random_crop_to_shape 256 640 256 768 \
# --augmentation_random_gaussian_blur_kernel_size 3 5 \
# --augmentation_random_gaussian_blur_sigma_range 0.05 0.1 \
