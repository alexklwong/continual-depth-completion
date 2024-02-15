#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=58888
L=1e-4

python depth_completion/src/ddp_train_depth_completion.py \
--train_image_paths training/kitti/supervised/kitti_train_image.txt \
--train_sparse_depth_paths training/kitti/supervised/kitti_train_sparse_depth.txt \
--train_ground_truth_paths training/kitti/supervised/kitti_train_ground_truth.txt \
--train_intrinsics_paths training/kitti/supervised/kitti_train_intrinsics.txt \
--val_image_path validation/kitti/kitti_val_image.txt \
--val_sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--val_intrinsics_path validation/kitti/kitti_val_intrinsics.txt \
--val_ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--n_batch 12 \
--n_height 240 \
--n_width 1216 \
--model_name nlspn \
--min_predict_depth 0.0 \
--max_predict_depth 90.0 \
--learning_rates 1e-3 \
--learning_schedule 10 \
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
--min_evaluate_depth 1e-3 \
--max_evaluate_depth 90.0 \
--evaluation_protocol kitti \
--checkpoint_path \
    model_save/nlspn_synthia_kitti_1e-3/ \
--n_step_per_checkpoint 5000 \
--n_step_per_summary 1000 \
--n_image_per_summary 4 \
--validation_start_step 5000 \
--restore_paths_model model_ckpt/nlspn/nlspn_synthia.pth \
--device gpu \
--n_thread 4

#256 640 320 768 \
# --augmentation_random_crop_to_shape 256 640 256 768 \
# --augmentation_random_gaussian_blur_kernel_size 3 5 \
# --augmentation_random_gaussian_blur_sigma_range 0.05 0.1 \
