#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python depth_completion/src/train_depth_completion.py \
--train_image_paths \
    training/waymo/waymo_train_image.txt \
--train_sparse_depth_path \
    training/waymo/waymo_train_lidar.txt \
--train_intrinsics_path \
    training/waymo/waymo_train_intrinsics.txt \
--val_image_paths \
    validation/kitti/kitti_val_image.txt \
    validation/waymo/waymo_val_image-subset.txt \
--val_sparse_depth_paths \
    validation/kitti/kitti_val_sparse_depth.txt \
    validation/waymo/waymo_val_lidar-subset.txt \
--val_intrinsics_paths \
    validation/kitti/kitti_val_intrinsics.txt \
    validation/waymo/waymo_val_intrinsics-subset.txt \
--val_ground_truth_paths \
    validation/kitti/kitti_val_ground_truth.txt \
    validation/waymo/waymo_val_ground_truth-subset.txt \
--model_name kbnet_kitti \
--network_modules depth pose fisher spatial_pyramid_pool ewc \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--train_batch_size 8 \
--train_crop_shapes \
        800 640 \
--learning_rates 1e-4 5e-5 \
--learning_schedule 10 15 \
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
    w_ewc=1.0 \
    w_color=0.15 \
    w_structure=0.95 \
    w_sparse_depth=2.0 \
    w_smoothness=2.0 \
    w_weight_decay_depth=0.0 \
    w_weight_decay_pose=0.0 \
--min_evaluate_depth 1e-3 1e-3 \
--max_evaluate_depth 100.0 80.0 \
--evaluation_protocols kitti waymo \
--n_step_per_summary 1000 \
--n_step_per_checkpoint 1000 \
--start_step_validation 1000 \
--restore_paths \
    trained_completion/kitti_pretrained/fusionnet/fusionnet-kitti.pth \
    trained_completion/kitti_pretrained/fusionnet/posenet-kitti.pth \
    trained_completion/outdoor/kitti/fusionnet/fusionnet_kitti_fisher/checkpoints_fusionnet_kitti-256033/fisher-info_256033.pth \
--frozen_model_paths \
    trained_completion/kitti_pretrained/fusionnet/fusionnet-kitti.pth \
    trained_completion/kitti_pretrained/fusionnet/posenet-kitti.pth \
--checkpoint_path \
    trained_completion/outdoor/waymo/kbnet/kbnet_kitti_waymo_ewc \
--device gpu \
--n_thread 8
