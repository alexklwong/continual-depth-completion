#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python depth_completion/src/train_depth_completion.py \
--train_image_paths \
    training/kitti/unsupervised/kitti_train_nonstatic_images.txt \
--train_sparse_depth_path \
    training/kitti/unsupervised/kitti_train_nonstatic_sparse_depth.txt \
--train_intrinsics_path \
    training/kitti/unsupervised/kitti_train_nonstatic_intrinsics.txt \
--val_image_path validation/kitti/kitti_val_image.txt \
--val_sparse_depth_path validation/kitti/kitti_val_sparse_depth.txt \
--val_intrinsics_path validation/kitti/kitti_val_intrinsics.txt \
--val_ground_truth_path validation/kitti/kitti_val_ground_truth.txt \
--model_name fusionnet_kitti \
--network_modules depth pose spatial_pyramid_pool fisher \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--train_batch_size 8 \
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
--min_evaluate_depth 0.0 \
--max_evaluate_depth 100.0 \
--evaluation_protocol kitti \
--n_step_per_summary 1000 \
--n_step_per_checkpoint 1000 \
--start_step_validation 1000 \
--restore_paths \
    trained_completion/kitti_pretrained/fusionnet/fusionnet-kitti.pth \
    trained_completion/kitti_pretrained/fusionnet/posenet-kitti.pth \
--checkpoint_path \
    trained_completion/outdoor/kitti/fusionnet/fusionnet_kitti_fisher \
--device gpu \
--n_thread 8
