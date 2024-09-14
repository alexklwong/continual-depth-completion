#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python depth_completion/src/train_depth_completion.py \
--train_image_paths \
    training/synthia-kitti-person/unsupervised/synthia_train_images.txt \
--train_sparse_depth_path \
    training/synthia-kitti-person/unsupervised/synthia_train_sparse_depth.txt \
--train_intrinsics_path \
    training/synthia-kitti-person/unsupervised/synthia_train_intrinsics.txt \
--val_image_paths \
    validation/kitti/kitti_val_image.txt \
    testing/synthia-kitti-person/synthia_test_image.txt \
--val_sparse_depth_paths \
    validation/kitti/kitti_val_sparse_depth.txt \
    testing/synthia-kitti-person/synthia_test_sparse_depth.txt \
--val_intrinsics_paths \
    validation/kitti/kitti_val_intrinsics.txt \
    testing/synthia-kitti-person/synthia_test_intrinsics.txt \
--val_ground_truth_paths \
    validation/kitti/kitti_val_ground_truth.txt \
    testing/synthia-kitti-person/synthia_test_ground_truth.txt \
--model_name kbnet_kitti \
--network_modules ewc depth pose fisher \
--min_predict_depth 0.0 \
--max_predict_depth 90.0 \
--train_batch_size 20 \
--train_crop_shapes \
        320 640 \
--learning_rates 5e-5 2.5e-5 \
--learning_schedule 3 6 \
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
    w_ewc=1.0 \
    w_color=0.15 \
    w_structure=0.95 \
    w_sparse_depth=2.0 \
    w_smoothness=2.0 \
    w_weight_decay_depth=0.0 \
    w_weight_decay_pose=0.0 \
--min_evaluate_depth 0.0 1e-3 \
--max_evaluate_depth 100.0 90.0 \
--evaluation_protocol kitti synthia \
--n_step_per_summary 2000 \
--n_step_per_checkpoint 2000 \
--start_step_validation 1000 \
--restore_paths \
    trained_completion/kitti_pretrained/kbnet/kbnet-kitti.pth \
    trained_completion/kitti_pretrained/kbnet/posenet-kitti.pth \
    trained_completion/outdoor/kitti/kbnet/kbnet_kitti_fisher/checkpoints_kbnet_kitti-486033/fisher-info_486033.pth \
--frozen_model_paths \
    trained_completion/kitti_pretrained/kbnet/kbnet-kitti.pth \
    trained_completion/kitti_pretrained/kbnet/posenet-kitti.pth \
--checkpoint_path \
    trained_completion/outdoor/synthia/kbnet/kbnet_kitti_synthia_ewc \
--device gpu \
--n_thread 8
