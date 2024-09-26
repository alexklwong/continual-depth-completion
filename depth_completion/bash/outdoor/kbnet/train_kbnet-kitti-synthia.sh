#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python depth_completion/src/train_depth_completion.py \
--train_image_paths \
    out_training/synthia-kitti-person/unsupervised/synthia_train_images.txt \
--train_sparse_depth_path \
    out_training/synthia-kitti-person/unsupervised/synthia_train_sparse_depth.txt \
--train_intrinsics_path \
    out_training/synthia-kitti-person/unsupervised/synthia_train_intrinsics.txt \
--val_image_paths \
    out_validation/kitti/kitti_val_image.txt \
    out_testing/synthia-kitti-person/synthia_test_image.txt \
--val_sparse_depth_paths \
    out_validation/kitti/kitti_val_sparse_depth.txt \
    out_testing/synthia-kitti-person/synthia_test_sparse_depth.txt \
--val_intrinsics_paths \
    out_validation/kitti/kitti_val_intrinsics.txt \
    out_testing/synthia-kitti-person/synthia_test_intrinsics.txt \
--val_ground_truth_paths \
    out_validation/kitti/kitti_val_ground_truth.txt \
    out_testing/synthia-kitti-person/synthia_test_ground_truth.txt \
--model_name kbnet_kitti \
--network_modules depth pose \
--min_predict_depth 1.5 \
--max_predict_depth 100.0 \
--train_batch_size 12 \
--train_crop_shapes \
        320 640 \
--learning_rates 1e-4 5e-5 \
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
--min_evaluate_depth 1e-3 1e-3 \
--max_evaluate_depth 100.0 100.0 \
--evaluation_protocols kitti synthia \
--n_step_per_summary 2500 \
--n_step_per_checkpoint 2500 \
--start_step_validation 2500 \
--restore_paths \
    trained_completion/kitti_pretrained/kbnet/kbnet-kitti.pth \
    trained_completion/kitti_pretrained/kbnet/posenet-kitti.pth \
--checkpoint_path \
    trained_completion/outdoor/kbnet/synthia/kbnet_kitti_synthia \
--device gpu \
--n_thread 1
