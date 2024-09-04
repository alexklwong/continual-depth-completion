#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python depth_completion/src/train_depth_completion.py \
--train_image_paths \
    training/nyu_v2/unsupervised/nyu_v2_train_image_corner.txt \
--train_sparse_depth_paths \
    training/nyu_v2/unsupervised/nyu_v2_train_sparse_depth_corner.txt \
--train_intrinsics_paths \
    training/nyu_v2/unsupervised/nyu_v2_train_intrinsics_corner.txt \
--val_image_path testing/nyu_v2/nyu_v2_test_image_corner.txt \
--val_sparse_depth_path testing/nyu_v2/nyu_v2_test_sparse_depth_corner.txt \
--val_intrinsics_path testing/nyu_v2/nyu_v2_test_intrinsics_corner.txt \
--val_ground_truth_path testing/nyu_v2/nyu_v2_test_ground_truth_corner.txt \
--model_name fusionnet_void \
--network_modules depth pose initialize_scaffnet spatial_pyramid_pool fisher \
--min_predict_depth 0.1 \
--max_predict_depth 8.0 \
--train_batch_size 12 \
--train_crop_shapes \
    416 576 \
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
--augmentation_random_resize_and_pad 0.70 1.00 \
--augmentation_random_resize_and_crop -1 -1 \
--augmentation_random_remove_patch_percent_range_image 1e-3 5e-3 \
--augmentation_random_remove_patch_size_image 5 5 \
--augmentation_random_remove_patch_percent_range_depth 0.60 0.70 \
--augmentation_random_remove_patch_size_depth 1 1 \
--supervision_type unsupervised \
--w_losses \
    w_color=0.20 \
    w_structure=0.80 \
    w_sparse_depth=1.0 \
    w_smoothness=1.0 \
    threshold_prior_depth=0.30 \
    w_prior_depth=0.10 \
    w_weight_decay_depth=0.0 \
    w_weight_decay_pose=0.0 \
--min_evaluate_depth 0.2 \
--max_evaluate_depth 5.0 \
--evaluation_protocol default \
--n_step_per_summary 1000 \
--n_step_per_checkpoint 1000 \
--start_step_validation 1000 \
--restore_path \
    external_models/scaffnet/scenenet/scaffnet-scenenet_v2.pth \
    trained_completion/fusionnet/nyu_v2/fusionnet_nyu_pretrained2/checkpoints_fusionnet_void-95000/fusionnet-95000.pth \
    trained_completion/fusionnet/nyu_v2/fusionnet_nyu_pretrained2/checkpoints_fusionnet_void-95000/posenet-95000.pth \
--checkpoint_path \
    trained_completion/fusionnet/nyu_v2/fusionnet_nyu_pretrained7 \
--device gpu \
--n_thread 8