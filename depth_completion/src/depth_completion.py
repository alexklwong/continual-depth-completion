import os, time, sys, tqdm
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
sys.path.insert(0, os.getcwd())
import datasets
from utils.src import data_utils, eval_utils, net_utils
from utils.src import data_utils, eval_utils, net_utils
from utils.src.log_utils import log
from depth_completion_model import DepthCompletionModel
from utils.src.transforms import Transforms
from PIL import Image






def train(train_image_paths,
          train_sparse_depth_paths,
          train_intrinsics_paths,
          train_ground_truth_paths,
          # Replay filepaths
          # TODO: Uncomment filepaths to use
          replay_image_paths,
          replay_sparse_depth_paths,
          replay_intrinsics_paths,
          replay_ground_truth_paths,
          # Validation filepaths
          val_image_paths,  # Added support for multiple val datasets
          val_sparse_depth_paths,
          val_intrinsics_paths,
          val_ground_truth_paths,
          # TODO: Uncomment to use
          replay_batch_size,
          replay_crop_shapes,
          replay_dataset_size,
          # Depth network settings
          model_name,
          network_modules,
          min_predict_depth,
          max_predict_depth,
          # Training settings
          train_batch_size,
          train_crop_shapes,
          learning_rates,
          learning_schedule,
          augmentation_probabilities,
          augmentation_schedule,
          # Photometric data augmentations
          augmentation_random_brightness,
          augmentation_random_contrast,
          augmentation_random_gamma,
          augmentation_random_hue,
          augmentation_random_saturation,
          augmentation_random_gaussian_blur_kernel_size,
          augmentation_random_gaussian_blur_sigma_range,
          augmentation_random_noise_type,
          augmentation_random_noise_spread,
          # Geometric data augmentations
          augmentation_padding_mode,
          augmentation_random_crop_type,
          augmentation_random_flip_type,
          augmentation_random_rotate_max,
          augmentation_random_crop_and_pad,
          augmentation_random_resize_to_shape,
          augmentation_random_resize_and_pad,
          augmentation_random_resize_and_crop,
          # Occlusion data augmentations
          augmentation_random_remove_patch_percent_range_image,
          augmentation_random_remove_patch_size_image,
          augmentation_random_remove_patch_percent_range_depth,
          augmentation_random_remove_patch_size_depth,
          # Loss function settings
          supervision_type,
          w_losses,
          # TODO: Uncomment to use frozen model for loss
          frozen_model_paths,
          # Evaluation settings
          min_evaluate_depths,  # allows multiple val datasets
          max_evaluate_depths,  # allows multiple val datasets
          evaluation_protocol,
          # Checkpoint settings
          checkpoint_path,
          n_step_per_checkpoint,
          n_step_per_summary,
          n_image_per_summary,
          start_step_validation,
          restore_paths,
          # Hardware settings
          device='cuda',
          n_thread=8):

    # Select device to run on
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device: {}'.format(device))

    # Set up checkpoint and event paths
    os.makedirs(checkpoint_path, exist_ok=True)

    checkpoint_dirpath = os.path.join(checkpoint_path, 'checkpoints_{}'.format(model_name) + '-{}')
    log_path = os.path.join(checkpoint_path, 'results.txt')
    event_path = os.path.join(checkpoint_path, 'tensorboard')

    os.makedirs(os.path.join(event_path, 'events-train'), exist_ok=True)
    os.makedirs(os.path.join(event_path, 'events-val'), exist_ok=True)

    '''
    Read input paths and assert paths
    '''
    assert len(train_image_paths) == len(train_sparse_depth_paths)

    # Read training input paths
    train_image_paths_arr = [
        data_utils.read_paths(train_image_path)
        for train_image_path in train_image_paths
    ]

    n_train_samples = [
        len(paths) for paths in train_image_paths_arr
    ]

    train_sparse_depth_paths_arr = [
        data_utils.read_paths(train_sparse_depth_path)
        for train_sparse_depth_path in train_sparse_depth_paths
    ]

    # Make sure each set of paths have same number of samples
    for n_train_sample, sparse_depth_paths in zip(n_train_samples, train_sparse_depth_paths_arr):
        assert n_train_sample == len(sparse_depth_paths)

    # Read OPTIONAL ground truth paths
    if train_ground_truth_paths is not None and len(train_ground_truth_paths) > 0:
        assert len(train_image_paths) == len(train_ground_truth_paths)

        train_ground_truth_paths_arr = [
            data_utils.read_paths(train_ground_truth_path)
            for train_ground_truth_path in train_ground_truth_paths
        ]

        for n_train_sample, ground_truth_paths in zip(n_train_samples, train_ground_truth_paths_arr):
            assert n_train_sample == len(ground_truth_paths)

        is_available_ground_truth = True
    else:
        train_ground_truth_paths_arr = [
            [None] * n_train_sample
            for n_train_sample in n_train_samples
        ]

        is_available_ground_truth = False

    # Read OPTIONAL intrinsics input paths
    if train_intrinsics_paths is not None and len(train_intrinsics_paths) > 0:
        assert len(train_image_paths) == len(train_intrinsics_paths)

        train_intrinsics_paths_arr = [
            data_utils.read_paths(train_intrinsics_path)
            for train_intrinsics_path in train_intrinsics_paths
        ]

        for n_train_sample, intrinsics_paths in zip(n_train_samples, train_intrinsics_paths_arr):
            assert n_train_sample == len(intrinsics_paths)
    else:
        train_intrinsics_paths_arr = [
            [None] * n_train_sample
            for n_train_sample in n_train_samples
        ]

    '''
    Setup training dataloaders
    '''    
    # Get number of train samples and training step
    # Note: zipping up iterators will pad based on largest one
    max_train_sample = max(n_train_samples)

    n_multiplier_sample_padding_arr = [
        max_train_sample // n_train_sample
        for n_train_sample in n_train_samples
    ]

    n_remainder_sample_padding_arr = [
        max_train_sample % n_train_sample
        for n_train_sample in n_train_samples
    ]

    n_train_sample = max_train_sample * len(n_train_samples)

    # Make sure batch size is divisible by datasets
    n_dataset = len(train_image_paths_arr)
    assert train_batch_size % n_dataset == 0

    n_train_step = \
        learning_schedule[-1] * np.floor(n_train_sample / train_batch_size).astype(np.int32)

    # Crop shapes are defined for each dataset
    assert len(train_crop_shapes) % 2 == 0 and len(train_crop_shapes) // 2 == n_dataset

    # Set up batch size and crop shape for each dataset
    batch_size = train_batch_size // n_dataset
    train_batch_sizes_arr = [batch_size] * n_dataset

    train_crop_shapes_arr = [
        (height, width)
        for height, width in zip(train_crop_shapes[::2], train_crop_shapes[1::2])
    ]

    n_step_per_epoch = max_train_sample // batch_size

    # Set up data augmentations
    train_transforms_geometric = Transforms(
        random_flip_type=augmentation_random_flip_type,
        random_rotate_max=augmentation_random_rotate_max,
        random_crop_and_pad=augmentation_random_crop_and_pad,
        random_resize_to_shape=augmentation_random_resize_to_shape,
        random_resize_and_pad=augmentation_random_resize_and_pad,
        random_resize_and_crop=augmentation_random_resize_and_crop)

    train_transforms_photometric = Transforms(
        random_brightness=augmentation_random_brightness,
        random_contrast=augmentation_random_contrast,
        random_gamma=augmentation_random_gamma,
        random_hue=augmentation_random_hue,
        random_saturation=augmentation_random_saturation,
        random_gaussian_blur_kernel_size=augmentation_random_gaussian_blur_kernel_size,
        random_gaussian_blur_sigma_range=augmentation_random_gaussian_blur_sigma_range,
        random_noise_type=augmentation_random_noise_type,
        random_noise_spread=augmentation_random_noise_spread,
        random_remove_patch_percent_range=augmentation_random_remove_patch_percent_range_image,
        random_remove_patch_size=augmentation_random_remove_patch_size_image)

    train_transforms_point_cloud = Transforms(
        random_remove_patch_percent_range=augmentation_random_remove_patch_percent_range_depth,
        random_remove_patch_size=augmentation_random_remove_patch_size_depth)

    # Map interpolation mode names to enums
    # Augmentation for image, sparse depth, validity map, ground truth
    interpolation_modes = \
        train_transforms_geometric.map_interpolation_mode_names_to_enums(
            ['bilinear', 'nearest', 'nearest', 'nearest'])

    padding_modes = ['edge', 'constant', 'constant', 'constant']

    # TODO: Load replay data if it is available
    if supervision_type == "unsupervised":
        is_available_replay = \
            replay_image_paths is not None and \
            replay_sparse_depth_paths is not None
    else:
        is_available_replay = \
            replay_image_paths is not None and \
            replay_sparse_depth_paths is not None and \
            replay_ground_truth_paths is not None

    calculate_fisher_enabled = 'fisher' in network_modules

    if is_available_replay:
        '''
        Read input paths and assert paths
        '''
        assert len(replay_image_paths) == len(replay_sparse_depth_paths)

        # Read training input paths
        replay_image_paths_arr = [
            data_utils.read_paths(replay_image_path)
            for replay_image_path in replay_image_paths
        ]

        n_replay_samples = [
            len(paths) for paths in replay_image_paths_arr
        ]

        replay_sparse_depth_paths_arr = [
            data_utils.read_paths(replay_sparse_depth_path)
            for replay_sparse_depth_path in replay_sparse_depth_paths
        ]

        # Make sure each set of paths have same number of samples
        for n_replay_sample, sparse_depth_paths in zip(n_replay_samples, replay_sparse_depth_paths_arr):
            assert n_replay_sample == len(sparse_depth_paths)

        # Read optional ground truth paths
        if replay_ground_truth_paths is not None and len(replay_ground_truth_paths) > 0:
            assert len(replay_image_paths) == len(replay_ground_truth_paths)

            replay_ground_truth_paths_arr = [
                data_utils.read_paths(replay_ground_truth_path)
                for replay_ground_truth_path in replay_ground_truth_paths
            ]

            for n_replay_sample, ground_truth_paths in zip(n_replay_samples, replay_ground_truth_paths_arr):
                assert n_replay_sample == len(ground_truth_paths)

            is_available_ground_truth = True
        else:
            replay_ground_truth_paths_arr = [
                [None] * n_replay_sample
                for n_replay_sample in n_replay_samples
            ]

            is_available_ground_truth = False

        # Read optional intrinsics input paths
        if replay_intrinsics_paths is not None and len(replay_intrinsics_paths) > 0:
            assert len(replay_image_paths) == len(replay_intrinsics_paths)

            replay_intrinsics_paths_arr = [
                data_utils.read_paths(replay_intrinsics_path)
                for replay_intrinsics_path in replay_intrinsics_paths
            ]

            for n_replay_sample, intrinsics_paths in zip(n_replay_samples, replay_intrinsics_paths_arr):
                assert n_replay_sample == len(intrinsics_paths)
        else:
            replay_intrinsics_paths_arr = [
                [None] * n_replay_sample
                for n_replay_sample in n_replay_samples
            ]

        '''
        Setup replaying dataloader
        '''

        # Sample replay datasets down to replay_dataset_size
        for n_replay_sample in n_replay_samples:
            assert replay_dataset_size <= n_replay_sample

        replay_input_paths_arr = zip(
            replay_image_paths_arr,
            replay_sparse_depth_paths_arr,
            replay_intrinsics_paths_arr,
            replay_ground_truth_paths_arr)

        truncated_replay_image_paths_arr = []
        truncated_replay_sparse_depth_paths_arr = []
        truncated_replay_intrinsics_paths_arr = []
        truncated_replay_ground_truth_paths_arr = []

        for inputs in replay_input_paths_arr:
            # Unpack for each dataset
            image_paths, \
                sparse_depth_paths, \
                intrinsics_paths, \
                ground_truth_paths = inputs

            # Compute indices to select 
            idx_replay_samples = np.random.permutation(range(len(image_paths)))[:replay_dataset_size]

            truncated_replay_image_paths_arr.append((np.array(image_paths)[idx_replay_samples]).tolist())
            truncated_replay_sparse_depth_paths_arr.append((np.array(sparse_depth_paths)[idx_replay_samples]).tolist())
            truncated_replay_intrinsics_paths_arr.append((np.array(intrinsics_paths)[idx_replay_samples]).tolist())
            truncated_replay_ground_truth_paths_arr.append((np.array(ground_truth_paths)[idx_replay_samples]).tolist())

        replay_image_paths_arr = truncated_replay_image_paths_arr
        replay_sparse_depth_paths_arr = truncated_replay_sparse_depth_paths_arr
        replay_intrinsics_paths_arr = truncated_replay_intrinsics_paths_arr
        replay_ground_truth_paths_arr = truncated_replay_ground_truth_paths_arr

        replay_multiplier_sample_padding_arr = [
            (n_step_per_epoch * replay_batch_size) // replay_dataset_size
            for n_replay_sample in n_replay_samples
        ]

        replay_remainder_sample_padding_arr = [
            (n_step_per_epoch * replay_batch_size) % replay_dataset_size
            for n_replay_sample in n_replay_samples
        ]

        # Make sure batch size is divisible by datasets
        n_dataset = len(replay_image_paths_arr)
        assert replay_batch_size % n_dataset == 0

        # Crop shapes are defined for each dataset
        assert len(replay_crop_shapes) % 2 == 0 and len(replay_crop_shapes) // 2 == n_dataset

        # Set up batch size and crop shape for each dataset
        batch_size = replay_batch_size // n_dataset
        replay_batch_sizes_arr = [batch_size] * n_dataset

        replay_crop_shapes_arr = [
            (height, width)
            for height, width in zip(replay_crop_shapes[::2], replay_crop_shapes[1::2])
        ]

    # Load validation data if it is available
    is_available_validation = \
        val_image_paths is not None and \
        val_sparse_depth_paths is not None and \
        val_ground_truth_paths is not None

    if is_available_validation:

        # Extended validation setup to handle multiple datasets (similar to training above)
        # This is so that we can gauge performance on current and previous datasets
        # We will keep the validation dataloaders in a list and
        # iterate through them during validation step

        # Images <=> Sparse depths
        assert len(val_image_paths) == len(val_sparse_depth_paths)

        # Read val input paths
        val_image_paths_arr = [
            data_utils.read_paths(val_image_path)
            for val_image_path in val_image_paths
        ]

        n_val_samples = [
            len(paths) for paths in val_image_paths_arr
        ]

        val_sparse_depth_paths_arr = [
            data_utils.read_paths(val_sparse_depth_path)
            for val_sparse_depth_path in val_sparse_depth_paths
        ]

        # Make sure each set of paths have same number of samples
        for n_val_sample, sparse_depth_paths in zip(n_val_samples, val_sparse_depth_paths_arr):
            assert n_val_sample == len(sparse_depth_paths)

        # Images <=> Ground truths
        assert len(val_image_paths) == len(val_ground_truth_paths)

        val_ground_truth_paths_arr = [
            data_utils.read_paths(val_ground_truth_path)
            for val_ground_truth_path in val_ground_truth_paths
        ]

        # Make sure each set of paths have same number of samples
        for n_val_sample, ground_truth_paths in zip(n_val_samples, val_ground_truth_paths_arr):
            assert n_val_sample == len(ground_truth_paths)

        # Images <=>? Intrinsics
        if val_intrinsics_paths is not None and len(val_intrinsics_paths) > 0:
            assert len(val_image_paths) == len(val_intrinsics_paths)

            val_intrinsics_paths_arr = [
                data_utils.read_paths(val_intrinsics_path)
                for val_intrinsics_path in val_intrinsics_paths
            ]

            # Make sure each set of paths have same number of samples
            for n_val_sample, intrinsics_paths in zip(n_val_samples, val_intrinsics_paths_arr):
                assert n_val_sample == len(intrinsics_paths)
        else:
            val_intrinsics_paths_arr = [
                [None] * n_val_sample
                for n_val_sample in n_val_samples
            ]

        '''
        Setup validation dataloaders
        '''
        val_dataloaders = []

        val_input_paths_arr = zip(
            val_image_paths_arr,
            val_sparse_depth_paths_arr,
            val_intrinsics_paths_arr,
            val_ground_truth_paths_arr)

        # For each dataset
        for inputs in val_input_paths_arr:

            # Unpack for each dataset
            image_paths, \
                sparse_depth_paths, \
                intrinsics_paths, \
                ground_truth_paths = inputs

            val_dataloader = torch.utils.data.DataLoader(
                datasets.DepthCompletionInferenceDataset(
                    image_paths=image_paths,
                    sparse_depth_paths=sparse_depth_paths,
                    intrinsics_paths=intrinsics_paths,
                    ground_truth_paths=ground_truth_paths,
                    load_image_triplets=False),
                batch_size=1,
                shuffle=False,
                num_workers=1,
                drop_last=False)

            val_dataloaders.append(val_dataloader)

        # Moved here because we need to know how many val datasets
        best_results = {
            'step': [-1] * len(val_dataloaders),
            'mae': [np.infty] * len(val_dataloaders),
            'rmse': [np.infty] * len(val_dataloaders),
            'imae': [np.infty] * len(val_dataloaders),
            'irmse': [np.infty] * len(val_dataloaders)
        }

    '''
    Set up the model
    '''
    # Build depth completion network
    depth_completion_model = DepthCompletionModel(
        model_name=model_name,
        network_modules=network_modules,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    parameters_depth_model = depth_completion_model.parameters_depth()

    if supervision_type == 'unsupervised':
        parameters_pose_model = depth_completion_model.parameters_pose()
    else:
        parameters_pose_model = []

    depth_completion_model.train()

    # TODO: If using loss based (e.g. EWC or LWF) then create another instance of the model
    # Also will need to introduce an argument for restoring weights from previous dataset

    if len(frozen_model_paths) > 0:
        frozen_model =  DepthCompletionModel(
            model_name=model_name,
            network_modules=network_modules,
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            device=device)

        frozen_model.restore_model(frozen_model_paths, frozen_model=True)
        frozen_model.eval()
    else:
        frozen_model = None

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    '''
    Log input paths
    '''
    log('Training input paths:', log_path)

    train_input_paths = [
        train_image_paths,
        train_sparse_depth_paths,
        train_intrinsics_paths
    ]

    if is_available_ground_truth:
        train_input_paths.append(train_ground_truth_paths)
    else:
        train_input_paths.append([None] * len(train_image_paths))

    for dataset_id, paths in enumerate(zip(*train_input_paths)):

        log('dataset_id={}'.format(dataset_id))
        for path in paths:
            if path is not None:
                log(path, log_path)

    log('', log_path)

    # TODO: if replay filepaths is available then log paths
    if is_available_replay:
        log('Replay input paths:', log_path)
        replay_input_paths = [
            replay_image_paths,
            replay_sparse_depth_paths,
            replay_intrinsics_paths
        ]

        if is_available_ground_truth:
            replay_input_paths.append(replay_ground_truth_paths)
        else:
            replay_input_paths.append([None] * len(replay_image_paths))

        for dataset_id, paths in enumerate(zip(*replay_input_paths)):

            log('dataset_id={}'.format(dataset_id))
            for path in paths:
                if path is not None:
                    log(path, log_path)
        
        log('', log_path)

    # Added multiple dataset support
    if is_available_validation:
        log('Validation input paths:', log_path)
        val_input_paths = [
            val_image_paths,
            val_sparse_depth_paths,
            val_intrinsics_paths,
            val_ground_truth_paths
        ]

        for dataset_id, paths in enumerate(zip(*val_input_paths)):

            log('dataset_id={}'.format(dataset_id))
            for path in paths:
                if path is not None:
                    log(path, log_path)

        log('', log_path)

    log_network_settings(
        log_path,
        # Depth network settings
        model_name=model_name,
        network_modules=network_modules,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        parameters_depth_model=parameters_depth_model,
        parameters_pose_model=parameters_pose_model)

    log_training_settings(
        log_path,
        # Training settings
        train_batch_sizes=train_batch_sizes_arr,
        train_crop_shapes=train_crop_shapes_arr,
        n_train_sample=n_train_sample,
        n_train_step=n_train_step,
        learning_rates=learning_rates,
        learning_schedule=learning_schedule,
        # Replay settings
        is_available_replay=is_available_replay,
        replay_batch_size=replay_batch_size,
        replay_crop_shapes=replay_crop_shapes,
        replay_dataset_size=replay_dataset_size,
        # Augmentation settings
        augmentation_probabilities=augmentation_probabilities,
        augmentation_schedule=augmentation_schedule,
        # Photometric data augmentations
        augmentation_random_brightness=augmentation_random_brightness,
        augmentation_random_contrast=augmentation_random_contrast,
        augmentation_random_gamma=augmentation_random_gamma,
        augmentation_random_hue=augmentation_random_hue,
        augmentation_random_saturation=augmentation_random_saturation,
        augmentation_random_gaussian_blur_kernel_size=augmentation_random_gaussian_blur_kernel_size,
        augmentation_random_gaussian_blur_sigma_range=augmentation_random_gaussian_blur_sigma_range,
        augmentation_random_noise_type=augmentation_random_noise_type,
        augmentation_random_noise_spread=augmentation_random_noise_spread,
        # Geometric data augmentations
        augmentation_padding_mode=augmentation_padding_mode,
        augmentation_random_crop_type=augmentation_random_crop_type,
        augmentation_random_crop_to_shape=None,
        augmentation_reverse_crop_to_shape=None,
        augmentation_random_flip_type=augmentation_random_flip_type,
        augmentation_random_rotate_max=augmentation_random_rotate_max,
        augmentation_random_crop_and_pad=augmentation_random_crop_and_pad,
        augmentation_random_resize_to_shape=augmentation_random_resize_to_shape,
        augmentation_random_resize_and_pad=augmentation_random_resize_and_pad,
        augmentation_random_resize_and_crop=augmentation_random_resize_and_crop,
        # Occlusion data augmentations
        augmentation_random_remove_patch_percent_range_image=augmentation_random_remove_patch_percent_range_image,
        augmentation_random_remove_patch_size_image=augmentation_random_remove_patch_size_image,
        augmentation_random_remove_patch_percent_range_depth=augmentation_random_remove_patch_percent_range_depth,
        augmentation_random_remove_patch_size_depth=augmentation_random_remove_patch_size_depth)

    log_loss_func_settings(
        log_path,
        # Loss function settings
        supervision_type=supervision_type,
        w_losses=w_losses)

    log_evaluation_settings(
        log_path,
        min_evaluate_depths=min_evaluate_depths,
        max_evaluate_depths=max_evaluate_depths,
        evaluation_protocol=evaluation_protocol)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=checkpoint_path,
        n_step_per_checkpoint=n_step_per_checkpoint,
        summary_event_path=event_path,
        n_step_per_summary=n_step_per_summary,
        n_image_per_summary=n_image_per_summary,
        start_step_validation=start_step_validation,
        restore_paths=restore_paths,
        # Hardware settings
        device=device,
        n_thread=n_thread)

    # Set up tensorboard summary writers
    train_summary_writer = SummaryWriter(event_path + '-train')
    val_summary_writer = SummaryWriter(event_path + '-val')

    '''
    Train model
    '''
    # Initialize optimizer with starting learning rate
    learning_schedule_pos = 0
    learning_rate = learning_rates[0]

    augmentation_schedule_pos = 0
    augmentation_probability = augmentation_probabilities[0]

    w_weight_decay_depth = \
        w_losses['w_weight_decay_depth'] if 'w_weight_decay_depth' in w_losses else 0.0
    w_weight_decay_pose = \
         w_losses['w_weight_decay_pose'] if 'w_weight_decay_pose' in w_losses else 0.0

    # TODO: Set allow depth completion model to specify optimizer as a pointer
    optimizer_depth = torch.optim.Adam([
        {
            'params' : parameters_depth_model,
            'weight_decay' : w_weight_decay_depth
        }],
        lr=learning_rate)

    if supervision_type == 'unsupervised':
        optimizer_pose = torch.optim.Adam([
            {
                'params' : parameters_pose_model,
                'weight_decay' : w_weight_decay_pose
            }],
            lr=learning_rate)
    else:
        optimizer_pose = None

    # Start training
    depth_completion_model.train()

    train_step = 0

    if len(restore_paths) > 0:
        try:
            train_step, optimizer_depth, optimizer_pose = depth_completion_model.restore_model(
                restore_paths,
                optimizer_depth=optimizer_depth,
                optimizer_pose=optimizer_pose)
        except Exception:
            print('Failed to restore optimizer for depth network: Ignoring...')
            train_step = depth_completion_model.restore_model(
                restore_paths)

        for g in optimizer_depth.param_groups:
            g['lr'] = learning_rate

        n_train_step = n_train_step + train_step

    time_start = time.time()

    log('Begin training...', log_path)

    for epoch in range(1, learning_schedule[-1] + 1):

        # Set learning rate schedule
        if epoch > learning_schedule[learning_schedule_pos]:
            learning_schedule_pos = learning_schedule_pos + 1
            learning_rate = learning_rates[learning_schedule_pos]

            # Update optimizer learning rates for depth network
            for g in optimizer_depth.param_groups:
                g['lr'] = learning_rate

            if supervision_type == 'unsupervised':
                # Update optimizer learning rates for pose network
                for g in optimizer_pose.param_groups:
                    g['lr'] = learning_rate

        # Set augmentation schedule
        if -1 not in augmentation_schedule and epoch > augmentation_schedule[augmentation_schedule_pos]:
            augmentation_schedule_pos = augmentation_schedule_pos + 1
            augmentation_probability = augmentation_probabilities[augmentation_schedule_pos]

        # Pad all datasets
        train_image_paths_arr_epoch = []
        train_sparse_depth_paths_arr_epoch = []
        train_intrinsics_paths_arr_epoch = []
        train_ground_truth_paths_arr_epoch = []
        train_batch_sizes_arr_epoch = train_batch_sizes_arr
        train_crop_shapes_arr_epoch = train_crop_shapes_arr

        train_input_paths_arr = zip(
            train_image_paths_arr,
            train_sparse_depth_paths_arr,
            train_intrinsics_paths_arr,
            train_ground_truth_paths_arr,
            n_multiplier_sample_padding_arr,
            n_remainder_sample_padding_arr)

        for inputs in train_input_paths_arr:
            # Unpack for each dataset
            image_paths, \
                sparse_depth_paths, \
                intrinsics_paths, \
                ground_truth_paths, \
                multiplier_sample_padding,\
                remainder_sample_padding = inputs
            
            # Compute indices to select remainder for this epoch
            idx_remainder = np.random.permutation(range(len(image_paths)))[:remainder_sample_padding]

            # Extend image paths
            image_paths_epoch = image_paths + \
                image_paths * (multiplier_sample_padding - 1) + \
                (np.array(image_paths)[idx_remainder]).tolist()

            # Extend sparse depth paths
            sparse_depth_paths_epoch = sparse_depth_paths + \
                sparse_depth_paths * (multiplier_sample_padding - 1) + \
                (np.array(sparse_depth_paths)[idx_remainder]).tolist()

            # Extend intrinsics paths
            intrinsics_paths_epoch = intrinsics_paths + \
                intrinsics_paths * (multiplier_sample_padding - 1) + \
                (np.array(intrinsics_paths)[idx_remainder]).tolist()
            
            # Extend ground truth paths
            ground_truth_paths_epoch = ground_truth_paths + \
                ground_truth_paths * (multiplier_sample_padding - 1) + \
                (np.array(ground_truth_paths)[idx_remainder]).tolist()
            
            # Append extended paths for each dataset
            train_image_paths_arr_epoch.append(image_paths_epoch)
            train_sparse_depth_paths_arr_epoch.append(sparse_depth_paths_epoch)
            train_intrinsics_paths_arr_epoch.append(intrinsics_paths_epoch)
            train_ground_truth_paths_arr_epoch.append(ground_truth_paths_epoch)

        train_input_paths_arr_epoch = zip(
            train_image_paths_arr_epoch,
            train_sparse_depth_paths_arr_epoch,
            train_intrinsics_paths_arr_epoch,
            train_ground_truth_paths_arr_epoch,
            train_batch_sizes_arr_epoch,
            train_crop_shapes_arr_epoch)
        
        train_dataloaders = []
        # For each dataset
        for inputs in train_input_paths_arr_epoch:

            # Unpack for each dataset
            image_paths, \
                sparse_depth_paths, \
                intrinsics_paths, \
                ground_truth_paths, \
                batch_size, \
                crop_shape = inputs

            if supervision_type == 'supervised':
                train_dataset = datasets.DepthCompletionSupervisedTrainingDataset(
                    image_paths=image_paths,
                    sparse_depth_paths=sparse_depth_paths,
                    intrinsics_paths=intrinsics_paths,
                    ground_truth_paths=ground_truth_paths,
                    random_crop_shape=crop_shape,
                    random_crop_type=augmentation_random_crop_type)
            elif supervision_type == 'unsupervised':
                train_dataset = datasets.DepthCompletionMonocularTrainingDataset(
                    images_paths=image_paths,
                    sparse_depth_paths=sparse_depth_paths,
                    intrinsics_paths=intrinsics_paths,
                    random_crop_shape=crop_shape,
                    random_crop_type=augmentation_random_crop_type)
            else:
                raise ValueError('Unsupported supervision type: {}'.format(supervision_type))
            

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=n_thread,
                pin_memory=False,
                drop_last=True)

            train_dataloaders.append(train_dataloader)

        # TODO: If replay is available, incorporate it to training loop
        if is_available_replay:
            # TODO: One design is to add it to the list of train dataloaders at the setup stage
            # Another is to have it in a separate loop to allow for separate logging

            # Pad all datasets
            replay_image_paths_arr_epoch = []
            replay_sparse_depth_paths_arr_epoch = []
            replay_intrinsics_paths_arr_epoch = []
            replay_ground_truth_paths_arr_epoch = []
            replay_batch_sizes_arr_epoch = replay_batch_sizes_arr
            replay_crop_shapes_arr_epoch = replay_crop_shapes_arr

            replay_input_paths_arr = zip(
                replay_image_paths_arr,
                replay_sparse_depth_paths_arr,
                replay_intrinsics_paths_arr,
                replay_ground_truth_paths_arr,
                replay_multiplier_sample_padding_arr,
                replay_remainder_sample_padding_arr)

            for inputs in replay_input_paths_arr:
                # Unpack for each dataset
                image_paths, \
                    sparse_depth_paths, \
                    intrinsics_paths, \
                    ground_truth_paths, \
                    multiplier_sample_padding,\
                    remainder_sample_padding = inputs
                
                # Compute indices to select remainder for this epoch
                idx_remainder = np.random.permutation(range(len(image_paths)))[:remainder_sample_padding]

                # Extend image paths
                image_paths_epoch = image_paths + \
                    image_paths * (multiplier_sample_padding - 1) + \
                    (np.array(image_paths)[idx_remainder]).tolist()

                # Extend sparse depth paths
                sparse_depth_paths_epoch = sparse_depth_paths + \
                    sparse_depth_paths * (multiplier_sample_padding - 1) + \
                    (np.array(sparse_depth_paths)[idx_remainder]).tolist()

                # Extend intrinsics paths
                intrinsics_paths_epoch = intrinsics_paths + \
                    intrinsics_paths * (multiplier_sample_padding - 1) + \
                    (np.array(intrinsics_paths)[idx_remainder]).tolist()
                
                # Extend ground truth paths
                ground_truth_paths_epoch = ground_truth_paths + \
                    ground_truth_paths * (multiplier_sample_padding - 1) + \
                    (np.array(ground_truth_paths)[idx_remainder]).tolist()
                
                # Append extended paths for each dataset
                replay_image_paths_arr_epoch.append(image_paths_epoch)
                replay_sparse_depth_paths_arr_epoch.append(sparse_depth_paths_epoch)
                replay_intrinsics_paths_arr_epoch.append(intrinsics_paths_epoch)
                replay_ground_truth_paths_arr_epoch.append(ground_truth_paths_epoch)

            replay_input_paths_arr_epoch = zip(
                replay_image_paths_arr_epoch,
                replay_sparse_depth_paths_arr_epoch,
                replay_intrinsics_paths_arr_epoch,
                replay_ground_truth_paths_arr_epoch,
                replay_batch_sizes_arr_epoch,
                replay_crop_shapes_arr_epoch)

            for inputs in replay_input_paths_arr_epoch:

                # Unpack for each dataset
                image_paths, \
                    sparse_depth_paths, \
                    intrinsics_paths, \
                    ground_truth_paths, \
                    batch_size, \
                    crop_shape = inputs

                if supervision_type == 'supervised':
                    replay_dataset = datasets.DepthCompletionSupervisedTrainingDataset(
                        image_paths=image_paths,
                        sparse_depth_paths=sparse_depth_paths,
                        intrinsics_paths=intrinsics_paths,
                        ground_truth_paths=ground_truth_paths,
                        random_crop_shape=crop_shape,
                        random_crop_type=augmentation_random_crop_type)
                elif supervision_type == 'unsupervised':
                    replay_dataset = datasets.DepthCompletionMonocularTrainingDataset(
                        images_paths=image_paths,
                        sparse_depth_paths=sparse_depth_paths,
                        intrinsics_paths=intrinsics_paths,
                        random_crop_shape=crop_shape,
                        random_crop_type=augmentation_random_crop_type)
                else:
                    raise ValueError('Unsupported supervision type: {}'.format(supervision_type))

                replay_dataloader = torch.utils.data.DataLoader(
                    replay_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=n_thread,
                    pin_memory=False,
                    drop_last=True)

                train_dataloaders.append(replay_dataloader)
        
        # Zip all dataloaders together to get batches from each
        train_dataloaders_epoch = tqdm.tqdm(
            zip(*train_dataloaders),
            desc='Epoch: {}/{}  Batch'.format(epoch, learning_schedule[-1]),
            total=n_step_per_epoch)

        # Each train_batches is a n_dataset-length tuple with one batch from each dataset
        for train_batches in train_dataloaders_epoch:
            train_step = train_step + 1
            loss = 0.0
            loss_info = {}

            '''
            Iterate over batches from different datasets
            '''
            for dataset_id, train_batch in enumerate(train_batches):

                # Fetch data
                train_batch = [
                    in_.to(device) for in_ in train_batch
                ]

                if supervision_type == 'supervised':
                    image0, \
                        sparse_depth0, \
                        intrinsics, \
                        ground_truth0 = train_batch

                    image1 = image0.detach().clone()
                    image2 = image0.detach().clone()
                elif supervision_type == 'unsupervised':
                    image0, \
                        image1, \
                        image2, \
                        sparse_depth0, \
                        intrinsics = train_batch

                    ground_truth0 = None
                else:
                    raise ValueError('Unsupported supervision type: {}'.format(supervision_type))

                # Validity map is where sparse depth is available
                validity_map0 = torch.where(
                    sparse_depth0 > 0,
                    torch.ones_like(sparse_depth0),
                    sparse_depth0)

                # Perform geometric augmentation i.e. crop, flip, etc. on the input image
                [input_image0, input_sparse_depth0, input_validity_map0], \
                    [input_intrinsics], \
                    transform_performed_geometric = train_transforms_geometric.transform(
                        images_arr=[image0, sparse_depth0, validity_map0],
                        intrinsics_arr=[intrinsics],
                        padding_modes=padding_modes,
                        interpolation_modes=interpolation_modes,
                        random_transform_probability=augmentation_probability)

                # TODO: Refactor this as a function inside transforms
                # Erode to prevent block artifacts from resampling
                if 'random_resize_to_shape' in transform_performed_geometric:
                    _, factor = transform_performed_geometric['random_resize_to_shape']

                    # Erode if factor greater than 1
                    if factor > 1:
                        erosion_kernel = torch.tensor([
                            [1., 1., 1.],
                            [1., 1., 1.],
                            [1., 1., 1.]], device=device)

                        erosion_map0 = torch.nn.functional.conv2d(
                            input_validity_map0,
                            erosion_kernel.view(1, 1, 3, 3),
                            padding='same')

                        # Keep single point
                        single_point_map0 = torch.where(
                            erosion_map0 == 1,
                            torch.ones_like(erosion_map0),
                            torch.zeros_like(erosion_map0))

                        # Erode multiple points
                        multi_point_map0 = torch.where(
                            erosion_map0 > 9,
                            torch.ones_like(erosion_map0),
                            torch.zeros_like(erosion_map0))

                        input_validity_map0 = torch.where(
                            single_point_map0 + multi_point_map0 > 0,
                            torch.ones_like(input_validity_map0),
                            torch.zeros_like(input_validity_map0))

                        input_sparse_depth0 = input_sparse_depth0 * input_validity_map0

                # Perform point removal from sparse depth
                [input_sparse_depth0], _ = train_transforms_point_cloud.transform(
                    images_arr=[input_sparse_depth0],
                    random_transform_probability=augmentation_probability)

                # Perform photometric augmentation i.e. masking, brightness, contrast, etc. on the input image
                [input_image0], _ = train_transforms_photometric.transform(
                    images_arr=[input_image0],
                    random_transform_probability=augmentation_probability)

                '''
                Forward through the network and compute loss
                '''
                # Inputs: augmented image, augmented sparse depth map, original (but aligned) validity map
                output_depth0 = depth_completion_model.forward_depth(
                    image=input_image0,
                    sparse_depth=input_sparse_depth0,
                    validity_map=input_validity_map0,
                    intrinsics=input_intrinsics,
                    return_all_outputs=True)

                if supervision_type == 'unsupervised':
                    pose0to1 = depth_completion_model.forward_pose(image0, image1)
                    pose0to2 = depth_completion_model.forward_pose(image0, image2)
                else:
                    pose0to1 = None
                    pose0to2 = None

                # For visualization
                if (train_step % n_step_per_summary) == 0:
                    output_depth0_initial = output_depth0[0].detach().clone()

                output_depth0, validity_map_image0 = train_transforms_geometric.reverse_transform(
                    images_arr=output_depth0,
                    transform_performed=transform_performed_geometric,
                    return_all_outputs=True,
                    padding_modes=[padding_modes[0]])

                # Compute loss function
                validity_map_depth0 = validity_map0

                # TODO: Add argument to allow frozen model to be passed in
                loss_batch, loss_info_batch = depth_completion_model.compute_loss(
                    image0=image0,
                    image1=image1,
                    image2=image2,
                    output_depth0=output_depth0,
                    sparse_depth0=sparse_depth0,
                    validity_map_depth0=validity_map_depth0,
                    validity_map_image0=validity_map_image0,
                    ground_truth0=ground_truth0,
                    intrinsics=intrinsics,
                    pose0to1=pose0to1,
                    pose0to2=pose0to2,
                    supervision_type=supervision_type,
                    w_losses=w_losses,
                    frozen_model=frozen_model)

                # Accumulate loss over batches and update loss info
                loss = loss + loss_batch

                '''
                Log training summary
                '''
                if (train_step % n_step_per_summary) == 0:

                    if supervision_type == 'unsupervised':
                        image1to0 = loss_info_batch.pop('image1to0')
                        image2to0 = loss_info_batch.pop('image2to0')
                    else:
                        image1to0 = image0
                        image2to0 = image0

                    for key, value in loss_info_batch.items():
                        if key in loss_info:
                            loss_info[key] = loss_info[key] + value
                        else:
                            loss_info[key] = value

                    # Log summary
                    depth_completion_model.log_summary(
                        summary_writer=train_summary_writer,
                        tag='inputs' + '-{}'.format(dataset_id),
                        step=train_step,
                        image0=input_image0,
                        output_depth0=output_depth0_initial.detach().clone(),
                        sparse_depth0=input_sparse_depth0,
                        validity_map0=input_validity_map0,
                        n_image_per_summary=min(batch_size, n_image_per_summary))

                    depth_completion_model.log_summary(
                        summary_writer=train_summary_writer,
                        tag='train' + '-{}'.format(dataset_id),
                        step=train_step,
                        image0=image0,
                        image1to0=image1to0.detach().clone(),
                        image2to0=image2to0.detach().clone(),
                        output_depth0=output_depth0[0].detach().clone(),
                        sparse_depth0=sparse_depth0,
                        validity_map0=validity_map0,
                        ground_truth0=ground_truth0,
                        pose0to1=pose0to1,
                        pose0to2=pose0to2,
                        scalars=loss_info_batch,
                        n_image_per_summary=min(batch_size, n_image_per_summary))

            '''
            Compute gradient and backpropagate
            '''
            optimizer_depth.zero_grad()

            if supervision_type == 'unsupervised':
                optimizer_pose.zero_grad()

            loss.backward()

            optimizer_depth.step()

            if supervision_type == 'unsupervised':
                optimizer_pose.step()

            # Compute fisher information for EWC
            if calculate_fisher_enabled:
                depth_completion_model.calculate_fisher(normalization=len(train_dataloaders[0].dataset))

            '''
            Log results and save checkpoints
            '''
            if (train_step % n_step_per_summary) == 0:

                depth_completion_model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train' ,
                    step=train_step,
                    scalars=loss_info_batch)

            if (train_step % n_step_per_checkpoint) == 0:

                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step) * time_elapse / train_step

                log('Step={:6}/{}  Loss={:.5f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step, n_train_step, loss.item(), time_elapse, time_remain),
                    log_path)

                if train_step >= start_step_validation and is_available_validation:
                    # Switch to validation mode
                    depth_completion_model.eval()

                    # Added support for validating multiple datasets with a loop over validation dataloaders

                    with torch.no_grad():
                        # Perform validation
                        best_results = validate(
                            depth_model=depth_completion_model,
                            dataloaders=val_dataloaders,  # multiple dataloaders
                            step=train_step,
                            best_results=best_results,
                            min_evaluate_depths=min_evaluate_depths,
                            max_evaluate_depths=max_evaluate_depths,
                            evaluation_protocol=evaluation_protocol,
                            device=device,
                            summary_writer=val_summary_writer,
                            n_image_per_summary=n_image_per_summary,
                            log_path=log_path)

                    # Switch back to training
                    depth_completion_model.train()

                # Save checkpoints
                depth_completion_model.save_model(
                    checkpoint_dirpath.format(train_step),
                    train_step,
                    optimizer_depth,
                    optimizer_pose)

        # update fisher at the end of epoch
        if calculate_fisher_enabled:
            depth_completion_model.update_fisher()


    '''
    Perform validation for final step and save checkpoint
    '''
    depth_completion_model.eval()

    with torch.no_grad():
        best_results = validate(
            depth_model=depth_completion_model,
            dataloaders=val_dataloaders,
            step=train_step,
            best_results=best_results,
            min_evaluate_depths=min_evaluate_depths,
            max_evaluate_depths=max_evaluate_depths,
            evaluation_protocol=evaluation_protocol,
            device=device,
            summary_writer=val_summary_writer,
            n_image_per_summary=n_image_per_summary,
            log_path=log_path)

    # Save checkpoints
    depth_completion_model.save_model(
        checkpoint_dirpath.format(train_step),
        train_step,
        optimizer_depth,
        optimizer_pose)

def validate(depth_model,
             dataloaders,  # multiple dataloaders
             step,
             best_results,
             min_evaluate_depths,
             max_evaluate_depths,
             evaluation_protocol,
             device,
             summary_writer,
             n_image_per_summary=4,
             n_interval_per_summary=250,
             log_path=None):

    n_val_steps = min([len(dataloader) for dataloader in dataloaders])
    n_dataloaders = len(dataloaders)
    mae = np.zeros((n_dataloaders, n_val_steps))
    rmse = np.zeros((n_dataloaders, n_val_steps))
    imae = np.zeros((n_dataloaders, n_val_steps))
    irmse = np.zeros((n_dataloaders, n_val_steps))

    image_summary = [[] for _ in range(n_dataloaders)]
    output_depth_summary = [[] for _ in range(n_dataloaders)]
    sparse_depth_summary = [[] for _ in range(n_dataloaders)]
    validity_map_summary = [[] for _ in range(n_dataloaders)]
    ground_truth_summary = [[] for _ in range(n_dataloaders)]

    val_dataloaders_epoch = tqdm.tqdm(
        zip(*dataloaders),
        desc='Batch',
        total=n_val_steps)

    for idx, val_batches in enumerate(val_dataloaders_epoch):
        '''
        Iterate over batches from different datasets
        '''
        for dataset_id, val_batch in enumerate(val_batches):

            # Fetch data
            val_batch = [
                in_.to(device) for in_ in val_batch
            ]

            image, sparse_depth, intrinsics, ground_truth = val_batch

            with torch.no_grad():
                # Validity map is where sparse depth is available
                validity_map = torch.where(
                    sparse_depth > 0,
                    torch.ones_like(sparse_depth),
                    sparse_depth)

                # Forward through network
                output_depth = depth_model.forward_depth(
                    image=image,
                    sparse_depth=sparse_depth,
                    validity_map=validity_map,
                    intrinsics=intrinsics,
                    return_all_outputs=False)

            if (idx % n_interval_per_summary) == 0 and summary_writer is not None:
                image_summary[dataset_id].append(image)
                output_depth_summary[dataset_id].append(output_depth)
                sparse_depth_summary[dataset_id].append(sparse_depth)
                validity_map_summary[dataset_id].append(validity_map)
                ground_truth_summary[dataset_id].append(ground_truth)

            # Convert to numpy to validate
            output_depth = np.squeeze(output_depth.cpu().numpy())
            ground_truth = np.squeeze(ground_truth.cpu().numpy())

            if evaluation_protocol == 'vkitti':
                # Crop output_depth and ground_truth
                crop_height = 240
                crop_width = 1216
                crop_mask = [crop_height, crop_width]
            elif evaluation_protocol == 'nuscenes':
                # Crop output_depth and ground_truth
                crop_height = 540
                crop_width = 1600
                crop_mask = [crop_height, crop_width]
            else:
                crop_mask = None

            if crop_mask is not None:
                height, width = ground_truth.shape[-2], ground_truth.shape[-1]
                center = width // 2
                start_x = center - crop_width // 2
                end_x = center + crop_width // 2
                # bottom crop
                end_y = height
                start_y = end_y - crop_height
                output_depth = output_depth[start_y:end_y, start_x:end_x]
                ground_truth = ground_truth[start_y:end_y, start_x:end_x]

            # Select valid regions to evaluate
            validity_mask = np.where(ground_truth > 0, 1, 0)

            min_max_mask = np.logical_and(
                ground_truth > min_evaluate_depths[dataset_id],
                ground_truth < max_evaluate_depths[dataset_id])

            mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

            output_depth = output_depth[mask]
            ground_truth = ground_truth[mask]

            # Compute validation metrics
            mae[dataset_id, idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
            rmse[dataset_id, idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
            imae[dataset_id, idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
            irmse[dataset_id, idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

    # Compute mean metrics
    mae   = np.mean(mae, axis=1)
    rmse  = np.mean(rmse, axis=1)
    imae  = np.mean(imae, axis=1)
    irmse = np.mean(irmse, axis=1)

    # Log for each dataset:
    for dataset_id in range(n_dataloaders):

        # Log to tensorboard
        if summary_writer is not None:
            depth_model.log_summary(
                summary_writer=summary_writer,
                tag='eval' + '-{}'.format(dataset_id),
                step=step,
                image0=torch.cat(image_summary[dataset_id], dim=0),
                output_depth0=torch.cat(output_depth_summary[dataset_id], dim=0),
                sparse_depth0=torch.cat(sparse_depth_summary[dataset_id], dim=0),
                validity_map0=torch.cat(validity_map_summary[dataset_id], dim=0),
                ground_truth0=torch.cat(ground_truth_summary[dataset_id], dim=0),
                scalars={'mae' : mae[dataset_id],
                         'rmse' : rmse[dataset_id],
                         'imae' : imae[dataset_id],
                         'irmse': irmse[dataset_id]},
                n_image_per_summary=n_image_per_summary)

        # Print validation results to console
        log('Validation results for dataset {}:'.format(dataset_id), log_path)
        log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
            'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
            log_path)
        log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            step, mae[dataset_id], rmse[dataset_id], imae[dataset_id], irmse[dataset_id]),
            log_path)

        n_improve = 0
        if np.round(mae[dataset_id], 2) <= np.round(best_results['mae'][dataset_id], 2):
            n_improve = n_improve + 1
        if np.round(rmse[dataset_id], 2) <= np.round(best_results['rmse'][dataset_id], 2):
            n_improve = n_improve + 1
        if np.round(imae[dataset_id], 2) <= np.round(best_results['imae'][dataset_id], 2):
            n_improve = n_improve + 1
        if np.round(irmse[dataset_id], 2) <= np.round(best_results['irmse'][dataset_id], 2):
            n_improve = n_improve + 1

        if n_improve > 2:
            best_results['step'][dataset_id] = step
            best_results['mae'][dataset_id] = mae[dataset_id]
            best_results['rmse'][dataset_id] = rmse[dataset_id]
            best_results['imae'][dataset_id] = imae[dataset_id]
            best_results['irmse'][dataset_id] = irmse[dataset_id]

        log('Best results for dataset {}:'.format(dataset_id), log_path)
        log('{:>8}  {:>8}  {:>8}  {:>8}  {:>8}'.format(
            'Step', 'MAE', 'RMSE', 'iMAE', 'iRMSE'),
            log_path)
        log('{:8}  {:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            best_results['step'][dataset_id],
            best_results['mae'][dataset_id],
            best_results['rmse'][dataset_id],
            best_results['imae'][dataset_id],
            best_results['irmse'][dataset_id]), log_path)

    return best_results


# NOT set up for multiple val dataloaders
def run(image_path,
        sparse_depth_path,
        intrinsics_path,
        ground_truth_path,
        # Restore path settings
        restore_paths,
        # Input settings
        input_channels_image,
        input_channels_depth,
        normalized_image_range,
        # Depth network settings
        model_name,
        network_modules,
        min_predict_depth,
        max_predict_depth,
        # Evaluation settings
        min_evaluate_depth,
        max_evaluate_depth,
        evaluation_protocol,
        # Output settings
        output_path,
        save_outputs,
        keep_input_filenames,
        # Hardware settings
        device):

    # Select device to run on
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device: {}'.format(device))

    '''
    Set up output paths
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    log_path = os.path.join(output_path, 'results.txt')
    output_dirpath = os.path.join(output_path, 'outputs')

    if save_outputs:
        # Create output directories
        image_dirpath = os.path.join(output_dirpath, 'image')
        output_depth_dirpath = os.path.join(output_dirpath, 'output_depth')
        sparse_depth_dirpath = os.path.join(output_dirpath, 'sparse_depth')
        ground_truth_dirpath = os.path.join(output_dirpath, 'ground_truth')

        dirpaths = [
            output_dirpath,
            image_dirpath,
            output_depth_dirpath,
            sparse_depth_dirpath,
            ground_truth_dirpath
        ]

        for dirpath in dirpaths:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

    '''
    Load input paths and set up dataloader
    '''
    image_paths = data_utils.read_paths(image_path)
    sparse_depth_paths = data_utils.read_paths(sparse_depth_path)
    intrinsics_paths = data_utils.read_paths(intrinsics_path)

    is_available_ground_truth = False

    if ground_truth_path is not None and ground_truth_path != '':
        is_available_ground_truth = True
        ground_truth_paths = data_utils.read_paths(ground_truth_path)
    else:
        ground_truth_paths = None

    n_sample = len(image_paths)

    input_paths = [
        image_paths,
        sparse_depth_paths,
        intrinsics_paths
    ]

    if is_available_ground_truth:
        input_paths.append(ground_truth_paths)

    for paths in input_paths:
        assert n_sample == len(paths)

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(
        datasets.DepthCompletionInferenceDataset(
            image_paths=image_paths,
            sparse_depth_paths=sparse_depth_paths,
            intrinsics_paths=intrinsics_paths,
            ground_truth_paths=ground_truth_paths,
            load_image_triplets=False),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    '''
    Set up the model
    '''
    # Build depth completion network
    depth_model = DepthCompletionModel(
        model_name=model_name,
        network_modules=network_modules,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        device=device)

    # Restore model and set to evaluation mode
    depth_model.restore_model(restore_paths)
    depth_model.eval()

    parameters_depth_model = depth_model.parameters_depth()

    '''
    Log input paths
    '''
    log('Input paths:', log_path)
    input_paths = [
        image_path,
        sparse_depth_path,
        intrinsics_path,
    ]

    if is_available_ground_truth:
        input_paths.append(ground_truth_path)

    for path in input_paths:
        log(path, log_path)
    log('', log_path)

    '''
    Log all settings
    '''
    log_network_settings(
        log_path,
        # Depth network settings
        model_name=model_name,
        network_modules=network_modules,
        min_predict_depth=min_predict_depth,
        max_predict_depth=max_predict_depth,
        # Weight settings
        parameters_depth_model=parameters_depth_model)

    log_evaluation_settings(
        log_path,
        min_evaluate_depths=min_evaluate_depth,
        max_evaluate_depths=max_evaluate_depth,
        evaluation_protocol=evaluation_protocol)

    log_system_settings(
        log_path,
        # Checkpoint settings
        checkpoint_path=output_path,
        restore_paths=restore_paths,
        # Hardware settings
        device=device,
        n_thread=1)

    '''
    Run model
    '''
    # Set up metrics in case groundtruth is available
    mae = np.zeros(n_sample)
    rmse = np.zeros(n_sample)
    imae = np.zeros(n_sample)
    irmse = np.zeros(n_sample)

    time_elapse = 0.0

    for idx, inputs in enumerate(dataloader):

        # Move inputs to device
        inputs = [
            in_.to(device) for in_ in inputs
        ]

        if dataloader.dataset.is_available_ground_truth:
            image, sparse_depth, intrinsics, ground_truth = inputs
        else:
            image, sparse_depth, intrinsics = inputs

        time_start = time.time()

        with torch.no_grad():
            # Validity map is where sparse depth is available
            validity_map = torch.where(
                sparse_depth > 0,
                torch.ones_like(sparse_depth),
                sparse_depth)

            # Forward through network
            output_depth = depth_model.forward_depth(
                image=image,
                sparse_depth=sparse_depth,
                validity_map=validity_map,
                intrinsics=intrinsics,
                return_all_outputs=False)

        time_elapse = time_elapse + (time.time() - time_start)

        # Convert to numpy
        output_depth = np.squeeze(output_depth.detach().cpu().numpy())

        # Save to output
        if save_outputs:
            image = np.transpose(np.squeeze(image.cpu().numpy()), (1, 2, 0))
            sparse_depth = np.squeeze(sparse_depth.cpu().numpy())

            if keep_input_filenames:
                filename = os.path.splitext(os.path.basename(image_paths[idx]))[0] + '.png'
            else:
                filename = '{:010d}.png'.format(idx)

            image_path = os.path.join(image_dirpath, filename)
            image = (255 * image).astype(np.uint8)
            Image.fromarray(image).save(image_path)

            output_depth_path = os.path.join(output_depth_dirpath, filename)
            data_utils.save_depth(output_depth, output_depth_path)

            sparse_depth_path = os.path.join(sparse_depth_dirpath, filename)
            data_utils.save_depth(sparse_depth, sparse_depth_path)

        if is_available_ground_truth:

            ground_truth = np.squeeze(ground_truth.cpu().numpy())
            validity_map = np.where(ground_truth > 0, 1, 0)

            if save_outputs:
                ground_truth_path = os.path.join(ground_truth_dirpath, filename)
                data_utils.save_depth(ground_truth, ground_truth_path)

            validity_mask = np.where(validity_map > 0, 1, 0)
            min_max_mask = np.logical_and(
                ground_truth > min_evaluate_depth,
                ground_truth < max_evaluate_depth)
            mask = np.where(np.logical_and(validity_mask, min_max_mask) > 0)

            output_depth = output_depth[mask]
            ground_truth = ground_truth[mask]

            mae[idx] = eval_utils.mean_abs_err(1000.0 * output_depth, 1000.0 * ground_truth)
            rmse[idx] = eval_utils.root_mean_sq_err(1000.0 * output_depth, 1000.0 * ground_truth)
            imae[idx] = eval_utils.inv_mean_abs_err(0.001 * output_depth, 0.001 * ground_truth)
            irmse[idx] = eval_utils.inv_root_mean_sq_err(0.001 * output_depth, 0.001 * ground_truth)

    # Compute total time elapse in ms
    time_elapse = time_elapse * 1000.0

    if is_available_ground_truth:
        mae_mean   = np.mean(mae)
        rmse_mean  = np.mean(rmse)
        imae_mean  = np.mean(imae)
        irmse_mean = np.mean(irmse)

        mae_std = np.std(mae)
        rmse_std = np.std(rmse)
        imae_std = np.std(imae)
        irmse_std = np.std(irmse)

        # Print evaluation results to console and file
        log('Evaluation results:', log_path)
        log('{:>8}  {:>8}  {:>8}  {:>8}'.format(
            'MAE', 'RMSE', 'iMAE', 'iRMSE'),
            log_path)
        log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            mae_mean, rmse_mean, imae_mean, irmse_mean),
            log_path)

        log('{:>8}  {:>8}  {:>8}  {:>8}'.format(
            '+/-', '+/-', '+/-', '+/-'),
            log_path)
        log('{:8.3f}  {:8.3f}  {:8.3f}  {:8.3f}'.format(
            mae_std, rmse_std, imae_std, irmse_std),
            log_path)

    # Log run time
    log('Total time: {:.2f} ms  Average time per sample: {:.2f} ms'.format(
        time_elapse, time_elapse / float(n_sample)))


'''
Helper functions for logging
'''
def log_network_settings(log_path,
                         # Depth network settings
                         model_name,
                         network_modules,
                         min_predict_depth,
                         max_predict_depth,
                         # Pose network settings
                         encoder_type_pose=None,
                         rotation_parameterization_pose=None,
                         # Weight settings
                         parameters_depth_model=[],
                         parameters_pose_model=[]):

    # Computer number of parameters
    n_parameter_depth = sum(p.numel() for p in parameters_depth_model)
    n_parameter_pose = sum(p.numel() for p in parameters_pose_model)

    n_parameter = n_parameter_depth + n_parameter_pose

    n_parameter_text = 'n_parameter={}'.format(n_parameter)
    n_parameter_vars = []

    if n_parameter_depth > 0 :
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_depth={}'
        n_parameter_vars.append(n_parameter_depth)

    if n_parameter_pose > 0 :
        n_parameter_text = \
            n_parameter_text + '  ' if len(n_parameter_text) > 0 else n_parameter_text

        n_parameter_text = n_parameter_text + 'n_parameter_pose={}'
        n_parameter_vars.append(n_parameter_pose)

    log('Depth network settings:', log_path)
    log('model_name={}'.format(model_name),
        log_path)
    log('network_modules={}'.format(network_modules),
        log_path)
    log('min_predict_depth={:.2f}  max_predict_depth={:.2f}'.format(
        min_predict_depth, max_predict_depth),
        log_path)
    log('', log_path)

    if encoder_type_pose is not None and rotation_parameterization_pose is not None:
        log('Pose network settings:', log_path)
        log('encoder_type_pose={}'.format(encoder_type_pose),
            log_path)
        log('rotation_parameterization_pose={}'.format(
            rotation_parameterization_pose),
            log_path)
        log('', log_path)

    log('Weight settings:', log_path)
    log(n_parameter_text.format(*n_parameter_vars),
        log_path)
    log('', log_path)

def log_training_settings(log_path,
                          # Training settings
                          train_batch_sizes,
                          train_crop_shapes,
                          n_train_sample,
                          n_train_step,
                          # Learning rate settings
                          learning_rates,
                          learning_schedule,
                          # Replay settings
                          is_available_replay,
                          replay_batch_size,
                          replay_crop_shapes,
                          replay_dataset_size,
                          # Augmentation settings
                          augmentation_probabilities,
                          augmentation_schedule,
                          # Photometric data augmentations
                          augmentation_random_brightness,
                          augmentation_random_contrast,
                          augmentation_random_gamma,
                          augmentation_random_hue,
                          augmentation_random_saturation,
                          augmentation_random_gaussian_blur_kernel_size,
                          augmentation_random_gaussian_blur_sigma_range,
                          augmentation_random_noise_type,
                          augmentation_random_noise_spread,
                          # Geometric data augmentations
                          augmentation_padding_mode,
                          augmentation_random_crop_type,
                          augmentation_random_crop_to_shape,
                          augmentation_reverse_crop_to_shape,
                          augmentation_random_flip_type,
                          augmentation_random_rotate_max,
                          augmentation_random_crop_and_pad,
                          augmentation_random_resize_to_shape,
                          augmentation_random_resize_and_pad,
                          augmentation_random_resize_and_crop,
                          # Occlusion data augmentations
                          augmentation_random_remove_patch_percent_range_image,
                          augmentation_random_remove_patch_size_image,
                          augmentation_random_remove_patch_percent_range_depth,
                          augmentation_random_remove_patch_size_depth):

    log('Batch settings', log_path)
    log('batch_size={}'.format(sum(train_batch_sizes)),
        log_path)
    for dataset_id, (batch_size, crop_shape) in enumerate(zip(train_batch_sizes, train_crop_shapes)):
        log('dataset_id={}  n_batch={}  n_height={}  n_width={}'.format(
            dataset_id, batch_size, crop_shape[0], crop_shape[1]),
            log_path)

    log('Training settings:', log_path)
    log('n_sample={}  n_epoch={}  n_step={}'.format(
        n_train_sample, learning_schedule[-1], n_train_step),
        log_path)

    log('learning_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // train_batch_sizes[0]), le * (n_train_sample // train_batch_sizes[0]), v)
            for ls, le, v in zip([0] + learning_schedule[:-1], learning_schedule, learning_rates)),
        log_path)
    log('', log_path)

    if is_available_replay:
        log('Replay settings:', log_path)
        log('replay_batch_size={}'.format(replay_batch_size), log_path)
        log('replay_crop_shapes={}'.format(replay_crop_shapes), log_path)
        log('replay_dataset_size={}'.format(replay_dataset_size), log_path)

    log('Augmentation settings:', log_path)
    log('augmentation_schedule=[%s]' %
        ', '.join('{}-{} : {}'.format(
            ls * (n_train_sample // train_batch_sizes[0]), le * (n_train_sample // train_batch_sizes[0]), v)
            for ls, le, v in zip([0] + augmentation_schedule[:-1], augmentation_schedule, augmentation_probabilities)),
        log_path)
    log('Photometric data augmentations:', log_path)
    log('augmentation_random_brightness={}'.format(augmentation_random_brightness),
        log_path)
    log('augmentation_random_contrast={}'.format(augmentation_random_contrast),
        log_path)
    log('augmentation_random_gamma={}'.format(augmentation_random_gamma),
        log_path)
    log('augmentation_random_hue={}'.format(augmentation_random_hue),
        log_path)
    log('augmentation_random_saturation={}'.format(augmentation_random_saturation),
        log_path)
    log('augmentation_random_gaussian_blur_kernel_size={}  augmentation_random_gaussian_blur_sigma_range={}'.format(
        augmentation_random_gaussian_blur_kernel_size, augmentation_random_gaussian_blur_sigma_range),
        log_path)
    log('augmentation_random_noise_type={}  augmentation_random_noise_spread={}'.format(
        augmentation_random_noise_type, augmentation_random_noise_spread),
        log_path)

    log('Geometric data augmentations:', log_path)
    log('augmentation_padding_mode={}'.format(augmentation_padding_mode),
        log_path)
    log('augmentation_random_crop_type={}'.format(augmentation_random_crop_type),
        log_path)
    log('augmentation_random_crop_to_shape={}'.format(augmentation_random_crop_to_shape),
        log_path)
    log('augmentation_reverse_crop_to_shape={}'.format(augmentation_reverse_crop_to_shape),
        log_path)
    log('augmentation_random_flip_type={}'.format(augmentation_random_flip_type),
        log_path)
    log('augmentation_random_rotate_max={}'.format(augmentation_random_rotate_max),
        log_path)
    log('augmentation_random_crop_and_pad={}'.format(augmentation_random_crop_and_pad),
        log_path)
    log('augmentation_random_resize_to_shape={}'.format(augmentation_random_resize_to_shape),
        log_path)
    log('augmentation_random_resize_and_pad={}'.format(augmentation_random_resize_and_pad),
        log_path)
    log('augmentation_random_resize_and_crop={}'.format(augmentation_random_resize_and_crop),
        log_path)

    log('Occlusion data augmentations:', log_path)
    log('augmentation_random_remove_patch_percent_range_image={}  augmentation_random_remove_patch_size_image={}'.format(
        augmentation_random_remove_patch_percent_range_image, augmentation_random_remove_patch_size_image),
        log_path)
    log('augmentation_random_remove_patch_percent_range_depth={}  augmentation_random_remove_patch_size_depth={}'.format(
        augmentation_random_remove_patch_percent_range_depth, augmentation_random_remove_patch_size_depth),
        log_path)
    log('', log_path)

def log_loss_func_settings(log_path,
                           # Loss function settings
                           supervision_type,
                           w_losses):

    w_losses_text = ''
    for idx, (key, value) in enumerate(w_losses.items()):

        if idx > 0 and idx % 3 == 0:
            w_losses_text = w_losses_text + '\n'

        w_losses_text = w_losses_text + '{}={:.1e}'.format(key, value) + '  '

    log('Loss function settings:', log_path)
    log('supervision_type={}'.format(supervision_type), log_path)
    log(w_losses_text, log_path)
    log('', log_path)

def log_evaluation_settings(log_path,
                            min_evaluate_depths,
                            max_evaluate_depths,
                            evaluation_protocol):

    log('Evaluation settings:', log_path)
    log('evaluation_protocol={}'.format(evaluation_protocol),
        log_path)
    for i in range(len(min_evaluate_depths)):
        log('Dataset {}: min_evaluate_depth={:.2f}  max_evaluate_depth={:.2f}'.format(
        i, min_evaluate_depths[i], max_evaluate_depths[i]),
        log_path)
    log('', log_path)

def log_system_settings(log_path,
                        # Checkpoint settings
                        checkpoint_path,
                        n_step_per_checkpoint=None,
                        summary_event_path=None,
                        n_step_per_summary=None,
                        n_image_per_summary=None,
                        start_step_validation=None,
                        restore_paths=None,
                        # Hardware settings
                        device=torch.device('cuda'),
                        n_thread=8):

    log('Checkpoint settings:', log_path)

    if checkpoint_path is not None:
        log('checkpoint_path={}'.format(checkpoint_path), log_path)

        if n_step_per_checkpoint is not None:
            log('n_step_per_checkpoint={}'.format(n_step_per_checkpoint), log_path)

        if start_step_validation is not None:
            log('start_step_validation={}'.format(start_step_validation), log_path)

        log('', log_path)

        summary_settings_text = ''
        summary_settings_vars = []

    if summary_event_path is not None:
        log('Tensorboard settings:', log_path)
        log('event_path={}'.format(summary_event_path), log_path)

    if n_step_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_step_per_summary={}'
        summary_settings_vars.append(n_step_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if n_image_per_summary is not None:
        summary_settings_text = summary_settings_text + 'n_image_per_summary={}'
        summary_settings_vars.append(n_image_per_summary)

        summary_settings_text = \
            summary_settings_text + '  ' if len(summary_settings_text) > 0 else summary_settings_text

    if len(summary_settings_text) > 0:
        log(summary_settings_text.format(*summary_settings_vars), log_path)

    if restore_paths is not None and restore_paths != '':
        log('restore_paths={}'.format(restore_paths),
            log_path)

    log('', log_path)

    log('Hardware settings:', log_path)
    log('device={}'.format(device.type), log_path)
    log('n_thread={}'.format(n_thread), log_path)
    log('', log_path)
