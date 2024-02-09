import os, sys
import torch
sys.path.insert(0, os.path.join('external_src', 'scaffnet'))
sys.path.insert(0, os.path.join('external_src', 'scaffnet', 'src'))
from scaffnet_model import ScaffNetModel as ScaffNet


class ScaffNetModel(object):
    '''
    Class for interfacing with ScaffNet model

    Arg(s):
        dataset_name : str
            model for a given dataset
        network_modules : list[str]
            list of additional network modules to build for model i.e. uncertainty decoder, freeze_all, freeze_depth
        min_predict_depth : float
            minimum value of predicted depth
        max_predict_depth : flaot
            maximum value of predicted depth
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 dataset_name='vkitti',
                 network_modules=['depth'],
                 min_predict_depth=1.5,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):

        encoder_type = ['batch_norm', 'spatial_pyramid_pool']
        decoder_type = ['multi-scale', 'batch_norm']

        if 'resnet18' in network_modules:
            encoder_type.append('resnet18')
        elif 'vggnet08' in network_modules:
            encoder_type.append('vggnet08')
        else:
            encoder_type.append('vggnet08')

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth

        if 'uncertainty' in network_modules:
            decoder_type.append('uncertainty')

        # Instantiate depth completion model
        if dataset_name == 'kitti' or dataset_name == 'vkitti':
            max_pool_sizes_spatial_pyramid_pool = [13, 17, 19, 21, 25]
            n_convolution_spatial_pyramid_pool = 3
            n_filter_spatial_pyramid_pool = 8
            n_filters_encoder = [16, 32, 64, 128, 256]
            n_filters_decoder = [256, 128, 128, 64, 32]
        elif dataset_name == 'nuscenes':
            max_pool_sizes_spatial_pyramid_pool = [13, 17, 19, 21, 25, 33, 39, 43, 47]
            n_convolution_spatial_pyramid_pool = 3
            n_filter_spatial_pyramid_pool = 8
            n_filters_encoder = [16, 32, 64, 128, 256]
            n_filters_decoder = [256, 128, 128, 64, 32]
        elif dataset_name == 'void' or dataset_name == 'scenenet' or dataset_name == 'nyu_v2':
            max_pool_sizes_spatial_pyramid_pool = [13, 17, 19, 21, 25]
            n_convolution_spatial_pyramid_pool = 3
            n_filter_spatial_pyramid_pool = 8
            n_filters_encoder = [16, 32, 64, 128, 256]
            n_filters_decoder = [256, 128, 128, 64, 32]
        else:
            raise ValueError('Unsupported dataset settings: {}'.format(dataset_name))

        self.network_modules = network_modules
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type

        # Build ScaffNet
        self.model = ScaffNet(
            max_pool_sizes_spatial_pyramid_pool=max_pool_sizes_spatial_pyramid_pool,
            n_convolution_spatial_pyramid_pool=n_convolution_spatial_pyramid_pool,
            n_filter_spatial_pyramid_pool=n_filter_spatial_pyramid_pool,
            encoder_type=encoder_type,
            n_filters_encoder=n_filters_encoder,
            decoder_type=decoder_type,
            n_filters_decoder=n_filters_decoder,
            n_output_resolution=1,
            weight_initializer='xavier_normal',
            activation_func='leaky_relu',
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            device=device)

        # Move to device
        self.device = device
        self.to(self.device)

        # Freeze ScaffNet
        if 'freeze_depth' in self.network_modules:
            self.frozen_module_names = ['spatial_pyramid_pool', 'encoder', 'decoder_depth']
            self.do_freeze = True
        elif 'freeze_all' in self.network_modules:
            self.frozen_module_names = ['all']
            self.do_freeze = True
        else:
            self.do_freeze = False

        if self.do_freeze:
            self.model.freeze(self.frozen_module_names)

    def forward(self, image, sparse_depth, intrinsics, return_all_outputs=False):
        '''
        Forwards inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
            return_all_outputs : bool
                if set, return all outputs
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map or list of all outputs
        '''

        _, sparse_depth = self.transform_inputs(image, sparse_depth)

        if self.do_freeze:
            self.model.freeze(self.frozen_module_names)

        # Forward through ScaffNet
        output_depth = self.model.forward(sparse_depth)

        if return_all_outputs:
            if 'uncertainty' in self.model.decoder_type:
                return torch.chunk(output_depth, chunks=2, dim=1)

            return [output_depth]
        else:
            if 'uncertainty' in self.model.decoder_type:
                output_depth = output_depth[:, 0:1, :, :]

            return output_depth

    def transform_inputs(self, image, sparse_depth):
        '''
        Transforms the input based on any required preprocessing step

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
        Returns:
            torch.Tensor[float32] : N x 3 x H x W image
            torch.Tensor[float32] : N x 1 x H x W sparse depth map
        '''

        # Clamping the sparse depth input
        sparse_depth = torch.clamp(
            sparse_depth,
            min=0,
            max=self.max_predict_depth)

        return image, sparse_depth

    def compute_loss(self, output_depth, ground_truth, image=None, w_losses={}):
        '''
        Call the model's compute loss function

        Arg(s):
            output_depth : torch.Tensor[float32]
                N x 1 x H x W dense output depth and uncertainty map
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground_truth depth with only valid values
            image : torch.Tensor[float32]
                N x 3 x H x W image for edge awareness weights
            w_losses : dict[str, float]
                dictionary of weights for each loss
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        # Check if loss weighting was passed in, if not then use default weighting
        if 'w_supervised_l1' in w_losses :
            w_supervised = w_losses['w_supervised_l1']
            loss_func = 'supervised_l1'
        elif 'w_supervised_l1_normalized' in w_losses:
            w_supervised = w_losses['w_supervised_l1_normalized']
            loss_func = 'supervised_l1_normalized'
        elif 'w_supervised_l2' in w_losses:
            w_supervised = w_losses['w_supervised_l2']
            loss_func = 'supervised_l2'
        elif 'w_supervised_l2_normalized' in w_losses:
            w_supervised = w_losses['w_supervised_l2_normalized']
            loss_func = 'supervised_l2_normalized'
        else:
            w_supervised = 1.00
            loss_func = 'supervised_l1_normalized'

        if 'uncertainty' in self.model.decoder_type:
            output_uncertainty = output_depth[1]
            output_depth = output_depth[0]
        else:
            output_uncertainty = None

        # Compute loss function
        loss, loss_info = self.model.compute_loss(
            loss_func=loss_func,
            target_depth=ground_truth,
            output_depths=output_depth,
            output_uncertainties=output_uncertainty,
            w_supervised=w_supervised)

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.model.parameters()

    def train(self):
        '''
        Sets model to training mode
        '''

        self.model.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device
        self.model.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.model.data_parallel()

    def restore_model(self, restore_paths, optimizer=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_paths : list[str]
                path to model weights
            optimizer : torch.optim
                optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        if isinstance(restore_paths, list):
            restore_paths = restore_paths[0]

        train_step, optimizer = self.model.restore_model(
            checkpoint_path=restore_paths,
            optimizer=optimizer)

        return train_step, optimizer

    def save_model(self, checkpoint_path, step, optimizer):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        self.model.save_model(
            checkpoint_path=checkpoint_path,
            step=step,
            optimizer=optimizer)
