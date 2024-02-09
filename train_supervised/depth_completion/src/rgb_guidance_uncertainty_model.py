import os, sys
import torch
import utils.src.loss_utils as loss_utils
sys.path.insert(0, os.path.join('external_src', 'Sparse-Depth-Completion'))
sys.path.insert(0, os.path.join('external_src', 'Sparse-Depth-Completion', 'Models'))
from model import uncertainty_net as RGBGuidanceUncertainty


class RGBGuidanceUncertaintyModel(object):
    '''
    Class for interfacing with RGB Guidance and Uncertainty model

    Arg(s):
        max_predict_depth : float
            value to clamp ground truths to in computing loss
        use_pretrained : bool
            if set, then configure using legacy settings
        device : torch.device
            device to run model on
    '''
    def __init__(self, max_predict_depth=100.0, use_pretrained=False, device=torch.device('cuda')):

        # Instantiate model
        self.model = RGBGuidanceUncertainty(in_channels=4)

        self.max_predict_depth = max_predict_depth

        pretrained_path_erfnet = os.path.join(
            'external_models',
            'rgb_guidance_uncertainty',
            'kitti',
            'erfnet_pretrained.pth')

        if use_pretrained:
            checkpoint = torch.load(pretrained_path_erfnet)

            state_dict = {
                k[7:] : v
                for k, v in checkpoint.items() if k[7:] in self.model.depthnet.state_dict()
            }

            state_dict.pop('encoder.initial_block.conv.weight')
            state_dict.pop('encoder.initial_block.conv.bias')
            state_dict.pop('decoder.output_conv.weight')
            state_dict.pop('decoder.output_conv.bias')
            self.model.depthnet.load_state_dict(state_dict, strict=False)

        self.device = device
        self.to(self.device)

    def forward(self, image, sparse_depth, intrinsics=None, return_all_outputs=False):
        '''
        Forwards inputs through the network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
        Returns:
            torch.Tensor[float32] : N x 1 x H x W dense depth map or list of all outputs
        '''

        image, sparse_depth = self.transform_inputs(image, sparse_depth)

        n_height, n_width = image.shape[-2:]

        do_padding = False

        # Pad to width and height such that it is divisible by 16
        if n_height % 16 != 0:
            times = n_height // 16
            padding_top = (times + 1) * 16 - n_height
            do_padding = True
        else:
            padding_top = 0

        if n_width % 16 != 0:
            times = n_width // 16
            padding_right = (times + 1) * 16 - n_width
            do_padding = True
        else:
            padding_right = 0

        if do_padding:
            # Pad the images and expand at 0-th dimension to get batch
            image0 = torch.nn.functional.pad(
                image,
                (0, padding_right, padding_top, 0, 0, 0),
                mode='constant',
                value=0)

            sparse_depth0 = torch.nn.functional.pad(
                sparse_depth,
                (0, padding_right, padding_top, 0, 0, 0),
                mode='constant',
                value=0)

            image1 = torch.nn.functional.pad(
                image,
                (padding_right, 0, 0, padding_top, 0, 0),
                mode='constant',
                value=0)

            sparse_depth1 = torch.nn.functional.pad(
                sparse_depth,
                (padding_right, 0, 0, padding_top, 0, 0),
                mode='constant',
                value=0)

            image = torch.cat([image0, image1], dim=0)
            sparse_depth = torch.cat([sparse_depth0, sparse_depth1], dim=0)

        outputs_ = self.model.forward(
            torch.cat([sparse_depth, image], axis=1))

        outputs = []

        for output in outputs_:

            if do_padding:
                output0, output1 = torch.chunk(output, chunks=2, dim=0)

                if padding_right == 0 :
                    output0 = output0[:, :, padding_top:, :]
                    output1 = output1[:, :, :-padding_top, :]
                elif padding_top == 0:
                    output0 = output0[:, :, :, :-padding_right]
                    output1 = output1[:, :, :, padding_right:]
                else:
                    output0 = output0[:, :, padding_top:, :-padding_right]
                    output1 = output1[:, :, :-padding_top, padding_right:]

                output = torch.cat([
                    torch.unsqueeze(output0, dim=1),
                    torch.unsqueeze(output1, dim=1)],
                    dim=1)
                output = torch.mean(output, dim=1, keepdim=False)

            if not self.model.training:
                output = torch.clamp(output, 0, self.max_predict_depth)

            outputs.append(output)

        if return_all_outputs:
            return outputs
        else:
            return outputs[0]

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
            output_depth : list[torch.Tensor[float32]]
                N x 1 x H x W dense output depth, lidar branch, precision, guidance
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
        w_pred = w_losses['w_pred'] if 'w_pred' in w_losses else 1.0
        w_lidar = w_losses['w_lidar'] if 'w_lidar' in w_losses else 0.1
        w_precise = w_losses['w_precise'] if 'w_precise' in w_losses else 0.1
        w_guide = w_losses['w_guide'] if 'w_guide' in w_losses else 0.1
        w_smoothness = w_losses['w_smoothness'] if 'w_smoothness' in w_losses else 0.0

        loss_info = {}

        assert isinstance(output_depth, list)

        # Clamping ground truth values
        ground_truth = torch.clamp(
            ground_truth,
            min=0.0,
            max=self.max_predict_depth)

        # Obtain valid values
        validity_map = torch.where(
            ground_truth > 0,
            torch.ones_like(ground_truth),
            ground_truth)

        # Note: lidar_out = lidar_to_depth, lidar_to_conf
        pred, lidar_out, precise, guide = output_depth

        lidar_to_depth, lidar_to_conf = torch.chunk(lidar_out, 2, dim=1)

        pred_loss = w_pred * loss_utils.mse_focal(pred, ground_truth, w=validity_map)
        lidar_to_depth_loss = loss_utils.mse_focal(lidar_to_conf, ground_truth, w=validity_map)
        lidar_to_conf_loss = loss_utils.mse_focal(lidar_to_depth, ground_truth, w=validity_map)
        lidar_loss = w_lidar * (lidar_to_depth_loss + lidar_to_conf_loss)
        precise_loss = w_precise * loss_utils.mse_focal(precise, ground_truth, w=validity_map)
        guide_loss = w_guide * loss_utils.mse_focal(guide, ground_truth, w=validity_map)

        loss = pred_loss + lidar_loss + precise_loss + guide_loss

        # Store loss info
        loss_info['pred_loss'] = pred_loss
        loss_info['lidar_loss'] = lidar_loss
        loss_info['precise_loss'] = precise_loss
        loss_info['guide_loss'] = guide_loss

        if w_smoothness > 0 and image is not None:
            loss_smoothness = w_smoothness * loss_utils.smoothness_loss_func(pred, image)
            loss_info['loss_smoothness'] = loss_smoothness

            loss = loss + loss_smoothness

        # Store loss info
        loss_info['loss'] = loss

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        parameters = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                parameters.append(param)

        parameters = torch.nn.ParameterList(parameters)

        return parameters

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

        self.model.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.model = torch.nn.DataParallel(self.model)

    def restore_model(self, restore_paths, optimizer=None):
        '''
        Loads weights from checkpoint and loads and returns optimizer

        Arg(s):
            restore_paths : list[str]
                path to model weights
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        if isinstance(restore_paths, list):
            restore_paths = restore_paths[0]

        checkpoint = torch.load(restore_paths, map_location=self.device)

        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint['state_dict'])

        if optimizer is not None and 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])

        try:
            train_step = checkpoint['train_step']
        except Exception:
            train_step = 0

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

        if isinstance(self.model, torch.nn.DataParallel):
            checkpoint = {
                'state_dict': self.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        else:
            checkpoint = {
                'state_dict': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }

        torch.save(checkpoint, checkpoint_path)
