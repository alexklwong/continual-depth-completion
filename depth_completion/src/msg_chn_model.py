import torch
import os, sys
import utils.src.loss_utils as loss_utils
sys.path.insert(0, os.path.join('external_src', 'MSG_CHN', 'workspace', 'exp_msg_chn'))
from network_exp_msg_chn import network


class MsgChnModel(object):
    '''
    Class for interfacing with MSGCHN model

    Arg(s):
        max_predict_depth : float
            value to clamp ground truths to in computing loss
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):

        # Initialize model
        self.model = network()

        self.max_predict_depth = max_predict_depth

        # Move to device
        self.device = device
        self.to(self.device)

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

        image, sparse_depth, intrinsics = self.transform_inputs(
            image,
            sparse_depth,
            intrinsics)

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

        outputs = self.model.forward(sparse_depth, image)

        output_depths = []

        for output in outputs:

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
                output = torch.clamp(output, min=0.0, max=self.max_predict_depth)

            output_depths.append(output)

        if return_all_outputs:
            return output_depths
        else:
            return output_depths[0]

    def transform_inputs(self, image, sparse_depth, intrinsics):
        '''
        Transforms the input based on any required preprocessing step

        Arg(s):
            image : tensor[float32]
                N x 3 x H x W image
            sparse_depth : tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
        Returns:
            tensor[float32] : N x 3 x H x W image
            tensor[float32] : N x 1 x H x W input sparse depth map
        '''

        # Clamping the sparse depth input
        sparse_depth = torch.clamp(
            sparse_depth,
            min=0,
            max=self.max_predict_depth)

        # Normalization
        image = image / 255.0

        return image, sparse_depth, intrinsics

    def compute_loss(self, output_depth, ground_truth, image=None, w_losses={}):
        '''
        Call the model's compute loss function

        Arg(s):
            output_depth : list[torch.Tensor[float32]]
                N x 1 x H x W dense output depth already masked with validity map or list of all outputs
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
        w_scale0 = w_losses['w_scale0'] if 'w_scale0' in w_losses else 1.0
        w_scale1 = w_losses['w_scale1'] if 'w_scale1' in w_losses else 0.5
        w_scale2 = w_losses['w_scale2'] if 'w_scale1' in w_losses else 0.1
        w_smoothness = w_losses['w_smoothness'] if 'w_smoothness' in w_losses else 0.0

        loss_info = {}

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

        w = [w_scale0, w_scale1, w_scale2]
        loss = 0.0

        for i in range(len(output_depth)):
            # Compute loss

            loss_scale = w[i] * loss_utils.l2_loss(
                src=output_depth[i],
                tgt=ground_truth,
                w=validity_map)

            loss = loss + loss_scale

            # Store loss info
            loss_info['loss-{}'.format(i)] = loss_scale

            if w_smoothness > 0 and image is not None:
                loss_smoothness_scale = \
                    w_smoothness * loss_utils.smoothness_loss_func(output_depth[i], image)

                loss_info['loss_smoothness-{}'.format(i)] = loss_smoothness_scale

                loss = loss + loss_smoothness_scale

        # Store loss info
        loss_info['loss'] = loss

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.tensor[float32]] : list of parameters
        '''

        parameters = list(self.model.parameters())
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

        self.device = device
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
            self.model.module.load_state_dict(checkpoint['net'])
        else:
            self.model.load_state_dict(checkpoint['net'])

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
                'net': self.model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }
        else:
            checkpoint = {
                'net': self.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_step': step
            }

        torch.save(checkpoint, checkpoint_path)