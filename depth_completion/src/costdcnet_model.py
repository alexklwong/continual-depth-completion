import os, sys, argparse
sys.path.insert(0, os.path.join('external_src', 'depth_completion', 'costdcnet'))
sys.path.insert(0, os.path.join('external_src', 'depth_completion', 'costdcnet', 'models'))
from CostDCNet import CostDCNet as CostDCNetBaseModel
import torch
import torch.nn as nn
import torchvision
import MinkowskiEngine as ME


class CostDCNetModel(object):
    '''
    Class for interfacing with NLSPN model

    Arg(s):
        device : torch.device
            device to run model on
        max_predict_depth : float
            value to clamp ground truths to in computing loss
        use_pretrained : bool
            if set, then configure using legacy settings
    '''
    def __init__(self,
                 device=torch.device('cuda'),
                 max_predict_depth=10.0,
                 ):

        # Settings to reproduce NLSPN numbers on KITTI
        args = argparse.Namespace(
            time=False,
            res=16,  # for NYU v2
            up_scale=4,  # for NYU v2
            max=max_predict_depth,
            device=device)
        self.args = args

        # Instantiate depth completion model
        self.model = CostDCNetBaseModel(args)

        # Move to device
        self.device = device
        self.to(self.device)

    def forward_depth(self,
                image,
                sparse_depth,
                intrinsics=None):
        '''
        Forwards inputs through the network

        Arg(s):
            image : tensor[float32]
                N x 3 x H x W image
            sparse_depth : tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            loss_type : forward types
                {'pretrain' : get output depth,
                 'prepare' : get embedding for self-supervised learning,
                 'adapt' : get both output depth and embeddings for self-supervised learning}
        Returns:
            loss_type == 'pretrain':
                torch.Tensor[float32] N x 1 x H x W dense depth map
            loss_type == 'prepare':
                torch.Tensor[float32] N x (H * W) x C prediction embedding
                torch.Tensor[float32].detach() N x (H * W) x C projection embedding
            loss_type == 'adapt':
                torch.Tensor[float32] N x 1 x H x W dense depth map
                torch.Tensor[float32] N x (H * W) x C prediction embedding
                torch.Tensor[float32].detach() N x (H * W) x C projection embedding
        '''
        batch_size, _, og_height, og_width = image.shape
        
        image, sparse_depth = self.transform_inputs(image, sparse_depth)
        image, sparse_depth, intrinsics = \
            self.pad_inputs(image, sparse_depth, intrinsics)

        output_depth = self.model(
            image=image,
            sparse_depth=sparse_depth)

        output_depth = self.recover_inputs(output_depth, og_height, og_width)

        return output_depth

    def transform_inputs(self, image, sparse_depth):
        '''
        Transforms the input based on any required preprocessing step

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
        Returns:
            torch.Tensor[float32] : N x 3 x H x W image
            torch.Tensor[float32] : N x 1 x H x W sparse depth map
        '''

        
        # Clamping the sparse depth input
        sparse_depth = torch.clamp(
            sparse_depth,
            min=0,
            max=self.max_predict_depth)

        image = image / 255.0

        for batch in range(image.shape[0]):

            image[batch, ...] = torchvision.transforms.functional.normalize(
                image[batch, ...],
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))
            
        return image, sparse_depth

    def pad_inputs(self, image, sparse_depth, intrinsics):

        n_height, n_width = image.shape[-2:]

        do_padding = False
        # Pad the images and expand at 0-th dimension to get batch
        if n_height % 128 != 0:
            times = n_height // 128
            padding_top = (times + 1) * 128 - n_height
            do_padding = True
        else:
            padding_top = 0

        if n_width % 128 != 0:
            times = n_width // 128
            padding_right = (times + 1) * 128 - n_width
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
            intrinsics = torch.cat([intrinsics, intrinsics], dim=0)

        return image, sparse_depth, intrinsics

    def recover_inputs(self, output_depth, n_height, n_width):
        height, width = output_depth.size()[-2:]
        do_padding = False if (n_height == height and n_width == width) else True
        if do_padding:
            padding_top = height - n_height
            padding_right = width - n_width
            output0, output1 = torch.chunk(output_depth, chunks=2, dim=0)
            if padding_right == 0 :
                output0 = output0[:, :, padding_top:, :]
                output1 = output1[:, :, :-padding_top, :]
            elif padding_top == 0:
                output0 = output0[:, :, :, :-padding_right]
                output1 = output1[:, :, :, padding_right:]
            else:
                output0 = output0[:, :, padding_top:, :-padding_right]
                output1 = output1[:, :, :-padding_top, padding_right:]

            output_depth = torch.cat([
                torch.unsqueeze(output0, dim=1),
                torch.unsqueeze(output1, dim=1)],
                dim=1)

            output_depth = torch.mean(output_depth, dim=1, keepdim=False)

        return output_depth

    def _prepare_head(self, mode):
        '''
        Initialize the self-supervised MLP heads
        '''
        return self.model._prepare_head(mode=mode)

    def get_offset(self):
        '''
        Get offset values
        '''
        if isinstance(self.model, torch.nn.parallel.DataParallel):
            self.model.module.get_offset()
        else:
            self.model.get_offset()

    def compute_loss(self,
                     output_depth=None,
                     target_depth=None,
                     image=None,
                     w_losses=None):
        '''

        Arg(s):
            output_depth : torch.Tensor[float32]
                N x 1 x H x W dense output depth already masked with validity map
            ground_truth_depth : torch.Tensor[float32]
                N x 2 x H x W ground_truth depth and ground truth validity map
            l1_weight : float
                weight of l1 loss
            l2_weight : float
                weight of l2 loss
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''
        if self.dataset_name == 'void':
            l1_weight=1.0
            l2_weight=0.0
        elif self.dataset_name == 'kitti':
            l1_weight=1.0
            l2_weight=1.0

        m = torch.where(target_depth>0, torch.ones_like(target_depth), torch.zeros_like(target_depth))
        
        num_valid = torch.sum(m)
        d = torch.abs(target_depth - output_depth) * m
        d = torch.sum(d, dim=[1, 2, 3])
        loss_l1 = d / (num_valid + 1e-8)
        loss_l1 = loss_l1.sum()

        d2 = torch.pow(target_depth - output_depth, 2) * m
        d2 = torch.sum(d2, dim=[1, 2, 3])
        loss_l2 = d2 / (num_valid + 1e-8)
        loss_l2 = loss_l2.sum()
        loss = l1_weight * loss_l1 + l2_weight * loss_l2

        # Store loss info
        loss_info = {
            'loss_l1': loss_l1.detach().item(),
            'loss_l2': loss_l2.detach().item(),
            'loss': loss.detach().item()
        }

        return loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        parameters = []
        parameters = torch.nn.ParameterList(self.model.parameters())

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

    def set_device(self, rank):
        self.model.module.set_device(rank)

    def distributed_data_parallel(self, rank):
        '''
        Allows multi-gpu split along batch
        '''
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[rank], find_unused_parameters=True)

    def restore_model(self, restore_path, optimizer=None, learning_schedule=None, learning_rates=None, n_step_per_epoch=None):
        '''
        Loads weights from checkpoint and loads and returns optimizer

        Arg(s):
            restore_path : str
                path to model weights
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            torch.optimizer if optimizer is passed in
        '''
        checkpoint_dict = torch.load(restore_path, map_location=self.device)
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint_dict['net'])
        elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model.load_state_dict(checkpoint_dict['net'])
        else:
            self.model.load_state_dict(checkpoint_dict['net'])

        if 'meanvar' in checkpoint_dict.keys():
            for k in checkpoint_dict['meanvar'].keys():
                if k != 'length':
                    checkpoint_dict['meanvar'][k] = checkpoint_dict['meanvar'][k].cuda()
            self.model.glob_mean = checkpoint_dict['meanvar']

            self.mean_var_dict = checkpoint_dict['meanvar']

        if optimizer is not None and 'optimizer' in checkpoint_dict.keys():
            optimizer.load_state_dict(checkpoint_dict['optimizer'])
            return optimizer

    def save_model(self, checkpoint_path, step, optimizer, meanvar=None):
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
        elif isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
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
        if meanvar is not None:
            checkpoint['meanvar'] = meanvar
        torch.save(checkpoint, checkpoint_path)

    def convert_syncbn(self, apex):
        '''
        Convert BN layers to SyncBN layers.
        SyncBN merge the BN layer weights in every backward step.
        '''
        if apex:
            apex.parallel.convert_syncbn_model(self.model)
        else:
            from torch.nn import SyncBatchNorm
            SyncBatchNorm.convert_sync_batchnorm(self.model)
