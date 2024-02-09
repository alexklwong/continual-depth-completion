import os, sys
import torch
import utils.src.loss_utils as loss_utils

sys.path.insert(0, os.path.join('depth_completion', 'kbformer'))
# sys.path.insert(0, os.path.join('external_src', 'NLSPN', 'src', 'model'))
from kbformer_renew import KBformer as KBformerBaseModel

class KBformer_Model(object):
    '''
    Class for interfacing with NLSPN model

    Arg(s):
        max_predict_depth : float
            value to clamp ground truths to in computing loss
        use_pretraineda : bool
            if set, then configure using legacy settings
        device : torch.device
            device to run model on
    '''

    def __init__(self, min_predict_depth=1.0,
                 max_predict_depth=100.0,
                 ffn_bias=False,
                 large=False,
                 device=torch.device('cuda')):
        if ffn_bias:
            kb_token_mlp = 'kbff'
            token_mlp = 'leff'
        else:
            kb_token_mlp = 'kbff_wo_bias'
            token_mlp = 'leff_wo_bias'
        if large:
            depths = [2 for _ in range(9)]
            shift_flag = True
        else:
            depths = [1 for _ in range(9)]
            shift_flag = False
        # Instantiate depth completion model
        cost_volume = False
        self.cost_volume = cost_volume
        self.model = KBformerBaseModel(input_channels_depth=2,
                 min_pool_sizes_sparse_to_dense_pool=[5, 7, 9, 11, 13],
                 max_pool_sizes_sparse_to_dense_pool=[15, 17],
                 n_convolution_sparse_to_dense_pool=3,
                 n_filter_sparse_to_dense_pool=8,
                 weight_initializer='kaiming_uniform',
                 # Backprojection_option
                 resolutions_backprojection=[0, 1, 2],
                #  resolutions_backprojection=[],
                 # Transformer_block
                 img_size=[256], in_chans=3, dd_in=3,
                 embed_dim=32, embed_dim_z=16,
                 depths=depths, num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
                 win_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
                 patch_norm=True,
                 use_checkpoint=False,
                 token_projection='linear',
                 token_mlp=token_mlp,
                 kb_token_mlp=kb_token_mlp,
                 shift_flag=shift_flag, modulator=True,
                 cross_modulator=False,
                 # For cost volume
                 cost_volume=cost_volume,
                 max_predict_depth=max_predict_depth,
                 ffn_bias=ffn_bias)
        self.min_predict_depth = min_predict_depth
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

        # Transform inputs - Scaling
        image, sparse_depth = self.transform_inputs(image, sparse_depth)

        n_height, n_width = image.size()[-2:]

        validity_map = torch.where(
            sparse_depth > 0,
            torch.ones_like(sparse_depth),
            sparse_depth)

        # Transform inputs - Spatial align
        image, sparse_depth, validity_map, intrinsics = \
            self.pad_inputs(image, sparse_depth, validity_map, intrinsics)

        output = self.model.forward(image, sparse_depth, validity_map, intrinsics)
        if self.cost_volume:
            pass
        else:
            output = torch.sigmoid(output)
            output = self.min_predict_depth / (output + self.min_predict_depth / self.max_predict_depth)
        output = self.recover_inputs(output, n_height, n_width)

        if return_all_outputs:
            return [output]
        else:
            return output

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
        # sparse_depth = torch.clamp(
        #     sparse_depth,
        #     min=0,
        #     max=self.max_predict_depth)

        image = image / 255.0
        # sparse_depth = sparse_depth / 256.0

        return image, sparse_depth

    def pad_inputs(self, image, sparse_depth, validity_map_depth, intrinsics):

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

            validity_map_depth0 = torch.nn.functional.pad(
                validity_map_depth,
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

            validity_map_depth1 = torch.nn.functional.pad(
                validity_map_depth,
                (padding_right, 0, 0, padding_top, 0, 0),
                mode='constant',
                value=0)

            image = torch.cat([image0, image1], dim=0)
            sparse_depth = torch.cat([sparse_depth0, sparse_depth1], dim=0)
            validity_map_depth = torch.cat([validity_map_depth0, validity_map_depth1], dim=0)
            intrinsics = torch.cat([intrinsics, intrinsics], dim=0)

        return image, sparse_depth, validity_map_depth, intrinsics

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
        w_l1 = w_losses['w_l1'] if 'w_l1' in w_losses else 1.0
        w_l2 = w_losses['w_l2'] if 'w_l2' in w_losses else 1.0
        w_smoothness = w_losses['w_smoothness'] if 'w_smoothness' in w_losses else 0.0

        loss_info = {}

        if isinstance(output_depth, list):
            output_depth = output_depth[0]

        # NLSPN clamps predictions during loss computation
        # output_depth = torch.clamp(output_depth, min=0, max=self.max_predict_depth)
        # ground_truth = torch.clamp(ground_truth, min=0, max=self.max_predict_depth)

        # Obtain valid values
        validity_map = torch.where(
            ground_truth > 0,
            torch.ones_like(ground_truth),
            ground_truth)

        # Compute individual losses
        l1_loss = w_l1 * loss_utils.l1_loss(
            src=output_depth,
            tgt=ground_truth,
            w=validity_map)

        l2_loss = w_l2 * loss_utils.l2_loss(
            src=output_depth,
            tgt=ground_truth,
            w=validity_map)

        loss = l1_loss + l2_loss

        # Store loss info
        loss_info['l1_loss'] = l1_loss
        loss_info['l2_loss'] = l2_loss

        if w_smoothness > 0 and image is not None:
            loss_smoothness = w_smoothness * loss_utils.smoothness_loss_func(output_depth, image)
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

        if isinstance(self.model, torch.nn.DataParallel) or isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
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
