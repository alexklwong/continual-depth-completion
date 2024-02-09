import os, sys
import torch
import utils.src.loss_utils as loss_utils

sys.path.insert(0, os.path.join('depth_completion', 'kbformer'))
# sys.path.insert(0, os.path.join('external_src', 'NLSPN', 'src', 'model'))
from uformer import Uformer as UformerBaseModel
from uformer_kb import KBformer_SUM_deckb_multiscale
from uformer_kb_Nov5 import KBformer_SUM
from uformer_kb_Nov5 import KBformer_SUM_3D_lastlayer
from uformer_kb_Nov5 import KBformer_SUM_3D_lastlayer_kb
from uformer_kb_Nov5 import KBformer_SUM_KB
from uformer_kb_Nov5 import KBformer_SUM_3D_lastlayer_kb_V2
class Uformer_Model_SS(object):
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
                 kb=False,
                 dec_3d=False,
                 dec_bn=False,
                 dec_kb=False,
                 squeeze_all=False,
                 half=False,
                 last_layer=False,
                 multi_scale=0,
                 frustum=False,
                 double_3d=False,
                 binning=False,
                 legacy=False,
                 v2=False,
                 xatt=False,
                 device=torch.device('cuda')):
    
        # if ffn_bias:
        #     kb_token_mlp = 'kbff'
        #     token_mlp = 'leff'
        # else:
        #     kb_token_mlp = 'kbff_wo_bias'
        #     token_mlp = 'leff_wo_bias'
        # if large:
        #     depths = [2 for _ in range(9)]
        #     shift_flag = True
        # else:
        #     depths = [1 for _ in range(9)]
        #     shift_flag = False
        # Instantiate depth completion model
        self.cost_volume = dec_3d

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        print('multi-scale-loss with dec-level-feature-prediction')
        assert multi_scale
        if not dec_3d:
            if not kb:
                raise NotImplementedError
            
            if xatt:
                self.model = KBformer_SUM_KB(img_size=256,
                                        dd_in=3,
                                        embed_dim=32,
                                        win_size=8,
                                        token_projection='linear',
                                        token_mlp='leff',
                                        conv_dec=True,
                                        dec_bn=dec_bn,
                                        dec_kb=dec_kb,
                                        shift_flag=True,
                                        multi_scale=multi_scale,
                                        modulator=False)
            else:
                self.model = KBformer_SUM(img_size=256,
                                        dd_in=3,
                                        embed_dim=32,
                                        win_size=8,
                                        token_projection='linear',
                                        token_mlp='leff',
                                        conv_dec=True,
                                        dec_bn=dec_bn,
                                        dec_kb=dec_kb,
                                        shift_flag=True,
                                        multi_scale=multi_scale,
                                        modulator=False)
        else:
            if squeeze_all:
                print('multi-scale-loss with 3D last layer feature with squeezing all')
            else:
                print('multi-scale-loss with 3D last layer feature with squeezing D->C')

            if xatt:
                if v2:
                    self.model = KBformer_SUM_3D_lastlayer_kb_V2(img_size=256,
                                        dd_in=3,
                                        embed_dim=24,
                                        win_size=8,
                                        token_projection='linear',
                                        token_mlp='leff',
                                        conv_dec=True,
                                        dec_bn=dec_bn,
                                        dec_kb=dec_kb,
                                        res=8,
                                        shift_flag=True,
                                        multi_scale=multi_scale,
                                        frustum=frustum,
                                        binning=binning,
                                        squeeze_all=squeeze_all,
                                        double_3d=double_3d,
                                        modulator=False)
                else:
                    self.model = KBformer_SUM_3D_lastlayer_kb(img_size=256,
                                        dd_in=3,
                                        embed_dim=24,
                                        win_size=8,
                                        token_projection='linear',
                                        token_mlp='leff',
                                        conv_dec=True,
                                        dec_bn=dec_bn,
                                        dec_kb=dec_kb,
                                        res=8,
                                        shift_flag=True,
                                        multi_scale=multi_scale,
                                        frustum=frustum,
                                        binning=binning,
                                        squeeze_all=squeeze_all,
                                        double_3d=double_3d,
                                        modulator=False)
            else:
                self.model = KBformer_SUM_3D_lastlayer(img_size=256,
                                        dd_in=3,
                                        embed_dim=32,
                                        win_size=8,
                                        token_projection='linear',
                                        token_mlp='leff',
                                        conv_dec=True,
                                        dec_bn=dec_bn,
                                        dec_kb=dec_kb,
                                        res=10,
                                        legacy=legacy,
                                        shift_flag=True,
                                        multi_scale=multi_scale,
                                        frustum=frustum,
                                        binning=binning,
                                        squeeze_all=squeeze_all,
                                        double_3d=double_3d,
                                        modulator=False)
            self.return_all_outputs=False


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

        output = self.model.forward(x=torch.cat([image, sparse_depth, validity_map], dim=1), intrinsics=intrinsics)

        for i in range(len(output)):
            output[i] = self.recover_inputs(output[i], n_height, n_width)
        if not self.model.training:
            return output[0]
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

            # Right top
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

            intrinsics_arr0 = self.adjust_intrinsics(
                intrinsics,
                x_scales=1.0,
                y_scales=1.0,
                x_offsets=padding_right,
                y_offsets=-padding_top)

            # Left Bottom
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
            intrinsics_arr1 = self.adjust_intrinsics(
                intrinsics,
                x_scales=1.0,
                y_scales=1.0,
                 x_offsets=-padding_right,
                y_offsets=padding_top)


            image = torch.cat([image0, image1], dim=0)
            sparse_depth = torch.cat([sparse_depth0, sparse_depth1], dim=0)
            validity_map_depth = torch.cat([validity_map_depth0, validity_map_depth1], dim=0)
            intrinsics = torch.cat([intrinsics_arr0, intrinsics_arr1], dim=0)

        return image, sparse_depth, validity_map_depth, intrinsics

    def adjust_intrinsics(self,
                          intrinsics,
                          x_scales=1.0,
                          y_scales=1.0,
                          x_offsets=0.0,
                          y_offsets=0.0):
        '''
        Adjust the each camera intrinsics based on the provided scaling factors and offsets

        Arg(s):
            intrinsics : torch.Tensor[float32]
                3 x 3 camera intrinsics
            x_scales : list[float]
                scaling factor for x-direction focal lengths and optical centers
            y_scales : list[float]
                scaling factor for y-direction focal lengths and optical centers
            x_offsets : list[float]
                amount of horizontal offset to SUBTRACT from optical center
            y_offsets : list[float]
                amount of vertical offset to SUBTRACT from optical center
        Returns:
            torch.Tensor[float32] : 3 x 3 adjusted camera intrinsics
        '''

        # for i, intrinsics in enumerate(intrinsics_arr):

        #     length = len(intrinsics)

        #     x_scales = [x_scales] * length if isinstance(x_scales, float) else x_scales
        #     y_scales = [y_scales] * length if isinstance(y_scales, float) else y_scales

        #     x_offsets = [x_offsets] * length if isinstance(x_offsets, float) else x_offsets
        #     y_offsets = [y_offsets] * length if isinstance(y_offsets, float) else y_offsets

            # for b, K in enumerate(intrinsics):
            #     x_scale = x_scales[b]
            #     y_scale = y_scales[b]
            #     x_offset = x_offsets[b]
            #     y_offset = y_offsets[b]

            # Scale and subtract offset
        K = intrinsics
        K[:, 0, 0] = K[:, 0, 0] * x_scales
        K[:, 0, 2] = K[:, 0, 2] * x_scales - x_offsets
        K[:, 1, 1] = K[:, 1, 1] * y_scales
        K[:, 1, 2] = K[:, 1, 2] * y_scales - y_offsets
        return K

    def recover_inputs(self, output_depth, n_height, n_width):
        height, width = output_depth.size()[-2:]
        do_padding = False if (n_height == height and n_width == width) else True
        if do_padding:
            padding_top = height - n_height
            padding_right = width - n_width
            output0, output1 = torch.chunk(output_depth, chunks=2, dim=0)
            if padding_right == 0:
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
        w_l1 = w_losses['w_l1'] if 'w_l1' in w_losses else 1.0
        w_l2 = w_losses['w_l2'] if 'w_l2' in w_losses else 1.0
        # Check if loss weighting was passed in, if not then use default weighting
        w_l1_0 = w_losses['w_l1_0'] if 'w_l1_0' in w_losses else w_l1 * 1.0
        w_l2_0 = w_losses['w_l2_0'] if 'w_l2_0' in w_losses else w_l2 * 1.0

        w_l1_1 = w_losses['w_l1_1'] * w_l1 if 'w_l1_1' in w_losses else w_l1 * 1/4
        w_l2_1 = w_losses['w_l2_1'] * w_l2 if 'w_l2_1' in w_losses else w_l2 * 1/4

        w_l1_2 = w_losses['w_l1_2'] if 'w_l1_2' in w_losses else w_l1 * 1/16
        w_l2_2 = w_losses['w_l2_2'] if 'w_l2_2' in w_losses else w_l2 * 1/16

        w_l1_3 = w_losses['w_l1_3'] if 'w_l1_3' in w_losses else w_l1 * 1/64
        w_l2_3 = w_losses['w_l2_3'] if 'w_l2_3' in w_losses else w_l2 * 1/64

        w_smoothness = w_losses['w_smoothness'] if 'w_smoothness' in w_losses else 0.0

        loss_info = {}

        # NLSPN clamps predictions during loss computation
        # output_depth = torch.clamp(output_depth, min=0, max=self.max_predict_depth)
        # ground_truth = torch.clamp(ground_truth, min=0, max=self.max_predict_depth)

        # Obtain valid values
        validity_map = torch.where(
            ground_truth > 0,
            torch.ones_like(ground_truth),
            ground_truth)

        # Compute individual losses

        l1_loss_0 = w_l1_0 * loss_utils.l1_loss(
            src=output_depth[0],
            tgt=ground_truth,
            w=validity_map)

        l2_loss_0 = w_l2_0 * loss_utils.l2_loss(
            src=output_depth[0],
            tgt=ground_truth,
            w=validity_map)

        loss = l1_loss_0 + l2_loss_0

        # Store loss info
        loss_info['l1_loss_0'] = l1_loss_0
        loss_info['l2_loss_0'] = l2_loss_0

        l1_loss_1 = w_l1_1 * loss_utils.l1_loss(
            src=output_depth[1],
            tgt=ground_truth,
            w=validity_map)

        l2_loss_1 = w_l2_1 * loss_utils.l2_loss(
            src=output_depth[1],
            tgt=ground_truth,
            w=validity_map)

        loss = loss + l1_loss_1 + l2_loss_1


        # Store loss info
        loss_info['l1_loss_1'] = l1_loss_1
        loss_info['l2_loss_1'] = l2_loss_1

        if len(output_depth) > 2:

            l1_loss_2 = w_l1_2 * loss_utils.l1_loss(
                src=output_depth[2],
                tgt=ground_truth,
                w=validity_map)

            l2_loss_2 = w_l2_2 * loss_utils.l2_loss(
                src=output_depth[2],
                tgt=ground_truth,
                w=validity_map)


            loss = loss + l1_loss_2 + l2_loss_2
            # Store loss info
            loss_info['l1_loss_2'] = l1_loss_2
            loss_info['l2_loss_2'] = l2_loss_2

        if len(output_depth) > 3:
            l1_loss_3 = w_l1_3 * loss_utils.l1_loss(
                src=output_depth[3],
                tgt=ground_truth,
                w=validity_map)
            l2_loss_3 = w_l2_3 * loss_utils.l2_loss(
                src=output_depth[3],
                tgt=ground_truth,
                w=validity_map)
            loss = loss + l1_loss_3 + l2_loss_3

            # Store loss info
            loss_info['l1_loss_3'] = l1_loss_3
            loss_info['l2_loss_3'] = l2_loss_3

        if w_smoothness > 0 and image is not None:
            loss_smoothness = w_smoothness * loss_utils.smoothness_loss_func(output_depth[0], image)
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
