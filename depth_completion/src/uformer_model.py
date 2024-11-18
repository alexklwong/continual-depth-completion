import os, sys
import torch
from outlier_removal import OutlierRemoval

sys.path.insert(0, os.path.join('depth_completion', 'kbformer'))
# sys.path.insert(0, os.path.join('external_src', 'NLSPN', 'src', 'model'))
from uformer import Uformer as UformerBaseModel
sys.path.insert(0, os.path.join('depth_completion', 'kbformer', 'posenet'))
from posenet_model import PoseNetModel as PoseNet

class Uformer_Model(object):
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

    def __init__(self, 
                 dataset_name='kitti',
                 network_modules=['depth', 'pose'],
                 dec_bn=False,
                 min_predict_depth=1.0,
                 max_predict_depth=100.0,
                 device=torch.device('cuda')):
        
        self.network_modules = network_modules

        self.model_depth = UformerBaseModel(img_size=256,
                                            dd_in=3,
                                            embed_dim=32,
                                            win_size=4,
                                            token_projection='linear',
                                            token_mlp='leff',
                                            conv_dec=True,
                                            dec_bn=dec_bn,
                                            modulator=True,
                                            min_predict_depth=min_predict_depth,
                                            max_predict_depth=max_predict_depth,
                                            device=device)
        
        if 'pose' in network_modules:
            self.model_pose = PoseNet(
                encoder_type='resnet18',
                rotation_parameterization='axis',
                weight_initializer='xavier_normal',
                activation_func='relu',
                device=device)
        else:
            self.model_pose = None

        self.outlier_removal = OutlierRemoval(
            kernel_size=7,
            threshold=1.5)

        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth

        # Move to device
        self.device = device
        self.to(self.device)
        self.eval()


    def forward_depth_encoder(self, image, sparse_depth, validity_map, intrinsics, image_prompts=None, depth_prompts=None, return_all_outputs=False):
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

        latent, skips, shape = self.model_depth.forward_encoder(x=torch.cat([image, sparse_depth, validity_map], dim=1), n_height=n_height, n_width=n_width, intrinsics=intrinsics,
                                                                image_prompts=image_prompts, depth_prompts=depth_prompts)
        return latent, skips, shape
    
        
    def forward_depth_decoder(self, latent, skips, shape, return_all_outputs=False):
        
        output, n_height, n_width, second_last = self.model_depth.forward_decoder(latent, skips, shape)
        
        # Linear output
        if isinstance(output, list):
            for i in range(len(output)):
                output[i] = self.recover_inputs(output[i], n_height, n_width)
            if not self.model_depth.training:
                return output[0]
            else:
                return output
        else:
            return self.recover_inputs(output, n_height, n_width), second_last
        

    def forward_pose(self, image0, image1):
        '''
        Forwards a pair of images through the network to output pose from time 0 to 1

        Arg(s):
            image0 : torch.Tensor[float32]
                N x C x H x W tensor
            image1 : torch.Tensor[float32]
                N x C x H x W tensor
        Returns:
            torch.Tensor[float32] : N x 4 x 4  pose matrix
        '''

        assert self.model_pose is not None

        # Normalize image between [0, 1]
        image0, image1 = [
            image / 255.0 for image in [image0, image1] if torch.max(image) > 1.0
        ]

        return self.model_pose.forward(image0, image1)
    

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

    def compute_loss(self,
                     image0,
                     image1,
                     image2,
                     output_depth0,
                     sparse_depth0,
                     validity_map_depth0,
                     validity_map_image0,
                     intrinsics,
                     pose0to1,
                     pose0to2,
                     w_losses):
        '''
        Computes loss function
        l = w_{ph}l_{ph} + w_{sz}l_{sz} + w_{sm}l_{sm}

        Arg(s):
            image0 : torch.Tensor[float32]
                N x 3 x H x W image at time step t
            image1 : torch.Tensor[float32]
                N x 3 x H x W image at time step t-1
            image2 : torch.Tensor[float32]
                N x 3 x H x W image at time step t+1
            output_depth0 : list[torch.Tensor[float32]]
                list of N x 1 x H x W output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                N x 1 x H x W sparse depth at time t
            validity_map_depth0 : torch.Tensor[float32]
                N x 1 x H x W validity map of sparse depth at time t
            validity_map_image0 : torch.Tensor[float32]
                N x 1 x H x W validity map of image at time t
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 camera intrinsics matrix
            pose0to1 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t-1
            pose0to2 : torch.Tensor[float32]
                N x 4 x 4 relative pose from image at time t to t+1
            w_color : float
                weight of color consistency term
            w_structure : float
                weight of structure consistency term (SSIM)
            w_sparse_depth : float
                weight of sparse depth consistency term
            w_smoothness : float
                weight of local smoothness term
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : dictionary of loss related tensors
        '''

        # Check if loss weighting was passed in, if not then use default weighting
        w_color = w_losses['w_color'] if 'w_color' in w_losses else 0.15
        w_structure = w_losses['w_structure'] if 'w_structure' in w_losses else 0.95
        w_sparse_depth = w_losses['w_sparse_depth'] if 'w_sparse_depth' in w_losses else 0.60
        w_smoothness = w_losses['w_smoothness'] if 'w_smoothness' in w_losses else 0.04

        # Unwrap from list
        output_depth0 = output_depth0[0]
        validity_map_image0 = validity_map_image0[0] if validity_map_image0 is not None else None

        # Remove outlier points and update sparse depth and validity map
        filtered_sparse_depth0, \
            filtered_validity_map_depth0 = self.outlier_removal.remove_outliers(
                sparse_depth=sparse_depth0,
                validity_map=validity_map_depth0)

        # Normalize images between [0, 1]
        image0, image1, image2 = [
            image / 255.0 for image in [image0, image1, image2] if torch.max(image) > 1.0
        ]

        loss, loss_info = self.model_depth.compute_loss(
            image0=image0,
            image1=image1,
            image2=image2,
            output_depth0=output_depth0,
            sparse_depth0=filtered_sparse_depth0,
            validity_map_depth0=filtered_validity_map_depth0,
            validity_map_image0=validity_map_image0,
            intrinsics=intrinsics,
            pose0to1=pose0to1,
            pose0to2=pose0to2,
            w_color=w_color,
            w_structure=w_structure,
            w_sparse_depth=w_sparse_depth,
            w_smoothness=w_smoothness)

        return loss, loss_info


    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        parameters = []
        for name, param in self.model_depth.named_parameters():
            if param.requires_grad:
                parameters.append(param)

        parameters = torch.nn.ParameterList(parameters)

        if 'pose' in self.network_modules:
            parameters = parameters + list(self.model_pose.parameters())

        return parameters
    
    def parameters_depth(self):
        '''
        Fetches model parameters for depth network modules

        Returns:
            list[torch.Tensor[float32]] : list of model parameters for depth network modules
        '''

        return self.model_depth.parameters()

    def parameters_pose(self):
        '''
        Fetches model parameters for pose network modules

        Returns:
            list[torch.Tensor[float32]] : list of model parameters for pose network modules
        '''

        if 'pose' in self.network_modules:
            return self.model_pose.parameters()
        else:
            raise ValueError('Unsupported pose network architecture: {}'.format(self.network_modules))

    def train(self):
        '''
        Sets model to training mode
        '''

        self.model_depth.train()

        if 'pose' in self.network_modules:
            self.model_pose.train()

    def eval(self):
        '''
        Sets model to evaluation mode
        '''

        self.model_depth.eval()

        if 'pose' in self.network_modules:
            self.model_pose.eval()

    def to(self, device):
        '''
        Move model to a device

        Arg(s):
            device : torch.device
                device to use
        '''

        self.device = device

        self.model_depth.to(device)

        if 'pose' in self.network_modules:
            self.model_pose.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.model_depth = torch.nn.DataParallel(self.model)


    def restore_model(self,
                      model_depth_restore_path,
                      model_pose_restore_path=None,
                      optimizer_depth=None,
                      optimizer_pose=None):
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

        checkpoint = torch.load(model_depth_restore_path, map_location=self.device)

        if isinstance(self.model_depth, torch.nn.DataParallel):
            self.model_depth.module.load_state_dict(checkpoint['net'])
        else:
            self.model_depth.load_state_dict(checkpoint['net'])

        if optimizer_depth is not None and 'optimizer' in checkpoint.keys():
            optimizer_depth.load_state_dict(checkpoint['optimizer'])

        try:
            train_step = checkpoint['train_step']
        except Exception:
            train_step = 0

        if 'pose' in self.network_modules and model_pose_restore_path is not None:
            _, optimizer_pose = self.model_pose.restore_model(
                model_pose_restore_path,
                optimizer_pose)

        return train_step, optimizer_depth, optimizer_pose


    def save_model(self,
                   model_depth_checkpoint_path,
                   step,
                   optimizer_depth,
                   model_pose_checkpoint_path=None,
                   optimizer_pose=None):
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

        if isinstance(self.model_depth, torch.nn.DataParallel) or isinstance(self.model_depth, torch.nn.parallel.DistributedDataParallel):
            checkpoint = {
                'net': self.model_depth.module.state_dict(),
                'optimizer': optimizer_depth.state_dict(),
                'train_step': step
            }
        else:
            checkpoint = {
                'net': self.model_depth.state_dict(),
                'optimizer': optimizer_depth.state_dict(),
                'train_step': step
            }

        torch.save(checkpoint, model_depth_checkpoint_path)

        if 'pose' in self.network_modules and model_pose_checkpoint_path is not None:
            self.model_pose.save_model(
                checkpoint_path=model_pose_checkpoint_path,
                step=step,
                optimizer=optimizer_pose)
