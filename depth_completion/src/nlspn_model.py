import os, sys, argparse
import torch, torchvision
import utils.src.loss_utils as loss_utils
from utils.src.data_utils import inpainting
sys.path.insert(0, os.path.join('external_src', 'NLSPN', 'src'))
sys.path.insert(0, os.path.join('external_src', 'NLSPN', 'src', 'model'))
from nlspnmodel import NLSPNModel as NLSPNBaseModel


class NLSPNModel(object):
    '''
    Class for interfacing with NLSPN model

    Arg(s):
        max_predict_depth : float
            value to clamp ground truths to in computing loss
        use_pretrained : bool
            if set, then configure using legacy settings
        device : torch.device
            device to run model on
    '''

    def __init__(self, max_predict_depth=100.0, use_pretrained=False, device=torch.device('cuda')):

        # Settings to reproduce NLSPN numbers on KITTI
        args = argparse.Namespace(
            affinity='TGASS',
            affinity_gamma=0.5,
            conf_prop=True,
            from_scratch=True,
            legacy=use_pretrained,
            lr=0.001,
            max_depth=max_predict_depth,
            network='resnet34',
            preserve_input=True,
            prop_kernel=3,
            prop_time=18,
            test_only=True)
        # args = argparse.Namespace(
        #     affinity='TGASS',
        #     affinity_gamma=0.5,
        #     conf_prop=True,
        #     from_scratch=True,
        #     legacy=use_pretrained,
        #     lr=0.001,
        #     max_depth=max_depth,
        #     network='resnet34',
        #     preserve_input=True,
        #     prop_kernel=3,
        #     prop_time=256,
        #     test_only=True)

        # Instantiate depth completion model
        self.model = NLSPNBaseModel(args)
        self.use_pretrained = use_pretrained
        self.max_predict_depth = max_predict_depth

        # Move to device
        self.device = device
        self.to(self.device)

    def forward_depth(self, image, sparse_depth, intrinsics, return_all_outputs=False):
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

        # Transform inputs
        image, sparse_depth, = self.transform_inputs(image, sparse_depth)

        # Forward through the model
        sample = {
            'rgb': image,
            'dep': sparse_depth
        }

        output = self.model.forward(sample)

        output_depth = output['pred']

        # Fill in any holes with inpainting
        if not self.model.training:
            output_depth = output_depth.detach().cpu().numpy()
            output_depth = inpainting(output_depth)
            output_depth = torch.from_numpy(output_depth).to(self.device)

        if return_all_outputs:
            return [output_depth]
        else:
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

        image = image / 255.0

        for batch in range(image.shape[0]):

            image[batch, ...] = torchvision.transforms.functional.normalize(
                image[batch, ...],
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225))

        return image, sparse_depth

    def compute_loss(self, output_depth, target_depth, image=None, w_losses={}):
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
        output_depth = torch.clamp(output_depth, min=0, max=self.max_predict_depth)
        target_depth = torch.clamp(target_depth, min=0, max=self.max_predict_depth)

        # Obtain valid values
        validity_map = torch.where(
            target_depth > 0,
            torch.ones_like(target_depth),
            target_depth)

        # Compute individual losses
        l1_loss = w_l1 * loss_utils.l1_loss(
            src=output_depth,
            tgt=target_depth,
            w=validity_map)

        l2_loss = w_l2 * loss_utils.l2_loss(
            src=output_depth,
            tgt=target_depth,
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
