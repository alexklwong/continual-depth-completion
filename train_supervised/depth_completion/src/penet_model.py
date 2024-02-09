import os, sys
import torch, torchvision
import utils.src.loss_utils as loss_utils
sys.path.insert(0, os.path.join('external_src', 'PENet_ICRA2021'))
from model_modified import PENet_C2
import CoordConv


class PENetModel(object):
    '''
    Class for interfacing with PENet model

    Arg(s):
        max_predict_depth : float
            value to clamp ground truths to in computing loss
        device : torch.device
            device to run model on
    '''

    def __init__(self, max_predict_depth=5.0, device=torch.device('cuda')):

        # Instantiate depth completion model
        network_model = 'pe'
        convolutional_layer_encoding = 'xyz'
        dilation_rate = 2

        self.model = PENet_C2(
            convolutional_layer_encoding=convolutional_layer_encoding,
            network_model=network_model,
            dilation_rate=dilation_rate)

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

        _, _, og_height, og_width = image.shape

        image, sparse_depth, intrinsics = self.transform_inputs(image, sparse_depth, intrinsics)

        _, _, height, width = image.shape

        position = CoordConv.AddCoordsNp(height, width)
        position = torch.from_numpy(position.call())
        position = torch.unsqueeze(position, dim=0)
        position = position.permute(0, 3, 1, 2)
        position = position.to(self.device)

        position = position.repeat((image.shape[0], 1, 1, 1))

        output = self.model.forward(image, sparse_depth, position, intrinsics)

        # take_avg() handles not taking the average when shapes are compatible
        if not self.model.training:
            output = torch.clamp(output, min=0, max=self.max_predict_depth)
            final_output = self.take_avg(output, og_height, og_width)
        else:
            final_output = output

        if return_all_outputs:
            return [final_output]
        else:
            return final_output

    def transform_inputs(self, image, sparse_depth, intrinsics):
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

        # If image shape is already compatible with network, simply return inputs
        _, c, h, w = image.shape

        if h % 32 == 0 and w % 32 == 0:
            return image, sparse_depth, intrinsics

        if not self.model.training:
            image_list = []
            sparse_depth_list = []
            intrinsics_list = []

            # Create top, bottom, left, right adjusted crops
            for i in range(4):

                if i == 0:
                    x_start = 0
                    y_start = 0
                elif i == 1:
                    x_start = w - 1216
                    y_start = h - 352
                elif i == 2:
                    x_start = 0
                    y_start = h - 352
                else:
                    x_start = w - 1216
                    y_start = 0

                x_end = x_start + 1216
                y_end = y_start + 352

                # Adjust intrinsics based on crop position
                crop_intrinsics = torch.tensor([[0.0, 0.0, -x_start],
                                                [0.0, 0.0, -y_start],
                                                [0.0, 0.0, 0.0]],
                                                dtype=torch.float32,
                                                device=intrinsics.device)

                image_cropped = image[:, :, y_start:y_end, x_start:x_end]
                sparse_depth_cropped = sparse_depth[:, :, y_start:y_end, x_start:x_end]
                intrinsics_cropped = intrinsics + crop_intrinsics

                image_list.append(image_cropped)
                sparse_depth_list.append(sparse_depth_cropped)
                intrinsics_list.append(intrinsics_cropped)

            # Concatenate along batch for a single forward pass
            image = torch.cat(image_list, dim=0)
            sparse_depth = torch.cat(sparse_depth_list, dim=0)
            intrinsics = torch.cat(intrinsics_list, dim=0)

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
        w_l2 = w_losses['w_l2'] if 'w_l2' in w_losses else 1.0
        w_smoothness = w_losses['w_smoothness'] if 'w_smoothness' in w_losses else 0.0

        loss_info = {}

        if isinstance(output_depth, list):
            output_depth = output_depth[0]

        # Clamp ground truth values
        ground_truth = torch.clamp(ground_truth, min=0, max=self.max_predict_depth)

        # Obtain valid values
        validity_map = torch.where(
            ground_truth > 0,
            torch.ones_like(ground_truth),
            ground_truth)

        l2_loss = w_l2 * loss_utils.l2_loss(
            src=output_depth,
            tgt=ground_truth,
            w=validity_map)

        loss = l2_loss

        # Store loss info
        loss_info['l2_loss'] = l2_loss

        if w_smoothness > 0 and image is not None:
            loss_smoothness = w_smoothness * loss_utils.smoothness_loss_func(output_depth, image)
            loss_info['loss_smoothness'] = loss_smoothness

            loss = loss + loss_smoothness

        # Store loss info
        loss_info['loss'] = loss

        return l2_loss, loss_info

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
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
            self.model.module.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint['model'])

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
                'model' : self.model.module.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'train_step' : step
            }
        else:
            checkpoint = {
                'model' : self.model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'train_step' : step
            }

        torch.save(checkpoint, checkpoint_path)

    def take_avg(self, output, height, width):
        '''
        If validating by anchoring in each corner, compute average and output final prediction

        Arg(s):
            output : torch.Tensor[float32]
                N x 1 x out_H x out_W tensor; output of model
                if N == 1 then do not average
            height : int
                original image height
            width : int
                original image width

        Returns:
            torch.Tensor[float32] : N x 1 x H x W final output (N should be 1 in validation)
        '''

        n_batch, _, out_height, out_width = output.shape

        # If in validation without anchoring 4 corners for full output, simply return output
        if n_batch == 1 and out_height == height and out_width == width:
            return output

        output_list = []
        validity_map_list = []

        for i in range(4):

            output_tmp = torch.zeros((1, height, width), device=output.device)
            validity_map = torch.zeros((1, height, width), device=output.device)

            if i == 0:
                x_start = 0
                y_start = 0
            elif i == 1:
                x_start = width - 1216
                y_start = height - 352
            elif i == 2:
                x_start = 0
                y_start = height - 352
            else:
                x_start = width - 1216
                y_start = 0

            x_end = x_start + 1216
            y_end = y_start + 352

            # Collect positions that have valid predictions
            output_tmp[..., y_start:y_end, x_start:x_end] = output[i]
            validity_map = torch.where(output_tmp > 0,
                torch.ones_like(validity_map),
                torch.zeros_like(validity_map))

            output_list.append(output_tmp)
            validity_map_list.append(validity_map)

        # Average over the overlaps for 4 x h x w
        output_full_size = torch.cat(output_list, dim=0)
        validity_maps = torch.cat(validity_map_list, dim=0)

        final_output = torch.sum(output_full_size, dim=0, keepdims=True) / torch.sum(validity_maps, dim=0, keepdims=True)
        final_output = final_output.unsqueeze(0)

        return final_output
