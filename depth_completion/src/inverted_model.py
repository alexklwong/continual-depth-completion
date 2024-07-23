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
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim

sys.path.insert(0, os.path.join('external_src', 'depth_completion'))
sys.path.insert(0, os.path.join('external_src', 'depth_completion', 'scaffnet'))
sys.path.insert(0, os.path.join('external_src', 'depth_completion', 'scaffnet', 'src'))
import losses

class InvertedModel():
    '''
    Class for model inversion that will iteratively adjust an image to produce lower loss when passed through the trained model. 
    '''

    def __init__(self,
                model_name,
                network_modules,
                min_predict_depth,
                max_predict_depth,
                crop_shape,
                n_thread,
                restore_paths,
                device=torch.device('cuda'),
                lr=1e-3):

        self.model_name = model_name
        self.network_modules = network_modules
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.device = device
        self.crop_shape = crop_shape
        self.n_thread = n_thread
        self.restore_paths = restore_paths
        self.n_output_resolution = 1

        # Build depth completion network
        self.depth_completion_model = DepthCompletionModel(
            model_name=model_name,
            network_modules=network_modules,
            min_predict_depth=min_predict_depth,
            max_predict_depth=max_predict_depth,
            device=self.device)

        # Restore model and set to evaluation mode
        self.depth_completion_model.restore_model(self.restore_paths)
        self.depth_completion_model.eval()

        # Freeze all model parameters
        for param in self.depth_completion_model.parameters():
            param.requires_grad = False

        # Example input size, adjust as necessary for your model
        self.generated_image = torch.randn((1, 3, 416, 512), device=self.device, requires_grad=True)
        # self.generated_image = torch.zeros((1, 3, 416, 512), device=self.device, requires_grad=True)
        self.generated_image = torch.nn.Parameter(self.generated_image)

        self.optimizer = torch.optim.Adam([self.generated_image], lr=lr)

    def load_single_data_point(self, image_paths, sparse_depth_paths, intrinsics_paths, ground_truth_paths):
        # Load inputs
        train_dataset = datasets.DepthCompletionMonocularTrainingDataset(
            images_paths=image_paths,
            sparse_depth_paths=sparse_depth_paths,
            intrinsics_paths=intrinsics_paths,
            random_crop_shape=self.crop_shape,
            random_crop_type=['none'])

        print(len(train_dataset))

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.n_thread,
            pin_memory=False,
            drop_last=False)

        print(len(train_dataloader))

        for train_batch in train_dataloader:

            # Fetch data
            train_batch = [
                in_.to(self.device) for in_ in train_batch
            ]

            image0, \
                image1, \
                image2, \
                sparse_depth0, \
                intrinsics = train_batch

            ground_truth0 = None

            validity_map0 = torch.where(
                sparse_depth0 > 0,
                torch.ones_like(sparse_depth0),
                sparse_depth0)

            input_image0 = image0
            input_sparse_depth0 = sparse_depth0
            input_intrinsics = intrinsics
            input_validity_map0 = validity_map0

            validity_map_depth0 = validity_map0

            # Forward pass to get target depth
            target_depth0 = self.depth_completion_model.forward_depth(
                image=input_image0,
                sparse_depth=input_sparse_depth0,
                validity_map=input_validity_map0,
                intrinsics=input_intrinsics,
                return_all_outputs=True)

            return image0, image1, image2, input_image0, input_sparse_depth0, input_intrinsics, input_validity_map0, target_depth0, validity_map_depth0

    def train_image(self, 
                    sample_path, 
                    experiment_name, 
                    image0, 
                    image1, 
                    image2, 
                    input_image0, 
                    input_sparse_depth0, 
                    input_intrinsics, 
                    input_validity_map0, 
                    target_depth0, 
                    validity_map_depth0,
                    iterations=10000):

        if not os.path.exists(sample_path + '/' + experiment_name):
            os.mkdir(sample_path + '/' + experiment_name)

        loss = torch.tensor(0)

        # input_sparse_depth0 = torch.randn((1, 1, 416, 512), device=device)
        # input_validity_map0 = torch.randn((1, 1, 416, 512), device=device)

        pose0to1 = self.depth_completion_model.forward_pose(image0, image1)
        pose0to2 = self.depth_completion_model.forward_pose(image0, image2)

        # Training loop
        for iteration in range(iterations):  # Adjust the number of iterations as needed
            self.optimizer.zero_grad()

            # Forward pass
            self.output_depth0 = self.depth_completion_model.forward_depth(
                image=self.generated_image,
                sparse_depth=input_sparse_depth0,
                validity_map=input_validity_map0,
                intrinsics=input_intrinsics,
                return_all_outputs=True)

            # pose0to1 = depth_completion_model.forward_pose(generated_image, image1)
            # pose0to2 = depth_completion_model.forward_pose(generated_image, image2)
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss.item()}")

                print("saving image...")
                self.save_tensor_as_image(self.generated_image, '{}/{}/generated_image-{}.png'.format(sample_path, experiment_name, iteration+1))
                print("saving depth...")
                self.save_depth_map(self.output_depth0[0], '{}/{}/generated_depth-{}.png'.format(sample_path, experiment_name, iteration+1))

            # Compute loss function
            loss, loss_info = self.compute_loss(
                loss_func='supervised_l1_normalized',
                target_depth=target_depth0[0],
                output_depths=self.output_depth0,
                w_supervised=100.00)
                
            loss.backward(retain_graph=True)
            
            # Update the generated image
            self.optimizer.step()

    def compute_loss(self,
                     loss_func,
                     target_depth,
                     output_depths,
                     output_uncertainties=None,
                     w_supervised=1.00):
        '''
        Computes loss function

        Arg(s):
            loss_func : list[str]
                loss functions to minimize
            target_depth : torch.Tensor[float32]
                N x 1 x H x W groundtruth target depth
            output_depths : list[torch.Tensor[float32]]
                list of N x 1 x H x W output depth
            output_uncertainties : list[torch.Tensor[float32]]
                N x 1 x H x W uncertainty
            w_supervised : float
                weight of supervised loss
        Returns:
            torch.Tensor[float32] : loss
            dict[str, torch.Tensor[float32]] : loss related infor
        '''

        if not isinstance(output_depths, list):
            output_depths = [output_depths]

        if not isinstance(output_uncertainties, list):
            output_uncertainties = [output_uncertainties]

        target_depth = torch.where(
            target_depth > self.max_predict_depth,
            torch.full_like(target_depth, fill_value=self.max_predict_depth),
            target_depth)

        validity_map_target_depth = torch.where(
            target_depth > 0.0,
            torch.ones_like(target_depth),
            torch.zeros_like(target_depth))

        shape = target_depth.shape[-2:]

        loss_supervised = 0.0

        for s, (output_depth, output_uncertainty) in enumerate(zip(output_depths, output_uncertainties)):

            w = 1.0 / (4 ** (float(self.n_output_resolution) - float(s) - 1))

            output_depth = torch.nn.functional.interpolate(
                input=output_depth,
                size=shape,
                mode='bilinear',
                align_corners=True)

            if 'supervised_l1_normalized' in loss_func:
                loss_supervised += w * losses.l1_loss_func(
                    src=output_depth,
                    tgt=target_depth,
                    w=validity_map_target_depth,
                    normalize=True)
            elif 'supervised_l1' in loss_func:
                loss_supervised += w * losses.l1_loss_func(
                    src=output_depth,
                    tgt=target_depth,
                    w=validity_map_target_depth)

        loss = w_supervised * loss_supervised

        loss_info = {
            'loss' : loss
        }

        return loss, loss_info

    def save_depth_map(self, depth_tensor, file_path):
        """
        Saves a depth map tensor as a PNG image to the specified file path.

        Args:
        depth_tensor (torch.Tensor): A torch tensor of shape (1, 1, 416, 512).
        file_path (str): The path where the image will be saved.
        """
        
        # Normalize the depth map to the range [0, 1] for better visualization
        depth_tensor = depth_tensor.squeeze(0)  # Remove batch dimension
        min_val = depth_tensor.min()
        max_val = depth_tensor.max()
        if max_val > min_val:  # Avoid division by zero
            depth_tensor = (depth_tensor - min_val) / (max_val - min_val)
        
        # Save the image
        save_image(depth_tensor, file_path)

    def save_tensor_as_image(self, tensor, file_path):
        """
        Saves a tensor as an image to the specified file path.

        Args:
        tensor (torch.Tensor): A torch tensor of shape (1, 3, 416, 512).
        file_path (str): The path where the image will be saved.
        """
        
        # Check and adjust the range if necessary
        if tensor.dtype == torch.uint8:
            tensor = tensor.float() / 255  # Normalize if the tensor is uint8
        elif tensor.max() > 1:
            tensor = tensor / tensor.max()  # Scale down if the max exceeds 1

        # Remove the batch dimension (1 in the shape) as save_image expects (C, H, W)
        tensor = tensor.squeeze(0)
        
        # Save the image
        save_image(tensor, file_path)