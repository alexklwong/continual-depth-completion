import os, torch, torchvision
import torch.nn.functional as F
from utils.src import log_utils, net_utils
from continual_learning_losses import token_loss, ewc_loss, lwf_loss
import net_utils


class DepthCompletionModel(object):
    '''
    Wrapper class for all external depth completion models

    Arg(s):
        model_name : str
            depth completion model to use
        network_modules : list[str]
            network modules to build for model
        min_predict_depth : float
            minimum depth to predict
        max_predict_depth : float
            maximum depth to predict
        frozen : bool
            for TokenCDC, freeze the model if True
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 model_name,
                 network_modules,
                 min_predict_depth,
                 max_predict_depth,
                 key_token_pool_size,
                 unfrozen=False,
                 device=torch.device('cuda')):

        self.model_name = model_name
        self.network_modules = network_modules
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
        self.key_token_pool_size = key_token_pool_size  # TokenCDC
        self.frozen = not unfrozen  # TokenCDC: freeze model if unfrozen=False
        self.device = device

        # Parse dataset name
        if 'kitti' in model_name:
            dataset_name = 'kitti'
        elif 'vkitti' in model_name:
            dataset_name = 'vkitti'
        elif 'void' in model_name:
            dataset_name = 'void'
        elif 'scenenet' in model_name:
            dataset_name = 'scenenet'
        elif 'nyu_v2' in model_name:
            dataset_name = 'nyu_v2'
        else:
            dataset_name = 'kitti'

        if 'kbnet' in model_name:
            from kbnet_models import KBNetModel

            self.model = KBNetModel(
                dataset_name=dataset_name,
                network_modules=network_modules,
                min_predict_depth=min_predict_depth,
                max_predict_depth=max_predict_depth,
                device=device)
        elif 'scaffnet' in model_name:
            from scaffnet_models import ScaffNetModel

            self.model = ScaffNetModel(
                dataset_name=dataset_name,
                network_modules=network_modules,
                min_predict_depth=min_predict_depth,
                max_predict_depth=max_predict_depth,
                device=device)
        elif 'fusionnet' in model_name:
            from fusionnet_models import FusionNetModel

            self.model = FusionNetModel(
                dataset_name=dataset_name,
                network_modules=network_modules,
                min_predict_depth=min_predict_depth,
                max_predict_depth=max_predict_depth,
                device=device)
        elif 'voiced' in model_name:
            from voiced_models import VOICEDModel

            self.model = VOICEDModel(
                network_modules=network_modules,
                min_predict_depth=min_predict_depth,
                max_predict_depth=max_predict_depth,
                device=device)
        else:
            raise ValueError('Unsupported depth completion model: {}'.format(model_name))

        if 'ewc' in network_modules:
            self.ewc = True
            self.prev_fisher = None
        else:
            self.ewc = False

        if 'fisher' in network_modules:
            self.calculate_fisher_enabled = True
            self.fisher = None
            self.epoch_fisher = net_utils.init_fisher(self.model.parameters_depth())
        else:
            self.calculate_fisher_enabled = False

        # TokenCDC: Freeze depth model parameters if unfrozen=False
        if self.frozen:
            for param in self.model.parameters_depth():
                param.requires_grad = False
            for param in self.model.parameters_pose():
                param.requires_grad = False

        # TokenCDC: initialize two ParameterDicts to store learnable key & token POOLS for each dataset
        self.image_key_weights = torch.nn.ParameterDict() 
        self.depth_key_weights = torch.nn.ParameterDict() 
        self.image_token_pools = torch.nn.ParameterDict()
        self.depth_token_pools = torch.nn.ParameterDict() 
        self.new_params = []  # LIST OF PARAMS to be added to the depth optimizer!
 

    def add_new_key_token_pool(self, dataset_uid, image_key_dim, depth_key_dim, token_dim):
        '''
        Add and return token for a new unseen dataset

        Arg(s):
            dataset_uid : str
                unique id of dataset
            n_keys : int
                number of keys/tokens in the pool
            key_dim : int
                dimension of image/depth features
            token_dim : int
                dimension of fused latent space
        Returns:
            torch.Tensor[float32] : added token
        '''
        new_image_key_weight = torch.nn.Parameter(torch.zeros((image_key_dim, token_dim),
                                        device=self.device),
                                        requires_grad=True)
        new_depth_key_weight = torch.nn.Parameter(torch.zeros((depth_key_dim, token_dim),
                                        device=self.device),
                                        requires_grad=True)
        new_image_token_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, token_dim),
                                        device=self.device),
                                        requires_grad=True)
        new_depth_token_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, token_dim),
                                        device=self.device),
                                        requires_grad=True) 
        
        self.image_key_weights[dataset_uid] = new_image_key_weight
        self.depth_key_weights[dataset_uid] = new_depth_key_weight
        self.image_token_pools[dataset_uid] = new_image_token_pool
        self.depth_token_pools[dataset_uid] = new_depth_token_pool

        self.new_params.append(new_image_key_weight)  # to be added to the optimizer
        self.new_params.append(new_depth_key_weight)  # to be added to the optimizer
        self.new_params.append(new_image_token_pool)  # to be added to the optimizer
        self.new_params.append(new_depth_token_pool)  # to be added to the optimizer

        assert set(self.image_key_weights.keys()) == set(self.depth_key_weights.keys()), "Image/Depth Key pools must have same keys!"
        assert set(self.image_key_weights.keys()) == set(self.depth_token_pools.keys()), "Key and Token pools must have same keys!"
        return new_image_key_weight, new_depth_key_weight, new_image_token_pool, new_depth_token_pool


    def forward_depth(self, 
                      image, 
                      sparse_depth, 
                      validity_map, 
                      dataset_uid,
                      intrinsics=None, 
                      return_all_outputs=False):
        '''
        Forwards stereo pair through network

        Arg(s):
            image : torch.Tensor[float32]
                N x 3 x H x W image
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W projected sparse point cloud (depth map)
            validity_map : torch.Tensor[float32]
                N x 1 x H x W valid locations of projected sparse point cloud
            intrinsics : torch.Tensor[float32]
                N x 3 x 3 intrinsic camera calibration matrix
            dataset_uid : str
                unique id of dataset
            return_all_outputs : bool
                if set, then return list of N x 1 x H x W depth maps else a single N x 1 x H x W depth map
        Returns:
            list[torch.Tensor[float32]] : a single or list of N x 1 x H x W outputs
        '''

        # Encoder Forward Pass
        latent, skips, shape, image_features, depth_features = self.model.forward_depth_encoder(
            image, 
            sparse_depth, 
            validity_map, 
            intrinsics)

        ##### BEGIN TokenCDC Implementation 

        # TokenCDC TEST: number of learnable params
        # count = 0
        # count += sum(p.numel() for p in self.token_pools.values() if p.requires_grad)
        # count += sum(p.numel() for p in self.model.parameters_depth() if p.requires_grad)
        # print("Learnable parameters in the model: {}".format(count))
        # print("Current token keys: {}".format(self.token_pools.keys()))

        # QUERY function (deterministic and frozen) for both image and depth features
        Ni, image_key_dim, Hi, Wi = image_features.shape
        Nd, depth_key_dim, Hd, Wd = depth_features.shape
        assert Ni == Nd and Hi == Hd and Wi == Wd, "Image and depth features must have the same non-channel dims!"
        # Pad to multiple of 32, then average pool
        DOWNSAMPLE = 32
        pad_H = (DOWNSAMPLE - Hi % DOWNSAMPLE) if Hi % DOWNSAMPLE != 0 else 0
        pad_W = (DOWNSAMPLE - Wi % DOWNSAMPLE) if Wi % DOWNSAMPLE != 0 else 0
        pad_left = pad_W // 2
        pad_right = pad_W - pad_left
        pad_top = pad_H // 2
        pad_bottom = pad_H - pad_top
        padded_image_features = F.pad(image_features, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        image_queries = F.avg_pool2d(padded_image_features, kernel_size=(DOWNSAMPLE, DOWNSAMPLE), stride=(DOWNSAMPLE, DOWNSAMPLE))
        padded_depth_features = F.pad(depth_features, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')
        depth_queries = F.avg_pool2d(padded_depth_features, kernel_size=(DOWNSAMPLE, DOWNSAMPLE), stride=(DOWNSAMPLE, DOWNSAMPLE))
        # Flatten spatial dimensions
        new_H = (Hi + pad_H) // DOWNSAMPLE
        new_W = (Wi + pad_W) // DOWNSAMPLE
        image_queries = image_queries.permute(0, 2, 3, 1).view(Ni, new_H*new_W, image_key_dim)
        depth_queries = depth_queries.permute(0, 2, 3, 1).view(Nd, new_H*new_W, depth_key_dim)

        # Get key weights/token pools
        N, token_dim, H, W = latent.shape
        assert H == new_H and W == new_W, "Latent and query spatial dimensions must match!"
        if dataset_uid not in self.image_key_weights:
            curr_image_key_weight, curr_depth_key_weight, curr_image_token_pool, curr_depth_token_pool = \
                self.add_new_key_token_pool(dataset_uid, image_key_dim, depth_key_dim, token_dim)
        else:
            curr_image_key_weight, curr_depth_key_weight, curr_image_token_pool, curr_depth_token_pool = \
                self.image_key_weights[dataset_uid], self.depth_key_weights[dataset_uid], \
                self.image_token_pools[dataset_uid], self.depth_token_pools[dataset_uid]

        # Compute the KEYS for image using FROZEN token pool
        curr_image_token_pool_no_grad = curr_image_token_pool.detach().clone()
        curr_image_token_pool_no_grad = curr_image_token_pool_no_grad.transpose(-2, -1)
        image_keys = torch.matmul(curr_image_key_weight, curr_image_token_pool_no_grad)
        # Same thing for KEYS for depth
        curr_depth_token_pool_no_grad = curr_depth_token_pool.detach().clone()
        curr_depth_token_pool_no_grad = curr_depth_token_pool_no_grad.transpose(-2, -1)
        depth_keys = torch.matmul(curr_depth_key_weight, curr_depth_token_pool_no_grad)

        # Compute TOKENS using attention
        image_attention = torch.matmul(image_queries, image_keys) / torch.sqrt(torch.tensor(image_key_dim, device=self.device, dtype=torch.float32))
        image_attention = F.softmax(image_attention, dim=-1)
        depth_attention = torch.matmul(depth_queries, depth_keys) / torch.sqrt(torch.tensor(depth_key_dim, device=self.device, dtype=torch.float32))
        depth_attention = F.softmax(depth_attention, dim=-1)
        image_tokens = torch.matmul(image_attention, curr_image_token_pool).view(N, H, W, token_dim).permute(0, 3, 1, 2)
        depth_tokens = torch.matmul(depth_attention, curr_depth_token_pool).view(N, H, W, token_dim).permute(0, 3, 1, 2)
        final_tokens = (image_tokens + depth_tokens) / 2.0

        # CONCAT tokens to latent and then 1x1 conv back to latent dims
        latent_with_tokens = latent + final_tokens

        ##### END TokenCDC Implementation

        # Decoder Forward Pass
        output = self.model.forward_depth_decoder(
                    latent_with_tokens,
                    skips,
                    shape,
                    return_all_outputs)

        return output


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

        return self.model.forward_pose(image0, image1)


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
                     queries=None,
                     keys=None,
                     domain_incremental=False,
                     ground_truth0=None,
                     supervision_type='unsupervised',
                     w_losses={},
                     frozen_model=None):
        '''
        Call model's compute loss function

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
            queries : torch.Tensor[float32]
                N x key_dim x H x W queries (FROZEN)
            keys : torch.Tensor[float32]
                N x key_dim x H x W keys
            ground_truth0 : torch.Tensor[float32]
                N x 1 x H x W ground truth depth at time t
            supervision_type : str
                type of supervision for training
            w_losses : dict[str, float]
                dictionary of weights for each loss
            frozen_model : object
                instance of pretrained model frozen for loss computations
        Returns:
            float : loss averaged over the batch
            dict[str, float] : loss info
        '''

        if supervision_type == 'supervised':
            loss, loss_info = self.model.compute_loss(
                target_depth=ground_truth0,
                output_depth=output_depth0)
        elif supervision_type == 'unsupervised':
            loss, loss_info = self.model.compute_loss(
                image0=image0,
                image1=image1,
                image2=image2,
                output_depth0=output_depth0,
                sparse_depth0=sparse_depth0,
                validity_map_depth0=validity_map_depth0,
                validity_map_image0=validity_map_image0,
                intrinsics=intrinsics,
                pose0to1=pose0to1,
                pose0to2=pose0to2,
                w_losses=w_losses)
        else:
            raise ValueError('Unsupported supervision type: {}'.format(supervision_type))

        # TokenCDC LEGACY: Loss between queries/keys and between keys in the key pool
        if 'w_token' in w_losses:
            loss_token = token_loss(
                            queries=queries,
                            keys=keys,
                            key_pools=self.key_pools,
                            lambda_token=w_losses['w_token'],
                            domain_incremental=domain_incremental)

            loss += loss_token
            loss_info['loss_token'] = loss_token

        if 'w_ewc' in w_losses:
            loss_ewc = ewc_loss(
                current_parameters=self.model.parameters_depth(),
                frozen_parameters=frozen_model.parameters_depth(),
                lambda_ewc=w_losses['w_ewc'],
                fisher_info=self.prev_fisher)

            loss += loss_ewc
            loss_info['loss_ewc'] = loss_ewc

        if 'w_lwf' in w_losses:
            #to debug this, I modified a few lines in random crop, need to fix back later
            frozen_model_output_depth0 = frozen_model.forward_depth(image0, sparse_depth0, validity_map_depth0, intrinsics, return_all_outputs=True)

            loss_lwf = lwf_loss(output_depth0, frozen_model_output_depth0, w_losses['w_lwf'])
            loss += loss_lwf
            loss_info['loss_lwf'] = loss_lwf

        return loss, loss_info


    def get_new_params(self):
        '''
        Returns the list of new parameters added to the model and resets the list

        Returns:
            list[torch.Tensor[float32]] : list of new parameters
        '''
        new_params = self.new_params
        self.new_params = []  # Clear the list after returning the new params
        # TokenCDC TEST
        #print("Number of NEW learnable parameters: {}\n\n\n\n".format(sum(p.numel() for p in new_params)))
        return new_params


    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return list(self.model.parameters())

    def parameters_depth(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return list(self.model.parameters_depth())

    def parameters_pose(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.model.parameters_pose()


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
        self.tokens = self.tokens.to(device)

    def data_parallel(self):
        '''
        Allows multi-gpu split along batch
        '''

        self.model.data_parallel()

    def distributed_data_parallel(self, rank):
        '''
        Allows multi-gpu split along batch with 'torch.nn.parallel.DistributedDataParallel'
        '''

        self.model.distributed_data_parallel(rank)

    def convert_syncbn(self):
        '''
        Convert BN layers to SyncBN layers.
        SyncBN merge the BN layer weights in every backward step.
        '''

        self.model.convert_syncbn()

    def restore_model(self,
                      restore_paths,
                      optimizer_depth=None,
                      optimizer_pose=None,
                      frozen_model=False):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_paths : list[str]
                path to model weights, 1st for depth model and 2nd for pose model (if exists)
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            int : training step
            torch.optimizer : optimizer for depth or None if no optimizer is passed in
            torch.optimizer : optimizer for pose or None if no optimizer is passed in
        '''

        # TokenCDC: Restore ALL TokenCDC params
        if 'new_pool_size' not in self.network_modules:
            # Get state dicts
            checkpoint = torch.load(restore_paths[-1], map_location=self.device)
            image_key_weights_state_dict = checkpoint['image_key_weights']
            depth_key_weights_state_dict = checkpoint['depth_key_weights']
            image_token_pools_state_dict = checkpoint['image_token_pools']
            depth_token_pools_state_dict = checkpoint['depth_token_pools']

            # Get the keys from the model's param_dicts and the saved state_dicts
            image_key_weights_state_dict_keys = set(image_key_weights_state_dict.keys())
            image_key_weights_curr_keys = set(self.image_key_weights.keys())
            depth_key_weights_state_dict_keys = set(depth_key_weights_state_dict.keys())
            depth_key_weights_curr_keys = set(self.depth_key_weights.keys())
            image_token_pools_state_dict_keys = set(image_token_pools_state_dict.keys())
            image_token_pools_curr_keys = set(self.image_token_pools.keys())
            depth_token_pools_state_dict_keys = set(depth_token_pools_state_dict.keys())
            depth_token_pools_curr_keys = set(self.depth_token_pools.keys())

            # Identify missing keys (keys in state_dict but not in model)
            image_key_weights_missing_keys = image_key_weights_state_dict_keys - image_key_weights_curr_keys
            depth_key_weights_missing_keys = depth_key_weights_state_dict_keys - depth_key_weights_curr_keys
            image_token_pools_missing_keys = image_token_pools_state_dict_keys - image_token_pools_curr_keys
            depth_token_pools_missing_keys = depth_token_pools_state_dict_keys - depth_token_pools_curr_keys
            assert image_key_weights_missing_keys == depth_key_weights_missing_keys, "Image/Depth Key pools must have the same keys!"
            assert image_key_weights_missing_keys == depth_token_pools_missing_keys, "Key and Token pools must have the same keys!"

            # Add pools for the missing keys (and add to new_params list to be added to optimizer)
            for mk in image_key_weights_missing_keys:
                self.add_new_key_token_pool(mk,
                                            image_key_weights_state_dict[mk].shape[0],
                                            depth_key_weights_state_dict[mk].shape[0],
                                            image_token_pools_state_dict[mk].shape[1])

            # Now, load the state dicts
            self.image_key_weights.load_state_dict(image_key_weights_state_dict)
            self.depth_key_weights.load_state_dict(depth_key_weights_state_dict)
            self.image_token_pools.load_state_dict(image_token_pools_state_dict)
            self.depth_token_pools.load_state_dict(depth_token_pools_state_dict)
            print("ALL TokenCDC PARAMS RESTORED!\n\n\n\n\n\n")

        if 'kbnet' in self.model_name:
            return self.model.restore_model(
                model_depth_restore_path=restore_paths[0],
                model_pose_restore_path=restore_paths[1] if len(restore_paths) > 1 else None,
                optimizer_depth=optimizer_depth,
                optimizer_pose=optimizer_pose)
        elif 'scaffnet' in self.model_name:
            return self.model.restore_model(
                restore_path=restore_paths[0],
                optimizer=optimizer_depth)
        elif 'fusionnet' in self.model_name:
            if 'initialize_scaffnet' in self.network_modules:
                self.model.scaffnet_model.restore_model(
                    restore_path=restore_paths[0])
                return 0, optimizer_depth, optimizer_pose
            else:
                return self.model.restore_model(
                    model_depth_restore_path=restore_paths[0],
                    model_pose_restore_path=restore_paths[1] if len(restore_paths) > 1 else None,
                    optimizer_depth=optimizer_depth,
                    optimizer_pose=optimizer_pose)
        elif 'voiced' in self.model_name:
            return self.model.restore_model(
                model_depth_restore_path=restore_paths[0],
                model_pose_restore_path=restore_paths[1] if len(restore_paths) > 1 else None,
                optimizer_depth=optimizer_depth,
                optimizer_pose=optimizer_pose)
        else:
            raise ValueError('Unsupported depth completion model: {}'.format(self.model_name))


    def save_model(self,
                   checkpoint_dirpath,
                   step,
                   optimizer_depth=None,
                   optimizer_pose=None):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_dirpath : str
                path to save directory to save checkpoints
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''

        os.makedirs(checkpoint_dirpath, exist_ok=True)

        if 'kbnet' in self.model_name:
            self.model.save_model(
                os.path.join(checkpoint_dirpath, 'kbnet-{}.pth'.format(step)),
                step,
                optimizer_depth,
                model_pose_checkpoint_path=os.path.join(checkpoint_dirpath, 'posenet-{}.pth'.format(step)),
                optimizer_pose=optimizer_pose)
        elif 'scaffnet' in self.model_name:
            self.model.save_model(
                os.path.join(checkpoint_dirpath, 'scaffnet-{}.pth'.format(step)),
                step=step,
                optimizer=optimizer_depth)
        elif 'fusionnet' in self.model_name:
            self.model.save_model(
                os.path.join(checkpoint_dirpath, 'fusionnet-{}.pth'.format(step)),
                step,
                optimizer_depth,
                model_pose_checkpoint_path=os.path.join(checkpoint_dirpath, 'posenet-{}.pth'.format(step)),
                optimizer_pose=optimizer_pose)
        elif 'voiced' in self.model_name:
            self.model.save_model(
                os.path.join(checkpoint_dirpath, 'voiced-{}.pth'.format(step)),
                step,
                optimizer_depth,
                model_pose_checkpoint_path=os.path.join(checkpoint_dirpath, 'posenet-{}.pth'.format(step)),
                optimizer_pose=optimizer_pose)
        else:
            raise ValueError('Unsupported depth completion model: {}'.format(self.model_name))

        # TokenCDC: Save ALL TokenCDC params
        torch.save({'image_key_weights': self.image_key_weights.state_dict(),
                    'depth_key_weights': self.depth_key_weights.state_dict(),
                    'image_token_pools': self.image_token_pools.state_dict(),
                    'depth_token_pools': self.depth_token_pools.state_dict()},
                    os.path.join(checkpoint_dirpath, 'tokens-{}.pth'.format(step)))


    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image0=None,
                    image1to0=None,
                    image2to0=None,
                    output_depth0=None,
                    sparse_depth0=None,
                    validity_map0=None,
                    ground_truth0=None,
                    pose0to1=None,
                    pose0to2=None,
                    scalars={},
                    n_image_per_summary=4):
        '''
        Logs summary to Tensorboard

        Arg(s):
            summary_writer : SummaryWriter
                Tensorboard summary writer
            tag : str
                tag that prefixes names to log
            step : int
                current step in training
            image0 : torch.Tensor[float32]
                image at time step t
            image1to0 : torch.Tensor[float32]
                image at time step t-1 warped to time step t
            image2to0 : torch.Tensor[float32]
                image at time step t+1 warped to time step t
            output_depth0 : torch.Tensor[float32]
                output depth at time t
            sparse_depth0 : torch.Tensor[float32]
                sparse_depth at time t
            validity_map0 : torch.Tensor[float32]
                validity map of sparse depth at time t
            ground_truth0 : torch.Tensor[float32]
                ground truth depth at time t
            pose0to1 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t to t-1
            pose0to2 : torch.Tensor[float32]
                4 x 4 relative pose from image at time t to t+1
            scalars : dict[str, float]
                dictionary of scalars to log
            n_image_per_summary : int
                number of images to display within a summary
        '''

        with torch.no_grad():

            display_summary_image = []
            display_summary_depth = []

            display_summary_image_text = tag
            display_summary_depth_text = tag

            if image0 is not None:
                image0_summary = image0[0:n_image_per_summary, ...]

                display_summary_image_text += '_image0'
                display_summary_depth_text += '_image0'

                # Normalize for display if necessary
                if torch.max(image0_summary) > 1:
                    image0_summary = image0_summary / 255.0

                # Add to list of images to log
                display_summary_image.append(
                    torch.cat([
                        image0_summary.cpu(),
                        torch.zeros_like(image0_summary, device=torch.device('cpu'))],
                        dim=-1))

                display_summary_depth.append(display_summary_image[-1])

            if image0 is not None and image1to0 is not None:
                image1to0_summary = image1to0[0:n_image_per_summary, ...]

                display_summary_image_text += '_image1to0-error'

                # Normalize for display if necessary
                if torch.max(image1to0_summary) > 1:
                    image1to0_summary = image1to0_summary / 255.0

                # Compute reconstruction error w.r.t. image 0
                image1to0_error_summary = torch.mean(
                    torch.abs(image0_summary - image1to0_summary),
                    dim=1,
                    keepdim=True)

                # Add to list of images to log
                image1to0_error_summary = log_utils.colorize(
                    (image1to0_error_summary / 0.10).cpu(),
                    colormap='inferno')

                display_summary_image.append(
                    torch.cat([
                        image1to0_summary.cpu(),
                        image1to0_error_summary],
                        dim=3))

            if image0 is not None and image2to0 is not None:
                image2to0_summary = image2to0[0:n_image_per_summary, ...]

                display_summary_image_text += '_image2to0-error'

                # Normalize for display if necessary
                if torch.max(image2to0_summary) > 1:
                    image2to0_summary = image2to0_summary / 255.0

                # Compute reconstruction error w.r.t. image 0
                image2to0_error_summary = torch.mean(
                    torch.abs(image0_summary - image2to0_summary),
                    dim=1,
                    keepdim=True)

                # Add to list of images to log
                image2to0_error_summary = log_utils.colorize(
                    (image2to0_error_summary / 0.10).cpu(),
                    colormap='inferno')

                display_summary_image.append(
                    torch.cat([
                        image2to0_summary.cpu(),
                        image2to0_error_summary],
                        dim=3))

            if output_depth0 is not None:
                output_depth0_summary = output_depth0[0:n_image_per_summary, ...]

                display_summary_depth_text += '_output0'

                # Add to list of images to log
                n_batch, _, n_height, n_width = output_depth0_summary.shape

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            (output_depth0_summary / self.max_predict_depth).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth0_distro', output_depth0, global_step=step)

            if output_depth0 is not None and sparse_depth0 is not None and validity_map0 is not None:
                sparse_depth0_summary = sparse_depth0[0:n_image_per_summary, ...]
                validity_map0_summary = validity_map0[0:n_image_per_summary, ...]

                display_summary_depth_text += '_sparse0-error'

                # Compute output error w.r.t. input sparse depth
                sparse_depth0_error_summary = \
                    torch.abs(output_depth0_summary - sparse_depth0_summary)

                sparse_depth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    (sparse_depth0_error_summary + 1e-8) / (sparse_depth0_summary + 1e-8),
                    validity_map0_summary)

                # Add to list of images to log
                sparse_depth0_summary = torch.clamp(sparse_depth0_summary, 0.0, self.max_predict_depth)

                sparse_depth0_summary = log_utils.colorize(
                    (sparse_depth0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                sparse_depth0_error_summary = log_utils.colorize(
                    (sparse_depth0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        sparse_depth0_summary,
                        sparse_depth0_error_summary],
                        dim=3))

                # Log distribution of sparse depth
                summary_writer.add_histogram(tag + '_sparse_depth0_distro', sparse_depth0, global_step=step)

            if output_depth0 is not None and ground_truth0 is not None:

                ground_truth0_summary = ground_truth0[0:n_image_per_summary, ...]
                validity_map0_summary = torch.where(
                    ground_truth0_summary > 0,
                    torch.ones_like(ground_truth0_summary),
                    ground_truth0_summary)

                display_summary_depth_text += '_groundtruth0-error'

                ground_truth0_summary = torch.clamp(ground_truth0_summary, 0.0, self.max_predict_depth)

                # Compute output error w.r.t. ground truth
                ground_truth0_error_summary = \
                    torch.abs(output_depth0_summary - ground_truth0_summary)

                ground_truth0_error_summary = torch.where(
                    validity_map0_summary == 1.0,
                    (ground_truth0_error_summary + 1e-8) / (ground_truth0_summary + 1e-8),
                    validity_map0_summary)

                # Add to list of images to log
                ground_truth0_summary = log_utils.colorize(
                    (ground_truth0_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                ground_truth0_error_summary = log_utils.colorize(
                    (ground_truth0_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        ground_truth0_summary,
                        ground_truth0_error_summary],
                        dim=3))

                # Log distribution of ground truth
                summary_writer.add_histogram(tag + '_ground_truth0_distro', ground_truth0, global_step=step)

            if pose0to1 is not None:
                # Log distribution of pose 1 to 0translation vector
                summary_writer.add_histogram(tag + '_tx0to1_distro', pose0to1[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty0to1_distro', pose0to1[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz0to1_distro', pose0to1[:, 2, 3], global_step=step)

            if pose0to2 is not None:
                # Log distribution of pose 2 to 0 translation vector
                summary_writer.add_histogram(tag + '_tx0to2_distro', pose0to2[:, 0, 3], global_step=step)
                summary_writer.add_histogram(tag + '_ty0to2_distro', pose0to2[:, 1, 3], global_step=step)
                summary_writer.add_histogram(tag + '_tz0to2_distro', pose0to2[:, 2, 3], global_step=step)

        # Log scalars to tensorboard
        for (name, value) in scalars.items():
            summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

        # Log image summaries to tensorboard
        if len(display_summary_image) > 1:
            display_summary_image = torch.cat(display_summary_image, dim=2)

            summary_writer.add_image(
                display_summary_image_text,
                torchvision.utils.make_grid(display_summary_image, nrow=n_image_per_summary),
                global_step=step)

        if len(display_summary_depth) > 1:
            display_summary_depth = torch.cat(display_summary_depth, dim=2)

            summary_writer.add_image(
                display_summary_depth_text,
                torchvision.utils.make_grid(display_summary_depth, nrow=n_image_per_summary),
                global_step=step)
