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
            # for param in self.model.parameters_pose():
            #     param.requires_grad = False

        # TokenCDC: initialize ParameterDicts to store learnable key & token POOLS for each dataset
        # self.i1_key_pools = torch.nn.ParameterDict() 
        # self.i1_token_pools = torch.nn.ParameterDict()
        # self.d1_key_pools = torch.nn.ParameterDict() 
        # self.d1_token_pools = torch.nn.ParameterDict() 

        self.i2_key_pools = torch.nn.ParameterDict() 
        self.i2_token_pools = torch.nn.ParameterDict()
        self.d2_key_pools = torch.nn.ParameterDict() 
        self.d2_token_pools = torch.nn.ParameterDict() 

        self.i3_key_pools = torch.nn.ParameterDict() 
        self.i3_token_pools = torch.nn.ParameterDict()
        self.d3_key_pools = torch.nn.ParameterDict() 
        self.d3_token_pools = torch.nn.ParameterDict() 

        self.i4_key_pools = torch.nn.ParameterDict() 
        self.i4_token_pools = torch.nn.ParameterDict()
        self.d4_key_pools = torch.nn.ParameterDict() 
        self.d4_token_pools = torch.nn.ParameterDict() 

        self.latent_key_pools = torch.nn.ParameterDict()
        self.latent_token_pools = torch.nn.ParameterDict()

        self.new_params = []  # LIST OF PARAMS to be added to the depth optimizer!
 

    def add_new_key_token_pool(self, dataset_uid, #i1_key_dim, d1_key_dim, i1_token_dim, d1_token_dim,
                                                i2_key_dim, d2_key_dim, i2_token_dim, d2_token_dim,
                                                i3_key_dim, d3_key_dim, i3_token_dim, d3_token_dim,
                                                i4_key_dim, d4_key_dim, i4_token_dim, d4_token_dim,
                                                latent_key_dim, latent_token_dim):
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
        # CREATE key and token pool
        # new_i1_key_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, i1_key_dim), device=self.device), requires_grad=True)
        # new_i1_token_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, i1_token_dim), device=self.device), requires_grad=True)
        # new_d1_key_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, d1_key_dim), device=self.device), requires_grad=True)
        # new_d1_token_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, d1_token_dim), device=self.device), requires_grad=True)

        new_i2_key_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, i2_key_dim), device=self.device), requires_grad=True)
        new_i2_token_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, i2_token_dim), device=self.device), requires_grad=True)
        new_d2_key_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, d2_key_dim), device=self.device), requires_grad=True)
        new_d2_token_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, d2_token_dim), device=self.device), requires_grad=True)

        new_i3_key_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, i3_key_dim), device=self.device), requires_grad=True)
        new_i3_token_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, i3_token_dim), device=self.device), requires_grad=True)
        new_d3_key_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, d3_key_dim), device=self.device), requires_grad=True)
        new_d3_token_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, d3_token_dim), device=self.device), requires_grad=True)

        new_i4_key_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, i4_key_dim), device=self.device), requires_grad=True)
        new_i4_token_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, i4_token_dim), device=self.device), requires_grad=True)
        new_d4_key_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, d4_key_dim), device=self.device), requires_grad=True)
        new_d4_token_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, d4_token_dim), device=self.device), requires_grad=True)

        new_latent_key_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, latent_key_dim), device=self.device), requires_grad=True)
        new_latent_token_pool = torch.nn.Parameter(torch.zeros((self.key_token_pool_size, latent_token_dim), device=self.device), requires_grad=True)

        # ADD to the key and token pool dicts
        # self.i1_key_pools[dataset_uid] = new_i1_key_pool
        # self.i1_token_pools[dataset_uid] = new_i1_token_pool
        # self.d1_key_pools[dataset_uid] = new_d1_key_pool
        # self.d1_token_pools[dataset_uid] = new_d1_token_pool
        
        self.i2_key_pools[dataset_uid] = new_i2_key_pool
        self.i2_token_pools[dataset_uid] = new_i2_token_pool
        self.d2_key_pools[dataset_uid] = new_d2_key_pool
        self.d2_token_pools[dataset_uid] = new_d2_token_pool
        
        self.i3_key_pools[dataset_uid] = new_i3_key_pool
        self.i3_token_pools[dataset_uid] = new_i3_token_pool
        self.d3_key_pools[dataset_uid] = new_d3_key_pool
        self.d3_token_pools[dataset_uid] = new_d3_token_pool
        
        self.i4_key_pools[dataset_uid] = new_i4_key_pool
        self.i4_token_pools[dataset_uid] = new_i4_token_pool
        self.d4_key_pools[dataset_uid] = new_d4_key_pool
        self.d4_token_pools[dataset_uid] = new_d4_token_pool
        
        self.latent_key_pools[dataset_uid] = new_latent_key_pool
        self.latent_token_pools[dataset_uid] = new_latent_token_pool

        # Add params to be added the optimizer (in the train loop in depth_completion.py)
        # self.new_params.append(new_i1_key_pool)
        # self.new_params.append(new_d1_key_pool)
        # self.new_params.append(new_i1_token_pool)
        # self.new_params.append(new_d1_token_pool)
        
        self.new_params.append(new_i2_key_pool)
        self.new_params.append(new_d2_key_pool)
        self.new_params.append(new_i2_token_pool)
        self.new_params.append(new_d2_token_pool)
        
        self.new_params.append(new_i3_key_pool)
        self.new_params.append(new_d3_key_pool)
        self.new_params.append(new_i3_token_pool)
        self.new_params.append(new_d3_token_pool)
        
        self.new_params.append(new_i4_key_pool)
        self.new_params.append(new_d4_key_pool)
        self.new_params.append(new_i4_token_pool)
        self.new_params.append(new_d4_token_pool)
        
        self.new_params.append(new_latent_key_pool)
        self.new_params.append(new_latent_token_pool)

        assert set(self.i4_key_pools.keys()) == set(self.d4_key_pools.keys()), "Image/Depth Key pools must have same keys!"
        assert set(self.i4_key_pools.keys()) == set(self.i4_token_pools.keys()), "Key and Token pools must have same keys!"
        #return new_i1_key_pool, new_d1_key_pool, new_i1_token_pool, new_d1_token_pool, \
        return new_i2_key_pool, new_d2_key_pool, new_i2_token_pool, new_d2_token_pool, \
                new_i3_key_pool, new_d3_key_pool, new_i3_token_pool, new_d3_token_pool, \
                new_i4_key_pool, new_d4_key_pool, new_i4_token_pool, new_d4_token_pool, \
                new_latent_key_pool, new_latent_token_pool


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
        latent, skips, shape = self.model.forward_depth_encoder(
            image, 
            sparse_depth, 
            validity_map, 
            intrinsics)

        ##### BEGIN TokenCDC Implementation 

        # TokenCDC TEST: number of learnable params
        # token_count = 0
        # depth_count = sum(p.numel() for p in self.model.parameters_depth() if p.requires_grad)
        # pose_count = sum(p.numel() for p in self.model.parameters_pose() if p.requires_grad)
        # print("Learnable parameters in the model: Tokens {}, Depth {}, Pose {}".format(token_count, depth_count, pose_count))
        # print("Current token keys: {}".format(self.token_pools.keys()))

        # FIRST, get key and token pools
        i1_C, d1_C, i2_C, d2_C, i3_C, d3_C, i4_C, d4_C, latent_C = skips[0][0].shape[1], skips[0][1].shape[1], \
                                                                    skips[1][0].shape[1], skips[1][1].shape[1], \
                                                                    skips[2][0].shape[1], skips[2][1].shape[1], \
                                                                    skips[3][0].shape[1], skips[3][1].shape[1], \
                                                                    latent.shape[1]
        if dataset_uid not in self.latent_key_pools:
            #curr_i1_key_pool, curr_d1_key_pool, curr_i1_token_pool, curr_d1_token_pool, \
            curr_i2_key_pool, curr_d2_key_pool, curr_i2_token_pool, curr_d2_token_pool, \
            curr_i3_key_pool, curr_d3_key_pool, curr_i3_token_pool, curr_d3_token_pool, \
            curr_i4_key_pool, curr_d4_key_pool, curr_i4_token_pool, curr_d4_token_pool, \
            curr_latent_key_pool, curr_latent_token_pool = \
                self.add_new_key_token_pool(dataset_uid, #i1_C, d1_C, i1_C, d1_C,
                                                        i2_C, d2_C, i2_C, d2_C,
                                                        i3_C, d3_C, i3_C, d3_C,
                                                        i4_C, d4_C, i4_C, d4_C,
                                                        latent_C, latent_C)
        else:
            #curr_i1_key_pool, curr_d1_key_pool, curr_i1_token_pool, curr_d1_token_pool, \
            curr_i2_key_pool, curr_d2_key_pool, curr_i2_token_pool, curr_d2_token_pool, \
            curr_i3_key_pool, curr_d3_key_pool, curr_i3_token_pool, curr_d3_token_pool, \
            curr_i4_key_pool, curr_d4_key_pool, curr_i4_token_pool, curr_d4_token_pool, \
            curr_latent_key_pool, curr_latent_token_pool = \
                self.i2_key_pools[dataset_uid], self.d2_key_pools[dataset_uid], \
                self.i2_token_pools[dataset_uid], self.d2_token_pools[dataset_uid], \
                self.i3_key_pools[dataset_uid], self.d3_key_pools[dataset_uid], \
                self.i3_token_pools[dataset_uid], self.d3_token_pools[dataset_uid], \
                self.i4_key_pools[dataset_uid], self.d4_key_pools[dataset_uid], \
                self.i4_token_pools[dataset_uid], self.d4_token_pools[dataset_uid], \
                self.latent_key_pools[dataset_uid], self.latent_token_pools[dataset_uid]
                #self.i1_key_pools[dataset_uid], self.d1_key_pools[dataset_uid], \
                #self.i1_token_pools[dataset_uid], self.d1_token_pools[dataset_uid], \
                

        ### Skip Conection 1
        # QUERIES
        # i1_skip, d1_skip = skips[0][0], skips[0][1]
        # i1_N, i1_C, i1_H, i1_W = i1_skip.shape
        # d1_N, d1_C, d1_H, d1_W = d1_skip.shape
        # assert i1_N == d1_N and i1_H == d1_H and i1_W == d1_W, "Image/depth must have the same non-channel dims!"
        # i1_queries = i1_skip.permute(0, 2, 3, 1).view(i1_N, i1_H*i1_W, i1_C)
        # d1_queries = d1_skip.permute(0, 2, 3, 1).view(d1_N, d1_H*d1_W, d1_C)
        # # Compute TOKENS using attention
        # i1_scores = torch.matmul(i1_queries, curr_i1_key_pool.transpose(-2, -1)) / torch.sqrt(torch.tensor(i1_C, device=self.device, dtype=torch.float32))
        # i1_scores = F.softmax(i1_scores, dim=-1)
        # d1_scores = torch.matmul(d1_queries, curr_d1_key_pool.transpose(-2, -1)) / torch.sqrt(torch.tensor(d1_C, device=self.device, dtype=torch.float32))
        # d1_scores = F.softmax(d1_scores, dim=-1)
        # i1_tokens = torch.matmul(i1_scores, curr_i1_token_pool).view(i1_N, i1_H, i1_W, i1_C).permute(0, 3, 1, 2)
        # d1_tokens = torch.matmul(d1_scores, curr_d1_token_pool).view(d1_N, d1_H, d1_W, d1_C).permute(0, 3, 1, 2)
        # # CONCAT tokens to skips
        # i1_skip_with_tokens = i1_skip + i1_tokens
        # d1_skip_with_tokens = d1_skip + d1_tokens
        i1_skip_with_tokens, d1_skip_with_tokens = skips[0][0], skips[0][1]

        ### Skip Conection 2
        # QUERIES
        # i2_skip, d2_skip = skips[1][0], skips[1][1]
        # i2_N, i2_C, i2_H, i2_W = i2_skip.shape
        # d2_N, d2_C, d2_H, d2_W = d2_skip.shape
        # assert i2_N == d2_N and i2_H == d2_H and i2_W == d2_W, "Image/depth must have the same non-channel dims!"
        # i2_queries = i2_skip.permute(0, 2, 3, 1).view(i2_N, i2_H*i2_W, i2_C)
        # d2_queries = d2_skip.permute(0, 2, 3, 1).view(d2_N, d2_H*d2_W, d2_C)
        # # Compute TOKENS using attention
        # i2_scores = torch.matmul(i2_queries, curr_i2_key_pool.transpose(-2, -1)) / torch.sqrt(torch.tensor(i2_C, device=self.device, dtype=torch.float32))
        # i2_scores = F.softmax(i2_scores, dim=-1)
        # d2_scores = torch.matmul(d2_queries, curr_d2_key_pool.transpose(-2, -1)) / torch.sqrt(torch.tensor(d2_C, device=self.device, dtype=torch.float32))
        # d2_scores = F.softmax(d2_scores, dim=-1)
        # i2_tokens = torch.matmul(i2_scores, curr_i2_token_pool).view(i2_N, i2_H, i2_W, i2_C).permute(0, 3, 1, 2)
        # d2_tokens = torch.matmul(d2_scores, curr_d2_token_pool).view(d2_N, d2_H, d2_W, d2_C).permute(0, 3, 1, 2)
        # # CONCAT tokens to skips
        # i2_skip_with_tokens = i2_skip + i2_tokens
        # d2_skip_with_tokens = d2_skip + d2_tokens
        i2_skip_with_tokens, d2_skip_with_tokens = skips[1][0], skips[1][1]

        ### Skip Conection 3
        # QUERIES
        # i3_skip, d3_skip = skips[2][0], skips[2][1]
        # i3_N, i3_C, i3_H, i3_W = i3_skip.shape
        # d3_N, d3_C, d3_H, d3_W = d3_skip.shape
        # assert i3_N == d3_N and i3_H == d3_H and i3_W == d3_W, "Image/depth must have the same non-channel dims!"
        # i3_queries = i3_skip.permute(0, 2, 3, 1).view(i3_N, i3_H*i3_W, i3_C)
        # d3_queries = d3_skip.permute(0, 2, 3, 1).view(d3_N, d3_H*d3_W, d3_C)
        # # Compute TOKENS using attention
        # i3_scores = torch.matmul(i3_queries, curr_i3_key_pool.transpose(-2, -1)) / torch.sqrt(torch.tensor(i3_C, device=self.device, dtype=torch.float32))
        # i3_scores = F.softmax(i3_scores, dim=-1)
        # d3_scores = torch.matmul(d3_queries, curr_d3_key_pool.transpose(-2, -1)) / torch.sqrt(torch.tensor(d3_C, device=self.device, dtype=torch.float32))
        # d3_scores = F.softmax(d3_scores, dim=-1)
        # i3_tokens = torch.matmul(i3_scores, curr_i3_token_pool).view(i3_N, i3_H, i3_W, i3_C).permute(0, 3, 1, 2)
        # d3_tokens = torch.matmul(d3_scores, curr_d3_token_pool).view(d3_N, d3_H, d3_W, d3_C).permute(0, 3, 1, 2)
        # # CONCAT tokens to skips
        # i3_skip_with_tokens = i3_skip + i3_tokens
        # d3_skip_with_tokens = d3_skip + d3_tokens
        i3_skip_with_tokens, d3_skip_with_tokens = skips[2][0], skips[2][1]
        
        ### Skip Conection 4
        # QUERIES
        # i4_skip, d4_skip = skips[3][0], skips[3][1]
        # i4_N, i4_C, i4_H, i4_W = i4_skip.shape
        # d4_N, d4_C, d4_H, d4_W = d4_skip.shape
        # assert i4_N == d4_N and i4_H == d4_H and i4_W == d4_W, "Image/depth must have the same non-channel dims!"
        # i4_queries = i4_skip.permute(0, 2, 3, 1).view(i4_N, i4_H*i4_W, i4_C)
        # d4_queries = d4_skip.permute(0, 2, 3, 1).view(d4_N, d4_H*d4_W, d4_C)
        # # Compute TOKENS using attention
        # i4_scores = torch.matmul(i4_queries, curr_i4_key_pool.transpose(-2, -1)) / torch.sqrt(torch.tensor(i4_C, device=self.device, dtype=torch.float32))
        # i4_scores = F.softmax(i4_scores, dim=-1)
        # d4_scores = torch.matmul(d4_queries, curr_d4_key_pool.transpose(-2, -1)) / torch.sqrt(torch.tensor(d4_C, device=self.device, dtype=torch.float32))
        # d4_scores = F.softmax(d4_scores, dim=-1)
        # i4_tokens = torch.matmul(i4_scores, curr_i4_token_pool).view(i4_N, i4_H, i4_W, i4_C).permute(0, 3, 1, 2)
        # d4_tokens = torch.matmul(d4_scores, curr_d4_token_pool).view(d4_N, d4_H, d4_W, d4_C).permute(0, 3, 1, 2)
        # # CONCAT tokens to skips
        # i4_skip_with_tokens = i4_skip + i4_tokens
        # d4_skip_with_tokens = d4_skip + d4_tokens
        i4_skip_with_tokens, d4_skip_with_tokens = skips[3][0], skips[3][1]

        ### Latent Space
        # QUERIES
        # latent_queries = latent.permute(0, 2, 3, 1).view(latent.shape[0], latent.shape[2]*latent.shape[3], latent.shape[1])
        # # Compute TOKENS using attention
        # latent_scores = torch.matmul(latent_queries, curr_latent_key_pool.transpose(-2, -1)) / torch.sqrt(torch.tensor(latent_C, device=self.device, dtype=torch.float32))
        # latent_scores = F.softmax(latent_scores, dim=-1)
        # latent_tokens = torch.matmul(latent_scores, curr_latent_token_pool).view(latent.shape[0], latent.shape[2], latent.shape[3], latent.shape[1]).permute(0, 3, 1, 2)
        # # CONCAT tokens to latent
        # latent_with_tokens = latent + latent_tokens
        latent_with_tokens = latent

        ### Combine all skip connections
        skips_with_tokens = [torch.cat([i1_skip_with_tokens, d1_skip_with_tokens], dim=1), 
                                torch.cat([i2_skip_with_tokens, d2_skip_with_tokens], dim=1),
                                torch.cat([i3_skip_with_tokens, d3_skip_with_tokens], dim=1),
                                torch.cat([i4_skip_with_tokens, d4_skip_with_tokens], dim=1)]

        ##### END TokenCDC Implementation

        # Decoder Forward Pass
        output = self.model.forward_depth_decoder(
                    latent_with_tokens,
                    skips_with_tokens,
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
            # i1_key_pools_state_dict = checkpoint['i1_key_pools']
            # d1_key_pools_state_dict = checkpoint['d1_key_pools']
            # i1_token_pools_state_dict = checkpoint['i1_token_pools']
            # d1_token_pools_state_dict = checkpoint['d1_token_pools']
            i2_key_pools_state_dict = checkpoint['i2_key_pools']
            d2_key_pools_state_dict = checkpoint['d2_key_pools']
            i2_token_pools_state_dict = checkpoint['i2_token_pools']
            d2_token_pools_state_dict = checkpoint['d2_token_pools']
            i3_key_pools_state_dict = checkpoint['i3_key_pools']
            d3_key_pools_state_dict = checkpoint['d3_key_pools']
            i3_token_pools_state_dict = checkpoint['i3_token_pools']
            d3_token_pools_state_dict = checkpoint['d3_token_pools']
            i4_key_pools_state_dict = checkpoint['i4_key_pools']
            d4_key_pools_state_dict = checkpoint['d4_key_pools']
            i4_token_pools_state_dict = checkpoint['i4_token_pools']
            d4_token_pools_state_dict = checkpoint['d4_token_pools']
            latent_key_pools_state_dict = checkpoint['latent_key_pools']
            latent_token_pools_state_dict = checkpoint['latent_token_pools']

            # Identify missing keys (keys in state_dict but not in model)
            # i1_key_pools_missing_keys = set(i1_key_pools_state_dict.keys()) - self.i1_key_pools.keys()
            # d1_key_pools_missing_keys = set(d1_key_pools_state_dict.keys()) - self.d1_key_pools.keys()
            # i1_token_pools_missing_keys = set(i1_token_pools_state_dict.keys()) - self.i1_token_pools.keys()
            # d1_token_pools_missing_keys = set(d1_token_pools_state_dict.keys()) - self.d1_token_pools.keys()
            i2_key_pools_missing_keys = set(i2_key_pools_state_dict.keys()) - self.i2_key_pools.keys()
            d2_key_pools_missing_keys = set(d2_key_pools_state_dict.keys()) - self.d2_key_pools.keys()
            i2_token_pools_missing_keys = set(i2_token_pools_state_dict.keys()) - self.i2_token_pools.keys()
            d2_token_pools_missing_keys = set(d2_token_pools_state_dict.keys()) - self.d2_token_pools.keys()
            i3_key_pools_missing_keys = set(i3_key_pools_state_dict.keys()) - self.i3_key_pools.keys()
            d3_key_pools_missing_keys = set(d3_key_pools_state_dict.keys()) - self.d3_key_pools.keys()
            i3_token_pools_missing_keys = set(i3_token_pools_state_dict.keys()) - self.i3_token_pools.keys()
            d3_token_pools_missing_keys = set(d3_token_pools_state_dict.keys()) - self.d3_token_pools.keys()
            i4_key_pools_missing_keys = set(i4_key_pools_state_dict.keys()) - self.i4_key_pools.keys()
            d4_key_pools_missing_keys = set(d4_key_pools_state_dict.keys()) - self.d4_key_pools.keys()
            i4_token_pools_missing_keys = set(i4_token_pools_state_dict.keys()) - self.i4_token_pools.keys()
            d4_token_pools_missing_keys = set(d4_token_pools_state_dict.keys()) - self.d4_token_pools.keys()
            latent_key_pools_missing_keys = set(latent_key_pools_state_dict.keys()) - self.latent_key_pools.keys()
            latent_token_pools_missing_keys = set(latent_token_pools_state_dict.keys()) - self.latent_token_pools.keys()
            assert i4_key_pools_missing_keys == d4_key_pools_missing_keys, "Image/Depth Key pools must have the same keys!"
            assert i4_key_pools_missing_keys == d4_token_pools_missing_keys, "Key and Token pools must have the same keys!"

            # Add pools for the missing keys (and add to new_params list to be added to optimizer)
            for mk in i4_key_pools_missing_keys:
                self.add_new_key_token_pool(mk,
                                            # i1_key_pools_state_dict[mk].shape[1],
                                            # d1_key_pools_state_dict[mk].shape[1],
                                            # i1_token_pools_state_dict[mk].shape[1],
                                            # d1_token_pools_state_dict[mk].shape[1],
                                            i2_key_pools_state_dict[mk].shape[1],
                                            d2_key_pools_state_dict[mk].shape[1],
                                            i2_token_pools_state_dict[mk].shape[1],
                                            d2_token_pools_state_dict[mk].shape[1],
                                            i3_key_pools_state_dict[mk].shape[1],
                                            d3_key_pools_state_dict[mk].shape[1],
                                            i3_token_pools_state_dict[mk].shape[1],
                                            d3_token_pools_state_dict[mk].shape[1],
                                            i4_key_pools_state_dict[mk].shape[1],
                                            d4_key_pools_state_dict[mk].shape[1],
                                            i4_token_pools_state_dict[mk].shape[1],
                                            d4_token_pools_state_dict[mk].shape[1],
                                            latent_key_pools_state_dict[mk].shape[1],
                                            latent_token_pools_state_dict[mk].shape[1])

            # Now, load the state dicts
            # self.i1_key_pools.load_state_dict(i1_key_pools_state_dict)
            # self.d1_key_pools.load_state_dict(d1_key_pools_state_dict)
            # self.i1_token_pools.load_state_dict(i1_token_pools_state_dict)
            # self.d1_token_pools.load_state_dict(d1_token_pools_state_dict)
            self.i2_key_pools.load_state_dict(i2_key_pools_state_dict)
            self.d2_key_pools.load_state_dict(d2_key_pools_state_dict)
            self.i2_token_pools.load_state_dict(i2_token_pools_state_dict)
            self.d2_token_pools.load_state_dict(d2_token_pools_state_dict)
            self.i3_key_pools.load_state_dict(i3_key_pools_state_dict)
            self.d3_key_pools.load_state_dict(d3_key_pools_state_dict)
            self.i3_token_pools.load_state_dict(i3_token_pools_state_dict)
            self.d3_token_pools.load_state_dict(d3_token_pools_state_dict)
            self.i4_key_pools.load_state_dict(i4_key_pools_state_dict)
            self.d4_key_pools.load_state_dict(d4_key_pools_state_dict)
            self.i4_token_pools.load_state_dict(i4_token_pools_state_dict)
            self.d4_token_pools.load_state_dict(d4_token_pools_state_dict)
            self.latent_key_pools.load_state_dict(latent_key_pools_state_dict)
            self.latent_token_pools.load_state_dict(latent_token_pools_state_dict)
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
        torch.save({#'i1_key_pools_state_dict': self.i1_key_pools.state_dict(),
                # 'd1_key_pools_state_dict': self.d1_key_pools.state_dict(),
                # 'i1_token_pools_state_dict': self.i1_token_pools.state_dict(),
                # 'd1_token_pools_state_dict': self.d1_token_pools.state_dict(),
                'i2_key_pools_state_dict': self.i2_key_pools.state_dict(),
                'd2_key_pools_state_dict': self.d2_key_pools.state_dict(),
                'i2_token_pools_state_dict': self.i2_token_pools.state_dict(),
                'd2_token_pools_state_dict': self.d2_token_pools.state_dict(),
                'i3_key_pools_state_dict': self.i3_key_pools.state_dict(),
                'd3_key_pools_state_dict': self.d3_key_pools.state_dict(),
                'i3_token_pools_state_dict': self.i3_token_pools.state_dict(),
                'd3_token_pools_state_dict': self.d3_token_pools.state_dict(),
                'i4_key_pools_state_dict': self.i4_key_pools.state_dict(),
                'd4_key_pools_state_dict': self.d4_key_pools.state_dict(),
                'i4_token_pools_state_dict': self.i4_token_pools.state_dict(),
                'd4_token_pools_state_dict': self.d4_token_pools.state_dict(),
                'latent_key_pools_state_dict': self.latent_key_pools.state_dict(),
                'latent_token_pools_state_dict': self.latent_token_pools.state_dict()},
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
