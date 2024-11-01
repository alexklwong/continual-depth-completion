import os, torch, torchvision
import torch.nn.functional as F
from utils.src import log_utils, net_utils
import net_utils


class ContinualLearningModel(torch.nn.Module):
    '''
    Wrapper class for continual learning parameters

    Arg(s):
        key_token_pool_size : int
            number of keys/tokens in the pool
    '''

    def __init__(self,
                 image_pool_size,
                 depth_pool_size,
                 device):
        super(ContinualLearningModel, self).__init__()
        
        self.image_pool_size = image_pool_size
        self.depth_pool_size = depth_pool_size
        self.device = device
        self.dataset_uids = []  # Master list of seen datasets
        self.new_params = []  # LIST OF PARAMS to be added to the depth optimizer!

        self.i1_key_pools = torch.nn.ParameterDict() 
        self.i1_token_pools = torch.nn.ParameterDict()
        self.i1_linear = torch.nn.ParameterDict()
        self.d1_key_pools = torch.nn.ParameterDict() 
        self.d1_token_pools = torch.nn.ParameterDict() 
        self.d1_linear = torch.nn.ParameterDict()

        self.i2_key_pools = torch.nn.ParameterDict() 
        self.i2_token_pools = torch.nn.ParameterDict()
        self.i2_linear = torch.nn.ParameterDict()
        self.d2_key_pools = torch.nn.ParameterDict() 
        self.d2_token_pools = torch.nn.ParameterDict() 
        self.d2_linear = torch.nn.ParameterDict()

        self.i3_key_pools = torch.nn.ParameterDict() 
        self.i3_token_pools = torch.nn.ParameterDict()
        self.i3_linear = torch.nn.ParameterDict()
        self.d3_key_pools = torch.nn.ParameterDict() 
        self.d3_token_pools = torch.nn.ParameterDict() 
        self.d3_linear = torch.nn.ParameterDict()

        self.i4_key_pools = torch.nn.ParameterDict() 
        self.i4_token_pools = torch.nn.ParameterDict()
        self.i4_linear = torch.nn.ParameterDict()
        self.d4_key_pools = torch.nn.ParameterDict() 
        self.d4_token_pools = torch.nn.ParameterDict() 
        self.d4_linear = torch.nn.ParameterDict()

        self.latent_key_pools = torch.nn.ParameterDict()
        self.latent_token_pools = torch.nn.ParameterDict()
        self.latent_linear = torch.nn.ParameterDict()

        # Move to device
        self.to(self.device)
        self.eval()
 
    
    def get_key_token_pool(self, dataset_uid, dims):
        """
        Get key and token pools for a dataset given its uid
        
        Arg(s):
            dataset_uid : str
                unique id of dataset
            dims: tuple
                tuple of dimensions for the key and token pools
        Returns:
            key and token pools for the dataset
        """ 
        if dataset_uid not in self.dataset_uids:
            return self.add_new_key_token_pool(dataset_uid, dims)
        else:
            return (self.i1_key_pools[dataset_uid], self.i1_key_pools[dataset_uid+'_up']), self.i1_token_pools[dataset_uid], self.i1_linear[dataset_uid], \
                    (self.d1_key_pools[dataset_uid], self.d1_key_pools[dataset_uid+'_up']), self.d1_token_pools[dataset_uid], self.d1_linear[dataset_uid], \
                    (self.i2_key_pools[dataset_uid], self.i2_key_pools[dataset_uid+'_up']), self.i2_token_pools[dataset_uid], self.i2_linear[dataset_uid], \
                    (self.d2_key_pools[dataset_uid], self.d2_key_pools[dataset_uid+'_up']), self.d2_token_pools[dataset_uid], self.d2_linear[dataset_uid], \
                    (self.i3_key_pools[dataset_uid], self.i3_key_pools[dataset_uid+'_up']), self.i3_token_pools[dataset_uid], self.i3_linear[dataset_uid], \
                    (self.d3_key_pools[dataset_uid], self.d3_key_pools[dataset_uid+'_up']), self.d3_token_pools[dataset_uid], self.d3_linear[dataset_uid], \
                    (self.i4_key_pools[dataset_uid], self.i4_key_pools[dataset_uid+'_up']), self.i4_token_pools[dataset_uid], self.i4_linear[dataset_uid], \
                    (self.d4_key_pools[dataset_uid], self.d4_key_pools[dataset_uid+'_up']), self.d4_token_pools[dataset_uid], self.d4_linear[dataset_uid], \
                    (self.latent_key_pools[dataset_uid], self.latent_key_pools[dataset_uid+'_up']), self.latent_token_pools[dataset_uid], self.latent_linear[dataset_uid]


    def add_new_key_token_pool(self, dataset_uid, dims):
        '''
        Add and return token for a new unseen dataset

        Arg(s):
            dataset_uid : str
                unique id of dataset
            dims: tuple
                tuple of dimensions for the key and token
            manual: bool
                if manually added, don't re-add to optimizer
        Returns:
            torch.Tensor[float32] : added token
        '''
        print("Added {}\n\n\n\n".format(dataset_uid))
        # Unpack dimensions
        i1_dim, d1_dim, i2_dim, d2_dim, i3_dim, d3_dim, i4_dim, d4_dim, latent_dim = dims
        DOWN = 4

        # CREATE key and token pools
        new_i1_key_pool = (torch.nn.Parameter(torch.empty((i1_dim, i1_dim//DOWN), device=self.device), requires_grad=True), torch.nn.Parameter(torch.empty((i1_dim//DOWN, i1_dim), device=self.device), requires_grad=True))
        new_i1_token_pool = torch.nn.Parameter(torch.empty((self.image_pool_size, i1_dim), device=self.device), requires_grad=True)
        new_i1_linear = torch.nn.Parameter(torch.empty((i1_dim,1,1), device=self.device), requires_grad=True)
        new_d1_key_pool = (torch.nn.Parameter(torch.empty((d1_dim, d1_dim//DOWN), device=self.device), requires_grad=True), torch.nn.Parameter(torch.empty((d1_dim//DOWN, d1_dim), device=self.device), requires_grad=True))
        new_d1_token_pool = torch.nn.Parameter(torch.empty((self.depth_pool_size, d1_dim), device=self.device), requires_grad=True)
        new_d1_linear = torch.nn.Parameter(torch.empty((d1_dim,1,1), device=self.device), requires_grad=True)

        new_i2_key_pool = (torch.nn.Parameter(torch.empty((i2_dim, i2_dim//DOWN), device=self.device), requires_grad=True), torch.nn.Parameter(torch.empty((i2_dim//DOWN, i2_dim), device=self.device), requires_grad=True))
        new_i2_token_pool = torch.nn.Parameter(torch.empty((self.image_pool_size, i2_dim), device=self.device), requires_grad=True)
        new_i2_linear = torch.nn.Parameter(torch.empty((i2_dim,1,1), device=self.device), requires_grad=True)
        new_d2_key_pool = (torch.nn.Parameter(torch.empty((d2_dim, d2_dim//DOWN), device=self.device), requires_grad=True), torch.nn.Parameter(torch.empty((d2_dim//DOWN, d2_dim), device=self.device), requires_grad=True))
        new_d2_token_pool = torch.nn.Parameter(torch.empty((self.depth_pool_size, d2_dim), device=self.device), requires_grad=True)
        new_d2_linear = torch.nn.Parameter(torch.empty((d2_dim,1,1), device=self.device), requires_grad=True)

        new_i3_key_pool = torch.nn.Parameter(torch.empty((i3_dim, i3_dim//DOWN), device=self.device), requires_grad=True), torch.nn.Parameter(torch.empty((i3_dim//DOWN, i3_dim), device=self.device), requires_grad=True)
        new_i3_token_pool = torch.nn.Parameter(torch.empty((self.image_pool_size, i3_dim), device=self.device), requires_grad=True)
        new_i3_linear = torch.nn.Parameter(torch.empty((i3_dim,1,1), device=self.device), requires_grad=True)
        new_d3_key_pool = torch.nn.Parameter(torch.empty((d3_dim, d3_dim//DOWN), device=self.device), requires_grad=True), torch.nn.Parameter(torch.empty((d3_dim//DOWN, d3_dim), device=self.device), requires_grad=True)
        new_d3_token_pool = torch.nn.Parameter(torch.empty((self.depth_pool_size, d3_dim), device=self.device), requires_grad=True)
        new_d3_linear = torch.nn.Parameter(torch.empty((d3_dim,1,1), device=self.device), requires_grad=True)

        new_i4_key_pool = torch.nn.Parameter(torch.empty((i4_dim, i4_dim//DOWN), device=self.device), requires_grad=True), torch.nn.Parameter(torch.empty((i4_dim//DOWN, i4_dim), device=self.device), requires_grad=True)
        new_i4_token_pool = torch.nn.Parameter(torch.empty((self.image_pool_size, i4_dim), device=self.device), requires_grad=True)
        new_i4_linear = torch.nn.Parameter(torch.empty((i4_dim,1,1), device=self.device), requires_grad=True)
        new_d4_key_pool = torch.nn.Parameter(torch.empty((d4_dim, d4_dim//DOWN), device=self.device), requires_grad=True), torch.nn.Parameter(torch.empty((d4_dim//DOWN, d4_dim), device=self.device), requires_grad=True)
        new_d4_token_pool = torch.nn.Parameter(torch.empty((self.depth_pool_size, d4_dim), device=self.device), requires_grad=True)
        new_d4_linear = torch.nn.Parameter(torch.empty((d4_dim,1,1), device=self.device), requires_grad=True)

        new_latent_key_pool = torch.nn.Parameter(torch.empty((latent_dim, latent_dim//DOWN), device=self.device), requires_grad=True), torch.nn.Parameter(torch.empty((latent_dim//DOWN, latent_dim), device=self.device), requires_grad=True)
        new_latent_token_pool = torch.nn.Parameter(torch.empty((self.image_pool_size, latent_dim), device=self.device), requires_grad=True)
        new_latent_linear = torch.nn.Parameter(torch.empty((latent_dim,1,1), device=self.device), requires_grad=True)

        # Initialize parameters using kaiming_normal_
        torch.nn.init.kaiming_normal_(new_i1_key_pool[0])
        torch.nn.init.kaiming_normal_(new_i1_key_pool[1])
        torch.nn.init.kaiming_normal_(new_i1_token_pool)
        torch.nn.init.kaiming_normal_(new_i1_linear)
        torch.nn.init.kaiming_normal_(new_d1_key_pool[0])
        torch.nn.init.kaiming_normal_(new_d1_key_pool[1])
        torch.nn.init.kaiming_normal_(new_d1_token_pool)
        torch.nn.init.kaiming_normal_(new_d1_linear)

        torch.nn.init.kaiming_normal_(new_i2_key_pool[0])
        torch.nn.init.kaiming_normal_(new_i2_key_pool[1])
        torch.nn.init.kaiming_normal_(new_i2_token_pool)
        torch.nn.init.kaiming_normal_(new_i2_linear)
        torch.nn.init.kaiming_normal_(new_d2_key_pool[0])
        torch.nn.init.kaiming_normal_(new_d2_key_pool[1])
        torch.nn.init.kaiming_normal_(new_d2_token_pool)
        torch.nn.init.kaiming_normal_(new_d2_linear)

        torch.nn.init.kaiming_normal_(new_i3_key_pool[0])
        torch.nn.init.kaiming_normal_(new_i3_key_pool[1])
        torch.nn.init.kaiming_normal_(new_i3_token_pool)
        torch.nn.init.kaiming_normal_(new_i3_linear)
        torch.nn.init.kaiming_normal_(new_d3_key_pool[0])
        torch.nn.init.kaiming_normal_(new_d3_key_pool[1])
        torch.nn.init.kaiming_normal_(new_d3_token_pool)
        torch.nn.init.kaiming_normal_(new_d3_linear)

        torch.nn.init.kaiming_normal_(new_i4_key_pool[0])
        torch.nn.init.kaiming_normal_(new_i4_key_pool[1])
        torch.nn.init.kaiming_normal_(new_i4_token_pool)
        torch.nn.init.kaiming_normal_(new_i4_linear)
        torch.nn.init.kaiming_normal_(new_d4_key_pool[0])
        torch.nn.init.kaiming_normal_(new_d4_key_pool[1])
        torch.nn.init.kaiming_normal_(new_d4_token_pool)
        torch.nn.init.kaiming_normal_(new_d4_linear)

        torch.nn.init.kaiming_normal_(new_latent_key_pool[0])
        torch.nn.init.kaiming_normal_(new_latent_key_pool[1])
        torch.nn.init.kaiming_normal_(new_latent_token_pool)
        torch.nn.init.kaiming_normal_(new_latent_linear)

        # ADD to the key and token pool dicts
        self.dataset_uids.append(dataset_uid)
        
        self.i1_key_pools[dataset_uid] = new_i1_key_pool[0]
        self.i1_key_pools[dataset_uid+'_up'] = new_i1_key_pool[1]
        self.i1_token_pools[dataset_uid] = new_i1_token_pool
        self.i1_linear[dataset_uid] = new_i1_linear
        self.d1_key_pools[dataset_uid] = new_d1_key_pool[0]
        self.d1_key_pools[dataset_uid+'_up'] = new_d1_key_pool[1]
        self.d1_token_pools[dataset_uid] = new_d1_token_pool
        self.d1_linear[dataset_uid] = new_d1_linear

        self.i2_key_pools[dataset_uid] = new_i2_key_pool[0]
        self.i2_key_pools[dataset_uid+'_up'] = new_i2_key_pool[1]
        self.i2_token_pools[dataset_uid] = new_i2_token_pool
        self.i2_linear[dataset_uid] = new_i2_linear
        self.d2_key_pools[dataset_uid] = new_d2_key_pool[0]
        self.d2_key_pools[dataset_uid+'_up'] = new_d2_key_pool[1]
        self.d2_token_pools[dataset_uid] = new_d2_token_pool
        self.d2_linear[dataset_uid] = new_d2_linear
        
        self.i3_key_pools[dataset_uid] = new_i3_key_pool[0]
        self.i3_key_pools[dataset_uid+'_up'] = new_i3_key_pool[1]
        self.i3_token_pools[dataset_uid] = new_i3_token_pool
        self.i3_linear[dataset_uid] = new_i3_linear
        self.d3_key_pools[dataset_uid] = new_d3_key_pool[0]
        self.d3_key_pools[dataset_uid+'_up'] = new_d3_key_pool[1]
        self.d3_token_pools[dataset_uid] = new_d3_token_pool
        self.d3_linear[dataset_uid] = new_d3_linear
        
        self.i4_key_pools[dataset_uid] = new_i4_key_pool[0]
        self.i4_key_pools[dataset_uid+'_up'] = new_i4_key_pool[1]
        self.i4_token_pools[dataset_uid] = new_i4_token_pool
        self.i4_linear[dataset_uid] = new_i4_linear
        self.d4_key_pools[dataset_uid] = new_d4_key_pool[0]
        self.d4_key_pools[dataset_uid+'_up'] = new_d4_key_pool[1]
        self.d4_token_pools[dataset_uid] = new_d4_token_pool
        self.d4_linear[dataset_uid] = new_d4_linear
        
        self.latent_key_pools[dataset_uid] = new_latent_key_pool[0]
        self.latent_key_pools[dataset_uid+'_up'] = new_latent_key_pool[1]
        self.latent_token_pools[dataset_uid] = new_latent_token_pool
        self.latent_linear[dataset_uid] = new_latent_linear
        # assert set(self.i4_key_pools.keys()) == set(self.dataset_uids)
        # assert set(self.d4_token_pools.keys()) == set(self.dataset_uids)

        # Add params to be added the optimizer (in the train loop in depth_completion.py)
        self.new_params.append(new_i1_key_pool[0])
        self.new_params.append(new_i1_key_pool[1])
        self.new_params.append(new_i1_token_pool)
        self.new_params.append(new_i1_linear)
        self.new_params.append(new_d1_key_pool[0])
        self.new_params.append(new_d1_key_pool[1])
        self.new_params.append(new_d1_token_pool)
        self.new_params.append(new_d1_linear)

        self.new_params.append(new_i2_key_pool[0])
        self.new_params.append(new_i2_key_pool[1])
        self.new_params.append(new_i2_token_pool)
        self.new_params.append(new_i2_linear)
        self.new_params.append(new_d2_key_pool[0])
        self.new_params.append(new_d2_key_pool[1])
        self.new_params.append(new_d2_token_pool)
        self.new_params.append(new_d2_linear)
        
        self.new_params.append(new_i3_key_pool[0])
        self.new_params.append(new_i3_key_pool[1])
        self.new_params.append(new_i3_token_pool)
        self.new_params.append(new_i3_linear)
        self.new_params.append(new_d3_key_pool[0])
        self.new_params.append(new_d3_key_pool[1])
        self.new_params.append(new_d3_token_pool)
        self.new_params.append(new_d3_linear)
        
        self.new_params.append(new_i4_key_pool[0])
        self.new_params.append(new_i4_key_pool[1])
        self.new_params.append(new_i4_token_pool)
        self.new_params.append(new_i4_linear)
        self.new_params.append(new_d4_key_pool[0])
        self.new_params.append(new_d4_key_pool[1])
        self.new_params.append(new_d4_token_pool)
        self.new_params.append(new_d4_linear)
        
        self.new_params.append(new_latent_key_pool[0])
        self.new_params.append(new_latent_key_pool[1])
        self.new_params.append(new_latent_token_pool)
        self.new_params.append(new_latent_linear)

        return new_i1_key_pool, new_i1_token_pool, new_i1_linear, new_d1_key_pool, new_d1_token_pool, new_d1_linear, \
                new_i2_key_pool, new_i2_token_pool, new_i2_linear, new_d2_key_pool, new_d2_token_pool, new_d2_linear, \
                new_i3_key_pool, new_i3_token_pool, new_i3_linear, new_d3_key_pool, new_d3_token_pool, new_d3_linear, \
                new_i4_key_pool, new_i4_token_pool, new_i4_linear, new_d4_key_pool, new_d4_token_pool, new_d4_linear, \
                new_latent_key_pool, new_latent_token_pool, new_latent_linear


    def get_new_params(self):
        '''
        Returns the list of new parameters added to the model and resets the list

        Returns:
            list[torch.Tensor[float32]] : list of new parameters
        '''
        new_params = self.new_params
        self.new_params = []  # Clear the list after returning the new params
        # TokenCDC TEST:
        count = sum(p.numel() for p in new_params)
        if count > 0:
            print("Number of NEW learnable parameters: {}\n\n\n\n".format(count))
        return new_params


    def restore_model(self,
                      restore_path,
                      optimizer):
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
        '''

        # TokenCDC: Restore ALL TokenCDC params
        checkpoint = torch.load(restore_path, map_location=self.device)

        i1_key_pools_state_dict = checkpoint['i1_key_pools_state_dict']
        i1_token_pools_state_dict = checkpoint['i1_token_pools_state_dict']
        i1_linear_state_dict = checkpoint['i1_linear_state_dict']
        d1_key_pools_state_dict = checkpoint['d1_key_pools_state_dict']
        d1_token_pools_state_dict = checkpoint['d1_token_pools_state_dict']
        d1_linear_state_dict = checkpoint['d1_linear_state_dict']
        i2_key_pools_state_dict = checkpoint['i2_key_pools_state_dict']
        i2_token_pools_state_dict = checkpoint['i2_token_pools_state_dict']
        i2_linear_state_dict = checkpoint['i2_linear_state_dict']
        d2_key_pools_state_dict = checkpoint['d2_key_pools_state_dict']
        d2_token_pools_state_dict = checkpoint['d2_token_pools_state_dict']
        d2_linear_state_dict = checkpoint['d2_linear_state_dict']
        i3_key_pools_state_dict = checkpoint['i3_key_pools_state_dict']
        i3_token_pools_state_dict = checkpoint['i3_token_pools_state_dict']
        i3_linear_state_dict = checkpoint['i3_linear_state_dict']
        d3_key_pools_state_dict = checkpoint['d3_key_pools_state_dict']
        d3_token_pools_state_dict = checkpoint['d3_token_pools_state_dict']
        d3_linear_state_dict = checkpoint['d3_linear_state_dict']
        i4_key_pools_state_dict = checkpoint['i4_key_pools_state_dict']
        i4_token_pools_state_dict = checkpoint['i4_token_pools_state_dict']
        i4_linear_state_dict = checkpoint['i4_linear_state_dict']
        d4_key_pools_state_dict = checkpoint['d4_key_pools_state_dict']
        d4_token_pools_state_dict = checkpoint['d4_token_pools_state_dict']
        d4_linear_state_dict = checkpoint['d4_linear_state_dict']
        latent_key_pools_state_dict = checkpoint['latent_key_pools_state_dict']
        latent_token_pools_state_dict = checkpoint['latent_token_pools_state_dict']
        latent_linear_state_dict = checkpoint['latent_linear_state_dict']

        # Identify missing keys (keys in state_dict but not in model)
        i1_key_pools_missing_keys = set(i1_key_pools_state_dict.keys()) - self.i1_key_pools.keys()
        i1_token_pools_missing_keys = set(i1_token_pools_state_dict.keys()) - self.i1_token_pools.keys()
        i1_linear_missing_keys = set(i1_linear_state_dict.keys()) - self.i1_linear.keys()
        d1_key_pools_missing_keys = set(d1_key_pools_state_dict.keys()) - self.d1_key_pools.keys()
        d1_token_pools_missing_keys = set(d1_token_pools_state_dict.keys()) - self.d1_token_pools.keys()
        d1_linear_missing_keys = set(d1_linear_state_dict.keys()) - self.d1_linear.keys()
        i2_key_pools_missing_keys = set(i2_key_pools_state_dict.keys()) - self.i2_key_pools.keys()
        i2_linear_missing_keys = set(i2_linear_state_dict.keys()) - self.i2_linear.keys()
        i2_token_pools_missing_keys = set(i2_token_pools_state_dict.keys()) - self.i2_token_pools.keys()
        d2_key_pools_missing_keys = set(d2_key_pools_state_dict.keys()) - self.d2_key_pools.keys()
        d2_token_pools_missing_keys = set(d2_token_pools_state_dict.keys()) - self.d2_token_pools.keys()
        d2_linear_missing_keys = set(d2_linear_state_dict.keys()) - self.d2_linear.keys()
        i3_key_pools_missing_keys = set(i3_key_pools_state_dict.keys()) - self.i3_key_pools.keys()
        i3_token_pools_missing_keys = set(i3_token_pools_state_dict.keys()) - self.i3_token_pools.keys()
        i3_linear_missing_keys = set(i3_linear_state_dict.keys()) - self.i3_linear.keys()
        d3_key_pools_missing_keys = set(d3_key_pools_state_dict.keys()) - self.d3_key_pools.keys()
        d3_token_pools_missing_keys = set(d3_token_pools_state_dict.keys()) - self.d3_token_pools.keys()
        d3_linear_missing_keys = set(d3_linear_state_dict.keys()) - self.d3_linear.keys()
        i4_key_pools_missing_keys = set(i4_key_pools_state_dict.keys()) - self.i4_key_pools.keys()
        i4_token_pools_missing_keys = set(i4_token_pools_state_dict.keys()) - self.i4_token_pools.keys()
        i4_linear_missing_keys = set(i4_linear_state_dict.keys()) - self.i4_linear.keys()
        d4_key_pools_missing_keys = set(d4_key_pools_state_dict.keys()) - self.d4_key_pools.keys()
        d4_token_pools_missing_keys = set(d4_token_pools_state_dict.keys()) - self.d4_token_pools.keys()
        d4_linear_missing_keys = set(d4_linear_state_dict.keys()) - self.d4_linear.keys()
        latent_key_pools_missing_keys = set(latent_key_pools_state_dict.keys()) - self.latent_key_pools.keys()
        latent_token_pools_missing_keys = set(latent_token_pools_state_dict.keys()) - self.latent_token_pools.keys()
        latent_linear_missing_keys = set(latent_linear_state_dict.keys()) - self.latent_linear.keys()
        assert i3_key_pools_missing_keys == d4_token_pools_missing_keys, "Image/Depth Key/Token pools must have the same keys!"

        # Add pools for the missing keys (and add to new_params list to be added to optimizer)
        for mk in i4_key_pools_missing_keys:
            self.add_new_key_token_pool(mk,
                                        (i1_key_pools_state_dict[mk].shape[1],
                                            d1_key_pools_state_dict[mk].shape[1],
                                            i2_key_pools_state_dict[mk].shape[1],
                                            d2_key_pools_state_dict[mk].shape[1],
                                            i3_key_pools_state_dict[mk].shape[1],
                                            d3_key_pools_state_dict[mk].shape[1],
                                            i4_key_pools_state_dict[mk].shape[1],
                                            d4_key_pools_state_dict[mk].shape[1],
                                            latent_key_pools_state_dict[mk].shape[1]))
            if optimizer is not None:
                optimizer.add_param_group({'params' : self.get_new_params()})  # Must also add all restored params to the optimizer!
                print('NEW PARAMS ADDED TO OPTIMIZER!\n\n')

        # Now, load the state dicts
        self.i1_key_pools.load_state_dict(i1_key_pools_state_dict)
        self.i1_token_pools.load_state_dict(i1_token_pools_state_dict)
        self.i1_linear.load_state_dict(i1_linear_state_dict)
        self.d1_key_pools.load_state_dict(d1_key_pools_state_dict)
        self.d1_token_pools.load_state_dict(d1_token_pools_state_dict)
        self.d1_linear.load_state_dict(d1_linear_state_dict)
        self.i2_key_pools.load_state_dict(i2_key_pools_state_dict)
        self.i2_linear.load_state_dict(i2_linear_state_dict)
        self.i2_token_pools.load_state_dict(i2_token_pools_state_dict)
        self.d2_key_pools.load_state_dict(d2_key_pools_state_dict)
        self.d2_token_pools.load_state_dict(d2_token_pools_state_dict)
        self.d2_linear.load_state_dict(d2_linear_state_dict)
        self.i3_key_pools.load_state_dict(i3_key_pools_state_dict)
        self.i3_token_pools.load_state_dict(i3_token_pools_state_dict)
        self.i3_linear.load_state_dict(i3_linear_state_dict)
        self.d3_key_pools.load_state_dict(d3_key_pools_state_dict)
        self.d3_token_pools.load_state_dict(d3_token_pools_state_dict)
        self.d3_linear.load_state_dict(d3_linear_state_dict)
        self.i4_key_pools.load_state_dict(i4_key_pools_state_dict)
        self.i4_token_pools.load_state_dict(i4_token_pools_state_dict)
        self.i4_linear.load_state_dict(i4_linear_state_dict)
        self.d4_key_pools.load_state_dict(d4_key_pools_state_dict)
        self.d4_token_pools.load_state_dict(d4_token_pools_state_dict)
        self.d4_linear.load_state_dict(d4_linear_state_dict)
        self.latent_key_pools.load_state_dict(latent_key_pools_state_dict)
        self.latent_token_pools.load_state_dict(latent_token_pools_state_dict)
        self.latent_linear.load_state_dict(latent_linear_state_dict)
        print("ALL TokenCDC PARAMS RESTORED!\n\n\n\n\n\n")

        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("OPTIMIZER RESTORED!\n\n\n\n\n\n")
            except Exception as e:
                print(e)
                print("OPTIMIZER NOT RESTORED!\n\n\n\n\n\n")
                pass

        # Return the current step and optimizer
        return checkpoint['train_step'], optimizer


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

        # TokenCDC: Save ALL TokenCDC params
        torch.save({
                    'train_step': step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'i1_key_pools_state_dict': self.i1_key_pools.state_dict(),
                    'd1_key_pools_state_dict': self.d1_key_pools.state_dict(),
                    'i1_token_pools_state_dict': self.i1_token_pools.state_dict(),
                    'd1_token_pools_state_dict': self.d1_token_pools.state_dict(),
                    'i1_linear_state_dict': self.i1_linear.state_dict(),
                    'd1_linear_state_dict': self.d1_linear.state_dict(),
                    'i2_key_pools_state_dict': self.i2_key_pools.state_dict(),
                    'd2_key_pools_state_dict': self.d2_key_pools.state_dict(),
                    'i2_token_pools_state_dict': self.i2_token_pools.state_dict(),
                    'd2_token_pools_state_dict': self.d2_token_pools.state_dict(),
                    'i2_linear_state_dict': self.i2_linear.state_dict(),
                    'd2_linear_state_dict': self.d2_linear.state_dict(),
                    'i3_key_pools_state_dict': self.i3_key_pools.state_dict(),
                    'd3_key_pools_state_dict': self.d3_key_pools.state_dict(),
                    'i3_token_pools_state_dict': self.i3_token_pools.state_dict(),
                    'd3_token_pools_state_dict': self.d3_token_pools.state_dict(),
                    'i3_linear_state_dict': self.i3_linear.state_dict(),
                    'd3_linear_state_dict': self.d3_linear.state_dict(),
                    'i4_key_pools_state_dict': self.i4_key_pools.state_dict(),
                    'd4_key_pools_state_dict': self.d4_key_pools.state_dict(),
                    'i4_token_pools_state_dict': self.i4_token_pools.state_dict(),
                    'd4_token_pools_state_dict': self.d4_token_pools.state_dict(),
                    'i4_linear_state_dict': self.i4_linear.state_dict(),
                    'd4_linear_state_dict': self.d4_linear.state_dict(),
                    'latent_key_pools_state_dict': self.latent_key_pools.state_dict(),
                    'latent_token_pools_state_dict': self.latent_token_pools.state_dict(),
                    'latent_linear_state_dict': self.latent_linear.state_dict()
                    },
                    checkpoint_path)
