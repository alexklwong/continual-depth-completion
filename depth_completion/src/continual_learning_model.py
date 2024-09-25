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
                 key_token_pool_size,
                 device):
        super(ContinualLearningModel, self).__init__()
        
        self.key_token_pool_size = key_token_pool_size
        self.device = device
        self.dataset_uids = []  # Master list of seen datasets
        self.new_params = []  # LIST OF PARAMS to be added to the depth optimizer!

        self.image_key_pools = torch.nn.ParameterDict()
        self.depth_key_pools = torch.nn.ParameterDict()
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
            return self.image_key_pools[dataset_uid], self.depth_key_pools[dataset_uid], \
                    self.latent_token_pools[dataset_uid], self.latent_linear[dataset_uid]
                    

    def add_new_key_token_pool(self, dataset_uid, dims, manual=False):
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
        latent_dim, image_dim, depth_dim = dims

        new_image_key_pool = torch.nn.Parameter(torch.empty((image_dim, latent_dim), device=self.device), requires_grad=True)
        new_depth_key_pool = torch.nn.Parameter(torch.empty((depth_dim, latent_dim), device=self.device), requires_grad=True)
        new_latent_token_pool = torch.nn.Parameter(torch.empty((self.key_token_pool_size, latent_dim), device=self.device), requires_grad=True)
        new_latent_linear = torch.nn.Parameter(torch.empty((latent_dim,1,1), device=self.device), requires_grad=True)

        # Initialize parameters using kaiming_normal_
        torch.nn.init.kaiming_normal_(new_image_key_pool)
        torch.nn.init.kaiming_normal_(new_depth_key_pool)
        torch.nn.init.kaiming_normal_(new_latent_token_pool)
        torch.nn.init.kaiming_normal_(new_latent_linear)

        # ADD to the key and token pool dicts
        self.dataset_uids.append(dataset_uid)
        self.image_key_pools[dataset_uid] = new_image_key_pool
        self.depth_key_pools[dataset_uid] = new_depth_key_pool
        self.latent_token_pools[dataset_uid] = new_latent_token_pool
        self.latent_linear[dataset_uid] = new_latent_linear
        assert set(self.image_key_pools.keys()) == set(self.dataset_uids)
        assert set(self.latent_token_pools.keys()) == set(self.dataset_uids)

        # Add params to be added the optimizer (in the train loop in depth_completion.py)
        self.new_params.append(new_image_key_pool)
        self.new_params.append(new_depth_key_pool)
        self.new_params.append(new_latent_token_pool)
        self.new_params.append(new_latent_linear)

        return new_image_key_pool, new_depth_key_pool, new_latent_token_pool, new_latent_linear


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

        image_key_pools_state_dict = checkpoint['image_key_pools_state_dict']
        depth_key_pools_state_dict = checkpoint['depth_key_pools_state_dict']
        latent_token_pools_state_dict = checkpoint['latent_token_pools_state_dict']
        latent_linear_state_dict = checkpoint['latent_linear_state_dict']

        # Identify missing keys (keys in state_dict but not in model)
        image_key_pools_missing_keys = set(image_key_pools_state_dict.keys()) - self.image_key_pools.keys()
        depth_key_pools_missing_keys = set(depth_key_pools_state_dict.keys()) - self.depth_key_pools.keys()
        latent_token_pools_missing_keys = set(latent_token_pools_state_dict.keys()) - self.latent_token_pools.keys()
        latent_linear_missing_keys = set(latent_linear_state_dict.keys()) - self.latent_linear.keys()
        assert image_key_pools_missing_keys == depth_key_pools_missing_keys, "Image/Depth Key pools must have the same keys!"
        
        # Add pools for the missing keys (and add to new_params list to be added to optimizer)
        for mk in image_key_pools_missing_keys:
            self.add_new_key_token_pool(mk,
                                        latent_token_pools_state_dict[mk].shape[1])
            if optimizer is not None:
                new_params = self.get_new_params()
                optimizer.add_param_group({'params' : self.get_new_params()})  # Must also add all restored params to the optimizer!
                print('{} NEW PARAMS ADDED TO OPTIMIZER!\n\n'.format(sum(p.numel() for p in new_params)))

        # Now, load the state dicts
        self.image_key_pools.load_state_dict(image_key_pools_state_dict)
        self.depth_key_pools.load_state_dict(depth_key_pools_state_dict)
        self.latent_token_pools.load_state_dict(latent_token_pools_state_dict)
        self.latent_linear.load_state_dict(latent_linear_state_dict)
        print("ALL TokenCDC PARAMS RESTORED!\n\n\n\n\n\n")

        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("OPTIMIZER RESTORED!\n\n\n\n\n\n")
            except Exception as e:
                print(e, "\nOPTIMIZER NOT RESTORED!\n\n\n\n\n\n")
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
                    'image_key_pools_state_dict': self.image_key_pools.state_dict(),
                    'depth_key_pools_state_dict': self.depth_key_pools.state_dict(),
                    'latent_token_pools_state_dict': self.latent_token_pools.state_dict(),
                    'latent_linear_state_dict': self.latent_linear.state_dict()
                    },
                    checkpoint_path)
