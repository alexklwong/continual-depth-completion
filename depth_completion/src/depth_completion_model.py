import os, torch, torchvision
from utils.src import log_utils
from continual_learning_losses import ewc_loss, lwf_loss


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
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 model_name,
                 network_modules,
                 min_predict_depth,
                 max_predict_depth,
                 device=torch.device('cuda')):

        self.model_name = model_name
        self.network_modules = network_modules
        self.min_predict_depth = min_predict_depth
        self.max_predict_depth = max_predict_depth
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

    def forward_depth(self, image, sparse_depth, validity_map, intrinsics=None, return_all_outputs=False):
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
            return_all_outputs : bool
                if set, then return list of N x 1 x H x W depth maps else a single N x 1 x H x W depth map
        Returns:
            list[torch.Tensor[float32]] : a single or list of N x 1 x H x W outputs
        '''

        return self.model.forward_depth(
            image,
            sparse_depth,
            validity_map,
            intrinsics,
            return_all_outputs)

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

        # TODO: Add frozen model as argument to loss computation (i.e. EWC, LWF)

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

        if w_losses['w_lwf']:
            #to debug this, I modified a few lines in random crop, need to fix back later
            frozen_model_output_depth0 = frozen_model.forward_depth(image0, sparse_depth0, validity_map_depth0, intrinsics, return_all_outputs=True)
            
            loss_lwf = lwf_loss(output_depth0, frozen_model_output_depth0, w_losses['w_lwf'])
            loss += loss_lwf
            loss_info['loss_lwf'] = loss_lwf

        return loss, loss_info

    def parameters_depth(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.model.parameters_depth()

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
        if self.ewc and self.prev_fisher is None and not frozen_model:
            # Assuming that if ewc we should have fisher information matrix path at the last of restore paths
            assert 'fisher' in restore_paths[-1]
            self.prev_fisher = torch.load(restore_paths[-1])

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

        if self.calculate_fisher_enabled and self.fisher is not None:
            torch.save(self.fisher, os.path.join(checkpoint_dirpath, 'fisher-info_{}.pth'.format(step)))

    def update_fisher(self):
        '''
        Update fisher information matrix at the end of an epoch
        '''
        self.fisher = self.epoch_fisher
        self.epoch_fisher = net_utils.init_fisher(self.model.parameters_depth())

    def calculate_fisher(self, normalization):
        '''
        Calculate fisher information matrix at the end of a batch

        Arg(s):
            normalization : int
                length of dataset
        '''
        self.epoch_fisher = net_utils.compute_fisher(
                    fisher_info=self.epoch_fisher,
                    parameters=self.model.parameters_depth(),
                    normalization=normalization)

    # TODO add fisher matrix to model save
    # def save_fisher(self,
    #                checkpoint_dirpath,
    #                step,
    #                fisher_info):

    #     if 'kbnet' in self.model_name:
    #         checkpoint_path = os.path.join(checkpoint_dirpath, 'kbnet-{}.pth'.format(step))
    #     elif 'scaffnet' in self.model_name:
    #          checkpoint_path = os.path.join(checkpoint_dirpath, 'scaffnet-{}.pth'.format(step))
    #     elif 'fusionnet' in self.model_name:
    #          checkpoint_path = os.path.join(checkpoint_dirpath, 'fusionnet-{}.pth'.format(step))
    #     elif 'voiced' in self.model_name:
    #          checkpoint_path = os.path.join(checkpoint_dirpath, 'voiced-{}.pth'.format(step))
    #     else:
    #         raise ValueError('Unsupported depth completion model: {}'.format(self.model_name))

    #     if not os.path.exists(checkpoint_path):
    #         raise FileExistsError('Model checkpoint does not exist {}'.format(checkpoint_dirpath))

    #     checkpoint = torch.load(checkpoint_dirpath)

    #     checkpoint['fisher_info'] = fisher_info

    #     torch.save(checkpoint, checkpoint_dirpath)

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
                print("output_depth0 shape:", output_depth0.shape)
                print("output_depth0 contents:", output_depth0)

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