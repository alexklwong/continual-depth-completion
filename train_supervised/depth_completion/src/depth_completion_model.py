import torch, torchvision
import utils.src.log_utils as log_utils


class DepthCompletionModel(object):
    '''
    Wrapper class for all depth completion models

    Arg(s):
        model_name : str
            depth completion model to use
        min_predict_depth : float
            minimum depth to predict
        max_predict_depth : float
            maximum depth to predict
        device : torch.device
            device to run model on
    '''

    def __init__(self,
                 model_name,
                 min_predict_depth,
                 max_predict_depth,
                 device=torch.device('cuda')):

        self.model_name = model_name
        self.device = device
        self.max_predict_depth = max_predict_depth
        print(model_name)

        kb = 'kb' in model_name
        v2 = 'v2' in model_name
        dec_bn = 'dec_bn' in model_name
        dec_kb = 'dec_kb' in model_name
        dec_3d = 'dec_3d' in model_name
        half = 'half' in model_name
        last_layer = 'last_layer' in model_name
        xatt = 'xatt' in model_name
        test = 'test' in model_name
        binning = 'binning' in model_name
        frustum = 'frustum' in model_name
        legacy = 'legacy' in model_name
        double_3d = 'double_3d' in model_name
        truncated = 'truncated' in model_name
        aggregation = 'aggregation' in model_name

        for idx, model_args in enumerate(model_name):
            if 'multi_scale' in model_args and idx != len(model_name)-1:
                print('You should input multi_scale argument at last')
                raise AssertionError
        if 'uformer' in model_name:
            from uformer_model import Uformer_Model
            from uformer_model_singlestream import Uformer_Model_SS
            if 'multi_scale' not in model_name[-1]:
                self.model = Uformer_Model(min_predict_depth=min_predict_depth,
                                        max_predict_depth=max_predict_depth,
                                        kb=kb,
                                        dec_3d=dec_3d,
                                        dec_bn=dec_bn,
                                        dec_kb=dec_kb,
                                        half=half,
                                        last_layer=last_layer,
                                        test=test,
                                        device=device)
            else:
                multi_scale = int(model_name[-1].split('multi_scale')[-1])
                squeeze_all = 'squeeze_all' in model_name
                conv_weight = False if 'linear_weight' in model_name else True
                guidance_feat = 'coord' if 'guidance_coord' in model_name else ('coord_proj' if 'guidance_proj_coord' in model_name else '')
                if aggregation:
                    self.model = Uformer_Model(min_predict_depth=min_predict_depth,
                                        max_predict_depth=max_predict_depth,
                                        kb=kb,
                                        dec_3d=dec_3d,
                                        dec_bn=dec_bn,
                                        dec_kb=dec_kb,
                                        half=half,
                                        last_layer=last_layer,
                                        test=test,
                                        aggregation=aggregation,
                                        multi_scale=multi_scale,
                                        conv_weight=conv_weight,
                                        guidance_feat=guidance_feat,
                                        device=device
                                        )
                else:
                    self.model = Uformer_Model_SS(min_predict_depth=min_predict_depth,
                                            max_predict_depth=max_predict_depth,
                                            kb=kb,
                                            dec_3d=dec_3d,
                                            dec_bn=dec_bn,
                                            dec_kb=dec_kb,
                                            last_layer=last_layer,
                                            multi_scale=multi_scale,
                                            squeeze_all=squeeze_all,
                                            half=half,
                                            frustum=frustum,
                                            binning=binning,
                                            xatt=xatt,
                                            v2=v2,
                                            legacy=legacy,
                                            double_3d=double_3d,
                                            truncated=truncated,
                                            device=device)
        elif 'nlspn' in model_name:
            from nlspn_model import NLSPNModel
            self.model = NLSPNModel(
                max_predict_depth=max_predict_depth,
                use_pretrained='pretrained' in model_name,
                device=device)
        elif 'enet' in model_name:
            from enet_model import ENetModel
            self.model = ENetModel(
                max_predict_depth=max_predict_depth,
                device=device)
        elif 'penet' in model_name:
            from penet_model import PENetModel
            self.model = PENetModel(
                max_predict_depth=max_predict_depth,
                device=device)
        elif 'msg_chn' in model_name:
            from msg_chn_model import MsgChnModel
            self.model = MsgChnModel(
                max_predict_depth=max_predict_depth,
                device=device)
        elif 'rgb_guidance_uncertainty' in model_name:
            from rgb_guidance_uncertainty_model import RGBGuidanceUncertaintyModel
            self.model = RGBGuidanceUncertaintyModel(
                use_pretrained='pretrained' in model_name,
                device=device)
        elif 'scaffnet' in model_name:
            from scaffnet_models import ScaffNetModel

            if 'vkitti' in model_name:
                dataset_name = 'vkitti'
            elif 'scenenet' in model_name:
                dataset_name = 'scenenet'
            elif 'nuscenes' in model_name:
                dataset_name = 'nuscenes'
            else:
                dataset_name = 'vkitti'

            network_modules = model_name

            self.model = ScaffNetModel(
                dataset_name=dataset_name,
                network_modules=network_modules,
                min_predict_depth=min_predict_depth,
                max_predict_depth=max_predict_depth,
                device=device)
        else:
            raise ValueError('Unsupported depth completion model: {}'.format(model_name))

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
            torch.Tensor[float32] : N x 1 x H x W dense depth map
        '''

        return self.model.forward(image, sparse_depth, intrinsics, return_all_outputs)

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

        do_supervised_loss = \
            'nlspn' in self.model_name or \
            'enet' in self.model_name or \
            'penet' in self.model_name or \
            'msg_chn' in self.model_name or \
            'scaffnet' in self.model_name or \
            'rgb_guidance_uncertainty' in self.model_name or \
            'kbformer' in self.model_name or \
            'uformer' in self.model_name[0]

        if do_supervised_loss:
            return self.model.compute_loss(
                output_depth=output_depth,
                ground_truth=ground_truth,
                image=image,
                w_losses=w_losses)
        else:
            raise ValueError('Unsupported depth completion model: {}'.format(self.model_name))

    def parameters(self):
        '''
        Returns the list of parameters in the model

        Returns:
            list[torch.Tensor[float32]] : list of parameters
        '''

        return self.model.parameters()

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

        self.model.model = torch.nn.DataParallel(self.model.model)

    def distributed_data_parallel(self, rank):
        '''
        Allows multi-gpu split along batch with 'torch.nn.parallel.DistributedDataParallel'
        '''
        self.model.model = torch.nn.parallel.DistributedDataParallel(self.model.model, device_ids=[rank], find_unused_parameters=True)

    def convert_syncbn(self):
        '''
        Convert BN layers to SyncBN layers.
        SyncBN merge the BN layer weights in every backward step.
        '''
        from torch.nn import SyncBatchNorm
        SyncBatchNorm.convert_sync_batchnorm(self.model.model)

    def restore_model(self, restore_paths, optimizer=None):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_paths : list[str]
                path to model weights
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            int : current step in optimization
            torch.optim : optimizer with restored state
        '''

        return self.model.restore_model(restore_paths, optimizer)

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

        self.model.save_model(checkpoint_path, step, optimizer)

    def log_summary(self,
                    summary_writer,
                    tag,
                    step,
                    image=None,
                    sparse_depth=None,
                    output_depth=None,
                    validity_map=None,
                    ground_truth=None,
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
            image : torch.Tensor[float32]
                N x 3 x H x W image from camera
            sparse_depth : torch.Tensor[float32]
                N x 1 x H x W sparse_depth from LiDAR
            output_depth : torch.Tensor[float32]
                N x 1 x H x W output depth for image
            validity_map : torch.Tensor[float32]
                N x 1 x H x W validity map from sparse depth
            ground_truth : torch.Tensor[float32]
                N x 1 x H x W ground truth depth image
            scalars : dict[str, float]
                dictionary of scalars to log
            n_image_per_summary : int
                number of images to display
        '''

        with torch.no_grad():
            display_summary_image = []
            display_summary_depth = []

            display_summary_image_text = tag
            display_summary_depth_text = tag

            # Log image
            if image is not None:
                # Normalize for display if necessary
                if torch.max(image) > 1.0:
                    image = image / 255.0

                image_summary = image[0:n_image_per_summary, ...]
                display_summary_image_text += '_image'
                display_summary_depth_text += '_image'

                display_summary_image.append(
                    torch.cat([
                        image_summary.cpu(),
                        torch.zeros_like(image_summary, device=torch.device('cpu'))],
                        dim=-1))

                display_summary_depth.append(display_summary_image[-1])

            if output_depth is not None:

                if output_depth.shape[1] > 1:
                    output_depth = output_depth[:, 0:1, :, :]

                output_depth_summary = output_depth[0:n_image_per_summary]
                display_summary_depth_text += '_output'

                # Add to list of images to log
                n_batch, _, n_height, n_width = output_depth_summary.shape

                display_summary_depth.append(
                    torch.cat([
                        log_utils.colorize(
                            (output_depth_summary / self.max_predict_depth).cpu(),
                            colormap='viridis'),
                        torch.zeros(n_batch, 3, n_height, n_width, device=torch.device('cpu'))],
                        dim=3))

                # Log distribution of output depth
                summary_writer.add_histogram(tag + '_output_depth_distro', output_depth, global_step=step)

            # Log output depth vs sparse depth
            if output_depth is not None and sparse_depth is not None and validity_map is not None:
                sparse_depth_summary = sparse_depth[0:n_image_per_summary]
                validity_map_summary = validity_map[0:n_image_per_summary]

                display_summary_depth_text += '_sparse-error'

                # Compute output error w.r.t. input sparse depth
                sparse_depth_error_summary = \
                    torch.abs(output_depth_summary - sparse_depth_summary)

                sparse_depth_error_summary = torch.where(
                    validity_map_summary == 1.0,
                    sparse_depth_error_summary / (sparse_depth_summary + 1e-8),
                    validity_map_summary)

                # Add to list of images to log
                sparse_depth_summary = log_utils.colorize(
                    (sparse_depth_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                sparse_depth_error_summary = log_utils.colorize(
                    (sparse_depth_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        sparse_depth_summary,
                        sparse_depth_error_summary],
                        dim=3))

                # Log distribution of sparse depth
                summary_writer.add_histogram(tag + '_sparse_depth_distro', sparse_depth, global_step=step)

            # Log output depth vs ground truth depth
            if output_depth is not None and ground_truth is not None:
                validity_map_ground_truth = torch.where(
                    ground_truth > 0,
                    torch.ones_like(ground_truth),
                    torch.zeros_like(ground_truth))

                validity_map_ground_truth_summary = validity_map_ground_truth[0:n_image_per_summary]
                ground_truth_summary = ground_truth[0:n_image_per_summary]

                display_summary_depth_text += '_groundtruth-error'

                # Compute output error w.r.t. ground truth
                ground_truth_error_summary = \
                    torch.abs(output_depth_summary - ground_truth_summary)

                ground_truth_error_summary = torch.where(
                    validity_map_ground_truth_summary == 1.0,
                    (ground_truth_error_summary + 1e-8) / (ground_truth_summary + 1e-8),
                    validity_map_ground_truth_summary)

                # Add to list of images to log
                ground_truth_summary = log_utils.colorize(
                    (ground_truth_summary / self.max_predict_depth).cpu(),
                    colormap='viridis')
                ground_truth_error_summary = log_utils.colorize(
                    (ground_truth_error_summary / 0.05).cpu(),
                    colormap='inferno')

                display_summary_depth.append(
                    torch.cat([
                        ground_truth_summary,
                        ground_truth_error_summary],
                        dim=3))

                # Log distribution of ground truth
                summary_writer.add_histogram(tag + '_ground_truth_distro', ground_truth, global_step=step)

            # Log scalars to tensorboard
            for (name, value) in scalars.items():
                summary_writer.add_scalar(tag + '_' + name, value, global_step=step)

            # Log image summaries to tensorboard
            if len(display_summary_image) >= 1:
                display_summary_image = torch.cat(display_summary_image, dim=2)

                summary_writer.add_image(
                    display_summary_image_text,
                    torchvision.utils.make_grid(display_summary_image, nrow=n_image_per_summary),
                    global_step=step)

            if len(display_summary_depth) >= 1:
                display_summary_depth = torch.cat(display_summary_depth, dim=2)

                summary_writer.add_image(
                    display_summary_depth_text,
                    torchvision.utils.make_grid(display_summary_depth, nrow=n_image_per_summary),
                    global_step=step)
