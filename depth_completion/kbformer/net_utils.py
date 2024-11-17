import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum
# import DCN
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable

'''
Utility functions for rigid warping
'''
def meshgrid(n_batch, n_height, n_width, device, homogeneous=True, normalize=False):
    '''
    Creates N x 2 x H x W meshgrid in x, y directions

    Arg(s):
        n_batch : int
            batch size
        n_height : int
            height of tensor
        n_width : int
            width of tensor
        device : torch.device
            device on which to create meshgrid
        homoegenous : bool
            if set, then add homogeneous coordinates (N x H x W x 3)
    Return:
        torch.Tensor[float32]: N x 2 x H x W meshgrid of x, y and 1 (if homogeneous)
    '''

    if normalize:
        x = torch.linspace(start=0.0, end=1-1/n_width, steps=n_width, device=device)
        y = torch.linspace(start=0.0, end=1-1/n_height, steps=n_height, device=device)
    else:
        x = torch.linspace(start=0.0, end=n_width-1, steps=n_width, device=device)
        y = torch.linspace(start=0.0, end=n_height-1, steps=n_height, device=device)

    # Create H x W grids
    grid_y, grid_x = torch.meshgrid(y, x)

    if homogeneous:
        # Create 3 x H x W grid (x, y, 1)
        grid_xy = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=0)
    else:
        # Create 2 x H x W grid (x, y)
        grid_xy = torch.stack([grid_x, grid_y], dim=0)

    grid_xy = torch.unsqueeze(grid_xy, dim=0) \
        .repeat(n_batch, 1, 1, 1)

    return grid_xy

def backproject_to_camera(depth, intrinsics, shape):
    '''
    Backprojects pixel coordinates to 3D camera coordinates

    Arg(s):
        depth : torch.Tensor[float32]
            N x 1 x H x W depth map
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 4 x (H x W)
    '''
    n_batch, _, n_height, n_width = shape

    # Create homogeneous coordinates [x, y, 1]
    xy_h = meshgrid(n_batch, n_height, n_width, device=depth.device, homogeneous=True)

    # Reshape pixel coordinates to N x 3 x (H x W)
    xy_h = xy_h.view(n_batch, 3, -1)

    # Reshape depth as N x 1 x (H x W)
    depth = depth.reshape(n_batch, 1, -1)

    # K^-1 [x, y, 1] z
    points = torch.matmul(torch.inverse(intrinsics), xy_h) * depth

    # Make homogeneous
    return torch.cat([points, torch.ones_like(depth)], dim=1)

def project_to_pixel(points, pose, intrinsics, shape):
    '''
    Projects points in camera coordinates to 2D pixel coordinates

    Arg(s):
        points : torch.Tensor[float32]
            N x 4 x (H x W) depth map
        pose : torch.Tensor[float32]
            N x 4 x 4 transformation matrix
        intrinsics : torch.Tensor[float32]
            N x 3 x 3 camera intrinsics
        shape : list[int]
            shape of tensor in (N, C, H, W)
    Return:
        torch.Tensor[float32] : N x 2 x H x W
    '''

    n_batch, _, n_height, n_width = shape

    # Convert camera intrinsics to homogeneous coordinates
    column = torch.zeros([n_batch, 3, 1], device=points.device)
    row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=points.device) \
        .view(1, 1, 4) \
        .repeat(n_batch, 1, 1)
    intrinsics = torch.cat([intrinsics, column], dim=2)
    intrinsics = torch.cat([intrinsics, row], dim=1)

    # Apply the transformation and project: \pi K g p
    T = torch.matmul(intrinsics, pose)
    T = T[:, 0:3, :]
    points = torch.matmul(T, points)
    points = points / (torch.unsqueeze(points[:, 2, :], dim=1) + 1e-7)
    points = points[:, 0:2, :]

    # Reshape to N x 2 x H x W
    return points.view(n_batch, 2, n_height, n_width)

def grid_sample(image, target_xy, shape, padding_mode='border'):
    '''
    Samples the image at x, y locations to target x, y locations

    Arg(s):
        image : torch.Tensor[float32]
            N x 3 x H x W RGB image
        target_xy : torch.Tensor[float32]
            N x 2 x H x W target x, y locations in image space
        shape : list[int]
            shape of tensor in (N, C, H, W)
        padding_mode : str
            padding to use when sampled out of bounds
    Return:
        torch.Tensor[float32] : N x 3 x H x W RGB image
    '''

    n_batch, _, n_height, n_width = shape

    # Swap dimensions to N x H x W x 2 for grid sample
    target_xy = target_xy.permute(0, 2, 3, 1)

    # Normalize coordinates between -1 and 1
    target_xy[..., 0] /= (n_width - 1.0)
    target_xy[..., 1] /= (n_height - 1.0)
    target_xy = 2.0 * (target_xy - 0.5)

    # Sample the image at normalized target x, y locations
    return torch.nn.functional.grid_sample(
        image,
        grid=target_xy,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True)


def _activation_func(activation_fn):
    '''
    Select activation function

    Arg(s):
        activation_fn : str
            name of activation function
    Returns:
        torch.nn.Module : activation function
    '''

    if 'linear' in activation_fn:
        return None
    elif 'leaky_relu' in activation_fn:
        return torch.nn.LeakyReLU(negative_slope=0.20, inplace=True)
    elif 'relu' in activation_fn:
        return torch.nn.ReLU()
    elif 'elu' in activation_fn:
        return torch.nn.ELU()
    elif 'sigmoid' in activation_fn:
        return torch.nn.Sigmoid()
    elif 'gelu' in activation_fn:
        return torch.nn.GELU()
    else:
        raise ValueError('Unsupported activation function: {}'.format(activation_fn))


class Conv3D(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(Conv3D, self).__init__()
        self.conv = nn.Conv3d(nin, nout, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm3d(nout)
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return out

# Reference : https://github.com/qijiezhao/pseudo-3d-pytorch/blob/master/p3d_model.py
class P3D(nn.Module):
    def __init__(self, nin, nout, activation_func=nn.ELU(inplace=False)):
        super(P3D, self).__init__()
        self.conv1 = nn.Conv3d(nin,nout,kernel_size=(1,3,3),stride=1, padding=(0,1,1),bias=False)
        self.bn1 = nn.BatchNorm3d(nout)
        self.conv2 = nn.Conv3d(nout,nout,kernel_size=(3,1,1),stride=1,padding=(1,0,0),bias=False)
        self.bn2 = nn.BatchNorm3d(nout)
        self.act = activation_func

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)
        return x


class UpConv3d(nn.Module):
    '''
    Up-convolution 3d (upsample + convolution) block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
    '''


    def __init__(self, in_channels, out_channels, mode='nearest', activation_func=torch.nn.LeakyReLU(negative_slope=0.20, inplace=True), conv_type=P3D):
        super().__init__()

        self.conv = P3D(in_channels, out_channels)
        self.up_mode = mode
        self.activation_func = activation_func
    def forward(self, x1, shape):
        x1 = F.interpolate(x1, size=shape, mode='nearest')
        return self.activation_func(self.conv(x1))


class UpConv2d(torch.nn.Module):
    '''
    Up-convolution (upsample + convolution) block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        shape : list[int]
            two element tuple of ints (height, width)
        kernel_size : int
            size of kernel (k x k)
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=True,
                 use_instance_norm=False):
        super(UpConv2d, self).__init__()

        self.conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x, shape):
        '''
        Forward input x through an up convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        upsample = torch.nn.functional.interpolate(x, size=shape, mode='nearest')
        conv = self.conv(upsample)
        return conv

'''
Network decoder blocks
'''
class DecoderBlock(torch.nn.Module):
    '''
    Decoder block with skip connection

    Arg(s):
        in_channels : int
            number of input channels
        skip_channels : int
            number of skip connection channels
        out_channels : int
            number of output channels
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        deconv_type : str
            deconvolution types: transpose, up
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 skip_channels,
                 out_channels,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 deconv_type='up',
                 use_depthwise_separable=False):
        super(DecoderBlock, self).__init__()

        self.skip_channels = skip_channels

        self.deconv = UpConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

        concat_channels = skip_channels + out_channels

        conv2d = Conv2d

        self.conv = conv2d(
            concat_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)

    def forward(self, x, skip=None, shape=None):
        '''
        Forward input x through a decoder block and fuse with skip connection

        Arg(s):
            x : torch.Tensor[float32]
                N x C x h x w input tensor
            skip : torch.Tensor[float32]
                N x F x H x W skip connection
            shape : tuple[int]
                height, width (H, W) tuple denoting output shape
        Returns:
            torch.Tensor[float32] : N x K x H x W output tensor
        '''

        if self.deconv_type == 'transpose':
            deconv = self.deconv(x)
        elif self.deconv_type == 'up':

            if skip is not None:
                shape = skip.shape[2:4]
            elif shape is not None:
                pass
            else:
                n_height, n_width = x.shape[2:4]
                shape = (int(2 * n_height), int(2 * n_width))

            deconv = self.deconv(x, shape=shape)

        if self.skip_channels > 0:
            concat = torch.cat([deconv, skip], dim=1)
        else:
            concat = deconv

        return self.conv(concat)

class SparseToDensePool(torch.nn.Module):
    '''
    Converts sparse inputs to dense outputs using max and min pooling
    with different kernel sizes and combines them with 1 x 1 convolutions

    Arg(s):
        input_channels : int
            number of channels to be fed to max and/or average pool(s)
        min_pool_sizes : list[int]
            list of min pool sizes s (kernel size is s x s)
        max_pool_sizes : list[int]
            list of max pool sizes s (kernel size is s x s)
        n_filter : int
            number of filters for 1 x 1 convolutions
        n_convolution : int
            number of 1 x 1 convolutions to use for balancing detail and density
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
    '''

    def __init__(self,
                 input_channels,
                 min_pool_sizes=[3, 5, 7, 9],
                 max_pool_sizes=[3, 5, 7, 9],
                 n_filter=8,
                 n_convolution=3,
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu'):
        super(SparseToDensePool, self).__init__()

        activation_func = _activation_func(activation_func)

        self.min_pool_sizes = [
            s for s in min_pool_sizes if s > 1
        ]

        self.max_pool_sizes = [
            s for s in max_pool_sizes if s > 1
        ]

        # Construct min pools
        self.min_pools = []
        for s in self.min_pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.min_pools.append(pool)

        # Construct max pools
        self.max_pools = []
        for s in self.max_pool_sizes:
            padding = s // 2
            pool = torch.nn.MaxPool2d(kernel_size=s, stride=1, padding=padding)
            self.max_pools.append(pool)

        self.len_pool_sizes = len(self.min_pool_sizes) + len(self.max_pool_sizes)

        in_channels = len(self.min_pool_sizes) + len(self.max_pool_sizes)

        pool_convs = []
        for _ in range(n_convolution):
            conv = Conv2d(
                in_channels,
                n_filter,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=True,
                use_instance_norm=False)
            pool_convs.append(conv)

            # Set new input channels as output channels
            in_channels = n_filter

        self.pool_convs = torch.nn.Sequential(*pool_convs)

        in_channels = n_filter + input_channels

        self.conv = Conv2d(
            in_channels,
            n_filter,
            kernel_size=3,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=True,
            use_instance_norm=False)

    def forward(self, x):
        # Input depth
        z = torch.unsqueeze(x[:, 0, ...], dim=1)

        pool_pyramid = []

        # Use min and max pooling to densify and increase receptive field
        for pool, s in zip(self.min_pools, self.min_pool_sizes):
            # Set flag (999) for any zeros and max pool on -z then revert the values
            z_pool = -pool(torch.where(z == 0, -999 * torch.ones_like(z), -z))
            # Remove any 999 from the results
            z_pool = torch.where(z_pool == 999, torch.zeros_like(z), z_pool)

            pool_pyramid.append(z_pool)
        for pool, s in zip(self.max_pools, self.max_pool_sizes):
            z_pool = pool(z)

            pool_pyramid.append(z_pool)
        # Stack max and minpools into pyramid
        pool_pyramid = torch.cat(pool_pyramid, dim=1)

        # Learn weights for different kernel sizes, and near and far structures
        pool_convs = self.pool_convs(pool_pyramid)

        pool_convs = torch.cat([pool_convs, x], dim=1)
        return self.conv(pool_convs)


class VGGNetBlock(torch.nn.Module):
    '''
    VGGNet block class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        n_convolution : int
            number of convolution layers
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
        use_depthwise_separable : bool
            if set, then use depthwise separable convolutions instead of convolutions
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 n_convolution=1,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False,
                 use_depthwise_separable=False):
        super(VGGNetBlock, self).__init__()

        if use_depthwise_separable:
            conv2d = DepthwiseSeparableConv2d
        else:
            conv2d = Conv2d

        layers = []
        for n in range(n_convolution - 1):
            conv = conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm,
                use_instance_norm=use_instance_norm)
            layers.append(conv)
            in_channels = out_channels

        conv = conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm,
            use_instance_norm=use_instance_norm)
        layers.append(conv)

        self.conv_block = torch.nn.Sequential(*layers)

    def forward(self, x):
        '''
        Forward input x through a VGG block

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        return self.conv_block(x)

'''
Network layers
'''
class Conv2d(torch.nn.Module):
    '''
    2D convolution class

    Arg(s):
        in_channels : int
            number of input channels
        out_channels : int
            number of output channels
        kernel_size : int
            size of kernel
        stride : int
            stride of convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then apply batch normalization
        use_instance_norm : bool
            if set, then apply instance normalization
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),
                 use_batch_norm=False,
                 use_instance_norm=False):
        super(Conv2d, self).__init__()

        padding = kernel_size // 2

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

        # Select the type of weight initialization, by default kaiming_uniform
        if weight_initializer == 'kaiming_normal':
            torch.nn.init.kaiming_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_normal':
            torch.nn.init.xavier_normal_(self.conv.weight)
        elif weight_initializer == 'xavier_uniform':
            torch.nn.init.xavier_uniform_(self.conv.weight)
        elif weight_initializer == 'kaiming_uniform':
            pass
        else:
            raise ValueError('Unsupported weight initializer: {}'.format(weight_initializer))

        self.activation_func = activation_func

        assert not (use_batch_norm and use_instance_norm), \
            'Unable to apply both batch and instance normalization'

        self.use_batch_norm = use_batch_norm
        self.use_instance_norm = use_instance_norm

        if use_batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        elif use_instance_norm:
            self.instance_norm = torch.nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        '''
        Forward input x through a convolution layer

        Arg(s):
            x : torch.Tensor[float32]
                N x C x H x W input tensor
        Returns:
            torch.Tensor[float32] : N x K x h x w output tensor
        '''

        conv = self.conv(x)

        if self.use_batch_norm:
            conv = self.batch_norm(conv)
        elif self.use_instance_norm:
            conv = self.instance_norm(conv)

        if self.activation_func is not None:
            return self.activation_func(conv)
        else:
            return conv


class CalibratedBackprojectionBlock(torch.nn.Module):
    '''
    Calibrated backprojection (KB) layer class

    Arg(s):
        in_channels_image : int
            number of input channels for image (RGB) branch
        in_channels_depth : int
            number of input channels for depth branch
        in_channels_fused : int
            number of input channels for RGB 3D fusion branch
        n_filter_image : int
            number of filters for image (RGB) branch
        n_filter_depth : int
            number of filters for depth branch
        n_filter_fused : int
            number of filters for RGB 3D fusion branch
        n_convolution_image : int
            number of convolution layers in image branch
        n_convolution_depth : int
            number of convolution layers in depth branch
        n_convolution_fused : int
            number of convolution layers in RGB 3D fusion branch
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
    '''
    def __init__(self,
                 in_channels_image,
                 in_channels_depth,
                 in_channels_fused,
                 n_filter_image=48,
                 n_filter_depth=16,
                 n_filter_fused=48,
                 n_convolution_image=1,
                 n_convolution_depth=1,
                 n_convolution_fused=1,
                 weight_initializer='kaiming_uniform',
                 activation_func=torch.nn.LeakyReLU(negative_slope=0.10, inplace=True)):
        super(CalibratedBackprojectionBlock, self).__init__()

        self.conv_image = VGGNetBlock(
            in_channels=in_channels_image,
            out_channels=n_filter_image,
            n_convolution=n_convolution_image,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.conv_depth = VGGNetBlock(
            in_channels=in_channels_depth + 3,
            out_channels=n_filter_depth,
            n_convolution=n_convolution_depth,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.proj_depth = Conv2d(
            in_channels=in_channels_depth,
            out_channels=1,
            kernel_size=1,
            stride=1,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

        self.conv_fused = Conv2d(
            in_channels=in_channels_fused + 3,
            out_channels=n_filter_fused,
            kernel_size=1,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func)

    def forward(self, image, depth, coordinates, fused=None):

        layers_fused = []

        # Obtain image (RGB) features
        conv_image = self.conv_image(image)

        # Obtain depth (Z) features
        conv_depth = self.conv_depth(torch.cat([depth, coordinates], dim=1))

        # Include image (RGB) features
        layers_fused.append(image)

        # Project depth features to 1 dimension
        z = self.proj_depth(depth)

        # Include backprojected 3D positional (XYZ) encoding: K^-1 [x y 1] z
        xyz = coordinates * z
        layers_fused.append(xyz)

        # Include previous RGBXYZ representation
        if fused is not None:
            layers_fused.append(fused)

        # Obtain fused (RGBXYZ) representation
        layers_fused = torch.cat(layers_fused, dim=1)
        conv_fused = self.conv_fused(layers_fused)

        return conv_image, conv_depth, conv_fused




class FastLeFF(nn.Module):

    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0.):
        super().__init__()

        from torch_dwconv import depthwise_conv2d, DepthwiseConv2d

        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(DepthwiseConv2d(hidden_dim, hidden_dim, kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.dim*self.hidden_dim
        # dwconv
        flops += H*W*self.hidden_dim*3*3
        # fc2
        flops += H*W*self.hidden_dim*self.dim
        print("LeFF:{%.2f}"%(flops/1e9))
        return flops


def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


# Supervised Attention Module

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img


class LPU(nn.Module):
    """
    Local Perception Unit to extract local infomation.
    LPU(X) = DWConv(X) + X
    """
    def __init__(self, in_channels, out_channels, stride = 1):
        super(LPU, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, out_channels, kernel_size = 3,
            stride = stride, padding = 1, groups = in_channels, bias = True
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        result = (self.depthwise(x) + x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return result

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.out_channels*3*3
        return flops

#########################################
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s = s

    def forward(self, x, H=None, W=None):
        B, N, C = x.shape
        H = H or int(math.sqrt(N))
        W = W or int(math.sqrt(N))
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.reduction = reduction

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x

    def flops(self):
        flops = 0
        flops += self.channel*self.channel/self.reduction*2

        return flops

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size =k_size

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel*self.channel*self.k_size

        return flops

class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer_1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.channel = channel
        self.k_size =k_size

    def forward(self, x):
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))

        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

    def flops(self):
        flops = 0
        flops += self.channel*self.channel*self.k_size

        return flops

class SepConv2d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,act_layer=nn.ReLU):
        super(SepConv2d, self).__init__()
        self.depthwise = torch.nn.Conv2d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, HW):
        flops = 0
        flops += HW*self.in_channels*self.kernel_size**2/self.stride**2
        flops += HW*self.in_channels*self.out_channels
        print("SeqConv2d:{%.2f}"%(flops/1e9))
        return flops

######## Embedding for q,k,v ########
class ConvProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, kernel_size=3, q_stride=1, k_stride=1, v_stride=1, dropout = 0.,
                 last_stage=False,bias=True):

        super().__init__()

        inner_dim = dim_head *  heads
        self.heads = heads
        pad = (kernel_size - q_stride)//2
        self.to_q = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x, attn_kv=None):
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x = rearrange(x, 'b (l w) c -> b c l w', l=l, w=w)
        attn_kv = rearrange(attn_kv, 'b (l w) c -> b c l w', l=l, w=w)
        # print(attn_kv)
        q = self.to_q(x)
        q = rearrange(q, 'b (h d) l w -> b h (l w) d', h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, 'b (h d) l w -> b h (l w) d', h=h)
        v = rearrange(v, 'b (h d) l w -> b h (l w) d', h=h)
        return q,k,v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q,k,v

    def flops(self, q_L, kv_L=None):
        kv_L = kv_L or q_L
        flops = q_L*self.dim*self.inner_dim+kv_L*self.dim*self.inner_dim*2
        return flops


#########################################
########### window-based self-attention #############
class WindowAttention(nn.Module):
    def __init__(self, dim, win_size,num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0]) # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1]) # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection =='conv':
            self.qkv = ConvProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        elif token_projection =='linear':
            self.qkv = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x,attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1)//relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N*ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N*ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}'

    def flops(self, H, W):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        N = self.win_size[0]*self.win_size[1]
        nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(H*W, H*W)

        # attn = (q @ k.transpose(-2, -1))

        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)

        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}" % (flops/1e9))
        return flops

# self-attention
class Attention(nn.Module):
    def __init__(self, dim, num_heads, token_projection='linear',
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = LinearProjection(dim, num_heads, dim//num_heads ,bias=qkv_bias)

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_kv=None, mask=None):
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # ratio = attn.size(-1)//relative_position_bias.size(-1)
        # relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)

        # attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, q_num, kv_num):
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        # N = self.win_size[0]*self.win_size[1]
        # nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(q_num, kv_num)
        # attn = (q @ k.transpose(-2, -1))

        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num

        # x = self.proj(x)
        flops += q_num * self.dim * self.dim
        print("MCA:{%.2f}"%(flops/1e9))
        return flops


# feed-forward network
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.in_features*self.hidden_features
        # fc2
        flops += H*W*self.hidden_features*self.out_features
        print("MLP:{%.2f}"%(flops/1e9))
        return flops


class LeFF(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU,drop = 0., use_eca=False):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim),
                                act_layer())
        self.dwconv = nn.Sequential(nn.Conv2d(hidden_dim,hidden_dim,groups=hidden_dim,kernel_size=3,stride=1,padding=1),
                        act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x):
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))

        x = self.linear1(x)

        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h = hh, w = hh)
        # bs,hidden_dim,32x32

        x = self.dwconv(x)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h = hh, w = hh)

        x = self.linear2(x)
        x = self.eca(x)

        return x

    def flops(self, H, W):
        flops = 0
        # fc1
        flops += H*W*self.dim*self.hidden_dim
        # dwconv
        flops += H*W*self.hidden_dim*3*3
        # fc2
        flops += H*W*self.hidden_dim*self.dim
        print("LeFF:{%.2f}"%(flops/1e9))
        # eca
        if hasattr(self.eca, 'flops'):
            flops += self.eca.flops()
        return flops

# window operation


def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


#########################################
# Downsample Block
class Downsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        # import pdb;pdb.set_trace()
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.conv(x).flatten(2).transpose(1,2).contiguous()  # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H/2*W/2*self.in_channel*self.out_channel*4*4
        print("Downsample:{%.2f}"%(flops/1e9))
        return flops

# Upsample Block
class Upsample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).contiguous().view(B, C, H, W)
        out = self.deconv(x).flatten(2).transpose(1,2).contiguous() # B H*W C
        return out

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*2*W*2*self.in_channel*self.out_channel*2*2
        print("Upsample:{%.2f}"%(flops/1e9))
        return flops


# Input Projection
class InputProj(nn.Module):
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None,act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        print("Input_proj:{%.2f}"%(flops/1e9))
        return flops


# Output Projection
class OutputProj(nn.Module):
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None,act_layer=None):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size//2),
        )
        if act_layer is not None:
            self.proj.add_module(act_layer(inplace=True))
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel = in_channel
        self.out_channel = out_channel

    def forward(self, x):
        B, L, C = x.shape
        H = int(math.sqrt(L))
        W = int(math.sqrt(L))
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, H, W):
        flops = 0
        # conv
        flops += H*W*self.in_channel*self.out_channel*3*3

        if self.norm is not None:
            flops += H*W*self.out_channel
        print("Output_proj:{%.2f}"%(flops/1e9))
        return flops


class ModulatedDeformConvFunction(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias,
                stride, padding, dilation, groups, deformable_groups, im2col_step):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.kernel_size = _pair(weight.shape[2:4])
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.im2col_step = im2col_step
        import DCN
        output = DCN.modulated_deform_conv_forward(input, weight, bias, offset,
                                                   mask,
                                                   ctx.kernel_size[0],
                                                   ctx.kernel_size[1],
                                                   ctx.stride[0],
                                                   ctx.stride[1],
                                                   ctx.padding[0],
                                                   ctx.padding[1],
                                                   ctx.dilation[0],
                                                   ctx.dilation[1],
                                                   ctx.groups,
                                                   ctx.deformable_groups,
                                                   ctx.im2col_step)
        ctx.save_for_backward(input, offset, mask, weight, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, offset, mask, weight, bias = ctx.saved_tensors
        grad_input, grad_offset, grad_mask, grad_weight, grad_bias = \
            DCN.modulated_deform_conv_backward(input, weight,
                                               bias,
                                               offset, mask,
                                               grad_output,
                                               ctx.kernel_size[0],
                                               ctx.kernel_size[1],
                                               ctx.stride[0],
                                               ctx.stride[1],
                                               ctx.padding[0],
                                               ctx.padding[1],
                                               ctx.dilation[0],
                                               ctx.dilation[1],
                                               ctx.groups,
                                               ctx.deformable_groups,
                                               ctx.im2col_step)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias, \
            None, None, None, None, None, None
