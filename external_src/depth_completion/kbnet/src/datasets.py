'''
Author: Alex Wong <alexw@cs.ucla.edu>

If you use this code, please cite the following paper:

A. Wong, and S. Soatto. Unsupervised Depth Completion with Calibrated Backprojection Layers.
https://arxiv.org/pdf/2108.10531.pdf

@inproceedings{wong2021unsupervised,
  title={Unsupervised Depth Completion with Calibrated Backprojection Layers},
  author={Wong, Alex and Soatto, Stefano},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12747--12756},
  year={2021}
}
'''
import numpy as np
import torch.utils.data
import data_utils


def load_image_triplet(path, normalize=True, data_format='CHW'):
    '''
    Load in triplet frames from path

    Arg(s):
        path : str
            path to image triplet
        normalize : bool
            if set, normalize to [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : image at t - 1
        numpy[float32] : image at t
        numpy[float32] : image at t + 1
    '''

    # Load image triplet and split into images at t-1, t, t+1
    images = data_utils.load_image(
        path,
        normalize=normalize,
        data_format=data_format)

    # Split along width
    image1, image0, image2 = np.split(images, indices_or_sections=3, axis=-1)

    return image1, image0, image2

def load_depth_with_validity_map(depth_path, data_format='CHW'):
    '''
    Load depth and validity maps


    Arg(s):
        depth_path : str
            path to depth map
        data_format : str
            'CHW', or 'HWC'
        Returns:
            numpy[float32] : depth and validity map (2 x H x W)
    '''

    depth_map, validity_map = data_utils.load_depth_with_validity_map(
        depth_path,
        data_format=data_format)

    if data_format == 'CHW':
        return np.concatenate([depth_map, validity_map], axis=0)
    elif data_format == 'HWC':
        return np.concatenate([depth_map, validity_map], axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

def random_crop(inputs, shape, intrinsics=None, crop_type=['none']):
    '''
    Apply crop to inputs e.g. images, depth and if available adjust camera intrinsics

    Arg(s):
        inputs : list[numpy[float32]]
            list of numpy arrays e.g. images, depth, and validity maps
        shape : list[int]
            shape (height, width) to crop inputs
        intrinsics : numpy[float32]
            3 x 3 camera intrinsics matrix
        crop_type : str
            none, horizontal, vertical, anchored, bottom
    Return:
        list[numpy[float32]] : list of cropped inputs
        numpy[float32] : if given, 3 x 3 adjusted camera intrinsics matrix
    '''

    n_height, n_width = shape
    _, o_height, o_width = inputs[0].shape

    # Get delta of crop and original height and width
    d_height = o_height - n_height
    d_width = o_width - n_width

    # By default, perform center crop
    y_start = d_height // 2
    x_start = d_width // 2

    if 'horizontal' in crop_type:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.0, 0.50, 1.0
            ]

            widths = [
                anchor * d_width for anchor in crop_anchors
            ]
            x_start = int(widths[np.random.randint(low=0, high=len(widths))])

        # Randomly select a crop location
        else:
            x_start = np.random.randint(low=0, high=d_width)

    # If bottom alignment, then set starting height to bottom position
    if 'bottom' in crop_type:
        y_start = d_height

    elif 'vertical' in crop_type and np.random.rand() <= 0.30:

        # Select from one of the pre-defined anchored locations
        if 'anchored' in crop_type:
            # Create anchor positions
            crop_anchors = [
                0.50, 1.0
            ]

            heights = [
                anchor * d_height for anchor in crop_anchors
            ]
            y_start = int(heights[np.random.randint(low=0, high=len(heights))])

        # Randomly select a crop location
        else:
            y_start = np.random.randint(low=0, high=d_height)

    # Crop each input into (n_height, n_width)
    y_end = y_start + n_height
    x_end = x_start + n_width
    outputs = [
        T[:, y_start:y_end, x_start:x_end] for T in inputs
    ]

    if intrinsics is not None:
        # Adjust intrinsics
        intrinsics = intrinsics + [[0.0, 0.0, -x_start],
                                   [0.0, 0.0, -y_start],
                                   [0.0, 0.0, 0.0     ]]

        return outputs, intrinsics
    else:
        return outputs


class KBNetTrainingDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image at time t-1, t, and t+1
        (2) sparse depth
        (3) camera intrinsics matrix

    Arg(s):
        image_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to 3 x 3 camera intrinsics matrix
        shape : tuple[int]
            shape (height, width) to crop inputs
        random_crop_type : list[str]
            none, horizontal, vertical, anchored, bottom
    '''

    def __init__(self,
                 images_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 shape=None,
                 random_crop_type=['none']):

        self.images_paths = images_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.shape = shape
        self.do_random_crop = \
            self.shape is not None and all([x > 0 for x in self.shape])

        # Augmentation
        self.random_crop_type = random_crop_type

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        image1, image0, image2 = load_image_triplet(
            self.images_paths[index],
            normalize=False,
            data_format=self.data_format)

        # Load depth
        sparse_depth0 = data_utils.load_depth(
            self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)

        # Crop image, depth and adjust intrinsics
        if self.do_random_crop:
            [image0, image1, image2, sparse_depth0], intrinsics = random_crop(
                inputs=[image0, image1, image2, sparse_depth0],
                shape=self.shape,
                intrinsics=intrinsics,
                crop_type=self.random_crop_type)

        # Convert to float32
        image0, image1, image2, sparse_depth0, intrinsics = [
            T.astype(np.float32)
            for T in [image0, image1, image2, sparse_depth0, intrinsics]
        ]

        return image0, image1, image2, sparse_depth0, intrinsics

    def __len__(self):
        return len(self.image_paths)


class KBNetInferenceDataset(torch.utils.data.Dataset):
    '''
    Dataset for fetching:
        (1) image
        (2) sparse depth
        (3) camera intrinsics matrix

    Arg(s):
        image_paths : list[str]
            paths to image triplets
        sparse_depth_paths : list[str]
            paths to sparse depth maps
        intrinsics_paths : list[str]
            paths to 3 x 3 camera intrinsics matrix
        ground_truth_paths : list[str]
            paths to ground truth depth maps
        load_image_triplets : bool
            if set, then load 3 images
    '''

    def __init__(self,
                 image_paths,
                 sparse_depth_paths,
                 intrinsics_paths,
                 ground_truth_paths=None,
                 load_image_triplets=False):

        self.image_paths = image_paths
        self.sparse_depth_paths = sparse_depth_paths
        self.intrinsics_paths = intrinsics_paths

        self.is_available_ground_truth = \
           ground_truth_paths is not None and all([x is not None for x in ground_truth_paths])

        if self.is_available_ground_truth:
            self.ground_truth_paths = ground_truth_paths

        self.load_image_triplets = load_image_triplets

        self.data_format = 'CHW'

    def __getitem__(self, index):

        # Load image
        if self.load_image_triplets:
            _, image, _ = load_image_triplet(
                self.image_paths[index],
                normalize=False,
                data_format=self.data_format)
        else:
            image = data_utils.load_image(
                self.image_paths[index],
                normalize=False,
                data_format=self.data_format)

        # Load depth
        sparse_depth = data_utils.load_depth(
            self.sparse_depth_paths[index],
            data_format=self.data_format)

        # Load camera intrinsics
        intrinsics = np.load(self.intrinsics_paths[index]).astype(np.float32)

        inputs = [
            image,
            sparse_depth,
            intrinsics
        ]

        # Load ground truth if available
        if self.is_available_ground_truth:
            ground_truth = data_utils.load_depth(
                self.ground_truth_paths[index],
                data_format=self.data_format)
            inputs.append(ground_truth)

        # Convert to float32
        inputs = [
            T.astype(np.float32)
            for T in inputs
        ]

        # Return image, sparse_depth, intrinsics, and if available, ground_truth
        return inputs

    def __len__(self):
        return len(self.image_paths)
