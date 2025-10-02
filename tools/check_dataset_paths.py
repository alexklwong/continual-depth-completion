import argparse, os, sys, tqdm
import numpy as np

sys.path.insert(0, os.path.join('utils', 'src'))
import data_utils


parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--image_paths_filepath',          type=str, required=True)
parser.add_argument('--sparse_depth_paths_filepath',   type=str, required=True)
parser.add_argument('--intrinsics_paths_filepath',     type=str, default=True)
parser.add_argument('--ground_truth_paths_filepath',   type=str, default=True)

args = parser.parse_args()

assert os.path.exists(args.image_paths_filepath)
assert os.path.exists(args.sparse_depth_paths_filepath)
assert os.path.exists(args.intrinsics_paths_filepath)
assert os.path.exists(args.ground_truth_paths_filepath)

image_paths = data_utils.read_paths(args.image_paths_filepath)
sparse_depth_paths = data_utils.read_paths(args.sparse_depth_paths_filepath)
intrinsics_paths = data_utils.read_paths(args.intrinsics_paths_filepath)
ground_truth_paths = data_utils.read_paths(args.ground_truth_paths_filepath)

data_paths = [
    image_paths,
    sparse_depth_paths,
    intrinsics_paths,
    ground_truth_paths
]

n_sample = len(image_paths)

for paths in data_paths:
    assert len(paths) == n_sample

data_paths_loader = tqdm.tqdm(
    zip(data_paths),
    total=n_sample)

for paths in data_paths_loader:

    for path in paths:
        assert os.path.exists(path), \
            'File does not exist {}'.format(path)

    image_path, \
        sparse_depth_path, \
        intrinsics_path, \
        ground_truth_path = paths

    try:
        data_utils.load_image(image_path, normalize=True, data_format='HWC')
    except Exception:
        print('Error loading {}'.format(image_path))

    try:
        data_utils.load_depth(sparse_depth_path, multiplier=256.0, data_format='HW')
    except Exception:
        print('Error loading {}'.format(sparse_depth_path))

    try:
        np.load(intrinsics_path)
    except Exception:
        print('Error loading {}'.format(intrinsics_path))

    try:
        data_utils.load_depth(ground_truth_path, multiplier=256.0, data_format='HW')
    except Exception:
        print('Error loading {}'.format(ground_truth_path))
