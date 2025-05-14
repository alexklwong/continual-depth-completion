import os, sys, glob, subprocess, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import gridspec
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join('utils', 'src'))
import data_utils


def config_plt():
    plt.box(False)
    plt.axis('off')


parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--output_root_dirpath',     type=str, required=True)
parser.add_argument('--visualization_dirpath',   type=str, required=True)
parser.add_argument('--image_ext',               type=str, default='.png')
parser.add_argument('--depth_ext',               type=str, default='.png')
parser.add_argument('--task',                    type=str, default='depth_prediction')

# Visualization
parser.add_argument('--median_scale_depth',      action='store_true')
parser.add_argument('--visualize_error',         action='store_true')
parser.add_argument('--cmap',                    type=str, default='jet')
parser.add_argument('--vmin',                    type=float, default=0.10,
    help='For VOID, use 0.1, for KITTI use 1.0, for NYUv2 use 0.1')
parser.add_argument('--vmax',                    type=float, default=100.0,
    help='For VOID, use 6.0, for KITTI use 70.0, for NYUv2 use 8.0')
parser.add_argument('--max_error_percent',       type=float, default=0.20)


args = parser.parse_args()


if not os.path.exists(args.visualization_dirpath):
    os.makedirs(args.visualization_dirpath)

assert args.task in ['depth_completion', 'depth_prediction']

'''
Fetch file paths from input directories
'''
image_dirpath = os.path.join(args.output_root_dirpath, 'image')
sparse_depth_dirpath = os.path.join(args.output_root_dirpath, 'sparse_depth')
output_depth_dirpath = os.path.join(args.output_root_dirpath, 'output_depth')

assert os.path.exists(image_dirpath)

if args.task == 'depth_completion':
    assert os.path.exists(sparse_depth_dirpath)

assert os.path.exists(output_depth_dirpath)

image_paths = \
    sorted(glob.glob(os.path.join(image_dirpath, '*' + args.image_ext)))
sparse_depth_paths = \
    sorted(glob.glob(os.path.join(sparse_depth_dirpath, '*' + args.depth_ext)))
output_depth_paths = \
    sorted(glob.glob(os.path.join(output_depth_dirpath, '*' + args.depth_ext)))

n_sample = len(image_paths)

if args.task == "depth_completion":
    assert n_sample == len(sparse_depth_paths)
assert n_sample == len(output_depth_paths)

if args.visualize_error or args.median_scale_depth:

    ground_truth_dirpath = os.path.join(args.output_root_dirpath, 'ground_truth')

    assert os.path.exists(ground_truth_dirpath)

    ground_truth_paths = \
        sorted(glob.glob(os.path.join(ground_truth_dirpath, '*' + args.depth_ext)))

    assert n_sample == len(ground_truth_paths)


cmap = cm.get_cmap(name=args.cmap)
cmap.set_under(color='black')

'''
Process image, sparse depth and output depth (and groundtruth)
'''
for idx in range(n_sample):

    sys.stdout.write(
        'Processing {}/{} samples...\r'.format(idx + 1, n_sample))
    sys.stdout.flush()

    image_path = image_paths[idx]

    if args.task == 'depth_completion':
        sparse_depth_path = sparse_depth_paths[idx]

    output_depth_path = output_depth_paths[idx]

    # Set up output path
    filename = os.path.basename(image_path)
    visualization_path = os.path.join(args.visualization_dirpath, filename)

    # Load image, sparse depth and output depth (and groundtruth)
    image = Image.open(image_paths[idx]).convert('RGB')
    image = np.asarray(image, dtype=np.uint8)
    if args.task == 'depth_completion':
        sparse_depth = data_utils.load_depth(sparse_depth_path)

    output_depth = data_utils.load_depth(output_depth_path)

    if args.visualize_error or args.median_scale_depth:
        ground_truth_path = ground_truth_paths[idx]
        ground_truth = data_utils.load_depth(ground_truth_path)

    if args.median_scale_depth:
        mask = np.where(ground_truth > 0)
        output_depth = output_depth * np.median(ground_truth[mask]) / np.median(output_depth[mask])

    # Set number of rows in output visualization
    n_row = 3 if args.task == 'depth_completion' else 2

    if args.visualize_error:
        n_row = 5 if args.task == 'depth_completion' else 4

    # Create figure and grid
    plt.figure(figsize=(75, 25), dpi=40, facecolor='w', edgecolor='k')

    gs = gridspec.GridSpec(n_row, 1, wspace=0.0, hspace=0.0)

    row_idx = 0

    # Plot image, sparse depth, output depth
    ax = plt.subplot(gs[row_idx, 0])
    config_plt()
    ax.imshow(image)
    row_idx = row_idx + 1

    if args.task == 'depth_completion':
        ax = plt.subplot(gs[row_idx, 0])
        config_plt()
        ax.imshow(sparse_depth, vmin=args.vmin, vmax=args.vmax, cmap=cmap)
        row_idx = row_idx + 1

    ax = plt.subplot(gs[row_idx, 0])
    config_plt()
    ax.imshow(output_depth, vmin=args.vmin, vmax=args.vmax, cmap=cmap)
    row_idx = row_idx + 1

    # Plot groundtruth if available
    if args.visualize_error:
        error_depth = np.where(
            ground_truth > 0,
            np.abs(output_depth - ground_truth) / ground_truth,
            0.0)

        ax = plt.subplot(gs[row_idx, 0])
        config_plt()
        ax.imshow(error_depth, vmin=0.00, vmax=args.max_error_percent, cmap='hot')
        row_idx = row_idx + 1

        ax = plt.subplot(gs[row_idx, 0])
        config_plt()
        ax.imshow(ground_truth, vmin=args.vmin, vmax=args.vmax, cmap=cmap)
        row_idx = row_idx + 1

    plt.savefig(visualization_path)
    plt.close()
    subprocess.call(["convert", "-trim", visualization_path, visualization_path])
