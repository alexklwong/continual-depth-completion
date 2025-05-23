'''
Authors:
Tian Yu Liu <tianyu@cs.ucla.edu>
Parth Agrawal <parthagrawal24@ucla.edu>
Allison Chen <allisonchen2@ucla.edu>
Alex Wong <alex.wong@yale.edu>

If you use this code, please cite the following paper:
T.Y. Liu, P. Agrawal, A. Chen, B.W. Hong, and A. Wong. Monitored Distillation for Positive Congruent Depth Completion.
https://arxiv.org/abs/2203.16034

@inproceedings{liu2022monitored,
  title={Monitored distillation for positive congruent depth completion},
  author={Liu, Tian Yu and Agrawal, Parth and Chen, Allison and Hong, Byung-Woo and Wong, Alex},
  booktitle={European Conference on Computer Vision},
  year={2022},
  organization={Springer}
}
'''

import warnings
warnings.filterwarnings("ignore")

import os, sys, glob, cv2, argparse, random
import multiprocessing as mp
import numpy as np
sys.path.insert(0, './')
import utils.src.data_utils as data_utils
from sklearn.cluster import MiniBatchKMeans


N_CLUSTER = 1500
O_HEIGHT = 480
O_WIDTH = 640
N_HEIGHT = 416
N_WIDTH = 576
MIN_POINTS = 1100
TEMPORAL_WINDOW = 21
RANDOM_SEED = 1

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

parser = argparse.ArgumentParser()

parser.add_argument('--sparse_depth_distro_type', type=str, default='corner')
parser.add_argument('--n_points',                 type=int, default=N_CLUSTER)
parser.add_argument('--min_points',               type=int, default=MIN_POINTS)
parser.add_argument('--temporal_window',          type=int, default=TEMPORAL_WINDOW)
parser.add_argument('--n_height',                 type=int, default=N_HEIGHT)
parser.add_argument('--n_width',                  type=int, default=N_WIDTH)
parser.add_argument('--fast_forward',             action='store_true')
parser.add_argument('--subset',                   action='store_true')

args = parser.parse_args()


NYU_ROOT_DIRPATH = \
    os.path.join('data', 'nyu_v2')
NYU_OUTPUT_DIRPATH = \
    os.path.join('data', 'nyu_v2_derived-{}'.format(args.sparse_depth_distro_type))

NYU_TEST_IMAGE_SPLIT_FILEPATH = \
    os.path.join('setup', 'nyu_v2', 'nyu_v2_test_image.txt')
NYU_TEST_DEPTH_SPLIT_FILEPATH = \
    os.path.join('setup', 'nyu_v2', 'nyu_v2_test_depth.txt')

TRAIN_REF_DIRPATH = os.path.join('training', 'nyu_v2')
VAL_REF_DIRPATH = os.path.join('validation', 'nyu_v2')
TEST_REF_DIRPATH = os.path.join('testing', 'nyu_v2')

TRAIN_SUPERVISED_REF_DIRPATH = os.path.join(TRAIN_REF_DIRPATH, 'supervised')
TRAIN_UNSUPERVISED_REF_DIRPATH = os.path.join(TRAIN_REF_DIRPATH, 'unsupervised')

# Define output paths for supervised training
TRAIN_SUPERVISED_IMAGE_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'nyu_v2_train_image_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_SUPERVISED_SPARSE_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'nyu_v2_train_sparse_depth_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_SUPERVISED_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'nyu_v2_train_ground_truth_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_SUPERVISED_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_SUPERVISED_REF_DIRPATH, 'nyu_v2_train_intrinsics_{}.txt'.format(args.sparse_depth_distro_type))

# Define output paths for unsupervised training
TRAIN_UNSUPERVISED_IMAGE_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'nyu_v2_train_image_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_UNSUPERVISED_SPARSE_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'nyu_v2_train_sparse_depth_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_UNSUPERVISED_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'nyu_v2_train_ground_truth_{}.txt'.format(args.sparse_depth_distro_type))
TRAIN_UNSUPERVISED_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(TRAIN_UNSUPERVISED_REF_DIRPATH, 'nyu_v2_train_intrinsics_{}.txt'.format(args.sparse_depth_distro_type))

# Validation file paths
VAL_IMAGE_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_image_{}.txt'.format(args.sparse_depth_distro_type))
VAL_SPARSE_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_sparse_depth_{}.txt'.format(args.sparse_depth_distro_type))
VAL_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_ground_truth_{}.txt'.format(args.sparse_depth_distro_type))
VAL_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(VAL_REF_DIRPATH, 'nyu_v2_val_intrinsics_{}.txt'.format(args.sparse_depth_distro_type))

# Test file paths
TEST_IMAGE_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_image_{}.txt'.format(args.sparse_depth_distro_type))
TEST_SPARSE_DEPTH_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_sparse_depth_{}.txt'.format(args.sparse_depth_distro_type))
TEST_GROUND_TRUTH_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_ground_truth_{}.txt'.format(args.sparse_depth_distro_type))
TEST_INTRINSICS_OUTPUT_FILEPATH = \
    os.path.join(TEST_REF_DIRPATH, 'nyu_v2_test_intrinsics_{}.txt'.format(args.sparse_depth_distro_type))


def process_frame(inputs):
    '''
    Processes a single frame

    Arg(s):
        inputs : tuple
            image path at time t=0,
            image path at time t=1,
            image path at time t=-1,
            ground truth path at time t=0
    Returns:
        str : output image path at time t=0
        str : output concatenated image path at time t=0
        str : output sparse depth path at time t=0
        str : output ground truth path at time t=0
    '''

    image0_path, image1_path, image2_path, ground_truth_path, save_image_triplet = inputs

    # Load image (for corner detection) to generate valid map
    image0 = cv2.imread(image0_path)
    image0 = np.float32(cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY))

    # Load dense depth
    ground_truth = data_utils.load_depth(ground_truth_path)

    # Crop away white borders
    if args.n_height != O_HEIGHT or args.n_width != O_WIDTH:
        d_height = O_HEIGHT - args.n_height
        d_width = O_WIDTH - args.n_width

        y_start = d_height // 2
        x_start = d_width // 2
        y_end = y_start + args.n_height
        x_end = x_start + args.n_width

        image0 = image0[y_start:y_end, x_start:x_end]
        ground_truth = ground_truth[y_start:y_end, x_start:x_end]

    if args.sparse_depth_distro_type == 'corner':
        N_INIT_CORNER = 30000

        # Run Harris corner detector
        corners = cv2.cornerHarris(image0, blockSize=5, ksize=3, k=0.04)

        # Remove the corners that are located on invalid depth locations
        corners = corners * np.where(ground_truth > 0.0, 1.0, 0.0)

        # Vectorize corner map to 1D vector and select N_INIT_CORNER corner locations
        corners = corners.ravel()
        corner_locations = np.argsort(corners)[0:N_INIT_CORNER]

        # Get locations of corners as indices as (x, y)
        corner_locations = np.unravel_index(
            corner_locations,
            (image0.shape[0], image0.shape[1]))

        # Convert to (y, x) convention
        corner_locations = \
            np.transpose(np.array([corner_locations[0], corner_locations[1]]))

        # Cluster them into n_points (number of output points)
        kmeans = MiniBatchKMeans(
            n_clusters=args.n_points,
            max_iter=2,
            n_init=1,
            init_size=None,
            random_state=RANDOM_SEED,
            reassignment_ratio=1e-11)
        kmeans.fit(corner_locations)

        # Use k-Means means as corners
        selected_indices = kmeans.cluster_centers_.astype(np.uint16)

    elif args.sparse_depth_distro_type == 'uniform':
        indices = \
            np.array([[h, w] for h in range(args.n_height) for w in range(args.n_width)])

        # Randomly select n_points number of points
        selected_indices = \
            np.random.permutation(range(args.n_height * args.n_width))[0:args.n_points]
        selected_indices = indices[selected_indices]

    else:
        raise ValueError('Unsupported sparse depth distribution type: {}'.format(
            args.sparse_depth_distro_type))

    # Convert the indices into validity map
    validity_map = np.zeros_like(image0).astype(np.int16)
    validity_map[selected_indices[:, 0], selected_indices[:, 1]] = 1.0

    # Build validity map from selected points, keep only ones greater than 0
    validity_map = np.where(validity_map * ground_truth > 0.0, 1.0, 0.0)

    # Get sparse depth based on validity map
    sparse_depth = validity_map * ground_truth

    # Shape check
    error_flag = False

    if np.squeeze(sparse_depth).shape != (args.n_height, args.n_width):
        error_flag = True
        print('FAILED: np.squeeze(sparse_depth).shape != ({}, {})'.format(args.n_height, args.n_width))

    # Depth value check
    if np.min(ground_truth) < 0.0 or np.max(ground_truth) > 256.0:
        error_flag = True
        print('FAILED: np.min(ground_truth) < 0.0 or np.max(ground_truth) > 256.0')

    if np.sum(np.where(validity_map > 0.0, 1.0, 0.0)) < args.min_points:
        error_flag = True
        print('FAILED: np.sum(np.where(validity_map > 0.0, 1.0, 0.0)) < MIN_POINTS', np.sum(np.where(validity_map > 0.0, 1.0, 0.0)))

    if np.sum(np.where(ground_truth > 0.0, 1.0, 0.0)) < args.min_points:
        error_flag = True
        print('FAILED: np.sum(np.where(ground_truth > 0.0, 1.0, 0.0)) < MIN_POINTS')

    # NaN check
    if np.any(np.isnan(sparse_depth)):
        error_flag = True
        print('FAILED: np.any(np.isnan(sparse_depth))')

    if not error_flag:

        image0 = cv2.imread(image0_path)

        if args.n_height != O_HEIGHT or args.n_width != O_WIDTH:
            image0 = image0[y_start:y_end, x_start:x_end, :]

        if save_image_triplet:
            # Read images and concatenate together
            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)

            if args.n_height != O_HEIGHT or args.n_width != O_WIDTH:
                image1 = image1[y_start:y_end, x_start:x_end, :]
                image2 = image2[y_start:y_end, x_start:x_end, :]

            imagec = np.concatenate([image1, image0, image2], axis=1)
        else:
            imagec = None

        # Example: nyu/training/depths/raw_data/bedroom_0001/r-1294886360.208451-2996770081.png
        image_output_path = image0_path \
            .replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH)

        if imagec is None:
            images_output_path = None
        else:
            images_output_path = image0_path \
                .replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH) \
                .replace('images', 'image_triplet')

        sparse_depth_output_path = ground_truth_path \
            .replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH) \
            .replace('depth', 'sparse_depth')
        ground_truth_output_path = ground_truth_path \
            .replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH) \
            .replace('depth', 'ground_truth')

        image_output_dirpath = os.path.dirname(image_output_path)

        if images_output_path is not None:
            images_output_dirpath = os.path.dirname(images_output_path)
        else:
            images_output_dirpath = None

        sparse_depth_output_dirpath = os.path.dirname(sparse_depth_output_path)
        ground_truth_output_dirpath = os.path.dirname(ground_truth_output_path)

        # Create output directories
        output_dirpaths = [
            image_output_dirpath,
            images_output_dirpath,
            sparse_depth_output_dirpath,
            ground_truth_output_dirpath
        ]

        for dirpath in output_dirpaths:
            if dirpath is not None and not os.path.exists(dirpath):
                os.makedirs(dirpath, exist_ok=True)

        # Write to file
        cv2.imwrite(image_output_path, image0)
        if imagec is not None:
            cv2.imwrite(images_output_path, imagec)
        data_utils.save_depth(sparse_depth, sparse_depth_output_path)
        data_utils.save_depth(ground_truth, ground_truth_output_path)
    else:
        print('Found error in {}'.format(ground_truth_path))
        image_output_path = 'error'
        images_output_path = 'error'
        sparse_depth_output_path = 'error'
        ground_truth_output_path = 'error'

    return (image_output_path,
            images_output_path,
            sparse_depth_output_path,
            ground_truth_output_path)

def filter_sequence(seq):
    keep_sequence = \
        '_0000/' in seq or \
        '_0001/' in seq or \
        '_0002/' in seq or \
        '_0003/' in seq or \
        '_0004/' in seq or \
        '_0005/' in seq or \
        '_0006/' in seq or \
        '_0007/' in seq or \
        '_0008/' in seq or \
        '_0009/' in seq or \
        '_0010/' in seq or \
        '_0011/' in seq or \
        '_0012/' in seq or \
        '_0013/' in seq or \
        '_0014/' in seq or \
        '_0014/' in seq or \
        '_0001a/' in seq or \
        '_0001b/' in seq or \
        '_0001c/' in seq or \
        '_0001d/' in seq or \
        '_0001e/' in seq or \
        '_0001f/' in seq or \
        '_0001g/' in seq or \
        '_0002a/' in seq or \
        '_0002b/' in seq or \
        '_0002c/' in seq or \
        '_0002d/' in seq

    return keep_sequence

def filter_paths(paths):
    paths_ = []

    for path in paths:
        if filter_sequence(path):
            paths_.append(path)

    return paths_


# Create output directories first
dirpaths = [
    NYU_OUTPUT_DIRPATH,
    TRAIN_REF_DIRPATH,
    VAL_REF_DIRPATH,
    TEST_REF_DIRPATH,
    TRAIN_SUPERVISED_REF_DIRPATH,
    TRAIN_UNSUPERVISED_REF_DIRPATH
]

for dirpath in dirpaths:
    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)


'''
Setup intrinsics (values are copied from camera_params.m)
'''
fx_rgb = 518.85790117450188
fy_rgb = 519.46961112127485
cx_rgb = 325.58244941119034
cy_rgb = 253.73616633400465
intrinsic_matrix = np.array([
    [fx_rgb,   0.0,    cx_rgb],
    [0.0,      fy_rgb, cy_rgb],
    [0.0,      0.0,    1.0   ]], dtype=np.float32)


if args.n_height != O_HEIGHT or args.n_width != O_WIDTH:

    d_height = O_HEIGHT - args.n_height
    d_width = O_WIDTH - args.n_width

    y_start = d_height // 2
    x_start = d_width // 2

    intrinsic_matrix = intrinsic_matrix + [[0.0, 0.0, -x_start],
                                           [0.0, 0.0, -y_start],
                                           [0.0, 0.0, 0.0     ]]

intrinsics_output_path = os.path.join(NYU_OUTPUT_DIRPATH, 'intrinsics.npy')
np.save(intrinsics_output_path, intrinsic_matrix)


'''
Process training paths
'''
train_supervised_image_output_paths = []
train_supervised_sparse_depth_output_paths = []
train_supervised_ground_truth_output_paths = []
train_supervised_intrinsics_output_paths = [intrinsics_output_path]

train_unsupervised_images_output_paths = []
train_unsupervised_sparse_depth_output_paths = []
train_unsupervised_ground_truth_output_paths = []
train_unsupervised_intrinsics_output_paths = [intrinsics_output_path]

train_image_sequences = sorted(glob.glob(
    os.path.join(NYU_ROOT_DIRPATH, 'training', 'images', 'raw_data', '*/')))
train_depth_sequences = sorted(glob.glob(
    os.path.join(NYU_ROOT_DIRPATH, 'training', 'depths', 'raw_data', '*/')))

# Check option for use subset to see if we use subset of the data
if args.subset:
    train_image_sequences = filter_paths(train_image_sequences)
    train_depth_sequences = filter_paths(train_depth_sequences)

w = int(args.temporal_window // 2)

for image_sequence, depth_sequence in zip(train_image_sequences, train_depth_sequences):

    # Fetch image and dense depth from sequence directory
    image_paths = \
        sorted(glob.glob(os.path.join(image_sequence, '*.png')))
    ground_truth_paths = \
        sorted(glob.glob(os.path.join(depth_sequence, '*.png')))

    n_sample = len(image_paths)

    for image_path, ground_truth_path in zip(image_paths, ground_truth_paths):
        assert os.path.join(*(image_path.split(os.sep)[-3:])) == os.path.join(*(image_path.split(os.sep)[-3:]))

    image_output_dirpath = os.path.dirname(
        image_paths[0].replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH))
    images_output_dirpath = os.path.dirname(
        image_paths[0].replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH).replace('images', 'image_triplet'))
    sparse_depth_output_dirpath = os.path.dirname(
        ground_truth_paths[0].replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH).replace('depth', 'sparse_depth'))
    ground_truth_output_dirpath = os.path.dirname(
        ground_truth_paths[0].replace(NYU_ROOT_DIRPATH, NYU_OUTPUT_DIRPATH).replace('depth', 'ground_truth'))

    image_output_paths = sorted(glob.glob(os.path.join(image_output_dirpath, '*.png')))
    images_output_paths = sorted(glob.glob(os.path.join(images_output_dirpath, '*.png')))
    sparse_depth_output_paths = sorted(glob.glob(os.path.join(sparse_depth_output_dirpath, '*.png')))
    ground_truth_output_paths = sorted(glob.glob(os.path.join(ground_truth_output_dirpath, '*.png')))

    is_exists_output_dirpaths = \
        os.path.exists(image_output_dirpath) and \
        os.path.exists(images_output_dirpath) and \
        os.path.exists(sparse_depth_output_dirpath) and \
        os.path.exists(ground_truth_output_dirpath) and \
        len(image_output_paths) == len(sparse_depth_output_paths) and \
        len(image_output_paths) == len(ground_truth_output_paths)

    if args.fast_forward and is_exists_output_dirpaths:

        print('Found {} samples for supervised and {} samples for unsupervised training in: {}'.format(
            len(image_output_paths), len(images_output_paths), image_sequence))

        # Append all supervised training paths
        train_supervised_image_output_paths.extend(image_output_paths)
        train_supervised_sparse_depth_output_paths.extend(sparse_depth_output_paths)
        train_supervised_ground_truth_output_paths.extend(ground_truth_output_paths)

        images_output_filenames = [
            os.path.splitext(os.path.basename(path))[0]
            for path in images_output_paths
        ]

        for depth_idx, sparse_depth_output_path in enumerate(sparse_depth_output_paths):
            sparse_depth_output_filename = os.path.splitext(os.path.basename(sparse_depth_output_path))[0]

            try:
                images_idx = images_output_filenames.index(sparse_depth_output_filename)
            except ValueError:
                continue

            ground_truth_output_path = ground_truth_output_paths[depth_idx]
            images_output_path = images_output_paths[images_idx]

            train_unsupervised_images_output_paths.append(images_output_path)
            train_unsupervised_sparse_depth_output_paths.append(sparse_depth_output_path)
            train_unsupervised_ground_truth_output_paths.append(ground_truth_output_path)
    else:
        pool_input = []
        for idx in range(n_sample):
            if idx in range(w, n_sample - w):
                pool_input.append((image_paths[idx], image_paths[idx-w], image_paths[idx+w], ground_truth_paths[idx], True))
            else:
                pool_input.append((image_paths[idx], image_paths[idx], image_paths[idx], ground_truth_paths[idx], False))

        print('Processing {} samples in: {}'.format(n_sample - 2 * w + 1, image_sequence))

        with mp.Pool() as pool:
            pool_results = pool.map(process_frame, pool_input)

            for result in pool_results:
                image_output_path, \
                    images_output_path, \
                    sparse_depth_output_path, \
                    ground_truth_output_path = result

                error_encountered = \
                    image_output_path == 'error' or \
                    images_output_path == 'error' or \
                    sparse_depth_output_path == 'error' or \
                    ground_truth_output_path == 'error'

                if error_encountered:
                    continue

                # Collect filepaths
                if images_output_path is not None:
                    train_unsupervised_images_output_paths.append(images_output_path)
                    train_unsupervised_sparse_depth_output_paths.append(sparse_depth_output_path)
                    train_unsupervised_ground_truth_output_paths.append(ground_truth_output_path)

                train_supervised_image_output_paths.append(image_output_path)
                train_supervised_sparse_depth_output_paths.append(sparse_depth_output_path)
                train_supervised_ground_truth_output_paths.append(ground_truth_output_path)

train_supervised_intrinsics_output_paths = \
    train_supervised_intrinsics_output_paths * len(train_supervised_image_output_paths)
train_unsupervised_intrinsics_output_paths = \
    train_unsupervised_intrinsics_output_paths * len(train_unsupervised_images_output_paths)

# Storing paths for supervised training
print('Storing {} training supervised image file paths into: {}'.format(
    len(train_supervised_image_output_paths), TRAIN_SUPERVISED_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_SUPERVISED_IMAGE_OUTPUT_FILEPATH, train_supervised_image_output_paths)

print('Storing {} training supervised sparse depth file paths into: {}'.format(
    len(train_supervised_sparse_depth_output_paths), TRAIN_SUPERVISED_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_SUPERVISED_SPARSE_DEPTH_OUTPUT_FILEPATH, train_supervised_sparse_depth_output_paths)

print('Storing {} training supervised ground truth file paths into: {}'.format(
    len(train_supervised_ground_truth_output_paths), TRAIN_SUPERVISED_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_SUPERVISED_GROUND_TRUTH_OUTPUT_FILEPATH, train_supervised_ground_truth_output_paths)

print('Storing {} training supervised intrinsics file paths into: {}'.format(
    len(train_supervised_intrinsics_output_paths), TRAIN_SUPERVISED_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_SUPERVISED_INTRINSICS_OUTPUT_FILEPATH, train_supervised_intrinsics_output_paths)

# Storing paths for unsupervised training
print('Storing {} training image file paths into: {}'.format(
    len(train_unsupervised_images_output_paths), TRAIN_UNSUPERVISED_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_UNSUPERVISED_IMAGE_OUTPUT_FILEPATH, train_unsupervised_images_output_paths)

print('Storing {} training sparse depth file paths into: {}'.format(
    len(train_unsupervised_sparse_depth_output_paths), TRAIN_UNSUPERVISED_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_UNSUPERVISED_SPARSE_DEPTH_OUTPUT_FILEPATH, train_unsupervised_sparse_depth_output_paths)

print('Storing {} training ground truth file paths into: {}'.format(
    len(train_unsupervised_ground_truth_output_paths), TRAIN_UNSUPERVISED_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_UNSUPERVISED_GROUND_TRUTH_OUTPUT_FILEPATH, train_unsupervised_ground_truth_output_paths)

print('Storing {} training intrinsics file paths into: {}'.format(
    len(train_unsupervised_intrinsics_output_paths), TRAIN_UNSUPERVISED_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(TRAIN_UNSUPERVISED_INTRINSICS_OUTPUT_FILEPATH, train_unsupervised_intrinsics_output_paths)

'''
Process validation and testing paths
'''
test_image_split_paths = data_utils.read_paths(NYU_TEST_IMAGE_SPLIT_FILEPATH)

val_image_output_paths = []
val_sparse_depth_output_paths = []
val_ground_truth_output_paths = []
val_intrinsics_output_paths = [intrinsics_output_path]

test_image_output_paths = []
test_sparse_depth_output_paths = []
test_ground_truth_output_paths = []
test_intrinsics_output_paths = [intrinsics_output_path]

test_image_paths = sorted(glob.glob(
    os.path.join(NYU_ROOT_DIRPATH, 'testing', 'images', '*.png')))
test_ground_truth_paths = sorted(glob.glob(
    os.path.join(NYU_ROOT_DIRPATH, 'testing', 'depths', '*.png')))

n_sample = len(test_image_paths)

for image_path, ground_truth_path in zip(test_image_paths, test_ground_truth_paths):
    assert os.path.join(*(image_path.split(os.sep)[-3:])) == os.path.join(*(image_path.split(os.sep)[-3:]))

pool_input = [
    (test_image_paths[idx], test_image_paths[idx], test_image_paths[idx], test_ground_truth_paths[idx], False)
    for idx in range(n_sample)
]

print('Processing {} samples for validation and testing'.format(n_sample))

with mp.Pool() as pool:
    pool_results = pool.map(process_frame, pool_input)

    for result in pool_results:
        image_output_path, \
            _, \
            sparse_depth_output_path, \
            ground_truth_output_path = result

        error_encountered = \
            image_output_path == 'error' or \
            sparse_depth_output_path == 'error' or \
            ground_truth_output_path == 'error'

        if error_encountered:
            continue

        test_split = False
        for test_image_path in test_image_split_paths:
            if test_image_path in image_output_path:
                test_split = True

        if test_split:
            # Collect test filepaths
            test_image_output_paths.append(image_output_path)
            test_sparse_depth_output_paths.append(sparse_depth_output_path)
            test_ground_truth_output_paths.append(ground_truth_output_path)
        else:
            # Collect validation filepaths
            val_image_output_paths.append(image_output_path)
            val_sparse_depth_output_paths.append(sparse_depth_output_path)
            val_ground_truth_output_paths.append(ground_truth_output_path)

val_intrinsics_output_paths = val_intrinsics_output_paths * len(val_image_output_paths)
test_intrinsics_output_paths = test_intrinsics_output_paths * len(test_image_output_paths)

'''
Write validation output paths
'''
print('Storing {} validation image file paths into: {}'.format(
    len(val_image_output_paths), VAL_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_IMAGE_OUTPUT_FILEPATH, val_image_output_paths)

print('Storing {} validation sparse depth file paths into: {}'.format(
    len(val_sparse_depth_output_paths), VAL_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_SPARSE_DEPTH_OUTPUT_FILEPATH, val_sparse_depth_output_paths)

print('Storing {} validation dense depth file paths into: {}'.format(
    len(val_ground_truth_output_paths), VAL_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_GROUND_TRUTH_OUTPUT_FILEPATH, val_ground_truth_output_paths)

print('Storing {} validation intrinsics file paths into: {}'.format(
    len(val_intrinsics_output_paths), VAL_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(VAL_INTRINSICS_OUTPUT_FILEPATH, val_intrinsics_output_paths)


'''
Write testing output paths
'''
print('Storing {} testing image file paths into: {}'.format(
    len(test_image_output_paths), TEST_IMAGE_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_IMAGE_OUTPUT_FILEPATH, test_image_output_paths)

print('Storing {} testing sparse depth file paths into: {}'.format(
    len(test_sparse_depth_output_paths), TEST_SPARSE_DEPTH_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_SPARSE_DEPTH_OUTPUT_FILEPATH, test_sparse_depth_output_paths)

print('Storing {} testing dense depth file paths into: {}'.format(
    len(test_ground_truth_output_paths), TEST_GROUND_TRUTH_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_GROUND_TRUTH_OUTPUT_FILEPATH, test_ground_truth_output_paths)

print('Storing {} testing intrinsics file paths into: {}'.format(
    len(test_intrinsics_output_paths), TEST_INTRINSICS_OUTPUT_FILEPATH))
data_utils.write_paths(TEST_INTRINSICS_OUTPUT_FILEPATH, test_intrinsics_output_paths)
