import cv2
import numpy as np
from skimage.restoration import inpaint
from scipy.interpolate import LinearNDInterpolator
from PIL import Image


def read_paths(filepath):
    '''
    Reads a newline delimited file containing paths

    Arg(s):
        filepath : str
            path to file to be read
    Return:
        list[str] : list of paths
    '''

    path_list = []
    with open(filepath) as f:
        while True:
            path = f.readline().rstrip('\n')

            # If there was nothing to read
            if path == '':
                break

            path_list.append(path.encode('utf-8').decode())

    return path_list

def write_paths(filepath, paths):
    '''
    Stores line delimited paths into file

    Arg(s):
        filepath : str
            path to file to save paths
        paths : list[str]
            paths to write into file
    '''

    with open(filepath, 'w') as o:
        for idx in range(len(paths)):
            o.write(paths[idx] + '\n')

def load_image(path, normalize=True, data_format='HWC'):
    '''
    Loads an RGB image

    Arg(s):
        path : str
            path to RGB image
        normalize : bool
            if set, then normalize image between [0, 1]
        data_format : str
            'CHW', or 'HWC'
    Returns:
        numpy[float32] : H x W x C or C x H x W image
    '''

    # Load image
    image = Image.open(path).convert('RGB')

    # Convert to numpy
    image = np.asarray(image, np.float32)

    if data_format == 'HWC':
        pass
    elif data_format == 'CHW':
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    # Normalize
    image = image / 255.0 if normalize else image

    return image


def save_image(image, path, normalized=True, data_type='color', data_format='HWC'):
    '''
    Saves an RGB, gray, or label (with validity map in alpha) image to 8-bit PNG

    Arg(s):
        image : numpy
            RGB, gray, or label (with validity map in alpha) image
        path : str
            path to store image
        normalized : bool
            if set, then treat image as normalized range [0, 1] and multiply by 255
        data_type : str
            color or label
        data_format : str
            data format of input 'CHW', or 'HWC'
    '''

    # Put image between [0, 255] range, if it was normalized to [0, 1]
    image = 255.0 * image if normalized else image
    image = np.uint8(image)

    if image.ndim == 2:
        # Append channel dimension
        image = np.expand_dims(image, axis=-1)

    # Put image into H x W x C format
    if data_format == 'HWC':
        pass
    elif data_format == 'CHW':
        image = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    if data_type == 'color':
        image = Image.fromarray(np.uint8(image))
    elif data_type == 'gray':
        image = np.squeeze(image)
        image = Image.fromarray(np.uint8(image), mode='L')
    elif data_type == 'label':
        # Add alpha channel
        if image.ndim == 2 or (image.ndim == 3 and image.shape[-1] == 1):
            image = np.concatenate([image, 255.0 * np.ones_like(image)], axis=-1)
        elif image.ndim == 3 and image.shape[-1] > 2:
            raise ValueError('ERROR: too many channels in label')

        image = Image.fromarray(np.uint8(image), mode='LA')
    else:
        raise ValueError('Unsupported data type: {}'.format(data_type))

    image.save(path)

def load_depth_with_validity_map(path, multiplier=256.0, data_format='HW'):
    '''
    Loads a depth map and validity map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
        numpy[float32] : binary validity map for available depth measurement locations
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0
    v = z.astype(np.float32)
    v[z > 0] = 1.0

    # Expand dimensions based on output format
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
        v = np.expand_dims(v, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
        v = np.expand_dims(v, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z, v

def load_depth(path, multiplier=256.0, data_format='HW'):
    '''
    Loads a depth map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : depth map
    '''

    # Loads depth map from 16-bit PNG file
    z = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    z = z / multiplier
    z[z <= 0] = 0.0

    # Expand dimensions based on output format
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z

def save_depth(z, path, multiplier=256.0):
    '''
    Saves a depth map to a 16-bit PNG file

    Arg(s):
        z : numpy[float32]
            depth map
        path : str
            path to store depth map
        multiplier : float
            multiplier for encoding float as unsigned integer
    '''

    z = np.uint32(z * multiplier)
    z = Image.fromarray(z, mode='I')
    z.save(path)

def interpolate_depth(depth_map, validity_map, log_space=False):
    '''
    Interpolate sparse depth with barycentric coordinates

    Arg(s):
        depth_map : np.float32
            H x W depth map
        validity_map : np.float32
            H x W depth map
        log_space : bool
            if set then produce in log space
    Returns:
        np.float32 : H x W interpolated depth map
    '''

    assert depth_map.ndim == 2 and validity_map.ndim == 2

    rows, cols = depth_map.shape
    data_row_idx, data_col_idx = np.where(validity_map)
    depth_values = depth_map[data_row_idx, data_col_idx]

    # Perform linear interpolation in log space
    if log_space:
        depth_values = np.log(depth_values)

    interpolator = LinearNDInterpolator(
        # points=Delaunay(np.stack([data_row_idx, data_col_idx], axis=1).astype(np.float32)),
        points=np.stack([data_row_idx, data_col_idx], axis=1),
        values=depth_values,
        fill_value=0 if not log_space else np.log(1e-3))

    query_row_idx, query_col_idx = np.meshgrid(
        np.arange(rows), np.arange(cols), indexing='ij')

    query_coord = np.stack(
        [query_row_idx.ravel(), query_col_idx.ravel()], axis=1)

    Z = interpolator(query_coord).reshape([rows, cols])

    if log_space:
        Z = np.exp(Z)
        Z[Z < 1e-1] = 0.0

    return Z

def load_validity_map(path, ignore_empty=False, data_format='HW'):
    '''
    Loads a validity map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : binary validity map for available depth measurement locations
    '''

    # Loads validity map from 16-bit PNG file
    v = np.array(Image.open(path), dtype=np.float32)

    if ignore_empty:
        values = np.unique(v)
        assert np.all(values == [0, 256]) or np.all(values == [0]) or np.all(values == [256])
    else:
        assert np.all(np.unique(v) == [0, 256])

    v[v > 0] = 1

    # Expand dimensions based on output format
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        v = np.expand_dims(v, axis=0)
    elif data_format == 'HWC':
        v = np.expand_dims(v, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return v

def save_validity_map(v, path):
    '''
    Saves a validity map to a 16-bit PNG file

    Arg(s):
        v : numpy[float32]
            validity map
        path : str
            path to store validity map
    '''

    v[v <= 0] = 0.0
    v[v > 0] = 1.0
    v = np.uint32(v * 256.0)
    v = Image.fromarray(v, mode='I')
    v.save(path)

def load_exr(path, data_format='HW'):
    '''
    Loads an exr image

    Arg(s):
        path : str
            path to exr file
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : exr depth image
    '''

    z = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    z = np.array(z, dtype=np.float32)

    # Expand dimensions based on output format
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        z = np.expand_dims(z, axis=0)
    elif data_format == 'HWC':
        z = np.expand_dims(z, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return z

def load_uncertainty(path, multiplier=256.0, offset=128.0, data_format='HW'):
    '''
    Loads a uncertainty map from a 16-bit PNG file

    Arg(s):
        path : str
            path to 16-bit PNG file
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        offset : float
            offset for negative values in uncertainty map
        data_format : str
            HW, CHW, HWC
    Returns:
        numpy[float32] : uncertainty map corresponding to estimates
    '''

    # Loads uncertainty map from 16-bit PNG file
    uncertainty = np.array(Image.open(path), dtype=np.float32)

    # Assert 16-bit (not 8-bit) depth map
    uncertainty = uncertainty / multiplier - offset

    # Expand dimensions based on output format
    if data_format == 'HW':
        pass
    elif data_format == 'CHW':
        uncertainty = np.expand_dims(uncertainty, axis=0)
    elif data_format == 'HWC':
        uncertainty = np.expand_dims(uncertainty, axis=-1)
    else:
        raise ValueError('Unsupported data format: {}'.format(data_format))

    return uncertainty

def save_uncertainty(uncertainty, path, multiplier=256.0, offset=128.0):
    '''
    Saves a uncertainty map to a 16-bit PNG file

    Arg(s):
        uncertainty : numpy[float32]
            H x W uncertainty map
        path : str
            path to store validity map
        multiplier : float
            multiplier for encoding float as 16/32 bit unsigned integer
        offset : float
            offset for negative values in uncertainty map
    '''

    uncertainty = np.uint32(uncertainty + offset * multiplier)
    uncertainty = Image.fromarray(uncertainty, mode='I')
    uncertainty.save(path)

def load_calibration(path):
    '''
    Loads the calibration matrices for each camera (KITTI) and stores it as map

    Arg(s):
        path : str
            path to file to be read
    Returns:
        dict[str, float] : map containing camera intrinsics keyed by camera id
    '''

    float_chars = set("0123456789.e+- ")
    data = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                try:
                    data[key] = np.asarray(
                        [float(x) for x in value.split(' ')])
                except ValueError:
                    pass
    return data


def create_pincushion_grid(shape, stride=5):
    '''
    Create grid for pincushion distortion

    Arg(s):
        shape : list[int]
            height, width of image
        stride : int
            space between each grid point
    Returns:
        list[numpy[int]] : x and y meshgrid
    '''

    n_height, n_width = shape

    # Generate grid points
    x_odd, y_odd = np.meshgrid(
        np.arange(stride // 2, n_height, stride * 2),
        np.arange(stride // 2, n_width, stride))

    x_even, y_even = np.meshgrid(
        np.arange(stride // 2 + stride, n_height, stride * 2),
        np.arange(stride, n_width, stride))

    x_u = np.concatenate((x_odd.ravel(), x_even.ravel()))
    y_u = np.concatenate((y_odd.ravel(), y_even.ravel()))

    return (x_u, y_u)

def create_pincushion_validity_map(shape, grid, offset=[0, 0], dist_coef=2e-5, noise=0):
    '''
    Simulate pincushion distortion for validity map

    Arg(s):
        shape : list[int]
            height, width of image
        grid : list[numpy[int]]
            x and y meshgrid
        offset : list[int]
            min, max offset from center
        dist_coef : float32
            controls the curvature of the spot pattern: larger distorts the pattern more. (0 ~ 5e-5)
        noise : float32
            standard deviation of the spot shift (0 ~ 0.5)
    Returns:
        numpy[float32] : validity map
    '''

    n_height, n_width = shape
    x_u, y_u = grid

    x_c = n_height // 2 + np.random.rand() * offset[1] + offset[0]
    y_c = n_width // 2 + np.random.rand() * offset[1] + offset[0]
    x_u = x_u - x_c
    y_u = y_u - y_c

    # Distortion
    r_u = np.sqrt(x_u ** 2 + y_u ** 2)
    r_d = r_u + dist_coef * r_u ** 3
    num_d = r_d.size
    sin_theta = x_u / r_u
    cos_theta = y_u / r_u
    x_d = np.round(r_d * sin_theta + n_height // 2 + np.random.normal(0, noise, num_d))
    y_d = np.round(r_d * cos_theta + n_width // 2 + np.random.normal(0, noise, num_d))
    idx_mask = (x_d < n_height) & (x_d > 0) & (y_d < n_width) & (y_d > 0)
    x_d = x_d[idx_mask].astype('int')
    y_d = y_d[idx_mask].astype('int')

    validity_map = np.zeros(shape)
    validity_map[x_d, y_d] = 1.0

    return validity_map

def create_uniform_validity_map(shape, n_points):
    '''
    Uniform random validity map

    Arg(s):
        shape : list[int]
            height, width of image
        n_points : int
            number of points to sample
    Returns:
        numpy[float32] : validity map

    '''

    n_height, n_width = shape

    indicies = [
        [h, w]
        for h in range(n_height)
        for w in range(n_width)
    ]

    meshgrid_uniform = np.array(indicies)

    selected_indices = \
        np.random.permutation(range(n_height * n_width))[0:n_points]
    selected_indices = meshgrid_uniform[selected_indices]

    validity_map = np.zeros(shape).astype(np.int16)
    validity_map[selected_indices[:, 0], selected_indices[:, 1]] = 1.0

    return validity_map

def create_fast_feature_validity_map(fast_feature_detector, image, n_points):
    '''
    Creates validity map using a FAST feature detector

    Arg(s):
        fast_feature_detector : cv2.FastFeatureDetector
            FAST feature detector instance
        image : numpy[uint8]
            H x W x 3 RGB image in range [0, 255]
        n_points : int
            number of points to sample
    '''

    n_height, n_width = image.shape[0:2]

    # Run fast feature detector
    key_points = fast_feature_detector.detect(image, None)
    key_points = np.array([p.pt for p in key_points]).astype(int)

    # Sample up to the number of points
    selected = np.random.permutation(np.arange(key_points.shape[0]))[:n_points]

    try:
        points = key_points[selected, :]
        validity_map = np.zeros((n_height, n_width))
        validity_map[points[:, 1], points[:, 0]] = 1

    except Exception:
        validity_map = create_uniform_validity_map((n_height, n_width), n_points)

    return validity_map

def create_harris_corner_validity_map(image, n_points, n_max_corners=15000):
    '''
    Creates validity map using a Harris corner detector

    Arg(s):
        image : numpy[uint8]
            H x W x 3 RGB image in range [0, 255]
        n_points : int
            number of points to sample
        n_max_corners : int
            number of top N corners to select
    '''

    n_height, n_width = image.shape[0:2]

    image_gray = np.float32(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    try:
        # Extract corner locations: H x W response map
        corners = cv2.cornerHarris(image_gray, blockSize=5, ksize=3, k=0.04)

        # Flatten and take the top N corners
        corners = corners.ravel()
        corners = np.argsort(corners)[:n_max_corners]

        # Sample up to the number of points
        selected = np.random.permutation(np.arange(corners.shape[0]))[:n_points]
        corners = corners[selected]

        validity_map = np.zeros((n_height, n_width))
        validity_map[selected] = 1

    except Exception:
        validity_map = create_uniform_validity_map((n_height, n_width), n_points)

    return validity_map

def inpainting(depth_map):
    '''
    Fill in almost dense depth_map with mask using Scikit Learn Image's inpaint_biharmonic
    Used for post processing depth maps

    Arg(s):
        depth_map : numpy[float32]
            N x 1 x H x W almost dense depth map
    Returns:
        numpy[float32] : N x 1 x H x W inpainted depth map
    '''

    # Create a mask from the depth map (1 represents holes, 0 represents background)
    n_batch = depth_map.shape[0]
    mask = np.where(depth_map == 0, 1, 0)

    if (mask == np.zeros_like(mask)).all():
        return depth_map

    # Perform inpainting with biharmonic interpolation
    for i in range(n_batch):
        depth = depth_map[i, ...]
        mask = np.where(depth == 0, 1, 0)
        if (mask == np.zeros_like(mask)).all():
            pass
        depth_map[i] = inpaint.inpaint_biharmonic(depth, mask)

    return depth_map
