import argparse, torch
from depth_completion import run


parser = argparse.ArgumentParser()

# Input filepaths
parser.add_argument('--image_path',
    type=str, required=True, help='Path to list of image paths')
parser.add_argument('--sparse_depth_path',
    type=str, required=True, help='Path to list of sparse depth paths')
parser.add_argument('--ground_truth_path',
    type=str, required=None, help='Path to list of ground truth paths')
parser.add_argument('--intrinsics_path',
    type=str, default=None, help='Path to list of intrinsics paths')
parser.add_argument('--restore_path_model',
    type=str, required=True, help='Path to restore depth model from checkpoint')

# Network settings
parser.add_argument('--model_name',
    nargs='+', type=str, help='Depth completion model to instantiate')
parser.add_argument('--min_predict_depth',
    type=float, default=1.5, help='Minimum depth prediction value')
parser.add_argument('--max_predict_depth',
    type=float, default=100.0, help='Maximum depth prediction value')

# Evaluation settings
parser.add_argument('--min_evaluate_depth',
    type=float, default=0.1, help='Minimum value of depth to evaluate')
parser.add_argument('--max_evaluate_depth',
    type=float, default=100.0, help='Maximum value of depth to evaluate')
parser.add_argument('--evaluation_protocol',
    type=str, default='default', help='Protocol for evaluation i.e. vkitti, nuscenes, default')

# Output settings
parser.add_argument('--output_dirpath',
    type=str, required=True, help='Path to directory to log results')
parser.add_argument('--save_outputs',
    action='store_true', help='If set then strore inputs and outputs into output directory')
parser.add_argument('--keep_input_filenames',
    action='store_true', help='If set then keep input filenames')

# Outlier Removal
parser.add_argument('--outlier_removal_kernel_size', default=7, type=int)
parser.add_argument('--outlier_removal_threshold', default=1.5, type=float)

# Hardware settings
parser.add_argument('--device',
    type=str, default='gpu', help='Device to use: gpu, cpu')


args = parser.parse_args()


if __name__ == '__main__':

    # Network settings
    args.model_name = [
        name.lower() for name in args.model_name
    ]

    # Hardware settings
    args.device = args.device.lower()
    if args.device not in ['cpu', 'gpu', 'cuda']:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.device = 'cuda' if args.device == 'gpu' else args.device

    run(image_path=args.image_path,
        sparse_depth_path=args.sparse_depth_path,
        ground_truth_path=args.ground_truth_path,
        intrinsics_path=args.intrinsics_path,
        restore_path_model=args.restore_path_model,
        # Network settings
        model_name=args.model_name,
        min_predict_depth=args.min_predict_depth,
        max_predict_depth=args.max_predict_depth,
        # Evaluation settings
        min_evaluate_depth=args.min_evaluate_depth,
        max_evaluate_depth=args.max_evaluate_depth,
        evaluation_protocol=args.evaluation_protocol,
        # Output settings
        output_dirpath=args.output_dirpath,
        save_outputs=args.save_outputs,
        keep_input_filenames=args.keep_input_filenames,
        # Outlier Removal
        outlier_removal_kernel_size=args.outlier_removal_kernel_size,
        outlier_removal_threshold=args.outlier_removal_threshold,
        # Hardware settings
        device=args.device)
