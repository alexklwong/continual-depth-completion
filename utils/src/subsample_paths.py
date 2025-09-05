import os, argparse
import data_utils

parser = argparse.ArgumentParser()

parser.add_argument('--input_filepath', type=str, required=True)
parser.add_argument('--output_filepath', type=str, required=True)
parser.add_argument('--subsample_factor', type=int, required=True)

args = parser.parse_args()

input_paths = data_utils.read_paths(args.input_filepath)
output_paths = input_paths[::args.subsample_factor]

assert input_paths[0] == output_paths[0], \
    (input_paths[0], output_paths[0])
assert input_paths[args.subsample_factor] == output_paths[1], \
    (input_paths[args.subsample_factor], output_paths[1])

print('# of Input Paths: {}'.format(len(input_paths)))
print('# of Output Paths: {}'.format(len(output_paths)))

data_utils.write_paths(args.output_filepath, output_paths)
