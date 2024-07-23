import torch
from inverted_model import InvertedModel

##### Parameters #######
model_name = 'kbnet_void'
network_modules = ['depth', 'pose']
min_predict_depth = 0.1
max_predict_depth = 8.0
device = torch.device('cuda')
crop_shape = (416, 512)
n_thread = 8


w_losses = {
    'w_color': 0.15,
    'w_structure': 0.95,
    'w_sparse_depth': 2.0,
    'w_smoothness': 2.0,
    'w_weight_decay_depth': 0.0,
    'w_weight_decay_pose': 0.0,
}

# restore_paths = ['/media/home/sugangopadhyay/workspace/continual-depth-completion/trained_completion/kbnet/nyu_v2/nyu_pretrained7/checkpoints_kbnet_nyu_v2-370000/kbnet-370000.pth', '/media/home/sugangopadhyay/workspace/continual-depth-completion/trained_completion/kbnet/nyu_v2/nyu_pretrained7/checkpoints_kbnet_nyu_v2-370000/posenet-370000.pth']
restore_paths = ['/media/home/alwong/workspace/continual-depth-completion/pretrained_models/depth_completion/kbnet/void/kbnet-void1500-new.pth', '/media/home/alwong/workspace/continual-depth-completion/pretrained_models/depth_completion/kbnet/void/posenet-void1500-new.pth']

image_paths = ['data/void_derived/void_1500/data/classroom2/image/1552096902.5217.png']
sparse_depth_paths = ['data/void_release/void_1500/data/classroom2/sparse_depth/1552096902.5217.png']
intrinsics_paths = ['data/void_derived/void_1500/data/classroom2/K.npy']
ground_truth_paths = ['data/void_release/void_1500/data/classroom2/ground_truth/1552096902.5217.png']

sample_path = "/media/home/sugangopadhyay/workspace/continual-depth-completion/inversion_experiments"
experiment_name = "void_trained_void_retrieval_supervised_lr1e-3_wloss100"
iterations = 10000
lr = 1e-3

########################

inverted_model = InvertedModel(
    model_name,
    network_modules,
    min_predict_depth,
    max_predict_depth,
    crop_shape,
    n_thread,
    restore_paths,
    device,
    lr
)

image0, \
image1, \
image2, \
input_image0, \
input_sparse_depth0, \
input_intrinsics, \
input_validity_map0, \
target_depth0, \
validity_map_depth0 = inverted_model.load_single_data_point(image_paths, sparse_depth_paths, intrinsics_paths, ground_truth_paths)

inverted_model.train_image(
    sample_path, 
    experiment_name,  
    image0, 
    image1, 
    image2, 
    input_image0, 
    input_sparse_depth0, 
    input_intrinsics, 
    input_validity_map0, 
    target_depth0, 
    validity_map_depth0,
    iterations
)