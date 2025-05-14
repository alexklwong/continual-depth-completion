# Continual Depth Completion

This repository provides standardized training scripts and evaluation pipelines for continual depth completion using a variety of methods (EWC, ANCL, LwF, Replay, CMP). It supports six major RGB-D datasets across indoor and outdoor environments.

## Dataset Setup

You will need to manually download the following datasets and structure them under the `training/` and `testing/` directories as shown in the bash scripts.

| Dataset       | Download Link                                                                 |
|---------------|-------------------------------------------------------------------------------|
| NYU-Depth-V2  | https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html                     |
| ScanNet       | http://www.scan-net.org/                                                     |
| VOID          | https://github.com/alexklwong/void-dataset                                   |
| KITTI         | http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction |
| Virtual KITTI | https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds/  |
| Waymo         | https://waymo.com/open/download/                                              |

Please preprocess and structure each dataset according to your experiment needs (see `training/` and `testing/` folders used in bash scripts for reference).

## Bash Scripts for Training

All bash scripts are located in:
```bash
continual-depth-completion/bash/
```

Each file corresponds to a specific experiment with a given method (e.g., under folder ancl, `train_fusionnet_kitti_waymo.sh` trains FusionNet using ANCL on KITTI→Waymo). Scripts are provided for the following methods:

- `ewc`: Elastic Weight Consolidation
- `ancl`: Auxiliary Network CL 
- `lwf`: Learning without Forgetting
- `replay`: Experience Replay
- `cmp`: Optimized Buffer Replay

## Running a Script

To train a model using one of the predefined bash scripts, simply run:

```bash
bash continual-depth-completion/bash/train_kbnet_nyu_v2_void.sh
```
Before you run, make sure to fill the required fields in bash scripts. 
## Filling in Restore/Frozen Paths

Some bash scripts omit the `--restore_paths`, `--frozen_model_paths`, and `--replay_*_paths`. You must fill them in manually based on your method.

### Restore Paths

Required for all methods to load pretrained models. Order:

```bash
--restore_paths \
    path/to/model.pth \
    path/to/posenet.pth \
    path/to/fisher-info.pth  # Fisher only required for EWC and ANCL
```
### Frozen Model Paths 
```bash
--frozen_model_paths \ 
    # Use the same paths as restore_paths
    path/to/model.pth \
    path/to/posenet.pth \
```
### Replay Paths 
Use the *_replay_dataset_*.txt files stored in the folder of the pretrained checkpoint you are continually training from:
```bash
# example
--replay_image_paths \
    trained_completion/cmp/fusionnet_nyu_void/checkpoints.../next_replay_dataset_image_paths.txt
--replay_sparse_depth_paths \
    trained_completion/cmp/fusionnet_nyu_void/checkpoints.../next_replay_dataset_sparse_depth_paths.txt
--replay_intrinsics_paths \
    trained_completion/cmp/fusionnet_nyu_void/checkpoints.../next_replay_dataset_intrinsics_paths.txt
```

### evaluation
Validation is automatically run during training using the dataset sequences defined in the bash script. 

### output

Trained checkpoints and logs are saved to the folder specified via:
```bash
--checkpoint_path trained_completion/{method}/{model_name}_{dataset_sequence}
# example: --checkpoint_path trained_completion/ancl/kbnet_nyu_v2_void
```
