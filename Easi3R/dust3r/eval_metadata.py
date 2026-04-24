import os
import glob
from tqdm import tqdm

# Define the merged dataset metadata dictionary
dataset_metadata = {
    'davis': {
        'img_path': "data/davis/DAVIS/JPEGImages/480p",
        'mask_path': "data/davis/DAVIS/masked_images/480p",
        'dir_path_func': lambda img_path, seq: os.path.join(img_path, seq),
        'gt_traj_func': lambda img_path, anno_path, seq: None,
        'traj_format': None,
        "seq_list": None,
        'full_seq': True,
        'mask_path_seq_func': lambda mask_path, seq: os.path.join(mask_path, seq),
        'skip_condition': None,
        'process_func': None,
    },
    'tum': {
        'img_path': "data/tum",
        'mask_path': "None",
        'dir_path_func': lambda img_path, seq: os.path.join(img_path, seq, 'rgb_all_stride30'),
        'gt_traj_func': lambda img_path, anno_path, seq: os.path.join(img_path, seq, 'groundtruth_all_stride30.txt'),
        'traj_format': 'tum',
        'seq_list': None,
        'full_seq': True,
        'mask_path_seq_func': lambda mask_path, seq: None,
        'skip_condition': None,
        'process_func': None,
    },
    'iphone': {
        'img_path': "data/iphone",
        'mask_path': "data/iphone",
        'dir_path_func': lambda img_path, seq: os.path.join(img_path, seq, 'prepare_all_stride10/rgb'),
        'gt_traj_func': lambda img_path, anno_path, seq: os.path.join(img_path, seq, 'prepare_all_stride10/camera'),
        'traj_format': 'iphone',
        'seq_list': None,
        'full_seq': True,
        'mask_path_seq_func': lambda mask_path, seq: os.path.join(mask_path, seq, 'prepare_all_stride10/masks'),
        'skip_condition': None,
        'process_func': None,
    },
    'adt': {
        'img_path': "data/adt",
        'mask_path': None,
        'dir_path_func': lambda img_path, seq: os.path.join(img_path, seq, 'prepare_all_stride5/rgb'),
        'gt_traj_func': lambda img_path, anno_path, seq: os.path.join(img_path, seq, 'prepare_all_stride5/camera'),
        'traj_format': 'iphone',
        'seq_list': None,
        'full_seq': True,
        'mask_path_seq_func': lambda mask_path, seq: None,
        'skip_condition': None,
        'process_func': None,
    },
}