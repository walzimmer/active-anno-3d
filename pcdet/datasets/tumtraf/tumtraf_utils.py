import os
import ruamel.yaml
import shutil
import numpy as np
import random
import pickle
import json

from tqdm import tqdm
from glob import glob
from pathlib import Path
from pypcd import pypcd
from typing import List

from scipy.spatial.transform import Rotation

from pcdet.datasets.tumtraf.tumtraf_converter import TUMTraf2KITTI
from pcdet.utils import common_utils



lidar2s1image = np.asarray(
    [
        [7.04216073e02, -1.37317442e03, -4.32235765e02, -2.03369364e04],
        [-9.28351327e01, -1.77543929e01, -1.45629177e03, 9.80290034e02],
        [8.71736000e-01, -9.03453000e-02, -4.81574000e-01, -2.58546000e00],
    ],
    dtype=np.float32,
)

lidar2s2image = np.asarray(
    [
        [1546.63215008, -436.92407115, -295.58362676, 1319.79271737],
        [93.20805656, 47.90351592, -1482.13403199, 687.84781276],
        [0.73326062, 0.59708904, -0.32528854, -1.30114325],
    ],
    dtype=np.float32,
)

south1intrinsics = np.asarray(
    [
        [1400.3096617691212, 0.0, 967.7899705163408],
        [0.0, 1403.041082755918, 581.7195041357244],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

south12ego = np.asarray(
    [
        [-0.06377762, -0.91003007, 0.15246652, -10.409943],
        [-0.41296193, -0.10492031, -0.8399004, -16.2729],
        [0.8820865, -0.11257353, -0.45447016, -11.557314],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]

south12lidar = np.linalg.inv(
    np.asarray(
        [
            [-0.10087585, -0.51122875, 0.88484734, 1.90816304],
            [-1.0776537, 0.03094424, -0.10792235, -14.05913251],
            [0.01956882, -0.93122171, -0.45454375, 0.72290242],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
)[:-1, :]

south2intrinsics = np.asarray(
    [
        [1029.2795655594014, 0.0, 982.0311857478633],
        [0.0, 1122.2781391971948, 1129.1480997238505],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

south22ego = np.asarray(
    [
        [0.650906, -0.7435749, 0.15303044, 4.6059465],
        [-0.14764456, -0.32172203, -0.935252, -15.00049],
        [0.74466264, 0.5861663, -0.3191956, -9.351643],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]

south22lidar = np.linalg.inv(
    np.asarray(
        [
            [0.49709212, -0.19863714, 0.64202357, -0.03734614],
            [-0.60406415, -0.17852863, 0.50214409, 2.52095055],
            [0.01173726, -0.77546627, -0.70523436, 0.54322305],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
)[:-1, :]

def get_objects_and_names_from_label(label_file):
    label_ext = label_file.split('/')[-1].split('.')[-1]

    if label_ext == 'txt':
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        gt_names = []
        gt_boxes = []
        for line in lines:
            label = line.strip().split(' ')
            name = label[0]
            loc = np.array([float(label[7]), float(label[8]), float(label[9])], dtype=np.float32)
            dim = np.array([float(label[6]), float(label[5]), float(label[4])], dtype=np.float32)
            yaw = float(label[10])

            gt_names.append(name)
            gt_boxes.append(np.concatenate([loc, dim, yaw], axis=None))
        gt_names = np.array(gt_names)
        gt_boxes = np.array(gt_boxes)
            

    elif label_ext == 'json':
        with open(label_file, 'r') as f:
            data = json.load(f)
        

        gt_names = []
        gt_boxes = []
        for frame_idx, frame_obj in data["openlabel"]["frames"].items():
            for obj_id, obj in  frame_obj["objects"].items():
                obj_data = obj["object_data"]

                name = obj_data["type"]
                loc = np.array([
                    float(obj_data["cuboid"]["val"][0]),
                    float(obj_data["cuboid"]["val"][1]),
                    float(obj_data["cuboid"]["val"][2])
                ], dtype=np.float32)
                dim = np.array([
                    float(obj_data["cuboid"]["val"][9]),
                    float(obj_data["cuboid"]["val"][8]),
                    float(obj_data["cuboid"]["val"][7])
                ], dtype=np.float32)

                rot = np.asarray(obj_data["cuboid"]["val"][3:7], dtype=np.float32)
                yaw = np.asarray(Rotation.from_quat(rot).as_euler("xyz", degrees=False)[2], dtype=np.float32)

                gt_names.append(name)
                gt_boxes.append(np.concatenate([loc, dim, yaw], axis=None))
        gt_names = np.array(gt_names)
        gt_boxes = np.array(gt_boxes)

    else:
        raise ValueError("label file extension should be either txt or json.")
    
    return gt_boxes, gt_names

def create_image_sets(data_root, splits: List[str]):
    imagesets_dir = os.path.join(data_root, "ImageSets")
    os.makedirs(imagesets_dir, exist_ok=True)

    for split in splits:
        split_dir = os.path.join(data_root, split)
        lidar_name = os.listdir(os.path.join(split_dir, 'point_clouds'))[0]
        pcd_files = sorted(glob(os.path.join(split_dir, 'point_clouds', lidar_name, '*.bin')))
        pcd_names = [i.split('.')[0].split('/')[-1] for i in pcd_files]

        split_txtfile = os.path.join(imagesets_dir, split + '.txt')
        with open(split_txtfile, 'w') as f:
            for name in tqdm(pcd_names):
                f.write(name + '\n')

def create_tumtraf_demo(data_root, output_dir, splits=["train"], logger=None):

    os.makedirs(output_dir, exist_ok=True)

    demo_size = 100
    lidars = ['s110_lidar_ouster_south']
    cameras = ['s110_camera_basler_south1_8mm',
               's110_camera_basler_south2_8mm']
    for split in splits:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)

        # define the indices of the demo items. 
        # since camera and lidar files do not have the same timestamps.
        # we cannot sample them based on the timestamps but rather on the indices.
        data_len = len(glob(os.path.join(data_root, split, 
                                         'point_clouds', 's110_lidar_ouster_south', '*')))
        full_indices = list(range(data_len))
        demo_indices = random.sample(full_indices, demo_size)

        for lidar in lidars:
            os.makedirs(os.path.join(split_dir, 'point_clouds', lidar), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'labels_point_clouds', lidar), exist_ok=True)

            pcd_files = sorted(glob(os.path.join(data_root, split, 
                                                 'point_clouds', lidar, '*')))
            pcd_label_files = sorted(glob(os.path.join(data_root, split, 
                                                       'labels_point_clouds', lidar, '*')))

            demo_pcd_files = [pcd_files[i] for i in demo_indices]
            demo_pcd_label_files = [pcd_label_files[i] for i in demo_indices]

            for i in range(demo_size):
                shutil.copy(demo_pcd_files[i], os.path.join(split_dir, 'point_clouds', lidar))
                shutil.copy(demo_pcd_label_files[i], os.path.join(split_dir, 'labels_point_clouds', lidar))
        
        for cam in cameras:
            os.makedirs(os.path.join(split_dir, 'images', cam), exist_ok=True)
            os.makedirs(os.path.join(split_dir, 'labels_images', cam), exist_ok=True)

            imgs = sorted(glob(os.path.join(data_root, split, 'images', cam, '*')))
            img_labels = sorted(glob(os.path.join(data_root, split, 'labels_images', cam, '*')))

            demo_img_files = [imgs[i] for i in demo_indices]
            demo_img_label_files = [img_labels[i] for i in demo_indices]

            for i in range(demo_size):
                shutil.copy(demo_img_files[i], os.path.join(split_dir, 'images', cam))
                shutil.copy(demo_img_label_files[i], os.path.join(split_dir, 'labels_images', cam))

    
    tumtraf_converter = TUMTraf2KITTI(
        splits=splits,
        load_dir=output_dir,
        save_dir=os.path.join(output_dir, "data_kitti_format"),
        logger=logger
    )
    tumtraf_converter.convert()
    
def create_demo_cfg(parent_cfg_path: str, model_cfg_path: str, save_path: str):

    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True

    cfg_name = parent_cfg_path.split('/')[-1].split('.')[0]

    with open(parent_cfg_path, 'r') as f:
        dataset_cfg = yaml.load(f)
    
    dataset_cfg['DATASET'] = 'TUMTrafDATASET_DEMO'
    dataset_cfg['DATA_PATH'] = os.path.join(save_path, 'data_kitti_format')

    new_cfg = os.path.join(save_path, cfg_name + '_demo.yaml')
    with open(new_cfg, 'w') as f:
        yaml.dump(dataset_cfg, f) 

    with open(model_cfg_path, 'r') as f:
        model_cfg = yaml.load(f)
    
    model_name = model_cfg_path.split('/')[-1].split('.')[0]
    model_cfg['DATA_CONFIG']['_BASE_CONFIG_'] = new_cfg
    model_cfg_new = os.path.join(save_path, model_name + '.yaml')

    with open(model_cfg_new, 'w') as f:
        yaml.dump(model_cfg, f)


if __name__ == "__main__":
    logger = common_utils.create_logger()
    # create_tumtraf_demo(
    #     data_root="/ahmed/data/tumtraf",
    #     output_dir="/ahmed/data/tumtraf_demo_data",
    #     splits=["train"],
    #     logger=logger
    # )
    # create_demo_cfg(
    #     parent_cfg_path='/ahmed/tools/cfgs/dataset_configs/tumtraf_dataset.yaml',
    #     model_cfg_path='/ahmed/tools/cfgs/tumtraf_models/pv_rcnn.yaml',
    #     save_path='/ahmed/data/tumtraf_demo_data'
    # )
    create_image_sets(
        data_root='/ahmed/data/tumtraf/tumtraf_kitti_format',
        splits=['train', 'test', 'val']
    )

