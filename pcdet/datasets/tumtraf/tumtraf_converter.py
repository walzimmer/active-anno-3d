import json
import os
import shutil

from pypcd import pypcd
from glob import glob
from typing import Any, Dict, List
from tqdm import tqdm 

import pickle
import numpy as np
import open3d as o3d

from scipy.spatial.transform import Rotation

# from pcdet.datasets.tumtraf.tumtraf_utils import *
from pcdet.utils import common_utils

lidar2ego = np.asarray(
    [
        [0.99011437, -0.13753536, -0.02752358, 2.3728100375737995],
        [0.13828977, 0.99000475, 0.02768645, -16.19297517556697],
        [0.02344061, -0.03121898, 0.99923766, -8.620000000000005],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)[:-1, :]

class TUMTraf2KITTI:
    def __init__(self, load_dir, save_dir, splits: List[str], logger=None):
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.splits = splits
        self.logger = logger

        self.imagesets: Dict[str, list] = {"training": [], "validation": [], "testing": []}

        self.map_split_to_dir_idx = {"training": 0, "validation": 1, "testing": 2}

        self.occlusion_map = {"NOT_OCCLUDED": 0, "PARTIALLY_OCCLUDED": 1, "MOSTLY_OCCLUDED": 2}
    
    def convert(self) -> None:
        self.logger.info("TUMTraf Conversion - start")
        for split in self.splits:
            self.logger.info(f"TUMTraf Conversion - split: {split}")

            split_source_path = os.path.join(self.load_dir, split)
            self.create_folder(split)

            pcd_list = sorted(glob(os.path.join(split_source_path, 
                                                'point_clouds', 
                                                's110_lidar_ouster_south_and_north_registered', '*')))
            pcd_labels_list = sorted(glob(os.path.join(split_source_path, 
                                                       'annotations', 
                                                       's110_lidar_ouster_south_and_north_registered', '*')))
            
            for idx, (pcd_path, pcd_label_path) in tqdm(enumerate(zip(pcd_list, pcd_labels_list))):
                out_pcd = pcd_path.split("/")[-1][:-4]
                out_label = pcd_label_path.split("/")[-1][:-5]

                self.convert_pcd_to_bin(
                    file=pcd_path, 
                    out_file=os.path.join(self.point_cloud_save_dir, out_pcd)
                )
                pcd_list[idx] = os.path.join(self.point_cloud_save_dir, out_pcd) + ".bin"

                self.convert_label2kitti(
                    file=pcd_label_path,
                    out_file=os.path.join(self.label_save_dir, out_label)
                )
                pcd_labels_list[idx] = os.path.join(self.label_save_dir, out_label) + '.txt'

    def convert_pcd_to_bin(self, file: str, out_file: str) -> None:
        """
        Convert file from .pcd to .bin

        Args:
            file: Filepath to .pcd
            out_file: Filepath of .bin
        """
        point_cloud = pypcd.PointCloud.from_path(file)
        np_x = np.array(point_cloud.pc_data["x"], dtype=np.float32)
        np_y = np.array(point_cloud.pc_data["y"], dtype=np.float32)
        np_z = np.array(point_cloud.pc_data["z"], dtype=np.float32)
        np_i = np.array(point_cloud.pc_data["intensity"], dtype=np.float32) # They are already normalized, check the min and max values
        # np_ts = np.zeros((np_x.shape[0],), dtype=np.float32)

        bin_format = np.column_stack((np_x, np_y, np_z, np_i)).flatten()
        bin_format.tofile(os.path.join(f"{out_file}.bin"))

    def convert_label2kitti(self, file: str, out_file: str) -> None:
        
        trunc = -1
        occ = -1
        alpha = -1
        score = -1

        with open(file, 'r') as f:
            data = json.load(f)
        
        labels = []
        for frame_idx, frame_obj in data["openlabel"]["frames"].items():
            for obj_id, obj in  frame_obj["objects"].items():
                obj_data = obj["object_data"]

                obj_type = obj_data["type"]
                loc_x = obj_data["cuboid"]["val"][0]
                loc_y = obj_data["cuboid"]["val"][1]
                loc_z = obj_data["cuboid"]["val"][2]
                length = obj_data["cuboid"]["val"][7]
                width = obj_data["cuboid"]["val"][8]
                height = obj_data["cuboid"]["val"][9]

                rot = np.asarray(obj_data["cuboid"]["val"][3:7], dtype=np.float32)
                yaw = np.asarray(Rotation.from_quat(rot).as_euler("xyz", degrees=False)[2], dtype=np.float32)

                label = [obj_type, trunc, occ, alpha, height, width, length, loc_x, loc_y, loc_z, yaw, score]

                labels.append(label)
            
            with open(out_file + '.txt', 'w') as f:
                    for label in labels:
                        for val in label:
                            f.write(str(val) + ' ')
                        f.write('\n')
    
    def create_folder(self, split: str) -> None:
        """
        Create folder for data preprocessing.
        """
        split_path = split
        self.logger.info(f"Creating folder - split_path: {split_path}")
        pcd_dir = "point_clouds/s110_lidar_ouster_south_and_north_registered"
        label_dir = "annotations/s110_lidar_ouster_south_and_north_registered"
        
        self.point_cloud_save_dir = os.path.join(self.save_dir, split_path, pcd_dir)
        self.label_save_dir = os.path.join(self.save_dir, split_path, label_dir)

        self.logger.info(f"Creating folder : {self.point_cloud_save_dir}")
        os.makedirs(self.point_cloud_save_dir, exist_ok=True, mode=0o777)
        os.makedirs(self.label_save_dir, exist_ok=True, mode=0o777)


class TUMTraf2KITTI_v2:
    def __init__(self, pcd_dict, pcd_labels_dict, save_dir, splits: List[str], logger=None):
        self.save_dir = save_dir
        self.splits = splits
        self.logger = logger

        self.pcd_dict = pcd_dict
        self.pcd_labels_dict = pcd_labels_dict

        self.imagesets: Dict[str, list] = {"training": [], "validation": [], "testing": []}

        self.map_split_to_dir_idx = {"training": 0, "validation": 1, "testing": 2}

        self.occlusion_map = {"NOT_OCCLUDED": 0, "PARTIALLY_OCCLUDED": 1, "MOSTLY_OCCLUDED": 2}
    
    def convert_label_file(self, label_file):
        trunc = -1
        occ = -1
        alpha = -1
        score = -1

        with open(label_file, 'r') as f:
            data = json.load(f)
        
        labels = []
        for frame_idx, frame_obj in data["openlabel"]["frames"].items():
            for obj_id, obj in  frame_obj["objects"].items():
                obj_data = obj["object_data"]

                obj_type = obj_data["type"]
                loc_x = obj_data["cuboid"]["val"][0]
                loc_y = obj_data["cuboid"]["val"][1]
                loc_z = obj_data["cuboid"]["val"][2]
                length = obj_data["cuboid"]["val"][7]
                width = obj_data["cuboid"]["val"][8]
                height = obj_data["cuboid"]["val"][9]

                rot = np.asarray(obj_data["cuboid"]["val"][3:7], dtype=np.float32)
                yaw = np.asarray(Rotation.from_quat(rot).as_euler("xyz", degrees=False)[2], dtype=np.float32)

                label = [obj_type, trunc, occ, alpha, height, width, length, loc_x, loc_y, loc_z, yaw, score]

                labels.append(label)
        return labels

    def convert(self) -> None:
        self.logger.info("TUMTraf Conversion - start")
        for split in self.splits:
            self.logger.info(f"TUMTraf Conversion - split: {split}")

            self.create_folder(split)

            pcd_list = self.pcd_dict[split]
            pcd_labels_list = self.pcd_labels_dict[split]

            for idx, (pcd_path, pcd_label_path) in tqdm(enumerate(zip(pcd_list, pcd_labels_list))):

                out_pcd = '_'.join(pcd_path.split("/")[-1][:-4].split('_')[:2])
                out_label = '_'.join(pcd_label_path.split("/")[-1][:-5].split('_')[:2])
                    
                self.convert_pcd_to_bin(
                    file=pcd_path, 
                    out_file=os.path.join(self.point_cloud_save_dir, out_pcd)
                )
                pcd_list[idx] = os.path.join(self.point_cloud_save_dir, out_pcd) + ".bin"

                self.convert_label2kitti(
                    file=pcd_label_path,
                    out_file=os.path.join(self.label_save_dir, out_label)
                )
                pcd_labels_list[idx] = os.path.join(self.label_save_dir, out_label) + '.txt'

    def convert_pcd_to_bin(self, file: str, out_file: str) -> None:
        """
        Convert file from .pcd to .bin

        Args:
            file: Filepath to .pcd
            out_file: Filepath of .bin
        """
        point_cloud = pypcd.PointCloud.from_path(file)
        np_x = np.array(point_cloud.pc_data["x"], dtype=np.float32)
        np_y = np.array(point_cloud.pc_data["y"], dtype=np.float32)
        np_z = np.array(point_cloud.pc_data["z"], dtype=np.float32)
        np_i = np.array(point_cloud.pc_data["intensity"], dtype=np.float32) # They are already normalized, check the min and max values
        # np_ts = np.zeros((np_x.shape[0],), dtype=np.float32)

        bin_format = np.column_stack((np_x, np_y, np_z, np_i)).flatten()
        bin_format.tofile(os.path.join(f"{out_file}.bin"))

    def convert_label2kitti(self, file: str, out_file: str) -> None:
        
        trunc = -1
        occ = -1
        alpha = -1
        score = -1

        with open(file, 'r') as f:
            data = json.load(f)
        
        labels = []
        for frame_idx, frame_obj in data["openlabel"]["frames"].items():
            for obj_id, obj in  frame_obj["objects"].items():
                obj_data = obj["object_data"]

                obj_type = obj_data["type"]
                loc_x = obj_data["cuboid"]["val"][0]
                loc_y = obj_data["cuboid"]["val"][1]
                loc_z = obj_data["cuboid"]["val"][2]
                length = obj_data["cuboid"]["val"][7]
                width = obj_data["cuboid"]["val"][8]
                height = obj_data["cuboid"]["val"][9]

                rot = np.asarray(obj_data["cuboid"]["val"][3:7], dtype=np.float32)
                yaw = np.asarray(Rotation.from_quat(rot).as_euler("xyz", degrees=False)[2], dtype=np.float32)

                label = [obj_type, trunc, occ, alpha, height, width, length, loc_x, loc_y, loc_z, yaw, score]

                labels.append(label)
            
            with open(out_file + '.txt', 'w') as f:
                    for label in labels:
                        for val in label:
                            f.write(str(val) + ' ')
                        f.write('\n')
    
    def create_folder(self, split: str) -> None:
        """
        Create folder for data preprocessing.
        """
        split_path = split
        self.logger.info(f"Creating folder - split_path: {split_path}")
        pcd_dir = "point_clouds/s110_lidar_ouster_south_and_north_registered"
        label_dir = "annotations/s110_lidar_ouster_south_and_north_registered"
        
        self.point_cloud_save_dir = os.path.join(self.save_dir, split_path, pcd_dir)
        self.label_save_dir = os.path.join(self.save_dir, split_path, label_dir)

        self.logger.info(f"Creating folder : {self.point_cloud_save_dir}")
        os.makedirs(self.point_cloud_save_dir, exist_ok=True, mode=0o777)
        os.makedirs(self.label_save_dir, exist_ok=True, mode=0o777)

if __name__ == "__main__":
    logger = common_utils.create_logger()
    converter = TUMTraf2KITTI(
        load_dir='/ahmed/data/tumtraf/OpenLABEL/stratified',
        save_dir='/ahmed/data/tumtraf/tumtraf_kitti_format',
        splits=['train', 'test', 'val'],
        logger=logger   
    )
    converter.convert()