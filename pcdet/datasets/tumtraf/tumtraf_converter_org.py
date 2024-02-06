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

# from pcdet.datasets.a9.a9_utils import lidar2ego
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


class A92KITTI:
    def __init__(self, load_dir, save_dir, splits: List[str], logger=None):
        self.load_dir = load_dir
        self.save_dir = save_dir
        self.splits = splits
        self.logger = logger

        self.imagesets: Dict[str, list] = {"training": [], "validation": [], "testing": []}

        self.map_split_to_dir_idx = {"training": 0, "validation": 1, "testing": 2}
        self.map_split_to_dir = {"training": "train", "validation": "val", "testing": "test"}

        self.occlusion_map = {"NOT_OCCLUDED": 0, "PARTIALLY_OCCLUDED": 1, "MOSTLY_OCCLUDED": 2}

    def convert(self, info_prefix: str) -> None:
        self.logger.info("A9 Conversion - start")
        for split in self.splits:
            self.logger.info(f"A9 Conversion - split: {self.map_split_to_dir[split]}")

            split_source_path = os.path.join(self.load_dir, self.map_split_to_dir[split])
            self.create_folder(split)

            test = True if split == "testing" else False

            pcd_list = sorted(glob(os.path.join(split_source_path, 
                                                'point_clouds', 
                                                's110_lidar_ouster_south', '*')))
            pcd_labels_list = sorted(glob(os.path.join(split_source_path, 
                                                       'labels_point_clouds', 
                                                       's110_lidar_ouster_south', '*')))

            for idx, pcd_path in tqdm(enumerate(pcd_list)):
                out_filename = pcd_path.split("/")[-1][:-4]

                # self.convert_pcd_to_bin(
                #     pcd_path, os.path.join(self.point_cloud_save_dir, out_filename)
                # )
                pcd_list[idx] = os.path.join(self.point_cloud_save_dir, out_filename) + ".bin"


            infos_list = self.write_infos(
                pcd_list=pcd_list,
                pcd_labels_list=pcd_labels_list,
                test=test
            )
            metadata = dict(version="r2")

            if test:
                self.logger.info(f"No. test samples: {len(infos_list)}")
                data = dict(infos=infos_list, metadata=metadata)
                info_path = os.path.join(self.save_dir, f"{info_prefix}_infos_test.pkl")
                with open(info_path, 'wb') as file:
                    pickle.dump(data, file)
            else:
                if split == "training":
                    self.logger.info(f"No. train samples: {len(infos_list)}")
                    data = dict(infos=infos_list, metadata=metadata)
                    info_path = os.path.join(self.save_dir, f"{info_prefix}_infos_train.pkl")
                    with open(info_path, 'wb') as file:
                        pickle.dump(data, file)
                elif split == "validation":
                    self.logger.info(f"No. val samples: {len(infos_list)}")
                    data = dict(infos=infos_list, metadata=metadata)
                    info_path = os.path.join(self.save_dir, f"{info_prefix}_infos_val.pkl")
                    with open(info_path, 'wb') as file:
                        pickle.dump(data, file)

        self.logger.info("A9 Conversion - end")

    def write_infos(self, pcd_list, pcd_labels_list, test=False) -> List[Dict[str, Any]]:
        infos_list = []

        for i, pcd_path in enumerate(pcd_list):
            json_file = open(pcd_labels_list[i])
            json_str = json_file.read()

            lidar_annotation = json.loads(json_str)
            lidar_anno_frame = {}

            for frame_idx in lidar_annotation["openlabel"]["frames"]:
                lidar_anno_frame = lidar_annotation["openlabel"]["frames"][frame_idx]

            info = {
                "lidar_path": pcd_path,
                "lidar_anno_path": pcd_labels_list[i],
                "lidar2ego": lidar2ego,
                "timestamp": lidar_anno_frame["frame_properties"]["timestamp"],
                "location": lidar_anno_frame["frame_properties"]["point_cloud_file_names"][0].split(
                    "_"
                )[2],
            }

            if not test:
                gt_boxes = []
                gt_names = []
                velocity = []
                valid_flag = []
                num_lidar_pts = []
                num_radar_pts = []

                for id in lidar_anno_frame["objects"]:
                    object_data = lidar_anno_frame["objects"][id]["object_data"]

                    loc = np.asarray(object_data["cuboid"]["val"][:3], dtype=np.float32)
                    dim = np.asarray(object_data["cuboid"]["val"][7:], dtype=np.float32)
                    rot = np.asarray(
                        object_data["cuboid"]["val"][3:7], dtype=np.float32
                    )  # quaternion in x,y,z,w

                    rot_temp = Rotation.from_quat(rot)
                    rot_temp = rot_temp.as_euler("xyz", degrees=False)

                    yaw = np.asarray(rot_temp[2], dtype=np.float32)

                    gt_box = np.concatenate([loc, dim, -yaw], axis=None)

                    gt_boxes.append(gt_box)
                    gt_names.append(object_data["type"])
                    velocity.append([0, 0])
                    valid_flag.append(True)

                    for n in object_data["cuboid"]["attributes"]["num"]:
                        if n["name"] == "num_points":
                            num_lidar_pts.append(n["val"])

                    num_radar_pts.append(0)

                gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
                info["gt_boxes"] = gt_boxes
                info["gt_names"] = np.array(gt_names)
                info["gt_velocity"] = np.array(velocity).reshape(-1, 2)
                info["num_lidar_pts"] = np.array(num_lidar_pts)
                info["num_radar_pts"] = np.array(num_radar_pts)
                info["valid_flag"] = np.array(valid_flag, dtype=bool)

            infos_list.append(info)

        return infos_list

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

    def create_folder(self, split: str) -> None:
        """
        Create folder for data preprocessing.
        """
        split_path = self.map_split_to_dir[split]
        self.logger.info(f"Creating folder - split_path: {split_path}")
        dir_list1 = ["point_clouds/s110_lidar_ouster_south"]
        for d in dir_list1:
            self.point_cloud_save_dir = os.path.join(self.save_dir, split_path, d)
            self.logger.info(f"Creating folder : {self.point_cloud_save_dir}")
            os.makedirs(self.point_cloud_save_dir, exist_ok=True, mode=0o777)


def create_groundtruth_dataset(info_path: str, used_classes: List[str], split: str):
        #@TODO: re-write
        
        database_save_path = Path(root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(root_path) / ('a9_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)

        all_db_infos = {}

        with open(info_path, 'rb') as f:
            data = pickle.load(f)

        info_list = data['infos']
 
        for k in range(len(info_list)):
            print('gt_database sample: %d/%d' % (k + 1, len(info_list)))

            info = info_list[k]
            lidar_file = info['lidar_path']
            points = self.get_lidar(lidar_file=lidar_file)

            names = info['gt_names']
            gt_boxes = info['gt_boxes']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):

                filename = '%s_%s_%d.bin' % (lidar_file, names[i], i)
                filepath = database_save_path / filename

                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]

                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'lidar_file': lidar_file, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}

                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

if __name__ == "__main__":
    logger = common_utils.create_logger()
    converter = A92KITTI(
        load_dir='/ahmed/data/a9-r2',
        save_dir='/ahmed/data/a9-r2/a9_kitti_format',
        splits=['training', 'testing', 'validation'],
        logger=logger
    )
    converter.convert(info_prefix='a9')