import os
import copy
import pickle
import numpy as np
import skimage as io
import yaml
import torch
import cv2
import open3d as o3d

from easydict import EasyDict
from typing import Any, Dict, List
from glob import glob

from pcdet.datasets.dataset import DatasetTemplate
from pcdet.utils import box_utils, common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

from pcdet.datasets.tumtraf.tumtraf_converter import *
from pcdet.datasets.tumtraf.tumtraf_utils import *


class TUMTrafDataset(DatasetTemplate):

    def __init__(self, 
                 dataset_cfg, 
                 class_names, 
                 training=True, 
                 validation=False, 
                 root_path=None, 
                 logger=None, 
                 format='kitti'):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=training,
            validation=validation,
            root_path=root_path,
            logger=logger
        )
        # assert dataset_cfg.DATA_PATH == root_path, "confusion between data path and root path!"
        
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        # self.split = 'val'
        split_info = os.path.join(self.root_path, 'ImageSets', self.split + '.txt')
            
        self.sample_id_list = [x.strip() for x in open(split_info).readlines()] if os.path.exists(split_info) else None

        self.tumtraf_infos = []
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI

        self.data_format=format

    def include_data(self, mode):

        if self.logger:
            self.logger.info('Loading TUMTraf dataset.')
        tumtraf_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = os.path.join(self.root_path, info_path)
            if not os.path.isfile(info_path):
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                tumtraf_infos.extend(infos)

        self.tumtraf_infos.extend(tumtraf_infos)
        if self.logger:
            self.logger.info('Total samples for TUMTraf dataset: %d' % (len(tumtraf_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg,
            class_names=self.class_names,
            training=self.training,
            root_path=self.root_path,
            logger=self.logger
        )
        self.split = split
        split_dir = os.path.join(self.root_path, 'ImageSets', self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if os.path.exists(split_dir) else None

    def get_lidar(self, idx: str):
        print("lidar_file: ", idx)
        lidar_name = os.listdir(os.path.join(self.root_path, self.split, 'point_clouds'))[0]
        if self.data_format == "kitti":
            lidar_file = os.path.join(self.root_path, self.split, 
                                      'point_clouds', lidar_name, idx + '.bin')
            assert os.path.exists(lidar_file), f"{lidar_file}, Wrong path, or file does not exist."
            point_features = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)
        
        else:   # OpenLABEL format
            lidar_file = os.path.join(self.root_path, self.split, 
                                      'point_clouds', lidar_name, idx + '.pcd')
            assert os.path.isfile(lidar_file), f"{lidar_file}, Wrong path, or file does not exist."
            point_features = o3d.io.read_point_cloud(lidar_file)

        return point_features

    def get_image(self, idx: str, which_cam):
        """
        Load an image for a sample.
        Args:
            idx: str, Sample file name
            which_cam: str, which camera of the two cameras.
        Returns:
            image:  (H, W, 3), RGB Image
        """
        img_file = os.path.join(self.root_path, self.split, which_cam, idx + '.png')
        assert os.path.isfile(img_file), "Wrong path, or file does not exist."
        img = io.imread(img_file)
        img = img.astype(np.float32)
        img /= 255.0

        return img

    def get_image_shape(self, idx: str):
        pass

    def get_label(self, idx: str):
        lidar_name = os.listdir(os.path.join(self.root_path, self.split, 'annotations'))[0]
        if self.data_format == 'kitti':
            anno_file = os.path.join(self.root_path, self.split, 
                                     'annotations', lidar_name, idx + '.txt')
        
        else:
            anno_file = os.path.join(self.root_path, self.split, 
                                     'annotations', lidar_name, idx + '.json')

        assert os.path.exists(anno_file), f"{anno_file},Wrong path, or file does not exist."
        gt_boxes, gt_names = get_objects_and_names_from_label(label_file=anno_file)
        return gt_boxes, gt_names
        
    def get_infos(self, class_names, num_workers=4, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_idx)
                annotations['name'] = name
                annotations['boxes_3d'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)

        return list(infos)


    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):

        database_save_path = os.path.join(self.root_path, 'gt_database' if split == 'train' else 'gt_database_%s' % split)
        db_info_save_path = os.path.join(self.root_path, 'tumtraf_dbinfos_%s.pkl' % split)

        os.makedirs(database_save_path, exist_ok=True)

        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
 
        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))

            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(idx=sample_idx)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['boxes_3d']

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):

                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = os.path.join(database_save_path, filename)

                gt_points = points[point_indices[i] > 0]
                gt_points[:, :3] -= gt_boxes[i, :3]

                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = os.path.relpath(filepath, start=self.root_path)
                    db_info = {'name': names[i], 'path': db_path, 'lidar_idx': sample_idx, 'gt_idx': i,
                               'gt_box3d': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}

                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 
                'score': np.zeros(num_samples),
                'boxes_3d': np.zeros([num_samples, 7]),
                'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            if box_dict['pred_scores'].requires_grad:
                pred_scores = box_dict['pred_scores'].detach().cpu().numpy()
                pred_boxes = box_dict['pred_boxes'].detach().cpu().numpy()
                pred_labels = box_dict['pred_labels'].detach().cpu().numpy()

            else:
                pred_scores = box_dict['pred_scores'].cpu().numpy()
                pred_boxes = box_dict['pred_boxes'].cpu().numpy()
                pred_labels = box_dict['pred_labels'].cpu().numpy()

            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_3d'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            if 'metadata' in batch_dict:
                single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    
    def evaluation(self, det_annos, class_names, **kwargs):
        from .tumtraf_eval import evaluation as tumtraf_eval
        if 'annos' not in self.tumtraf_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils


            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)

            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.tumtraf_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        
        elif kwargs['eval_metric'] == 'tumtraf':
            ap_result_str, ap_dict = tumtraf_eval.get_evaluation_results(eval_gt_annos, eval_det_annos, class_names)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.tumtraf_infos)

    def __getitem__(self, index):

        if self.merge_all_iters_to_one_epoch:
            index = index % len(self.tumtraf_infos)

        info = copy.deepcopy(self.tumtraf_infos[index])
        
        sample_idx = info['point_cloud']['lidar_idx']
        points = self.get_lidar(sample_idx)
        input_dict = {
            'frame_id': self.sample_id_list[index],
            'points': points
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='OTHER')
            gt_names = annos['name']
            gt_boxes_lidar = annos['boxes_3d']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            input_dict['sample_id_list'] = self.sample_id_list

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def create_tumtraf_infos(dataset_cfg, class_names, data_path, save_path, splits=['train', 'val']):
    dataset = TUMTrafDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger(),
        format='kitti'
    )
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    for split in splits:
        split_filename = os.path.join(save_path, 'tumtraf_infos_%s.pkl' % split)

        print('------------------------Start to generate data infos------------------------')
        dataset.set_split(split)

        has_label = True
        if split == 'test':
            has_label = False
        tumtraf_infos = dataset.get_infos(
            class_names=class_names,
            has_label=has_label, 
            num_features=num_features
        )
        with open(split_filename, 'wb') as f:
            pickle.dump(tumtraf_infos, f)
        print('TUMTraf info %s file is save to %s' % (split, split_filename))

        print('------------------------Start create groundtruth database for data augmentation------------------------')
        dataset.set_split(split)
        if split != 'test':
            dataset.create_groundtruth_database(split_filename, split=split)
        # dataset.create_groundtruth_database(split_filename, split=split)

        print('------------------------Data preparation done------------------------')


if __name__ == "__main__":

    with open('/ahmed/tools/cfgs/dataset_configs/tumtraf_dataset.yaml', 'r') as file:
        DATASET_CFG = EasyDict(yaml.safe_load(file))

    ROOT_DIR = "/ahmed/data/tumtraf/tumtraf_kitti_format"
    CLASS_NAMES = ['CAR', 'VAN', 'BICYCLE', 'MOTORCYCLE', 'TRUCK', 'TRAILER', 'BUS','PEDESTRIAN']

    tumtraf_data = TUMTrafDataset(
            dataset_cfg=DATASET_CFG,
            class_names=CLASS_NAMES,
            training=False,  # because this is to generate ground-truth database
            root_path=ROOT_DIR
    )
    create_tumtraf_infos(dataset_cfg=DATASET_CFG,
                    class_names=CLASS_NAMES,
                    data_path=ROOT_DIR,
                    save_path=ROOT_DIR,
                    splits=['train', 'val', 'test']
                    )



