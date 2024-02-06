import os
import glob
import json
import pickle as pkl
import datetime
import shutil
from pathlib import Path

import numpy as np
from pcdet.datasets.tumtraf.tumtraf_dataset import TUMTrafDataset
from pcdet.datasets.tumtraf.tumtraf_converter import TUMTraf2KITTI_v2
from pcdet.datasets.tumtraf.tumtraf_utils import create_image_sets


class DataManager():
    def __init__(self, data_openlabel_dir, operation):
        self.operation = operation
        self.data_dir = data_openlabel_dir 

        self.datalog = self.get_datalog()
        self.curr_data_status = self.get_curr_data_status()

    def get_datalog(self):
        datalog_path = self.data_dir / 'datalog.pkl'
        if datalog_path.is_file():
            with open(datalog_path, 'rb') as f:
                datalog = pkl.load(f)
        else:
            datalog = {
                'base_data_config': {
                    'train': [],
                    'val': [],
                    'test': [],
                    'currently_annotating': [],
                    'latest_annotated': []
                },
                'operation_log': [            {
                    'date': None,
                    'operation': None,
                    'pre_data_config': {
                        'train': [],
                        'val': [],
                        'test': [],
                        'currently_annotating': [],
                        'latest_annotated': []
                    },
                    'post_data_config': {
                        'train': [],
                        'val': [],
                        'test': [],
                        'currently_annotating': [],
                        'latest_annotated': []
                    },
                }]
            }


        return datalog
    
    def get_curr_data_status(self):
        dirs = [
            'currently_annotating',
            'latest_annotated',
            'train',
            'test',
            'val'
        ]
        data_status = {
            dir: [] for dir in dirs
        }
        for dir in dirs:
            dir_path = os.path.join(self.data_dir, dir)
            lidar_channel = os.listdir(os.path.join(dir_path, 'annotations'))[0]
            anno_path = os.path.join(dir_path, 'annotations', lidar_channel)

            anno_filenames = sorted(glob.glob(os.path.join(anno_path, '*.json')))
            anno_filenames = [i.split('.')[0].split('/')[-1] for i in anno_filenames]

            data_status[dir] = anno_filenames
        
        return data_status
    
    def split_data(self):
        
        latest_annotated_path = os.path.join(self.data_dir, 'latest_annotated')
        latest_lidar_name = os.listdir(os.path.join(latest_annotated_path, 'point_clouds'))[0]

        pcd_dict = {}
        labels_dict = {}

        splits = ['train', 'test']
        for split in splits:
            split_path = os.path.join(self.data_dir, split)
            lidar_name = os.listdir(os.path.join(split_path, 'point_clouds'))[0]

            if ((self.operation == 'train_active_all' or self.operation == 'train_all_only') and (split == 'train')):
                
                pcd_path = os.path.join(split_path, 'point_clouds', lidar_name)
                pcd_labels_path = os.path.join(split_path, 'annotations', lidar_name)

                pcd_filenames = sorted(glob.glob(os.path.join(pcd_path, '*.pcd')))
                labels_filenames = sorted(glob.glob(os.path.join(pcd_labels_path, '*.json')))

                pcd_latest_path = os.path.join(latest_annotated_path, 'point_clouds', latest_lidar_name)
                labels_latest_path = os.path.join(latest_annotated_path, 'annotations', latest_lidar_name)

                pcd_latest_filenames = sorted(glob.glob(os.path.join(pcd_latest_path, '*.pcd')))
                labels_latest_filenames = sorted(glob.glob(os.path.join(labels_latest_path, '*.json')))

                labels_all_list = labels_filenames + labels_latest_filenames
                pcd_all_list = pcd_filenames + pcd_latest_filenames

                pcd_dict[split] = pcd_all_list
                labels_dict[split] = labels_all_list

            elif ((self.operation == 'train_active_latest' or self.operation == 'train_latest_only') and (split == 'train')):
                pcd_latest_path = os.path.join(latest_annotated_path, 'point_clouds', latest_lidar_name)
                labels_latest_path = os.path.join(latest_annotated_path, 'annotations', latest_lidar_name)

                pcd_latest_filenames = sorted(glob.glob(os.path.join(pcd_latest_path, '*.pcd')))
                labels_latest_filenames = sorted(glob.glob(os.path.join(labels_latest_path, '*.json')))

                pcd_dict[split] = pcd_latest_filenames
                labels_dict[split] = labels_latest_filenames
            
            else:
                pcd_path = os.path.join(split_path, 'point_clouds', lidar_name)
                pcd_labels_path = os.path.join(split_path, 'annotations', lidar_name)

                pcd_filenames = sorted(glob.glob(os.path.join(pcd_path, '*.pcd')))
                labels_filenames = sorted(glob.glob(os.path.join(pcd_labels_path, '*.json')))

                pcd_dict[split] = pcd_filenames
                labels_dict[split] = labels_filenames

        return pcd_dict, labels_dict
    
    def update_base_config(self):
        pass

    def construct_data_to_base_config(self):
        pass
    
    @staticmethod
    def clear_data_dir(dir):
        """
        this will clear the data in the kitti_format directory
        """
        dir = str(dir)
        for item in os.listdir(dir):
            item_path = os.path.join(dir, item)

            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

    def process(self):
        self.clear_data_dir()

        data_status = self.get_curr_data_status()
        self.datalog['operation_log'] = {
            'date': datetime.datetime.now().strftime('%Y_%m_%d_%H'),
            'operation': self.operation,
            'pre_data_config': {
                k: data_status[k] for k in data_status.keys()
            }
        }

        pcd_dict, labels_dict = self.split_data()
        


if __name__ == "__main__":
    pass
        
