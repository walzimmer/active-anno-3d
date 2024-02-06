import os
import numpy as np
import argparse
import json
import pickle as pkl
import pickle as pkl
import datetime
import argparse
import os
import numpy as np
import torch
import tqdm
import json
from pathlib import Path
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.tumtraf.tumtraf_dataset import TUMTrafDataset
from pcdet.datasets.tumtraf.tumtraf_converter import TUMTraf2KITTI_v2
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets.tumtraf.tumtraf_utils import get_objects_and_names_from_label


def parse_config():
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--op', type=str, default='inference')
        parser.add_argument('--filenames', type=bool, default=False)
        parser.add_argument('--ckpt', type=str, default='Oracle_100.pth', help='ckpt used for inference or further training')
        parser.add_argument('--cfg_file', type=str, default='/ahmed/tools/cfgs/tumtraf_models/pv_rcnn.yaml')

        args = parser.parse_args()

        cfg_from_yaml_file(args.cfg_file, cfg)

        args.filenames = args.filenames.split(',')
        
        cfg.MODEL.OPERATION = args.op
        # the openlabel directory. we don't need the kitti format directory in evaluation
        # since we have the detections and annotations in OpenLABEL format already.
        cfg.DATA_CONFIG.ROOT_DIR = Path('/'.join(cfg.DATA_CONFIG.ROOT_DIR.split('/')[:-1]) + '/OpenLABEL') / '/currently_annotating'
        
        return args, cfg


def main(cfg, frame_ids, logger):

    curr_eval_dir = os.path.join(cfg.DATA_CONFIG_ROOT_DIR, 'evaluation')
    os.makedirs(curr_eval_dir, exist_ok=True)
    curr_lidar_name = os.listdir(os.path.join(cfg.DATA_CONFIG_ROOT_DIR, 'point_clouds'))[0]

    dataset = TUMTrafDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=cfg.DATA_CONFIG.ROOT_DIR,
        training=False,
        logger=logger
    )

    # frame_idx = dataset.sample_id_list.index(frame_id)
    frame_indices = [dataset.sample_id_list.index(frame_id) for frame_id in frame_ids if frame_id in dataset.sample_id_list]
    
    dataset.sample_id_list = [dataset.sample_id_list[idx] for idx in frame_indices]
    dataset.tumtraf_infos = [dataset.tumtraf_infos[idx] for idx in frame_indices]
    
    filtered_dets = []
    batch_dict = {'frame_id': []}
    for idx, frame_id in enumerate(frame_ids):
        detections_file = os.path.join(cfg.DATA_CONFIG_ROOT_DIR, 'detections', f'detections_{frame_id}.pkl')
        with open(detections_file, 'rb') as det_file:
            data = pkl.load(det_file)
            filtered_dets.append(data['filtered'])
            batch_dict['frame_id'].append(frame_id) 

        labels_file = os.path.join(cfg.DATA_CONFIG_ROOT_DIR, 'annotations', curr_lidar_name, f'{frame_id}.json')
        gt_boxes, gt_names = get_objects_and_names_from_label(label_file=labels_file)
        dataset.tumtraf_infos[idx]['annos'] = {
            'name': gt_names,
            'boxes_3d': gt_boxes[:, :7]
        }
    pred_dicts = dataset.generate_prediction_dicts(batch_dict=batch_dict,
                                                   pred_dicts=filtered_dets,
                                                   class_names=cfg.CLASS_NAMES)
    
    unique_labels = set(label for prediction in pred_dicts for label in np.unique(prediction['name']))

    result_str, result_dict = dataset.evaluation(
        pred_dicts, 
        unique_labels,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=curr_eval_dir
    )
    


if __name__ == "__main__":
    
    args, cfg = parse_config()

    output_dir = Path(f'/ahmed/output/proannoV2/{args.op}')
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_inference_%s.txt' % (datetime.datetime.now().strftime('%Y_%m_%d_%H')))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)


    main(cfg, args.filenames, logger)


