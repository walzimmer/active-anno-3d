import numpy as np
import os
import datetime
import wandb
from pathlib import Path
import torch
import torch.nn as nn
import glob
from pcdet.models import build_network, model_fn_decorator
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets.tumtraf.tumtraf_converter import TUMTraf2KITTI, TUMTraf2KITTI_v2
from tools.utils.train_utils.optimization import build_optimizer, build_scheduler

from tools.proannoV2.utils.active_utils import train_model, build_dataloader_active, build_dataloader_train, query_samples_for_annotation
from tools.proannoV2.utils.infer_utils import model_inference, build_dataloader_inference, inference_post_processing
from tools.proannoV2.utils.infer_utils import prepare_data_current_annotation
from tools.proannoV2.utils.utils import TorchEncoder, NumpyEncoder, idx_to_label
from tools.proannoV2.inference import parse_config
from tools.proannoV2.utils.active_utils import prepare_data

if __name__ == "__main__":


    args, cfg = parse_config()

    op = 'inference'

    output_dir = Path(f'/ahmed/output/proannoV2/{op}')
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_inference_%s.txt' % (datetime.datetime.now().strftime('%Y_%m_%d_%H')))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # because we save the detections in the openlabel format
    openlabel_root_dir = Path('/'.join(cfg.DATA_CONFIG.ROOT_DIR.split('/')[:-2]) + '/OpenLABEL')
    dets_save_dir = openlabel_root_dir / 'currently_annotating' / 'detections'
    dets_save_dir.mkdir(parents=True, exist_ok=True)
    vis_save_dir = openlabel_root_dir / 'currently_annotating'/ 'visualizaions'
    vis_save_dir.mkdir(parents=True, exist_ok=True)

    cfg.MODEL.OPERATION = op
    # cfg.DATA_CONFIG.ROOT_DIR = cfg.DATA_CONFIG.ROOT_DIR + '/currently_annotating'
    if not os.path.isdir(cfg.DATA_CONFIG.ROOT_DIR):
        os.makedirs(cfg.DATA_CONFIG.ROOT_DIR, exist_ok=True)
        prepare_data_current_annotation(cfg, logger)

    filenames = [
        '1688626890_040199717',
        '1688626890_140238582',
        '1688626890_240150553',
        '1688626890_340243534'
    ]
    # ckpt = '/ahmed/output/proannoV2/train_active_all/testing/ckpts/checkpoint_2023_11_21_10.pth'
    ckpt = '/ahmed/tools/proannoV2_models/Oracle_3.pth'
    preds = model_inference(frame_ids=filenames,
                            cfg=cfg,
                            ckpt=ckpt,
                            logger=logger)
    
    preds_json = inference_post_processing(predictions=preds,
                                           frame_ids=filenames,
                                           openlabel_dir=openlabel_root_dir)

    # # to be captured by the flask app
    # print(preds_json)


    # from tools.utils.visual_utils.vis_tumtraf import VisualizationTUMTraf

    # vis = VisualizationTUMTraf()

    # root_dir = '/ahmed/data/tumtraf/proannoV2/OpenLABEL/currently_annotating'
    # pcd_path = os.path.join(root_dir, 'point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
    # labels_path = os.path.join(root_dir, 'labels_point_clouds/s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered')
    # dets_path = os.path.join(root_dir, 'detections')

    # frame_ids = [
    #         '1688626899_940260020',
    #         '1688626899_840264832',
    #         '1688626899_740300803',
    #         '1688626899_640174766'
    #     ]
    # frame_id = frame_ids[0]

    # vis.visualize_pcd_with_boxes(
    #     pcd_file_path=os.path.join(pcd_path, frame_id + '_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered' + '.pcd'),
    #     labels_file_path=os.path.join(labels_path, frame_id + '_s110_lidar_ouster_south_and_vehicle_lidar_robosense_registered' + '.json'),
    #     dets_file_path=os.path.join(dets_path, frame_id + '.json'),
    #     use_detections_in_base=None,
    #     view='bev',
    #     save_vis='/ahmed/data/tumtraf/proannoV2/OpenLABEL/currently_annotating/visualizaions',
    #     show_vis=False,
    #     return_vis=False)
