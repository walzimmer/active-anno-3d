import os
import numpy as np
import argparse
import json
import pickle as pkl
import datetime
import glob
import wandb
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pcdet.models import build_network, model_fn_decorator
from pcdet.datasets.tumtraf.tumtraf_dataset import TUMTrafDataset
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.tumtraf.tumtraf_converter import TUMTraf2KITTI, TUMTraf2KITTI_v2
from pcdet.datasets.tumtraf.tumtraf_dataset import create_tumtraf_infos

from tools.utils.train_utils.optimization import build_optimizer, build_scheduler
from tools.proannoV2.utils.utils import TorchEncoder, NumpyEncoder
from tools.proannoV2.utils.active_utils import build_dataloader_active, build_dataloader_train, train_model, query_samples_for_annotation
from tools.proannoV2.utils.active_utils import prepare_data


def parse_config():
        parser = argparse.ArgumentParser(description='arg parser')

        parser.add_argument('--op', type=str, default='train_active_all')
        parser.add_argument('--query', type=str, default='crb')
        parser.add_argument('--n_select', type=int, default=20)
        parser.add_argument('--epochs', type=int, default=5)
        parser.add_argument('--ckpt', type=str, default='Oracle_1.pth', help='ckpt used for inference or further training')
        parser.add_argument('--cfg_file', type=str, default='/ahmed/tools/cfgs/proannoV2/pv_rcnn_active.yaml')

        args = parser.parse_args()

        cfg_from_yaml_file(args.cfg_file, cfg)

        # cfg.MODEL.OPERATION = args.op
        cfg.ACTIVE_TRAIN.METHOD = args.query
        cfg.ACTIVE_TRAIN.SELECT_NUMS = args.n_select
        cfg.ACTIVE_TRAIN.TOTAL_BUDGET_NUMS = args.n_select

        cfg.MODEL.OPERATION = 'query_only'

        return args, cfg


def post_processes_selected_frames(active_label_dir):
     pass

if __name__ == "__main__":
    
    args, cfg = parse_config()

    output_dir = Path(f'/ahmed/output/proannoV2/{args.op}')
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dir = Path('/ahmed/data/tumtraf/proannoV2/test_KITTI_format')
    test_dir.mkdir(parents=True, exist_ok=True)
    active_label_dir = test_dir / 'active_selection'
    active_label_dir.mkdir(parents=True, exist_ok=True)
    

    log_file = output_dir / ('log_%s_%s.txt' % (args.op, datetime.datetime.now().strftime('%Y_%m_%d_%H')))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    op = 'query_only'
    cfg.DATA_CONFIG.ROOT_DIR = '/ahmed/data/tumtraf/proannoV2/test_KITTI_format'

#     prepare_data(cfg=cfg, op=op, logger=logger)

    labelled_set, unlabelled_set,\
        labelled_loader, unlabelled_loader,\
        _, _ = build_dataloader_active(cfg)
    
    pretrained_model = Path('/ahmed/tools/proannoV2_models/' + args.ckpt)

    model = build_network(model_cfg=cfg.MODEL,
                          num_class=len(cfg.CLASS_NAMES),
                          dataset=labelled_set)
    
    model.load_params_from_file(filename=pretrained_model, to_cpu=False, logger=logger)
    model.cuda()
    logger.info('device count: {}'.format(torch.cuda.device_count()))
    logger.info(model)

    query_samples_for_annotation(
          model=model,
          labelled_loader=labelled_loader,
          unlabelled_loader=unlabelled_loader,
          method=cfg.ACTIVE_TRAIN.METHOD,
          cur_epoch=0,
          active_label_dir=active_label_dir
        )
        