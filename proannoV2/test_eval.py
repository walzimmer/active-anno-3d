import numpy as np
import os
import datetime
import wandb
from pathlib import Path
import torch
import torch.nn as nn

from pcdet.models import build_network, model_fn_decorator
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets.tumtraf.tumtraf_converter import TUMTraf2KITTI, TUMTraf2KITTI_v2
from tools.utils.train_utils.optimization import build_optimizer, build_scheduler

from tools.proannoV2.utils.eval_utils import prepare_data_eval
from tools.proannoV2.utils.active_utils import train_model, build_dataloader_active, build_dataloader_train, query_samples_for_annotation
from tools.proannoV2.utils.infer_utils import model_inference, build_dataloader_inference, inference_post_processing
from tools.proannoV2.utils.infer_utils import prepare_data_current_annotation
from tools.proannoV2.utils.utils import TorchEncoder, NumpyEncoder, idx_to_label
from tools.proannoV2.inference import parse_config
from tools.proannoV2.utils.active_utils import prepare_data


if __name__ == "__main__":

    args, cfg = parse_config()

    op = 'evaluation'

    output_dir = Path(f'/ahmed/output/proannoV2/{op}')
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_evaluation_%s.txt' % (datetime.datetime.now().strftime('%Y_%m_%d_%H')))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    cfg.MODEL.OPERATION = op
    cfg.DATA_CONFIG.ROOT_DIR = cfg.DATA_CONFIG.ROOT_DIR + '/currently_annotating'

    filenames = [
        '1688626890_040199717',
        '1688626890_140238582',
        '1688626890_240150553',
        '1688626890_340243534'
    ]

    prepare_data_eval(cfg, 
                      filenames, 
                      logger)