
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

from tools.proannoV2.utils.active_utils import train_model, build_dataloader_active, build_dataloader_train, query_samples_for_annotation
from tools.proannoV2.utils.infer_utils import model_inference, build_dataloader_inference, inference_post_processing
from tools.proannoV2.utils.utils import TorchEncoder, NumpyEncoder, prepare_data, idx_to_label

if __name__ == "__main__":

    op = 'train_active_all'
    output_dir = Path(f'/ahmed/output/proannoV2/{op}')
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_save_dir = output_dir / 'testing' / 'ckpts'
    ckpt_save_dir.mkdir(parents=True, exist_ok=True)
    active_label_dir = output_dir / 'testing' / 'active_selection'
    active_label_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_%s_%s.txt' % (op, datetime.datetime.now().strftime('%Y_%m_%d_%H')))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)
    
    cfg_file = '/ahmed/tools/cfgs/proannoV2/pv_rcnn_active.yaml'
    cfg = cfg_from_yaml_file(cfg_file, cfg)
    cfg.MODEL.OPERATION = op

    # wandb.init(project= 'proannoV2_' + cfg.DATA_CONFIG._BASE_CONFIG_.split('/')[-1].split('.')[0] + '_' + op,
    #            entity="ahmedalaa-gh1",
    #            tags=cfg_file.split('/')[-1].split('.')[0])

    # prepare_data(cfg=cfg, op=op, logger=logger)

    labelled_set, unlabelled_set,\
    labelled_loader, unlabelled_loader,\
    _,_ = build_dataloader_active(cfg)
    
    # ckpt = 'Oracle_80.pth'
    # epochs = 1
    # pretrained_model = Path('/ahmed/tools/pvrcnn_oracle_ckpts/' + ckpt)

    pretrained_model = '/ahmed/output/proannoV2/train_active_all/testing/ckpts/checkpoint_2023_11_21_10.pth'
    model = build_network(model_cfg=cfg.MODEL,
                        num_class=len(cfg.CLASS_NAMES),
                        dataset=labelled_set)
    
    model.load_params_from_file(filename=pretrained_model, to_cpu=False, logger=logger)
    model.cuda()


    # optimizer = build_optimizer(model, cfg.OPTIMIZATION)
                
    model.train()

    logger.info('device count: {}'.format(torch.cuda.device_count()))
    logger.info(model)

    # total_iters_each_epoch = len(labelled_loader)
    # start_epoch = it = 0
    # last_epoch = -1

    # lr_scheduler, lr_warmup_scheduler = build_scheduler(
    #     optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=epochs,
    #     last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    #     )
        
    # train_func = train_model
    # cur_epoch, accumulated_iter =train_func(
    #         model=model,
    #         optimizer=optimizer,
    #         labelled_loader=labelled_loader,
    #         model_func=model_fn_decorator(),
    #         lr_scheduler=lr_scheduler,
    #         optim_cfg=cfg.OPTIMIZATION,
    #         start_epoch=start_epoch,
    #         total_epochs=epochs,
    #         start_iter=it,
    #         ckpt_save_dir=ckpt_save_dir,
    #         ckpt_save_interval=1,
    #         labelled_sampler=None,
    #         unlabelled_sampler=None,
    #         logger=logger
    # )

    # query_samples_for_annotation(
    #         model=model,
    #         labelled_loader=labelled_loader,
    #         unlabelled_loader=unlabelled_loader,
    #         logger=logger,
    #         method='tcrb',
    #         cur_epoch=0,
    #         active_label_dir=active_label_dir
    #     )
    
    print("rabna kareem")
