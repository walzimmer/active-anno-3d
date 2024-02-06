
import argparse
import datetime
import glob
import os

import cv2

from pathlib import Path
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import wandb

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader, build_active_dataloader
from pcdet.models import build_network, model_fn_decorator

from pcdet.utils import common_utils
from tools.utils.train_utils.optimization import build_optimizer, build_scheduler
from tools.utils.train_utils.train_utils import train_model

from tools.utils.train_utils.train_st_utils import train_model_st
from tools.utils.train_utils.train_active_utils import train_model_active

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"


def parse_config():

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/ahmed/tools/cfgs/active-tumtraf_models/pv_rcnn_active_crb.yaml',
                        help='specify the configuration for training')
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of training epochs')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloaders')
    parser.add_argument('--extra_tag', type=str, default='tunedAll', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=str, default='/ahmed/tools/pretrained_models/pv_rcnn_8369.pth', help='pretrained model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'bash'], default='none')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=True, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=10, help='save a checkpoint every ... epochs')
    parser.add_argument('--local_rank', type=int, default=1, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=60, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = args.cfg_file.split('/')[1:-1][-1]

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    cfg.MODEL.OPERATION = ''

    return args, cfg


def main():
    
    args, cfg = parse_config()

    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of GPUs'
        args.batch_size = args.batch_size // total_gpus

    if cfg.get('ACTIVE_TRAIN', None):
        args.epochs = cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS if args.epochs is None else args.epochs
    else:
        args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    ckpt_dir = output_dir / 'ckpt'

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg.get('ACTIVE_TRAIN', None):
        active_label_dir = output_dir / 'active_label'
        active_label_dir.mkdir(parents=True, exist_ok=True)

    # for all active methods, they share the same pre-trained model weights.
    backbone_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / 'backbone' / args.extra_tag / 'ckpt'
    backbone_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_train_%s.txt' % datetime.datetime.now().strftime('%Y_%m_%d_%H'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    print('********************** Start Logging **********************')
    logger.info('********************** Start Logging **********************')
    
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    
    log_config_to_file(cfg, logger=logger)

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    if cfg.LOCAL_RANK == 0:
        _train_ = '_active_train' if cfg.get('ACTIVE_TRAIN', None) else '_train'
        dataset_name = cfg.DATA_CONFIG._BASE_CONFIG_.split('/')[-1].split('.')[0].split('_')[0]

        if cfg.get('ACTIVE_TRAIN', None):
            e_pre = cfg.ACTIVE_TRAIN.PRE_TRAIN_EPOCH_NUMS # pre-training epochs
            e_int = cfg.ACTIVE_TRAIN.SELECT_LABEL_EPOCH_INTERVAL # epoch interval between selections
            pretrain_size = cfg.ACTIVE_TRAIN.PRE_TRAIN_SAMPLE_NUMS
            select_nums = cfg.ACTIVE_TRAIN.SELECT_NUMS

        time_now = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

        os.system('cp %s %s' % (args.cfg_file, output_dir))
        wandb.init(project=cfg.DATA_CONFIG._BASE_CONFIG_.split('/')[-1].split('.')[0] + _train_, 
                   entity="ahmedalaa-gh1",
                   tags=args.cfg_file.split('/')[-1].split('.')[0])
        
        if cfg.get('ACTIVE_TRAIN', None):
            run_name_elements = [args.extra_tag] + [dataset_name] + [cfg.TAG] + [e_pre] + [e_int] + [pretrain_size] + [select_nums] + [time_now]
        else:
            
            run_name_elements = [args.extra_tag] + [dataset_name] + [cfg.TAG] + [datetime.datetime.now().strftime('%Y-%m%d-%H%M%S')]
        
        run_name_elements = '_'.join([str(i) for i in run_name_elements])
        wandb.run.name = run_name_elements
        wandb.config.update(args)
        wandb.config.update(cfg)

    print('********************** Dataloader, Network, and Optimizer **********************')

    # ----------------- ACTIVE TRAINING -----------------

    if cfg.get('ACTIVE_TRAIN', None):
        labelled_set, unlabelled_set,\
        labelled_loader, unlabelled_loader, \
        labelled_sampler, unlabelled_sampler, = build_active_dataloader(dataset_cfg=cfg.DATA_CONFIG,
                                                                       class_names=cfg.CLASS_NAMES,
                                                                       batch_size=args.batch_size,
                                                                       root_path=None,
                                                                       dist=dist_train,
                                                                       workers=args.workers,
                                                                       logger=logger,
                                                                       training=True)
    # Normal Training or Self-training
    else:
        source_set, source_loader, source_sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_train,
            workers=args.workers,
            logger=logger,
            training=True,
            validation=False,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            total_epochs=args.epochs
        )
    

    if cfg.get('ACTIVE_TRAIN', None):
        model = build_network(model_cfg=cfg.MODEL,
                              num_class=len(cfg.CLASS_NAMES),
                              dataset=labelled_set)
    else:
        model = build_network(model_cfg=cfg.MODEL,
                              num_class=len(cfg.CLASS_NAMES),
                              dataset=source_set)
    
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model.cuda()

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    # load checkpoint if possible
    start_epoch = it = 0
    last_epoch = -1

    if args.pretrained_model is not None:
        model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger)
        for name, params in model.named_parameters():
            if 'backbone_3d' in name:
                params.requires_grad = False
    
    if args.ckpt is not None:
        it, start_epoch = model.load_params_with_optimizer(args.ckpt, to_cpu=dist_train, optimizer=optimizer, logger=logger)
        last_epoch = start_epoch + 1
    else:
        ckpt_list = glob.glob(str(ckpt_dir / '*checkpoint_epoch_*.pth'))
        if len(ckpt_list) > 0:
            ckpt_list.sort(key=os.path.getmtime)
            it, start_epoch = model.load_params_with_optimizer(
                ckpt_list[-1], to_cpu=dist_train, optimizer=optimizer, logger=logger
            )
            last_epoch = start_epoch + 1
    
    model.train()
    
    logger.info('device count: {}'.format(torch.cuda.device_count()))

    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])

    logger.info(model)

    # ----------------- ACTIVE TRAINING -----------------
    
    if cfg.get('ACTIVE_TRAIN', None):
        total_iters_each_epoch = len(labelled_loader) if not args.merge_all_iters_to_one_epoch \
        else len(labelled_loader) // args.epochs
    
    else:
        total_iters_each_epoch = len(source_loader) if not args.merge_all_iters_to_one_epoch \
        else len(source_loader) // args.epochs

    
    lr_scheduler, lr_warmup_scheduler = build_scheduler(
        optimizer, total_iters_each_epoch=total_iters_each_epoch, total_epochs=args.epochs,
        last_epoch=last_epoch, optim_cfg=cfg.OPTIMIZATION
    )
    train_func = train_model_active if cfg.get('ACTIVE_TRAIN', None) else train_model

    
    print("********************** Start Training **********************")
    logger.info('********************** Start Training %s/%s (%s) **********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    if cfg.get('ACTIVE_TRAIN', None):
        
        train_func(
            model=model,
            optimizer=optimizer,
            labelled_loader=labelled_loader,
            unlabelled_loader=unlabelled_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=lr_scheduler,
            optim_cfg=cfg.OPTIMIZATION,
            start_epoch=start_epoch,
            total_epochs=args.epochs,
            start_iter=it,
            rank=cfg.LOCAL_RANK,
            tb_log=tb_log,
            ckpt_save_dir=ckpt_dir,
            active_label_dir=active_label_dir,
            backbone_dir=backbone_dir,
            labelled_sampler=labelled_sampler,
            unlabelled_sampler=unlabelled_sampler,
            lr_warmup_scheduler=lr_warmup_scheduler,
            max_ckpt_save_num=args.max_ckpt_save_num,
            ckpt_save_interval=args.ckpt_save_interval,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            logger=logger,
            ema_model=None,
            dist_train=dist_train
        )

    
    else:
        train_func(
            model=model,
            optimizer=optimizer,
            train_loader=source_loader,
            model_func=model_fn_decorator(),
            lr_scheduler=lr_scheduler,
            optim_cfg=cfg.OPTIMIZATION,
            start_epoch=start_epoch,
            total_epochs=args.epochs,
            start_iter=it,
            rank=cfg.LOCAL_RANK,
            tb_log=tb_log,
            ckpt_save_dir=ckpt_dir,
            train_sampler=source_sampler,
            lr_warmup_scheduler=lr_warmup_scheduler,
            ckpt_save_interval=args.ckpt_save_interval,
            max_ckpt_save_num=args.max_ckpt_save_num,
            merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
            logger=logger
        )
    
    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    


if __name__ == "__main__":
    main()