import tqdm
import wandb
import os
from pathlib import Path
import datetime
import pickle as pkl
import glob
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
from pcdet.datasets.tumtraf.tumtraf_utils import create_image_sets
from tools.utils.train_utils.optimization import build_optimizer, build_scheduler
from tools.utils.det_utils.detections import detections_to_openlabel, Detection
from tools.utils.train_utils.train_utils import model_state_to_cpu, save_checkpoint, checkpoint_state
from torch.nn.utils import clip_grad_norm_

from pcdet import query_strategies



def prepare_data(cfg, op, logger):
    print("*** Preparing Data for Active Learning ***")

    root_dir = '/ahmed/data/tumtraf/proannoV2/OpenLABEL/'

    latest_annotated_path = os.path.join(root_dir, 'latest_annotated')
    latest_lidar_name = os.listdir(os.path.join(latest_annotated_path, 'point_clouds'))[0]

    pcd_dict = {}
    labels_dict = {}

    splits = ['train', 'test']
    for split in splits:
        split_path = os.path.join(root_dir, split)
        lidar_name = os.listdir(os.path.join(split_path, 'point_clouds'))[0]

        if ((op == 'train_active_all' or op == 'train_all_only') and (split == 'train')):
            
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
        

        elif ((op == 'train_active_latest' or op == 'train_latest_only') and (split == 'train')):
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


    converter = TUMTraf2KITTI_v2(
    pcd_dict=pcd_dict,
    pcd_labels_dict=labels_dict,
    save_dir=cfg.DATA_CONFIG.ROOT_DIR,
    splits=['train', 'test'],
    logger=logger)

    converter.convert()
    print("*** Data converted to KITTI format ***")

    create_image_sets(
        data_root=cfg.DATA_CONFIG.ROOT_DIR,
        splits=['train', 'test']
    )

    create_tumtraf_infos(dataset_cfg=cfg.DATA_CONFIG,
                    class_names=cfg.CLASS_NAMES,
                    data_path=cfg.DATA_CONFIG.ROOT_DIR,
                    save_path=cfg.DATA_CONFIG.ROOT_DIR,
                    splits=['train', 'test'])
    
print("*** Data infos generated! ***")
    
def active_post_processing(cfg, selected):
    pass


def build_dataloader_active(cfg):

    labelled_set = TUMTrafDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=True,
        root_path=Path(cfg.DATA_CONFIG.ROOT_DIR)
    )
    unlabelled_set = TUMTrafDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=False,
        root_path=Path(cfg.DATA_CONFIG.ROOT_DIR)
    )

    sampler_labelled, sampler_unlabelled =  None, None

    labelled_dataloader = DataLoader(
        labelled_set, 
        batch_size=cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU, 
        pin_memory=True, 
        num_workers=4,
        shuffle=True, 
        collate_fn=labelled_set.collate_batch,
        drop_last=False, 
        sampler=sampler_labelled, 
        timeout=0
        )
    unlabelled_dataloader = DataLoader(
        unlabelled_set, 
        batch_size=1, 
        pin_memory=True, 
        num_workers=4,
        shuffle=True, 
        collate_fn=unlabelled_set.collate_batch,
        drop_last=False, 
        sampler=sampler_unlabelled, 
        timeout=0
        )
    

    

    return labelled_set, unlabelled_set, \
            labelled_dataloader, unlabelled_dataloader, \
            sampler_labelled, sampler_unlabelled



def build_dataloader_train(cfg):
    dataset = TUMTrafDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        training=True,
        root_path=cfg.DATA_CONFIG.ROOT_DIR
    )
    sampler = None
    dataloader = DataLoader(
        dataset, 
        batch_size=cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU, 
        pin_memory=True, 
        num_workers=4,
        shuffle=True, 
        collate_fn=dataset.collate_batch,
        drop_last=False, 
        sampler=sampler, 
        timeout=0
        )
    
    return dataset, dataloader, sampler



def query_samples_for_annotation(model,
                                 labelled_loader,
                                 unlabelled_loader,
                                 method,
                                 cur_epoch=None,
                                 active_label_dir=None):

    strategy = query_strategies.build_strategy(
        method=method,
        model=model,
        labelled_loader=labelled_loader,
        unlabelled_loader=unlabelled_loader,
        active_label_dir=active_label_dir,
        rank=0,
        cfg=cfg
    )
    selected_frames = strategy.query(leave_pbar=True,
                                    cur_epoch=cur_epoch,
                                    use_test_set=True,
                                    proanno=True)
    save_active_labels(selected_frames, active_label_dir)
    


def save_active_labels(selected_frames, active_label_dir):
    datetimenow = datetime.datetime.now().strftime('%Y_%m_%d_%H')
    file_name = active_label_dir / f'selected_frames_{cfg.ACTIVE_TRAIN.METHOD}_{cfg.ACTIVE_TRAIN.SELECT_NUMS}_{datetimenow}.pkl'

    with open(file_name, 'wb') as f:
        pkl.dump(selected_frames, f)



def train_one_epoch(model,
                    optimizer,
                    train_loader,
                    model_func,
                    lr_scheduler,
                    accumulated_iter,
                    optim_cfg,
                    tbar,
                    total_it_each_epoch,
                    dataloader_iter):
    
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)
    
    pbar = tqdm.tqdm(total=total_it_each_epoch, leave=None, desc='train', dynamic_ncols=True)
    model.train()

    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        
        optimizer.zero_grad()
        loss, tb_dict, disp_dict = model_func(model, batch)

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        lr_scheduler.step(accumulated_iter)

        accumulated_iter += 1
        log_accumulated_iter = accumulated_iter

        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        pbar.update()
        pbar.set_postfix(dict(total_it=log_accumulated_iter))
        tbar.set_postfix(disp_dict)
        tbar.refresh()

        wandb.log({'train/loss': loss, 'meta_data/learning_rate': cur_lr}, step=log_accumulated_iter)
        for key, val in tb_dict.items():
            wandb.log({'train/' + key: val}, step=log_accumulated_iter)

    return accumulated_iter



def train_model(model,
                optimizer,
                labelled_loader,
                model_func,
                lr_scheduler,
                optim_cfg,
                start_epoch,
                total_epochs,
                start_iter,
                ckpt_save_dir,
                ckpt_save_interval,
                labelled_sampler=None,
                unlabelled_sampler=None,
                logger=None):
    

    accumulated_iter = start_iter

    logger.info("***** Start Active Train Loop *****")
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True,leave=None) as tbar:
        
        selection_num = 0
        for cur_epoch in tbar:
            if labelled_sampler is not None:
                labelled_sampler.set_epoch(cur_epoch)
            
            total_it_each_epoch = len(labelled_loader)
            logger.info("currently {} iterations to learn per epoch".format(total_it_each_epoch))

            dataloader_iter = iter(labelled_loader)

            if labelled_sampler is not None:
                labelled_sampler.set_epoch(cur_epoch)

            cur_scheduler = lr_scheduler

            accumulated_iter = train_one_epoch(
                    model, 
                    optimizer, 
                    labelled_loader, 
                    model_func,
                    lr_scheduler=cur_scheduler,
                    accumulated_iter=accumulated_iter, 
                    optim_cfg=optim_cfg,
                    tbar=tbar,
                    total_it_each_epoch=total_it_each_epoch,
                    dataloader_iter=dataloader_iter
                )
            
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0:
                datetimenow = datetime.datetime.now().strftime('%Y_%m_%d_%H')
                ckpt_name = ckpt_save_dir / ('checkpoint_%s' % datetimenow)
                save_checkpoint(
                    checkpoint_state(model, optimizer, cur_epoch, accumulated_iter, lr_scheduler),
                    filename=ckpt_name)
    
    return cur_epoch, accumulated_iter
     


if __name__ == "__main__":
    pass