
import torch
import os
import glob
import tqdm
import wandb
import numpy as np
import torch.distributed as dist
import pickle as pkl
import re
import ast

from itertools import accumulate
from pcdet.config import cfg
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils, commu_utils, memory_ensemble_utils
from tools.utils.train_utils.train_utils import model_state_to_cpu, save_checkpoint, checkpoint_state, resume_dataset

from pcdet.datasets import build_active_dataloader
from .. import query_strategies


ACTIVE_LABELS = {}
NEW_ACTIVE_LABELS = {}


def check_already_exist_active_label(active_label_dir, start_epoch):
    """
    if we continue training, use this to directly
    load pseudo labels from exsiting result pkl

    if exsit, load latest result pkl to PSEUDO LABEL
    otherwise, return false and

    Args:
        active_label_dir: dir to save pseudo label results pkls.
        start_epoch: start epoc
    Returns:

    """
    # support init ps_label given by cfg
    if start_epoch == 0 and cfg.ACTIVE_TRAIN.get('INITIAL_SELECTION', None):
        if os.path.exists(cfg.ACTIVE_TRAIN.INITIAL_SELECTION):
            init_active_label = pkl.load(open(cfg.ACTIVE_TRAIN.INITIAL_SELECTION, 'rb'))
            ACTIVE_LABELS.update(init_active_label)

            if cfg.LOCAL_RANK == 0:
                ps_path = os.path.join(active_label_dir, "ps_label_e0.pkl")
                with open(ps_path, 'wb') as f:
                    pkl.dump(ACTIVE_LABELS, f)

            return cfg.ACTIVE_TRAIN.INITIAL_SELECTION

    active_label_list = glob.glob(os.path.join(active_label_dir, 'active_label_e*.pkl'))
    if len(active_label_list) == 0:
        return

    active_label_list.sort(key=os.path.getmtime, reverse=True)
    for cur_pkl in active_label_list:
        num_epoch = re.findall('ps_label_e(.*).pkl', cur_pkl)
        assert len(num_epoch) == 1

        # load pseudo label and return
        if int(num_epoch[0]) <= start_epoch:
            latest_ps_label = pkl.load(open(cur_pkl, 'rb'))
            ACTIVE_LABELS.update(latest_ps_label)
            return cur_pkl

    return None


def save_active_label_epoch(model, val_loader, rank, leave_pbar, active_label_dir, cur_epoch):
    """
    Generate active with given model.

    Args:
        model: model to predict result for active label
        val_loader: data_loader to predict labels
        rank: process rank
        leave_pbar: tqdm bar controller
        active_label_dir: dir to save active label
        cur_epoch
    """
    val_dataloader_iter = iter(val_loader)
    total_it_each_epoch = len(val_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                         desc='generate_active_e%d' % cur_epoch, dynamic_ncols=True)

    # pos_ps_meter = common_utils.AverageMeter()
    # ign_ps_meter = common_utils.AverageMeter()

    model.eval()

    for cur_it in range(total_it_each_epoch):
        try:
            unlabelled_batch = next(val_dataloader_iter)
        except StopIteration:
            unlabelled_dataloader_iter = iter(val_loader)
            unlabelled_batch = next(unlabelled_dataloader_iter)

        # generate gt_boxes for unlabelled_batch and update model weights
        with torch.no_grad():
            load_data_to_gpu(unlabelled_batch)
            pred_dicts, recall_dicts = model(unlabelled_batch)

        # select and save active labels
        # random

        save_active_label_batch(
            unlabelled_batch, pred_dicts=pred_dicts
        )

        # log to console and tensorboard
        # pos_ps_meter.update(pos_ps_batch)
        # ign_ps_meter.update(ign_ps_batch)
        # disp_dict = {'pos_ps_box': "{:.3f}({:.3f})".format(pos_ps_meter.val, pos_ps_meter.avg),
        #              'ign_ps_box': "{:.3f}({:.3f})".format(ign_ps_meter.val, ign_ps_meter.avg)}

        if rank == 0:
            pbar.update()
            # pbar.set_postfix(disp_dict)
            pbar.refresh()

    if rank == 0:
        pbar.close()

    gather_and_dump_pseudo_label_result(rank, active_label_dir, cur_epoch)




def save_active_label_batch(input_dict,
                            pred_dicts=None,
                            need_update=True):
    """
    Save pseudo label for given batch.
    If model is given, use model to inference pred_dicts,
    otherwise, directly use given pred_dicts.

    Args:
        input_dict: batch data read from dataloader
        pred_dicts: Dict if not given model.
            predict results to be generated pseudo label and saved
        need_update: Bool.
            If set to true, use consistency matching to update pseudo label
    """
    pos_ps_meter = common_utils.AverageMeter()
    ign_ps_meter = common_utils.AverageMeter()

    batch_size = len(pred_dicts)

    for b_idx in range(batch_size):
        # pred_cls_scores = pred_iou_scores = None
        # if 'pred_boxes' in pred_dicts[b_idx]:
        #     # Exist predicted boxes passing self-training score threshold
        #     pred_boxes = pred_dicts[b_idx]['pred_boxes'].detach().cpu().numpy()
        #     pred_labels = pred_dicts[b_idx]['pred_labels'].detach().cpu().numpy()
        #     pred_scores = pred_dicts[b_idx]['pred_scores'].detach().cpu().numpy()
        #     if 'pred_cls_scores' in pred_dicts[b_idx]:
        #         pred_cls_scores = pred_dicts[b_idx]['pred_cls_scores'].detach().cpu().numpy()
        #     if 'pred_iou_scores' in pred_dicts[b_idx]:
        #         pred_iou_scores = pred_dicts[b_idx]['pred_iou_scores'].detach().cpu().numpy()
        #
        #     # remove boxes under negative threshold
        #     if cfg.SELF_TRAIN.get('NEG_THRESH', None):
        #         labels_remove_scores = np.array(cfg.SELF_TRAIN.NEG_THRESH)[pred_labels - 1]
        #         remain_mask = pred_scores >= labels_remove_scores
        #         pred_labels = pred_labels[remain_mask]
        #         pred_scores = pred_scores[remain_mask]
        #         pred_boxes = pred_boxes[remain_mask]
        #         if 'pred_cls_scores' in pred_dicts[b_idx]:
        #             pred_cls_scores = pred_cls_scores[remain_mask]
        #         if 'pred_iou_scores' in pred_dicts[b_idx]:
        #             pred_iou_scores = pred_iou_scores[remain_mask]
        #
        #     labels_ignore_scores = np.array(cfg.SELF_TRAIN.SCORE_THRESH)[pred_labels - 1]
        #     ignore_mask = pred_scores < labels_ignore_scores
        #     pred_labels[ignore_mask] = -1
        #
        #     gt_box = np.concatenate((pred_boxes,
        #                              pred_labels.reshape(-1, 1),
        #                              pred_scores.reshape(-1, 1)), axis=1)
        #
        # else:
        #     # no predicted boxes passes self-training score threshold
        #     gt_box = np.zeros((0, 9), dtype=np.float32)
        #
        # gt_infos = {
        #     'gt_boxes': gt_box,
        #     'cls_scores': pred_cls_scores,
        #     'iou_scores': pred_iou_scores,
        #     'memory_counter': np.zeros(gt_box.shape[0])
        # }

        # record pseudo label to pseudo label dict
        # if need_update:
        #     ensemble_func = getattr(memory_ensemble_utils, cfg.SELF_TRAIN.MEMORY_ENSEMBLE.NAME)
        #     gt_infos = ensemble_func(ACTIVE_LABELS[input_dict['frame_id'][b_idx]],
        #                              gt_infos, cfg.SELF_TRAIN.MEMORY_ENSEMBLE)

        # if gt_infos['gt_boxes'].shape[0] > 0:
        #     ign_ps_meter.update((gt_infos['gt_boxes'][:, 7] < 0).sum())
        # else:
        #     ign_ps_meter.update(0)
        # pos_ps_meter.update(gt_infos['gt_boxes'].shape[0] - ign_ps_meter.val)

        NEW_ACTIVE_LABELS[input_dict['frame_id'][b_idx]] = input_dict['gt_boxes']

    return



def select_active_labels(model,
                         labelled_loader,
                         unlabelled_loader,
                         rank,
                         logger,
                         method,
                         leave_pbar=True,
                         cur_epoch=None,
                         dist_train=False,
                         active_label_dir=None,
                         accumulated_iter=None):


    strategy = query_strategies.build_strategy(method=method, 
                                               model=model,
                                               labelled_loader=labelled_loader,
                                               unlabelled_loader=unlabelled_loader,
                                               rank=rank,
                                               active_label_dir=active_label_dir,
                                               cfg=cfg)
    

    if os.path.isfile(os.path.join(active_label_dir, 'selected_frames_epoch_{}_rank_{}.pkl'.format(cur_epoch, rank))):
            print('found {} epoch saved selections...start resuming...'.format(cur_epoch))
            selected_frames_list = sorted([str(active_label_dir / i) for i in glob.glob(str(active_label_dir / 'selected_frames_epoch_*.pkl'))])

            labelled_loader, unlabelled_loader, selected_frames = resume_dataset(labelled_loader=labelled_loader,
                                                                                unlabelled_loader=unlabelled_loader,
                                                                                selected_frames_list=selected_frames_list,
                                                                                dist_train=dist_train,
                                                                                logger=logger,
                                                                                cfg=cfg)

            # return labelled_loader, unlabelled_loader
            return selected_frames, strategy, labelled_loader, unlabelled_loader
                
    else:
        if method == 'crb' or method == 'tcrb':
            selected_frames, grad_embeddings = strategy.query(leave_pbar, cur_epoch)
            strategy.save_active_labels(selected_frames=selected_frames, grad_embeddings=grad_embeddings, cur_epoch=cur_epoch)
            strategy.update_dashboard(cur_epoch=cur_epoch, accumulated_iter=accumulated_iter)
        
        elif method == 'wcrb':
            class_weights = get_latest_class_weights(active_label_dir=active_label_dir)
            selected_frames, grad_embeddings = strategy.query(leave_pbar, cur_epoch, class_weights=class_weights)
            strategy.save_active_labels(selected_frames=selected_frames, grad_embeddings=None, cur_epoch=cur_epoch)
            strategy.update_dashboard(cur_epoch=cur_epoch, accumulated_iter=accumulated_iter)
            update_class_weights(active_label_dir=active_label_dir)

        else:
            selected_frames = strategy.query(leave_pbar, cur_epoch, accumulated_iter)
            strategy.save_active_labels(selected_frames=selected_frames, cur_epoch=cur_epoch)
            strategy.update_dashboard(cur_epoch=cur_epoch, accumulated_iter=accumulated_iter)

        

    return selected_frames, strategy, labelled_loader, unlabelled_loader

    


def prepare_network_and_dataset(model,
                                strategy,
                                labelled_loader,
                                unlabelled_loader,
                                selected_frames,
                                logger,
                                ckpt_dir,
                                backbone_dir,
                                dist_train=False):
    """
    Prepares the network and dataset for active learning by updating the model weights and creating new data loaders.

    Parameters:
    - model (torch.nn.Module): The model to be updated.
    - strategy (ActiveLearningStrategy): The strategy used for active learning.
    - labelled_loader (DataLoader): DataLoader for labelled data.
    - unlabelled_loader (DataLoader): DataLoader for unlabelled data.
    - selected_frames (list): List of selected frame IDs.
    - logger (Logger): Logger for logging information.
    - ckpt_dir (Path or str): Directory containing model checkpoints.
    - backbone_dir (Path or str): Directory containing backbone model checkpoints.
    - dist_train (bool): Flag for distributed training mode.

    Returns:
    - tuple: A tuple of two DataLoader objects (labelled_loader, unlabelled_loader).
    """
    
    network = cfg.ACTIVE_TRAIN.NETWORK_UPDATE.NETWORK
    data_set = cfg.ACTIVE_TRAIN.NETWORK_UPDATE.DATASET

    if network == 'INITIAL_NETWORK' and data_set == 'ALL_AVAILABLE_DATA':
        
        # *************************
        # ********  MODEL  ********
        # *************************

        logger.info("**finished selection: reload initial weights of the model")
        backbone_init_ckpt = torch.load(str(backbone_dir / 'init_checkpoint.pth'))
        model.load_state_dict(backbone_init_ckpt, strict=cfg.ACTIVE_TRAIN.METHOD!='llal')


        # ************************
        # ********  DATA  ********
        # ************************

        if cfg.DATA_CONFIG.DATASET == 'TUMTrafDataset':
            selected_id_list, selected_infos = list(strategy.labelled_set.sample_id_list), list(strategy.labelled_set.tumtraf_infos)
            unselected_id_list, unselected_infos = list(strategy.unlabelled_set.sample_id_list), list(strategy.unlabelled_set.tumtraf_infos)
        
        elif cfg.DATA_CONFIG.DATASET == 'KittiDataset':
            selected_id_list, selected_infos = list(strategy.labelled_set.sample_id_list),  list(strategy.labelled_set.kitti_infos)
            unselected_id_list, unselected_infos = list(strategy.unlabelled_set.sample_id_list), list(strategy.unlabelled_set.kitti_infos)
        
        for i in range(len(strategy.pairs)):
            if strategy.pairs[i][0] in selected_frames:
                selected_id_list.append(strategy.pairs[i][0])
                selected_infos.append(strategy.pairs[i][1])
                unselected_id_list.remove(strategy.pairs[i][0])
                unselected_infos.remove(strategy.pairs[i][1])
        
        selected_id_list = tuple(selected_id_list)
        selected_infos = tuple(selected_infos)
        unselected_id_list = tuple(unselected_id_list)
        unselected_infos = tuple(unselected_infos)

        active_training = [selected_id_list, selected_infos, unselected_id_list, unselected_infos]

        # Create new dataloaders
        batch_size = unlabelled_loader.batch_size
        print("Batch_size of a single loader: %d" % (batch_size))
        workers = unlabelled_loader.num_workers

        # delete old sets and loaders, save space for new sets and loaders
        del labelled_loader.dataset, unlabelled_loader.dataset, labelled_loader, unlabelled_loader

        labelled_set, unlabelled_set, \
        labelled_loader, unlabelled_loader, \
        sampler_labelled, sampler_unlabelled = build_active_dataloader(
            cfg.DATA_CONFIG,
            cfg.CLASS_NAMES,
            batch_size,
            dist_train,
            workers=workers,
            logger=logger,
            training=True,
            active_training=active_training
        )
        return labelled_loader, unlabelled_loader

    elif network == 'INITIAL_NETWORK' and data_set == 'CURRENT_SELECTED_DATA':

        # *************************
        # ********  MODEL  ********
        # *************************

        logger.info("**finished selection: reload initial weights of the model")
        backbone_init_ckpt = torch.load(str(backbone_dir / 'init_checkpoint.pth'))
        model.load_state_dict(backbone_init_ckpt, strict=cfg.ACTIVE_TRAIN.METHOD!='llal')


        # ************************
        # ********  DATA  ********
        # ************************

        selected_id_list, selected_infos = [], []
        if cfg.DATA_CONFIG.DATASET == 'TUMTrafDataset':

            unselected_id_list, unselected_infos = list(strategy.unlabelled_set.sample_id_list), list(strategy.unlabelled_set.tumtraf_infos)
        

        for i in range(len(strategy.pairs)):
            if strategy.pairs[i][0] in selected_frames:
                selected_id_list.append(strategy.pairs[i][0])
                selected_infos.append(strategy.pairs[i][1])
                unselected_id_list.remove(strategy.pairs[i][0])
                unselected_infos.remove(strategy.pairs[i][1])

        selected_id_list = tuple(selected_id_list)
        selected_infos = tuple(selected_infos)
        unselected_id_list = tuple(unselected_id_list)
        unselected_infos = tuple(unselected_infos)
        
        active_training = [selected_id_list, selected_infos, unselected_id_list, unselected_infos]

        # Create new dataloaders
        batch_size = unlabelled_loader.batch_size
        print("Batch_size of a single loader: %d" % (batch_size))
        workers = unlabelled_loader.num_workers

        # delete old sets and loaders, save space for new sets and loaders
        del labelled_loader.dataset, unlabelled_loader.dataset, labelled_loader, unlabelled_loader

        labelled_set, unlabelled_set, \
        labelled_loader, unlabelled_loader, \
        sampler_labelled, sampler_unlabelled = build_active_dataloader(
            cfg.DATA_CONFIG,
            cfg.CLASS_NAMES,
            batch_size,
            dist_train,
            workers=workers,
            logger=logger,
            training=True,
            active_training=active_training
        )

        # check that the dataloader is correct --> loading the selected_frames only
        set_frames = [i.item() for batch in labelled_loader for i in batch['frame_id']]
        assert all(frame in selected_frames for frame in set_frames) and all(frame in set_frames for frame in selected_frames)

        return labelled_loader, unlabelled_loader

    elif network == 'LATEST_NETWORK' and data_set == 'ALL_AVAILABLE_DATA':
        
        # *************************
        # ********  MODEL  ********
        # *************************

        logger.info("**finished selection: reload latest weights of the model")
        ckpt_list = [i for i in glob.glob(str(ckpt_dir / 'checkpoint_epoch_*.pth'))]

        if len(ckpt_list) < 1:
            backbone_ckpt_list = [i for i in glob.glob(str(backbone_dir / 'checkpoint_epoch_*.pth'))]
            backbone_ckpt_list.sort(key=os.path.getmtime)
            ckpt_last_epoch = torch.load(str(backbone_dir / backbone_ckpt_list[-1]))
            epoch_id = str(backbone_dir / backbone_ckpt_list[-1]).split('.')[0].split('_')[-1]

            checksums = {}
            for i in backbone_ckpt_list:
                ckpt_last_epoch = torch.load(str(backbone_dir / i))
                model.load_state_dict(ckpt_last_epoch['model_state'], strict=cfg.ACTIVE_TRAIN.METHOD!='llal')
                epoch_id_i = i.split('.')[0].split('_')[-1]
                cs_val = get_model_checksum(model)
                checksums[f'epoch_{epoch_id_i}'] = cs_val
        else:
            ckpt_last_epoch = torch.load(str(ckpt_dir / ckpt_list[-1]))
            ckpt_list.sort(key=os.path.getmtime)
            epoch_id = str(ckpt_dir / ckpt_list[-1]).split('.')[0].split('_')[-1]

            checksums = {}
            for i in ckpt_list:
                ckpt_last_epoch = torch.load(str(ckpt_dir / i))
                model.load_state_dict(ckpt_last_epoch['model_state'], strict=cfg.ACTIVE_TRAIN.METHOD!='llal')
                epoch_id_i = i.split('.')[0].split('_')[-1]
                cs_val = get_model_checksum(model)
                checksums[f'epoch_{epoch_id_i}'] = cs_val

        if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(ckpt_last_epoch['model_state'], strict=cfg.ACTIVE_TRAIN.METHOD!='llal')

        else:
            model.load_state_dict(ckpt_last_epoch['model_state'], strict=cfg.ACTIVE_TRAIN.METHOD!='llal')

        # Check that we're loading the correct model
        checksum_val = get_model_checksum(model)
        assert checksum_val == checksums[f'epoch_{epoch_id}'], "!!! loading the wrong model model !!!"


        # ************************
        # ********  DATA  ********
        # ************************

        if cfg.DATA_CONFIG.DATASET == 'TUMTrafDataset':
            selected_id_list, selected_infos = list(strategy.labelled_set.sample_id_list), list(strategy.labelled_set.tumtraf_infos)
            unselected_id_list, unselected_infos = list(strategy.unlabelled_set.sample_id_list), list(strategy.unlabelled_set.tumtraf_infos)
        
        elif cfg.DATA_CONFIG.DATASET == 'KittiDataset':
            selected_id_list, selected_infos = list(strategy.labelled_set.sample_id_list),  list(strategy.labelled_set.kitti_infos)
            unselected_id_list, unselected_infos = list(strategy.unlabelled_set.sample_id_list), list(strategy.unlabelled_set.kitti_infos)
        
        for i in range(len(strategy.pairs)):
            if strategy.pairs[i][0] in selected_frames:
                selected_id_list.append(strategy.pairs[i][0])
                selected_infos.append(strategy.pairs[i][1])
                unselected_id_list.remove(strategy.pairs[i][0])
                unselected_infos.remove(strategy.pairs[i][1])
        
        selected_id_list = tuple(selected_id_list)
        selected_infos = tuple(selected_infos)
        unselected_id_list = tuple(unselected_id_list)
        unselected_infos = tuple(unselected_infos)

        active_training = [selected_id_list, selected_infos, unselected_id_list, unselected_infos]

        # Create new dataloaders
        batch_size = unlabelled_loader.batch_size
        print("Batch_size of a single loader: %d" % (batch_size))
        workers = unlabelled_loader.num_workers

        # delete old sets and loaders, save space for new sets and loaders
        del labelled_loader.dataset, unlabelled_loader.dataset, labelled_loader, unlabelled_loader

        labelled_set, unlabelled_set, \
        labelled_loader, unlabelled_loader, \
        sampler_labelled, sampler_unlabelled = build_active_dataloader(
            cfg.DATA_CONFIG,
            cfg.CLASS_NAMES,
            batch_size,
            dist_train,
            workers=workers,
            logger=logger,
            training=True,
            active_training=active_training
        )

        # check that the dataloader is correct --> loading the selected_frames only
        set_frames = [i.item() for batch in labelled_loader for i in batch['frame_id']]
        assert all(frame in set_frames for frame in selected_frames)

        return labelled_loader, unlabelled_loader

    elif network == 'LATEST_NETWORK' and data_set == 'CURRENT_SELECTED_DATA':

        # *************************
        # ********  MODEL  ********
        # *************************

        logger.info("**finished selection: reload latest weights of the model")
        ckpt_list = [i for i in glob.glob(str(ckpt_dir / 'checkpoint_epoch_*.pth'))]

        if len(ckpt_list) < 1:
            backbone_ckpt_list = [i for i in glob.glob(str(backbone_dir / 'checkpoint_epoch_*.pth'))]
            backbone_ckpt_list.sort(key=os.path.getmtime)
            ckpt_last_epoch = torch.load(str(backbone_dir / backbone_ckpt_list[-1]))
            epoch_id = str(backbone_dir / backbone_ckpt_list[-1]).split('.')[0].split('_')[-1]

            checksums = {}
            for i in backbone_ckpt_list:
                ckpt_last_epoch = torch.load(str(backbone_dir / i))
                model.load_state_dict(ckpt_last_epoch['model_state'], strict=cfg.ACTIVE_TRAIN.METHOD!='llal')
                epoch_id_i = i.split('.')[0].split('_')[-1]
                cs_val = get_model_checksum(model)
                checksums[f'epoch_{epoch_id_i}'] = cs_val
        else:
            ckpt_last_epoch = torch.load(str(ckpt_dir / ckpt_list[-1]))
            ckpt_list.sort(key=os.path.getmtime)
            epoch_id = str(ckpt_dir / ckpt_list[-1]).split('.')[0].split('_')[-1]

            checksums = {}
            for i in ckpt_list:
                ckpt_last_epoch = torch.load(str(ckpt_dir / i))
                model.load_state_dict(ckpt_last_epoch['model_state'], strict=cfg.ACTIVE_TRAIN.METHOD!='llal')
                epoch_id_i = i.split('.')[0].split('_')[-1]
                cs_val = get_model_checksum(model)
                checksums[f'epoch_{epoch_id_i}'] = cs_val

        if isinstance(model, torch.nn.parallel.DistributedDataParallel) or isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(ckpt_last_epoch['model_state'], strict=cfg.ACTIVE_TRAIN.METHOD!='llal')

        else:
            model.load_state_dict(ckpt_last_epoch['model_state'], strict=cfg.ACTIVE_TRAIN.METHOD!='llal')

        # Check that we're loading the correct model
        checksum_val = get_model_checksum(model)
        assert checksum_val == checksums[f'epoch_{epoch_id}'], "!!! loading the wrong model model !!!"
        

        # ************************
        # ********  DATA  ********
        # ************************

        selected_id_list, selected_infos = [], []
        if cfg.DATA_CONFIG.DATASET == 'TUMTrafDataset':
            unselected_id_list, unselected_infos = list(strategy.unlabelled_set.sample_id_list), list(strategy.unlabelled_set.tumtraf_infos)
        
        elif cfg.DATA_CONFIG.DATASET == 'KittiDataset':
            unselected_id_list, unselected_infos = list(strategy.unlabelled_set.sample_id_list), list(strategy.unlabelled_set.kitti_infos)

        for i in range(len(strategy.pairs)):
            if strategy.pairs[i][0] in selected_frames:
                selected_id_list.append(strategy.pairs[i][0])
                selected_infos.append(strategy.pairs[i][1])
                unselected_id_list.remove(strategy.pairs[i][0])
                unselected_infos.remove(strategy.pairs[i][1])

        selected_id_list = tuple(selected_id_list)
        selected_infos = tuple(selected_infos)
        unselected_id_list = tuple(unselected_id_list)
        unselected_infos = tuple(unselected_infos)
        
        active_training = [selected_id_list, selected_infos, unselected_id_list, unselected_infos]

        # Create new dataloaders
        batch_size = unlabelled_loader.batch_size
        print("Batch_size of a single loader: %d" % (batch_size))
        workers = unlabelled_loader.num_workers

        # delete old sets and loaders, save space for new sets and loaders
        del labelled_loader.dataset, unlabelled_loader.dataset, labelled_loader, unlabelled_loader

        labelled_set, unlabelled_set, \
        labelled_loader, unlabelled_loader, \
        sampler_labelled, sampler_unlabelled = build_active_dataloader(
            cfg.DATA_CONFIG,
            cfg.CLASS_NAMES,
            batch_size,
            dist_train,
            workers=workers,
            logger=logger,
            training=True,
            active_training=active_training
        )

        # check that the dataloader is correct --> loading the selected_frames only
        set_frames = [i.item() for batch in labelled_loader for i in batch['frame_id']]
        assert all(frame in selected_frames for frame in set_frames) and all(frame in set_frames for frame in selected_frames)

        return labelled_loader, unlabelled_loader

    else:
        raise NotImplementedError("This case is not implemented. Revise your configurations!!")
    

def get_latest_class_weights(active_label_dir):
    """
    Retrieves the most recent class weights from a file within a given directory.

    Parameters:
    - active_label_dir (Path or str): The directory path where the 'class_weights.txt' file is stored or will be created.

    Returns:
    - dict: A dictionary containing the latest class weights, with the class names as keys and their corresponding weights as values.

    """

    # check if the prior distribution file exists, if not then create it
    weights_txt = active_label_dir / 'class_weights.txt'

    if not os.path.isfile(weights_txt):
        with open(weights_txt, 'a') as f:
            # construct uniform class weights
            idx_to_label = {
                '1': 'CAR',
                '2': 'VAN',
                '3': 'BICYCLE',
                '4': 'MOTORCYCLE',
                '5': 'TRUCK',
                '6': 'TRAILER',
                '7': 'BUS',
                '8': 'PEDESTRIAN',
            }
            class_weights = {value: 1 for value in idx_to_label.values()}
            class_weights = {k: v/sum(class_weights.values()) for k, v in class_weights.items()}
            weights_line = "weights_at_step_0: " + str(class_weights)
            f.write(weights_line + "\n")

    else:
        # the class weights file exist, get the class weights
        latest_weights = None
        with open(active_label_dir / 'class_weights.txt', 'r') as f:
            for line in f:
                if line.strip():
                    latest_weights = line
        dict_str = latest_weights.split(':', 1)[1].strip()
        class_weights = ast.literal_eval(dict_str)

    
    return class_weights


def update_class_weights(active_label_dir, min_weight=1):
    """
    Updates the class weights based on the distribution of object classes in the growing labeled dataset.

    Parameters:
    - active_label_dir (Path or str): The directory path where the .pkl files are stored and where 'class_weights.txt' will be updated.
    - min_weight (int, optional): The minimum weight to be assigned to any class. Default value is 1.

    Returns:
    - None: The function writes the updated class weights to a file and does not return any value.
    """

    idx_to_label = {
        '1': 'CAR',
        '2': 'VAN',
        '3': 'BICYCLE',
        '4': 'MOTORCYCLE',
        '5': 'TRUCK',
        '6': 'TRAILER',
        '7': 'BUS',
        '8': 'PEDESTRIAN',
    }

    class_distribution = {value: 1 for value in idx_to_label.values()}
    
    pkl_files = glob.glob(str(active_label_dir / '*.pkl'))
    pkl_files = [i for i in pkl_files if 'selected_frames' in i]

    for file in pkl_files:
        with open(file, 'rb') as f:
            data = pkl.load(f)

        bbox_data = data['selected_bbox']
        for bbox in bbox_data:
            for cls_name, count in bbox.items():
                class_distribution[cls_name] += count
    
    for cls_name in class_distribution.keys():
        class_distribution[cls_name] = class_distribution[cls_name].item()


    class_weights = {class_name: 1/count for class_name, count in class_distribution.items()}
    weight_sum = sum(class_weights.values())
    class_weights = {class_name: weight/weight_sum for class_name, weight in class_weights.items()}

    steps = len(pkl_files)
    with open(active_label_dir / 'class_weights.txt', 'a') as f:
        weights_line = f"weights_at_step_{steps}: " + str(class_weights)
        f.write(weights_line + "\n")


def get_model_checksum(model):
    import hashlib
    # Convert the model's state_dict to a string representation
    model_string = str(model.state_dict())
    
    # Compute the MD5 checksum of this string
    return hashlib.md5(model_string.encode()).hexdigest() 

