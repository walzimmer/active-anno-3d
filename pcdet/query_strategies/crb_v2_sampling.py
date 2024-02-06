
import torch
import tqdm
import numpy as np
import wandb
import time
import scipy
import pickle as pkl

import torch.nn.functional as F

from pcdet.datasets import build_active_dataloader
from .strategy import Strategy
from pcdet.models import load_data_to_gpu
from torch.distributions import Categorical
from sklearn.cluster import kmeans_plusplus, KMeans, MeanShift, Birch
from sklearn.mixture import GaussianMixture
from scipy.stats import uniform
from sklearn.neighbors import KernelDensity
from scipy.cluster.vq import vq
from typing import Dict, List

from tools.utils.eval_utils.eval_utils import eval_one_epoch

class v2CRBSampling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):

        super(v2CRBSampling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

        # coefficients controls the ratio of selected subset
        self.k1 = getattr(cfg.ACTIVE_TRAIN.ACTIVE_CONFIG, 'K1', 5)
        self.k2 = getattr(cfg.ACTIVE_TRAIN.ACTIVE_CONFIG, 'K2', 3)
        
        # bandwidth for the KDE in the GPDB module
        self.bandwidth = getattr(cfg.ACTIVE_TRAIN.ACTIVE_CONFIG, 'BANDWDITH', 5)
        # ablation study for prototype selection
        self.prototype = getattr(cfg.ACTIVE_TRAIN.ACTIVE_CONFIG, 'CLUSTERING', 'kmeans++')
        # controls the boundary of the uniform prior distribution
        self.alpha = 0.95

        self.active_label_dir = active_label_dir


    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        i = 0
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                i += 1
                m.train()
        print('**found and enabled {} Dropout layers for random sampling**'.format(i))


    def query(self, leave_pbar=True, cur_epoch=None, save_points=True, use_test_set=False):

        select_dic = {}

        val_dataloader_iter = iter(self.unlabelled_loader)
        val_loader = self.unlabelled_loader
        total_it_each_epoch = len(self.unlabelled_loader)
        
        # feed forward the model
        if self.rank == 0:
            pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                             desc='evaluating_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
        self.model.eval()
        self.enable_dropout(self.model)
        num_class = len(self.labelled_loader.dataset.class_names)
        check_value = []
        cls_results = {}
        reg_results = {}
        density_list = {}
        label_list = {}

        '''
        -------------  Stage 1: Consise Label Sampling ----------------------
        '''
        for cur_it in range(total_it_each_epoch):
            try:
                unlabelled_batch = next(val_dataloader_iter)
            except StopIteration:
                unlabelled_dataloader_iter = iter(val_loader)
                unlabelled_batch = next(unlabelled_dataloader_iter)
            with torch.no_grad():
                load_data_to_gpu(unlabelled_batch)
                pred_dicts, _ = self.model(unlabelled_batch)

                for batch_inx in range(len(pred_dicts)):
                    # save the meta information and project it to the wandb dashboard
                    
                    value, counts = torch.unique(pred_dicts[batch_inx]['pred_labels'], return_counts=True)
                    if len(value) == 0:
                        entropy = 0
                    else:
                        # calculates the shannon entropy of the predicted labels of bounding boxes
                        unique_proportions = torch.ones(num_class).cuda()
                        unique_proportions[value - 1] = counts.float()
                        entropy = Categorical(probs = unique_proportions / sum(counts)).entropy()
                        check_value.append(entropy)

                    # save the hypothetical labels for the regression heads at Stage 2
                    cls_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_cls']
                    reg_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_reg']
                    # used for sorting
                    select_dic[unlabelled_batch['frame_id'][batch_inx]] = entropy
                    # save the density records for the Stage 3
                    density_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['pred_box_unique_density']
                    label_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['pred_labels']

            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        check_value.sort()
        log_data = [[idx, value] for idx, value in enumerate(check_value)]
        table = wandb.Table(data=log_data, columns=['idx', 'selection_value'])
        wandb.log({'value_dist_epoch_{}'.format(cur_epoch) : wandb.plot.line(table, 'idx', 'selection_value',
            title='value_dist_epoch_{}'.format(cur_epoch))})

        # sort and get selected_frames
        select_dic = dict(sorted(select_dic.items(), key=lambda item: item[1]))
        # narrow down the scope
        selected_frames = list(select_dic.keys())[::-1][:int(self.k1 * self.cfg.ACTIVE_TRAIN.SELECT_NUMS)]

        selected_id_list, selected_infos = [], []
        unselected_id_list, unselected_infos = [], []
        '''
        -------------  Stage 2: Representative Prototype Selection ----------------------
        '''
        # rebuild a dataloader for K1 samples
        for i in range(len(self.pairs)):
            if self.pairs[i][0] in selected_frames:
                selected_id_list.append(self.pairs[i][0])
                selected_infos.append(self.pairs[i][1])
            else:
                # no need for unselected part
                if len(unselected_id_list) == 0:
                    unselected_id_list.append(self.pairs[i][0])
                    unselected_infos.append(self.pairs[i][1])

        selected_id_list, selected_infos, \
        unselected_id_list, unselected_infos = \
            tuple(selected_id_list), tuple(selected_infos), \
            tuple(unselected_id_list), tuple(unselected_infos)
        
        active_training = [selected_id_list, selected_infos, unselected_id_list, unselected_infos]

        labelled_set, _,\
        grad_loader, _,\
        _, _ = build_active_dataloader(
            self.cfg.DATA_CONFIG,
            self.cfg.CLASS_NAMES,
            1,
            False,
            workers=self.labelled_loader.num_workers,
            logger=None,
            training=False,
            active_training=active_training
        )
        grad_dataloader_iter = iter(grad_loader)
        total_it_each_epoch = len(grad_loader)
        

        self.model.eval()
            
        fc_grad_1_embedding_list = []
        # fc_grad_2_embedding_list = []
        index_list = []


        # start looping over the K1 samples        
        if self.rank == 0:
                pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar,
                                desc='inf_grads_unlabelled_set_epoch_%d' % cur_epoch, dynamic_ncols=True)
                
        for cur_it in range(total_it_each_epoch):
            try:
                unlabelled_batch = next(grad_dataloader_iter)
                
            except StopIteration:
                unlabelled_dataloader_iter = iter(grad_loader)
                unlabelled_batch = next(grad_dataloader_iter)

            load_data_to_gpu(unlabelled_batch)

            pred_dicts, _ = self.model(unlabelled_batch)
            # get sample wise grads
            rcnn_cls_preds_per_sample = pred_dicts[0]['batch_rcnn_cls']
            rcnn_cls_gt_per_sample = cls_results[unlabelled_batch['frame_id'][0]]

            rcnn_reg_preds_per_sample = pred_dicts[0]['batch_rcnn_reg']
            rcnn_reg_gt_per_sample = reg_results[unlabelled_batch['frame_id'][0]]

            
            cls_loss, _ = self.model.roi_head.get_box_cls_layer_loss({'rcnn_cls': rcnn_cls_preds_per_sample, 
                                                                      'rcnn_cls_labels': rcnn_cls_gt_per_sample})
            
            reg_loss = self.model.roi_head.get_box_reg_layer_loss({'rcnn_reg': rcnn_reg_preds_per_sample, 
                                                                   'reg_sample_targets': rcnn_reg_gt_per_sample})
            
            # clean cache
            # del rcnn_cls_preds_per_sample, rcnn_cls_gt_per_sample
            # del rcnn_reg_preds_per_sample, rcnn_reg_gt_per_sample
            # torch.cuda.empty_cache()

            loss = cls_loss + reg_loss.mean()
            self.model.zero_grad()
            loss.backward()

            fc_grads_1 = self.model.roi_head.shared_fc_layer[4].weight.grad.clone().detach().cpu()
            fc_grad_1_embedding_list.append(fc_grads_1)

            # fc_grads_2 = self.model.roi_head.shared_fc_layer[0].weight.grad.clone().detach().cpu()
            # fc_grad_2_embedding_list.append(fc_grads_2)
            index_list.append(unlabelled_batch['frame_id'][0])

            if self.rank == 0:
                pbar.update()
                    # pbar.set_postfix(disp_dict)
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        # stacking gradients for K1 candiates        
        fc_grad_embeddings = torch.stack(fc_grad_1_embedding_list, 0)
        num_sample = fc_grad_embeddings.shape[0]
        fc_grad_embeddings = fc_grad_embeddings.view(num_sample, -1)
        start_time = time.time()

        # choose the prefered prototype selection method and select the K2 medoids
        if self.prototype == 'kmeans++':
            _, selected_fc_idx = kmeans_plusplus(fc_grad_embeddings.numpy(), n_clusters=int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS * self.k2), random_state=0)
        elif self.prototype == 'kmeans':
            km = KMeans(n_clusters=int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS * self.k2), random_state=0).fit(fc_grad_embeddings.numpy())
            selected_fc_idx, _ = vq(km.cluster_centers_, fc_grad_embeddings.numpy())
        elif self.prototype == 'birch':
            ms = Birch(n_clusters=int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS * self.k2)).fit(fc_grad_embeddings.numpy())
            selected_fc_idx, _ = vq(ms.subcluster_centers_, fc_grad_embeddings.numpy())
        elif self.prototype == 'gmm':
            gmm = GaussianMixture(n_components=int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS * self.k2), random_state=0, covariance_type="diag").fit(fc_grad_embeddings.numpy())
            selected_fc_idx, _ = vq(gmm.means_, fc_grad_embeddings.numpy())
        else:
            raise NotImplementedError
        selected_frames = [index_list[i] for i in selected_fc_idx]
        print("--- {%s} running time: %s seconds for fc grads---" % (self.prototype, time.time() - start_time))


        '''
        -------------  Stage 3: Greedy Point Cloud Density Balancing ----------------------
        '''