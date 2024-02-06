
import torch
import tqdm
import numpy as np
import wandb
import time
import scipy
import pickle as pkl
import numpy as np
from scipy.special import gammaln, psi
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

class wCRBSampling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):

        super(wCRBSampling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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

        self.idx_to_label = {
                '1': 'CAR',
                '2': 'VAN',
                '3': 'BICYCLE',
                '4': 'MOTORCYCLE',
                '5': 'TRUCK',
                '6': 'TRAILER',
                '7': 'BUS',
                '8': 'PEDESTRIAN',
            }

    def compute_dirichlet_entropy(class_weights):
        """
        Compute the entropy of a Dirichlet distribution given class weights.
        :param class_weights: Array-like, class weights (alphas) for the Dirichlet distribution.
        :return: Entropy of the Dirichlet distribution.
        """
        alpha = np.array(class_weights)

        # Compute B(alpha)
        B_alpha = np.exp(np.sum(gammaln(alpha)) - gammaln(np.sum(alpha)))

        # Compute the sum part of the entropy formula
        sum_part = -np.sum((alpha - 1) * psi(alpha))

        # Final entropy calculation
        entropy = np.log(B_alpha) + sum_part
        return entropy



    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        i = 0
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                i += 1
                m.train()
        print('**found and enabled {} Dropout layers for random sampling**'.format(i))


    def query(self, leave_pbar=True, cur_epoch=None, use_test_set=False, class_weights=None):

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
                    if not use_test_set:
                        self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])
                    
                    value, counts = torch.unique(pred_dicts[batch_inx]['pred_labels'], return_counts=True)
                    if len(value) == 0:
                        weighted_entropy = 0
                    else:
                        # calculates the shannon entropy of the predicted labels of bounding boxes
                        unique_proportions = torch.ones(num_class).cuda()
                        unique_proportions[value - 1] = counts.float()
                        probs = unique_proportions / sum(counts)
                        

                    # Weighted entropy calculation
                    weighted_entropy = 0
                    for val, prob in zip(value, probs):
                        class_label = self.idx_to_label[str(val.item())] 
                        class_weight = class_weights.get(class_label, 1)
                        weighted_entropy -= class_weight * prob * torch.log(prob)

                    check_value.append(weighted_entropy)

                    # save the hypothetical labels for the regression heads at Stage 2
                    cls_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_cls']
                    reg_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_reg']
                    # used for sorting
                    select_dic[unlabelled_batch['frame_id'][batch_inx]] = weighted_entropy
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
            training=(not use_test_set),
            active_training=active_training
        )
        grad_dataloader_iter = iter(grad_loader)
        total_it_each_epoch = len(grad_loader)
        
        if use_test_set:
            self.model.eval()
            for name, params in self.model.named_parameters():
                if 'backbone_3d' in name:
                    params.requires_grad = False
        else:
            self.model.train()
            
        fc_grad_1_embedding_list = []
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
            if use_test_set:
                pred_dicts, _ = self.model(unlabelled_batch)

                rcnn_cls_preds_per_sample = pred_dicts[0]['batch_rcnn_cls']
                rcnn_cls_gt_per_sample = cls_results[unlabelled_batch['frame_id'][0]]
                rcnn_reg_preds_per_sample = pred_dicts[0]['batch_rcnn_reg']
                rcnn_reg_gt_per_sample = reg_results[unlabelled_batch['frame_id'][0]]
            else:
                pred_dicts, _, _= self.model(unlabelled_batch)
                rcnn_cls_preds_per_sample = pred_dicts['rcnn_cls']
                rcnn_cls_gt_per_sample = cls_results[unlabelled_batch['frame_id'][0]]
                rcnn_reg_preds_per_sample = pred_dicts['rcnn_reg']
                rcnn_reg_gt_per_sample = reg_results[unlabelled_batch['frame_id'][0]]

            
            cls_loss, _ = self.model.roi_head.get_box_cls_layer_loss({'rcnn_cls': rcnn_cls_preds_per_sample, 
                                                                      'rcnn_cls_labels': rcnn_cls_gt_per_sample})
            
            reg_loss = self.model.roi_head.get_box_reg_layer_loss({'rcnn_reg': rcnn_reg_preds_per_sample, 
                                                                   'reg_sample_targets': rcnn_reg_gt_per_sample})
            
            # clean cache
            del rcnn_cls_preds_per_sample, rcnn_cls_gt_per_sample
            del rcnn_reg_preds_per_sample, rcnn_reg_gt_per_sample
            torch.cuda.empty_cache()

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
            grad_centroids, selected_fc_idx = kmeans_plusplus(fc_grad_embeddings.numpy(), n_clusters=int(self.cfg.ACTIVE_TRAIN.SELECT_NUMS * self.k2), random_state=0)
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

        fc_grad_embedding_dict = {}
        fc_grad_embedding_dict[unlabelled_batch['frame_id'][0]] = {
            'fc_grads': fc_grad_embeddings,
            'centroid': grad_centroids,
            'centroid_indices': selected_fc_idx
        }

        '''
        -------------  Stage 3: Greedy Point Cloud Density Balancing ----------------------
        '''
        del fc_grad_1_embedding_list
        # del fc_grad_2_embedding_list

        sampled_density_list = [density_list[i] for i in selected_frames]
        sampled_label_list = [label_list[i] for i in selected_frames]

        """ Build the uniform distribution for each class """
        start_time = time.time()

        density_all = torch.cat(list(density_list.values()), 0)
        label_all = torch.cat(list(label_list.values()), 0)


        # **************** Here, I can just use the prior knowledge of the class point density distrubtion as
        # **************** my target distribution ***********************************************************

        label_counts_full = [0] * num_class

        unique_labels, label_counts = torch.unique(label_all, return_counts=True)
        unique_labels = unique_labels.cpu().numpy()
        label_counts = label_counts.cpu().numpy()

        for i, label in enumerate(unique_labels):
            label_counts_full[label-1] = label_counts[i]

        label_to_idx = {label.item(): label.item()-1 for label in unique_labels}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

        # sorted_density = [torch.sort(density_all[label_all==unique_label])[0] for unique_label in unique_labels]
        sorted_density = [0] * num_class
        for label in unique_labels:
            label_idx = label_to_idx[label.item()]
            sorted_density[label_idx] = torch.sort(density_all[label_all==label])[0]
    
        global_density_max = [0] * num_class
        global_density_high = [0] * num_class
        global_density_low = [0] * num_class

        for label in unique_labels:
            label_idx = label_to_idx[label.item()]

            if label_counts_full[label_idx].shape == 2:
                global_density_max[label_idx] = int(sorted_density[label_idx][-1])
                global_density_high[label_idx] = int(sorted_density[label_idx][-1])
                global_density_low[label_idx] = int(sorted_density[label_idx][0])
            else:
                global_density_max[label_idx] = int(sorted_density[label_idx][-1])
                global_density_high[label_idx] = int(sorted_density[label_idx][int(self.alpha * label_counts_full[label_idx])])
                global_density_low[label_idx] = int(sorted_density[label_idx][-int(self.alpha * label_counts_full[label_idx])])

        for label in unique_labels:
            label = label_to_idx[label.item()]

            if global_density_high[label] == global_density_low[label]:
                # cnst = int(torch.mean(sorted_density[label]).item())
                cnst = int(1)
                global_density_high[label] = global_density_high[label] + cnst
                global_density_low[label] = max(0, global_density_low[label] - cnst)
        
        x_axis = [np.linspace(-50, int(global_density_max[i])+50, 400) for i in range(num_class)]
        uniform_dist_per_cls = [uniform.pdf(x_axis[i], global_density_low[i], global_density_high[i] - global_density_low[i])
                                if global_density_max[i] != 0 else np.zeros_like(x_axis[i]) for i in range(num_class)]

        print("--- Build the uniform distribution running time: %s seconds ---" % (time.time() - start_time))

        density_list, label_list, frame_id_list = sampled_density_list, sampled_label_list, selected_frames

        selected_frames: List[str] = []
        selected_box_densities: torch.tensor = torch.tensor([]).cuda()
        selected_box_labels: torch.tensor = torch.tensor([]).cuda()


        # looping over N_r samples
        if self.rank == 0:
            pbar = tqdm.tqdm(total=self.cfg.ACTIVE_TRAIN.SELECT_NUMS, leave=leave_pbar,
                             desc='global_density_div_for_epoch_%d' % cur_epoch, dynamic_ncols=True)
        
        for j in range(self.cfg.ACTIVE_TRAIN.SELECT_NUMS):
            if j == 0: # initially, we randomly select a frame.

                selected_frames.append(frame_id_list[j])
                selected_box_densities = torch.cat((selected_box_densities, density_list[j]))
                selected_box_labels = torch.cat((selected_box_labels, label_list[j]))

                # remove selected frame
                del density_list[0]
                del label_list[0]
                del frame_id_list[0]

            else: # go through all the samples and choose the frame that can most reduce the KL divergence
                best_frame_id = None
                best_frame_index = None
                best_inverse_coff = -1

                for i in range(len(density_list)):
                    unique_proportions = np.zeros(num_class)
                    KL_scores_per_cls = np.zeros(num_class)

                    for cls in range(num_class):
                        if (label_list[i] == cls + 1).sum() == 0:
                            unique_proportions[cls] = 1
                            KL_scores_per_cls[cls] = np.inf
                        else:
                            # get existing selected box densities
                            selected_box_densities_cls = selected_box_densities[selected_box_labels==(cls + 1)]
                            # append new frame's box densities to existing one
                            selected_box_densities_cls = torch.cat((selected_box_densities_cls,
                                                                    density_list[i][label_list[i] == (cls + 1)]))
                            # initialize kde
                            kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(
                                selected_box_densities_cls.cpu().numpy()[:, None])

                            logprob = kde.score_samples(x_axis[cls][:, None])
                            KL_score_per_cls = scipy.stats.entropy(uniform_dist_per_cls[cls], np.exp(logprob))
                            KL_scores_per_cls[cls] = KL_score_per_cls
                            # ranging from 0 to 1
                            unique_proportions[cls] = 2 / np.pi * np.arctan(np.pi / 2 * KL_score_per_cls)

                    inverse_coff = np.mean(1 - unique_proportions)
                    # KL_save_list.append(inverse_coff)
                    if inverse_coff > best_inverse_coff:
                        best_inverse_coff = inverse_coff
                        best_frame_index = i
                        best_frame_id = frame_id_list[i]

                # remove selected frame
                selected_box_densities = torch.cat((selected_box_densities, density_list[best_frame_index]))
                selected_box_labels = torch.cat((selected_box_labels, label_list[best_frame_index]))

                del density_list[best_frame_index]
                del label_list[best_frame_index]
                del frame_id_list[best_frame_index]

                selected_frames.append(best_frame_id)

            if self.rank == 0:
                pbar.update()
                # pbar.set_postfix(disp_dict)
                pbar.refresh()

        if self.rank == 0:
            pbar.close()

        self.model.eval()
        # returned the index of acquired bounding boxes 

        # TODO: save selected densities and labels for target point density computation
        select_dict = {
            'entropies': select_dic,
            'gradient_info': fc_grad_embedding_dict,
            'selected_box_densities': selected_box_densities,
            'selected_box_labels': selected_box_labels,
            'density_all': density_all,
            'density_list': density_list,
            'labels_all': label_all,
            'labels_list': label_list
        }
        
        with open(self.active_label_dir / f'ablation_dict_epochs_{cur_epoch}.pkl', 'wb') as f:
            pkl.dump(select_dict, f)
        
        return selected_frames, fc_grad_embeddings


