import torch
import tqdm
import numpy as np
import wandb
import time
import scipy
import pickle as pkl
import torch.nn.functional as F
import random
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


class tCRBSampling(Strategy):
    def __init__(self, model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg):

        super(tCRBSampling, self).__init__(model, labelled_loader, unlabelled_loader, rank, active_label_dir, cfg)

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
    
    @staticmethod
    def temporal_stage_one(select_dict, window_size, stride, return_window_entropies=False):

        select_dic_sorted = dict(sorted(select_dict.items(), key=lambda item: item[0]))
        max_entropy_sum = 0
        selected_window_start_index = 0
        sorted_frame_ids = list(select_dic_sorted.keys())

        window_entropies = []

        for start_index in range(0, len(sorted_frame_ids) - window_size + 1, stride):
            current_window = sorted_frame_ids[start_index:start_index + window_size]
            entropy_sum = sum(select_dic_sorted[frame_id] for frame_id in current_window)

        
            if entropy_sum > max_entropy_sum:
                max_entropy_sum = entropy_sum
                selected_window_start_index = start_index
                window_info = {
                'start_frame': current_window[0],
                'end_frame': current_window[-1],
                'entropy_sum': entropy_sum,
                'entropies': [select_dict[i] for i in current_window]
                }
                window_entropies.append(window_info)

        selected_frames = sorted_frame_ids[selected_window_start_index:selected_window_start_index + window_size]

        if return_window_entropies:
            return selected_frames, window_entropies
        return selected_frames, None

    @staticmethod
    def temporal_stage_two(select_dict, window_size, stride, return_window_details=False):

        sorted_grad_dict = dict(sorted(select_dict.items(), key=lambda item: item[0]))
        window_details = []
        selected_window_start_index = 0
        sorted_frame_ids = list(sorted_grad_dict.keys())

        max_magnitude_sum = 1e-6
        max_orientation_sum = 1e-6
        best_combined_score = 0

        for start_index in range(0, len(sorted_frame_ids) - window_size + 1, stride):
            curr_window = sorted_frame_ids[start_index:start_index + window_size]
            curr_window_grads = [sorted_grad_dict[idx] for idx in curr_window]

            weights = [grad.norm() for grad in curr_window_grads]

            weighted_sum = sum(grad * weight for grad, weight in zip(curr_window_grads, weights))
            total_weight = sum(weights)
            weighted_mean_grad = weighted_sum / total_weight if total_weight != 0 else torch.zeros_like(weighted_sum)

            # compute magnitude and orientation sum
            magnitude_sum = 0
            orientation_sum = 0
            for grad in curr_window_grads:
                magnitude_sum += (grad).norm()
                orientation_sum += torch.nn.functional.cosine_similarity(grad.view(-1), weighted_mean_grad.view(-1), dim=0)

            max_magnitude_sum = max(max_magnitude_sum, magnitude_sum)
            max_orientation_sum = max(max_orientation_sum, orientation_sum)

            norm_magnitude_sum = magnitude_sum/max_magnitude_sum
            norm_orientation_sum = orientation_sum/max_orientation_sum


            combined_sum = norm_magnitude_sum - norm_orientation_sum

            if combined_sum > best_combined_score:
                best_combined_score = combined_sum
                selected_window_start_index = start_index
                window_details.append({
                'start_frame': curr_window[0],
                'end_frame': curr_window[-1],
                'window_grads': curr_window_grads,
                'weighted_mean_grad': weighted_mean_grad,
                'magnitude_sum': norm_magnitude_sum.item(),
                'orientation_sum': norm_orientation_sum.item(),
                'combined_sum': combined_sum.item()
            })

        selected_frames = sorted_frame_ids[selected_window_start_index:selected_window_start_index + window_size]

        if return_window_details:
            return selected_frames, window_details
        return selected_frames, None

    @staticmethod
    def temporal_stage_three(select_dict, window_size, stride, return_window_details=False):
        sorted_kl_div_dict = dict(sorted(select_dict.items(), key=lambda item: item[0]))
        min_kl_div_sum = 1e6
        selected_window_start_index = 0
        sorted_frame_ids = list(sorted_kl_div_dict.keys())

        window_details = []
        for start_index in range(0, len(sorted_frame_ids) - window_size + 1, stride):
            curr_window = sorted_frame_ids[start_index:start_index+window_size]
            kl_div_sum = sum(sorted_kl_div_dict[idx] for idx in curr_window)

            window_details.append({
                'start_frame': curr_window[0],
                'end_frame': curr_window[-1],
                'kl_div_sum': kl_div_sum
            })

            if kl_div_sum < min_kl_div_sum:
                min_kl_div_sum = kl_div_sum
                selected_window_start_index = start_index
        
        selected_frames = sorted_frame_ids[selected_window_start_index:selected_window_start_index + window_size]
        if return_window_details:
            return selected_frames, window_details
        return selected_frames, None


    def query(self, leave_pbar=True, cur_epoch=None, use_test_set=False, proanno=False):

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
        -------------  Stage 1: Concise Label Sampling ----------------------
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
                    if not use_test_set:
                        self.save_points(unlabelled_batch['frame_id'][batch_inx], pred_dicts[batch_inx])

                    value, counts = torch.unique(pred_dicts[batch_inx]['pred_labels'], return_counts=True)
                    if len(value) == 0:
                        entropy = 0
                    else:
                        # calculates the shannon entropy of the predicted labels of bounding boxes
                        unique_proportions = torch.ones(num_class).cuda()
                        unique_proportions[value - 1] = counts.float()
                        entropy = Categorical(probs = unique_proportions / sum(counts)).entropy()
                        check_value.append(entropy)

                    cls_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_cls']
                    reg_results[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['batch_rcnn_reg']
                    select_dic[unlabelled_batch['frame_id'][batch_inx]] = entropy
                    density_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['pred_box_unique_density']
                    label_list[unlabelled_batch['frame_id'][batch_inx]] = pred_dicts[batch_inx]['pred_labels']

            if self.rank == 0:
                pbar.update()
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        check_value.sort()

        select_dic = dict(sorted(select_dic.items(), key=lambda item: item[1]))
        selected_frames = list(select_dic.keys())[::-1][:int(self.k1 * self.cfg.ACTIVE_TRAIN.SELECT_NUMS)]
        
        start_time = time.time()
        selected_frames, entropies_window_info = self.temporal_stage_one(
            select_dict=select_dic,
            window_size=int(self.k1 * self.cfg.ACTIVE_TRAIN.SELECT_NUMS),
            stride=10,
            return_window_entropies=True
        )
        
        if not use_test_set:
            select_dict = {
                'all_entropies': select_dic,
                'stage_1_selected_entropies': [select_dic[i] for i in selected_frames],
                'selected_frames_stage_1': selected_frames,
                'density_list': density_list,
                'labels_list': label_list,
                'entropy_window_info': entropies_window_info
            }

        selected_id_list, selected_infos = [], []
        unselected_id_list, unselected_infos = [], []
        

        print("********************** STAGE 1 DONE **********************")
        '''
        -------------  Stage 2: Representative Prototype Selection ----------------------
        '''
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
        fc_grad_embedding_dict = {}

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
            fc_grad_embedding_dict[unlabelled_batch['frame_id'][0]] = fc_grads_1

            if self.rank == 0:
                pbar.update()
                pbar.refresh()
        if self.rank == 0:
            pbar.close()

        selected_frames, fc_grads_window_info = self.temporal_stage_two(
            select_dict=fc_grad_embedding_dict,
            window_size=int(self.k2 * self.cfg.ACTIVE_TRAIN.SELECT_NUMS),
            stride=4,
            return_window_details=True
        )
        fc_grad_embedding_dict = {key: val for key, val in fc_grad_embedding_dict.items() if key in selected_frames}
        fc_grad_embeddings = torch.stack(fc_grad_1_embedding_list, 0)
        num_sample = fc_grad_embeddings.shape[0]
        fc_grad_embeddings = fc_grad_embeddings.view(num_sample, -1)
        end_time = time.time()

        if not use_test_set:
            select_dict['gradient_info'] = fc_grads_window_info
            select_dict['selected_frames_stage_2'] = selected_frames

        print("********************** STAGE 2 DONE **********************")

        '''
        -------------  Stage 3: Greedy Point Cloud Density Balancing ----------------------
        '''
        del fc_grad_1_embedding_list

        sampled_density_list = [density_list[i] for i in selected_frames]
        sampled_label_list = [label_list[i] for i in selected_frames]

        """ Build the uniform distribution for each class """
        start_time = time.time()
        density_all = torch.cat(list(density_list.values()), 0)
        label_all = torch.cat(list(label_list.values()), 0)
        label_counts_full = [0] * num_class
        unique_labels, label_counts = torch.unique(label_all, return_counts=True)
        unique_labels = unique_labels.cpu().numpy()
        label_counts = label_counts.cpu().numpy()
        for i, label in enumerate(unique_labels):
            label_counts_full[label-1] = label_counts[i]
        label_to_idx = {label.item(): label.item()-1 for label in unique_labels}
        idx_to_label = {idx: label for label, idx in label_to_idx.items()}

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

        density_list, label_list, frame_id_list = sampled_density_list, sampled_label_list, selected_frames

        selected_frames: List[str] = []
        selected_box_densities: torch.tensor = torch.tensor([]).cuda()
        selected_box_labels: torch.tensor = torch.tensor([]).cuda()


        kl_div_dict = {}

        for i, densities in enumerate(density_list):
            labels = label_list[i]
            kl_div_frame = 0

            frame_id = frame_id_list[i]
            unique_classes = torch.unique(labels)
            class_weight = 1 / len(unique_classes) if len(unique_classes) > 0 else 1

            for cls in unique_classes:
                densities_cls = densities[labels == cls]
                if len(densities_cls) == 0:
                    continue
            
                kde = KernelDensity(kernel='gaussian', bandwidth=self.bandwidth).fit(densities_cls.cpu().numpy()[:, None])
                logprob = kde.score_samples(x_axis[cls - 1][:, None])

                kl_div_cls = scipy.stats.entropy(uniform_dist_per_cls[cls - 1], np.exp(logprob))
                kl_div_frame += (class_weight * kl_div_cls)
            kl_div_dict[frame_id] = kl_div_frame

        selected_frames, kl_window_info = self.temporal_stage_three(
            select_dict=kl_div_dict,
            window_size=self.cfg.ACTIVE_TRAIN.SELECT_NUMS,
            stride=2,
            return_window_details=True
        )

        self.model.eval()

        print("********************** SAMPLE SELECTION DONE **********************")
        if not use_test_set:
            select_dict['selected_box_densities'] = selected_box_densities
            select_dict['selected_box_labels'] = selected_box_labels
            select_dict['labels_all'] = label_all
            select_dict['density_all'] = density_all
        
            with open(self.active_label_dir / f'ablation_dict_epochs_{cur_epoch}.pkl', 'wb') as f:
                    pkl.dump(select_dict, f)
        
            return selected_frames, fc_grad_embeddings
        else:
            selected_frames = [i.item() for i in selected_frames]
            print(selected_frames)
            return selected_frames
