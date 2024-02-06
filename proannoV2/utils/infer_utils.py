import os
import numpy as np
import argparse
import json
import pickle as pkl
import datetime
import glob
import random
import uuid
import open3d
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pcdet.models import build_network, model_fn_decorator
from pcdet.datasets.tumtraf.tumtraf_dataset import TUMTrafDataset
from pcdet.models import load_data_to_gpu
from tools.utils.det_utils.detections import detections_to_openlabel, Detection

from tools.proannoV2.utils.utils import TorchEncoder, NumpyEncoder, idx_to_label
from pcdet.datasets.tumtraf.tumtraf_converter import TUMTraf2KITTI, TUMTraf2KITTI_v2
from pcdet.datasets.tumtraf.tumtraf_dataset import create_tumtraf_infos
from pcdet.datasets.tumtraf.tumtraf_utils import create_image_sets
from pcdet.datasets.tumtraf.tumtraf_dataset import TUMTrafDataset
from pytorch3d.ops import box3d_overlap
from scipy.spatial.transform.rotation import Rotation as R



def prepare_data_current_annotation(cfg, logger):
    """
    prepares the sequence currently being annotated for inference.
    """
    print("**** Preparing data for inference ****")
    curr_root_dir = '/ahmed/data/tumtraf/proannoV2/OpenLABEL/currently_annotating'
    curr_save_path = cfg.DATA_CONFIG.ROOT_DIR
    curr_lidar_name = os.listdir(os.path.join(curr_root_dir, 'point_clouds'))[0]

    pcd_path = os.path.join(curr_root_dir, 'point_clouds', curr_lidar_name)
    pcd_labels_path = os.path.join(curr_root_dir, 'annotations', curr_lidar_name)

    pcd_filenames = sorted(glob.glob(os.path.join(pcd_path, '*.pcd')))
    labels_filenames = sorted(glob.glob(os.path.join(pcd_labels_path, '*.json')))
    
    pcd_dict = {
        'test': pcd_filenames
    }
    labels_dict = {
        'test': labels_filenames
    }

    converter = TUMTraf2KITTI_v2(
        pcd_dict=pcd_dict,
        pcd_labels_dict=labels_dict,
        save_dir=curr_save_path,
        splits=['test'],
        logger=logger)
    
    converter.convert()

    create_image_sets(
        data_root=cfg.DATA_CONFIG.ROOT_DIR,
        splits=['test']
    )

    create_tumtraf_infos(dataset_cfg=cfg.DATA_CONFIG,
                        class_names=cfg.CLASS_NAMES,
                        data_path=cfg.DATA_CONFIG.ROOT_DIR,
                        save_path=cfg.DATA_CONFIG.ROOT_DIR,
                        splits=['test'])
    
    print("**** Data preparation done! ****")
    


def model_inference(frame_ids, cfg, ckpt, logger):

    print("**** Start Model Inference ****")

    dataset, dataloader, _ = build_dataloader_inference(frame_ids=frame_ids,
                                                        cfg=cfg,
                                                        logger=logger)

    model = build_network(model_cfg=cfg.MODEL,
                          num_class=len(cfg.CLASS_NAMES),
                          dataset=dataset)
    
    model.load_params_from_file(filename=ckpt, to_cpu=False, logger=logger)
    
    model.cuda()
    model.eval()
    
    pred_dicts_all = []
    total_iters = len(dataloader)
    dataloader_iter = iter(dataloader)
    for it in range(total_iters):
        data_batch = next(dataloader_iter)
        load_data_to_gpu(data_batch)
        pred_dicts, _ = model(data_batch)
        # pred_dicts = {key: val.detach().cpu().numpy() for key, val in pred_dicts[0].items()}
        pred_dicts_all.append(pred_dicts[0])

    print("**** Model Inference Done! ****")

    return pred_dicts_all



def build_dataloader_inference(frame_ids, cfg, logger):

    print("**** Building Dataloader for Inference ****")

    dataset = TUMTrafDataset(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        root_path=cfg.DATA_CONFIG.ROOT_DIR,
        training=False,
        logger=logger
    )

    frame_indices = [dataset.sample_id_list.index(frame_id) for frame_id in frame_ids if frame_id in dataset.sample_id_list]
    
    dataset.sample_id_list = [dataset.sample_id_list[idx] for idx in frame_indices]
    dataset.tumtraf_infos = [dataset.tumtraf_infos[idx] for idx in frame_indices]

    sampler = None

    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        pin_memory=True, 
        num_workers=1,
        shuffle=False,
        collate_fn=dataset.collate_batch,
        drop_last=False, 
        sampler=None, 
        timeout=0
)

    return dataset, dataloader, sampler



def get_corner_points(box: open3d.geometry.OrientedBoundingBox):
    center = np.array(box.center)
    extent = np.array(box.extent) / 2  # Half-lengths
    R = np.array(box.R)
    corners = np.empty((8, 3))
    for i in range(8):
        sign = np.array(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            dtype=np.float32)
        corner = center + R @ (sign[i] * extent)
        corners[i] = corner
    return corners



def filter_by_overlap(detections, boxes):
    """
    refer to this repo:
        https://gitlab.lrz.de/providentiaplusplus/dataset-dev-kit/-/blob/main/internal/src/postprocessing/filter_overlapping_boxes.py
    """
    overlapping_detections = []
    valid_detections = []
    valid_boxes = []

    for i, box in enumerate(boxes):
        if i in overlapping_detections:
            continue
        heading_angle = np.arctan2(box.R[1, 0], box.R[0, 0])
        source_corner_points = get_corner_points(box)
        source_corner_points = torch.from_numpy(source_corner_points).float()
        source_corner_points = source_corner_points.unsqueeze(0)

        current_max_score_idx = i
        current_to_be_filtered = []
        for j in range(i + 1, len(boxes)):
            if j in overlapping_detections:
                continue
            
            heading_angle = np.arctan2(boxes[j].R[1, 0], boxes[j].R[0, 0])
            target_corner_points = get_corner_points(boxes[j])
            target_corner_points = torch.from_numpy(target_corner_points).float()

            # add batch dimension
            target_corner_points = target_corner_points.unsqueeze(0)
            intersection_vol, iou_3d = box3d_overlap(source_corner_points, target_corner_points)

            epsilon = 1e-4
            # --- Add rules here ---
            if iou_3d.cpu().numpy()[0, 0] < epsilon:
                # no overlap found
                continue
            if detections[current_max_score_idx].category == "TRUCK" and detections[j].category == "TRAILER" or \
                    detections[current_max_score_idx].category == "TRAILER" and detections[j].category == "TRUCK":
                continue
            detections[current_max_score_idx].overlap = True
            detections[j].overlap = True
            # print overlap indices
            print("Overlap detected between detections: trackID1", detections[current_max_score_idx].uuid,
                  "category",
                  detections[current_max_score_idx].category,
                  "and trackID2:", detections[j].uuid, "category", detections[j].category, "IoU: ",
                  iou_3d.cpu().numpy()[0, 0])
            if detections[current_max_score_idx].score < detections[j].score:
                current_to_be_filtered.append(current_max_score_idx)
                current_max_score_idx = j
            else:
                current_to_be_filtered.append(j)

        overlapping_detections.extend(current_to_be_filtered)
        valid_detections.append(detections[current_max_score_idx])
        valid_boxes.append(boxes[current_max_score_idx])

    print("Detections before filtering:", len(detections), "Detections after filtering", len(valid_detections))
    return valid_detections, valid_boxes, overlapping_detections


def get_frame_properties(annos_dir, frame_id):
    with open(os.path.join(annos_dir, frame_id + '.json'), 'r') as f:
        data = json.load(f)
    frame_idx = list(data['openlabel']['frames'].keys())[0]
    frame_props =  data['openlabel']['frames'][frame_idx]['frame_properties']
    return frame_props


def inference_post_processing(predictions, frame_ids, openlabel_dir):

    dets_out_dir = openlabel_dir / 'currently_annotating' / 'detections'
    annos_dir = openlabel_dir / 'currently_annotating' / 'annotations'
    lidar_name = list(annos_dir.iterdir())[0]
    annos_dir = annos_dir / lidar_name
    print("**** Post-Processing Predictions ****")

    for idx, pred_dict in enumerate(predictions):
        pred_boxes = pred_dict['pred_boxes'].detach().cpu().numpy()
        pred_scores = pred_dict['pred_scores'].detach().cpu().numpy()
        pred_labels = pred_dict['pred_labels'].detach().cpu().numpy()

        pred_labels = [idx_to_label[str(label.item())] for label in pred_labels] 
    
        # selects the best detections
        num_objs = len(pred_boxes)

        filename = frame_ids[idx] + '.json'
        detections = []
        for i in range(num_objs):
            unique_id = uuid.uuid4()
            location = pred_boxes[i][0:3]
            dimensions = tuple(pred_boxes[i][3:6])
            yaw = pred_boxes[i][-1]
            category = pred_labels[i]
            score = pred_scores[i]

            obj = Detection(
                location=location,
                dimensions=dimensions,
                yaw=yaw,
                category=category,
                score=score,
                uuid=unique_id
            )
            detections.append(obj)

        
        boxes = []
        for detection in detections:
            bbox = open3d.geometry.OrientedBoundingBox()
            bbox.center = detection.location
            bbox.R = open3d.geometry.get_rotation_matrix_from_xyz(np.array([0, 0, detection.yaw]))
            bbox.extent = detection.dimensions
            bbox.color = np.array([1, 0, 0])
            boxes.append(bbox)

        detections, boxes, invalid_dets = filter_by_overlap(detections, boxes)
        valid_mask = ~np.isin(np.arange(len(predictions[idx]["pred_boxes"])), invalid_dets)
        filtered_predictions = {key: val[valid_mask] for key, val in predictions[idx].items()}

        all_predictions = {
            'raw': predictions[idx],
            'filtered': filtered_predictions,
            'overlap_indices': invalid_dets
        }

        with open(f"{dets_out_dir}/detections_{frame_ids[idx]}.pkl", 'wb') as f:
            pkl.dump(all_predictions, f)

        _ = detections_to_openlabel(
            detection_list=detections,
            filename=filename,
            output_folder_path=dets_out_dir,
            frame_id=str(idx),
            frame_properties=get_frame_properties(annos_dir=annos_dir,
                                                  frame_id=frame_ids[idx])
        )

    print("**** Post-Processing done! ****")

    return filtered_predictions

