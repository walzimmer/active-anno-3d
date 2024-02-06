"""
Evaluation Server
"""

import json
import os
import glob
import numpy as np
import numba
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from tqdm import tqdm
from pcdet.datasets.tumtraf.tumtraf_eval.iou_utils import rotate_iou_gpu_eval
from pcdet.datasets.tumtraf.tumtraf_eval.rotate_iou_cpu_eval import rotate_iou_cpu_eval
from pcdet.datasets.tumtraf.tumtraf_eval.eval_utils import compute_split_parts, overall_filter, distance_filter, overall_distance_filter

iou_threshold_dict = {
    'CAR': 0.1,
    'BUS': 0.01,
    'TRUCK': 0.01,
    'PEDESTRIAN': 0.01,
    'BICYCLE': 0.01,
    'TRAILER': 0.01,
    'VAN': 0.01,
    'MOTORCYCLE': 0.01,
    'EMERGENCY_VEHICLE': 0.01,
    'OTHER': 0.05
}

superclass_iou_threshold_dict = {
    'VEHICLE': 0.7,
    'PEDESTRIAN': 0.3,
    'BICYCLE': 0.5
}

def get_evaluation_results(gt_annos, pred_annos, classes,
                           use_superclass=False,
                           iou_thresholds=None,
                           num_pr_points=50,
                           difficulty_mode='Overall&Distance',
                           ap_with_heading=True,
                           num_parts=100,
                           print_ok=False
                           ):

    if iou_thresholds is None:
        if use_superclass:
            iou_thresholds = superclass_iou_threshold_dict
        else:
            iou_thresholds = iou_threshold_dict
            print("use threshold dict...")

    assert len(gt_annos) == len(pred_annos), f"the number of GT {len(gt_annos)} must match predictions {len(pred_annos)}"
    assert difficulty_mode in ['Overall&Distance', 'Overall', 'Distance'], "difficulty mode is not supported"
    if use_superclass:
        if ('Car' in classes) or ('Bus' in classes) or ('Truck' in classes) or ('Trailer' in classes) or ('Van' in classes) or ('Other' in classes):
            assert ('Car' in classes) and ('Bus' in classes) and ('Truck' in classes), "Car/Bus/Truck must all exist for vehicle detection"
        classes = list(superclass_iou_threshold_dict.keys())

    num_samples = len(gt_annos)
    split_parts = compute_split_parts(num_samples, num_parts)
    # commont out this line to use gpu version
    # ious = compute_iou3d(gt_annos, pred_annos, split_parts, with_heading=ap_with_heading)
    ious = compute_iou3d_cpu(gt_annos, pred_annos)
    num_classes = len(classes)
    if difficulty_mode == 'Distance':
        num_difficulties = 3
        difficulty_types = ['0-30m', '30-50m', '50m-inf']
    elif difficulty_mode == 'Overall':
        num_difficulties = 1
        difficulty_types = ['overall']
    elif difficulty_mode == 'Overall&Distance':
        num_difficulties = 4
        difficulty_types = ['overall', '0-30m', '30-50m', '50m-inf']
    else:
        raise NotImplementedError
    print("Init...")
    precision = np.zeros([num_classes, num_difficulties, num_pr_points+1])
    recall = np.zeros([num_classes, num_difficulties, num_pr_points+1])
    print("Run evaluation")
    gt_class_occurrence = {}
    pred_class_occurrence = {}
    for cur_class in classes:
        gt_class_occurrence[cur_class] = 0
        pred_class_occurrence[cur_class] = 0
    for sample_idx in tqdm(range(num_samples)):
        gt_anno = gt_annos[sample_idx]
        pred_anno = pred_annos[sample_idx]
        if use_superclass:
            if gt_anno['name'].size > 0:
                n_pedestrians = (gt_anno['name']=='PEDESTRIAN').sum()
                n_bicylces = (np.logical_or(gt_anno['name']=='BICYCLE', gt_anno['name']=='MOTORCYCLE')).sum()
                n_vehicles = len(gt_anno['name']) - n_pedestrians - n_bicylces
                gt_class_occurrence['PEDESTRIAN'] += n_pedestrians
                gt_class_occurrence['BICYCLE'] += n_bicylces
                gt_class_occurrence['VEHICLE'] += n_vehicles
            if pred_anno['name'].size > 0:
                n_pedestrians = (pred_anno['name']=='PEDESTRIAN').sum()
                n_bicylces = (np.logical_or(pred_anno['name']=='BICYCLE', pred_anno['name']=='MOTORCYCLE')).sum()
                n_vehicles = len(pred_anno['name']) - n_pedestrians - n_bicylces
                pred_class_occurrence['PEDESTRIAN'] += n_pedestrians
                pred_class_occurrence['BICYCLE'] += n_bicylces
                pred_class_occurrence['VEHICLE'] += n_vehicles
        else:
            for cur_class in classes:
                anno = np.asarray([x.upper() for x in gt_anno['name']])
                if gt_anno['name'].size > 0:
                    gt_class_occurrence[cur_class] += (anno==cur_class).sum()
                if pred_anno['name'].size > 0:
                    pred_class_occurrence[cur_class] += (pred_anno['name']==cur_class).sum()
    iou_size_larger = 0
    less = 0
    print("Determine score...")
    for cls_idx, cur_class in tqdm(enumerate(classes)):
        iou_threshold = iou_thresholds[cur_class]
        for diff_idx in range(num_difficulties):
            ### filter data & determine score thresholds on p-r curve ###
            accum_all_scores, gt_flags, pred_flags = [], [], []
            num_valid_gt = 0
            for sample_idx in range(num_samples):
                gt_anno = gt_annos[sample_idx]
                pred_anno = pred_annos[sample_idx]
                pred_score = pred_anno['score']
                iou = ious[sample_idx]
                gt_flag, pred_flag = filter_data(gt_anno, pred_anno, difficulty_mode,
                                                    difficulty_level=diff_idx, class_name=cur_class, use_superclass=use_superclass)
                gt_flags.append(gt_flag)
                pred_flags.append(pred_flag)
                num_valid_gt += sum(gt_flag == 0)
                if iou.size > 0:
                    accum_scores = accumulate_scores(iou, pred_score, gt_flag, pred_flag,
                                                    iou_threshold=iou_threshold)
                    iou_size_larger += 1
                else:
                    accum_scores = np.array([])
                    less += 1

                accum_all_scores.append(accum_scores)
            all_scores = np.concatenate(accum_all_scores, axis=0)
            thresholds = get_thresholds(all_scores, num_valid_gt, num_pr_points=num_pr_points)
            # print(accum_all_scores)
            ### compute tp/fp/fn ###
            confusion_matrix = np.zeros([len(thresholds), 3]) # only record tp/fp/fn
            for sample_idx in range(num_samples):
                pred_score = pred_annos[sample_idx]['score']
                iou = ious[sample_idx]
                gt_flag, pred_flag = gt_flags[sample_idx], pred_flags[sample_idx]
                for th_idx, score_th in enumerate(thresholds):
                    if iou.size > 0:
                        tp, fp, fn = compute_statistics(iou, pred_score, gt_flag, pred_flag,
                                                        score_threshold=score_th, iou_threshold=iou_threshold)
                        confusion_matrix[th_idx, 0] += tp
                        confusion_matrix[th_idx, 1] += fp
                        confusion_matrix[th_idx, 2] += fn

            ### draw p-r curve ###
            for th_idx in range(len(thresholds)):
                recall[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / \
                                                    (confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 2])
                precision[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / \
                                                       (confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 1])

            for th_idx in range(len(thresholds)):
                precision[cls_idx, diff_idx, th_idx] = np.max(
                    precision[cls_idx, diff_idx, th_idx:], axis=-1)
                recall[cls_idx, diff_idx, th_idx] = np.max(
                    recall[cls_idx, diff_idx, th_idx:], axis=-1)
    print(iou_size_larger, less)
    print("Aggregating data...")
    AP = 0
    for i in range(1, precision.shape[-1]):
        AP += precision[..., i]
    AP = AP / num_pr_points * 100
    print(AP)
    ret_dict = {}

    ret_str = "\n|AP@%-15s|" % (str(num_pr_points))
    for diff_type in difficulty_types:
        ret_str += '%-12s|' % diff_type
    ret_str += '%-20s|' % 'Occurrence (pred/gt)'
    ret_str += '\n'
    for cls_idx, cur_class in enumerate(classes):
        ret_str += "|%-18s|" % cur_class
        for diff_idx in range(num_difficulties):
            diff_type = difficulty_types[diff_idx]
            key = 'AP_' + cur_class + '/' + diff_type
            # TODO: Adopt correction of TP=0, FP=0 -> AP = 0 for all difficulty
            # types by counting occurrence individually for each difficulty type
            if pred_class_occurrence[cur_class] == 0 and gt_class_occurrence[cur_class] == 0:
                AP[cls_idx,diff_idx] = 100
            ap_score = AP[cls_idx,diff_idx]
            ret_dict[key] = ap_score
            ret_str += "%-12.2f|" % ap_score
        ret_str += '%-20s|' % (str(pred_class_occurrence[cur_class]) + "/" + str(gt_class_occurrence[cur_class]))
        ret_str += "\n"
    mAP = np.mean(AP, axis=0)
    ret_str += "|%-18s|" % 'mAP'
    for diff_idx in range(num_difficulties):
        diff_type = difficulty_types[diff_idx]
        key = 'AP_mean' + '/' + diff_type
        ap_score = mAP[diff_idx]
        ret_dict[key] = ap_score
        ret_str += "%-12.2f|" % ap_score
    ret_str += '%-20s|' % (str(np.sum(list(pred_class_occurrence.values()))) + '/' + str(np.sum(list(gt_class_occurrence.values()))) + ' (Total)')
    ret_str += "\n"

    if print_ok:
        print(ret_str)
    return ret_str, ret_dict

@numba.jit(nopython=True)
def get_thresholds(scores, num_gt, num_pr_points):
    eps = 1e-6
    scores.sort()
    scores = scores[::-1]
    recall_level = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (r_recall + l_recall < 2 * recall_level) and i < (len(scores) - 1):
            continue
        thresholds.append(score)
        recall_level += 1 / num_pr_points
        # avoid numerical errors
        # while r_recall + l_recall >= 2 * recall_level:
        while r_recall + l_recall + eps > 2 * recall_level:
            thresholds.append(score)
            recall_level += 1 / num_pr_points
    return thresholds

@numba.jit(nopython=True)
def accumulate_scores(iou, pred_scores, gt_flag, pred_flag, iou_threshold):
    num_gt = iou.shape[0]
    num_pred = iou.shape[1]
    assigned = np.full(num_pred, False)
    accum_scores = np.zeros(num_gt)
    accum_idx = 0
    for i in range(num_gt):
        if gt_flag[i] == -1: # not the same class
            continue
        det_idx = -1
        detected_score = -1
        for j in range(num_pred):
            if pred_flag[j] == -1: # not the same class
                continue
            if assigned[j]:
                continue
            iou_ij = iou[i, j]
            pred_score = pred_scores[j]
            if (iou_ij > iou_threshold) and (pred_score > detected_score):
                det_idx = j
                detected_score = pred_score

        if (detected_score == -1) and (gt_flag[i] == 0): # false negative
            pass
        elif (detected_score != -1) and (gt_flag[i] == 1 or pred_flag[det_idx] == 1): # ignore
            assigned[det_idx] = True
        elif detected_score != -1: # true positive
            accum_scores[accum_idx] = pred_scores[det_idx]
            accum_idx += 1
            assigned[det_idx] = True

    return accum_scores[:accum_idx]

@numba.jit(nopython=True)
def compute_statistics(iou, pred_scores, gt_flag, pred_flag, score_threshold, iou_threshold):
    num_gt = iou.shape[0]
    num_pred = iou.shape[1]
    assigned = np.full(num_pred, False)
    under_threshold = pred_scores < score_threshold

    tp, fp, fn = 0, 0, 0
    for i in range(num_gt):
        if gt_flag[i] == -1: # different classes
            continue
        det_idx = -1
        detected = False
        best_matched_iou = 0
        gt_assigned_to_ignore = False

        for j in range(num_pred):
            if pred_flag[j] == -1: # different classes
                continue
            if assigned[j]: # already assigned to other GT
                continue
            if under_threshold[j]: # compute only boxes above threshold
                continue
            iou_ij = iou[i, j]
            if (iou_ij > iou_threshold) and (iou_ij > best_matched_iou or gt_assigned_to_ignore) and pred_flag[j] == 0:
                best_matched_iou = iou_ij
                det_idx = j
                detected = True
                gt_assigned_to_ignore = False
            elif (iou_ij > iou_threshold) and (not detected) and pred_flag[j] == 1:
                det_idx = j
                detected = True
                gt_assigned_to_ignore = True

        if (not detected) and gt_flag[i] == 0: # false negative
            fn += 1
        elif detected and (gt_flag[i] == 1 or pred_flag[det_idx] == 1): # ignore
            assigned[det_idx] = True
        elif detected: # true positive
            tp += 1
            assigned[det_idx] = True

    for j in range(num_pred):
        if not (assigned[j] or pred_flag[j] == -1 or pred_flag[j] == 1 or under_threshold[j]):
            fp += 1

    return tp, fp, fn

def filter_data(gt_anno, pred_anno, difficulty_mode, difficulty_level, class_name, use_superclass):
    """
    Filter data by class name and difficulty

    Args:
        gt_anno:
        pred_anno:
        difficulty_mode:
        difficulty_level:
        class_name:

    Returns:
        gt_flags/pred_flags:
            1 : same class but ignored with different difficulty levels
            0 : accepted
           -1 : rejected with different classes
    """
    num_gt = len(gt_anno['name'])
    gt_flag = np.zeros(num_gt, dtype=np.int64)
    if num_gt > 0:
        if use_superclass:
            if class_name == 'VEHICLE':
                reject = np.logical_or(gt_anno['name']=='PEDESTRIAN', np.logical_or(gt_anno['name']=='BICYCLE', gt_anno['name']=='MOTORCYCLE'))
            elif class_name == 'BICYCLE':
                reject = ~np.logical_or(gt_anno['name']=='BICYCLE', gt_anno['name']=='MOTORCYCLE')
            else:
                reject = gt_anno['name'] != class_name
        else:
            print(gt_anno['name'], class_name)
            reject = gt_anno['name'] != class_name
        gt_flag[reject] = -1
    num_pred = len(pred_anno['name'])
    pred_flag = np.zeros(num_pred, dtype=np.int64)
    if num_pred > 0:
        if use_superclass:
            if class_name == 'VEHICLE':
                reject = np.logical_or(pred_anno['name']=='PEDESTRIAN', np.logical_or(pred_anno['name']=='BICYCLE', pred_anno['name']=='MOTORCYCLE'))
            elif class_name == 'BICYCLE':
                reject = ~np.logical_or(pred_anno['name']=='BICYCLE', pred_anno['name']=='MOTORCYCLE')
            else:
                reject = pred_anno['name'] != class_name
        else:
            print(pred_anno['name'], class_name)
            reject = pred_anno['name'] != class_name
        pred_flag[reject] = -1

    if difficulty_mode == 'Overall':
        ignore = overall_filter(gt_anno['boxes_3d'])
        gt_flag[ignore] = 1
        ignore = overall_filter(pred_anno['boxes_3d'])
        pred_flag[ignore] = 1
    elif difficulty_mode == 'Distance':
        ignore = distance_filter(gt_anno['boxes_3d'], difficulty_level)
        gt_flag[ignore] = 1
        ignore = distance_filter(pred_anno['boxes_3d'], difficulty_level)
        pred_flag[ignore] = 1
    elif difficulty_mode == 'Overall&Distance':
        ignore = overall_distance_filter(gt_anno['boxes_3d'], difficulty_level)
        gt_flag[ignore] = 1
        ignore = overall_distance_filter(pred_anno['boxes_3d'], difficulty_level)
        pred_flag[ignore] = 1
    else:
        raise NotImplementedError

    return gt_flag, pred_flag

def iou3d_kernel(gt_boxes, pred_boxes):
    """
    Core iou3d computation (with cuda)

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes: [M, 7]

    Returns:
        iou3d: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(gt_boxes[:, [0, 1, 3, 4, 6]], pred_boxes[:, [0, 1, 3, 4, 6]], criterion=2)
    gt_max_h = gt_boxes[:, [2]] + gt_boxes[:, [5]] * 0.5
    gt_min_h = gt_boxes[:, [2]] - gt_boxes[:, [5]] * 0.5
    pred_max_h = pred_boxes[:, [2]] + pred_boxes[:, [5]] * 0.5
    pred_min_h = pred_boxes[:, [2]] - pred_boxes[:, [5]] * 0.5
    max_of_min = np.maximum(gt_min_h, pred_min_h.T)
    min_of_max = np.minimum(gt_max_h, pred_max_h.T)
    inter_h = min_of_max - max_of_min
    inter_h[inter_h <= 0] = 0
    #inter_h[intersection_2d <= 0] = 0
    intersection_3d = intersection_2d * inter_h
    gt_vol = gt_boxes[:, [3]] * gt_boxes[:, [4]] * gt_boxes[:, [5]]
    pred_vol = pred_boxes[:, [3]] * pred_boxes[:, [4]] * pred_boxes[:, [5]]
    union_3d = gt_vol + pred_vol.T - intersection_3d
    #eps = 1e-6
    #union_3d[union_3d<eps] = eps
    iou3d = intersection_3d / union_3d
    return iou3d

def iou3d_kernel_with_heading(gt_boxes, pred_boxes):
    """
    Core iou3d computation (with cuda)

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes: [M, 7]

    Returns:
        iou3d: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(gt_boxes[:, [0, 1, 3, 4, 6]], pred_boxes[:, [0, 1, 3, 4, 6]], criterion=2)
    gt_max_h = gt_boxes[:, [2]] + gt_boxes[:, [5]] * 0.5
    gt_min_h = gt_boxes[:, [2]] - gt_boxes[:, [5]] * 0.5
    pred_max_h = pred_boxes[:, [2]] + pred_boxes[:, [5]] * 0.5
    pred_min_h = pred_boxes[:, [2]] - pred_boxes[:, [5]] * 0.5
    max_of_min = np.maximum(gt_min_h, pred_min_h.T)
    min_of_max = np.minimum(gt_max_h, pred_max_h.T)
    inter_h = min_of_max - max_of_min
    inter_h[inter_h <= 0] = 0
    #inter_h[intersection_2d <= 0] = 0
    intersection_3d = intersection_2d * inter_h
    gt_vol = gt_boxes[:, [3]] * gt_boxes[:, [4]] * gt_boxes[:, [5]]
    pred_vol = pred_boxes[:, [3]] * pred_boxes[:, [4]] * pred_boxes[:, [5]]
    union_3d = gt_vol + pred_vol.T - intersection_3d
    #eps = 1e-6
    #union_3d[union_3d<eps] = eps
    iou3d = intersection_3d / union_3d

    # rotation orientation filtering
    diff_rot = gt_boxes[:, [6]] - pred_boxes[:, [6]].T
    diff_rot = np.abs(diff_rot)
    reverse_diff_rot = 2 * np.pi - diff_rot
    diff_rot[diff_rot >= np.pi] = reverse_diff_rot[diff_rot >= np.pi] # constrain to [0-pi]
    iou3d[diff_rot > np.pi/2] = 0 # unmatched if diff_rot > 90
    return iou3d


def rotate_iou_kernel_eval(gt_boxes, pred_boxes):
    iou3d_cpu = rotate_iou_cpu_eval(gt_boxes, pred_boxes)
    return iou3d_cpu


def compute_iou3d(gt_annos, pred_annos, split_parts, with_heading):
    """
    Compute iou3d of all samples by parts

    Args:
        with_heading: filter with heading
        gt_annos: list of dicts for each sample
        pred_annos:
        split_parts: for part-based iou computation

    Returns:
        ious: list of iou arrays for each sample
    """
    gt_num_per_sample = np.stack([len(anno["name"]) for anno in gt_annos], 0)
    pred_num_per_sample = np.stack([len(anno["name"]) for anno in pred_annos], 0)
    ious = []
    sample_idx = 0
    for num_part_samples in split_parts:
        gt_annos_part = gt_annos[sample_idx:sample_idx + num_part_samples]
        pred_annos_part = pred_annos[sample_idx:sample_idx + num_part_samples]

        gt_boxes = np.concatenate([anno["boxes_3d"] for anno in gt_annos_part], 0)
        pred_boxes = np.concatenate([anno["boxes_3d"] for anno in pred_annos_part], 0)

        if with_heading:
            iou3d_part = iou3d_kernel_with_heading(gt_boxes, pred_boxes)
        else:
            iou3d_part = iou3d_kernel(gt_boxes, pred_boxes)

        gt_num_idx, pred_num_idx = 0, 0
        for idx in range(num_part_samples):
            gt_box_num = gt_num_per_sample[sample_idx + idx]
            pred_box_num = pred_num_per_sample[sample_idx + idx]
            ious.append(iou3d_part[gt_num_idx: gt_num_idx + gt_box_num, pred_num_idx: pred_num_idx+pred_box_num])
            gt_num_idx += gt_box_num
            pred_num_idx += pred_box_num
        sample_idx += num_part_samples
    return ious


def compute_iou3d_cpu(gt_annos, pred_annos):
    ious = []
    gt_num = len(gt_annos)
    print("Compute IoU...")
    for i in tqdm(range(gt_num)):
        gt_boxes = gt_annos[i]['boxes_3d']
        pred_boxes = pred_annos[i]['boxes_3d']

        iou3d_part = rotate_iou_cpu_eval(gt_boxes, pred_boxes)
        ious.append(iou3d_part)
    return ious


def prepare_a9_dataset_ground_truth(labels_path, object_min_points=0, ouster_lidar_only=False):
    def append_object(pc, l, w, h, rotation, position_3d, category):
        obb = o3d.geometry.OrientedBoundingBox(np.array(position_3d), np.array([[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]]), np.array([l, w, h]))
        n_object_points = len(obb.get_point_indices_within_bounding_box(pc.points))
        # Ground truth is labeled with camera data, so there are objects
        # contained in the ground truth without a single corresponding
        # point in the LiDAR point cloud.
        # You can specify how many minimum points there should be before a label
        # is included.
        if n_object_points >= object_min_points:
            name.append(category.capitalize())
            boxes_3d.append(np.hstack((position_3d, l, w, h, rotation)))
            num_points_in_gt.append(n_object_points)

    if os.path.isfile(labels_path):
        pc_path = get_pc_path(os.path.dirname(labels_path))
        labels_file_list = [labels_path]
    else:
        label_path, pc_path = get_pc_path(labels_path)
        labels_file_list = glob.glob(label_path)
        labels_file_list.sort()
    labels_list = []

    for label_file in tqdm(labels_file_list):
        if ouster_lidar_only and 'ouster' not in label_file:
            continue
        name = []
        boxes_3d = []
        num_points_in_gt = []
        json_file = open(label_file, )
        json_data = json.load(json_file)

        if "openlabel" in json_data:
            key_number = list(json_data["openlabel"]["frames"].keys())[0]
            pc_file = os.path.join(pc_path, json_data["openlabel"]["frames"][key_number]["frame_properties"]["point_cloud_file_name"])
            pc = o3d.io.read_point_cloud(pc_file)
            for k, v in json_data["openlabel"]["frames"].items():
                for label in v["objects"].values():
                    # Dataset in ASAM OpenLABEL format
                    l = float(label["object_data"]["cuboid"]["val"][7])
                    w = float(label["object_data"]["cuboid"]["val"][8])
                    h = float(label["object_data"]["cuboid"]["val"][9])
                    quat_x = float(label["object_data"]["cuboid"]["val"][3])
                    quat_y = float(label["object_data"]["cuboid"]["val"][4])
                    quat_z = float(label["object_data"]["cuboid"]["val"][5])
                    quat_w = float(label["object_data"]["cuboid"]["val"][6])
                    if np.linalg.norm([quat_x, quat_y, quat_z, quat_w]) == 0.0:
                        continue
                    rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler('zyx', degrees=False)[0]
                    position_3d = [
                        float(label["object_data"]["cuboid"]["val"][0]),
                        float(label["object_data"]["cuboid"]["val"][1]),
                        float(label["object_data"]["cuboid"]["val"][2])  # - h / 2  # To avoid floating bounding boxes
                    ]
                    append_object(pc, l, w, h, rotation, position_3d, label["object_data"]["type"])
        else:
            pc_file = os.path.join(pc_path, json_data["lidar_name"] + '.pcd')
            pc = o3d.io.read_point_cloud(pc_file)
            for label in json_data["labels"]:
                if "dimensions" in label:
                    # Dataset R1 NOT IN ASAM OpenLABEL format
                    l = float(label["dimensions"]["length"])
                    w = float(label["dimensions"]["width"])
                    h = float(label["dimensions"]["height"])
                    quat_x = float(label["rotation"]["_x"])
                    quat_y = float(label["rotation"]["_y"])
                    quat_z = float(label["rotation"]["_z"])
                    quat_w = float(label["rotation"]["_w"])
                    rotation = R.from_quat([quat_x, quat_y, quat_z, quat_w]).as_euler('zyx', degrees=False)[0]
                    position_3d = [
                        float(label["center"]["x"]),
                        float(label["center"]["y"]),
                        float(label["center"]["z"]) - h / 2  # To avoid floating bounding boxes
                    ]
                else:
                    # Dataset R0 NOT IN ASAM OpenLABEL format
                    l = float(label["box3d"]["dimension"]["length"])
                    w = float(label["box3d"]["dimension"]["width"])
                    h = float(label["box3d"]["dimension"]["height"])
                    rotation = float(label["box3d"]["orientation"]["rotationYaw"])
                    position_3d = [
                        float(label["box3d"]["location"]["x"]),
                        float(label["box3d"]["location"]["y"]),
                        float(label["box3d"]["location"]["z"])
                    ]
                append_object(pc, l, w, h, rotation, position_3d, label["category"])
        json_file.close()
        label_dict = {'name': np.array([x.upper() for x in name]), 'boxes_3d': np.array(boxes_3d), 'num_points_in_gt': np.array(num_points_in_gt)}
        labels_list.append(label_dict) 
    return labels_list


def get_pc_path(labels_path):
    parent_dir = os.path.dirname(labels_path)
    subdirs = os.listdir(parent_dir)
    label_dir = os.path.basename(labels_path)
    return os.path.join(parent_dir, subdirs[0], "*"), os.path.join(parent_dir, subdirs[1])


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]


def prepare_predictions(predictions_path):
    if os.path.isfile(predictions_path):
        predictions_file_list = [predictions_path]
    else:
        predictions_file_list = listdir_fullpath(predictions_path)
        predictions_file_list.sort()
    predictions_list = []
    for prediction_file in tqdm(predictions_file_list):
        name = []
        boxes_3d = []
        with open(prediction_file, 'r') as f:
            for line in f:
                line = line.rstrip().split()
                prediction = [float(item) for item in line[1:]]
                prediction.insert(0, line[0])
                name.append(prediction[0])
                boxes_3d.append(prediction[1:])
        prediction_dict = {'name': np.array(name), 'boxes_3d': np.array(boxes_3d)}
        predictions_list.append(prediction_dict) 
    return predictions_list


def visualize_bounding_boxes(label_path, prediction_path, object_min_points=0, ouster_lidar_only=False):
    def rreplace(s, old, new):
        """ Reverse replace """
        return (s[::-1].replace(old[::-1],new[::-1], 1))[::-1]
    gt_data = prepare_a9_dataset_ground_truth(label_path, object_min_points, ouster_lidar_only)
    pred_data = prepare_predictions(prediction_path)
    for item in pred_data:
        n_obj = item['name'].size
        item['score'] = np.full(n_obj, 1)
    classes = ['Car', 'Bus', 'Truck', 'PEDESTRIAN', 'BICYCLE', 'Trailer', 'Van', 'MOTORCYCLE', 'Emergency_Vehicle', 'Other']
    result_str, _ = get_evaluation_results(gt_data, pred_data, classes, use_superclass=False, difficulty_mode='Overall')
    print(result_str)

    pc_file = os.path.join(get_pc_path(os.path.dirname(label_path)), rreplace(os.path.basename(label_path), '.json', '.pcd'))
    pc = o3d.io.read_point_cloud(pc_file)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Point Cloud Visualizer')
    vis.get_render_option().background_color = [0.1, 0.1, 0.1]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True
    vis.add_geometry(pc)
    for box in gt_data[0]['boxes_3d']:
        obb = o3d.geometry.OrientedBoundingBox(box[:3], np.array([[np.cos(box[6]), np.sin(box[6]), 0], [-np.sin(box[6]), np.cos(box[6]), 0], [0, 0, 1]]), box[3:6])
        obb.color = np.array([0, 1, 0])
        vis.add_geometry(obb)
    for box in pred_data[0]['boxes_3d']:
        obb = o3d.geometry.OrientedBoundingBox(box[:3], np.array([[np.cos(box[6]), np.sin(box[6]), 0], [-np.sin(box[6]), np.cos(box[6]), 0], [0, 0, 1]]), box[3:6])
        obb.color = np.array([1, 0, 0])
        vis.add_geometry(obb)
    vis.get_view_control().set_zoom(0.05)
    vis.get_view_control().set_front([-0.940, 0.096, 0.327])
    vis.get_view_control().set_lookat([17.053, 0.544, -2.165])
    vis.get_view_control().set_up([0.327, -0.014, 0.945])
    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    import pickle
    import copy
    import argparse
    argparser = argparse.ArgumentParser(description='Evaluating A9-Dataset')
    argparser.add_argument('-gt', '--folder_path_ground_truth', type=str, default="",
                           help='Ground truth folder path.')
    argparser.add_argument('-p', '--folder_path_predictions', type=str, default="",
                           help='Predictions folder path.')
    argparser.add_argument('--object_min_points', type=int, default=0,
                           help='Minimum point per object before being considered.')
    args = argparser.parse_args()
    folder_path_ground_truth = args.folder_path_ground_truth
    folder_path_predictions = args.folder_path_predictions
    object_min_points = args.object_min_points
    if os.path.isfile(folder_path_ground_truth) and os.path.isfile(folder_path_predictions):
        visualize_bounding_boxes(folder_path_ground_truth,
                                folder_path_predictions,
                                object_min_points=object_min_points, ouster_lidar_only=False)
    else:                            
        # info_data = pickle.load(open(os.path.normpath(os.path.join(__file__, '../submission_format/once_infos_val.pkl')), 'rb'))[:3] # you can find this file in once_devkit/submission_format/
        # pred_data = pickle.load(open('result.pkl', 'rb')) # your prediction file
        # gt_data = list()
        # for item in info_data:
        #     if 'annos' in item:
        #         gt_data.append(item['annos'])
        # Overwriting original ONCE ground truth with A9-Dataset ground truth
        classes = ['CAR', 'BUS', 'TRUCK', 'PEDESTRIAN', 'BICYCLE', 'TRAILER', 'VAN', 'MOTORCYCLE', 'EMERGENCY_VEHICLE']
        print("Prepare ground truth")
        gt_data = prepare_a9_dataset_ground_truth(folder_path_ground_truth, object_min_points=object_min_points, ouster_lidar_only=False)
        # Use predictions or ground truth
        print("Preapre predictions")
        pred_data = prepare_predictions(folder_path_predictions)
        # pred_data = copy.deepcopy(gt_data)
        for item in pred_data:
            n_obj = item['name'].size
            item['score'] = np.full(n_obj, 1)
        
        print("Evaluating...")
        result_str, result_dict = get_evaluation_results(gt_data, pred_data, classes, use_superclass=False, difficulty_mode='Overall')
        print(result_str)
