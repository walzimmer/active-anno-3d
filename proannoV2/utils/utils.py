import os
import numpy as np
import argparse
import json
import pickle as pkl
import datetime
import glob
import random
import tqdm
import wandb
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pcdet.models import build_network, model_fn_decorator
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.utils import common_utils
from pcdet.datasets.tumtraf.tumtraf_converter import TUMTraf2KITTI_v2
from pcdet.datasets.tumtraf.tumtraf_dataset import create_tumtraf_infos
from pcdet.datasets.tumtraf.tumtraf_utils import create_image_sets
from tools.utils.train_utils.optimization import build_optimizer, build_scheduler



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

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class TorchEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist() 
        return json.JSONEncoder.default(self, obj)





if __name__ == "__main__":
    pass