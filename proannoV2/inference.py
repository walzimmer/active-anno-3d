import os
import numpy as np
import argparse
import json
import pickle as pkl
import datetime
from pathlib import Path
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.utils import common_utils
from proannoV2.utils.utils import TorchEncoder, NumpyEncoder
from proannoV2.utils.infer_utils import *



def parse_config():
        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--op', type=str, default='inference')
        parser.add_argument('--filenames', nargs='+', help='filename for inference')

        args = parser.parse_args()

        cfg_file = '/ahmed/tools/cfgs/proannoV2/pv_rcnn_inference.yaml'
        cfg_from_yaml_file(cfg_file, cfg)

        print("filenames: ", args.filenames)
        cfg.MODEL.OPERATION = args.op
        # the KITTI format directory
        cfg.DATA_CONFIG.ROOT_DIR = cfg.DATA_CONFIG.ROOT_DIR + '/currently_annotating'


        return args, cfg



if __name__ == "__main__":
    
    args, cfg = parse_config()

    ckpt_dir = '/ahmed/tools/proannoV2_models'
    ckpt = os.path.join(ckpt_dir, sorted(os.listdir(ckpt_dir))[-1])

    output_dir = Path(f'/ahmed/output/proannoV2/{args.op}')
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_inference_%s.txt' % (datetime.datetime.now().strftime('%Y_%m_%d_%H')))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # because we save the detections in the openlabel format
    openlabel_root_dir = Path('/'.join(cfg.DATA_CONFIG.ROOT_DIR.split('/')[:-2]) + '/OpenLABEL')
    dets_save_dir = openlabel_root_dir / 'currently_annotating' / 'detections'
    dets_save_dir.mkdir(parents=True, exist_ok=True)
    vis_save_dir = openlabel_root_dir / 'currently_annotating'/ 'visualizaions'
    vis_save_dir.mkdir(parents=True, exist_ok=True)


    if not os.path.isdir(cfg.DATA_CONFIG.ROOT_DIR):
        os.makedirs(cfg.DATA_CONFIG.ROOT_DIR, exist_ok=True)
        prepare_data_current_annotation(cfg, logger)


    preds = model_inference(frame_ids=args.filenames,
                            cfg=cfg,
                            ckpt=ckpt,
                            logger=logger)
    

    preds = inference_post_processing(predictions=preds,
                                      frame_ids=args.filenames,
                                      openlabel_dir=openlabel_root_dir)
    
    print(preds)


