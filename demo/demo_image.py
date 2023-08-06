# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import glob
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import tempfile
import time
import warnings

import cv2
import numpy as np
import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

from mask_former import (
    add_mask_former_config,
    add_lmseg_ade20k_input, add_lmseg_cityscapes_input, add_lmseg_cocostuff10k_input, add_lmseg_mapillary_vistas_input,
    add_lmseg_ade20k_panoptic_input, add_lmseg_cityscapes_panoptic_input, add_lmseg_coco_panoptic_input,
    add_lmseg_ade20k_full_input,
)


# constants
WINDOW_NAME = "MaskFormer demo"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_mask_former_config(cfg)
    add_lmseg_ade20k_input(cfg)
    add_lmseg_ade20k_full_input(cfg)
    add_lmseg_cityscapes_input(cfg)
    add_lmseg_cocostuff10k_input(cfg)
    add_lmseg_mapillary_vistas_input(cfg)
    add_lmseg_coco_panoptic_input(cfg)
    add_lmseg_ade20k_panoptic_input(cfg)
    add_lmseg_cityscapes_panoptic_input(cfg)

    cfg.merge_from_file(args.config_file)

    if args.test_config_file is not None:
        cfg.merge_from_file(args.test_config_file)

    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/ade20k-150/maskformer_R50_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--image_file", type=str)
    parser.add_argument("--image_dir", type=str)
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument('--test_config_file', type=str)
    parser.add_argument('--pdb_debug', action='store_true')

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    if args.pdb_debug:
        import pdb; pdb.set_trace()
        
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    image_files = []
    if args.image_file is not None:
        image_files.append(args.image_file)
    if args.image_dir is not None:
        image_files += list(glob.glob(os.path.join(args.image_dir, '*.png')))

    for path in tqdm.tqdm(image_files, disable=not args.output):
        # use PIL, to be consistent with evaluation
        img = read_image(path, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        # predictoins: dict_keys(['sem_seg', 'panoptic_seg', 'mask_cls_result'])
        
        logger.info(
            "{}: {} in {:.2f}s".format(
                path,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

        if args.output:
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(path))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)

