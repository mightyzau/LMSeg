# Cross datasets training
# 使用多个dataloaders 进行改造，使得每张卡采样的samples来自同一个dataset。避免不同datasets因为crop_size不同导致可能出现的精度问题。

import copy
import itertools
import logging
import os, sys
from collections import OrderedDict
from time import daylight
from typing import Any, Dict, List, Set
import argparse
import weakref
import time
import random
import numpy as np

import torch
import torch.utils.data as torchdata
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.config import configurable
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, MapDataset, build_batch_data_loader, get_detection_dataset_dicts, DatasetMapper
from detectron2.data.samplers import TrainingSampler, RepeatFactorTrainingSampler
#from detectron2.data.build import _train_loader_from_config
from detectron2.engine import DefaultTrainer, default_setup, launch, create_ddp_model, AMPTrainer, SimpleTrainer
#from detectron2.engine import default_argument_parser
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    SemSegEvaluator,
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger, _log_api_usage, log_first_n

# MaskFormer
from mask_former import (
    SemanticSegmentorWithTTA,
    add_mask_former_config,
    add_lmseg_ade20k_input, add_lmseg_cityscapes_input, add_lmseg_cocostuff10k_input, add_lmseg_mapillary_vistas_input,
    add_lmseg_ade20k_panoptic_input, add_lmseg_cityscapes_panoptic_input, add_lmseg_coco_panoptic_input,
    add_lmseg_ade20k_full_input,
    LMSegSemanticDatasetMapper,
    LMSegPanopticDatasetMapper
)


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
        Examples:

        Run on single machine:
            $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

        Change some config options:
            $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

        Run on multiple machines:
            (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
            (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    parser.add_argument('--local_rank', type=int, default=-1)

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2**15 + 2**14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2**14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )

    parser.add_argument('--test_config_file', type=str)
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


## rewrite for cross datasets
def _train_loader_from_config(cfg, mapper=None, *, dataset_list=[], sampler_list=[]):
    if len(dataset_list) == 0:
        for d in cfg.DATASETS.TRAIN:
            dataset = get_detection_dataset_dicts(
                d,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if cfg.MODEL.KEYPOINT_ON
                else 0,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            )
            _log_api_usage("dataset." + cfg.DATASETS.TRAIN[0])
            dataset_list.append(dataset)

    if mapper is None:
        mapper = DatasetMapper(cfg, True)

    if len(sampler_list) == 0:
        for dataset in dataset_list:
            sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
            logger = logging.getLogger(__name__)
            if isinstance(dataset, torchdata.IterableDataset):
                logger.info("Not using any sampler since the dataset is IterableDataset.")
                sampler = None
            else:
                logger.info("Using training sampler {}".format(sampler_name))
                if sampler_name == "TrainingSampler":
                    sampler = TrainingSampler(len(dataset))
                elif sampler_name == "RepeatFactorTrainingSampler":
                    repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                        dataset, cfg.DATALOADER.REPEAT_THRESHOLD
                    )
                    sampler = RepeatFactorTrainingSampler(repeat_factors)
                #elif sampler_name == "RandomSubsetTrainingSampler":
                #    sampler = RandomSubsetTrainingSampler(
                #        len(dataset), cfg.DATALOADER.RANDOM_SUBSET_RATIO
                #    )
                else:
                    raise ValueError("Unknown training sampler: {}".format(sampler_name))
            sampler_list.append(sampler)

    return {
        "dataset_list": dataset_list,
        "sampler_list": sampler_list,
        "mapper": mapper,
        "total_batch_size": cfg.SOLVER.IMS_PER_BATCH,
        "aspect_ratio_grouping": cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        "num_workers": cfg.DATALOADER.NUM_WORKERS,
        "cfg": cfg.clone(),
    }

## rewrite for cross datasets
@configurable(from_config=_train_loader_from_config)
def build_detection_train_loader(
    dataset_list,
    *,
    mapper,
    sampler_list=None,
    total_batch_size,
    aspect_ratio_grouping=True,
    num_workers=0,
    collate_fn=None,
    cfg=None,
):
    """
    Build a dataloader for object detection with some default features.

    Args:
        dataset (list or torch.utils.data.Dataset): a list of dataset dicts,
            or a pytorch dataset (either map-style or iterable). It can be obtained
            by using :func:`DatasetCatalog.get` or :func:`get_detection_dataset_dicts`.
        mapper (callable): a callable which takes a sample (dict) from dataset and
            returns the format to be consumed by the model.
            When using cfg, the default choice is ``DatasetMapper(cfg, is_train=True)``.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces
            indices to be applied on ``dataset``.
            If ``dataset`` is map-style, the default sampler is a :class:`TrainingSampler`,
            which coordinates an infinite random shuffle sequence across all workers.
            Sampler must be None if ``dataset`` is iterable.
        total_batch_size (int): total batch size across all workers.
        aspect_ratio_grouping (bool): whether to group images with similar
            aspect ratio for efficiency. When enabled, it requires each
            element in dataset be a dict with keys "width" and "height".
        num_workers (int): number of parallel data loading workers
        collate_fn: a function that determines how to do batching, same as the argument of
            `torch.utils.data.DataLoader`. Defaults to do no collation and return a list of
            data. No collation is OK for small batch size and simple data structures.
            If your batch size is large and each sample contains too many small tensors,
            it's more efficient to collate them in data loader.

    Returns:
        torch.utils.data.DataLoader:
            a dataloader. Each output from it is a ``list[mapped_element]`` of length
            ``total_batch_size / num_workers``, where ``mapped_element`` is produced
            by the ``mapper``.
    """
    assert isinstance(dataset_list, list)
    dataset_lens = [len(d) for d in dataset_list]
    logger = logging.getLogger("detectron2")
    logger.info('sampls of datasets: {}'.format(dataset_lens))
    logger.info('{} dataset used, with total {} samples'.format(len(dataset_list), sum(dataset_lens)))

    if mapper is not None:
        dataset_list = [MapDataset(d, mapper) for d in dataset_list]

    data_loads = []
    for _i, (_dataset, sampler) in enumerate(zip(dataset_list, sampler_list)):
        if isinstance(_dataset, torchdata.IterableDataset):
            assert sampler is None, "sampler must be None if dataset is IterableDataset"
        else:
            if sampler is None:
                sampler = TrainingSampler(len(_dataset))
            assert isinstance(sampler, torchdata.Sampler), f"Expect a Sampler but got {type(sampler)}"
        
        try:
            loader =  build_batch_data_loader(
                    _dataset,
                    sampler,
                    total_batch_size,
                    aspect_ratio_grouping=aspect_ratio_grouping,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                )
        except:
            loader =  build_batch_data_loader(
                    _dataset,
                    sampler,
                    total_batch_size,
                    aspect_ratio_grouping=aspect_ratio_grouping,
                    num_workers=num_workers,
                )

        data_loads.append(loader)
    return data_loads, dataset_lens

## rewrite for cross datasets
class SimpleTrainer_New(SimpleTrainer):
    def __init__(self, model, data_loaders, optimizer, dataset_lens, accum_iter=1):
        super(SimpleTrainer, self).__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loaders = data_loaders
        # to access the data loader iterator, call `self._data_loader_iter`
        self._data_loader_iter_objs = None
        self.optimizer = optimizer
        self.num_datasets = len(self.data_loaders)

        assert len(dataset_lens) == len(data_loaders)
        self.dataset_probs = [d / sum(dataset_lens) for d in dataset_lens]
        self.accum_iter = accum_iter
    
    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_objs is None:
            self._data_loader_iter_objs = [iter(d) for d in self.data_loaders]
        return self._data_loader_iter_objs
    
    def reset_data_loader(self, data_loaders):
        del self.data_loaders
        self.data_loaders = data_loaders
        self._data_loader_iter_objs = None

    def run_step(self, iter, same_dataset_cross_gpus=False, sample_strategy='', prob_given=None):
        """
        Implement the standard training logic described above.

        same_dataset_cross_gpus: if True, all gpus select the same dataset index for one batch
        """
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        if same_dataset_cross_gpus:
            np.random.seed(iter)
        
        if sample_strategy == 'uniform':
            data_ind = np.random.choice(self.num_datasets)                              # random choose one dataset
        elif sample_strategy == 'prob_data_length':
            data_ind = np.random.choice(self.num_datasets, p=self.dataset_probs)        # 根据dataset的样本数目比例，选择dataset
        elif sample_strategy == 'prob_given':
            assert len(prob_given) == self.num_datasets
            if sum(prob_given) != 1:
                prob_given = [d / sum(prob_given) for d in prob_given]
            data_ind = np.random.choice(self.num_datasets, p=prob_given)
        else:
            raise ValueError()

        data = next(self._data_loader_iter[data_ind])

        # do some check
        #_data_dir = os.path.join(*data[0]['file_name'].split('/')[:2])              # e.g., "datasets/ADEChallengeData2016/images/training/ADE_train_00012229.jpg"
        #for _d in data:
        #    assert os.path.join(*_d['file_name'].split('/')[:2]) == _data_dir

        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        losses.backward()

        self._write_metrics(loss_dict, data_time)

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4
        """
        self.optimizer.step()

## rewrite for cross datasets
class AMPTrainer_New(SimpleTrainer_New):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """
    def __init__(self, model, data_loaders, optimizer, grad_scaler=None):
        """
        Args:
            model, data_loader, optimizer: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super(AMPTrainer_New, self).__init__(model, data_loaders, optimizer)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler

    def run_step(self, iter, same_dataset_cross_gpus=False, sample_strategy='', prob_given=None):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()

        if same_dataset_cross_gpus:
            np.random.seed(iter)

        if sample_strategy == 'uniform':
            data_ind = np.random.choice(self.num_datasets)                              # random choose one dataset
        elif sample_strategy == 'prob_data_length':
            data_ind = np.random.choice(self.num_datasets, p=self.dataset_probs)        # 根据dataset的样本数目比例，选择dataset
        elif sample_strategy == 'prob_given':
            if sum(prob_given) != 1:
                prob_given = [d / sum(prob_given) for d in prob_given]
            data_ind = np.random.choice(self.num_datasets, p=prob_given)
        else:
            raise ValueError()

        data = next(self._data_loader_iter[data_ind])
        data_time = time.perf_counter() - start

        with autocast():
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])



class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loaders, dataset_lens = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
        self._trainer = (AMPTrainer_New if cfg.SOLVER.AMP.ENABLED else SimpleTrainer_New)(
            model, data_loaders, optimizer, dataset_lens)

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

        logger.info('sampling_strage setting')
        logger.info('    sample_strategy: {}'.format(cfg.LMSEG.SAMPLE_STRATEGY))
        logger.info('    prob_given: {}'.format(cfg.LMSEG.SAMPLE_PROB_GIVEN))
        logger.info('    same_dataset_cross_gpus: {}'.format(cfg.LMSEG.SAME_DATASET_CROSS_GPUS))
    
    def run_step(self):
        self._trainer.iter = self.iter
        same_dataset_cross_gpus = self.cfg.LMSEG.SAME_DATASET_CROSS_GPUS
        sample_strategy = self.cfg.LMSEG.SAMPLE_STRATEGY
        prob_given = self.cfg.LMSEG.SAMPLE_PROB_GIVEN
        self._trainer.run_step(self.iter, same_dataset_cross_gpus=same_dataset_cross_gpus, sample_strategy=sample_strategy, prob_given=prob_given)

        if (self.cfg.LMSEG.EMPTY_CACHE_STEP > 1) and (self.iter % self.cfg.LMSEG.EMPTY_CACHE_STEP == 0):
            torch.cuda.empty_cache()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
        ]:
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            assert (
                torch.cuda.device_count() >= comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            evaluator_list.append(CityscapesSemSegEvaluator(dataset_name))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.LMSEG.TASK_TYPE == 'semantic_segmentation':
            mapper = LMSegSemanticDatasetMapper(cfg, True)
        elif cfg.LMSEG.TASK_TYPE == 'panoptic_segmentation':
            mapper = LMSegPanopticDatasetMapper(cfg, True)
        else:
            raise ValueError('not supported cfg.LMSEG.TASK_TYPE: {}'.format(cfg.LMSEG.TASK_TYPE))

        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
                    
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                
                if "text_encoder" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.TEXTENCODER_MULTIPLIER
                    
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
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
    if args.eval_only:
        if args.test_config_file is not None:
            cfg.merge_from_file(args.test_config_file)
            
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "mask_former" module
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="mask_former")
    return cfg


def main(args):
    cfg = setup(args)
    if cfg.PDB_DEBUG:
        import pdb; pdb.set_trace()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    logger = logging.getLogger("detectron2.trainer")
    logger.info("Starting training from iteration {}, with max iteration {}".format(trainer.start_iter, trainer.max_iter))

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
