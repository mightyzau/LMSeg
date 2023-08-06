#coding=utf-8

from typing import Tuple
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from collections import defaultdict

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from .modeling.text_heads.build import build_text_encoder
from .modeling.text_heads.tokenize import tokenize

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from .lmseg_model import MLPLayers


@META_ARCH_REGISTRY.register()
class LMSEG_Panoptic(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,

        text_encoder,
        fixed_contexts,
        learnable_contexts,
        text_adapter,

        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        metadata,
        size_divisibility,
        sem_seg_postprocess_before_inference: bool,
        panoptic_test_configs,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone

        self.text_encoder = text_encoder
        self.fixed_contexts = fixed_contexts
        self.learnable_contexts = learnable_contexts
        self.text_adapter = text_adapter

        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.panoptic_test_configs = panoptic_test_configs

        self.metadata = metadata
        for k in size_divisibility:
            if size_divisibility[k] < 0:
                size_divisibility[k] = self.backbone.size_divisibility

        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.tau = nn.Parameter(torch.ones(1) * 0.07)
        if self.cfg.LMSEG.CLS_EMBED_MLP_LAYERS > 0:
            self.cls_embed_mlp = MLPLayers(256, 256, 256, self.cfg.LMSEG.CLS_EMBED_MLP_LAYERS)
        else:
            self.cls_embed_mlp = nn.Identity()

        self.text_embedding_buffer = {}

    
    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)

        text_encoder = build_text_encoder(cfg)

        fixed_contexts = {}
        # metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[3])
        for dataset_name in cfg.DATASETS.TRAIN + cfg.DATASETS.TEST:
            assert 'panoptic' in dataset_name

            class_names = MetadataCatalog.get(dataset_name).get('stuff_classes')
            # 如果有同义词，取第一个
            class_names = [c.split(',')[0] for c in class_names]
            class_names += ['background']
            fixed_contexts_d = torch.cat([tokenize(c, context_length=cfg.MODEL.TEXTENCODER.HARD_PROMPT_LENGTH) for c in class_names])

            if 'ade20k' in dataset_name:
                assert len(class_names) == 150 + 1
                fixed_contexts['ade20k_panoptic'] = (fixed_contexts_d.clone(), class_names)
            
            if 'cityscapes' in dataset_name:
                assert len(class_names) == 19 + 1
                fixed_contexts['cityscapes_panoptic'] = (fixed_contexts_d.clone(), class_names)

            if 'coco' in dataset_name:
                assert len(class_names) == 133 + 1
                fixed_contexts['coco_panoptic'] = (fixed_contexts_d.clone(), class_names)
                
        # learnable prompt embedding
        token_embed_dim = cfg.MODEL.TEXTENCODER.TOKEN_EMBED_DIM
        
        context_length = text_encoder.context_length - cfg.MODEL.TEXTENCODER.HARD_PROMPT_LENGTH
        if not cfg.LMSEG.ENABLE_LEARNABLE_CONTEXT:
            assert context_length == 0
        else:
            assert context_length > 0

        if context_length > 0:
            learnable_contexts = nn.Parameter(torch.randn(1, context_length, token_embed_dim))       # (1, 8, 512)
            nn.init.trunc_normal_(learnable_contexts)
        else:
            learnable_contexts = None

        text_dim = cfg.MODEL.TEXTENCODER.TEXT_DIM
        text_adapter = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, 256)
        )

        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION       # True
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT       # 0.1
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT                 # 1.0
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT                 # 20.0

        # building criterion
        matcher = HungarianMatcher(
            cost_class=1,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
        )

        weight_dict = {"loss_ce": 1, "loss_mask": mask_weight, "loss_dice": dice_weight}
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            num_classes=-1,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
        )

        size_divisibility = {
            'ade20k_panoptic': cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY_ADE20K_PANOPTIC,
            'coco_panoptic': cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY_COCO_PANOPTIC,
            'cityscapes_panoptic': cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY_CITYSCAPES_PANOPTIC,
        }

        panoptic_test_configs = {}
        panoptic_test_configs['coco_panoptic'] = {
            'panoptic_on': cfg.MODEL.MASK_FORMER.TEST_COCO_PANOPTIC.PANOPTIC_ON,
            'object_mask_threshold': cfg.MODEL.MASK_FORMER.TEST_COCO_PANOPTIC.OBJECT_MASK_THRESHOLD,
            'overlap_threshold': cfg.MODEL.MASK_FORMER.TEST_COCO_PANOPTIC.OVERLAP_THRESHOLD,
        }
        panoptic_test_configs['ade20k_panoptic'] = {
            'panoptic_on': cfg.MODEL.MASK_FORMER.TEST_ADE20K_PANOPTIC.PANOPTIC_ON,
            'object_mask_threshold': cfg.MODEL.MASK_FORMER.TEST_ADE20K_PANOPTIC.OBJECT_MASK_THRESHOLD,
            'overlap_threshold': cfg.MODEL.MASK_FORMER.TEST_ADE20K_PANOPTIC.OVERLAP_THRESHOLD,
        }

        panoptic_test_configs['cityscapes_panoptic'] = {
            'panoptic_on': cfg.MODEL.MASK_FORMER.TEST_CITYSCAPES_PANOPTIC.PANOPTIC_ON,
            'object_mask_threshold': cfg.MODEL.MASK_FORMER.TEST_CITYSCAPES_PANOPTIC.OBJECT_MASK_THRESHOLD,
            'overlap_threshold': cfg.MODEL.MASK_FORMER.TEST_CITYSCAPES_PANOPTIC.OVERLAP_THRESHOLD,
        }
 

        return {
            "backbone": backbone,

            "text_encoder": text_encoder,
            "fixed_contexts": fixed_contexts,
            "learnable_contexts": learnable_contexts,
            "text_adapter": text_adapter,

            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": size_divisibility,

            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or ('panoptic' in cfg.LMSEG.TASK_TYPE)),
            "panoptic_test_configs": panoptic_test_configs,

            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "cfg": cfg.clone(),
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    def after_extract_feat(self, dataset_type_list):  
        # assert all dataset_type in the same gpu is same
        assert len(set(dataset_type_list)) == 1

        text_embeddings = []
        for t in dataset_type_list:
            if (t in self.text_embedding_buffer) and (not self.cfg.LMSEG.ENABLE_LEARNABLE_CONTEXT) and (self.cfg.SOLVER.TEXTENCODER_MULTIPLIER <= 0):
                text_embeddings_per_img = self.text_embedding_buffer[t].clone().detach()
            else:
                text_embeddings_per_img, class_names = self.fixed_contexts[t]
                text_embeddings_per_img = text_embeddings_per_img.to(self.device)
                text_embeddings_per_img = self.text_encoder(text_embeddings_per_img, self.learnable_contexts)           # (1, 150, 1024)
                if t not in self.text_embedding_buffer:
                    self.text_embedding_buffer[t] = text_embeddings_per_img.clone().detach()
            
            text_embeddings.append(text_embeddings_per_img)
        text_embeddings = torch.cat(text_embeddings, dim=0)                                                             # (B, 150, 1024)
        
        return text_embeddings
        
    def forward(self, batched_inputs):
        #torch.cuda.empty_cache()

        if self.training:
            dataset_type_list = [b['dataset_type'] for b in batched_inputs]
        else:
            ## semantic segmenatation 
            assert 'panoptic' in self.cfg.DATASETS.TEST[0]

            if 'ade20k' in self.cfg.DATASETS.TEST[0]:
                dataset_type_list = ['ade20k_panoptic'] * len(batched_inputs)
            elif 'cityscapes' in self.cfg.DATASETS.TEST[0]:
                dataset_type_list = ['cityscapes_panoptic'] * len(batched_inputs)
            elif 'coco' in self.cfg.DATASETS.TEST[0]:
                dataset_type_list = ['coco_panoptic'] * len(batched_inputs)
            else:
                raise ValueError()
        
        assert len(set(dataset_type_list)) == 1
        dataset_type = dataset_type_list[0]
        size_divisibility = self.size_divisibility[dataset_type]

        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, size_divisibility)          # images.tensor: (B, 3, 512, 512)

        features = self.backbone(images.tensor)                             # dict_keys(['res2', 'res3', 'res4', 'res5'])

        text_embeddings_pre = self.after_extract_feat(dataset_type_list)                            # (B, 151, 256)
        text_embeddings = self.text_adapter(text_embeddings_pre)
        
        outputs = self.sem_seg_head(features, text_embeddings)
        # dict_keys(['pred_masks', 'aux_outputs', 'class_embeddings', 'mask_embddings'])
        #       pred_masks: (B, n_query, 128, 128)
        #       aux_outputs: list of dict
        #       class_embeddings/mask_embddings: (6, B, n_query, 256)


        outputs_class = torch.einsum('lbqc,bkc->lbqk', F.normalize(self.cls_embed_mlp(outputs['class_embeddings']), p=2, dim=-1), 
                                                        F.normalize(text_embeddings, p=2, dim=-1))

        outputs_class /= self.tau
        outputs['pred_logits'] = outputs_class[-1]                          # (B, n_query, n_classes+1)
        for k in range(len(outputs['aux_outputs'])):
            outputs['aux_outputs'][k]['pred_logits'] = outputs_class[k]
        
        
        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size, dataset_type in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes, dataset_type_list
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])

                if self.sem_seg_postprocess_before_inference:
                    mask_pred_result = sem_seg_postprocess(
                        mask_pred_result, image_size, height, width
                    )

                # semantic segmentation inference
                r = self.semantic_inference(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = sem_seg_postprocess(r, image_size, height, width)
                processed_results.append({"sem_seg": r})

                # panoptic segmentation inference
                #if self.panoptic_on:
                if ('panoptic' in dataset_type) and self.panoptic_test_configs[dataset_type]['panoptic_on']:
                    panoptic_r = self.panoptic_inference(mask_cls_result, mask_pred_result, 
                                                self.panoptic_test_configs[dataset_type]['object_mask_threshold'],
                                                self.panoptic_test_configs[dataset_type]['overlap_threshold'],
                                                dataset_type)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                
                processed_results[-1]['mask_cls_result'] = mask_cls_result
                
            return processed_results

    
    def prepare_targets(self, targets, images):
        h, w = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks            
            padded_masks = torch.zeros((gt_masks.shape[0], h, w), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg
    
    def panoptic_inference(self, mask_cls, mask_pred, object_mask_threshold, overlap_threshold, dataset_type):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        num_classes = len(self.fixed_contexts[dataset_type][1]) - 1    # exclude background

        #if True:
        #    scores_no_bg = scores[labels.ne(num_classes)]
        #    labels_no_bg = labels[labels.ne(num_classes)]
        #    topk_inds = scores_no_bg.topk(k=10)[1]
        #    scores_no_bg_tok = scores_no_bg[topk_inds]
        #    labels_no_bg_tok = labels_no_bg[topk_inds]

        keep = labels.ne(num_classes) & (scores > object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]

        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                metadata = MetadataCatalog.get(self.cfg.DATASETS.TEST[0])
                isthing = pred_class in metadata.thing_dataset_id_to_contiguous_id.values()
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info
