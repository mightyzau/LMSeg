# LMSeg: Language-guided Multi-dataset Segmentation

Qiang Zhou, Yuang Liu, Chaohui Yu, Jingliang Li, Zhibin Wang, Fan Wang

<a href='https://arxiv.org/abs/2302.13495'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>


## Prepare Datasets

```
ln -s /mnt/datasets/ade/ADEChallengeData2016 datasets/
ln -s /mnt/datasets/cityscapes datasets/
ln -s /mnt/datasets/mapillary_vistas_v1 datasets/mapillary_vistas
ln -s /mnt/datasets/coco_2017 datasets/coco

## additional
ln -s /mnt/datasets/ADE20K_full_847 datasets/
```


## Training
```
bash train_lmseg_net.sh
```

## Acknowledgement
If you're using LMSeg in your research, please cite using this BibTeX:
```
@inproceedings{LMSeg23Zhou,
  author       = {Qiang Zhou and Yuang Liu and Chaohui Yu and Jingliang Li and Zhibin Wang and Fan Wang},
  title        = {LMSeg: Language-guided Multi-dataset Segmentation},
  booktitle    = {The Eleventh International Conference on Learning Representations, {ICLR} 2023, Kigali, Rwanda, May 1-5, 2023},
  year         = {2023}
}
```


## Thanks
Many codes are based on [MaskFormer](https://github.com/facebookresearch/MaskFormer).