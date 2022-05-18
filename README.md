# Seed the views: Hierarchical semantic alignment for contrastive representation learning

Official implementation of the paper [Seed the views: Hierarchical semantic alignment for contrastive representation learning](https://arxiv.org/abs/2012.02733)


## Requirements
* Pytorch >= 1.4.0
* faiss-gpu
* absl-py
* easydict

## Pre-training
```
bash scripts/unsupervised/single_crop_200ep.sh # change --data_dir with the path in your server
```

## Linear Classification
```
bash scripts/lincls/single_crop_200ep.sh # change --data_dir with the path in your server
```
