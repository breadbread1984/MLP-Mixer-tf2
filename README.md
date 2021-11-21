# MLP Mixer

this project implements SOTA features extractor algorithm introduced in paper [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601v4)

## generate dataset

download cifar10 with the following command

```shell
python3 create_datasets.py
```

## train

train on cifar10 with the following command

```shell
python3 train.py --model=(b16|b32|l16) --batch_size=<batch size>
```
