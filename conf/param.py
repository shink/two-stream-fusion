# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: param.py
@date: 2022/3/10
@description: 
"""

import os

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))

dataset = 'ucf-crime'  # the dataset name, it can be 'ucf-101', 'ucf-crime', 'xd-violence'
dataset_dir = os.path.join(root_dir, 'dataset')  # look for datasets from the 'dataset' directory by default
save_model_dir = os.path.join(root_dir, 'model')  # models stored in the 'model' directory by default

epoch_num = 2
resume_epoch = 0  # resume from an epoch
snapshot = 2  # store a model every snapshot epochs
test_interval = 1
lr = 1e-5  # learning rate
