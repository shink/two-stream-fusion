# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: param.py
@date: 2022/3/10
@description: 
"""

import os
from utils.util import read_yaml

param_file_dir = os.path.dirname(os.path.abspath(__file__))
param_file_name = "param.yml"
param_file_path = os.path.join(param_file_dir, param_file_name)

param = read_yaml(param_file_path)
dataset = param['dataset']
dataset_dir = param['dataset_dir']
dataset_preprocess_dir = param['dataset_preprocess_dir']
save_model_dir = param['save_model_dir']
model_name = param['model_name']

epoch_num = param['param']['epoch_num']
resume_epoch = param['param']['resume_epoch']
snapshot = param['param']['snapshot']
test_interval = param['param']['test_interval']
lr = param['param']['lr']
