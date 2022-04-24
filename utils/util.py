# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: utils.py
@date: 2022/3/10
@description: 
"""

import time
import yaml


def get_localtime() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def read_yaml(file_path: str, encoding='utf-8'):
    with open(file_path, encoding=encoding) as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)
