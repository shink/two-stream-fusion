# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: utils.py
@date: 2022/3/10
@description: 
"""

import os
import time
import yaml
import csv


def get_localtime() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def read_yaml(file_path: str, encoding='utf-8'):
    with open(file_path, encoding=encoding) as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def read_labels_from_csv(csv_path) -> list:
    labels = []
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for item in reader:
            if len(item) == 2:
                labels.append([item[0], item[1]])
    return labels


def filename(file) -> str:
    return os.path.splitext(file)[0]


def frame_name(count: int, suffix: str) -> str:
    return '%.6d.%s' % (count, suffix)
