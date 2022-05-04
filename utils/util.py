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


def get_root_path() -> os.path:
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def get_localtime() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def read_yaml(file_path: str, encoding='utf-8'):
    with open(file_path, encoding=encoding) as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


def read_list_from_csv(file_path: str) -> list:
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        data = [item for item in reader]
    return data


def write_list_to_csv(file_path: str, data: list) -> None:
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(data)
