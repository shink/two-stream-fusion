# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: utils.py
@date: 2022/5/4
@description: 
"""

import os
from sklearn.model_selection import train_test_split

from utils import util


def save_divided_dataset_list(save_dir: str, dataset_list: list) -> None:
    train_dataset_list, test_dataset_list, valid_dataset_list = [], [], []

    for class_list in dataset_list:
        class_train_and_valid, class_test_dataset_list = train_test_split(class_list, test_size=0.1, random_state=42)
        class_train_dataset_list, class_valid_dataset_list = train_test_split(class_train_and_valid, test_size=0.1, random_state=42)
        train_dataset_list += class_train_dataset_list
        test_dataset_list += class_test_dataset_list
        valid_dataset_list += class_valid_dataset_list

    # write to file
    util.write_list_to_csv(os.path.join(save_dir, dataset_divide_filename('train')), sorted(train_dataset_list))
    util.write_list_to_csv(os.path.join(save_dir, dataset_divide_filename('test')), sorted(test_dataset_list))
    util.write_list_to_csv(os.path.join(save_dir, dataset_divide_filename('valid')), sorted(valid_dataset_list))


def resume_dataset(dataset_split_dir: str, phase: str) -> list:
    divide_filename = dataset_divide_filename(phase)
    return util.read_list_from_csv(file_path=os.path.join(dataset_split_dir, divide_filename))


def dataset_divide_filename(phase: str):
    if phase == 'train':
        return 'train_dataset.csv'

    elif phase == 'test':
        return 'test_dataset.csv'

    elif phase == 'valid':
        return 'valid_dataset.csv'

    else:
        return None
