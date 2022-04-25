# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: test_ucf_101.py
@date: 2022/3/12
@description: 
"""

import os.path
from os.path import join, abspath, dirname
import time
from tqdm import tqdm
from unittest import TestCase

import utils.util
from core.dataset.ucf_101 import Ucf101Dataset


class TestUcf101Dataset(TestCase):
    """
    """

    project_path = abspath(dirname(dirname(dirname(dirname(__file__)))))

    def test_preprocess(self):
        dataset_path = join(self.project_path, 'dataset', 'ucf-101', 'UCF101')
        frame_path = join(self.project_path, 'dataset', 'ucf-101', 'frames')
        ucf101 = Ucf101Dataset(dataset_path, frame_path, preprocess=True)

    def test_frame_filename(self):
        count, suffix = 1, 'jpg'
        frame_filename = utils.util.frame_name(count, suffix)
        assert frame_filename == '000001.jpg'
