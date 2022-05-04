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

from utils import video_util, project_root_path
from core.dataset.ucf_101 import Ucf101Dataset


class TestUcf101Dataset(TestCase):
    """
    """

    def test_preprocess(self):
        dataset_path = join(project_root_path, 'datasets', 'ucf-101', 'UCF101')
        frame_path = join(project_root_path, 'datasets', 'ucf-101', 'frames')
        # ucf101 = Ucf101Dataset(dataset_path, frame_path, preprocess=True)
