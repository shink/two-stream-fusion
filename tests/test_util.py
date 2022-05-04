# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: test_util.py
@date: 2022/3/12
@description: 
"""

from os.path import join, abspath, dirname
from unittest import TestCase

from utils import util, project_root_path


class TestUtil(TestCase):

    def test_read_labels_from_csv(self):
        csv_path = join(project_root_path, 'datasets', 'ucf-101', 'label.csv')
        labels = util.read_list_from_csv(csv_path)
        print(labels)
        assert len(labels) == 101
