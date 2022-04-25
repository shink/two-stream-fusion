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

from utils import util


class TestUtils(TestCase):

    def test_read_labels_from_csv(self):
        project_path = abspath(dirname(dirname(dirname(__file__))))
        csv_path = join(project_path, 'dataset', 'ucf-101', 'label.csv')
        labels = util.read_labels_from_csv(csv_path)
        assert len(labels) == 101
