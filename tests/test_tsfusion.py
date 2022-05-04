# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: test_tsfusion.py
@date: 2022/4/26
@description:
"""

from unittest import TestCase
from core.network.tsfusion import TwoStreamFusion


class TestTwoStreamFusion(TestCase):
    """
    """

    def test_network(self):
        nn = TwoStreamFusion(101)
        print(nn)
