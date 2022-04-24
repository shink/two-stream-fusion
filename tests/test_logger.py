# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: test_logger.py
@date: 2022/4/22
@description:
"""

import os
from conf.logger import log_file_path


class TestLogger:
    def test_logger(self):
        assert os.path.exists(log_file_path)
