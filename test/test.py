# coding=utf-8

"""
@author: shenke
@project: c3d
@file: test.py
@date: 2021/4/20
@description: 
"""

import os
import sys
import torch
from conf import logger

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != root_path:
    sys.path.insert(0, root_path)

if __name__ == '__main__':
    logger.info("Info")
    logger.debug(sys.path)

    x = torch.empty(5, 3)
    print(x)
