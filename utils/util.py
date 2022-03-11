# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: utils.py
@date: 2022/3/10
@description: 
"""

import time


def get_localtime() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

