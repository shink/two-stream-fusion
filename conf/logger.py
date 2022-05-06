# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: logger.py
@date: 2022/3/10
@description: 
"""

import os
import logging
from utils.util import read_yaml


class Logger(logging.Logger):
    def __init__(self,
                 name='root',
                 level=logging.DEBUG,
                 console_level=logging.DEBUG,
                 console_formatter=None,
                 file_level=logging.INFO,
                 file_path=None,
                 file_encoding=None,
                 file_formatter=None):
        super().__init__(name)
        self.setLevel(level)

        # if file exists, write log to it
        if file_path:
            file_handler = logging.FileHandler(file_path, encoding=file_encoding)
            file_handler.setLevel(file_level)
            file_handler.setFormatter(logging.Formatter(file_formatter))
            self.addHandler(file_handler)

        # print log to console
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(console_level)
        stream_handler.setFormatter(logging.Formatter(console_formatter))
        self.addHandler(stream_handler)


log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
log_file_name = 'two-stream-fusion.log'
log_file_path = os.path.join(log_dir, log_file_name)

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

config_dir = os.path.dirname(os.path.abspath(__file__))
config_file_name = "logger.yml"
config_file_path = os.path.join(config_dir, config_file_name)

config = read_yaml(config_file_path)
logger_config = config["logger"]
log = Logger(
    name=logger_config["name"],
    level=logger_config["level"],
    console_level=logger_config["console"]["level"],
    console_formatter=logger_config["console"]["formatter"],
    file_level=logger_config["file"]["level"],
    file_path=log_file_path,
    file_encoding=logger_config["file"]["encoding"],
    file_formatter=logger_config["file"]["formatter"]
)
