# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: logger.py
@date: 2022/3/10
@description: 
"""

import os
import logging.config

# 日志输出格式
standard_format = '[%(levelname)-5s] %(asctime)s [%(threadName)s] [%(name)s] %(filename)s:%(lineno)d - %(message)s'
simple_format = '[%(levelname)-5s] %(asctime)s [%(name)s] %(filename)s - %(message)s'

# log文件存放目录、文件名
logfile_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
logfile_name = 'two-stream-fusion.log'

if not os.path.isdir(logfile_dir):
    os.mkdir(logfile_dir)

logfile_path = os.path.join(logfile_dir, logfile_name)

# logging 配置
LOGGING_DIC = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': standard_format
        },
        'simple': {
            'format': simple_format
        },
    },
    'filters': {},
    'handlers': {
        # 打印到终端的日志
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',  # 打印到屏幕
            'formatter': 'simple'
        },
        # 打印到文件的日志，收集 info 及以上的日志
        'file': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',  # 保存到文件
            'formatter': 'standard',
            'filename': logfile_path,  # 日志文件
            'maxBytes': 5 * 1024 * 1024,  # 日志大小 5M
            'backupCount': 5,
            'encoding': 'utf-8',  # 日志文件的编码
        },
    },
    'loggers': {
        # logging.getLogger(__name__)拿到的logger配置
        '': {
            'handlers': ['console', 'file'],  # 这里把上面定义的两个 handler 都加上
            'level': 'DEBUG',
            'propagate': True,  # 向上（更高level的logger）传递
        },
    },
}

logging.config.dictConfig(LOGGING_DIC)
logger = logging.getLogger(__name__)  # 生成一个 logger 实例


def debug(message):
    logger.debug(message)


def info(message):
    logger.info(message)


def warn(message):
    logger.warn(message)


def error(message):
    logger.error(message)
