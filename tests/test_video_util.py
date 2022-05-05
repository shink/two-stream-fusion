# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: test_video_util.py
@date: 2022/3/13
@description: 
"""

import os
from os.path import join, abspath, dirname, exists
import cv2
from unittest import TestCase

from utils import video_util, project_root_path


class TestVideoUtil(TestCase):
    """
    """

    def test_frame_filename(self):
        count, suffix = 1, 'jpg'
        frame_filename = video_util.frame_name(count, suffix)
        assert frame_filename == '000001.jpg'

    def test_extract_frame_from_video(self):
        video_path = join(project_root_path, 'datasets', 'ucf-101', 'UCF101', 'ApplyEyeMakeup')
        video_name = 'v_ApplyEyeMakeup_g01_c01.avi'
        processed_path = join(project_root_path, 'datasets', 'ucf-101', 'ut-frames')

        # if not exists(processed_path):
        #     os.makedirs(processed_path)
        #
        # video_util.extract_video(video_path, video_name, processed_path)

        # path = join(processed_path, 'v_ApplyEyeMakeup_g01_c01', 'rgb', '000000.jpg')
        # img = cv2.imread(path)
        # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("img_gray", img_gray)
        # cv2.waitKey()
