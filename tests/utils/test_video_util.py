# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: test_video_util.py
@date: 2022/3/13
@description: 
"""

from os.path import join, abspath, dirname
import cv2
from unittest import TestCase

from utils import video_util


class TestVideoUtil(TestCase):

    def test_extract_frame_from_video(self):
        # video_path = "C:\\Users\\shenke\\Desktop"
        # video_name = 'Fighting030_x264.mp4'
        # frame_path = "C:\\Users\\shenke\\Desktop"
        #
        # video_util.extract_frame_from_video(video_path, video_name, frame_path)

        path = "C:\\Users\\shenke\\Desktop\\2\\000019.jpg"
        img = cv2.imread(path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imshow("img_gray", img_gray)
        cv2.waitKey()
