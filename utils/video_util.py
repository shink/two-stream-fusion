# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: video_util.py
@date: 2022/3/12
@description: 
"""

import os
import cv2

from utils import util
from conf import logger


def extract_videos(video_path, video_name_list, frame_category_path):
    if not os.path.exists(frame_category_path):
        os.mkdir(frame_category_path)

    for video_name in video_name_list:
        extract_frame_from_video(video_path, video_name, frame_category_path)


def extract_frame_from_video(video_path,
                             video_name,
                             frame_path,
                             frame_interval=1,
                             frame_suffix='jpg',
                             resize_height=None,
                             resize_width=None):
    """
    """

    video_filename = util.filename(video_name)
    frame_path = os.path.join(frame_path, video_filename)

    if not os.path.exists(frame_path):
        os.mkdir(frame_path)

    cap = cv2.VideoCapture(os.path.join(video_path, video_name))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.debug('The %s \'s frame rate: %dfps, count: %d, width: %dpx, height: %dpx' % (video_name, frame_rate, frame_count, frame_width, frame_height))
    need_resize = (resize_height is not None) and (resize_width is not None) and ((frame_height != resize_height) or (frame_width != resize_width))

    for count in range(frame_count):
        ret, frame = cap.read()
        assert ret
        if count % frame_interval == 0:
            if need_resize:
                frame = cv2.resize(frame, (resize_width, resize_height))
            cv2.imwrite(filename=os.path.join(frame_path, util.frame_name(count, frame_suffix)), img=frame)

    # release the VideoCapture once it is no longer needed
    cap.release()


def extract_optical_flow_from_frames():
    pass
