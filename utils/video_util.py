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
import random
import numpy as np

from conf import log


def extract_videos(processed_category_path, video_path_list: list) -> None:
    """
    """

    if not os.path.exists(processed_category_path):
        os.mkdir(processed_category_path)

    for video_path in video_path_list:
        extract_video(processed_category_path, video_path)


def extract_video(processed_root_path: str,
                  video_path: str,
                  save_interval=1,
                  frame_suffix='jpg') -> None:
    """
    """

    processed_video_path = get_processed_video_path(processed_root_path, video_path)
    processed_rgb_path = os.path.join(processed_video_path, 'rgb')
    processed_optical_path = os.path.join(processed_video_path, 'optical')

    if not os.path.exists(processed_video_path):
        os.mkdir(processed_video_path)

    if not os.path.exists(processed_rgb_path):
        os.mkdir(processed_rgb_path)

    if not os.path.exists(processed_optical_path):
        os.mkdir(processed_optical_path)

    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    log.debug('[%s] frame rate: %dfps, count: %d, width: %dpx, height: %dpx'
              % (os.path.split(video_path)[1], frame_rate, frame_count, frame_width, frame_height))

    global prev, hsv
    for count in range(frame_count):
        ret, frame = cap.read()
        assert ret

        # save RGB
        if count % save_interval == 0:
            cv2.imwrite(filename=os.path.join(processed_rgb_path, frame_name(count, frame_suffix)), img=frame)

        if count == 0:
            prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
        else:
            next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            prev = next

            # save optical flow
            if count % save_interval == 0:
                cv2.imwrite(filename=os.path.join(processed_optical_path, frame_name(count, frame_suffix)), img=rgb)

    # release the VideoCapture once it is no longer needed
    cap.release()


def extract_frame(frame_dir: str,
                  frame_suffix: str = 'jpg',
                  extract_count: int = 2,
                  extract_interval: int = 5) -> list:
    """
    """

    rgb_path_list = []
    cnt = len(os.listdir(frame_dir))
    for i in range(0, cnt, extract_interval):
        frame_cnt_list = random.sample(range(i, i + extract_interval), extract_count)
        rgb_path_list += [os.path.join(frame_dir, frame_name(frame_cnt, frame_suffix)) for frame_cnt in frame_cnt_list]
    return rgb_path_list


def collect_frame_to_tensor(frame_path_list: list):
    frame_cnt = len(frame_path_list)
    buffer = np.empty((frame_cnt, self.resize_height, self.resize_width, 3), np.dtype('float32'))
    for i, frame_name in enumerate(frames):
        frame = np.array(cv2.imread(frame_name)).astype(np.float64)
        buffer[i] = frame


def filename(file_path: str) -> str:
    file_name = os.path.split(file_path)[1]
    return os.path.splitext(file_name)[0]


def frame_name(count: int, suffix: str) -> str:
    return '%.6d.%s' % (count, suffix)


def get_processed_video_path(processed_root_path: str, video_path: str) -> str:
    return os.path.join(processed_root_path, filename(video_path))
