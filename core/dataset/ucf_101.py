# coding=utf-8

"""
@author: shenke
@project: two-stream-fusion
@file: ucf_101.py
@date: 2022/3/10
@description: 
"""

import os
import timeit
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from conf import logger
from utils import util


class Ucf101Dataset(Dataset):
    """
    The UCF-101 Dataset
    """

    __class_nums = 101
    __resize_height = 224
    __resize_width = 224

    __frame_interval: int
    __frame_suffix: str

    __dataset_path: str
    __frame_path: str
    __datasets: list
    __labels: list

    def __init__(self, dataset_path, frame_path=None, phase='train', preprocess=False, frame_interval=4, frame_suffix='jpg'):
        """
        """

        self.__dataset_path = dataset_path
        self.__frame_path = frame_path
        self.__frame_interval = frame_interval
        self.__frame_suffix = frame_suffix

        if not self.check_integrity():
            logger.error('Dataset not found or corrupted. You need to download it from official website.')
            raise RuntimeError('Dataset not found or corrupted. You need to download it from official website.')

        if preprocess or not self.check_preprocess():
            logger.info("Preprocessing of the UCF-Crime dataset, which will take a long time, but it will be done only once.")
            self.preprocess()

        # self.__labels = self.generate_labels()

        # logger.info('Number of %s videos: %d.' % (phase, len(self.__datasets)))

    def __len__(self):
        return len(self.__datasets)

    def __getitem__(self, index):
        pass

    def get_class_num(self) -> int:
        return self.__class_nums

    def preprocess(self) -> None:
        start_time = timeit.default_timer()

        if not os.path.exists(self.__frame_path):
            os.mkdir(self.__frame_path)
            os.mkdir(os.path.join(self.__frame_path, 'train'))
            os.mkdir(os.path.join(self.__frame_path, 'tests'))
            os.mkdir(os.path.join(self.__frame_path, 'valid'))

        # Split datasets
        for category in tqdm(os.listdir(self.__dataset_path), desc='Processing', leave=True):
            # category folder
            category_path = os.path.join(self.__dataset_path, category)

            # all videos in the folder
            video_list = [v for v in os.listdir(category_path)]

            # split datasets
            train_and_valid, test_video_name_list = train_test_split(video_list, test_size=0.2, random_state=42)
            train_video_name_list, valid_video_name_list = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            frame_train_path = os.path.join(self.__frame_path, 'train', category)
            frame_test_path = os.path.join(self.__frame_path, 'tests', category)
            frame_valid_path = os.path.join(self.__frame_path, 'valid', category)

            self.extract_videos(category_path, train_video_name_list, frame_train_path)
            self.extract_videos(category_path, test_video_name_list, frame_test_path)
            self.extract_videos(category_path, valid_video_name_list, frame_valid_path)

        stop_time = timeit.default_timer()
        logger.info('Preprocessing finished. Execution time: %ds' % (stop_time - start_time))

    def extract_videos(self, video_path, video_name_list, frame_category_path):
        if not os.path.exists(frame_category_path):
            os.mkdir(frame_category_path)

        for video_name in video_name_list:
            self.extract_video(video_path, video_name, frame_category_path)

    def extract_video(self, video_path, video_name, frame_category_path):
        """
        """

        video_filename = util.filename(video_name)
        frame_path = os.path.join(frame_category_path, video_filename)

        if not os.path.exists(frame_path):
            os.mkdir(frame_path)

        cap = cv2.VideoCapture(os.path.join(video_path, video_name))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.debug('The %s \'s frame rate: %dfps, count: %d, width: %dpx, height: %dpx' % (video_name, frame_rate, frame_count, frame_width, frame_height))

        for count in range(frame_count):
            ret, frame = cap.read()
            assert ret
            if count % self.__frame_interval == 0:
                if (frame_height != self.__resize_height) or (frame_width != self.__resize_width):
                    frame = cv2.resize(frame, (self.__resize_width, self.__resize_height))
                cv2.imwrite(filename=os.path.join(frame_path, util.frame_name(count, self.__frame_suffix)), img=frame)

        # release the VideoCapture once it is no longer needed
        cap.release()

    def generate_labels(self) -> list:
        labels = []
        label_path = os.path.join(self.__dataset_path, 'label.csv')
        if not os.path.exists(label_path):
            with open(label_path, 'w') as f:
                # TODO: write labels
                pass
        return labels

    def check_integrity(self) -> bool:
        print(self.__dataset_path)
        print(os.path.exists(self.__dataset_path))
        return os.path.exists(self.__dataset_path)

    def check_preprocess(self) -> bool:
        pass
