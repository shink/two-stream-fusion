# # coding=utf-8
#
# """
# @author: shenke
# @project: two-stream-fusion
# @file: ucf_crime.py
# @date: 2022/3/10
# @description:
# """
#
# import os
# import timeit
# from PIL import Image
# import cv2
# import numpy as np
# import torch
# from torch.utils.data import Dataset
#
# from conf import log
#
#
# class UcfCrimeDataset(Dataset):
#     """
#     The UCF-Crime Dataset
#     """
#
#     __class_nums = 13
#     __dataset_dir: str
#     __frame_dir: str
#     __datasets: list
#     __labels: list
#
#     def __int__(self, dataset_dir, frame_dir=None, phase='train', preprocess=False):
#         """
#         """
#
#         if not self.check_integrity():
#             log.error('Dataset not found or corrupted. You need to download it from official website.')
#             raise RuntimeError('Dataset not found or corrupted. You need to download it from official website.')
#
#         if preprocess or not self.check_preprocess():
#             log.info("Preprocessing of the UCF-Crime datasets, which will take a long time, but it will be done only once.")
#
#         self.__labels = self.generate_labels()
#
#         log.info('Number of %s videos: %d.' % (phase, len(self.__datasets)))
#
#     def __len__(self):
#         return len(self.__datasets)
#
#     def __getitem__(self, index):
#         pass
#
#     def get_class_num(self) -> int:
#         return self.__class_nums
#
#     def preprocess(self) -> None:
#         start_time = timeit.default_timer()
#
#         if not os.path.exists(self.frame_dir):
#             os.mkdir(self.frame_dir)
#             os.mkdir(os.path.join(self.frame_dir, 'train'))
#             os.mkdir(os.path.join(self.frame_dir, 'tests'))
#             os.mkdir(os.path.join(self.frame_dir, 'valid'))
#
#         # Split datasets
#         for file in os.listdir(self.__dataset_dir):
#             file_path = os.path.join(self.__dataset_dir, file)
#
#             train_dir = os.path.join(self.__frame_dir, 'train', file)
#             test_dir = os.path.join(self.__frame_dir, 'tests', file)
#             valid_dir = os.path.join(self.__frame_dir, 'valid', file)
#
#             if not os.path.exists(train_dir):
#                 os.mkdir(train_dir)
#             if not os.path.exists(test_dir):
#                 os.mkdir(test_dir)
#             if not os.path.exists(valid_dir):
#                 os.mkdir(valid_dir)
#
#         stop_time = timeit.default_timer()
#         log.info('Preprocessing finished. Execution time: %d' % (stop_time - start_time))
#
#     def extract_videos(self, video_dir, video_name, file, save_dir):
#         """
#         """
#
#         video_filename = video_name.split('.')[0]
#         if not os.path.exists(os.path.join(save_dir, video_filename)):
#             os.mkdir(os.path.join(save_dir, video_filename))
#
#         video_path = os.path.join(video_dir, video_name)
#         cap = cv2.VideoCapture(video_path)
#         frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
#         frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#         frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#         log.debug('The %s \'s frame rate: %dfps, count: %d, width: %dpx, height: %dpx' % (video_path, frame_rate, frame_count, frame_width, frame_height))
#
#     def generate_labels(self) -> list:
#         labels = []
#         label_path = os.path.join(self.__dataset_dir, 'label.csv')
#         if not os.path.exists(label_path):
#             with open(label_path, 'w') as f:
#                 # TODO: write labels
#                 pass
#         return labels
#
#     def check_integrity(self) -> bool:
#         return os.path.exists(self.dataset_dir)
#
#     def check_preprocess(self) -> bool:
#         pass
