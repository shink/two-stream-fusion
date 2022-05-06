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
from os import path

from PIL import Image
from torch.utils.data import Dataset

from conf import log
from utils import util, video_util, dataset_util, project_root_path


class Ucf101Dataset(Dataset):
    """
    The UCF-101 Dataset
    """

    url = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"

    def __init__(self,
                 dataset_path: str,
                 processed_path: str,
                 phase: str = 'train',
                 preprocess: bool = False,
                 download: bool = False,
                 frame_interval: int = 4,
                 frame_suffix: str = 'jpg',
                 transform=None):
        """
        """

        self.dataset_path = dataset_path
        self.processed_path = processed_path
        self.phase = phase
        self.frame_interval = frame_interval
        self.frame_suffix = frame_suffix
        self.transform = transform

        self.project_dataset_dir = path.join(project_root_path, 'datasets', 'ucf-101')

        label_file_path = path.join(self.project_dataset_dir, 'label.csv')
        label_list = util.read_list_from_csv(label_file_path)
        self.idx_label_dict = {item[0]: item[1] for item in label_list}
        self.label_idx_dict = {item[1]: item[0] for item in label_list}

        if download:
            self.download()

        elif not self.check_integrity():
            log.error("dataset (%s) not found or corrupted" % self.dataset_path)
            raise RuntimeError("Dataset not found or corrupted. You need to download it from official website.")

        if download or preprocess or not self.check_preprocess():
            log.info("preprocessing dataset, which will take a long time, but it will be done only once.")
            self.preprocess()

        self.dataset_list = dataset_util.resume_dataset(self.project_dataset_dir, phase)

        log.info('%s dataset size: %d' % (phase, self.__len__()))

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, index):
        """
        """

        label, processed_video_path = self.dataset_list[index]
        rgb_tensor, optical_tensor = video_util.extract_frame_to_tensor(processed_video_path)
        return rgb_tensor, optical_tensor, label

    def download(self):
        """
        download dataset
        """
        pass

    def check_integrity(self) -> bool:
        log.debug("checking dataset integrity, dataset path: %s" % self.dataset_path)
        return os.path.exists(self.dataset_path)

    def check_preprocess(self) -> bool:
        return os.path.exists(self.processed_path)

    def preprocess(self) -> None:
        start_time = timeit.default_timer()

        if not os.path.exists(self.processed_path):
            os.mkdir(self.processed_path)

        total_dataset_list = []
        for category in tqdm(os.listdir(self.dataset_path), desc='Preprocessing', leave=False, ncols=50):
            # category directory
            category_path = os.path.join(self.dataset_path, category)

            # all videos in the category directory
            video_path_list = [os.path.join(category_path, video_name) for video_name in os.listdir(category_path)]

            # extract videos
            processed_category_path = os.path.join(self.processed_path, category)
            video_util.extract_videos(processed_category_path, video_path_list)

            total_dataset_list.append(
                [
                    [self.label_idx_dict[category], video_util.get_processed_video_path(self.processed_path, video_path)]
                    for video_path in video_path_list
                ]
            )

        dataset_util.save_divided_dataset_list(self.project_dataset_dir, total_dataset_list)

        stop_time = timeit.default_timer()
        log.info('Preprocessing finished. Execution time: %ds' % (stop_time - start_time))
