import numpy as np
from PIL import Image
import copy
import os
import re

def os_walk(folder_dir):
    for root, dirs, files in os.walk(folder_dir):
        files = sorted(files, reverse=True)
        dirs = sorted(dirs, reverse=True)
        return root, dirs, files


class PersonReIDSamples:

    def __init__(self, samples_path, reorder=True):

        # parameters
        self.samples_path = samples_path
        self.reorder = reorder

        # load samples 所有有效图片的路径
        samples = self._load_images_path(self.samples_path)

        # reorder person identities and camera identities
        if self.reorder:
            samples = self._reorder_labels(samples, 1)
            samples = self._reorder_labels(samples, 2)
        self.samples = samples

    def _reorder_labels(self, samples, label_index):

        ids = []
        for sample in samples:
            ids.append(sample[label_index])

        # delete repetitive elments and order
        ids = list(set(ids))
        ids.sort()
        # reorder
        for sample in samples:
            sample[label_index] = ids.index(sample[label_index])

        return samples

    def _load_images_path(self, folder_dir):
        '''
        :param folder_dir:
        :return: [(path, identiti_id, camera_id)]
        '''
        samples = []
        root_path, _, files_name = os_walk(folder_dir)
        for file_name in files_name:
            if '.jpg' in file_name:
                identi_id, camera_id = self._analysis_file_name(file_name)
                # 去掉market里的junk数据
                if identi_id != -1:
                    samples.append([root_path + file_name, identi_id, camera_id])
        return samples

    def _analysis_file_name(self, file_name):
        '''

        :param file_name: format like 0844_c3s2_107328_01.jpg
        :return:
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').replace('s', '_').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

class Samples4OtherReid(PersonReIDSamples):
    '''
    其他三个reid dataset 都是这个文件格式
    Occluded_reid Dataset
    '''
    def _analysis_file_name(self, file_name):
        '''
        :param file_name: format like 001_01.jpg
        :return:
        '''
        split_list = file_name.replace('.jpg', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id

class Samples4Market(PersonReIDSamples):
    '''
    Market Dataset
    '''
    def _analysis_file_name(self, file_name):
        '''
        相机序号需要转化为数字后除10
        :param file_name: format like 0002_c1s1_000551_01.jpg
        :return:
        '''
        pattern = re.compile(r'([-\d]+)_c(\d)')
        identi_id, camera_id = map(int, pattern.search(file_name).groups())
        # identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id


class Samples4Duke(PersonReIDSamples):
    '''
    Duke dataset
    '''
    def _analysis_file_name(self, file_name):
        '''

        :param file_name: format like 0002_c1_f0044158.jpg
        :return:
        '''
        split_list = file_name.replace('.jpg', '').replace('c', '').split('_')
        identi_id, camera_id = int(split_list[0]), int(split_list[1])
        return identi_id, camera_id


class PersonReIDDataSet:

    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        this_sample = copy.deepcopy(self.samples[index])

        this_sample[0] = self._loader(this_sample[0])
        if self.transform is not None:
            this_sample[0] = self.transform(this_sample[0])
        this_sample[1] = np.array(this_sample[1])

        return this_sample

    def __len__(self):
        return len(self.samples)

    def _loader(self, img_path):
        return Image.open(img_path).convert('RGB')
