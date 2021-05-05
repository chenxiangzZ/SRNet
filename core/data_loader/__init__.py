import sys

sys.path.append('../')

from .dataset import *
from .loader import *

import torchvision.transforms as transforms
from tools import *


class Loaders:

    def __init__(self, config):

        self.transform_train = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
        ])
        self.transform_test = transforms.Compose([
            transforms.Resize(config.image_size, interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 新增market选项
        self.datasets = ['duke', 'market', "occluded_reid", "partial_reid", "partial_ilids"]

        # dataset
        self.dataset_path = config.dataset_path
        # 默认是duck
        self.train_dataset = config.train_dataset
        assert self.train_dataset in self.datasets

        # batch size
        self.p = config.p
        self.k = config.k

        # dataset paths
        # 针对是train还是test来区别Loader的加载
        if self.train_dataset == "occluded_reid" or self.train_dataset == "partial_ilids":
            self.samples_path = {
                'query_dir': os.path.join(self.dataset_path, 'query/'),
                'gallery_dir': os.path.join(self.dataset_path, 'gallery/')}
        elif self.train_dataset == "partial_reid":
            self.samples_path = {
                'query_dir': os.path.join(self.dataset_path, 'partial_body_images/'),
                'gallery_dir': os.path.join(self.dataset_path, 'whole_body_images/')}
        else:
            self.samples_path = {
                'train_dir': os.path.join(self.dataset_path, 'bounding_box_train/'),
                'query_dir': os.path.join(self.dataset_path, 'query/'),
                'gallery_dir': os.path.join(self.dataset_path, 'bounding_box_test/')}
        # load
        if config.mode == "train":
            self._load_trian_dataset()
        elif config.mode == "test" or config.mode == "visualize":
            self._load_test_dataset()

    def _load_trian_dataset(self):
        # train dataset and iter
        self.train_samples = self._get_train_samples(self.train_dataset)
        self.train_iter = self._get_uniform_iter(self.train_samples, self.transform_train, self.p, self.k)

        # duke test dataset and loader
        self.query_samples, self.gallery_samples = self._get_test_samples()
        self.query_loader = self._get_loader(self.query_samples, self.transform_test, 128)
        self.gallery_loader = self._get_loader(self.gallery_samples, self.transform_test, 128)

        # 得到数据集的信息
        num_train_pids, num_train_cams = self._parse_data(self.train_samples.samples)
        num_query_pids, num_query_cams = self._parse_data(self.query_samples.samples)
        num_gallery_pids, num_gallery_cams = self._parse_data(self.gallery_samples.samples)

        # 打印数据集的信息
        print('=> Loaded {}'.format(self.train_dataset))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print('  train    | {:5d} | {:8d} | {:9d}'.format(num_train_pids, len(self.train_samples.samples),
                                                          num_train_cams))
        print('  query    | {:5d} | {:8d} | {:9d}'.format(num_query_pids, len(self.query_samples.samples),
                                                          num_query_cams))
        print('  gallery  | {:5d} | {:8d} | {:9d}'.format(num_gallery_pids, len(self.gallery_samples.samples),
                                                          num_gallery_cams))
        print('  ----------------------------------------')

    def _load_test_dataset(self):

        # test dataset and loader
        self.query_samples, self.gallery_samples = self._get_test_samples()
        self.query_loader = self._get_loader(self.query_samples, self.transform_test, 128)
        self.gallery_loader = self._get_loader(self.gallery_samples, self.transform_test, 128)

        # 得到数据集的信息
        num_query_pids, num_query_cams = self._parse_data(self.query_samples.samples)
        num_gallery_pids, num_gallery_cams = self._parse_data(self.gallery_samples.samples)

        # 打印数据集的信息
        print('=> Loaded {}'.format(self.train_dataset))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print('  query    | {:5d} | {:8d} | {:9d}'.format(num_query_pids, len(self.query_samples.samples),
                                                          num_query_cams))
        print('  gallery  | {:5d} | {:8d} | {:9d}'.format(num_gallery_pids, len(self.gallery_samples.samples),
                                                          num_gallery_cams))
        print('  ----------------------------------------')

    def _get_train_samples(self, train_dataset):
        train_samples_path = self.samples_path['train_dir']
        if train_dataset == 'duke':
            return Samples4Duke(train_samples_path)
        elif train_dataset == 'market':
            return Samples4Market(train_samples_path)

    def _get_test_samples(self):
        query_data_path = self.samples_path['query_dir']
        gallery_data_path = self.samples_path['gallery_dir']
        if self.train_dataset == 'duke':
            query_samples = Samples4Duke(query_data_path, reorder=False)
            gallery_samples = Samples4Duke(gallery_data_path, reorder=False)
            return query_samples, gallery_samples
        elif self.train_dataset == 'market':
            query_samples = Samples4Market(query_data_path, reorder=False)
            gallery_samples = Samples4Market(gallery_data_path, reorder=False)
            return query_samples, gallery_samples
        else:
            query_samples = Samples4OtherReid(query_data_path, reorder=False)
            gallery_samples = Samples4OtherReid(gallery_data_path, reorder=False)
            return query_samples, gallery_samples

    def _get_uniform_iter(self, samples, transform, p, k):
        '''
        load person reid data_loader from images_folder
        and uniformly sample according to class
        :param images_folder_path:
        :param transform:
        :param p:
        :param k:
        :return:
        '''
        dataset = PersonReIDDataSet(samples.samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=p * k, num_workers=8, drop_last=False,
                                 sampler=ClassUniformlySampler(dataset, class_position=1, k=k))
        iters = IterLoader(loader)

        return iters

    def _get_random_iter(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples.samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        iters = IterLoader(loader)
        return iters

    def _get_random_loader(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples.samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=True)
        return loader

    def _get_loader(self, samples, transform, batch_size):
        dataset = PersonReIDDataSet(samples.samples, transform=transform)
        loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=8, drop_last=False, shuffle=False)
        return loader

    def _parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.

        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        for _, pid, camid in data:
            pids.add(pid)
            cams.add(camid)
        return len(pids), len(cams)
