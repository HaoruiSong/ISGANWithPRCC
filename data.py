from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from utils.Sampler import Sampler
from opt import opt
import os
import re
import random
from PIL import Image
class Data():
    def __init__(self, test=None):
        rgb_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3)
        ])
        gray_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.Grayscale(3)
        ])
        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        process_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])
        woEr_process_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.trainset = Market1501(data_path=opt.data_path, dtype="train",
                                   rgb_trans=rgb_transform, gray_trans=gray_transform, process_trans=process_transform)
        self.trainset_woEr = Market1501(data_path=opt.data_path, dtype="train",
                                        rgb_trans=rgb_transform, gray_trans=gray_transform,
                                        process_trans=woEr_process_transform)
        if test is None:
            self.testset = Market1501(data_path=opt.data_path, dtype="test", test_trans=test_transform)
        else:
            self.testset = Market1501(data_path=opt.data_path + '/' + str(test),
                                      dtype="test", test_trans=test_transform)
        self.queryset = Market1501(data_path=opt.data_path, dtype="query", test_trans=test_transform)

        self.train_loader = dataloader.DataLoader(
            self.trainset, 
            sampler=RandomSampler(self.trainset, batch_id=opt.batchid,batch_image=opt.batchimage),
            batch_size=opt.batchid * opt.batchimage, num_workers=0, pin_memory=True)  # 8
        self.train_loader_woEr = dataloader.DataLoader(
            self.trainset_woEr,
            sampler=RandomSampler(self.trainset_woEr, batch_id=opt.batchid, batch_image=opt.batchimage),
            batch_size=opt.batchid * opt.batchimage, num_workers=0, pin_memory=True)  # 8

        self.test_loader = dataloader.DataLoader(
            self.testset, batch_size=opt.batchtest, num_workers=0, pin_memory=True)  # 8
        self.query_loader = dataloader.DataLoader(
            self.queryset, batch_size=opt.batchtest, num_workers=0, pin_memory=True)  # 8

class Market1501(dataset.Dataset):
    def __init__(self, data_path, dtype, rgb_trans = None, gray_trans = None, process_trans = None, test_trans = None):

        self.rgb_transform = rgb_trans
        self.gray_transform = gray_trans
        self.process_transform = process_trans
        self.test_transform = test_trans
        self.loader = default_loader
        self.data_path = data_path
        self.dtype = dtype

        if self.dtype == 'train':
            self.data_path += '\\bounding_box_train'
        elif self.dtype == 'test':
            self.data_path += '\\bounding_box_test'
        else:
            self.data_path += '\\query'

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}
        self._path2cloth = {path: self.cloth(path) for path in self.imgs}

    def __getitem__(self, index):
        path = self.imgs[index]
        labels = self._id2label[self.id(path)]
        cloth = self._path2cloth[path]
        img = self.loader(path)
        if self.dtype == "train":
            # if self.rgb_transform is not None:
            rgb = self.rgb_transform(img)
            if self.gray_transform is not None:
                gray = self.gray_transform(img)
            rgb = self.process_transform(rgb)
            return rgb, cloth, labels
        else:
            if self.test_transform is not None:
                rgb = self.test_transform(img)
            return rgb, labels


    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('\\')[-1].split('_')[0])

    def cloth(self, file_path):
        c = int(file_path.split("\\")[-1].split('_')[1][1])
        if c == 0 or c == 1:
            return self._id2label[self.id(file_path)] * 2
        elif c == 2:
            return self._id2label[self.id(file_path)] * 2 + 1

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('\\')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])

