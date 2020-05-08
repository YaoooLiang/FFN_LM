from torch.utils import data
from typing import Sequence, List
import h5py
from .utils import *
import random
import imageio
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa


def geometric_transform(data1, data2):
        t1 = random.choice([np.fliplr, np.flipud])
        t2 = random.choice([np.fliplr, np.flipud])
        data1 = t2(t1(data1))
        data2 = t2(t1(data2))

        #t2 = random.choice([np.fliplr, np.flipud])
        #data1 = t2(data1)
        #data2 = t2(data2)


        return data1, data2


def hue_transform1(data1):
    z = data1.shape[0]
    degree = random.randrange(8, 12, 1)
    hue_m = degree / 10
    aug = iaa.MultiplyHueAndSaturation((hue_m, hue_m))

    for stack in range(z):
        data1[stack] = aug.augment_image(data1[stack])

    return data1


def hue_transform2(data1):
    z = data1.shape[0]
    para = random.randrange(-15, 15, 1)

    aug = iaa.AddToHueAndSaturation((para, para))

    for stack in range(z):
        data1[stack] = aug.augment_image(data1[stack])

    return data1


class BatchCreator(data.Dataset):
    def __init__(
            self,
            input_h5data: List[str],
            input_size: Sequence[int],
            delta: Sequence[int],
            threshold: float = 0.9,
            train: bool = True):
        super(BatchCreator, self).__init__()

        self.shifts = []
        self.deltas = delta
        for dx in (-self.deltas[0], 0, self.deltas[0]):
            for dy in (-self.deltas[1], 0, self.deltas[1]):
                for dz in (-self.deltas[2], 0, self.deltas[2]):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.shifts.append((dx, dy, dz))
        random.shuffle(self.shifts)

        self.input_data = []
        self.label_data = []
        self.seed = []
        self.coor = []
        self.parse_h5py(input_h5data)

        # Early checks
        if len(self.input_data) != len(self.label_data):
            raise ValueError("input_h5data and target_h5data must be lists of same length!")

        self.input_size = np.array(input_size)
        self.seed_shape = self.input_size + np.array(delta) * 2
        self.label_radii = self.seed_shape // 2

        self.data_idx = 0
        self.coor_idx = 0

        self.image_patch = None
        self.label_patch = None
        self.seed_patch = None
        self.offset = None

        self.threshold = threshold
        self.num_classes = 2
        self.reset = True


    def parse_h5py(self, input):
        """解析数据"""
        for input_data in input:
            with h5py.File(input_data, 'r') as raw:
                self.input_data.append(raw['image'][()])#.astype(np.float32)-128) / 33.0)
                self.label_data.append(raw['label'][()])
                self.coor.append(raw['coor'][()])#.astype(np.uint8))
                self.seed.append(logit(np.full(list(raw['label'][()].shape), 0.05, dtype=np.float32)))

    def __getitem__(self, idx):
        """根据记载的label坐标从250x250x250的立方中根据索引截取49x49x49的patch"""
        self.coor_patch = self.coor[self.data_idx][idx]

        self.image_patch = center_crop_and_pad(self.input_data[self.data_idx], self.coor_patch, self.seed_shape)

        self.label_patch = center_crop_and_pad(self.label_data[self.data_idx], self.coor_patch, self.seed_shape)

        self.image_patch ,self.label_patch = geometric_transform(self.image_patch, self.label_patch)



        self.image_patch =((self.image_patch.astype(np.float32) - 128) / 33.0)
        self.image_patch = self.image_patch.transpose(3, 0, 1, 2)

        """保证中心为当前label"""
        self.label_patch = np.logical_and(self.label_patch > 0, np.equal(self.label_patch, self.label_patch[tuple(self.label_radii)]))
        self.label_patch = np.where(self.label_patch, np.ones(self.label_patch.shape)*0.95, np.ones(self.label_patch.shape)*0.05)

        self.seed_patch = center_crop_and_pad(self.seed[self.data_idx], self.coor_patch, self.seed_shape)
        self.seed_patch[tuple(self.label_radii)] = logit(0.95)

        #skimage.io.imsave('./testinput/seed_patch_{}.tif'.format(idx), self.seed_patch)



        return self.image_patch, self.label_patch, self.seed_patch, self.coor_patch

    def __len__(self):
        return len(self.coor[self.data_idx])
2, 1)
    hue_m = degree / 10
    aug = iaa.MultiplyHueAndSaturation((hue_m, hue_m))

    for stack in range(z):
        data1[stack] = aug.augment_image(data1[stack])

    return data1


def hue_transform2(data1):
    z = data1.shape[0]
    para = random.randrange(-15, 15, 1)

    aug = iaa.AddToHueAndSaturation((para, para))

    for stack in range(z):
        data1[stack] = aug.augment_image(data1[stack])

    return data1


class BatchCreator(data.Dataset):
    def __init__(
            self,
            input_h5data: List[str],
            input_size: Sequence[int],
            delta: Sequence[int],
            threshold: float = 0.9,
            train: bool = True):
        super(BatchCreator, self).__init__()

        self.shifts = []
        self.deltas = delta
        for dx in (-self.deltas[0], 0, self.deltas[0]):
            for dy in (-self.deltas[1], 0, self.deltas[1]):
                for dz in (-self.deltas[2], 0, self.deltas[2]):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    self.shifts.append((dx, dy, dz))
        random.shuffle(self.shifts)

        self.input_data = []
        self.label_data = []
        self.seed = []
        self.coor = []
        self.parse_h5py(input_h5data)

        # Early checks
        if len(self.input_data) != len(self.label_data):
            raise ValueError("input_h5data and target_h5data must be lists of same length!")

        self.input_size = np.array(input_size)
        self.seed_shape = self.input_size + np.array(delta) * 2
        self.label_radii = self.seed_shape // 2

        self.data_idx = 0
        self.coor_idx = 0

        self.image_patch = None
        self.label_patch = None
        self.seed_patch = None
        self.offset = None

        self.threshold = threshold
        self.num_classes = 2
        self.reset = True


    def parse_h5py(self, input):
        """解析数据"""
        for input_data in input:
            with h5py.File(input_data, 'r') as raw:
                self.input_data.append(raw['image'][()])#.astype(np.float32)-128) / 33.0)
                self.label_data.append(raw['label'][()])
                self.coor.append(raw['coor'][()])#.astype(np.uint8))
                self.seed.append(logit(np.full(list(raw['label'][()].shape), 0.05, dtype=np.float32)))

    def __getitem__(self, idx):
        """根据记载的label坐标从250x250x250的立方中根据索引截取49x49x49的patch"""
        self.coor_patch = self.coor[self.data_idx][idx]

        self.image_patch = center_crop_and_pad(self.input_data[self.data_idx], self.coor_patch, self.seed_shape)
        self.label_patch = center_crop_and_pad(self.label_data[self.data_idx], self.coor_patch, self.seed_shape)


        self.image_patch ,self.label_patch = geometric_transform(self.image_patch, self.label_patch)


        #self.image_patch = hue_transform1(self.image_patch)#.astype('uint8'))
        #self.image_patch = hue_transform2(self.image_patch)#.astype('uint8'))



        #skimage.io.imsave('./testinput/image_patch_{}.tif'.format(idx),  self.image_patch)
        #skimage.io.imsave('./testinput/label_patch_{}.tif'.format(idx), self.label_patch)

        self.image_patch =((self.image_patch.astype(np.float32) - 128) / 33.0)

        #skimage.io.imsave('./testinput/image_patch_transed_{}.tif'.format(idx), self.image_patch)
        self.image_patch = self.image_patch.transpose(3, 0, 1, 2)

        """保证中心为当前label"""
        self.label_patch = np.logical_and(self.label_patch > 0, np.equal(self.label_patch, self.label_patch[tuple(self.label_radii)]))
        self.label_patch = np.where(self.label_patch, np.ones(self.label_patch.shape)*0.95, np.ones(self.label_patch.shape)*0.05)

        self.seed_patch = center_crop_and_pad(self.seed[self.data_idx], self.coor_patch, self.seed_shape)
        self.seed_patch[tuple(self.label_radii)] = logit(0.95)

        #skimage.io.imsave('./testinput/seed_patch_{}.tif'.format(idx), self.seed_patch)



        return self.image_patch, self.label_patch, self.seed_patch, self.coor_patch

    def __len__(self):
        return len(self.coor[self.data_idx])
