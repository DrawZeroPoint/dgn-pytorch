import os
import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import cv2

d1 = 0.0039215686


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = output_size
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img_d, img_l, img_r = sample['depth'], sample['left'], sample['right']

        h, w = img_d.shape[:2]
        new_h = np.random.randint(self.output_size, h)
        new_w = new_h

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img_d = img_d[top: top + new_h, left: left + new_w]
        img_l = img_l[top: top + new_h, left: left + new_w]
        img_r = img_r[top: top + new_h, left: left + new_w]
        return {'depth': img_d, 'left': img_l, 'right': img_r}


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img_d, img_l, img_r = sample['depth'], sample['left'], sample['right']

        h, w = img_d.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img_d = cv2.resize(img_d, (new_h, new_w))
        img_l = cv2.resize(img_l, (new_h, new_w))
        img_r = cv2.resize(img_r, (new_h, new_w))
        return {'depth': img_d, 'left': img_l, 'right': img_r}


class ToTensor(object):
    """Convert nd arrays in sample to Tensors."""

    def __call__(self, sample):
        img_d, img_l, img_r = sample['depth'], sample['left'], sample['right']
        img_d = torch.from_numpy(img_d.astype(np.float32))  # depth image has no channel
        img_d = normalize_depth(img_d)
        img_l = img_l.astype(np.float32).transpose((2, 0, 1))
        img_r = img_r.astype(np.float32).transpose((2, 0, 1))
        return {'depth': img_d.unsqueeze(0),
                'left': torch.from_numpy(img_l).mul(d1),
                'right': torch.from_numpy(img_r).mul(d1)}


class FATDataset(Dataset):
    """Falling things dataset."""

    def __init__(self, root_dir, phase, trans=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            phase (string): 'train' or 'test'
            trans (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.phase = phase
        self.transform = trans

        self.list_depth = []
        self.list_left = []
        self.list_right = []
        self.pair_num = self.gen_file_list()

    def gen_file_list(self):
        f_list = os.listdir(os.path.join(self.root_dir, self.phase))
        f_list.sort()
        for fn in f_list:
            if 'left.depth.png' in fn:
                self.list_depth.append(fn)
            elif 'left.jpg' in fn:
                self.list_left.append(fn)
            elif 'right.jpg' in fn:
                self.list_right.append(fn)
        assert (len(self.list_depth) == len(self.list_left) == len(self.list_right))
        return len(self.list_right)

    def __len__(self):
        return self.pair_num

    def __getitem__(self, idx):
        pre_fix = os.path.join(self.root_dir, self.phase)
        name_depth = os.path.join(pre_fix, self.list_depth[idx])
        name_left = os.path.join(pre_fix, self.list_left[idx])
        name_right = os.path.join(pre_fix, self.list_right[idx])
        image_depth = io.imread(name_depth)
        image_left = io.imread(name_left)
        image_right = io.imread(name_right)

        sample = {'depth': image_depth, 'left': image_left, 'right': image_right}

        if self.transform:
            sample = self.transform(sample)

        return sample


def normalize_depth(depth):
    """
    Normalize the uint16 depth into range [0, 1]
    :param depth: input depth image
    :return: normalized depth image
    """
    v_min = torch.min(depth)
    v_range = torch.max(depth) - v_min
    if v_range > 0:
        norm = (depth - v_min) / v_range
    else:
        norm = torch.zeros(depth.size())
    return norm.clamp(0.0, 1.0)


def custom_save_img(tensor, filename, n_row=8, padding=2):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=n_row, padding=padding)
    nd_arr = grid.mul(255).byte().transpose(0, 2).transpose(0, 1).numpy()
    # nd_arr = cv2.applyColorMap(nd_arr, cv2.COLORMAP_PARULA)
    im = Image.fromarray(nd_arr)
    im.save(filename)
