import os
import numpy as np
import torch
from skimage import io, transform
from torch.utils.data import Dataset
from torchvision.utils import make_grid
import cv2


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

        img_dn = transform.resize(img_d, (new_h, new_w))
        img_ln = transform.resize(img_l, (new_h, new_w))
        img_rn = transform.resize(img_r, (new_h, new_w))

        return {'depth': img_dn, 'left': img_ln, 'right': img_rn}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        img_d, img_l, img_r = sample['depth'], sample['left'], sample['right']

        h, w = img_d.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        img_dn = img_d[top: top + new_h, left: left + new_w]
        img_ln = img_l[top: top + new_h, left: left + new_w]
        img_rn = img_r[top: top + new_h, left: left + new_w]

        return {'depth': img_dn, 'left': img_ln, 'right': img_rn}


class ToTensor(object):
    """Convert nd arrays in sample to Tensors."""

    def __call__(self, sample):
        img_d, img_l, img_r = sample['depth'], sample['left'], sample['right']

        img_dn = torch.from_numpy(img_d.astype(np.float32))  # depth image has no channel
        img_dn = normalize_depth(img_dn)
        img_ln = img_l.astype(np.float32).transpose((2, 0, 1))
        img_rn = img_r.astype(np.float32).transpose((2, 0, 1))
        return {'depth': img_dn.unsqueeze(0),
                'left': torch.from_numpy(img_ln),
                'right': torch.from_numpy(img_rn)}


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
    minv = torch.min(depth)
    rangev = torch.max(depth) - minv
    if rangev > 0:
        norm = (depth - minv) / rangev
    else:
        norm = torch.zeros(depth.size())
    return norm.clamp(0., 1.)


def custom_save_img(tensor, filename, n_row=8, padding=2):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image
    tensor = tensor.cpu()
    tensor = normalize_depth(tensor)
    grid = make_grid(tensor, nrow=n_row, padding=padding)
    nd_arr = grid.mul(255.).byte().transpose(0, 2).transpose(0, 1).numpy()
    # nd_arr = cv2.applyColorMap(nd_arr, cv2.COLORMAP_PARULA)
    im = Image.fromarray(nd_arr)
    im.save(filename)
