import numpy as np
import numpy.ma as ma
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from dataset import FATDataset, RandomCrop, ToTensor
import matplotlib.pyplot as plt
import matplotlib.colors as cls


def show_batch(batched_samples):
    batch_depth = batched_samples['depth']

    grid = utils.make_grid(batch_depth).numpy().astype(np.int)
    norm = cls.Normalize(vmin=0, vmax=255)
    grid = ma.getdata(norm(grid))  # Convert the masked array to ndarray
    plt.imshow(grid.astype(np.int).transpose((1, 2, 0)))


fat_dataset = FATDataset("/home/dong/dgn-pytorch/dataset/fat",
                         "train", trans=transforms.Compose([RandomCrop(480), ToTensor()]))

dataloader = DataLoader(fat_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['depth'].size(),
          sample_batched['left'].size())

    # observe 4th batch and stop.
    if i_batch == 3:
        plt.figure()
        show_batch(sample_batched)
        plt.axis('off')
        plt.ioff()
        plt.show()
        break
