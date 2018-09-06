from fat_dataset import load_dataset
from torchvision.utils import make_grid


def custom_save_img(tensor, filename, n_row=10, padding=2):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image
    tensor = tensor.cpu()
    tensor = load_dataset.Fat.normalize_depth(tensor)
    grid = make_grid(tensor, nrow=n_row, padding=padding)
    nd_arr = grid.mul(255.).byte().transpose(0, 2).transpose(0, 1).numpy()
    im = Image.fromarray(nd_arr)
    im.save(filename)
    return tensor
