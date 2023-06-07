import os

from PIL import Image
from torch.utils.data import Dataset


class VidFrameLoader(Dataset):
    """
    Just load images from a video. Can define a subset of frames or a max
    number of frames.
    """
    def __init__(self, root,
                 transforms=None,
                 ext='.jpg'):
        """
        inputs:
           path     - path to images
           tgt_size - (w,h) the size of image
        """
        imgs = os.listdir(root)
        imgs.sort()
        self.imgs = [img for img in imgs if ext in img]
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_fn = self.imgs[index]
        img = Image.open(img_fn)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, img_fn
