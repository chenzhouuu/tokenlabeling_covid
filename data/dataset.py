""" Covid dataset with lung and infection masks
"""
from lib2to3.pytree import convert
from time import sleep
import torch.utils.data as data
import os
import torch
import logging

import re

from PIL import Image

_logger = logging.getLogger('xray dataset')

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg']

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]

def load_images_and_masks(folder, types=IMG_EXTENSIONS, class_to_idx=None, sort=True):
    labels = []
    image_names = []

    
    for root, subdirs, files in os.walk(folder, topdown=False):
        rel_path = os.path.relpath(root, folder) if (root != folder) else ''
        rel_path_split = rel_path.split(os.path.sep)
        if len(rel_path_split) == 2:
            label = rel_path_split[-2]
            for f in files:
                base, ext = os.path.splitext(f)
                if ext.lower() in types:
                    if rel_path_split[-1] == 'images':
                        image_names.append(os.path  .join(root, f))
                        labels.append(label)
                    
    if class_to_idx is None:
        # building class index
        unique_labels = set(labels)
        sorted_labels = list(sorted(unique_labels, key=natural_key))
        class_to_idx = {c: idx for idx, c in enumerate(sorted_labels)}
    images_and_targets = [(f, class_to_idx[l]) for f, l in zip(image_names, labels) if l in class_to_idx]
    if sort:
        images_and_targets = sorted(images_and_targets, key=lambda k: natural_key(k[0]))
    return images_and_targets, class_to_idx
    

class CovidQu(data.Dataset):

    def __init__(
        self,
        root,
        load_bytes=False,
        transform=None,
        greyscale=False
    ):
        class_to_idx = None
        images, class_to_idx = load_images_and_masks(root, class_to_idx=None)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transform
        self.greyscale = 'L' if greyscale else 'RGB'

    def __getitem__(self, index):
        path, target = self.samples[index]
        lung_mask_path = path.replace('images', 'lung masks')
        infection_mask_path = path.replace('images', 'infection masks')
        
        img = open(path, 'rb').read() if self.load_bytes else Image.open(path).convert(self.greyscale)
        lung_mask = open(lung_mask_path, 'rb').read() if self.load_bytes else Image.open(lung_mask_path).convert('L')
        infection_mask = open(infection_mask_path, 'rb').read() if self.load_bytes else Image.open(infection_mask_path).convert('L')

        if self.transform is not None:
            img, lung_mask, infection_mask = self.transform(img, lung_mask, infection_mask)

        return img, target, lung_mask, infection_mask

    def __len__(self):
        return len(self.samples)

    def filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

    def filenames(self, basename=False, absolute=False):
        fn = lambda x: x
        if basename:
            fn = os.path.basename
        elif not absolute:
            fn = lambda x: os.path.relpath(x, self.root)
        return [fn(x[0]) for x in self.samples]

def create_dataset(root, dataset_type='train', greyscale=False):
    dir = os.path.join(root, dataset_type)
    if not os.path.exists(dir):
        _logger.error('{} directory does not exist at: {}'.format(dataset_type, dir))
        exit(1)
    return CovidQu(root=dir, greyscale=greyscale)
    