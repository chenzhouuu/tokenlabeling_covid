""" Transforms Factory
Factory methods for building image and mask transforms
Adapted from Ross Wightman
"""
import math

import torch
from torchvision import transforms

from .mask_augment import auto_augment_transform, rand_augment_transform

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
# from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import str_to_interp_mode, str_to_pil_interp, RandomResizedCropAndInterpolation, ToNumpy, ToTensor
from timm.data.random_erasing import RandomErasing

import torchvision
from torchvision.transforms import functional as torchvision_F
import warnings
import random



_RANDOM_INTERPOLATION = (str_to_interp_mode('bilinear'), str_to_interp_mode('bicubic'))

class ComposeWithMask(torchvision.transforms.Compose):
    def __init__(self, **kwargs):
        super(ComposeWithMask, self).__init__(**kwargs)

    def __call__(self, img, mask):
        for t in self.transforms:
            if type(t).__name__ == 'RandomHorizontalFlipWithMask':
                img, mask = t(img, mask)
            elif type(t).__name__ == 'RandomVerticalFlipWithMask':
                img, mask = t(img, mask)
            elif type(t).__name__ == 'RandAugment':
                img, mask = t(img, mask)
            elif type(t).__name__ == 'AutoAugment':
                img, mask = t(img, mask)
            elif type(t).__name__ == 'RandomResizedCropAndInterpolationWithMask':
                # should ensure RandomResizedCropWithCoords after all trabsformation
                img, mask = t(img, mask)
            # elif type(t).__name__ == 'ToTensor':
            #     img, mask = t(img, mask)
            # elif type(t).__name__ == 'Normalize':
            #     img, mask = t(img, mask)
            else:
                img = t(img)
        return img, mask

class RandomResizedCropAndInterpolationWithMask(RandomResizedCropAndInterpolation):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 interpolation='bilinear'):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("range should be of kind (min, max)")

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = str_to_interp_mode(interpolation)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img, mask):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        if isinstance(self.interpolation, (tuple, list)):
            interpolation = random.choice(self.interpolation)
        else:
            interpolation = self.interpolation
        resized_img = torchvision_F.resized_crop(img, i, j, h, w, self.size, interpolation)
        resized_mask = torchvision_F.resized_crop(mask, i, j, h, w, self.size, interpolation)
        return resized_img, resized_mask

class RandomHorizontalFlipWithMask(torchvision.transforms.RandomHorizontalFlip):
    def __init__(self, **kwargs):
        super(RandomHorizontalFlipWithMask, self).__init__(**kwargs)

    def __call__(self, img, mask):
        if torch.rand(1) < self.p:
            return torchvision_F.hflip(img), torchvision_F.hflip(mask)
        return img, mask


class RandomVerticalFlipWithMask(torchvision.transforms.RandomVerticalFlip):
    def __init__(self, **kwargs):
        super(RandomVerticalFlipWithMask, self).__init__(**kwargs)

    def __call__(self, img, mask):
        if torch.rand(1) < self.p:
            return torchvision_F.vflip(img), torchvision_F.vflip(mask)
        return img, mask


def transforms_noaug_train_mask(
        img_size=224,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        toimg=False
):
    if interpolation == 'random':
        # random interpolation not supported with no-aug
        interpolation = 'bilinear'
    tfl = [
        RandomResizedCropAndInterpolationWithMask(img_size, scale=(1.0, 1.0), ratio=(1.0, 1.0), interpolation=interpolation)
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
    if toimg:
        tfl = [
        RandomResizedCropAndInterpolationWithMask(img_size, scale=(1.0, 1.0), ratio=(1.0, 1.0), interpolation=interpolation)
    ]
    return ComposeWithMask(transforms = tfl)


def transforms_train_mask(
        img_size=224,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='random',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        separate=False,
        toimg=False
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3./4., 4./3.))  # default imagenet ratio range
    primary_tfl = []
    if hflip > 0.:
        primary_tfl += [RandomHorizontalFlipWithMask(p=hflip)]
    if vflip > 0.:
        primary_tfl += [RandomVerticalFlipWithMask(p=vflip)]

    secondary_tfl = []
    if auto_augment:
        assert isinstance(auto_augment, str)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )

        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = str_to_pil_interp(interpolation)
        
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    elif color_jitter is not None:
        # color jitter is enabled when not using AA
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter),) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = [RandomResizedCropAndInterpolationWithMask(img_size, scale=scale, ratio=ratio, interpolation=interpolation)]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob, mode=re_mode, max_count=re_count, num_splits=re_num_splits, device='cpu'))
    if toimg:
        final_tfl = []
    if separate:
        return ComposeWithMask(primary_tfl), ComposeWithMask(secondary_tfl), ComposeWithMask(final_tfl)
    else:
        return ComposeWithMask(transforms = primary_tfl + secondary_tfl + final_tfl)

def transforms_eval(
        img_size=224,
        crop_pct=None,
        interpolation='bilinear',
        use_prefetcher=False,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD):
    crop_pct = crop_pct or DEFAULT_CROP_PCT

    if isinstance(img_size, (tuple, list)):
        assert len(img_size) == 2
        if img_size[-1] == img_size[-2]:
            # fall-back to older behaviour so Resize scales to shortest edge if target is square
            scale_size = int(math.floor(img_size[0] / crop_pct))
        else:
            scale_size = tuple([int(x / crop_pct) for x in img_size])
    else:
        scale_size = int(math.floor(img_size / crop_pct))

    tfl = [
        transforms.Resize(scale_size, interpolation=str_to_interp_mode(interpolation)),
        transforms.CenterCrop(img_size),
    ]
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        tfl += [ToNumpy()]
    else:
        tfl += [
            transforms.ToTensor(),
            transforms.Normalize(
                     mean=torch.tensor(mean),
                     std=torch.tensor(std))
        ]

    return transforms.Compose(tfl)


def create_mask_transform(
        input_size,
        is_training=False,
        use_prefetcher=False,
        no_aug=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
        crop_pct=None,
        separate=False, 
        toimg=False):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if is_training and no_aug:
        assert not separate, "Cannot perform split augmentation with no_aug"
        transform = transforms_noaug_train_mask(
            img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            toimg=toimg)
    elif is_training:
        transform = transforms_train_mask(
            img_size,
            scale=scale,
            ratio=ratio,
            hflip=hflip,
            vflip=vflip,
            color_jitter=color_jitter,
            auto_augment=auto_augment,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits,
            separate=separate,
            toimg=toimg)
    else:
        assert not separate, "Separate transforms not supported for validation preprocessing"
        transform = transforms_eval(
            img_size,
            interpolation=interpolation,
            use_prefetcher=use_prefetcher,
            mean=mean,
            std=std,
            crop_pct=crop_pct)

    return transform