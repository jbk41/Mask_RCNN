from cell import train
from imgaug import augmenters as iaa

import os
import sys
import random
import numpy as np
import cv2
import skimage.io
import warnings; warnings.simplefilter('ignore')
import time


####################################################################
# Augmentations 
####################################################################
# TGAR, add your own augmentations using the iaa package
# Sharpen and Emboss
SEaug = iaa.Sequential( [
    iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
    iaa.Emboss(alpha=(0.0, 1.0), strength=(0.5, 1.5))
], random_order=True)

# Gaussian Noise
GNaug = iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))

# Color
Caug = iaa.Sequential([
    iaa.WithChannels(0, iaa.Add((10, 100))),
    iaa.WithChannels(1, iaa.Add((10, 100))),
    iaa.WithChannels(2, iaa.Add((10, 100)))
])

# Brightness and Contrast
BCaug = iaa.Sequential([
    iaa.ContrastNormalization((0.5, 1.5)),
    iaa.Multiply((0.5, 1.5))
], random_order=True)

# Flips
Faug = iaa.Sequential( [
    iaa.Fliplr(.5),
    iaa.Flipud(.5),
])

#blur and brightness - intuition is that some cells are super bright and blurred so this will help identify cells like that
BBaug = iaa.Sequential([
    iaa.GaussianBlur(sigma=(0.0, 3.0)),
    iaa.AverageBlur(k=(2, 5)),
    iaa.MedianBlur(k=(3, 7)),
    iaa.Multiply((.5, 3))
], random_order=True)

# transformations - helps with making overlapping cells
Taug = iaa.Sequential( [
    iaa.ElasticTransformation(alpha=(0, 5.0), sigma=0.25),
    iaa.Superpixels(p_replace=0.5, n_segments=64),
    iaa.PiecewiseAffine(scale=(0.01, 0.05))
])

augmentation = iaa.Sometimes(.5, [
        SEaug,
        GNaug,
        Caug,
        BBaug,
        BCaug,
        Faug,
	Taug
    ])


# TRAIN THE MODEL

if __name__ == '__main__':
    import argparse
    import os
    import sys

    # Edit the augmentation variable to add or exclude certain augmentations to include during training

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mask R-CNN to detect cells.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train model, edit this file to modify augmentations used in training'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/cell/dataset/",
                        help='Directory of the Cell dataset')
    parser.add_argument('--init', required=True,
                        metavar="Weights to initialize training",
                        help="coco, imagenet, last, or /path/to/weights")
    parser.add_argument('--logs', required=True,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory')
    args = parser.parse_args()

    if args.init not in ['coco', 'last', 'imagenet']:
        if not os.path.exists(args.init):
            sys.exit('{} is not a valid initialization weights path'.format(args.init))
        
    if args.command == 'train':
        train(args.dataset, augmentation=augmentation, init_with=args.init, model_dir=args.logs)          
    else:
        sys.exit('{} is not a valid command'.format(args.command)) 
