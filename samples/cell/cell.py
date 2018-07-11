import tensorflow as tf
import os
import sys
import math
import random
import numpy as np
import cv2
import skimage.io
from PIL import Image
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.visualize import display_images


####################################################################
# CONFIGURATION
####################################################################

class CellsConfig(Config):
    NAME = "cells"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    
    NUM_CLASSES = 1+1 # background + cell
    
    # size of images are 256px X 256px
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 1024 
    
     # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    #TRAIN_ROIS_PER_IMAGE = 512
    TRAIN_ROIS_PER_IMAGE = 200
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50
    
    LEARNING_RATE = 1e-4 


####################################################################
# DATASET 
####################################################################
class CellsDataset(utils.Dataset):
    """Generates a cells dataset for training. Dataset consists of microscope images.
"""
        
    def load_cells(self, dataset_dir, subset):
        """Loads cell images from the dataset directory"""
        
        # Add class
        self.add_class("cells", 1, "cellobj")
        
        # Add every image in IMG_DIR
        #cells_dir = os.path.join(ROOT_DIR, os.path.join(dataset_dir))
        #count = 0
        #for image_dir in os.listdir(image_dir):
        #   if filename.endswith(".tif"):
        #        self.add_image("cells", count, os.path.join(image_dir, filename))
        #        count+=1 
        
        assert subset in ["train", "test"]
        if subset=="train":
            dataset_dir = os.path.join(dataset_dir, "training_data")
        elif subset=="test":
            dataset_dir = os.path.join(dataset_dir, "testing_data")

        count = 0
        for image_dir in next(os.walk(dataset_dir))[1]:
            path = os.path.join(dataset_dir, os.path.join(image_dir, 'image/image.tif'))
            self.add_image('cells', count, path)
            count +=1
    
    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])

        # If 16bit, convert to 8bit
        # if image.dtype=='uint16': 
        #    image = self.map_uint16_to_uint8(image, lower_bound=None, upper_bound=None)

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image
    
    
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        
        # Get mask directory from image path
        mask_dir = os.path.join(os.path.dirname(os.path.dirname(info['path'])), "masks")
        # print(mask_dir)
        # Read mask files from .tif image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith('.tif'):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32) 
 
