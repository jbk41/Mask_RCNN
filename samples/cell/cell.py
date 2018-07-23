import tensorflow as tf
import os
import sys
import random
import numpy as np
import cv2
import skimage.io
import warnings; warnings.simplefilter('ignore')
import time

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log


####################################################################
# CONFIGURATION
####################################################################

class CellsConfig(Config):
    NAME = "cells"
    
    GPU_COUNT = 1
    
    # To Reddy and George (TRAG), img/gpu could be increased to maximize training (i think I'm undersaturating the GPU so maybe we can increase this later)
    IMAGES_PER_GPU = 2
    
    NUM_CLASSES = 1+1 # background + cell
    
    # TRAG, change the following values based on the input size for training
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 1024 

    # TRAG, RPN_ANCHOR_SCALES can be decreased for smaller images. For example, the caltech images have very small cells so the following value can be decreased
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    #TRAIN_ROIS_PER_IMAGE = 512
    
    TRAIN_ROIS_PER_IMAGE = 200

    # batch_size = num_training_data/STEPS_PER_EPOCH
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50
    
    
    LEARNING_RATE = 1e-4 


####################################################################
# DATASET 
####################################################################
class CellsDataset(utils.Dataset):
    """Generates a cells dataset for training. Dataset consists of microscope images.
"""
    def generate_masks(mask_array):
        """
        Generate a dictionary of masks. The keys are instance numbers from the numpy stack and the values are the corresponding binary masks.

        Args:
            mask_array: numpy array of size [H,W]. 0 represents the background. Any non zero integer represents a individual instance

        Returns:
            Mask dictionary {instance_id: [H,W] numpy binary mask array}
        """
        masks = {} # keys are instances, values are corresponding binary mask array
        for (x,y), value in np.ndenumerate(mask_array): #go through entire array 
            if value != 0: # if cell
                if value not in masks: # if new instance introduced
                    masks[value] = np.zeros(mask_array.shape) #make new array
                dummy_array = masks[value]
                dummy_array[(x,y)] = 1
                masks[value] = dummy_array # change value of array to 1 to represent cell
        return masks
           
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
 


####################################################################
# TRAINING 
####################################################################

def train(dataset_dir, augmentation=None, init_with='coco', model_dir=None):
    MODEL_DIR = model_dir
    dataset_train = CellsDataset()
    dataset_train.load_cells(dataset_dir, subset="train")
    dataset_train.prepare()

    dataset_test = CellsDataset()
    dataset_test.load_cells(dataset_dir, subset="test")
    dataset_test.prepare()
    
    config = CellsConfig()
    config.display()

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)


    # Which weights to start with?
    # imagenet, coco, or last
    print('initializing with {}'.format(init_with))
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)


        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

    ### TRAIN THE MODEL
    DEVICE = '/device:GPU:0'
    with tf.device(DEVICE): 
        
        train_heads_start = time.time() 
        model.train(dataset_train, dataset_test, 
                    learning_rate=config.LEARNING_RATE,
                    augmentation=augmentation, 
                    epochs=75,
                    layers='heads')

        model.train(dataset_train, dataset_test, 
                    learning_rate=config.LEARNING_RATE / 10,
                    augmentation=augmentation, 
                    epochs=100,
                    layers='heads')

        model.train(dataset_train, dataset_test, 
                    learning_rate=config.LEARNING_RATE / 100,
                    augmentation=augmentation, 
                    epochs=125,
                    layers='heads')
        train_heads_end = time.time()
        train_heads_time = train_heads_end - train_heads_start
        print('\n Done training heads. Took {} seconds'.format(train_heads_time))

        # Fine tune all layers
        # Passing layers="all" trains all layers. You can also 
        # pass a regular expression to select which layers to
        # train by name pattern.
        train_all_start = time.time() 

        t1s = time.time()
        model.train(dataset_train, dataset_test, 
                    learning_rate=config.LEARNING_RATE / 10,
                    #augmentation=augmentation,
                    epochs=150, 
                    layers="all")
        t1e = time.time()
        print(t1e-t1s)

        t2s = time.time()
        model.train(dataset_train, dataset_test, 
                    learning_rate=config.LEARNING_RATE / 100,
                    #augmentation=augmentation,
                    epochs=175, 
                    layers="all")
        t2e = time.time()
        print(t2e-t2s)

        t3s = time.time()
        model.train(dataset_train, dataset_test, 
                    learning_rate=config.LEARNING_RATE / 1000,
                    #augmentation=augmentation,
                    epochs=200, 
                    layers="all")
        t3e = time.time()
        print(t3e-t3s)        

        train_all_end = time.time() 
        train_all_time = train_all_end - train_all_start
        print('\n Done training all layers. Took {} seconds'.format(train_all_time))

