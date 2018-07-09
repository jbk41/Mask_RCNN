# script for timing inference
# coding: utf-8

# In[ ]:

import tensorflow as tf


# In[ ]:

import warnings; warnings.simplefilter('ignore')


# In[ ]:

import os
import sys
import math
import random
import numpy as np
import cv2
#import Mask
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



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# In[ ]:

class CellsConfig(Config):
    NAME = "cells"
    
    GPU_COUNT = 1
    IMAGES_PER_GPU = 2
    
    NUM_CLASSES = 1+1 # background + cell
    
    # size of images are 256px X 256px
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256 
    
     # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    #TRAIN_ROIS_PER_IMAGE = 512
    TRAIN_ROIS_PER_IMAGE = 200
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50
    
    LEARNING_RATE = 1e-4 

config = CellsConfig()
config.display()


# In[ ]:

class CellsDataset(utils.Dataset):
    """Generates a cells dataset for training. Dataset consists of microscope images
    of one cell and a background. They each have a 256x256 resolution"""
        
    def load_cells(self, dataset_dir, subset):
        """Loads cell images from the dataset directory"""
        
        # Add class
        self.add_class("cells", 1, "cellobj")
 
        assert subset in ["train", "test", "fov"]
        if subset=="train":
            dataset_dir = os.path.join(dataset_dir, "training_data")
        elif subset=="test":
            dataset_dir = os.path.join(dataset_dir, "testing_data")
        elif subset=="fov":
            pass
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
        if image.dtype=='uint16': 
            image = self.map_uint16_to_uint8(image, lower_bound=None, upper_bound=None)
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
        print(mask_dir)
        # Read mask files from .jpg image
        mask = []
        for f in next(os.walk(mask_dir))[2]:
            if f.endswith('.tif'):
                m = skimage.io.imread(os.path.join(mask_dir, f)).astype(np.bool)
                mask.append(m)
        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32) 
    
    def map_uint16_to_uint8(self, img, lower_bound=None, upper_bound=None):
        '''
        Map a 16-bit image trough a lookup table to convert it to 8-bit.

        Parameters
        ----------
        img: numpy.ndarray[np.uint16]
            image that should be mapped
        lower_bound: int, optional
            lower bound of the range that should be mapped to ``[0, 255]``,
            value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
            (defaults to ``numpy.min(img)``)
        upper_bound: int, optional
           upper bound of the range that should be mapped to ``[0, 255]``,
           value must be in the range ``[0, 65535]`` and larger than `lower_bound`
           (defaults to ``numpy.max(img)``)

        Returns
        -------
        numpy.ndarray[uint8]
        '''
        
        if lower_bound is None:
            lower_bound = np.min(img)
        if not(0 <= lower_bound < 2**16):
            raise ValueError(
                '"lower_bound" must be in the range [0, 65535]')
        if upper_bound is None:
            upper_bound = np.max(img)
        if not(0 <= upper_bound < 2**16):
            raise ValueError(
                '"upper_bound" must be in the range [0, 65535]')    
        if lower_bound >= upper_bound:
            raise ValueError(
                '"lower_bound" must be smaller than "upper_bound"')
        lut = np.concatenate([
            np.zeros(lower_bound, dtype=np.uint16),
            np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
            np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
        ])
        return lut[img].astype(np.uint8)
        
    def pad(self, num):
        return '{0:04d}'.format(num)
    
    def size(self):
        return len(self._image_ids)

    


# In[ ]:
if __name__ == '__main__':
    dataset_dir = os.path.join(ROOT_DIR, '/data/kimjb/Mask_RCNN/images/images')


    dataset_train = CellsDataset()
    dataset_train.load_cells(dataset_dir, subset="train")
    dataset_train.prepare()

    dataset_test = CellsDataset()
    dataset_test.load_cells(dataset_dir, subset="test")
    dataset_test.prepare()



    # ### Image visualization

    # ### Inference (testing with and without GT)

    # In[ ]:

    
    # In[ ]:
    
    class InferenceConfig(CellsConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # below, comment if running inference on small crops
        #TRAIN_ROIS_PER_IMAGE = 2000
        POST_NMS_ROIS_INFERENCE = 2000
        DETECTION_MAX_INSTANCES = 200
        #DETECTION_NMS_THRESHOLD = 0.35

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")

    #model_path = model.find_last()[1]
    model_path = '/data/kimjb/Mask_RCNN_original/logs/cells20180628T1527/mask_rcnn_cells_0100.h5'

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)


    # In[ ]:

    import time
    cropped_list_time = []
    for i in range(50):
        # Run object detection
        image_id = random.choice(dataset_test.image_ids)
        cropped_time_start = time.time()
        image = dataset_test.load_image(image_id)
        results = model.detect([image]) 
        cropped_time_end = time.time()
        total_time = cropped_time_end - cropped_time_start
        cropped_list_time.append(total_time)
        #print('Time to load image, detect, and visualize for cropped (256x256 img): () '.format(cropped_time_end - cropped_time_start))
    print('Average time to load image, detect, and visualize for cropped (256x256 img) with 2000 region proposals: {} '.format(np.mean(cropped_list_time)))
    

    # In[ ]:

    dataset_dir = '/data/kimjb/Mask_RCNN/images/fov'

    fov_dataset = CellsDataset()
    fov_dataset.load_cells(dataset_dir, subset='fov')
    fov_dataset.prepare()


    # In[ ]:

    # Run object detection
    fov_2000_time = []
    for i in range(50):
        image_id = random.choice(fov_dataset.image_ids)
        fov_time_start_2000 = time.time()
        image = fov_dataset.load_image(image_id)
        results = model.detect([image])
        fov_time_end_2000 = time.time()
        total_time = fov_time_end_2000 - fov_time_start_2000
        fov_2000_time.append(total_time)
    print('Average time to load image, detect for fov (1024x1024 img) and 2000 region proposals: {} '.format(np.mean(fov_2000_time)))


    def pad(arrays, reference, offsets):
        """
        array: Array to be padded
        reference: Reference array with the desired shape
        offsets: list of offsets (number of elements must be equal to the dimension of the array)
        """
        # Create an array of zeros with the reference shape
        result = np.full((reference[0],reference[1],reference[2]), False, dtype=bool)
        print(result.shape)
        # Create a list of slices from offset to offset + shape in each dimension
        insertHere = [slice(offsets[dim], offsets[dim] + arrays.shape[dim]) for dim in range(arrays.ndim)]
        #print(insertHere)
        #print(arrays.shape)
        # Insert the array in the result at the specified offsets
        result[insertHere] = arrays
        return result


    # In[ ]:
    fov_stitched_2000_time = []
    for i in range(50):
        image_id = random.choice(fov_dataset.image_ids)
        fov_stitched_time_start_2000 = time.time()
        image = fov_dataset.load_image(image_id)

        b1 = image[0:512, 0:512,:]
        b1results = model.detect([b1])
        b1r = b1results[0]

        b2 = image[0:512, 512:,:]
        b2results = model.detect([b2])
        b2r = b2results[0]

        b3 = image[512:, 0:512,:]
        b3results = model.detect([b3])
        b3r = b3results[0]

        b4 = image[512:, 512:,:]
        b4results = model.detect([b4])
        b4r = b4results[0]

        b1roi = b1r['rois']
        b2roi = b2r['rois']
        b3roi = b3r['rois']
        b4roi = b4r['rois']

        b12 = np.concatenate((b1, b2), axis=1)
        b34 = np.concatenate((b3,b4), axis=1)
        b1234 = np.concatenate((b12, b34), axis=0)

        for instance in b2roi:
            instance[1] += 512
            instance[3] += 512
        for instance in b3roi:
            instance[0] += 512
            instance[2] += 512
        for instance in b4roi:
            instance[1] += 512
            instance[3] += 512
            instance[0] += 512
            instance[2] += 512
            
        

        #print(b1r['masks'].shape[2])
        b1masks = pad(b1r['masks'], [1024, 1024, b1r['masks'].shape[2]], [0,0,0])
        b2masks = pad(b2r['masks'], [1024, 1024, b2r['masks'].shape[2]], [0,512,0])
        b3masks = pad(b3r['masks'], [1024, 1024, b3r['masks'].shape[2]], [512,0,0])
        b4masks = pad(b4r['masks'], [1024, 1024, b4r['masks'].shape[2]], [512,512,0])



        b12roi = np.concatenate((b1roi, b2roi), axis=0)
        b12masks = np.concatenate((b1masks, b2masks), axis=2)
        b12ci = np.concatenate((b1r['class_ids'], b2r['class_ids']), axis=0)
        b12scores = np.concatenate((b1r['scores'], b2r['scores']), axis=0)

        b34roi = np.concatenate((b3roi, b4roi), axis=0)
        b34masks = np.concatenate((b3masks, b4masks), axis=2)
        b34ci = np.concatenate((b3r['class_ids'], b4r['class_ids']), axis=0)
        b34scores = np.concatenate((b3r['scores'], b4r['scores']), axis=0)

        b1234roi = np.concatenate((b12roi, b34roi), axis=0)
        b1234masks = np.concatenate((b12masks, b34masks), axis=2)
        b1234ci = np.concatenate((b12ci, b34ci), axis=0)
        b1234scores = np.concatenate((b12scores, b34scores), axis=0)

        fov_stitched_time_end_2000 = time.time()
        total_time = fov_stitched_time_end_2000 - fov_stitched_time_start_2000
        fov_stitched_2000_time.append(total_time)
    print('Average time to load image, detect for fov stitched (1024x1024 img) and 2000 region proposals: {} '.format(np.mean(fov_stitched_2000_time)))

        
    class InferenceConfig(CellsConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # below, comment if running inference on small crops
        #TRAIN_ROIS_PER_IMAGE = 2000
        POST_NMS_ROIS_INFERENCE = 13000
        DETECTION_MAX_INSTANCES = 200
        #DETECTION_NMS_THRESHOLD = 0.35

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=MODEL_DIR)

    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")

    #model_path = model.find_last()[1]
    model_path = '/data/kimjb/Mask_RCNN_original/logs/cells20180628T1527/mask_rcnn_cells_0100.h5'

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)


    # In[ ]:

    # FOV detect w 13000 region proposals
    fov_13000_time = []
    for i in range(50):
        image_id = random.choice(fov_dataset.image_ids)
        fov_time_start_13000 = time.time()
        image = fov_dataset.load_image(image_id)
        results = model.detect([image])
        fov_time_end_13000 = time.time()
        total_time = fov_time_end_13000 - fov_time_start_13000
        fov_13000_time.append(total_time)
    print('average time to load image, detect for fov (1024x1024 img) and 13000 region proposals: {} '.format(np.mean(fov_13000_time)))


    # In[ ]:
    fov_stitched_13000_time = []
    for i in range(50):
        image_id = random.choice(fov_dataset.image_ids)
        fov_stitched_time_start_13000 = time.time()
        image = fov_dataset.load_image(image_id)
        b1 = image[0:512, 0:512,:]
        b1results = model.detect([b1])
        b1r = b1results[0]

        b2 = image[0:512, 512:,:]
        b2results = model.detect([b2])
        b2r = b2results[0]

        b3 = image[512:, 0:512,:]
        b3results = model.detect([b3])
        b3r = b3results[0]

        b4 = image[512:, 512:,:]
        b4results = model.detect([b4])
        b4r = b4results[0]

        b1roi = b1r['rois']
        b2roi = b2r['rois']
        b3roi = b3r['rois']
        b4roi = b4r['rois']

        b12 = np.concatenate((b1, b2), axis=1)
        b34 = np.concatenate((b3,b4), axis=1)
        b1234 = np.concatenate((b12, b34), axis=0)

        for instance in b2roi:
            instance[1] += 512
            instance[3] += 512
        for instance in b3roi:
            instance[0] += 512
            instance[2] += 512
        for instance in b4roi:
            instance[1] += 512
            instance[3] += 512
            instance[0] += 512
            instance[2] += 512
            
        #print(b1r['masks'].shape[2])
        b1masks = pad(b1r['masks'], [1024, 1024, b1r['masks'].shape[2]], [0,0,0])
        b2masks = pad(b2r['masks'], [1024, 1024, b2r['masks'].shape[2]], [0,512,0])
        b3masks = pad(b3r['masks'], [1024, 1024, b3r['masks'].shape[2]], [512,0,0])
        b4masks = pad(b4r['masks'], [1024, 1024, b4r['masks'].shape[2]], [512,512,0])

        b12roi = np.concatenate((b1roi, b2roi), axis=0)
        b12masks = np.concatenate((b1masks, b2masks), axis=2)
        b12ci = np.concatenate((b1r['class_ids'], b2r['class_ids']), axis=0)
        b12scores = np.concatenate((b1r['scores'], b2r['scores']), axis=0)

        b34roi = np.concatenate((b3roi, b4roi), axis=0)
        b34masks = np.concatenate((b3masks, b4masks), axis=2)
        b34ci = np.concatenate((b3r['class_ids'], b4r['class_ids']), axis=0)
        b34scores = np.concatenate((b3r['scores'], b4r['scores']), axis=0)

        b1234roi = np.concatenate((b12roi, b34roi), axis=0)
        b1234masks = np.concatenate((b12masks, b34masks), axis=2)
        b1234ci = np.concatenate((b12ci, b34ci), axis=0)
        b1234scores = np.concatenate((b12scores, b34scores), axis=0)

        fov_stitched_time_end_13000 = time.time()
        total_time = fov_stitched_time_end_13000 - fov_stitched_time_start_13000 
        fov_stitched_13000_time.append(total_time)
    print('Time to load image, detect for fov stitched (4 1024x1024 imgs) and 13000 region proposals: {} '.format(np.mean(fov_stitched_13000_time)))

