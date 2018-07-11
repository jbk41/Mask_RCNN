import scipy.misc
import os
from PIL import Image
import numpy as np
import sys
from hashlib import sha224
import random
import shutil

### Basic tools



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

def makedir(directory): # makes directory 
    if not os.path.exists(directory):
        os.makedirs(directory)


def reset(train_directory, test_directory, image_dir)
    if os.path.exists(train_directory):
        for filename in os.listdir(train_directory):
            shutil.move(os.path.join(train_directory, filename), image_dir)

        if os.path.exists(test_directory):
            for filename in os.listdir(test_directory):
                shutil.move(os.path.join(test_directory, filename), image_dir)

        os.rmdir(train_directory)
        os.rmdir(test_directory)
        print('Done. All training and testing data merged to ' + image_dir)

