#!/usr/bin/env python
import scipy.misc
import os
from PIL import Image
import numpy as np
import sys
from hashlib import sha224
import random
import shutil

TRAIN = .7 # 70% training
# TEST = 1-TRAIN

# generates mask dictionary, each key with a single mask instance as a value
def generate_masks(mask_array):
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
        #print('Made directory ' + directory)


if __name__ == '__main__':
    task = sys.argv[1]
    assert task in ['reset', 'train_test_split', 'np_to_image', 'fov_to_image'], "first arg must be either 'np_to_image', 'train_test_split', 'reset', or 'fov_to_image'"
   
    if task == 'np_to_image':
        image_file = sys.argv[2]
        mask_file = sys.argv[3]

        if not os.path.isfile(image_file):
            print('{image_file} does not exist'.format(image_file=image_file))
            sys.exit()
        if not os.path.isfile(mask_file):
            print('{mask_file} does not exist'.format(mask_file=mask_file))
            sys.exit()
    
    if task == 'fov_to_image':
        image_file = sys.argv[2]
        if not os.path.isfile(image_file):
            print('{image_file} does not exist'.format(image_file=image_file))
            sys.exit()
            
    image_dir = "images" #parent dir to hold all train/test images
    makedir(image_dir)

    train_directory = os.path.join(image_dir, 'training_data')
    test_directory = os.path.join(image_dir, 'testing_data')

    if task == 'reset': # if data is split into train and test data, move all images back to image_dir
        if os.path.exists(train_directory):
            for filename in os.listdir(train_directory):
                shutil.move(os.path.join(train_directory, filename), image_dir)

        if os.path.exists(test_directory):
            for filename in os.listdir(test_directory):
                shutil.move(os.path.join(test_directory, filename), image_dir)

        os.rmdir(train_directory)
        os.rmdir(test_directory)
        print('Done. All training and testing data merged to ' + image_dir)

    if task == 'train_test_split':
        makedir(train_directory)
        makedir(test_directory)

        list_of_directories = []
        filenames= os.listdir(image_dir)
        for filename in filenames: # loop through all the files and folders
            if os.path.isdir(os.path.join(image_dir, filename)) and (filename != 'training_data' and filename != 'testing_data'): # check whether the current object is a folder or not
                list_of_directories.append(filename)
        """for directory in os.walk(image_dir)[1]:
            if directory != 'training_data' or directory != 'testing_data':
                list_of_directories.append(directory)
        """ 
        num_images = len(list_of_directories)
        num_train = int(num_images*TRAIN)
        num_test = num_images - num_train

        random.shuffle(list_of_directories)
        for i in range(num_train):
            destination = os.path.join(train_directory, list_of_directories[i])
            shutil.move(os.path.join(image_dir, list_of_directories[i]), destination)
        for i in range(num_train, num_images):
            destination = os.path.join(test_directory, list_of_directories[i])
            shutil.move(os.path.join(image_dir, list_of_directories[i]), destination)

        # print('Done. Completed a train/test split of the data in ' + image_dir + '. There were ' + num_train + ' training images and ' + num_test + ' testing images.')
        print('Done. Completed a train/test split of the data in {image_dir}. There were {num_train} training images and {num_test} testing images.'.format(image_dir=image_dir, num_train=str(num_train), num_test=str(num_test)))

    if task == 'np_to_image':
        image_stack_array = np.load(image_file) # loads .npy file that contains images
        mask_stack_array = np.load(mask_file) # loads .npy file that contains GT masks
        
        num_images = image_stack_array.shape[0]
        num_masks = mask_stack_array.shape[0]
        assert num_images==num_masks, 'number of images must equal number of mask images'

        for i in range(num_images):
            image_array = image_stack_array[i]
            mask_array = mask_stack_array[i]

            image_name = os.path.join(image_dir, sha224(image_array.tostring()).hexdigest()) # give unique name to each set of images/masks
            # make directory for each set and then image and masks subdirectories
             
            makedir(image_name)
            image_instance_dir = os.path.join(image_name, 'image')
            mask_instance_dir = os.path.join(image_name, 'masks')

            makedir(image_instance_dir)
            makedir(mask_instance_dir)
            
            scipy.misc.imsave(os.path.join(image_instance_dir, 'image.tif'), image_array)

            masks = generate_masks(mask_array)
            count = 0
            for index in masks:
                instance_file_name = 'mask_instance_' + '{:02d}'.format(count) + '.tif' # zero pad for naming
                instance_array = masks[index]
                scipy.misc.imsave(os.path.join(mask_instance_dir, instance_file_name),instance_array)
                count += 1
        print('Done. Converted numpy array to dataset.')

    if task == 'fov_to_image':
        image_dir = 'fov'
        makedir(image_dir)
        image_stack_array = np.load(image_file) # loads .npy file that contains FOV images
        num_images = image_stack_array.shape[0]        
        
        for i in range(num_images):
            image_array = image_stack_array[i]
            image_name = os.path.join(image_dir, sha224(image_array.tostring()).hexdigest())
            makedir(image_name)
            image_instance_dir = os.path.join(image_name, 'image')
            makedir(image_instance_dir)
            scipy.misc.imsave(os.path.join(image_instance_dir, 'image.tif'), image_array)