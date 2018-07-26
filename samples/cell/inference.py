import tensorflow as tf
import numpy as np
import time
from cell import CellsConfig
import mrcnn.model as modellib
import skimage.io
import sys

def mask_stack_to_single_image(image, masks, offset):
    """
    Merge a stack of masks containing multiple instances to one large image.

    Args:
        image: full fov np array
        masks: stack of masks of shape [h,w,n]. Note that image.shape != masks.shape, because the shape of the masks is the size of the inference call. Since we are doing inference in patches, the masks are going to be of size of the patch.
        offset: list of length 2 that describes where to place the masks in image

    Returns:
        image that is the same shape as the original raw image, containing all of the masks from the mask stack
    """
    # switch shape to [num_masks, h, w] from [h, w, num_masks]
    masks = masks.astype(int) 
    masks = np.moveaxis(masks, -1, 0) 
    
    num_masks = masks.shape[0] # shape = [num_masks,h,w]
    
    current_id = 1
    
    for i in range(num_masks):
        current_mask = masks[i]
        image = add_mask_to_ids(image, current_mask, offset, current_id)
        current_id += 1
        
    return image


def add_mask_to_ids(image, mask, offset, fill_int):
    """
    Same as mask_stack_to_single_image but is just a helper function. Merges one mask from the stack into image. Gives unique id to each mask
    """
    for (row,col), value in np.ndenumerate(mask):
        if value != 0 and image[row + offset[0]-1,col + offset[1]-1] == 0:
            image[row + offset[0]-1,col + offset[1]-1] = fill_int
    return image


def run_inference(model, image):
    """
    Runs inference on an image using model and returns the mask stack.
    """
    results = model.detect([image], verbose=1)
    r = results[0]
    masks = r['masks']
    return masks


def generate_inference_model(model_path, cropsize):
    """
    Generates an inference model from the model_path. cropsize is how big of a patch to run inference on.
    """
    class InferenceConfig(CellsConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        # comment below if running inference on small crops
        TRAIN_ROIS_PER_IMAGE = 2000
        POST_NMS_ROIS_INFERENCE = 13000
        DETECTION_MAX_INSTANCES = 200
        #DETECTION_NMS_THRESHOLD = 0.35
        IMAGE_MIN_DIM = cropsize #math.ceil(mindim / 256) * 256
        IMAGE_MAX_DIM = cropsize #math.ceil(maxdim / 256) * 256

    inference_config = InferenceConfig()

    # Recreate the model in inference mode
    DEVICE = '/device:GPU:0'
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=model_path)

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    return model


def stitched_inference(image, model_path, cropsize, padding=40):#, minsize=100):
    """
    Runs multiple inferences on different patches of the entire image and stitches them back together.

    Args:
        image: image to run inference on
        cropsize: size of patches to run inference on (must be multiple of 256)
        model_path: /path/to/model
        padding: number of pixels the patches will overlap

    Returns: 
        One image that contains all of the masks. Note: Due to patching, there are some split cells and cells that have overlapping. Use the CleanMasks class to fix this issue.
    """
    final_image = np.zeros(image.shape[0:2]) # make new image of zeros (exclude third dimension, not using rgb)
    print(final_image.shape)
    visited = set()
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image) 
    if image.dtype == 'float32':
        image = image.astype('uint16')
    num_row = image.shape[0] # num rows in the image
    num_col = image.shape[1]
    print(image.shape)
    
    row = 0
    col = 0
    
    model = generate_inference_model(model_path, cropsize)
    
    for row in np.arange(0, num_row, cropsize-padding): # row defines the rightbound side of box
        for col in np.arange(0, num_col, cropsize-padding): # col defines lowerbound of box
            upperbound = row
            lowerbound = row + cropsize
            leftbound  = col
            rightbound = col + cropsize
            
            if lowerbound > num_row:
                lowerbound = num_row
                upperbound = num_row-cropsize
            
            if rightbound > num_col:
                rightbound = num_col
                leftbound  = num_col-cropsize

            #print('bounds:')
            #print('upper: {}'.format(upperbound))
            #print('lower: {}'.format(lowerbound))
            #print('left : {}'.format(leftbound))
            #print('right: {}'.format(rightbound))
            
            
            cropped_image = image[upperbound:lowerbound, leftbound:rightbound, :]
            #print('cropped image shape: {}'.format(cropped_image.shape))
            #print(cropped_image.shape)
            
            masks = run_inference(model, cropped_image)
            
            #padded_masks = pad(masks, [num_row, num_col, masks.shape[2]], [upperbound,leftbound,0])
            #print('mask shape:')
            #print (padded_masks.shape)
            
            final_image = mask_stack_to_single_image(final_image, masks, [upperbound, leftbound])
            
    return final_image


def pad(arrays, reference, offsets):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.full((reference[0],reference[1],reference[2]), False, dtype=bool)
    print('result:')
    print(result.shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + arrays.shape[dim]) for dim in range(arrays.ndim)]
    #print(insertHere)
    #print(arrays.shape)
    # Insert the array in the result at the specified offsets
    result[insertHere] = arrays
    return result



class CleanMask():

    """
    Use the cleanup function to fix overlapping and split instances generated from stitched_inference.
    """
    def __init__(self, image):
        self.image = image
        self.num_row = self.image.shape[0]
        self.num_col = self.image.shape[1]
        self.masks = np.zeros((self.num_row, self.num_col))
        self.visitedPoints = set()
        self.id = 0
        sys.setrecursionlimit(25000)

        
    def inBounds(self, row, col):
        if (row < 0 or row >= self.num_row):
            return False
        if (col < 0 or col >= self.num_col):
            return False
        return True
    
    def getMasks(self):
        return self.masks
    
    def visited(self, row, col):
        if (row,col) in self.visitedPoints:
            return True
    
    def dfs(self, row, col):
        if not self.visited(row,col):
            if self.image[row,col] > 0:
                self.masks[row,col] = self.id
                #print('[{},{}]'.format(row, col))
            self.visitedPoints.add((row,col))

            if self.inBounds(row+1, col):
                if self.image[row+1, col] > 0:
                    self.dfs(row+1, col)
                    
            if self.inBounds(row, col+1):
                if self.image[row, col+1] > 0:
                    self.dfs(row, col+1)
                    
            if self.inBounds(row, col-1):
                if self.image[row, col-1] > 0:
                    self.dfs(row, col-1)
            
            if self.inBounds(row-1, col):
                if self.image[row-1, col] > 0:
                    self.dfs(row-1, col)
            
    def cleanup(self):
        for (row,col), value in np.ndenumerate(self.image):
            if value > 0 and not self.visited(row,col):
                self.id += 1
                self.dfs(row, col)

    def save(self, save_path):
        from scipy.misc import imsave
        imsave(save_path, self.getMasks())



