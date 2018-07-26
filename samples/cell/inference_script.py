from inference import stitched_inference
from inference import CleanMask
from PIL import Image
import numpy as np
# example
# python inference_script.py /data/kimjb/Mask_RCNN/image_test/to_caltech/exp3/AUTO0218_D08_T0001F004L01A01Z01C01.tif /data/kimjb/Mask_RCNN/logs/cells20180719T1559/mask_rcnn_cells.h5. --cropsize=256 --padding=40

if __name__ == '__main__':
    import argparse
    import os 
    import sys

    parser = argparse.ArgumentParser(description='Run inference on an image for cell segmentation')

    parser.add_argument('image',
                        help='/path/to/image')
    parser.add_argument('model',
                        help='/path/to/model')
    parser.add_argument('output',
                        help='/path/to/output/image.tif')
    parser.add_argument('--cropsize', required=False,
                        default='256',
                        help='Size of patches. Must be multiple of 256')
    parser.add_argument('--padding', required=False,
                        default='40',
                        help='Amount of overlapping pixels along one axis') 


    args = parser.parse_args()
    
    image = Image.open(args.image)
    imarray = np.array(image)
    padding = int(args.padding)
    cropsize = int(args.cropsize) 
    stitched_masks = stitched_inference(imarray, args.model, cropsize, padding)

    cleaned_masks = CleanMask(stitched_masks)
    cleaned_masks.cleanup()
    cleaned_masks.save(args.output)    
    print('Done. Saved masks to {}.'.format(args.output)) 
