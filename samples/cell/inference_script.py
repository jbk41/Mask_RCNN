from inference import stitched_inference
from inference import CleanMask
from inference import generate_inference_model
from inference import preprocess
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


    # TGAR, something weird - I trained the model by with 16bit images but when I ran inference, I got much better results passing in 8bit images. I have not figured out why the program has trouble with displaying and running inference on 16 bit images. So here, I am converting the image to 8bit before running inference.    
    image = preprocess(args.image)
    padding = int(args.padding)
    cropsize = int(args.cropsize) 

    model = generate_inference_model(args.model, 512)
    import time
    start = time.time()
    stitched_inference_stack, num_times_visited = stitched_inference(image, cropsize, model, padding=padding)
    masks = CleanMask(stitched_inference_stack, num_times_visited)
    masks.cleanup()
    masks.save(args.output)
    end = time.time()
    
    print('Done. Saved masks to {}. Took {} seconds'.format(args.output, end-start)) 
