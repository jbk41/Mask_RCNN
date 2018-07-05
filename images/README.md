# nih18
Scripts for data setup.

When training cell images using [Mask RCNN](https://github.com/matterport/Mask_RCNN), it is necessary to  separate a stack of images into individual files and then split the data into training and testing (validation) sets. The script `setup_data.py` can be used to first convert a numpy array of images and masks into individual images. Then, the data directory can be split into a training and testing set. 

To use the `setup_data.py` script, copy it into the directory where you want your output images to be. Then run

`
chmod +x setup_data.py
`

To convert .npy images and masks into image datasets, run 

`
python setup_data.py np_to_image <path_to_image_numpy_file> <path_to_mask_numpy_file>
`

This will create an `images` directory in your current directory and make a folder for each image/mask pair. In each subdirectory, there will be an image folder which will hold the `image` and a `masks` folder which will contain an image for each individual instance found in the image. 

To complete the train/test split, run 

`
python setup_data.py train_test_split
`

The default split is 70/30 but you can change line 11 in `setup_data.py`.
  
To reset or merge all training and testing data to the image directory, run

`
python setup_data.py reset
`
