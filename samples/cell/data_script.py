from data import np_to_image
from data import train_test_split 
from data import reset
import os

if __name__ == '__main__':
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate data, train/test split, reset train/test splits')
    parser.add_argument("command",
                        metavar="<command>",
                        help="either 'imgmask' 'split' 'reset'")
    parser.add_argument('--imgdir', required=True,
                        metavar="/path/to/image/directory/",
                        help='Directory of the np images')
    parser.add_argument('--images', required=False,
                        metavar="Image numpy file",
                        help='Name of image numpy file in imgdir')
    parser.add_argument('--masks', required=False,
                        metavar="Mask numpy file",
                        help='Name of mask numpy file in imgdir')
    parser.add_argument('--train-percent', required=False,
                        default=.7,
                        metavar='train percentage as decimal value',
                        help='Percentage of data to allocate for training')
    args = parser.parse_args()

    train_directory = os.path.join(imgdir, 'training_data')
    test_directory = os.path.join(imgdir, 'testing_data')
         
    if args.command == 'imgmask':
        assert [args.images, args.masks], 'imgmask requires the --images and --masks arguments'
        np_to_image(args.images, args.masks, args.imgdir)   
    
    if args.command == 'split':
        train_test_split(train_directory, test_directory, args.imgdir, args.train_percent)

    if args.command == 'reset':
        reset(train_directory, test_directory, args.imgdir)
        
