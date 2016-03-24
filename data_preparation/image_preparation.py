import glob
# from joblib import Parallel, delayed
from data_preparation.helper_functions import load_and_preprocess
import cPickle as pickle
import time

#----------------Configuration----------------------------
# n_images = 4000
n_images = None     # Read and process all test images
imsize   = 64       # Square images
#n_jobs = 3

#---------------------------------------------------------

class ImageLoader(object):
    def __init__(self, save_file_name='data/all_train_photos',
                 photo_dir='data/train_photos/'):
        self.save_file_name = save_file_name
        self.photo_dir = photo_dir

    def load_images(self):
        #%% Read in the images
        im_files = glob.glob(test_jpg_dir + '*.jpg')
        if n_images:
            im_selection = im_files[:n_images]
        else:
            im_selection = im_files
            n_images = len(im_selection)

        #test_images = Parallel(n_jobs=n_jobs)(delayed(load_and_preprocess)(im_file) for im_file in im_selection)
        test_images = [load_and_preprocess(im_file) for im_file in im_selection]

    def save_images_to_pickle(self):
        #%% Save the images and file names
        with open(save_file_name+'_images' + '.pkl', 'wb') as out_file:
           pickle.dump(test_images, out_file, 2)

        with open(save_file_name+'_im_files'+'.pkl', 'wb') as out_file:
           pickle.dump(im_selection, out_file, 2)
