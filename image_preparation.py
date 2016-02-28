import glob
from joblib import Parallel, delayed

#import numpy as np
#import pandas as pd
#import random
import time


#from helper_functions import generate_test_df, preprocess_image
from helper_functions import load_and_preprocess

import cPickle as pickle

'''
Pass in
im_mean
imsize
'''


#----------------Configuration----------------------------
# n_images = 40000
n_images = None     # Read and process all test images
imsize   = 64       # Square images
#n_jobs = 3
test_jpg_dir = 'data/test_photos/'

save_file_name = 'data/test_images'
#---------------------------------------------------------


im_files = glob.glob(test_jpg_dir + '*.jpg')


#%% Read in the images
if n_images:
    im_selection = im_files[:n_images]
else:
    im_selection = im_files
    n_images = len(im_selection)

print 'Read and preprocessing {} images'.format(n_images)
print 'Done in about {} minutes'.format(n_images*0.0037/60)  # 3.7ms/image
#print 'n_jobs = {}'.format(n_jobs)
start_time = time.time()

#test_images = Parallel(n_jobs=n_jobs)(delayed(load_and_preprocess)(im_file) for im_file in im_selection)
test_images = [load_and_preprocess(im_file) for im_file in im_selection]

elapsed_time = time.time() - start_time
print "Took %.1f seconds and %.1f ms per image" % (elapsed_time,
                                                   1000*elapsed_time/n_images)

#%% Save the images and file names
with open(save_file_name+'_images' + '.pkl', 'wb') as out_file:
   pickle.dump(test_images, out_file, 2)
#
with open(save_file_name+'_im_files'+'.pkl', 'wb') as out_file:
   pickle.dump(test_images, out_file, 2)
