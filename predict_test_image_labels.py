import glob
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import random
import time

from keras.models import Sequential
from keras.models import model_from_json

from helper_functions import generate_test_df, preprocess_image
from helper_functions import load_and_preprocess

# export THEANO_FLAGS=blas.ldflags=

'''
Pass in
im_mean
imsize
'''


#Configuration
#n_images = 100
n_images = None     # Read and process all test images
imsize   = 64       # Square images
test_jpg_dir = 'data/test_photos/'

#%% Read in the images

print 'Read and preprocessing {} images'.format(n_images)
start_time = time.time()
im_files = glob.glob(test_jpg_dir + '*.jpg')

if n_images:
    im_files = random.sample(im_files, n_images)        # Forget other files
    test_images = Parallel(n_jobs=6)(delayed(load_and_preprocess)(im_file) for im_file in im_files)
else:
    test_images = Parallel(n_jobs=6)(delayed(load_and_preprocess)(im_file) for im_file in im_files)
    n_images = len(test_images)


#del a, im_files        # conserve memory
elapsed_time = time.time() - start_time
print "Took %.1f seconds and %.1f ms per image" % (elapsed_time,
                                                   1000*elapsed_time/n_images)
#%% Read and join biz_ids on photo_id


test_df = []

test_df = pd.DataFrame(test_df, columns=['filepath'])

test_df['photo_id'] = test_df.filepath.str.extract('(\d+)')
test_df.photo_id = test_df.photo_id.astype('int')


photo_biz_ids_df = pd.read_csv('test_photo_to_biz.csv')
# Column names: photo_id, business_id


test_df = pd.merge(test_df, photo_biz_ids_df, on='photo_id')

#%% Make a tensor
print 'Making tensor...'

tensor = np.zeros((n_images,imsize,imsize,3))

for i in range(n_images):
    tensor[i] = test_df.image[i]

# Reshape to fit Theanos format
tensor = tensor.reshape(n_images,3,imsize,imsize)

# Clean up
del photo_biz_ids_df, i

test_df.drop('image', axis=1, inplace=True)


#%% Final processing and setup
if im_mean is None:
    im_mean = tensor.mean()
tensor -= im_mean       # Subtract the mean


#%% Load model reconstruction from JSON:

model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')


#%%
# Threshold at 0.5 and convert to 0 or 1

# model.predict_classes() -- use with class_mode = 'binary'
X_test_prediction = (model.predict(tensor) > .5)*1


#%% Compile a Test DataFrame

test_df = generate_test_df(test_df, ['photo_id', 'business_id'],
                           X_test_prediction, np.arange(n_images))

#%% Generate a submission

test_df[['business_id', 'predicted_labels']].to_csv('csv_testing_yo.csv',
                                                    index=False,
                                                    header=['business_id','labels']
                                                    compression='gzip')
