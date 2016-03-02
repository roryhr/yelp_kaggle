import cPickle as pickle

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

#------------------------Configuration----------------------------
csv_dir = 'data/'                           # Folder for csv files
test_jpg_dir = 'data/test_photos/'
save_file_name = 'data/all_test_photos'
#-----------------------------------------------------------------

#%% Load in preprocessed images from pickle files
with open(save_file_name+'_images' + '.pkl', 'rb') as in_file:
   test_images = pickle.load(in_file)

with open(save_file_name+'_im_files'+'.pkl', 'rb') as in_file:
   im_files = pickle.load(in_file)


#%% Read and join biz_ids on photo_id
test_df = pd.DataFrame(im_files, columns=['filepath'])

test_df['photo_id'] = test_df.filepath.str.extract('(\d+)')
test_df.photo_id = test_df.photo_id.astype('int')

# Column names: photo_id, business_id
photo_biz_ids_df = pd.read_csv(csv_dir+'test_photo_to_biz.csv')

test_df = pd.merge(test_df, photo_biz_ids_df, on='photo_id')

#%% Make a tensor
print 'Making tensor...'

tensor = np.zeros((n_images,imsize,imsize,3))

for i in range(n_images):
    tensor[i] = test_images[i]

# Reshape to fit Theanos format
tensor = tensor.reshape(n_images,3,imsize,imsize)

# Clean up
del photo_biz_ids_df, i


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
