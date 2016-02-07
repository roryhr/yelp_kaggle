import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np      # 1.10.1
import pandas as pd
import glob
import random
import time
from sklearn.cross_validation import train_test_split
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from helper_functions import load_and_preprocess
from helper_functions import mean_f1_score

from joblib import Parallel, delayed

# THEANO_FLAGS='floatX=float32,blas.ldflags=,OMP_NUM_THREADS=3,openmp=True' python cnn_training.py

# TODO: make sure I'm useing float32 instad of float64
#       use list comprehension and parallelize image preprocessing


#%% Configuration 

n_images = 20000
imsize   = 100  # Square images

model_name = 'feb_6'

#csv_dir = '/home/ubuntu/data/yelp/'   # Folder for csv files    
csv_dir = 'data/'   # Folder for csv files    

#jpg_dir = '/home/ubuntu/data/yelp/train_photos/'
jpg_dir = 'data/train_photos/'

models_dir = 'models/'


#%% Read in the images

print 'Read and preprocessing {} images'.format(n_images)

start_time = time.time()

im_files = glob.glob(jpg_dir + '*.jpg')
im_files = random.sample(im_files, n_images)  # Might as well forget other files for now

#%% Load and preprocess images

train_images = []

train_images = Parallel(n_jobs=4)(delayed(load_and_preprocess)(im_file) for im_file in im_files)

#%%
train_df = pd.DataFrame(im_files, columns=['filepath'])
#plt.imshow(train_images[0])

train_df['photo_id'] = train_df.filepath.str.extract('(\d+)')
train_df.photo_id = train_df.photo_id.astype('int')

elapsed_time = time.time() - start_time

print "Took %.1f seconds and %.1f ms per image" % (elapsed_time,
                                                   1000*elapsed_time/n_images)
#%% Read and join biz_ids on photo_id

photo_biz_ids_df = pd.read_csv(csv_dir + 'train_photo_to_biz_ids.csv')  
# Column names: photo_id, business_id

train_df = pd.merge(train_df, photo_biz_ids_df, on='photo_id')

#%% Read and join train labels, set to 0 or 1

train_labels_df = pd.read_csv(csv_dir + 'train.csv')
# Column names: business_id, labels

# Work column-wise to encode the labels string into 9 new columns
for i in '012345678':
    train_labels_df[i] = train_labels_df['labels'].str.contains(i) * 1

train_labels_df = train_labels_df.fillna(0)

train_df = pd.merge(train_df, train_labels_df, on='business_id')

# Convert labels to integer
train_df[train_df.columns[5:]] = train_df[train_df.columns[5:]].astype('int')

#%% Make a tensor
print 'Making tensor...'

if len(train_df) != n_images:
    print "Lost an image somewhere!"
    n_images = len(train_df) 

    
tensor = np.zeros((n_images,imsize,imsize,3))

for i in range(n_images):
    tensor[i] = train_images[i]

'''
Reshape to fit Theanos format 
dim_ordering='th'
(samples, channels, rows, columns)
'''
tensor = tensor.reshape(n_images,3,imsize,imsize)

tensor = tensor.astype('float32')

#%% Clean up and save memory

del train_labels_df, photo_biz_ids_df, i, train_images

label_start = 4  # Column number where labels in train_df start

#%% Final processing and setup

im_mean = tensor.mean()
tensor -= im_mean       # Subtract the mean

X_train_ind, X_test_ind, Y_train_ind, Y_test_ind = train_test_split(
                                                 range(n_images),
                                                 range(n_images), 
                                                 test_size=0.1, random_state=4)

print 'Mean for all images: {}'.format(im_mean)

#%% VGG-like covnet

''' 
Include BatchNormalization

Final layer use sigmoid activation, binary_crossentropy loss
'''
    
model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(16, 3, 3, input_shape=(3, imsize, imsize))) # 32,3,3
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(16, 3, 3, W_regularizer=l2(l=0.01)))  #32
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
#model.add(Dropout(0.25))

model.add(Convolution2D(32, 3, 3, W_regularizer=l2(.01)))   # 64, 3, 3
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(128, W_regularizer=l2(.01)))       # 256
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(128, W_regularizer=l2(l=0.01)))       # 256
model.add(Activation('relu'))
#model.add(Dropout(0.5))

model.add(Dense(9, W_regularizer=l2(.01)))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)


#%% Fit model

model.fit(tensor[X_train_ind],
          train_df.iloc[Y_train_ind,label_start:].values, 
          batch_size=32, nb_epoch=2,
          validation_data=(tensor[X_test_ind],
                           train_df.iloc[Y_test_ind,label_start:].values),
          show_accuracy=True, verbose=1)

#%% Compute mean_f1_score
''' Calculate the Mean F1 Score

                        Mean F1 Score
    Sample submission   0.36633 	
    Random submission   0.43468
    Benchmark           0.64590 	
    Leader (1/17)       0.81090
'''

# Threshold at 0.5 and convert to 0 or 1 
X_test_prediction = (model.predict(tensor[X_test_ind]) > .5)*1


print 'Mean F1 Score: %.2f' % mean_f1_score(X_test_prediction,
                                            train_df.iloc[Y_test_ind,label_start:].values)
    
    
#%% Save model as JSON
    
json_string = model.to_json()
open(models_dir + model_name + '.json', 'w').write(json_string)
model.save_weights(models_dir + model_name + '.h5')  # requires h5py


#%% Compile a Test DataFrame 

#test_df = generate_test_df(train_df, ['photo_id', 'business_id', 'labels'], 
#                           X_test_prediction, X_test_ind)


    