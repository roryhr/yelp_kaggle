#import matplotlib.image as mpimg
import numpy as np      # 1.10.1
import pandas as pd
import glob
import random
import time
from sklearn.cross_validation import train_test_split
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from helper_functions import load_and_preprocess, mean_f1_score
from helper_functions import show_image_labels

from joblib import Parallel, delayed
import tables

# THEANO_FLAGS='floatX=float32,blas.ldflags=,OMP_NUM_THREADS=2,openmp=True' python cnn_training.py


#%% Configuration 
number_of_epochs = 30
n_images = 19000
imsize   = 64  # Square images
save_model = False
show_plots = True
model_name = 'mar_7_0005'
csv_dir = 'data/'                        # Folder for csv files    
reload_images = False
jpg_dir = 'data/train_photos/'
models_dir = 'models/'
weight_decay = 0.01

#%% Read in the images
print 'Read and preprocessing {} images'.format(n_images)

start_time = time.time()

if reload_images:
    im_files = glob.glob(jpg_dir + '*.jpg')
    im_files = random.sample(im_files, n_images)  
    # Might as well forget other files for now
    
    train_images = []
    train_images = Parallel(n_jobs=3)(delayed(load_and_preprocess)(im_file) for im_file in im_files)
else:
    h5file = tables.open_file('data/all_photos.hdf5')
    im_table = h5file.root.train_images.images
    
    im_files = im_table.read(start=0, stop=n_images, field='file_name')
    train_images = im_table.read(start=0, stop=n_images, field='image')
    h5file.close()


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

if reload_images:
    tensor = np.zeros((n_images,imsize,imsize,3))
    
    for i in range(n_images):
        tensor[i] = train_images[i]
else:
    tensor = train_images
'''
Reshape to fit Theanos format 
dim_ordering='th'
(samples, channels, rows, columns)
vs
dim_ordering='tf'
(samples, rows, cols, channels)

'''
tensor = tensor.reshape(n_images,3,imsize,imsize)

tensor = tensor.astype('float32')

#%% Clean up and save memory

del train_labels_df, photo_biz_ids_df, i, train_images

label_start = 4  # Column number where labels in train_df start

#%% Final processing and setup

im_mean = tensor.mean()
tensor -= im_mean       # Subtract the mean

train_ind, test_ind, _, _ = train_test_split(range(n_images),
                                             range(n_images),
                                             test_size=0.1, 
                                             random_state=4)

print 'Mean for all images: {}'.format(im_mean)

#%% ResNet-like 

''' 
Include BatchNormalization

Final layer use sigmoid activation, binary_crossentropy loss

if strides is None:
    pool_size = pool_size
'''
    
model = Sequential()
# INPUT: 64x64 images with 3 channels -> (3, 64, 64) tensors.
# this applies 4 convolution filters of size 3x3 each.
model.add(Convolution2D(nb_filter=4, nb_row=7, nb_col=7,
                        W_regularizer=l2(weight_decay), 
                        input_shape=(3,imsize,imsize),
                        subsample=(2,2)
                        dim_ordering='th')) 
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(4, 3, 3, W_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
# OUTPUT: 32x32

model.add(Convolution2D(8, 3, 3, W_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(8, 3, 3, W_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3,3), pool_size=(2,2)))
# OUTPUT: 16x16       W_regularizer=l2(.01)

model.add(Convolution2D(16, 3, 3, W_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(16, 3, 3, W_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# OUTPUT: 8x8  

model.add(Convolution2D(16, 2, 2, W_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution2D(16, 2, 2, W_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# OUTPUT: 4x4x32

model.add(Flatten())
model.add(Dense(256, W_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(256, W_regularizer=l2(weight_decay)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(9))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd)


#%% Fit model
model.fit(tensor[train_ind],
          train_df.iloc[train_ind,label_start:].values, 
          batch_size=32, nb_epoch=number_of_epochs,
          validation_data=(tensor[test_ind],
                           train_df.iloc[test_ind,label_start:].values),
          shuffle=True,
#          show_accuracy=True, 
          verbose=1)

#%% Compute mean_f1_score
''' Calculate the Mean F1 Score

                        Mean F1 Score
    Sample submission   0.36633 	
    Random submission   0.43468
    Benchmark           0.64590 	
    Leader (1/17)       0.81090
'''

# Threshold at 0.5 and convert to 0 or 1 
X_test_prediction = (model.predict(tensor[test_ind]) > .5)*1


print 'Mean F1 Score: %.2f' % mean_f1_score(X_test_prediction,
                                            train_df.iloc[test_ind,label_start:].values)
    
    
#%% Save model as JSON
if save_model:
    json_string = model.to_json()
    open(models_dir + model_name + '.json', 'w').write(json_string)
    model.save_weights(models_dir + model_name + '.h5')  # requires h5py
    
#%% Plot a few images to get a feel of how I did
if show_plots:
    for i in range(10):
        show_image_labels(tensor[test_ind[i]], X_test_prediction[i], 
                          train_df['labels'][test_ind[i]], im_mean)

