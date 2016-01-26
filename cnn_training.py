import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import pandas as pd
import glob
import random
import time
from sklearn.cross_validation import train_test_split

from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

# export THEANO_FLAGS=blas.ldflags=

#Configuration 
n_images = 100
imsize   = 100  # Square images


def preprocess_image(im,width=imsize,height=imsize):
    ''' INPUT: PIL Image. OUTPUT: RGB PIL Image
        OUTPUT: Rescaled image and crop
    '''
    
#    im.thumbnail((300,300))   # Resize inplace to fit in 300x300
    
    
#    if im.size < (width, height):
#        print 'Image size is smaller than width and height'
#        return im.crop((0,0,width,height))
#            
#            
#    # Random crop
#    x_left = np.random.randint(0, im.size[0]-width)
#    y_top = np.random.randint(0, im.size[1]-height)
#    
#    return im.crop((x_left,y_top,x_left+width,y_top+height))
    
    return im.resize((width,height))


def mean_f1_score(Y_pred, Y_true):
    '''Y_pred int64 array size (n_samples, 9)'''
    
    tp = (((Y_pred == Y_true)) * (Y_pred == 1)).sum() # True positives
    fp = (((Y_pred != Y_true)) * (Y_pred == 1)).sum() # False positives
    # tp + fp == Y_pred.sum()
    
    fn = (((Y_pred != Y_true)) * (Y_pred == 0)).sum() 
    
    p = tp*1.0/(tp + fp)
    r = tp*1.0/(tp + fn)
    
    return 2.0*p*r/(p + r)
    

#%% Read in the images

print 'Read and preprocessing {} images'.format(n_images)

start_time = time.time()

im_files = glob.glob("/home/rory/kaggle/yelp/train_photos/*.jpg")

train_df = []


for file_path in random.sample(im_files, n_images):
    b = Image.open(file_path)
    if b.layers == 3:
        train_df.append([preprocess_image(b,imsize,imsize), 
                        file_path] )
    else:
        print "Doesn't have 3 layers, ignoring image"
    
train_df = pd.DataFrame(train_df, columns=['image', 'filepath'])
#plt.imshow(train_df.image[0])
#del a, im_files        # conserve memory


train_df['photo_id'] = train_df.filepath.str.extract('(\d+)')
train_df.photo_id = train_df.photo_id.astype('int')

elapsed_time = time.time() - start_time

print "Took %.1f seconds and %.1f ms per image" % (elapsed_time,
                                                   1000*elapsed_time/n_images)
#%% Read and join biz_ids on photo_id

photo_biz_ids_df = pd.read_csv('train_photo_to_biz_ids.csv')  
# Column names: photo_id, business_id


train_df = pd.merge(train_df, photo_biz_ids_df, on='photo_id')

#%% Read and join train labels, set to 0 or 1

train_labels_df = pd.read_csv('train.csv')
# Column names: business_id, labels

# DEBUG FILL NaNs with 0's
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
#    tensor[i] = train_df.image.iloc[i][:imsize,:imsize,:]
    tensor[i] = train_df.image[i]

# Reshape to fit Theanos format 
tensor = tensor.reshape(n_images,3,imsize,imsize)

# Clean up
del train_labels_df, photo_biz_ids_df, i    

train_df.drop('image', axis=1, inplace=True)
label_start = 4  

#%% Final processing and setup
im_mean = tensor.mean()
tensor -= im_mean       # Subtract the mean

X_train_ind, X_test_ind, Y_train_ind, Y_test_ind = train_test_split(
                                                 range(n_images),
                                                 range(n_images), 
                                                 test_size=0.1, random_state=4)

#%% VGG-like covnet
    
model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution2D(32, 3, 3, input_shape=(3, imsize, imsize))) # 32,3,3
model.add(Activation('relu'))
#model.add(BatchNormalization(input_shape=(3,imsize, imsize)))
model.add(Convolution2D(32, 3, 3))  #32
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3))   # 64, 3, 3
model.add(Activation('relu'))
#model.add(BatchNormalization())
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))       # 256
model.add(Activation('relu'))
model.add(Dense(256))       # 256
model.add(Activation('relu'))
#model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(9))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


#%% Fit model

model.fit(tensor[X_train_ind],
          train_df.iloc[Y_train_ind,label_start:].values, 
          batch_size=16, nb_epoch=1,
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

#model.evaluate(X_test, Y_test)
print 'Mean F1 Score: %.2f' % mean_f1_score(model.predict(tensor[X_test_ind]), train_df.iloc[Y_test_ind,label_start:].values)
    
    

