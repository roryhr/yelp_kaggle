import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import pandas as pd
import glob
import random
import time

from keras.models import Sequential
from keras.models import model_from_json


from helper_functions import generate_test_df, preprocess_image, mean_f1_score

# export THEANO_FLAGS=blas.ldflags=

'''
Pass in 
im_mean
imsize
'''


#Configuration 
n_images = 500
imsize   = 100  # Square images
    

#%% Read in the images

print 'Read and preprocessing {} images'.format(n_images)

start_time = time.time()

im_files = glob.glob("/home/rory/kaggle/yelp/test_photos/*.jpg")

test_df = []


for file_path in random.sample(im_files, n_images):
    b = Image.open(file_path)
    if b.layers == 3:
        test_df.append([preprocess_image(b,imsize,imsize), 
                        file_path] )
    else:
        print "Doesn't have 3 layers, ignoring image"
    
test_df = pd.DataFrame(test_df, columns=['image', 'filepath'])
#plt.imshow(train_df.image[0])
#del a, im_files        # conserve memory


test_df['photo_id'] = test_df.filepath.str.extract('(\d+)')
test_df.photo_id = test_df.photo_id.astype('int')

elapsed_time = time.time() - start_time

print "Took %.1f seconds and %.1f ms per image" % (elapsed_time,
                                                   1000*elapsed_time/n_images)
#%% Read and join biz_ids on photo_id

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
#im_mean = tensor.mean()
tensor -= im_mean       # Subtract the mean


#%% Load model reconstruction from JSON:

# elsewhere...
model = model_from_json(open('my_model_architecture.json').read())
model.load_weights('my_model_weights.h5')



# Threshold at 0.5 and convert to 0 or 1

# model.predict_classes() -- use with class_mode = 'binary'
X_test_prediction = (model.predict(tensor[X_test_ind]) > .5)*1


#%% Compile a Test DataFrame 

test_df = generate_test_df(test_df, ['photo_id', 'business_id', 'labels'], 
                           X_test_prediction, X_test_ind)



