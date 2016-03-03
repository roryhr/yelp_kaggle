import cPickle as pickle
import numpy as np
import pandas as pd
from keras.models import model_from_json
from helper_functions import generate_test_df

# export THEANO_FLAGS=blas.ldflags=

#%%----------------------Configuration----------------------------
model_name = 'mar_1_1852'

csv_dir = 'data/'                           # Folder for csv files
models_dir = 'models/'
#test_jpg_dir = 'data/test_photos/'
save_file_name = 'data/all_test_photos'
submission_file_name = 'submission_mar_3.csv.gz'

#im_mean = None
im_mean = 100
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
print 'Making tensor'
n_images = len(test_images)
im_shape = test_images[0].shape      
# im_shape = (64, 64, 3)
assert n_images==len(test_images), "Number of images doesn't match number of files!"
tensor = np.zeros(shape=(n_images,)+im_shape,dtype=np.float32)

for i in range(n_images):
    tensor[i] = test_images[i]

# Reshape to fit Theanos format
tensor = tensor.reshape(n_images,3,im_shape[0],im_shape[1])

# Clean up
del photo_biz_ids_df, test_images, im_files

#%% Final processing and setup
if im_mean is None:
    im_mean = tensor.mean()
tensor -= im_mean       # Subtract the mean


#%% Load model from JSON and weights from HDF5:
model = model_from_json(open(models_dir+model_name+'.json').read())
model.load_weights(models_dir+model_name+'.h5')

#%% Predict
print 'Generating predictions'
# Threshold at 0.5 and convert to 0 or 1
# model.predict_classes() -- use with class_mode = 'binary'
X_test_prediction = (model.predict(tensor) > .5)*1


#%% Compile a Test DataFrame
test_df = generate_test_df(test_df, ['photo_id', 'business_id'],
                           X_test_prediction, np.arange(n_images))

#%% Generate a submission
test_df[['business_id', 'predicted_labels']].to_csv(submission_file_name,
                                                    index=False,
                                                    header=['business_id','labels'],
                                                    compression='gzip')
