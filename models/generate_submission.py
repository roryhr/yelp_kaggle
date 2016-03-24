import numpy as np
import pandas as pd
from keras.models import model_from_json
# from helper_functions import generate_test_df
import tables


def prepare_tensor(test_images):
    print 'Making tensor'
    n_images = len(test_images)
    im_shape = test_images[0].shape      
    # im_shape = (64, 64, 3)
    
    test_images = test_images.astype('float32')
      
    test_images -= im_mean  
    
    # Reshape to fit Theanos format
    return test_images.reshape(n_images,3,im_shape[0],im_shape[1])
    
def generate_labels_df(all_predictions):
    df = pd.DataFrame(all_predictions)
    df2 = pd.DataFrame(index=df.index)
    df2['predicted_labels'] = ''
    
    for i in range(8):
        df2[df[i]==1] = df2[df[i]==1] + str(i) + ' '
        
    return df2

#%% Configuration
model_name = 'mar_7_0005'
csv_dir = 'data/'                           # Folder for csv files
models_dir = 'models/'
#save_file_name = 'data/all_test_photos'
submission_file_name = 'submissions/submission_mar_7.csv.gz'
split_point = 100000   # Split test_images into 2 batches, roughly in half
#im_mean = 106.7203
#im_mean = 106.74557           # mar_6_1136
im_mean = 106.81699           # mar_7_0005, trained on 0-1

#%% Load model from JSON and weights from HDF5:
model = model_from_json(open(models_dir+model_name+'.json').read())
model.load_weights(models_dir+model_name+'.h5')

#%% Load in pointers to the test photos hdf5 file
h5file = tables.open_file('data/photos.hdf5')
im_table = h5file.root.test_images.images

#%% Make a tensor
test_images = im_table.read(start=0, stop=split_point, field='image')
test_images = prepare_tensor(test_images)

#%% Predict
# Threshold at 0.5 and convert to 0 or 1
predictions = (model.predict(test_images, batch_size=1000, verbose=1) > .5)*1

#%% Make another tensor on a second batch of images
test_images = im_table.read(start=split_point, field='image')
test_images = prepare_tensor(test_images)

#%% Predict again
predictions2 = (model.predict(test_images, batch_size=10000, verbose=1) > .5)*1
#%%
total_predictions = np.append(predictions, predictions2, axis=0)

#%% Read and join biz_ids on photo_id
im_files = im_table.read(field='file_name')
test_df = pd.DataFrame(im_files, columns=['filepath'])

#Extract the photo id from the file name 
test_df['photo_id'] = test_df.filepath.str.extract('(\d+)')
test_df.photo_id = test_df.photo_id.astype('int')

#%%

test_df['labels'] = generate_labels_df(total_predictions)

# Column names: photo_id, business_id
photo_biz_ids_df = pd.read_csv(csv_dir+'test_photo_to_biz.csv')

test_df = pd.merge(test_df, photo_biz_ids_df, on='photo_id')
submission_df = test_df.drop_duplicates('business_id')


#%% Generate a submission
submission_df[['business_id', 'labels']].to_csv(submission_file_name,
                                                index=False,
                                                header=['business_id','labels'],
                                                compression='gzip')