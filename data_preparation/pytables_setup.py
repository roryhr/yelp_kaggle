import cPickle as pickle
import tables

#%% Configuration
hdf5_file_name = "data/all_photos.hdf5"
test_photos_file = 'data/all_test_photos'
train_photos_file = 'data/all_train_photos'

#%% Load in test images from pickle files
with open(test_photos_file+'_images'+'.pkl', 'rb') as in_file:
   test_images = pickle.load(in_file)

with open(test_photos_file+'_im_files'+'.pkl', 'rb') as in_file:
   im_files = pickle.load(in_file)


#%% Define Images table 
class Images(tables.IsDescription):
    image = tables.UInt8Col(shape=(64,64,3))      # Unsigned 8 byte integers
    file_name = tables.StringCol(50)               # Unsigned 8 byte integers

#%%

h5file = tables.open_file(hdf5_file_name, 
                          mode = "w", 
                          title = "Yelp Photos")

group = h5file.create_group('/', 'test_images', 'Test Images')
table = h5file.create_table(group, 'images', Images, 'Test photo and filename')

#%%

image_row = table.row
for image, file_name in zip(test_images, im_files):
    image_row['image']  = image
    image_row['file_name'] = file_name
    image_row.append()
        
table.flush()


#%% Load in test images from pickle files
with open(train_photos_file+'_images'+'.pkl', 'rb') as in_file:
   train_images = pickle.load(in_file)

with open(train_photos_file+'_im_files'+'.pkl', 'rb') as in_file:
   train_im_files = pickle.load(in_file)

#%%
group = h5file.create_group('/', 'train_images', 'Train Images')
table = h5file.create_table(group, 'images', Images, 'Train photo and filename')

image_row = table.row
for image, file_name in zip(train_images, train_im_files):
    image_row['image']  = image
    image_row['file_name'] = file_name
    image_row.append()
        
table.flush()


h5file.close()