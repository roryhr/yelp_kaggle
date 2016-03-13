#import matplotlib.image as mpimg
import numpy as np      # 1.10.1
import pandas as pd
import glob
import random
import time
from sklearn.cross_validation import train_test_split

from keras.callbacks import EarlyStopping, TensorBoard
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.models import Graph
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.optimizers import SGD

from helper_functions import mean_f1_score, resnet_image_processing
from helper_functions import show_image_labels

from joblib import Parallel, delayed
import tables


#%% Configuration 
number_of_epochs = 20
n_images = 1000
imsize   = 224  # Square images
save_model = False
show_plots = True
model_name = 'mar_7_0005'
csv_dir = 'data/'                        # Folder for csv files    
reload_images = True
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
    train_images = Parallel(n_jobs=3)(delayed(resnet_image_processing)(im_file) for im_file in im_files)
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

#train_ind, test_ind, _, _ = train_test_split(range(n_images),
#                                             range(n_images),
#                                             test_size=0.1, 
#                                             random_state=4)

print 'Mean for all images: {}'.format(im_mean)
#%%

def building_block(graph, layer_nb, conv_nb, input_name, nb_filters):
    """ Add a Residual building block
    

    """
#    First convolution 
    first_convolution = 'conv{}_{}'.format(layer_nb,conv_nb)
    first_normalization = 'bn_{}_{}'.format(layer_nb,conv_nb)
    first_activation = 'relu{}_{}'.format(layer_nb, conv_nb)

#    Second convolution 
    second_convolution = 'conv{}_{}'.format(layer_nb,conv_nb+1)
    second_normalization = 'bn_{}_{}'.format(layer_nb,conv_nb+1)
    second_activation = 'relu{}_{}'.format(layer_nb, conv_nb+1)
    
#    return [first_convolution, first_normalization, first_activation,
#            second_convolution, second_normalization, second_activation]
    
    graph.add_node(Convolution2D(nb_filters, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name=first_convolution, input=input_name)
    graph.add_node(BatchNormalization(), name=first_normalization,
                   input=first_convolution)
    graph.add_node(Activation('relu'), name=first_activation,
                   input=first_normalization)
    
#    Second Convolution
    graph.add_node(Convolution2D(nb_filters, 3, 3, W_regularizer=l2(weight_decay), 
                                 border_mode='same'),
                   name=second_convolution, input=first_activation)
    graph.add_node(BatchNormalization(), name=second_normalization,
                   input=second_convolution)
    graph.add_node(Activation('relu'), name=second_activation,
                   inputs=[input_name,second_normalization], 
                   merge_mode='sum')
    
    return graph, second_activation



#%% ResNet-like 
"""34-layer ResNet with skip connections 

    http://arxiv.org/abs/1512.03385
"""
print 'Building model'
nb_filters = 8

graph = Graph()
graph.add_input(name='input', input_shape=(3,imsize,imsize))   
graph.add_node(Convolution2D(nb_filter=nb_filters, nb_row=7, nb_col=7,
                             input_shape=(3,imsize,imsize),
                             border_mode='same',
                             subsample=(2,2),
                             dim_ordering='th',
                             W_regularizer=l2(weight_decay)), 
                             name='conv1', input='input')
# Output shape = (None,16,112,112)
graph.add_node(BatchNormalization(), name='bn1', input='conv1')
graph.add_node(Activation('relu'), name='relu1', input='bn1')
graph.add_node(MaxPooling2D(pool_size=(3,3), strides=(2,2),
                            border_mode='same'), 
               name='pool1', input='relu1')
# Output shape = (None,32,56,56)

#-------------------------- Layer Group 2 -------------------------------------
graph, output_name = building_block(graph, layer_nb=2, conv_nb=1,
                                    input_name='pool1', nb_filters=nb_filters)
graph, output_name = building_block(graph, layer_nb=2, conv_nb=3,
                                    input_name=output_name, nb_filters=nb_filters)
graph, output_name = building_block(graph, layer_nb=2, conv_nb=5,
                                    input_name=output_name, nb_filters=nb_filters)
# Output shape = (None,16,56,56)

#-------------------------- Layer Group 3 -------------------------------------
graph.add_node(Convolution2D(nb_filters*2, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same', subsample=(2,2)),
               name='conv3_1', input=output_name)
# Output shape = (None,32,28,28)
graph.add_node(BatchNormalization(), name='bn3_1', input='conv3_1')
graph.add_node(Activation('relu'), name='relu3_1', input='bn3_1')
graph.add_node(Convolution2D(nb_filters*2, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv3_2', input='relu3_1')
graph.add_node(BatchNormalization(), name='bn3_2', input='conv3_2')

graph.add_node(Convolution2D(nb_filters/2,1,1, W_regularizer=l2(weight_decay)),
               name='short3_1', input='relu2_2')
graph.add_node(BatchNormalization(), name='short3_2', input='short3_1')
# Output tensor shape = (32,56,56)
graph.add_node(Reshape((nb_filters*2,28,28)), name='short3_3', input='short3_2')               
graph.add_node(Activation('relu'), name='relu3_2', inputs=['short3_3','bn3_2'], 
               merge_mode='sum')
# Output shape = (None,64,28,28)

#nb_filters *= 2
graph, output_name = building_block(graph, layer_nb=3, conv_nb=3,
                                    input_name='relu3_2', 
                                    nb_filters=nb_filters*2)
graph, output_name = building_block(graph, layer_nb=3, conv_nb=5,
                                    input_name=output_name, 
                                    nb_filters=nb_filters*2)
graph, output_name = building_block(graph, layer_nb=3, conv_nb=7,
                                    input_name=output_name, 
                                    nb_filters=nb_filters*2)        

#-------------------------- Layer Group 4 -------------------------------------
graph.add_node(Convolution2D(nb_filters*4, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same', subsample=(2,2)),
               name='conv4_1', input=output_name)
graph.add_node(BatchNormalization(), name='bn4_1', input='conv4_1')
graph.add_node(Activation('relu'), name='relu4_1', input='bn4_1')
graph.add_node(Convolution2D(nb_filters*4, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv4_2', input='relu4_1')
graph.add_node(BatchNormalization(), name='bn4_2', input='conv4_2')
# Output shape = (256,14,14)

graph.add_node(Convolution2D(nb_filters,1,1, W_regularizer=l2(weight_decay)),
               name='short4_1', input='relu3_2')
graph.add_node(BatchNormalization(), name='short4_2', input='short4_1')
# Output shape = (64,28,28)
graph.add_node(Reshape((nb_filters*4,14,14)), name='short4_3', input='short4_2')               
graph.add_node(Activation('relu'), name='relu4_2', inputs=['short4_3','bn4_2'], 
               merge_mode='sum')
# Output shape = (256,14,14)

graph, output_name = building_block(graph, layer_nb=4, conv_nb=3,
                                    input_name='relu4_2', 
                                    nb_filters=nb_filters*4)
graph, output_name = building_block(graph, layer_nb=4, conv_nb=5,
                                    input_name=output_name, 
                                    nb_filters=nb_filters*4)
graph, output_name = building_block(graph, layer_nb=4, conv_nb=7,
                                    input_name=output_name, 
                                    nb_filters=nb_filters*4)        
graph, output_name = building_block(graph, layer_nb=4, conv_nb=9,
                                    input_name=output_name, 
                                    nb_filters=nb_filters*4)   
graph, output_name = building_block(graph, layer_nb=4, conv_nb=11,
                                    input_name=output_name, 
                                    nb_filters=nb_filters*4)   
# Output shape = (256,14,14)


#-------------------------- Layer Group 5 -------------------------------------
graph.add_node(Convolution2D(nb_filters*8, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same', subsample=(2,2)),
               name='conv5_1', input=output_name)
graph.add_node(BatchNormalization(), name='bn5_1', input='conv5_1')
graph.add_node(Activation('relu'), name='relu5_1', input='bn5_1')
graph.add_node(Convolution2D(nb_filters*8, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name='conv5_2', input='relu5_1')
graph.add_node(BatchNormalization(), name='bn5_2', input='conv5_2')
# Output shape = (256,14,14)

graph.add_node(Convolution2D(nb_filters*2,1,1, W_regularizer=l2(weight_decay)),
               name='short5_1', input='relu4_2')
graph.add_node(BatchNormalization(), name='short5_2', input='short5_1')
# Output shape = (64,28,28)
graph.add_node(Reshape((nb_filters*8,7,7)), name='short5_3', input='short5_2')               
graph.add_node(Activation('relu'), name='relu5_2', inputs=['short5_3','bn5_2'], 
               merge_mode='sum')
# Output shape = (512,7,7)
               
graph, output_name = building_block(graph, layer_nb=5, conv_nb=3,
                                    input_name='relu5_2', 
                                    nb_filters=nb_filters*8)
graph, output_name = building_block(graph, layer_nb=5, conv_nb=5,
                                    input_name=output_name, 
                                    nb_filters=nb_filters*8)
# Output shape = (128,7,7)
                      
graph.add_node(AveragePooling2D(pool_size=(3,3), strides=(2,2),
                                border_mode='same'),
               name='pool2', input=output_name)               
graph.add_node(Flatten(), name='flatten', input='pool2')
graph.add_node(Dense(9, activation='sigmoid'), name='dense', input='flatten')
graph.add_output(name='output', input='dense')
               
               
sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9)
graph.compile(optimizer=sgd, loss={'output':'binary_crossentropy'})


#%% Fit model
graph.fit({'input': tensor, 'output': train_df.iloc[:,label_start:].values}, 
          batch_size=64, nb_epoch=number_of_epochs, validation_split=0.1,
#          validation_data={'input': tensor[test_ind],
#                            'output': train_df.iloc[test_ind,label_start:].values},
          shuffle=True,
          callbacks=[TensorBoard('/home/rory/logs')],
          verbose=1)

#%% Compute mean_f1_score
''' Calculate the Mean F1 Score

                        Mean F1 Score
    Sample submission   0.36633 	
    Random submission   0.43468
    Benchmark           0.64590 	
    Leader (1/17)       0.81090
'''
X_val, y_val, _ = graph.validation_data
# Threshold at 0.5 and convert to 0 or 1 
predictions = (graph.predict({'input':X_val})['output'] > .5)*1

print 'Mean F1 Score: %.2f' % mean_f1_score(predictions,
                                            y_val)
    
    
#%% Save model as JSON
if save_model:
    json_string = model.to_json()
    open(models_dir + model_name + '.json', 'w').write(json_string)
    model.save_weights(models_dir + model_name + '.h5')  # requires h5py
    
#%% Plot a few images to get a feel of how I did
if show_plots:
    for i in range(10):
        show_image_labels(tensor[test_ind[i]], predictions[i], 
                          train_df['labels'][test_ind[i]], im_mean)

