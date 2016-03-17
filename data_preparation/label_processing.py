def lr_schedule(epoch):
    if epoch < 6:
        lr = 0.1
    elif epoch < 10:
        lr = 0.01
    else:
        lr = 0.001
    return lr

#%% Read in the images
print 'Read and preprocessing {} images'.format(n_images)
start_time = time.time()

if process_images:
    im_files = glob.glob(jpg_dir + '*.jpg')
    im_files = random.sample(im_files, n_images)
    # Might as well forget other files for now

    train_images = []
    train_images = Parallel(n_jobs=3)(delayed(resnet_image_processing)(im_file) for im_file in im_files)

    with open(photo_cache, 'wb') as out_file:
        pickle.dump((im_files, train_images), out_file, 2)
else:
    with open(photo_cache, 'rb') as in_file:
        im_files, train_images = pickle.load(in_file)


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


tensor = np.stack(train_images)
"""Reshape to fit Theanos format
dim_ordering='th'
(samples, channels, rows, columns)
vs
dim_ordering='tf'
(samples, rows, cols, channels)
"""
tensor = tensor.reshape(n_images,3,imsize,imsize)
tensor = tensor.astype('float32')

#%% Clean up and save memory
del train_labels_df, photo_biz_ids_df, i, train_images

label_start = 4  # Column number where labels in train_df start

#%% Final processing and setup
im_mean = tensor.mean()
tensor -= im_mean
# Subtract the mean for faster convergence
print 'Mean for all images: {}'.format(im_mean)
#%%