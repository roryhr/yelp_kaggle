from itertools import zip_longest
import pandas as pd
from pathlib import Path
# import pickle
import numpy as np
import random
from scipy import misc

CROP_SIZE = (224, 224)   # Size of final image passed into convolution network
#BATCH_SIZE = 1000        # Number of images to process
LABELS_CACHE = 'data/train_labels.pkl'


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


class ImageLoader(object):
    """ A few utilities to work with csv files and images."""
    def __init__(self,
                 save_file_name='data/all_train_photos',
                 csv_dir='data/',
                 photo_dir='data/train_photos/train_photos/',
                 random_horizontal_flip=False,
                 random_vertical_flip=False,
                 batch_size=30,
                 im_mean=106.7):
        self.save_file_name = save_file_name
        self.csv_dir = csv_dir
        self.photo_dir = photo_dir
        
#        self.batch_size = batch_size
        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        self.im_mean = im_mean
        self.train_df = self.compute_target_labels()

    def compute_target_labels(self):
        """ Compute the target labels and save to a pickle file

        photo_ids = ['1', '163', ...]
        """

        # Load the pickled train_df if it exists
        if Path(LABELS_CACHE).exists():
            print('Loaded training labels from cache')
            return pd.read_pickle(LABELS_CACHE)

        p = Path('../data/train_photos')
        photo_ids = list(p.glob('*[0-9].jpg'))
        # ['../data/train_photos/57459.jpg', '../data/train_photos/227860.jpg', ...]
        photo_ids = [int(path_id.stem) for path_id in photo_ids]
        # [57459, 227860, 302452, ...]

        train_df = pd.DataFrame(photo_ids, columns=['photo_id'])

        # %% Read and join biz_ids on photo_id
        photo_biz_ids_df = pd.read_csv(self.csv_dir + 'train_photo_to_biz_ids.csv')
        # Column names: photo_id, business_id

        train_df = pd.merge(train_df, photo_biz_ids_df, on='photo_id')

        # Read and join train labels, set to 0 or 1
        train_labels_df = pd.read_csv(self.csv_dir + 'train.csv')
        # Column names: business_id, labels

        # Work column-wise to encode the labels string into 9 new columns
        for i in '012345678':
            train_labels_df[i] = train_labels_df['labels'].str.contains(i) * 1

        train_labels_df = train_labels_df.fillna(0)

        train_df = pd.merge(train_df, train_labels_df, on='business_id')

        train_df.to_pickle(LABELS_CACHE)

        # Convert label columns, 3-12, to integers
        # train_df[train_df.columns[3:]] = train_df[train_df.columns[3:]].astype('int')
        return train_df

    @staticmethod
    def resnet_image_processing(file_path):
        """ Apply image processing as in He et al:
        Scale augmentation, per-pixel mean subtraction, horizontal flips,
        and color augmentation

        He et al. Deep Residual Learning for Image Recognition. arXiv, 2015.
        http://arxiv.org/abs/1512.03385

        K. Simonyan and A. Zisserman. Very deep convolutional networks
        for large-scale image recognition. In ICLR, 2015.

        A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification
        with deep convolutional neural networks. In NIPS, 2012.
        """

        img = misc.imread(file_path)
        (height, width, channels) = img.shape

        # Resize the image with so that shorter side is between 256-480
        scale = float(random.sample(range(256, 480), 1)[0]) / min(height, width)
        img = misc.imresize(img, size=scale)

        (height, width, channels) = img.shape

        # Take a CROP_SIZE = 224x224 randomly selected crop
        row = random.sample(range(height - CROP_SIZE[0]), 1)[0]
        col = random.sample(range(width - CROP_SIZE[1]), 1)[0]

        crop_img = img[row:row+CROP_SIZE[0], col:col+CROP_SIZE[1], :]
        return crop_img

    def horizontal_flip(self, value):
        pass

    def graph_train_generator(self, im_files, batch_size=30):
        """Read in the images and yield a training dict."""
        while True:
            im_selection = random.sample(im_files, batch_size)

            # Load images into a list
            images_tensor = [self.resnet_image_processing(im_file) for im_file in im_selection]

            # Convert list into a tensor
            images_tensor = np.stack(images_tensor)

            # Reshape tensor into Theano format
            images_tensor = images_tensor.reshape(batch_size, 3, CROP_SIZE[0], CROP_SIZE[1])
            images_tensor = images_tensor.astype('float32')
            images_tensor -= self.im_mean

            yield {'input': images_tensor, 'output': self.get_target_labels(im_selection)}

    def test_image_generator(self, im_files, batch_size=30):
        """Read in the test images and yield a test dict."""
        for im_selection in grouper(im_files, batch_size):
            # Load images into a list
            images_tensor = [self.resnet_image_processing(im_file) for im_file in im_selection]
            photo_ids = [int(path_id.stem) for path_id in im_selection]

            # Convert list into a tensor
            images_tensor = np.stack(images_tensor)

            # Reshape tensor into Theano format
            images_tensor = images_tensor.reshape(batch_size, 3, CROP_SIZE[0], CROP_SIZE[1])
            images_tensor = images_tensor.astype('float32')
            images_tensor -= self.im_mean

            yield {'input': images_tensor}, photo_ids

    def get_target_labels(self, image_ids):
        """ Return target labels from a pickled DataFrame

        Input:
        image_ids: list of Paths, list
                   [PosixPath('../data/train_photos/20548.jpg'),...]

        Output:
        train labels, numpy.ndarray
        array([[1, 0, 0, 1, 0, 0, 0, 0, 1],
               [1, 0, 0, 1, 0, 0, 0, 0, 1],
               ...                       ])
        """

        # Convert to list of integer photo_ids
        image_ids = [int(l.stem) for l in image_ids]

        labels = self.train_df[self.train_df.photo_id.isin(image_ids)]

        # TODO: Check order of values!
        return labels.iloc[:, 3:].values


if __name__ == '__main__':
    # p = Path('../data/train_photos')
    # im_files = list(p.glob('*[0-9].jpg'))
    # # misc.imshow(misc.imread(im_files[0]))
    #
    image_loader = ImageLoader()
    image_loader.compute_target_labels()
    # x = image_loader.get_target_labels([57459, 207144, 51954])

    im_files = list(Path('../data/train_photos').glob('*[0-9].jpg'))
    images = image_loader.graph_train_generator(im_files)

    test = next(images)

    print(test)