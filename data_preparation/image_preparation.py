import pickle
import glob
import numpy as np
import random
from scipy import misc

CROP_SIZE = (224, 224)   # Size of final image passed into convolution network
BATCH_SIZE = 100         # Number of images to process


class ImageLoader(object):
    def __init__(self,
                 save_file_name='data/all_train_photos',
                 photo_dir='data/train_photos/train_photos/',
                 random_horizontal_flip=False,
                 random_vertical_flip=False,
                 im_mean=None):
        self.save_file_name = save_file_name
        self.photo_dir = photo_dir

        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        self.im_mean = im_mean

    @staticmethod
    def resnet_image_processing(file_path):
        """ Apply image processing from MSFT's ResNet

        http://arxiv.org/abs/1512.03385
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
        return self.random_horizontal_flip

    def load_images(self, im_files):
        """ Read in the images."""
        # im_files = glob.glob(test_jpg_dir + '*.jpg')
        # if n_images:
        #     im_selection = im_files[:n_images]
        # else:
        #     im_selection = im_files
        #     n_images = len(im_selection)
        im_selection = random.sample(im_files, BATCH_SIZE)

        images = [self.resnet_image_processing(im_file) for im_file in im_selection]

        return images

    def yield_images(self, im_files):
        """ Read in the images and yield."""
        while True:
            im_selection = random.sample(im_files, BATCH_SIZE)

            # Load images into a list
            images_tensor = [self.resnet_image_processing(im_file) for im_file in im_selection]

            # Convert list into a tensor
            images_tensor = np.stack(images)

            # Reshape tensor into theano format
            images_tensor = images_tensor.reshape(BATCH_SIZE, 3, CROP_SIZE[0], CROP_SIZE[1])
            images_tensor = images_tensor.astype('float32')
            images_tensor -= self.im_mean

            yield images_tensor

    def save_images_to_pickle(self):
        #%% Save the images and file names
        with open(save_file_name+'_images' + '.pkl', 'wb') as out_file:
           pickle.dump(test_images, out_file, 2)

        with open(save_file_name+'_im_files'+'.pkl', 'wb') as out_file:
           pickle.dump(im_selection, out_file, 2)

if __name__ == '__main__':
    from pathlib import Path
    from scipy import misc
    p = Path('../data/train_photos')
    im_files = list(p.glob('*[0-9].jpg'))
    misc.imshow(misc.imread(im_files[0]))
    # TODO: get files from im_files[0].stem = '57459' -> integers

    image_loader = ImageLoader()
    images = image_loader.load_images(im_files)