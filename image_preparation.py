#import glob
# from joblib import Parallel, delayed
from helper_functions import load_and_preprocess
import cPickle as pickle
import time

#----------------Configuration----------------------------
# n_images = 4000
n_images = None     # Read and process all test images
imsize   = 64       # Square images
#n_jobs = 3
test_jpg_dir = 'data/test_photos/'

save_file_name = 'data/all_test_photos'
#---------------------------------------------------------


#im_files = glob.glob(test_jpg_dir + '*.jpg')


#%% Read in the images
if n_images:
    im_selection = im_files[:n_images]
else:
    im_selection = im_files
    n_images = len(im_selection)

print 'Read and preprocessing {} images'.format(n_images)
print "Done in about %.0f minutes" % (n_images*0.0072/60)  # 7.2 ms/image
#print 'n_jobs = {}'.format(n_jobs)
start_time = time.time()

#test_images = Parallel(n_jobs=n_jobs)(delayed(load_and_preprocess)(im_file) for im_file in im_selection)
test_images = [load_and_preprocess(im_file) for im_file in im_selection]

elapsed_time = time.time() - start_time
print "Took %.1f minutes and %.1f ms per image" % (elapsed_time/60,
                                                   1000*elapsed_time/n_images)

#%% Save the images and file names
with open(save_file_name+'_images' + '.pkl', 'wb') as out_file:
   pickle.dump(test_images, out_file, 2)

with open(save_file_name+'_im_files'+'.pkl', 'wb') as out_file:
   pickle.dump(im_selection, out_file, 2)
