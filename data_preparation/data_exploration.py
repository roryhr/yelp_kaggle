import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import numpy as np
import pandas as pd
import glob
import re

import tarfile

##%% 
#img = mpimg.imread('train_photos/5.jpg')
#plt.imshow(img)

#%% 

df = pd.read_csv('train_photo_to_biz_ids.csv')




#%% Read in the files an
im_files = glob.glob("/home/rory/kaggle/yelp/train_photos/*.jpg")
big_list = []
p = re.compile('\d+')

for i in range(30):
    a = p.search(im_files[i])
    big_list.append([mpimg.imread(im_files[i]), im_files[i], 
                   int(im_files[i][a.start():a.end()])])
    
    
