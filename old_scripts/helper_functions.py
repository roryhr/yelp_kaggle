import pandas as pd

from scipy import misc
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed


def generate_test_df(train_df, cols, X_test_prediction, X_test_ind):
    '''Helper function to make a test data frame with predicted and actual
    labels
    '''
    
    test_df = train_df[cols].iloc[X_test_ind]

    #New columns are integers 0-8
    test_df = test_df.join(pd.DataFrame(data=X_test_prediction,
                                        index=X_test_ind))

# TODO: Choose unique business IDs in a smart way --- average predictions? 

    # Make a dummy DataFrame because setting a column value didn't work
    df = pd.DataFrame(index = X_test_ind)
    df['predicted_labels']  = ''

    # Work column-wise to make a predicted labels string into 9 new columns
    for i in range(8):
        df[test_df[i]==1] = df[test_df[i]==1] + str(i) + ' '
    
    
    test_df = test_df.join(df)
    cols.append('predicted_labels')
    return test_df[cols]
        
    
def load_and_preprocess(file_path):
    ''' INPUT:  Image file path (ex 'data/train_photos/54500.jpg')
        OUTPUT: Rescaled and cropped? numpy array
    '''
    b = misc.imread(file_path)
    if b.shape[2] != 3:
        print "Doesn't have 3 layers, ignoring image"  
        return
    
    return preprocess_image(b)

def load_image(file_path):
    ''' INPUT:  Image file path (ex 'data/train_photos/54500.jpg')
        OUTPUT: Rescaled and cropped? numpy array
    '''
    b = misc.imread(file_path)
    if b.shape[2] != 3:
        print "Doesn't have 3 layers, ignoring image"  
        return
    
    return b


def resnet_image_processing(file_path):
    """Apply image processing from MSFT's ResNet
    
    http://arxiv.org/abs/1512.03385
    """
    
    img = misc.imread(file_path)
    (height, width, channels) = img.shape
    
    # Resize the image with so that shorter side is between 256-480
    scale = float(random.sample(xrange(256,480),1)[0])/min(height, width)
    img = misc.imresize(img, size=scale)

    (height, width, channels) = img.shape
    # Take a 224x224 randomly selected crop
    
#    Randomly take a crop from valid choices
    row = random.sample(xrange(height-224),1)[0]
    col = random.sample(xrange(width-224),1)[0]
    
    crop_img = img[row:row+224,col:col+224,:]
    
    return crop_img
    
def preprocess_image(im,width=64,height=64):
    ''' INPUT: numpy.ndarray
        OUTPUT: Rescaled image and crop
    '''
    
#    im.thumbnail((300,300))   # Resize inplace to fit in 300x300
    
    
#    if im.size < (width, height):
#        print 'Image size is smaller than width and height'
#        return im.crop((0,0,width,height))
#            
#            
#    # Random crop
#    x_left = np.random.randint(0, im.size[0]-width)
#    y_top = np.random.randint(0, im.size[1]-height)
#    
#    return im.crop((x_left,y_top,x_left+width,y_top+height))
    
    return misc.imresize(im, (width,height))
    
    

def mean_f1_score(Y_pred, Y_true):
    '''Y_pred int64 array size (n_samples, 9)'''
    
    tp = (((Y_pred == Y_true)) * (Y_pred == 1)).sum() # True positives
    fp = (((Y_pred != Y_true)) * (Y_pred == 1)).sum() # False positives
    # tp + fp == Y_pred.sum()
    
    fn = (((Y_pred != Y_true)) * (Y_pred == 0)).sum() 
    
    p = tp*1.0/(tp + fp)
    r = tp*1.0/(tp + fn)
    
    return 2.0*p*r/(p + r)
    
    
#%% Compare some predicted labels with true labels
    
def show_image_labels(im_slice, predicted_labels_encoded, true_labels, im_mean=0):
    """Plot the test image along with the true and predicted labels 
    
    Usage:
    show_image_labels(tensor[test_ind[0]], X_test_prediction[test_ind[0]], train_df['labels'][test_ind[0]], im_mean)
    
    predicted_labels_encoded = array([0, 1, 1, 1, 0, 1, 1, 0, 0])
    true_labels = '1 2 4 5 6 7'
    im_mean = 108.4926
    """
    
    if im_slice.shape[0] == 3:
        im_shape = im_slice.shape
        
        im_slice = im_slice.reshape(im_shape[1], im_shape[2], 3)
        
    im_slice += im_mean
    
#    im_slice = im_slice.astype('uint8')
    
    # generate a list of indicies which are 1 instead of 0
    predicted_labels = [str(ind) for (ind, val) in enumerate(predicted_labels_encoded) if val]
    
    predicted_labels = ' '.join(predicted_labels)
    
    
    plt.imshow(misc.toimage(im_slice))
    plt.axis('off')  # clear x- and y-axes
    plt.text(1, -6, 'Pred labels: ' + predicted_labels, fontsize=12)   
    plt.text(1, -16, 'True labels: ' + true_labels, fontsize=12)    
    plt.show()    
