import pandas as pd
#from PIL import Image


def generate_test_df(train_df, cols, X_test_prediction, X_test_ind):
    '''Helper function to make a test data frame with predicted and actual
    labels
    '''
    
    test_df = train_df[cols].iloc[X_test_ind]

    #New columns are integers 0-8
    test_df = test_df.join(pd.DataFrame(data=X_test_prediction, index=X_test_ind))

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
    
    
  

def preprocess_image(im,width=100,height=100):
    ''' INPUT: PIL Image. OUTPUT: RGB PIL Image
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
    
    return im.resize((width,height))
    
    

def mean_f1_score(Y_pred, Y_true):
    '''Y_pred int64 array size (n_samples, 9)'''
    
    tp = (((Y_pred == Y_true)) * (Y_pred == 1)).sum() # True positives
    fp = (((Y_pred != Y_true)) * (Y_pred == 1)).sum() # False positives
    # tp + fp == Y_pred.sum()
    
    fn = (((Y_pred != Y_true)) * (Y_pred == 0)).sum() 
    
    p = tp*1.0/(tp + fp)
    r = tp*1.0/(tp + fn)
    
    return 2.0*p*r/(p + r)