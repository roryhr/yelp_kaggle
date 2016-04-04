from pathlib import Path
from data_preparation.image_preparation import ImageLoader
from keras_models import KerasGraphModel

im_files = list(Path('data/train_photos').glob('*[0-9].jpg'))

# No validation data for now
image_loader = ImageLoader()
train_im_func = image_loader.graph_train_generator(im_files)
# test_images = next(train_im_func)

model = KerasGraphModel()
model.load_graph('small_test_model')

# Fit on 30 mini-batches of 200 samples for 3 epochs
model.graph.fit_generator(train_im_func, 200*30, 3)
