from pathlib import Path

from data_preparation.image_preparation import ImageLoader
from keras_models import KerasGraphModel

image_loader = ImageLoader()
image_loader.compute_target_labels()

im_files = list(Path('../data/train_photos').glob('*[0-9].jpg'))
images = image_loader.graph_train_generator(im_files)

model = KerasGraphModel()
model.build_residual_network(nb_blocks=[1, 2, 2, 2, 2])
model.graph.summary()

model.save('small_test_model')

model2 = KerasGraphModel()
model2.load_model('small_test_model')