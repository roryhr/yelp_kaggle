from pathlib import Path
from data_preparation.image_preparation import ImageLoader
from keras_models import KerasGraphModel

im_files = list(Path('data/train_photos').glob('*[0-9].jpg'))

# No validation data for now
image_loader = ImageLoader()
train_im_func = image_loader.graph_train_generator(im_files, batch_size=100)
# test_images = next(train_im_func)

model = KerasGraphModel()
#model.build_residual_network(nb_blocks=[1, 2, 2, 2, 2], initial_nb_filters=4)
model.load_graph('18_layer_4_filters')

# Fit on 30 mini-batches of 200 samples for 3 epoch
h = model.graph.fit_generator(train_im_func, 200*20, 3)
# test_data, test_photo_ids = next(test_data_generator)

# test_labels = model.graph.predict(test_data)

model.save_model(model.graph, model_stem='18_layer_4_filters', overwrite=True)


test_data_generator = image_loader.test_image_generator(im_files, batch_size=2000)
test = model.generate_submission(test_data_generator)