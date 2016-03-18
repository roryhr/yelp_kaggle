from convolutional_models.keras_model import KerasGraphModel


class Model(KerasGraphModel):
    def __init__(self, nb_epochs=10, mini_batch_size=100, weight_decay=0.0001):
        self.nb_epochs = nb_epochs
        self.weight_decay = weight_decay
        self.mini_batch_size = mini_batch_size
        self.imsize = None
        self.nb_images = None
        self.train_images = None
        self.dim_order = 'th'

    def fit(self, train_images, target):
        self.train_images = train_images
        self.nb_images = len(train_images)
        self.imsize = train_images[0].shape[:2]

    def evaluate(self):
        pass

    def generate_submission(self):
        pass

    def score(self):
        pass
