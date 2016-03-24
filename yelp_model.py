from models.keras_models import KerasGraphModel


model = KerasGraphModel()
    def __init__(self, nb_epochs=10, mini_batch_size=100, weight_decay=0.0001):
        self.nb_epochs = nb_epochs
        self.weight_decay = weight_decay
        self.mini_batch_size = mini_batch_size
        self.imsize = None
        self.nb_images = None
        self.train_images = None
        self.dim_order = 'th'