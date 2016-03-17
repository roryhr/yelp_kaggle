class Model:

    def __init__(self, nb_epochs=10, nb_images=1000, imsize=(224,224)):
        self.nb_epochs = nb_epochs
        self.nb_images = nb_images
        self.imsize = imsize
        self.save_model = False
        self.show_plots = True
        self.model_name = 'mar_7_0005'
        self.csv_dir = 'data/'                        # Folder for csv files
        # self.reload_images = False
        self.jpg_dir = 'data/train_photos/'
        self.models_dir = 'models/'
        self.weight_decay = 0.01

    def train(self):
        pass

    def test(self):
        pass

    def generate_submission(self):
        pass