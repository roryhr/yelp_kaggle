from keras.callbacks import TensorBoard, LearningRateScheduler
from keras.layers.convolutional import AveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers.normalization import BatchNormalization
from keras.models import Graph
from keras.models import model_from_json

from keras.optimizers import SGD
from keras.regularizers import l2

class BaseKerasModel(object):
    def __init__(self, nb_epochs=10, mini_batch_size=100):
        self.nb_epochs = nb_epochs
        self.mini_batch_size = mini_batch_size

    def save_model(self):
        pass

    def evaluate(self, X_val, y_val):
        """Calculate the Mean F1 Score

        :param X_val: training hold out data
        :param y_val: target from hold out data

                            Mean F1 Score
        Sample submission   0.36633
        Random submission   0.43468
        Benchmark           0.64590
        Leader (1/17)       0.81090
        """
        # Threshold at 0.5 and convert to 0 or 1
        predictions = (self.predict({'input':X_val})['output'] > .5)*1

    def load_model(self):
        """Load model from JSON and weights from HDF5"""
        self.model = model_from_json(open(models_dir+model_name+'.json').read())
        self.model.load_weights(models_dir+model_name+'.h5')

    def save(self, model_name):
        """Save model to model_name.json and model_name.h5"""
        json_string = model.to_json()
        open(self.models_dir + model_name + '.json', 'w').write(json_string)
        self.model.save_weights(self.models_dir + model_name + '.h5')


class KerasGraphModel(BaseKerasModel):
    def __init__(self, weight_decay=0.0001, nb_epochs=10, mini_batch_size=100):
        super(KerasGraphModel, self).__init__(nb_epochs, mini_batch_size)
        self.weight_decay = weight_decay
        self.graph = Graph()

    def base_convolution(self, input_name, nb_filters, layer_nb, conv_nb,
                         conv_shape=(3,3),
                         stride=(1,1),
                         relu_activation=True,
                         **kwargs):
        """Convolution2D -> BatchNormalization -> ReLU

        :param conv_nb: convolution number
        :param layer_nb: layer number
        :param nb_filters: number of filters
        :param input_name: name of input
        """
        convolution = 'conv{}_{}'.format(layer_nb, conv_nb)
        normalization = 'bn_{}_{}'.format(layer_nb, conv_nb)
        activation = 'relu{}_{}'.format(layer_nb, conv_nb)

        self.graph.add_node(Convolution2D(nb_filter=nb_filters,
                                          nb_row=conv_shape[0],
                                          nb_col=conv_shape[1],
                                          W_regularizer=l2(self.weight_decay),
                                          subsample=stride,
                                          border_mode='same',
                                          **kwargs),
                            name=convolution, input=input_name)

        self.graph.add_node(BatchNormalization(), name=normalization, input=convolution)

        if relu_activation:
            self.graph.add_node(Activation('relu'), name=activation, input=normalization)
            return activation
        else:
            return normalization

    def residual_block(self, input_name, nb_filters, layer_nb, conv_nb, stride=(1,1)):
        """Add a residual building block

        A residual block consists of 2 base convolutions with a short/identity
        connection between the input and output activation

        Input:
        input_name: name of input node, string
        :type nb_filters: int
        :type input_name: str

        Output:
        output_name: name of output node, string
        """
    #    First convolution
        output_name = self.base_convolution(input_name=input_name, nb_filters=nb_filters,
                                            layer_nb=layer_nb, conv_nb=conv_nb,
                                            stride=stride,
                                            relu_activation=True)
        # TODO: Add 1x1, nb_filter/2 short-circuit connection for stride of 2
        # Double the number of filters if we take a stride of 2
        if stride == (2,2):
            nb_filters *= 2

        # Second Convolution, with Batch Normalization, without ReLU activation
        output_name = self.base_convolution(input_name=output_name, nb_filters=nb_filters,
                                            layer_nb=layer_nb, conv_nb=conv_nb+1,
                                            relu_activation=False)

        second_activation = 'relu{}_{}'.format(layer_nb, conv_nb+1)
        self.graph.add_node(Activation('relu'), name=second_activation,
                            inputs=[input_name, output_name],
                            merge_mode='sum')

        return second_activation, nb_filters


    def build_residual_network(self, initial_nb_filters=4, first_conv_shape=(7,7)):
        """34-layer Residual Network with skip connections

            layer name      output size     18-layer        34-layer
            conv1           112x112      7x7, 64, stride 2 -> 3x3 max pool, stride 2
            conv2_x         56x56           [3x3, 64]x2     [3x3, 64]x3
                                            [3x3, 64]       [3x3, 64]
            conv3_x         28x28           [3x3, 128]x2    [3x3, 128]x4
                                            [3x3, 128]      [3x3, 128]
            conv4_x         14x14           [3x3, 256]x2    [3x3, 256]x6
                                            [3x3, 256]      [3x3, 256]
            conv5_x         7x7             [3x3, 512]x2    [3x3, 512]x3
                                            [3x3, 512]      [3x3, 512]
                            1x1          average pool, 1000-d fc, softmax

        Reference: http://arxiv.org/abs/1512.03385
        """
        imsize = 224
        # -------------------------- Layer Group 1 ----------------------------
        self.graph.add_input(name='input', input_shape=(3,imsize,imsize))
        output_name = self.base_convolution(input_name='input',
                                                        nb_filters=initial_nb_filters,
                                                        layer_nb=1,
                                                        conv_nb=1,
                                                        stride=(2,2),
                                                        conv_shape=first_conv_shape,
                                                        input_shape=(3,imsize,imsize),
                                                        dim_ordering='th')
        # Output shape = (None,16,112,112)
        self.graph.add_node(MaxPooling2D(pool_size=(3,3), strides=(2,2),
                                         border_mode='same'),
                            name='pool1', input=output_name)
        # Output shape = (None,32,56,56)
        # -------------------------- Layer Group 2 ----------------------------
        output_name, nb_filters = self.residual_block(input_name='pool1',
                                                      nb_filters=initial_nb_filters,
                                                      layer_nb=2, conv_nb=1)
        for i in xrange(1, 3):
            output_name, nb_filters = self.residual_block(input_name=output_name,
                                                          nb_filters=nb_filters,
                                                          layer_nb=2, conv_nb=(2*i+1))
        # Output shape = (None,16,56,56)
        # -------------------------- Layer Group 3 ----------------------------
        output_name, nb_filters = self.residual_block(input_name=output_name,
                                                      nb_filters=nb_filters,
                                                      layer_nb=3, conv_nb=1,
                                                      stride=(2,2))  # nb_filters *= 2
        for i in xrange(1, 4):
            output_name, nb_filters = self.residual_block(input_name=output_name,
                                                          nb_filters=nb_filters,
                                                          layer_nb=3, conv_nb=(2*i+1))
        # -------------------------- Layer Group 4 ----------------------------
        output_name, nb_filters = self.residual_block(input_name=output_name,
                                                      nb_filters=nb_filters,
                                                      layer_nb=4, conv_nb=1,
                                                      stride=(2,2))  # nb_filters *= 2
        for i in xrange(1, 6):
            output_name, nb_filters = self.residual_block(input_name=output_name,
                                                          nb_filters=nb_filters,
                                                          layer_nb=4, conv_nb=(2*i+1))
        # Output shape = (256,14,14)
        # -------------------------- Layer Group 5 ----------------------------
        output_name, nb_filters = self.residual_block(input_name=output_name,
                                                      nb_filters=nb_filters,
                                                      layer_nb=4, conv_nb=1,
                                                      stride=(2,2))  # nb_filters *= 2
        for i in xrange(1, 3):
            output_name, nb_filters = self.residual_block(input_name=output_name,
                                                          nb_filters=nb_filters,
                                                          layer_nb=5, conv_nb=(2*i+1))
        # Output shape = (None,64,7,7)

        self.graph.add_node(AveragePooling2D(pool_size=(3,3), strides=(2,2),
                                             border_mode='same'),
                            name='pool2', input=output_name)
        self.graph.add_node(Flatten(), name='flatten', input='pool2')
        self.graph.add_node(Dense(9, activation='sigmoid'), name='dense', input='flatten')
        self.graph.add_output(name='output', input='dense')
               
        sgd = SGD(lr=0.1, decay=1e-4, momentum=0.9)
        self.graph.compile(optimizer=sgd, loss={'output': 'binary_crossentropy'})


    def fit(self, input_tensor, target, validation_split=0.1):
        self.graph.fit({'input': input_tensor, 'output': train_df.iloc[:,label_start:].values},
                       batch_size=mini_batch_size, nb_epoch=number_of_epochs,
                       validation_split=validation_split,
                       # validation_data={'input': tensor[test_ind],
                       #                  'output': train_df.iloc[test_ind,label_start:].values},
                       shuffle=True,
                       callbacks=[TensorBoard('/home/rory/logs/2'),
                                  LearningRateScheduler(lr_schedule)],
                       verbose=1)


class KerasSequentialnModel(BaseKerasModel):
    def __init__(self, nb_filters=10, nb_epochs=10):
        super(KerasSequentialnModel, self).__init__()
        self.nb_filters = nb_filters
        n_images = 20000
        imsize   = 224  # Square images
        save_model = False
        show_plots = False
        model_name = 'mar_7_0005'
        csv_dir = 'data/'
        process_images = False
        photo_cache = 'data/photo_cache.pkl'
        jpg_dir = 'data/train_photos/'
        models_dir = 'models/'
        weight_decay = 0.0001

    # #%% Plot a few images to get a feel of how I did
    # def show_plots(self, nb_plots):
    #     for i in range(nb_plots):
    #         show_image_labels(tensor[test_ind[i]], predictions[i],
    #                           train_df['labels'][test_ind[i]], im_mean)

if __name__ == '__main__':
    test_model = KerasGraphModel()
    test_model.build_residual_network()
    test_model.graph.summary()