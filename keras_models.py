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
    """ Base class for Keras Sequential and Graph models."""

    def __init__(self, nb_epochs=10, mini_batch_size=100):
        self.nb_epochs = nb_epochs
        self.mini_batch_size = mini_batch_size

    def evaluate(self, X_val, y_val):
        """ Calculate the Mean F1 Score

        :param X_val: training hold out data
        :param y_val: target from hold out data

                            Mean F1 Score
        Sample submission   0.36633
        Random submission   0.43468
        Benchmark           0.64590
        Leader (1/17)       0.81090
        """
        # Threshold at 0.5 and convert to 0 or 1
        predictions = (self.predict({'input': X_val})['output'] > .5)*1
        return predictions

    @staticmethod
    def lr_schedule(epoch):
        if epoch < 6:
            lr = 0.1
        elif epoch < 10:
            lr = 0.01
        else:
            lr = 0.001
        return lr

    @staticmethod
    def _load_model(model_stem):
        """ Load model from JSON and weights from HDF5."""

        model = model_from_json(open(model_stem + '.json').read())
        model.load_weights(model_stem + '.h5')
        return model

    @staticmethod
    def _save(model, model_stem):
        """ Save model to model_name.json and model_name.h5"""

        json_string = model.to_json()
        open(model_stem + '.json', 'w').write(json_string)
        model.save_weights(model_stem + '.h5')

    def predict(self):
        pass


class KerasGraphModel(BaseKerasModel):
    """ Keras Graph model and a method to create an arbitrary residual network."""

    def __init__(self, weight_decay=0.0001, nb_epochs=10, mini_batch_size=100, graph=None):
        self.weight_decay = weight_decay
        super().__init__(nb_epochs, mini_batch_size)
        self.graph = graph

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
        normalization = 'bn{}_{}'.format(layer_nb, conv_nb)
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

    def residual_block(self, input_name, nb_filters, layer_nb, conv_nb, first_stride=(1, 1)):
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

        # First convolution
        first_relu = self.base_convolution(input_name=input_name, nb_filters=nb_filters,
                                           layer_nb=layer_nb, conv_nb=conv_nb,
                                           stride=first_stride)
        output_shape = self.graph.nodes[first_relu].output_shape

        # Second Convolution, with Batch Normalization, without ReLU activation
        second_bn = self.base_convolution(input_name=first_relu, nb_filters=nb_filters,
                                          layer_nb=layer_nb, conv_nb=conv_nb+1,
                                          stride=(1, 1),
                                          relu_activation=False)

        # Add the short convolution, with Batch Normalization
        if first_stride == (2, 2):
            short_conv = 'short{}_{}'.format(layer_nb, conv_nb)
            self.graph.add_node(Convolution2D(nb_filter=nb_filters//4,
                                              nb_row=1,
                                              nb_col=1,
                                              W_regularizer=l2(self.weight_decay),
                                              border_mode='same'),
                                name=short_conv, input=input_name)

            short_bn = 'short_bn{}_{}'.format(layer_nb, conv_nb+1)
            self.graph.add_node(BatchNormalization(), name=short_bn, input=short_conv)
            short_reshape = 'short_reshape{}_{}'.format(layer_nb, conv_nb)
            self.graph.add_node(Reshape(output_shape[1:]), name=short_reshape, input=short_bn)

            input_name = short_reshape       # Overwrite input_name with reshaped short circuit

        output_activation = 'relu{}_{}'.format(layer_nb, conv_nb+1)
        self.graph.add_node(Activation('relu'), name=output_activation,
                            inputs=[second_bn, input_name],
                            merge_mode='sum')

        return output_activation

    def build_residual_network(self, nb_blocks=[1,3,4,6,3],
                               initial_nb_filters=64,
                               first_conv_shape=(7, 7)):
        """Construct a residual convolutional network graph from scratch.

        Parameters
        ----------
        nb_blocks : list
           The number of residual blocks for each layer group. For the 18-layer
           model nb_blocks=[1,2,2,2,2] and 34-layer nb_blocks=[1,3,4,6,3].
        initial_nb_filters : int, optional
           The initial number of filters to use. The number of filters is doubled
           for each layer.
        first_conv_shape : tuple of ints
           The shape of the first convolution, also known as the kernel size.

        Returns
        -------
        self.graph : A new Keras graph


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
        self.graph = Graph()
        # -------------------------- Layer Group 1 ----------------------------
        self.graph.add_input(name='input', input_shape=(3, imsize, imsize))
        output_name = self.base_convolution(input_name='input',
                                            nb_filters=initial_nb_filters,
                                            layer_nb=1,
                                            conv_nb=1,
                                            stride=(2, 2),
                                            conv_shape=first_conv_shape,
                                            input_shape=(3, imsize, imsize),
                                            dim_ordering='th')
        # Output shape = (None,16,112,112)
        self.graph.add_node(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                                         border_mode='same'),
                            name='pool1', input=output_name)
        # Output shape = (None,initial_nb_filters,56,56)
        # -------------------------- Layer Group 2 ----------------------------
        output_name = self.residual_block(input_name='pool1',
                                          nb_filters=initial_nb_filters,
                                          layer_nb=2,
                                          conv_nb=1)
        for i in range(1, nb_blocks[1]):
            output_name = self.residual_block(input_name=output_name,
                                              nb_filters=initial_nb_filters,
                                              layer_nb=2,
                                              conv_nb=(2*i+1))
        # self.graph.nodes[output_name] = (None,initial_nb_filters,56,56)
        # output size = 14x14
        # -------------------------- Layer Group 3 ----------------------------
        output_name = self.residual_block(input_name=output_name,
                                          nb_filters=initial_nb_filters*2,
                                          layer_nb=3, conv_nb=1, first_stride=(2, 2))
        for i in range(1, nb_blocks[2]):
            output_name = self.residual_block(input_name=output_name,
                                              nb_filters=initial_nb_filters*2,
                                              layer_nb=3,
                                              conv_nb=(2*i + 1))
        # -------------------------- Layer Group 4 ----------------------------
        output_name = self.residual_block(input_name=output_name,
                                          nb_filters=initial_nb_filters*4,
                                          layer_nb=4, conv_nb=1, first_stride=(2,2))
        for i in range(1, nb_blocks[3]):
            output_name = self.residual_block(input_name=output_name,
                                              nb_filters=initial_nb_filters*4,
                                              layer_nb=4,
                                              conv_nb=(2*i+1))
        # output size = 14x14
        # -------------------------- Layer Group 5 ----------------------------
        output_name = self.residual_block(input_name=output_name,
                                          nb_filters=initial_nb_filters*8,
                                          layer_nb=5, conv_nb=1, first_stride=(2, 2))
        for i in range(1, nb_blocks[4]):
            output_name = self.residual_block(input_name=output_name,
                                              nb_filters=initial_nb_filters*8,
                                              layer_nb=5,
                                              conv_nb=(2*i + 1))
        # output size = 7x7
        self.graph.add_node(AveragePooling2D(pool_size=(7,7),
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

    def load_graph(self, model_name_stem):
        self.graph = self._load_model(model_stem=model_name_stem)


if __name__ == '__main__':
    test_model = KerasGraphModel()
    test_model.load_graph('../small_test_model')
    test_model.graph.summary()