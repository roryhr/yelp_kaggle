import unittest
from models.keras_models import KerasGraphModel
from keras.models import Graph

class TestBaseConvolution(unittest.TestCase):

    def setUp(self):
        self.model = KerasGraphModel()
        # self.graph = Graph()

    def test_base_convolution(self):
        self.model.graph.add_input(name='input', input_shape=(3,100,100))

        output_name = self.model.base_convolution(input_name='input',
                                                              nb_filters=4,
                                                              layer_nb=2,
                                                              conv_nb=1)
        self.assertEqual(output_name, 'relu2_1')


if __name__ == '__main__':
    unittest.main()