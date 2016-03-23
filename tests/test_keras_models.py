import unittest
from models.keras_models import KerasGraphModel

class TestBaseConvolution(unittest.TestCase):

    def setUp(self):
        self.graph = KerasGraphModel()

    def test_base_convolution(self):
        output_name, nb_filters = self.graph.base_convolution(input_name='in_name',
                                                              nb_filters=4,
                                                              layer_nb=1,
                                                              conv_nb=1)
        self.assertEqual(nb_filters, 4)


if __name__ == '__main__':
    unittest.main()