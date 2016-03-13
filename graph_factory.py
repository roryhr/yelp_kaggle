

def building_block(graph, layer_nb, conv_nb=1, input_name):
    """ Add a Residual building block
    

    """
#    First convolution 
    first_convolution = 'conv{}_{}'.format(layer_nb,conv_nb)
    first_normalization = 'bn_{}_{}'.format(layer_nb,conv_nb)
    first_activation = 'relu{}_{}'.format(layer_nb, conv_nb)

#    Second convolution 
    second_convolution = 'conv{}_{}'.format(layer_nb,conv_nb+1)
    second_normalization = 'bn_{}_{}'.format(layer_nb,conv_nb+1)
    second_activation = 'relu{}_{}'.format(layer_nb, conv_nb+1)
    
#    return [first_convolution, first_normalization, first_activation,
#            second_convolution, second_normalization, second_activation]
    
    graph.add_node(Convolution2D(nb_filters, 3, 3, W_regularizer=l2(weight_decay), 
                             border_mode='same'),
               name=first_convolution, input=input_name)
    graph.add_node(BatchNormalization(), name=first_normalization,
                   input=first_convolution)
    graph.add_node(Activation('relu'), name=first_activation,
                   input=first_normalization)
    
#    Second Convolution
    graph.add_node(Convolution2D(nb_filters, 3, 3, W_regularizer=l2(weight_decay), 
                                 border_mode='same'),
                   name=second_convolution, input=first_activation)
    graph.add_node(BatchNormalization(), name=second_normalization,
                   input=second_convolution)
    graph.add_node(Activation('relu'), name=second_activation,
                   inputs=[input_name,second_normalization], 
                   merge_mode='sum')
    
    return graph
    
    
if __name__=='__main__':
    print building_block(1,1)