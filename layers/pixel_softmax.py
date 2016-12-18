import lasagne

from layers.convs import MaskedConv2D


def pixel_softmax_reshape(network, n_colors=1):
    # save the original shape
    original_shape = network.output_shape

    # we'll now interpret the channel values at each pixel as softmax values
    # for each of the colors
    network = MaskedConv2D(incoming=network, num_filters=256 * n_colors,
                           filter_size=(1, 1), mask_type="b", nonlinearity=lasagne.nonlinearities.identity,
                           n_colors=n_colors)
    network = lasagne.layers.ReshapeLayer(network,
                                          (-1, n_colors, 256, original_shape[2], original_shape[3]))

    # dimensions are now:
    # (batch_size x color x height x width x 256)
    # the output of this still needs to be flattened
    network = lasagne.layers.DimshuffleLayer(network, (0, 1, 3, 4, 2))

    return network
