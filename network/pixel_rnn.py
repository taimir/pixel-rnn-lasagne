import abc

import lasagne
import theano
import theano.tensor as T

from layers import MaskedConv2D, DiagLSTMLayer
from layers.pixel_softmax import pixel_softmax_reshape


class PixelNetwork(object):
    def __init__(self, batch_size, image_shape, n_hidden):
        self.batch_size = batch_size
        self.input_channels = image_shape[0]
        self.h = n_hidden
        self.height = image_shape[1]
        self.width = image_shape[2]
        self.out_channels = n_hidden
        self.n_colors = image_shape[0]
        self.inputs = T.tensor4("images")
        self.labels = T.itensor4("labels")
        self.network, self.loss, self.output = self._define_network(self.inputs, self.labels)
        self._define_forward_passes(self.network, self.loss, self.output, self.inputs, self.labels)

    def _define_forward_passes(self, network, loss, output, inputs, labels):
        params = lasagne.layers.get_all_params(network, trainable=True)

        train_updates = lasagne.updates.adam(loss_or_grads=loss,
                                             params=params,
                                             learning_rate=1e-3)

        self.train_pass = theano.function(inputs=[inputs, labels],
                                          outputs=[loss, output],
                                          updates=train_updates)

        self.validation_pass = theano.function(inputs=[inputs, labels],
                                               outputs=[loss, output])

        self.test_pass = theano.function(inputs=[inputs],
                                         outputs=output)

    @abc.abstractmethod
    def _define_network(self, inputs, labels):
        """
        :param inputs - theano variable for the feature vectors
        :param labels - theano variable for the ground truths
        :return: network, loss, output tensor
        """
        return None, None, None


class PixelRNN(PixelNetwork):
    def __init__(self, batch_size, image_shape, n_hidden, depth=2):
        self.depth = depth
        super(PixelRNN, self).__init__(batch_size, image_shape, n_hidden)

    def _define_network(self, inputs, labels):
        input = lasagne.layers.InputLayer(shape=(None, self.input_channels, self.height, self.width),
                                          input_var=inputs)
        network = MaskedConv2D(incoming=input, num_filters=self.h, filter_size=(7, 7), mask_type="a",
                               n_colors=self.n_colors)
        for i in range(0, self.depth):
            forward_in = MaskedConv2D(incoming=network, num_filters=4 * self.h, filter_size=(1, 1), mask_type="b",
                                      n_colors=self.n_colors)
            backward_in = MaskedConv2D(incoming=network, num_filters=4 * self.h, filter_size=(1, 1), mask_type="b",
                                       n_colors=self.n_colors)
            forward = DiagLSTMLayer(incoming=forward_in)
            backward = DiagLSTMLayer(incoming=backward_in, backwards=True)

            shifted_backward = lasagne.layers.SliceLayer(incoming=backward, indices=slice(None, -1), axis=2)
            zero_pad = lasagne.layers.InputLayer(
                input_var=T.zeros((self.inputs.shape[0], self.h, 1, self.width), dtype=theano.config.floatX),
                shape=(None, self.h, 1, self.width))
            shifted_backward = lasagne.layers.ConcatLayer(incomings=[zero_pad, shifted_backward], axis=2)

            # add a residual connection, + network
            network = lasagne.layers.ElemwiseSumLayer(incomings=[forward, shifted_backward, network])

        for i in range(0, 2):
            network = MaskedConv2D(incoming=network, num_filters=self.out_channels,
                                   filter_size=(1, 1), mask_type="b", nonlinearity=lasagne.nonlinearities.rectify,
                                   n_colors=self.n_colors)

        if self.n_colors > 1:
            network = pixel_softmax_reshape(network, n_colors=self.n_colors)
            output = lasagne.layers.get_output(network)
            softmax_output_flat = T.nnet.softmax(T.reshape(output, newshape=(-1, output.shape[-1])))
            labels_flat = T.flatten(labels)

            loss = T.mean(lasagne.objectives.categorical_crossentropy(softmax_output_flat, labels_flat))
            output = T.argmax(output, axis=-1)
        else:
            network = MaskedConv2D(incoming=network, num_filters=1,
                                   filter_size=(1, 1), mask_type="b", nonlinearity=lasagne.nonlinearities.sigmoid,
                                   n_colors=self.n_colors)
            output = lasagne.layers.get_output(network)
            loss = T.mean(lasagne.objectives.binary_crossentropy(output, labels))
        return network, loss, output
