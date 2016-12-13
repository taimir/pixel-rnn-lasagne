import numpy as np
import theano
import theano.tensor as T
import lasagne
import os
import scipy
import colorlog as log
import logging

from pixel_softmax import pixel_softmax_reshape
from convs import MaskedConv2D
from diag_lstm import DiagLSTMLayer
from model_monitor import ModelMonitor

log.basicConfig(level=logging.DEBUG)

# from nolearn.lasagne.visualize import draw_to_file

os.environ["THEANO_FLAGS"] = "device=gpu0,lib.cnmem=0"


def save_image(image, filename):
    """
    images.shape: (channels x height x width)
    """
    scipy.misc.toimage(image, mode='P').save('{}.png'.format(filename))


if __name__ == "__main__":
    # constants
    batch_size = 16
    input_channels = 1
    h = 16
    height = width = 28
    out_channels = 32
    n_colors = 1

    images = T.tensor4("images")
    labels = T.itensor4("labels")
    input = lasagne.layers.InputLayer(shape=(None, input_channels, height, width), input_var=images)
    filter_size = (7, 7)
    network = MaskedConv2D(incoming=input, num_filters=h, filter_size=(7, 7), mask_type="a", n_colors=n_colors)
    for i in range(0, 1):
        forward_in = MaskedConv2D(incoming=network, num_filters=4 * h, filter_size=(1, 1), mask_type="b",
                                  n_colors=n_colors)
        backward_in = MaskedConv2D(incoming=network, num_filters=4 * h, filter_size=(1, 1), mask_type="b",
                                   n_colors=n_colors)
        forward = DiagLSTMLayer(incoming=forward_in)
        backward = DiagLSTMLayer(incoming=backward_in, backwards=True)

        shifted_backward = lasagne.layers.SliceLayer(incoming=backward, indices=slice(None, -1), axis=2)
        zero_pad = lasagne.layers.InputLayer(
            input_var=T.zeros((images.shape[0], h, 1, width), dtype=theano.config.floatX),
            shape=(None, h, 1, width))
        shifted_backward = lasagne.layers.ConcatLayer(incomings=[zero_pad, shifted_backward], axis=2)

        together = lasagne.layers.ElemwiseSumLayer(incomings=[forward, shifted_backward])

        # add a residual connection
        network = lasagne.layers.ElemwiseSumLayer(incomings=[together, network])

    for i in range(0, 2):
        network = lasagne.layers.NonlinearityLayer(network, lasagne.nonlinearities.rectify)
        network = MaskedConv2D(incoming=network, num_filters=out_channels,
                               filter_size=(1, 1), mask_type="b", nonlinearity=lasagne.nonlinearities.identity,
                               n_colors=n_colors)

    if n_colors > 1:
        network = pixel_softmax_reshape(network, n_colors=n_colors)
        output = lasagne.layers.get_output(network)
        softmax_output_flat = T.nnet.softmax(T.reshape(output, newshape=(-1, output.shape[-1])))
        labels_flat = T.flatten(labels)

        loss = T.mean(lasagne.objectives.categorical_crossentropy(softmax_output_flat, labels_flat))
        output = T.argmax(output, axis=-1)
    else:
        network = MaskedConv2D(incoming=network, num_filters=1,
                               filter_size=(1, 1), mask_type="b", nonlinearity=lasagne.nonlinearities.sigmoid,
                               n_colors=n_colors)
        output = lasagne.layers.get_output(network)
        loss = T.mean(lasagne.objectives.binary_crossentropy(output, labels))

    params = lasagne.layers.get_all_params(network, trainable=True)

    train_updates = lasagne.updates.adam(loss_or_grads=loss,
                                         params=params,
                                         learning_rate=1e-3)

    train_pass = theano.function(inputs=[images, labels],
                                 outputs=[loss, output],
                                 updates=train_updates)

    validation_pass = theano.function(inputs=[images, labels],
                                      outputs=[loss, output])

    test_pass = theano.function(inputs=[images],
                                outputs=output)

    # layers_debug = lasagne.layers.get_all_layers(network)
    # outputs_debug = lasagne.layers.get_output(layers_debug)
    # draw_to_file(layers_debug, "data/network_graph.png")
    # Print a theano graph for inspection
    # theano.printing.pydotprint(softmax_output, outfile="data/pixel_rnn.png", var_with_name_simple=True)
    # theano.printing.pydotprint(forward_pass, outfile="data/pixel_rnn_compiled.png", var_with_name_simple=True)

    model_monitor = ModelMonitor(outputs=network)

    from mnist import load_data, binarize

    data = load_data()
    x_train, x_valid, x_test = data['x_train'], data['x_valid'], data['x_test']

    y = x_train
    X = np.array(x_train, dtype=np.float32)

    y_valid = x_valid[:1000]
    X_valid = np.array(x_valid[:1000], dtype=np.float32)

    y_test = x_test
    X_test = np.array(x_test, dtype=np.float32)

    # # center all data
    # data_mean = np.mean(X, axis=0, keepdims=True)
    # X = X - data_mean
    # X_valid = X_valid - data_mean
    # X_test = X_test - data_mean

    best_loss = 100

    # train
    minibatch_count = X.shape[0] // batch_size
    val_minibatch_count = X_valid.shape[0] // batch_size
    # try:
    #     for e in range(0, 10):
    #         for i in range(0, minibatch_count):
    #             y_next = y[i * batch_size:(i + 1) * batch_size]
    #             X_next = X[i * batch_size:(i + 1) * batch_size]
    #             loss, train_image = train_pass(X_next, y_next)
    #             if i % 100 == 0:
    #                 log.info("minibatch {} loss: {}".format(i, loss))
    #                 save_image(binarize(train_image)[0, 0], filename="data/trained_images/image_{}.jpg".format(i))
    #                 save_image(y_next[0, 0], filename="data/original_images/image_{}.jpg".format(i))
    #
    #                 val_losses = list()
    #                 for j in range(0, val_minibatch_count):
    #                     y_val_next = y_valid[j * batch_size:(j + 1) * batch_size]
    #                     x_val_next = X_valid[j * batch_size:(j + 1) * batch_size]
    #                     val_loss, _ = validation_pass(x_val_next, y_val_next)
    #                     val_losses.append(val_loss)
    #                 mean_val_loss = np.array(val_losses).mean()
    #                 log.info("validation: epoch {}, iteration {}, loss: {}".format(e, i, mean_val_loss))
    #                 if mean_val_loss < best_loss:
    #                     best_loss = mean_val_loss
    #                     model_monitor.save_model(epoch_count=e, msg="new_best")
    # except KeyboardInterrupt:
    #     log.info("Training was interrupted. Proceeding with image generation.")
    model_monitor.load_model(model_name="params_2ep_new_best.npz", network=network)

    # test by generating images
    for i in range(0, 10):
        image = X_test[[i], :, :, :]
        image[:, :, height // 2:, :] = 0

        rest = height - (height // 2)
        for row_i in range(0, rest):
            for col_i in range(0, width):
                for chan_i in range(0, input_channels):
                    new_image = binarize(test_pass(image))
                    # copy one generated pixel of one channel, then use it for the next generation
                    image[:, chan_i, height // 2 + row_i, col_i] = new_image[:, chan_i, height // 2 + row_i, col_i]
        save_image(image[0, 0], filename="data/generated/image_{}.jpg".format(i))
        log.info("generated image {}".format(i))
