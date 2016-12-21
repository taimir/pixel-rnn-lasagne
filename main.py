import os
import numpy as np
import scipy
import colorlog as log
import logging

from utils.model_monitor import ModelMonitor
from network.pixel_rnn import PixelRNN

from utils.visualization import save_network_graph, dynamic_image, save_grayscale_images_grid

log.basicConfig(level=logging.DEBUG)


def train_model(model, model_monitor, X, y, X_valid, y_valid, batch_size):
    minibatch_count = X.shape[0] // batch_size
    val_minibatch_count = X_valid.shape[0] // batch_size

    best_loss = 100
    try:
        for e in range(0, 10):
            for i in range(0, minibatch_count):
                y_next = y[i * batch_size:(i + 1) * batch_size]
                X_next = X[i * batch_size:(i + 1) * batch_size]
                loss, train_image = model.train_pass(X_next, y_next)
                if i % 100 == 0:
                    log.info("epoch {} minibatch {} loss: {}".format(e, i, loss))

                    val_losses = list()
                    for j in range(0, val_minibatch_count):
                        y_val_next = y_valid[j * batch_size:(j + 1) * batch_size]
                        x_val_next = X_valid[j * batch_size:(j + 1) * batch_size]
                        val_loss, _ = model.validation_pass(x_val_next, y_val_next)
                        val_losses.append(val_loss)
                    mean_val_loss = np.array(val_losses).mean()
                    log.info("validation: epoch {}, iteration {}, loss: {}".format(e, i, mean_val_loss))
                    if mean_val_loss < best_loss:
                        best_loss = mean_val_loss
                        model_monitor.save_model()
    except KeyboardInterrupt:
        log.info("Training was interrupted. Proceeding with image generation.")


def generate_images(model, model_monitor):
    """
    Samples a grid of images from the model
    :param model: e.g. PixelRNN
    :param model_monitor: the monitor to load the model weights with
    """
    model_monitor.load_model(model_name="best_params.npz", network=model.network)

    images = np.zeros((100, 1, 28, 28), dtype=np.float32)

    for row_i in range(0, height):
        for col_i in range(0, width):
            for chan_i in range(0, input_channels):
                new_images = sample(model.test_pass(images))
                # copy one generated pixel of one channel, then use it for the next generation
                images[:, chan_i, row_i, col_i] = new_images[:, chan_i, row_i, col_i]

    save_grayscale_images_grid(images=images, image_size=(28, 28), grid_size=(10, 10),
                               filepath=os.path.join(os.path.dirname(__file__), "data/generated/images.jpg"))
    log.info("Images generated under data/generated :) ")


def test_model(model, model_monitor, X_test):
    """
    Dynamically completes the lower half an image pixel by pixel, showing the generation in a window.

    :param model: e.g. PixelRNN
    :param model_monitor: model monitor to load the model weights with
    :param X_test: an array of images, the lower half of which will be completed
    """
    model_monitor.load_model(model_name="best_params.npz", network=model.network)

    def image_gen():
        for i in range(0, 100):
            image = X_test[[i], :, :, :]
            image[:, :, height // 2:, :] = 0.5

            # show the image and keep refreshing it
            for row_i in range(height // 2, height):
                for col_i in range(0, width):
                    for chan_i in range(0, input_channels):
                        next = sample(model.test_pass(image))
                        image[:, chan_i, row_i, col_i] = next[:, chan_i, row_i, col_i]
                        img = scipy.misc.toimage(image[0, 0], cmin=0.0, cmax=1.0)
                        yield img

    dynamic_image(init_img=scipy.misc.toimage(X_test[0, 0], cmin=0.0, cmax=1.0), image_generator=image_gen)


if __name__ == "__main__":
    # CONSTANTS
    batch_size = 16
    h = 64
    height = width = 28
    input_channels = 1

    # INSTANTIATE MODEL
    pixel_rnn = PixelRNN(batch_size=batch_size, image_shape=(input_channels, height, width), n_hidden=h)

    save_network_graph(pixel_rnn.network, os.path.join(os.path.dirname(__file__), "data/network_graph.png"))
    model_monitor = ModelMonitor(outputs=pixel_rnn.network, name=pixel_rnn.get_name())

    # DATA PREP
    from mnist import load_data, sample

    data = load_data()
    x_train, x_valid, x_test = data['x_train'], data['x_valid'], data['x_test']

    y = x_train
    X = np.array(x_train, dtype=np.float32)

    y_valid = x_valid[:1000]
    X_valid = np.array(x_valid[:1000], dtype=np.float32)

    y_test = x_test
    X_test = np.array(x_test, dtype=np.float32)

    # USE MODEL
    # train_model(model=pixel_rnn, model_monitor=model_monitor, X=X, y=y, X_valid=X_valid, y_valid=y_valid,
    #             batch_size=batch_size)
    test_model(model=pixel_rnn, model_monitor=model_monitor, X_test=X_test)
    # generate_images(model=pixel_rnn, model_monitor=model_monitor)
