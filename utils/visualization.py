def save_grayscale_image(image, filename):
    import scipy
    """
    images.shape: (channels x height x width)
    """
    scipy.misc.toimage(image, mode='P').save('{}.png'.format(filename))


def save_grayscale_images_grid(images, image_size, grid_size, filepath):
    import scipy
    assert images.shape[0] == grid_size[0] * grid_size[1], "Image count %d does not fit into grid (%d x %d)" % (
        images.shape[0], grid_size[0], grid_size[1])

    new_shape = grid_size + image_size
    images = images.reshape(new_shape)
    images = images.transpose(1, 2, 0, 3)
    images = images.reshape((grid_size[0] * image_size[0], grid_size[1] * image_size[1]))
    scipy.misc.toimage(images, cmin=0.0, cmax=1.0).save(filepath)


def save_network_graph(network, filename):
    import lasagne
    from nolearn.lasagne.visualize import draw_to_file
    layers_debug = lasagne.layers.get_all_layers(network)
    draw_to_file(layers_debug, filename)


def dynamic_image(init_img, image_generator):
    from matplotlib.pyplot import subplots, close
    from matplotlib import cm
    from PIL import Image

    """ Visualise the simulation using matplotlib, using blit for
    improved speed"""
    fig, ax = subplots(1, 1)
    img = init_img.resize((500, 500), Image.ANTIALIAS)
    im = ax.imshow(img, interpolation='nearest', cmap=cm.hot, animated=True)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)  # cache the background
    fig.show()

    for img in image_generator():
        img = img.resize((500, 500), Image.ANTIALIAS)
        im.set_data(img)
        fig.canvas.restore_region(background)  # restore background
        ax.draw_artist(im)  # redraw the image
        fig.canvas.blit(ax.bbox)  # redraw the axes rectangle

    close(fig)
