import numpy as np
import theano
import theano.tensor as T
import colorlog as log
import logging

log.basicConfig(level=logging.DEBUG)


def mask_spatial_context(filter, center):
    # zero out context connections
    filter = T.set_subtensor(filter[:, :, center[0] + 1:, :], 0)
    filter = T.set_subtensor(filter[:, :, center[0], center[1] + 1:], 0)
    return filter


def mask(filter, n_colors, type):
    """
    :param filter: 4d symbolic filter
            (output_channels x input_channels x height x width)
    :return: masked filter
    """

    center = (filter.shape[-2] // 2, filter.shape[-1] // 2)
    filter = mask_spatial_context(filter, center)

    # mask the connections between input channels and output channels
    out_part_size = filter.shape[0] // n_colors
    in_part_size = filter.shape[1] // n_colors

    # consider the residuals in case the out- or in-channel shapes
    # are not divisible by the number of colors
    out_residual = filter.shape[0] % n_colors
    in_residual = filter.shape[1] % n_colors

    assert (type == "a" or type == "b"), "Mask type can be only 'a' or 'b'"

    curr_out_part_pos = 0
    curr_in_part_pos = 0
    # Count the residuals to the part of the first color.
    next_out_part_pos = out_residual + out_part_size
    next_in_part_pos = in_residual + in_part_size
    for i in range(n_colors):
        masked_subtensor = filter[curr_out_part_pos:next_out_part_pos,
                           curr_in_part_pos if type == "a" else next_in_part_pos:,
                           center[0], center[1]]
        curr_in_part_pos = next_in_part_pos
        curr_out_part_pos = next_out_part_pos
        next_in_part_pos += in_part_size
        next_out_part_pos += out_part_size
        filter = T.set_subtensor(masked_subtensor, 0)

    return filter


if __name__ == "__main__":
    input = T.tensor4("filter")
    output = mask(input, n_colors=3, type="b")
    f = theano.function(inputs=[input], outputs=output)

    masked = f(np.ones((4, 4, 3, 3), dtype=np.float32))

    log.info("red out")
    log.info(masked[0])

    log.info("green out")
    log.info(masked[1])

    log.info("blue out")
    log.info(masked[2])
